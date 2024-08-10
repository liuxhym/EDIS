import numpy as np
import collections
import random
import os
import datetime

import torch
import torch.nn.functional as F

eps = 0.3


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    
class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ProbNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ProbNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class SAC(torch.nn.Module):
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device, lfiw=0, foda=0):
        super(SAC, self).__init__()
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        self.prob_net = ProbNet(state_dim, hidden_dim, 1).to(device)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        self.prob_optimizer = torch.optim.Adam(self.prob_net.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.lfiw = lfiw
        self.foda = foda
        self.time = datetime.datetime.now().strftime('%H-%M-%S')
        self.env_step = 0

    def lfiw_update(self, slow_states, slow_actions, fast_states, fast_actions):
        slow_states = torch.tensor(slow_states,
                              dtype=torch.float).to(self.device).squeeze()
        slow_actions = torch.tensor(slow_actions).view(-1, 1).to(
            self.device)
        fast_states = torch.tensor(fast_states,
                                dtype=torch.float).to(self.device).squeeze() 
        fast_actions = torch.tensor(fast_actions).view(-1, 1).to(
            self.device)
        slow_samples = torch.cat([slow_states, slow_actions], dim=1)
        fast_samples = torch.cat([fast_states, fast_actions], dim=1)

        slow_preds = self.prob_net(slow_samples)
        fast_preds = self.prob_net(fast_samples)

        zeros = torch.zeros_like(slow_preds).to(self.device)
        ones = torch.ones_like(fast_preds).to(self.device)

        loss = F.binary_cross_entropy(torch.sigmoid(slow_preds), zeros) + \
            F.binary_cross_entropy(torch.sigmoid(fast_preds), ones)
        
        self.prob_optimizer.zero_grad()
        loss.backward()
        self.prob_optimizer.step()

    def take_action(self, state, stochastic = True):
        batch_size = state.shape[0]
        if stochastic and np.random.random() < eps:
            action = np.random.randint(4, size=(batch_size, 1)) 
            return action   
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        action = action.unsqueeze(1)
        return action.detach().cpu().numpy()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        # print(next_states, next_states.shape)
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
            
    def snapshot(self, timestamp):
        save_dir = 'model_oct'
        if self.lfiw == 0:
            save_dir = os.path.join(save_dir, "sac", self.time)
        elif self.foda == 0:
            save_dir = os.path.join(save_dir, "lfiw", self.time)
        else:
            save_dir = os.path.join(save_dir, "foda", self.time)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, "ite_{}.pt".format(timestamp))
        torch.save(self.state_dict(), model_save_path)

    def update_sac(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device).squeeze()
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device).squeeze()
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)


        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        # print(states.shape)
        # print(self.critic_1(states).shape)
        # print(actions.shape)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
 

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device).squeeze()
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        acc_rewards = torch.tensor(transition_dict['acc_rewards'],
                                   dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device).squeeze()
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_probs = self.actor(next_states)
            value_next = torch.sum(next_probs * torch.min(self.critic_1(next_states), self.critic_2(next_states)))
            init_states = torch.zeros_like(next_states)
            init_probs = self.actor(init_states)
            value_init = torch.sum(init_probs * torch.min(self.critic_1(init_states), self.critic_2(init_states)))
            acc_adv = acc_rewards + value_next - value_init
            acc_adv = torch.clip(acc_adv, 0, None)
            acc_adv = acc_adv / (torch.max(acc_adv) + 1.0)
            acc_adv = torch.sqrt(acc_adv)
            raw_weight = self.prob_net(torch.cat([states, actions], dim=1))
            weight = torch.sigmoid(raw_weight * self.lfiw)
            weight = weight / torch.mean(weight)
            weight = torch.sqrt(weight)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        # print(states.shape)
        # print(self.critic_1(states).shape)
        # print(actions.shape)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(weight * critic_1_q_values, weight * td_target.detach()) + \
                self.foda * F.mse_loss(acc_adv * critic_1_q_values, acc_adv * td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(weight * critic_2_q_values, weight * td_target.detach()) + \
                self.foda * F.mse_loss(acc_adv * critic_2_q_values, acc_adv * td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
