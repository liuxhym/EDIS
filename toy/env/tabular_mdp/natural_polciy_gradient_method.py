
"""
overview:
    solve small NOP using natural policy gradint
output:
    update policy
usage-example:
    python3 natural_polciy_gradient_method.py
"""

# import module
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys

# parameter of policy
theta_choice = np.array([[-5,-5],[-4,-4],[-3,-3],[-2.25,-3.25],[-1.5,-1],[-1.0, -2.5]])
color_list = ["forestgreen", "darkgreen", "seagreen", "mediumaquamarine", "turquoise", "darkturquoise"]

# reward function
r = np.zeros((2, 2))
r[0, 0] = 1.0
r[0, 1] = 0.0
r[1, 0] = 2.0
r[1, 1] = 0.0

# policy
pi = np.zeros((2,2))

# anther parameter
alpha         = 0.03
gamma         = 1
episode       = 20
epoch         = 1000

# 方策パラメータ可視化用のインスタンス
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# シグモイド関数
def sigmoid(s, theta):
    sigmoid_range = 34.5

    if theta[s] <= -sigmoid_range:
        return 1e-15
    if theta[s] >= sigmoid_range:
        return 1.0 - 1e-15

    return 1.0 / (1.0 + np.exp(-theta[s]))

def differential_log_pi(s_t, a_t, theta):
    nabla_theta = np.zeros(2)
    if a_t == 0:
        nabla_theta[s_t] = 1.0 - sigmoid(s_t, theta)
        nabla_theta[(s_t+1)%2] = 0
    else:
        nabla_theta[s_t] = (-1.0) * sigmoid(s_t, theta)
        nabla_theta[(s_t+1)%2] = 0

    return nabla_theta

def double_differential_log_pi(s_t, a_t, theta):
    double_nabla_theta = np.zeros((2,2))
    double_nabla_theta[s_t, s_t] = (1.0 - sigmoid(s_t, theta)) * sigmoid(s_t, theta)

    return double_nabla_theta

def act(s, theta):
    pi[s, 0] = sigmoid(s, theta)
    pi[s, 1] = 1 - pi[s, 0]

    p = np.random.rand()
    if p <= pi[s, 0]:
        action = 0
    else:
        action = 1

    return action

def natural_polciy_gradient(theta, nabla_eta, alpha, fisher_information_metrix):
    fisher_information_metrix_add_I = fisher_information_metrix + np.eye(2, 2) * 1.0e-2
    f_inv = np.linalg.inv(fisher_information_metrix_add_I)
    delta_theta = np.dot(f_inv, nabla_eta)
    theta_new   = list(np.array(theta) + alpha * delta_theta)

    theta_range = 500

    for i in range(len(theta_new)):
        if theta_new[i] <= -theta_range:
            theta_new[i] = -theta_range
        if theta_new[i] >=  theta_range:
            theta_new[i] =  theta_range

    return theta_new

def step(s, a, r):
    if a == 0:
        s_next = s
    elif a == 1:
        s_next = (s + 1)%2

    return s_next, r[s, a]

# 等高線の表示
theta1 = np.arange(-7.5, 10, 0.05)
theta2 = np.arange(-7.5, 10, 0.05)

R_ = np.zeros((len(theta1), len(theta2)))
# R_ = np.load('./R.npy')
i = 0

for x in theta1:
    j = 0
    for y in theta2:
        # 初期状態の観測
        for k in range(10):
            p_s = np.random.rand()
            if p_s <= 0.8:
                s = 0
            else:
                s = 1
            theta = [x,y]
            rewards = []
            scores_deque = []
            scores = []

            for t in range(episode):
                a = act(s, theta)
                s, reward = step(s, a, r)
                rewards.append(reward)

            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
                    
            discounts = [gamma**i for i in range(len(rewards)+1)]
            R_[j,i] += sum([a*b for a,b in zip(discounts, rewards)]) / 10
        j = j + 1
    i = i + 1

# np.save('./R', np.array(R_))

im = ax.pcolormesh(theta1, theta2, R_, cmap='PuOr',alpha=0.3)
im = ax.contourf(theta1, theta2, R_, cmap="inferno")
fig.colorbar(im, ax=ax)

i = 0
for theta in theta_choice:

    print(theta)

    ax.scatter(theta[0], theta[1], s=40, marker='o',  color = color_list[i])

    theta_total = []
    R_total = []

    for epoch_ in range(epoch):
        rewards = []
        scores_deque = []
        scores = []
        states = []
        actions = []
        q_values = []

        # 初期状態の観測
        p_s = np.random.rand()
        if p_s <= 0.8:
            s = 0
        else:
            s = 1

        states.append(s)
        theta_total.append(theta)

        # 1 episode分の状態・行動をサンプリング
        for t in range(episode):
            a = act(s, theta)
            next_state, reward = step(s, a, r)
            # 各要素の保存
            s = next_state
            rewards.append(reward)
            actions.append(a)
            states.append(s)

        # 収益の計算
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # 各時間ステップの割引報酬和を計算
        G = []
        for t in range(episode):
            discounts = [gamma**i for i in range(episode-t+1)]
            G.append(sum([a*b for a,b in zip(discounts, rewards[t:])]))

        # print(G)
                
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)]) / episode
        
        saved_nabla_log_pi = []
        saved_double_nabla_log_pi = []
        policy_gradient = []

        # log(pi)の勾配計算
        for t in range(episode):
            saved_nabla_log_pi.append(list(np.array(differential_log_pi(states[t], actions[t], theta)).T))

        # モンテカルロによる勾配近似(REINFOCE)
        for nabla_log_pi_, r_ in zip(saved_nabla_log_pi, G):   
            policy_gradient.append(list(np.array(nabla_log_pi_) * r_))
 
        # 方策パラメータの更新
        nabla_eta = np.sum(np.array(policy_gradient).T, axis=1) / episode

        # フィッシャー情報行列の計算
        for t in range(episode):
            saved_double_nabla_log_pi.append(list(double_differential_log_pi(states[t], actions[t], theta)))        
        fisher_information_metrix = np.sum(np.array(saved_double_nabla_log_pi), axis=0) / episode

        # 方策の更新
        theta = natural_polciy_gradient(theta, nabla_eta, alpha, fisher_information_metrix)

        R_total.append(R)

    theta_total_T = list(np.array(theta_total).T)
    ax.plot(theta_total_T[0], theta_total_T[1], linewidth=3, color = color_list[i])
    i = i + 1

# 方策パラメータの可視化
plt.title('Natural Policy Gradient (1000 episode) ')
plt.xlabel('theta 1')
plt.ylabel('theta 2')
plt.ylim(-7.5, 10)
plt.xlim(-7.5, 10)

plt.savefig('Natural_Policy_Gradient.png')