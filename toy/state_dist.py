import os
import uuid
from typing import Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import argparse
from dataclasses import dataclass

from common.buffer import calq_ReplayBuffer
from common.energy import energy_model

from diffusion.trainer import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator
from diffusion.utils import construct_diffusion_model
from diffusion.denoiser_network import ResidualMLPDenoiser

from env.tabular_mdp.maze import Maze, Status
from sac import SAC
from tqdm import trange
import gin
from env.tabular_mdp.maze import Maze, Status
from sac import SAC
from tqdm import trange
import gin


# @dataclass
# @dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-expert-v2"  # OpenAI gym environment name
    description: str = "accuracy of diffusion prediction"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = "offline_model.pth"  # Model load file name, "" doesn't load
    map_as_state: bool = False

    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization
    # Cal-QL
    is_sparse_reward: bool = False  # Use sparse reward
    log_name: str = "diff_pred"  # name for the log directory

    # diffusion
    utd_ratio: int = 1
    energy_hidden_layers: int = 3
    ebm_activation: str = "relu"
    ebm_layer_type: str = "MLP"
    ebm_spectral_norm: bool = True
    ebm_lr: float = 1e-3
    negative_sample: int = 10
    energy_train_epoch: int = 20
    grad_clip: float = 1.0
    ope_clip: float = 2
    te_clip: float = 0.01
    pe_clip: float = 0.01

    model_terminals: bool = False
    num_samples: int = 100000
    retrain_diffusion_every: int = 10000
    diffusion_start: int = 0
    gin_config_files: str = 'configs/synther.gin'
    rollout_step: int = 4

    def __post_init__(self):
        self.name = f"{self.log_name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def collect_data(agent, num_traj=100):
    states = []
    actions = []
    next_states = []
    rewards = []
    terminals = []

    for _ in range(num_traj):
        s = env.reset()
        done = False
        while not done:
            a = agent.take_action(s, stochastic=True)
            states.append(s[0])
            actions.append(a)
            assert a.shape[1] == 1, a
            next_s, r, done, info = env.step(a[0, 0])
            next_states.append(next_s[0])
            rewards.append(r)
            terminals.append(done)
            s = next_s
    states = np.array(states)
    actions = np.array(actions).reshape(-1, 1)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    return states, actions, rewards, next_states, terminals


def visitation_frequencey(states):
    state_visitation = np.zeros_like(map, dtype=float)
    l_max, w_max = state_visitation.shape
    l_max, w_max = int(l_max - 1), int(w_max - 1)
    for idx in range(len(states)):
        l = min(int(states[idx][1]), l_max)
        l = max(l, 0)
        w = min(int(states[idx][0]), w_max)
        w = max(w, 0)
        state_visitation[l, w] += 1.0
    state_visitation = state_visitation / 300
    return state_visitation


def plot(probs_1: np.ndarray, name: str):
    sns.set_theme(font="sans-serif", font_scale=1.2, rc={'figure.figsize': (6, 5)})
    sns.heatmap(probs_1, vmin=0, vmax=1.0, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    plt.savefig("toy/image/dist_{}.jpg".format(name))
    plt.clf()

def plot_diff(probs_1: np.ndarray, name: str, cmax=None, cmin=None):
    sns.set_theme(font="sans-serif", font_scale=1.2, rc={'figure.figsize': (6, 5)})
    sns.heatmap(probs_1, vmin=cmin, vmax=cmax, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    plt.savefig("toy/image/diff_{}.jpg".format(name))
    plt.clf()


def draw_dist(data_1, name):
    states_1 = data_1
    dist_1 = visitation_frequencey(states_1)
    plot(dist_1, name)

def draw_diff(data_1, data_2, name, cmax=None, cmin=None):
    states_1 = data_1
    states_2 = data_2

    dist_1 = visitation_frequencey(states_1)
    dist_2 = visitation_frequencey(states_2)

    diff = dist_2 - dist_1
    print(np.sum(np.abs(diff)))
    plot_diff(diff, name, cmax, cmin)

def compute_diff(data_pre, data_post, eta):
    state_pre, acc_adv = itemgetter("state", "acc_adv")(data_pre)
    state_post = itemgetter("state")(data_post)

    dist_pre = visitation_frequencey(state_pre)
    dist_post = visitation_frequencey(state_post)

    weight_dist_pre = visitation_frequencey(state_pre, acc_adv, eta)

    diff = dist_post - dist_pre
    diff_weighted = dist_post - weight_dist_pre

    return dist_pre, diff, diff_weighted


def load_plot(load_path_1, load_path_2, eta):
    data_pre = np.load(load_path_1)
    data_post = np.load(load_path_2)

    dist_pre, diff, diff_weighted = compute_diff(data_pre, data_post, eta)

    delta_pre = np.sum(np.abs(diff))
    delta_post = np.sum(np.abs(diff_weighted))
    ratio = delta_post / delta_pre

    sns.set_theme(font="sans-serif", font_scale=1.7, rc={'figure.figsize': (6, 5)})
    sns.heatmap(dist_pre, cmap="Reds", xticklabels=False, yticklabels=False)
    plt.show()

    cmax = np.max(np.array([diff, diff_weighted]))
    cmin = np.min(np.array([diff, diff_weighted]))
    sns.heatmap(diff, vmin=cmin, vmax=cmax, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    plt.show()
    sns.heatmap(diff_weighted, vmin=cmin, vmax=cmax, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    plt.show()

    print("Original dist diff: ", delta_pre)
    print("Adjusted dist diff:", delta_post)
    print("Ratio: ", ratio)
    print("---------------------------------")


def load_plot_save(load_path_1, load_path_2, save_dir, eta):
    data_pre = np.load(load_path_1)
    data_post = np.load(load_path_2)

    dist_pre, diff, diff_weighted = compute_diff(data_pre, data_post, eta)

    delta_pre = np.sum(np.abs(diff))
    delta_post = np.sum(np.abs(diff_weighted))
    ratio = delta_post / delta_pre

    sns.set_theme(font="sans-serif", font_scale=1.7, rc={'figure.figsize': (6, 5)})
    sns.heatmap(dist_pre, cmap="Reds", xticklabels=False, yticklabels=False)
    plt.savefig(save_dir + "_pre_dist.svg", format='svg', dpi=400)
    plt.clf()

    cmax = np.max(np.array([diff, diff_weighted]))
    cmin = np.min(np.array([diff, diff_weighted]))
    sns.heatmap(diff, vmin=cmin, vmax=cmax, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig(save_dir + "_diff.svg", format='svg', dpi=400)
    plt.clf()
    sns.heatmap(diff_weighted, vmin=cmin, vmax=cmax, cmap="RdBu_r", center=0.0, xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig(save_dir + "_weight_diff.svg", format='svg', dpi=400)
    plt.clf()

    print("Original dist diff: ", delta_pre)
    print("Adjusted dist diff:", delta_post)
    print("Ratio: ", ratio)
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2")  ################ task
    parser.add_argument("--seed", type=int, default=6)  ############################ seed
    parser.add_argument("--log_name", type=str, default="cal_ql")
    parser.add_argument("--policy_guide", action='store_true', default=True)
    parser.add_argument("--state_guide", action='store_true', default=True)
    parser.add_argument("--transition_guide", action='store_true', default=True)
    parser.add_argument("--test_divergence", type=bool, default=False)

    parser.add_argument("--ope_clip", type=float, default=10)
    parser.add_argument("--te_clip", type=float, default=0.1)
    parser.add_argument("--pe_clip", type=float, default=0.1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    map = np.array([[0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [1, 0, 0, 1, 0]])
    env = Maze(map)
    episode_length = 200
    ite = 20
    online_ite = 150
    offline_agent = SAC(2, 128, 4, 1e-3, 1e-2, 1e-2, -1, 0.005, 0.98, torch.device("cuda"))
    online_agent = SAC(2, 128, 4, 1e-3, 1e-2, 1e-2, -1, 0.005, 0.98, torch.device("cuda"))
    pre_agent_load_path = "toy/model/ite_" + str(ite) + ".pt"
    curr_agent_load_path = "toy/model/ite_" + str(online_ite) + ".pt"
    offline_agent.load_state_dict(torch.load(pre_agent_load_path, map_location=torch.device("cuda")))
    online_agent.load_state_dict(torch.load(curr_agent_load_path, map_location=torch.device("cuda")))

    print("start collecting data")
    offline_data = collect_data(offline_agent, 100)
    online_data = collect_data(online_agent, 10)

    state_dim = env.state_dim
    action_dim = env.action_dim

    config = TrainConfig()
    gin.parse_config_files_and_bindings([config.gin_config_files], [])

    state_energy_model = energy_model(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.energy_hidden_layers,
        activation=config.ebm_activation,
        with_reward=True,
        spectral_norm=config.ebm_spectral_norm,
        layer_type=config.ebm_layer_type,
        device=config.device
    )
    transition_energy_model = energy_model(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.energy_hidden_layers,
        activation=config.ebm_activation,
        with_reward=True,
        spectral_norm=config.ebm_spectral_norm,
        layer_type=config.ebm_layer_type,
        device=config.device
    )
    policy_energy_model = energy_model(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.energy_hidden_layers,
        activation=config.ebm_activation,
        with_reward=True,
        spectral_norm=config.ebm_spectral_norm,
        layer_type=config.ebm_layer_type,
        device=config.device
    )
    rew_model = None

    diff_dims = state_dim + action_dim + 1 + state_dim
    if config.model_terminals:
        diff_dims += 1

    inputs = torch.zeros((128, diff_dims)).float()

    state_energy_optimizer = torch.optim.Adam(
        list(state_energy_model.parameters()), config.ebm_lr
    )
    transition_energy_optimizer = torch.optim.Adam(
        list(transition_energy_model.parameters()), config.ebm_lr
    )
    policy_energy_optimizer = torch.optim.Adam(
        list(policy_energy_model.parameters()), config.ebm_lr
    )
    rew_model_optim = None

    diffusion_trainer = REDQTrainer(
        construct_diffusion_model(
            inputs=inputs,
            skip_dims=[state_dim + action_dim],
            disable_terminal_norm=config.model_terminals,
            args=args
        ),
        state_energy=state_energy_model,
        transition_energy=transition_energy_model,
        policy_energy=policy_energy_model,
        ope_optim=state_energy_optimizer,
        te_optim=transition_energy_optimizer,
        pe_optim=policy_energy_optimizer,
        energy_train_epoch=config.energy_train_epoch,

        results_folder=os.path.join("logs", config.log_name),
        model_terminals=config.model_terminals,
        args=args,
        rew_model=rew_model,
        rew_model_optim=rew_model_optim
    )

    offline_buffer = calq_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    diffusion_buffer = calq_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    online_buffer = calq_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    print("start adding data")
    offline_buffer.add_transition_batch(offline_data)
    online_buffer.add_transition_batch(online_data)

    stable_online_batch = collect_data(online_agent, 100)

    print("start training")
    diffusion_trainer.update_normalizer(online_buffer, device=config.device)
    diffusion_trainer.train_from_redq_buffer(offline_buffer, online_buffer, num_steps=8000)
    diffusion_trainer.train_energy(online_buffer, online_agent, config.negative_sample, env=env)

    if rew_model is not None:
        diffusion_trainer.train_rew_model(offline_buffer)

    # Add samples to agent replay buffer
    print("start sampling")
    generator = SimpleDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model, rew_model=rew_model, sample_batch_size=250)
    diffusion_batch = generator.sample(clip=config.grad_clip,
                                       num_samples=3000,
                                       state_energy=state_energy_model,
                                       transition_energy=transition_energy_model,
                                       policy_energy=policy_energy_model)
    print("start comparing")
    # TODO: clip pred states
    pred_state = diffusion_batch[0]
    if not config.map_as_state:
        pred_state = pred_state.astype(int)
    
    cmax = 0.1
    cmin = -0.2

    if not os.path.exists('toy/image'):
        os.mkdir('toy/image')

    draw_dist(offline_data[0], "offline")
    draw_dist(stable_online_batch[0], "real")
    draw_dist(diffusion_batch[0], "diffusion")
    draw_dist(online_data[0], "less")

    print("start conditional diffusion genration")
    start_states, _, _, _, _, _ = offline_buffer.sample(200)

    states = start_states
    cum_states = []
    cum_actions = []
    cum_next_states = []
    cum_rewards = []
    cum_terminals = []
    for i in range(config.rollout_step):
        states = torch.as_tensor(states, dtype=torch.float32, device=config.device)
        actions = online_agent.take_action(states, stochastic=True)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=config.device)
        state_actions = torch.cat([states, actions], dim=1)
        rewards, next_states, terminals = generator.sample_wo_guidance_cond(
            num_samples=750, cond=state_actions)
        cum_states.append(states.detach().cpu().numpy())
        cum_actions.append(actions.detach().cpu().numpy())
        cum_next_states.append(next_states)
        cum_rewards.append(rewards)
        cum_terminals.append(terminals)
        states = next_states
    cum_states = np.reshape(np.concatenate(cum_states, axis=0), [-1, cum_states[0].shape[-1]])
    cum_actions = np.reshape(np.concatenate(cum_actions, axis=0), [-1, cum_actions[0].shape[-1]])
    cum_next_states = np.reshape(np.concatenate(cum_next_states, axis=0), [-1, cum_next_states[0].shape[-1]])
    cum_rewards = np.concatenate(cum_rewards, axis=0)
    cum_terminals = np.concatenate(cum_terminals, axis=0)
    diffusion_batch = (cum_states, cum_actions, cum_rewards, cum_next_states, cum_terminals)

    print(len(diffusion_batch[0]))
    draw_dist(diffusion_batch[0], "diffusion_transition")

    print("start training mlp")
    hidden_dim = 256
    model = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for i in range(1000):
        optimizer.zero_grad()
        state, action, reward, next_state, terminal, _ = offline_buffer.sample(16)
        pred_state = model(torch.cat([state, action], dim=1))
        loss = F.mse_loss(pred_state, next_state)
        loss.backward()
        optimizer.step()

    print("start conditional mlp genration")
    start_states, _, _, _, _, _ = offline_buffer.sample(750)

    states = start_states
    mlp_states = []
    for i in range(config.rollout_step):
        states = torch.as_tensor(states, dtype=torch.float32, device=config.device)
        actions = online_agent.take_action(states, stochastic=True)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=config.device)
        state_actions = torch.cat([states, actions], dim=1)
        next_states = model(state_actions)
        mlp_states.append(states.detach().cpu().numpy())
        states = next_states
    mlp_states = np.reshape(np.concatenate(mlp_states, axis=0), [-1, mlp_states[0].shape[-1]])
    draw_dist(mlp_states, "mlp_transition")
