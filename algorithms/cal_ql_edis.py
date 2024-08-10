# source: https://github.com/nakamotoo/Cal-QL/tree/main
# https://arxiv.org/pdf/2303.05479.pdf
import os
import random
import uuid
import gin

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from common.logger import Logger
from common.network import MLP

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco_py
# import wandb
import argparse
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from common.buffer import calq_ReplayBuffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from common.energy import energy_model

from diffusion.trainer import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator 
from diffusion.utils import construct_diffusion_model
from diffusion.denoiser_network import ResidualMLPDenoiser



TensorBatch = List[torch.Tensor]

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


# @dataclass
@gin.configurable
class TrainConfig(object):
    # Experiment
    def __init__(self, 
                 utd_ratio = 1, 
                 offline_mixing_ratio = 0.5,
                 normalize = True,
                 normalize_reward = True,
                 q_n_hidden_layers = 2,
                 cql_alpha = 10,
                 cql_alpha_online = 0,
                 cql_lagrange = True,
                 reward_scale = 1, 
                 reward_bias = 0,
                 is_sparse_reward = True,
                 energy_hidden_layers = 3,
                 cql_max_target_backup = False,
                 num_negative_sample = 10,
                 cql_clip_diff_min = -np.inf,
                 cql_clip_diff_max = np.inf,
                 cql_target_action_gap = 10,
                 ope_clip = 2,
                 pe_clip = 0.1,
                 te_clip = 0.1,
                 retrain_diffusion_every = 10000) -> None:
        self.device: str = "cuda"
        self.env: str = "walker2d-random-v2"  # OpenAI gym environment name
        self.description: str = "test original"  # OpenAI gym environment name
        self.seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
        self.eval_seed: int = 0  # Eval environment seed
        self.eval_freq: int = int(1e4)  # How often (time steps) we evaluate
        self.n_episodes: int = 10  # How many episodes run during evaluation
        self.offline_iterations: int = int(3e5)  # Number of offline updates
        self.online_iterations: int = int(2e5+10)  # Number of online updates
        self.checkpoints_path: Optional[str] = None # Save path
        self.load_model: str = ""  # Model load file name, "" doesn't load
        # CQL
        self.buffer_size: int = 2_000_000  # Replay buffer size
        self.batch_size: int = 256  # Batch size for all networks
        self.discount: float = 0.99  # Discount factor
        self.alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
        self.use_automatic_entropy_tuning: bool = True  # Tune entropy
        self.backup_entropy: bool = True  # Use backup entropy
        self.policy_lr: float = 1e-4  # Policy learning rate
        self.qf_lr: float = 3e-4  # Critics learning rate
        self.soft_target_update_rate: float = 5e-3  # Target network update rate
        self.bc_steps: int = int(0)  # Number of BC steps at start
        self.target_update_period: int = 1  # Frequency of target nets updates
        self.cql_alpha: float = cql_alpha  # CQL offline regularization parameter
        self.cql_alpha_online: float = cql_alpha_online  # CQL online regularization parameter
        self.cql_n_actions: int = 10  # Number of sampled actions
        self.cql_importance_sample: bool = True  # Use importance sampling
        self.cql_lagrange: bool = cql_lagrange  # Use Lagrange version of CQL
        self.cql_target_action_gap: float = cql_target_action_gap  # Action gap
        self.cql_temp: float = 1.0  # CQL temperature
        self.cql_max_target_backup: bool = cql_max_target_backup  # Use max target backup
        self.cql_clip_diff_min: float = cql_clip_diff_min  # Q-function lower loss clipping
        self.cql_clip_diff_max: float = cql_clip_diff_max # Q-function upper loss clipping
        self.orthogonal_init: bool = True  # Orthogonal initialization
        self.normalize: bool = normalize  # Normalize states
        self.normalize_reward: bool = normalize_reward  # Normalize reward
        self.q_n_hidden_layers: int = q_n_hidden_layers  # Number of hidden layers in Q networks
        self.reward_scale: float = reward_scale  # Reward scale for normalization
        self.reward_bias: float = reward_bias  # Reward bias for normalization
        self.offline_mixing_ratio: float = offline_mixing_ratio  # Data mixing ratio for online tuning
        self.is_sparse_reward: bool = is_sparse_reward  # Use sparse reward
        self.log_name: str = "cal_ql_edis"  # name for the log directory 


        # diffusion
        self.utd_ratio: int = utd_ratio
        self.energy_hidden_layers: int = energy_hidden_layers
        self.ebm_activation: str = "relu"
        self.ebm_layer_type: str = "MLP"
        self.ebm_spectral_norm: bool = True
        self.ebm_lr: float = 1e-3
        self.num_negative_sample: int = num_negative_sample
        self.energy_train_epoch: int = 20
        self.grad_clip: float = 1
        self.ope_clip: float = ope_clip
        self.te_clip: float = te_clip
        self.pe_clip: float = pe_clip

        
        self.model_terminals: bool = False
        self.num_samples: int = 100000
        self.retrain_diffusion_every: int = retrain_diffusion_every
        self.diffusion_start: int = 0

    def __post_init__(self):
        self.name = f"{self.log_name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)



def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
                # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def get_return_to_go(dataset: Dict, env: gym.Env, config: TrainConfig) -> np.ndarray:
    returns = []
    ep_ret, ep_len = 0.0, 0
    cur_rewards = []
    terminals = []
    N = len(dataset["rewards"])
    for t, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len += 1
        is_last_step = (
            (t == N - 1)
            or (
                np.linalg.norm(
                    dataset["observations"][t + 1] - dataset["next_observations"][t]
                )
                > 1e-6
            )
            or ep_len == env._max_episode_steps
        )

        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0
            if (
                config.is_sparse_reward
                and r
                == env.ref_min_score * config.reward_scale + config.reward_bias
            ):
                discounted_returns = [r / (1 - config.discount)] * ep_len
            else:
                for i in reversed(range(ep_len)):
                    discounted_returns[i] = cur_rewards[
                        i
                    ] + config.discount * prev_return * (1 - terminals[i])
                    prev_return = discounted_returns[i]
            returns += discounted_returns
            ep_ret, ep_len = 0.0, 0
            cur_rewards = []
            terminals = []
    return returns


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
) -> Dict:
    modification_data = {}
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        modification_data = {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return modification_data


def modify_reward_online(
    reward: float,
    env_name: str,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    **kwargs,
) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    reward = reward * reward_scale + reward_bias
    return reward


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.base_network[-1], False)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        else:
            init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class CalQLEDIS:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mc_returns: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        # Calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(
            1, cql_q1_current_actions.shape[1]
        )

        num_vals = torch.sum(lower_bounds == lower_bounds)
        bound_rate_cql_q1_current_actions = (
            torch.sum(cql_q1_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_current_actions = (
            torch.sum(cql_q2_current_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q1_next_actions = (
            torch.sum(cql_q1_next_actions < lower_bounds) / num_vals
        )
        bound_rate_cql_q2_next_actions = (
            torch.sum(cql_q2_next_actions < lower_bounds) / num_vals
        )

        """ Cal-QL: bound Q-values with MC return-to-go """
        if self._calibration_enabled:
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
                bound_rate_cql_q1_current_actions=bound_rate_cql_q1_current_actions.item(),  # noqa
                bound_rate_cql_q2_current_actions=bound_rate_cql_q2_current_actions.item(),  # noqa
                bound_rate_cql_q1_next_actions=bound_rate_cql_q1_next_actions.item(),
                bound_rate_cql_q2_next_actions=bound_rate_cql_q2_next_actions.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            mc_returns,
        ) = batch
        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            dones,
            mc_returns,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])
        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        # # self.alpha_optimizer.load_state_dict(
        # #     state_dict=state_dict["sac_log_alpha_optim"]
        # # )
        self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        
        self.log_alpha_prime = state_dict["cql_log_alpha"]
        # # self.alpha_prime_optimizer.load_state_dict(
        # #     state_dict=state_dict["cql_log_alpha_optim"]
        # # )
        self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.qf_lr,
            )
        self.total_it = state_dict["total_it"]


# @pyrallis.wrap()
# def train(args, config=config):
def train(args):#, config=config):

    names = args.env.split('-')
    env_name = names[0]
    other_name = ''
    for name in names[1:-1]:
        other_name = other_name + name + '-'
    other_name = other_name[:-1]
    gin_config_files = 'configs/cal_ql_edis/' + env_name + '/' + other_name + '.gin'

    gin.parse_config_files_and_bindings([gin_config_files], [])

    config=TrainConfig()
    config.env = args.env 
    config.seed = args.seed
    config.log_name = args.log_name
    args.grad_clip = config.grad_clip
    args.ope_clip = config.ope_clip
    args.te_clip = config.te_clip
    args.pe_clip = config.pe_clip
    
    # env = gym.make(config.env)
    # eval_env = gym.make(config.env)
    log_dir = os.path.join("logs", config.log_name)
    logger = Logger(log_dir, config.env, config.seed)
    logger.log_str_object("parameters", log_dict = config.__dict__)

    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    batch_size_offline = int(config.batch_size * config.offline_mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )
    mc_returns = get_return_to_go(dataset, env, config)
    dataset["mc_returns"] = np.array(mc_returns)
    assert len(dataset["mc_returns"]) == len(dataset["rewards"])

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    
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
    rew_model_optim = None

    state_energy_optimizer = torch.optim.Adam(
        list(state_energy_model.parameters()), config.ebm_lr
    )
    transition_energy_optimizer = torch.optim.Adam(
        list(transition_energy_model.parameters()), config.ebm_lr
    )
    policy_energy_optimizer = torch.optim.Adam(
        list(policy_energy_model.parameters()), config.ebm_lr
    )

    offline_replay_buffer = calq_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    replay_buffer = calq_ReplayBuffer(
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

    diffusion_replay_buffer = calq_ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    offline_replay_buffer.load_d4rl_dataset(dataset)
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    delimiter_index = config.env.find("-")
    env_name = config.env[:delimiter_index]
    # real_model = mujoco_py.load_model_from_path("asset/{}.xml".format(env_name))
    # sim = mujoco_py.MjSim(real_model)
    if env_name == "hopper":
        test_env = gym.make('Hopper-v3')
    elif env_name == "halfcheetah":
        test_env = gym.make('HalfCheetah-v3') 
    elif env_name == "walker2d":
        test_env = gym.make('Walker2d-v3')

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    logger.log_str("---------------------------------------")
    print(f"Training Cal-QL, Env: {config.env}, Seed: {seed}")
    logger.log_str(f"Training Cal-QL-EDIS, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    logger.log_str("---------------------------------------")

    # Initialize actor
    trainer = CalQLEDIS(**kwargs)


    diff_dims = state_dim + action_dim + 1 + state_dim 
    if config.model_terminals:
        diff_dims += 1

    inputs = torch.zeros((128, diff_dims)).float()

    evaluations = []
    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    logger.log_str("Offline pretraining")
    if config.load_model != "":
        # policy_file = Path(config.load_model)
        policy_path = "save/cal_ql/{}".format(config.env)
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        policy_file = policy_path + "/{}".format(config.env, config.load_model)
        if os.path.exists(policy_file):
            trainer.load_state_dict(torch.load(policy_file))
            config.offline_iterations = 0
            actor = trainer.actor
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):

        if t == config.offline_iterations:
            logger.log_str("Online tuning")
            config.eval_freq = 1e4
            model_state = trainer.state_dict()
            torch.save(model_state, os.path.join(logger.log_path, "offline_model.pth"))

            trainer.switch_calibration()
            trainer.cql_alpha = config.cql_alpha_online
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1),
                    device=config.device,
                    dtype=torch.float32,
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(
                    reward,
                    config.env,
                    reward_scale=config.reward_scale,
                    reward_bias=config.reward_bias,
                    **reward_mod_dict,
                )
            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            online_buffer.add_transition(state, action, reward, next_state, real_done)

            state = next_state

            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = (
                    normalized_return * 100.0
                )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False


            if (t + 1) % config.retrain_diffusion_every == 0 or t == config.offline_iterations:
                # Train new diffusion model
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
                
                diffusion_trainer.update_normalizer(replay_buffer, device=config.device)
                diffusion_trainer.train_from_redq_buffer(replay_buffer)
                diffusion_trainer.train_energy(online_buffer, actor, config.num_negative_sample, env=env)
                if rew_model is not None:
                    diffusion_trainer.train_rew_model(replay_buffer)

                diffusion_replay_buffer = calq_ReplayBuffer(
                    state_dim,
                    action_dim,
                    config.buffer_size,
                    config.device,
                )

                # Add samples to agent replay buffer
                generator = SimpleDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model, rew_model=rew_model)
                diffusion_batch = generator.sample(clip=config.grad_clip,
                                                    num_samples=config.num_samples, 
                                                   state_energy=state_energy_model,
                                                   transition_energy=transition_energy_model,
                                                   policy_energy=policy_energy_model)


                diffusion_replay_buffer.add_transition_batch(diffusion_batch)
                batch = replay_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline, batch_size_online, config.device)
                if args.test_divergence:
                    real_act, real_log_prob = actor(batch[0])
                    assert(real_act.shape == batch[1].shape)
                    policy_est = F.mse_loss(real_act, batch[1])

                    state_div = 0
                    rew_div = 0
                    for i in range(len(batch[0])):
                        div_index = len(batch[0][i]) // 2
                        
                        qpos = batch[0][i][:div_index].cpu().numpy() 
                        qvel = batch[0][i][div_index:].cpu().numpy() 
                        qpos = np.append(0, qpos)
                        test_env.reset()
                        test_env.set_state(qpos, qvel)
                        # sim.data.qpos[:] = qpos
                        # sim.data.qvel[:] = qvel

                        action = batch[1][i].cpu().numpy() 
                        # sim.data.ctrl[:] = action
                        new_state, r, d, info = test_env.step(action)

                        # sim.step()

                        # new_qpos = sim.data.qpos[1:]
                        # new_qpos[0] = 0
                        # new_qvel = sim.data.qvel
                        # new_state = np.append(new_qpos, new_qvel)
                        state_div += np.sum((new_state - batch[3][i].cpu().numpy()) ** 2)
                        rew_div += (r - batch[2][i].cpu().numpy()) ** 2
                    rew_div = rew_div / len(batch[0])
                    state_div = state_div / len(batch[0]) / 11


        utd_ratio = config.utd_ratio if t >= config.offline_iterations  else 1 
            
        if t < config.offline_iterations:
            batch = offline_replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
        else:
            batch = online_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline, batch_size_online, config.device)
            batch = [b.to(config.device) for b in batch]
            for _ in range(utd_ratio - 1):
                if utd_ratio > 1 :
                    batch = online_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline, batch_size_online, config.device)
                batch = [b.to(config.device) for b in batch]
                log_dict = trainer.train(batch)


        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        


        # Evaluate episode
        if (t != 0 and t % config.eval_freq == 0) or t  == config.offline_iterations:
            if args.test_divergence and t >= config.offline_iterations:
                eval_index = {}
                eval_index["eval/policy_est"] = policy_est
                eval_index["eval/state_div"] = state_div
                log_dict.update(eval_index)

            logger.log(log_dict, step = trainer.total_it)
            print(f"Time steps: {t}")
            logger.log_str(f"Time steps: {t}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log, performance_log = {}, {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            if t < config.offline_iterations :
                performance_log["eval/d4rl_offline_normalized_score"] = normalized_eval_score
            else:
                performance_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            logger.log_str("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            logger.log_str(f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}")
            print("---------------------------------------")
            if args.test_divergence and t >= config.offline_iterations:
                logger.log_str(f"Policy divergence:{policy_est}")
                logger.log_str(f"State divergence:{state_div}")
                logger.log_str(f"Reward divergence:{rew_div}")
            print("---------------------------------------")
            logger.log_str("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            logger.log(eval_log, step=trainer.total_it)

            performance_log = {"eval/d4rl_offline_normalized_score": performance_log["eval/d4rl_offline_normalized_score"]} if t < config.offline_iterations  \
                else {"eval/d4rl_normalized_score": performance_log["eval/d4rl_normalized_score"]}
            logger.log(performance_log, step = t if t < config.offline_iterations else t - config.offline_iterations)
            # wandb.log(eval_log, step=trainer.total_it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-umaze-v2") ################ task
    parser.add_argument("--seed", type=int, default=10) ############################ seed
    parser.add_argument("--log_name", type=str, default="cal_ql_edis")
    parser.add_argument("--policy_guide", action='store_true', default=True)
    parser.add_argument("--state_guide", action='store_true', default=True)
    parser.add_argument("--transition_guide", action='store_true', default=True)
    parser.add_argument("--test_divergence", type=bool, default=False)

    parser.add_argument("--ope_clip", type=float, default=0.1)
    parser.add_argument("--te_clip", type=float, default=0.1)
    parser.add_argument("--pe_clip", type=float, default=0.1)
    args = parser.parse_args()
    train(args=args)
