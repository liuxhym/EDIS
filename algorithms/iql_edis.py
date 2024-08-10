# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import gin
import random
import uuid
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from common.logger import Logger

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.buffer import calq_ReplayBuffer, RewardNormalizer, StateNormalizer, DiffusionConfig
from common.energy import energy_model

from diffusion.trainer import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator 
from diffusion.utils import construct_diffusion_model
from diffusion.denoiser_network import ResidualMLPDenoiser

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@gin.configurable
class TrainConfig(object):
    # Experiment
    def __init__(self, utd_ratio=20, offline_mixing_ratio=0.5, beta=10.0, iql_tau=0.9, \
    tau=0.005, normalize=True, normalize_reward=False):
    # Experiment
        self.device: str = "cuda"
        self.env: str = "walker2d-random-v2"  # OpenAI gym environment name
        self.seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
        self.eval_seed: int = 0  # Eval environment seed
        self.eval_freq: int = int(5e4)  # How often (time steps) we evaluate
        self.n_episodes: int = 100  # How many episodes run during evaluation
        self.offline_iterations: int = int(1e6)  # Number of offline updates
        self.online_iterations: int = int(2e5+10)  # Number of online updates
        self.checkpoints_path: Optional[str] = None  # Save path
        self.load_model: str = ""  # Model load file name, "" doesn't load
        # IQL
        self.actor_dropout: float = 0.0  # Dropout in actor network
        self.buffer_size: int = 2_000_000  # Replay buffer size
        self.batch_size: int = 256  # Batch size for all networks
        self.discount: float = 0.99  # Discount factor
        self.tau: float = tau  # Target network update rate
        self.beta: float = beta  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
        self.iql_tau: float = iql_tau  # Coefficient for asymmetric loss
        self.expl_noise: float = 0.03  # Std of Gaussian exploration noise
        self.noise_clip: float = 0.5  # Range to clip noise
        self.iql_deterministic: bool = False  # Use deterministic actor
        self.normalize: bool = normalize  # Normalize states
        self.normalize_reward: bool = normalize_reward  # Normalize reward
        self.vf_lr: float = 3e-4  # V function learning rate
        self.qf_lr: float = 3e-4  # Critic learning rate
        self.actor_lr: float = 3e-4  # Actor learning rate
        self.log_name: str = "iql_edis"   # name for the log directory 

        self.utd_ratio: int = 20  # Update the diffusion model every utd_ratio steps
        self.energy_hidden_layers: int = 3
        self.ebm_activation: str = "relu"
        self.ebm_layer_type: str = "MLP"
        self.ebm_spectral_norm: bool = True
        self.ebm_lr: float = 1e-3
        self.num_negative_sample: int = 10 
        self.energy_train_epoch: int = 20
        self.grad_clip: float = 1
        self.ope_clip: float = 2.0
        self.te_clip: float = 0.1
        self.pe_clip: float = 0.1
        self.offline_mixing_ratio: float = offline_mixing_ratio
        self.model_terminals: bool = False
        self.num_samples: int = 100000
        self.retrain_diffusion_every: int = 10000
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


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class IQLEDIS:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            mc_returns,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


# @pyrallis.wrap()
config=None
def train(args, config=config):
    names = args.env.split('-')
    env_name = names[0]
    other_name = ''
    for name in names[1:-1]:
        other_name = other_name + name + '-'
    other_name = other_name[:-1]
    gin_config_files = 'configs/iql_edis/' + env_name + '/' + other_name + '.gin'

    gin.parse_config_files_and_bindings([gin_config_files], [])
    config = TrainConfig()
    config.env = args.env 
    config.seed = args.seed
    args.grad_clip = config.grad_clip
    args.ope_clip = config.ope_clip
    args.te_clip = config.te_clip
    args.pe_clip = config.pe_clip

    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    log_dir = os.path.join("logs", config.log_name)
    logger = Logger(log_dir, config.env, config.seed, info_str=args.comment)
    logger.log_str_object("parameters", log_dict = config.__dict__)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    batch_size_offline = int(config.batch_size * config.offline_mixing_ratio)
    batch_size_online = config.batch_size - batch_size_offline

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

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

    dataset["mc_returns"] = np.zeros_like(dataset["rewards"])
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

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
    }

    print("---------------------------------------")
    logger.log_str("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    logger.log_str(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    logger.log_str("---------------------------------------")

    # Initialize actor
    trainer = IQLEDIS(**kwargs)

    evaluations = []

    # set up diffusion model
    diff_dims = state_dim + action_dim + 1 + state_dim
    if config.model_terminals:
        diff_dims += 1

    inputs = torch.zeros((128, diff_dims)).float()

    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = [] 

    import time
    start_time = time.time()
    logger.log_str("Offline pretraining")
    if config.load_model != "":
        policy_path = "save/iql_edis/{}".format(config.env)
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        policy_file = policy_path + "/{}".format(config.env, config.load_model)
        if os.path.exists(policy_file):
            trainer.load_state_dict(torch.load(policy_file))
            config.offline_iterations = 0
            actor = trainer.actor

    for t in range(int(config.offline_iterations) + int(config.online_iterations)):


        if t == config.offline_iterations:
            online_time = time.time()
            offline_time = online_time - start_time
            print('-------------------------------')
            print("offline time", offline_time)
            print('-------------------------------')
            logger.log_str("Online tuning")
            config.eval_freq = 5e3
            model_state = trainer.state_dict()
            torch.save(model_state, os.path.join(logger.log_path, "offline_model.pth"))

        online_log = {}

        batch = replay_buffer.sample(config.batch_size)

        if t >= config.offline_iterations:
            episode_step += 1
            action = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            if not config.iql_deterministic:
                action = action.sample()
            else:
                noise = (torch.randn_like(action) * config.expl_noise).clamp(
                    -config.noise_clip, config.noise_clip
                )
                action += noise
            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)

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

            if (t + 1) % config.retrain_diffusion_every == 0 and (t + 1) >= config.diffusion_start:
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
                )
                diffusion_trainer.update_normalizer(replay_buffer, device=config.device)
                diffusion_trainer.train_from_redq_buffer(replay_buffer)
                diffusion_trainer.train_energy(online_buffer, actor, config.num_negative_sample, env=env)

                diffusion_replay_buffer = calq_ReplayBuffer(
                    state_dim,
                    action_dim,
                    config.buffer_size,
                    config.device,
                )

                # Add samples to agent replay buffer
                generator = SimpleDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model)
                diffusion_batch = generator.sample(clip=config.grad_clip,
                                                    num_samples=config.num_samples, 
                                                   state_energy=state_energy_model,
                                                   transition_energy=transition_energy_model,
                                                   policy_energy=policy_energy_model)
                # observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=config.num_samples)
                # for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                #     diffusion_replay_buffer.add_transition(o, a, r, o2, term)
                diffusion_replay_buffer.add_transition_batch(diffusion_batch)
                batch = online_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline, batch_size_online, config.device)
        
        utd_ratio = config.utd_ratio if t >= config.offline_iterations and (t + 1) % config.retrain_diffusion_every == 0 else 1 
        
        for _ in range(utd_ratio):
            if utd_ratio > 1 :
                batch = online_buffer.combine_replay_buffer(diffusion_replay_buffer, batch_size_offline, batch_size_online, config.device)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)


        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)

        # Evaluate episode
        if (t != 0 and t % config.eval_freq == 0) or t  == config.offline_iterations:
            logger.log(log_dict, step=trainer.total_it)

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
            normalized = eval_env.get_normalized_score(eval_score)
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            evaluations.append(normalized_eval_score)
            # eval_log["eval/d4rl_normalized_score"] = normalized_eval_score

            if t < config.offline_iterations :
                performance_log["eval/d4rl_offline_normalized_score"] = normalized_eval_score
            else:
                performance_log["eval/d4rl_normalized_score"] = normalized_eval_score

            print("---------------------------------------")
            logger.log_str("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            logger.log_str(f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}")
            print("---------------------------------------")
            logger.log_str("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            logger.log(eval_log, step=trainer.total_it)

            performance_log = {"eval/d4rl_offline_normalized_score": performance_log["eval/d4rl_offline_normalized_score"]} if t < config.offline_iterations  \
                else {"eval/d4rl_normalized_score": performance_log["eval/d4rl_normalized_score"]}
            logger.log(performance_log, step = t if t < config.offline_iterations else t - config.offline_iterations)

            try:
                evaluate_end = time.time()
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker2d-random-v2") ################ task
    parser.add_argument("--seed", type=int, default=0) ############################ seed
    parser.add_argument("--policy_guide", action='store_true', default=False)
    parser.add_argument("--state_guide", action='store_true', default=False)
    parser.add_argument("--transition_guide", action='store_true', default=False)
    parser.add_argument("--test_divergence", type=bool, default=False)
    parser.add_argument("--comment", type=str, default='')

    parser.add_argument("--ope_clip", type=float, default=1.0)
    parser.add_argument("--te_clip", type=float, default=0.1)
    parser.add_argument("--pe_clip", type=float, default=0.1)
    args = parser.parse_args()
    train(args)