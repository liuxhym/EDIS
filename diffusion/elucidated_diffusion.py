"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import gym
import math
from typing import Optional, Sequence, Tuple

import gin
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from tqdm import tqdm

from diffusion.norm import BaseNormalizer, MinMaxNormalizer


# from diffusion.utils import split_diffusion_samples
# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


EPSILON = 1e-20


def grad_wrt_next_s(
        energy_network,
        state_action,
        delta_state,
        create_graph: bool = False,
):
    delta_state.requires_grad = True
    energies = energy_network(state_action, delta_state)
    grad = torch.autograd.grad(energies.sum(), delta_state, create_graph=create_graph)[0]
    # grad = torch.autograd.grad(energies, delta_state, grad_outputs=torch.ones(energies.size()).to(energy_network.device), create_graph=create_graph)[0]
    return grad, energies


# main class
@gin.configurable
class ElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            normalizer: BaseNormalizer,
            event_shape: Sequence[int],  # shape of the input and output
            num_sample_steps: int = 32,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
            args=None
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.normalizer = normalizer
        self.clamp_samples = isinstance(self.normalizer, MinMaxNormalizer)
        # input dimensions
        self.event_shape = event_shape

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.args = args

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, clamp=False, cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        net_out = self.net(
            self.c_in(padded_sigma) * noised_inputs,
            self.c_noise(sigma),
            cond=cond,
        )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1 + EPSILON) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    def negative_generator(self,
                           condition,
                           num_samples,
                           num_sample_steps: int = 128,
                           sample_batch_size: int = 256,
                           env: Optional[gym.Env] = None
                           ):

        assert num_samples % sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // condition.shape[0]
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            # print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.negative_sample(
                condition=condition,
                batch_size=sample_batch_size,
                negative_step=num_sample_steps,
                clamp=self.clamp_samples,
            )

            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            # Split samples into (s, a, r, s') format
            obs = sampled_outputs[:, :obs_dim]
            acts = sampled_outputs[:, obs_dim:obs_dim + action_dim]
            rews = sampled_outputs[:, obs_dim + action_dim]
            next_obs = sampled_outputs[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
            transitions = obs, acts, rews, next_obs

            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew.reshape(-1, 1) if len(rew.shape) == 1 else rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals

    @torch.no_grad()
    def negative_sample(
            self,
            condition,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            negative_step = 10,
            clamp: bool = True,
            cond=None,
            disable_tqdm: bool = False,
    ):
        paddding = torch.zeros(condition.shape[0],
                               *[i - j for i, j in zip(self.event_shape, list(condition.shape)[1:])]).to(condition)
        condition_padded = torch.cat([condition, paddding], dim=-1)
        condition_padded = self.normalizer.normalize(condition_padded)
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        # shape = (batch_size, *self.event_shape)
        shape = (condition.shape[0], *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)
        mask = torch.zeros_like(inputs)
        mask[..., 0:condition.shape[0]] = 1

        # gradually denoise
        for sigma, sigma_next, gamma in sigmas_and_gammas[:negative_step]:  # tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1, disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            inputs_next = (1 - mask) * inputs_next + mask * condition_padded

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # @torch.no_grad()
    def energy_guidance(
            self,
            env,
            state_energy: Optional[nn.Module],
            transition_energy: Optional[nn.Module],
            policy_energy: Optional[nn.Module],
            inputs,
            grad_clip,
            timestep
    ):

        ope_mask, pe_mask, te_mask = torch.zeros_like(inputs), torch.zeros_like(inputs), torch.zeros_like(inputs)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        inputs = torch.cat([inputs, torch.ones(inputs.shape[0],1).to(inputs) * timestep], dim = -1)

        inputs.requires_grad = True
        state_derivative = torch.autograd.grad(state_energy(inputs).sum(), inputs)[0][:,:-1]
        ope_mask[..., 0:obs_dim] = 1
        state_guidance = ope_mask * state_derivative

        action_derivative = torch.autograd.grad(policy_energy(inputs).sum(), inputs)[0][:,:-1]
        pe_mask[..., obs_dim: obs_dim + action_dim] = 1
        policy_guidance = pe_mask * action_derivative

        next_state_derivative = torch.autograd.grad(transition_energy(inputs).sum(), inputs)[0][:,:-1]
        te_mask[..., obs_dim + action_dim + 1:] = 1
        transition_guidance = te_mask * next_state_derivative

        guidance = torch.zeros_like(state_guidance)
        if self.args.state_guide:
            guidance += state_guidance * self.args.ope_clip
        if self.args.transition_guide:
            guidance += transition_guidance * self.args.te_clip
        if self.args.policy_guide:
            guidance += policy_guidance * self.args.pe_clip

        if grad_clip:
            grad_norm = torch.norm(guidance, dim=-1, keepdim=True)
            target_norm = np.sqrt(guidance.shape[-1]) * 10.0
            ratio = torch.ones_like(grad_norm)
            ratio[grad_norm > target_norm] = target_norm / grad_norm[grad_norm > target_norm]
            # grad = grad / grad_norm * target_norm
            guidance = ratio * guidance
            # print("ratio:{}, guidance:{}".format(ratio.mean().item(), guidance.abs().mean().item()))
        inputs.requires_grad = False

        return guidance * grad_clip

    # @torch.no_grad()
    def sample(
            self,
            state_energy,
            transition_energy,
            policy_energy,
            env,
            clip,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            disable_tqdm: bool = False
    ):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        has_energy = state_energy is not None and transition_energy is not None and policy_energy is not None

        # gradually denoise
        timestep = 0
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            energy_inputs = inputs.clamp(-1., 1.) if clamp else inputs
            energy_inputs = self.normalizer.unnormalize(energy_inputs)
            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            if has_energy:
                denoised_over_sigma += self.energy_guidance(env, state_energy, transition_energy, policy_energy,
                                                            energy_inputs, clip, timestep)

            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                energy_inputs = inputs_next.clamp(-1., 1.) if clamp else inputs_next
                energy_inputs = self.normalizer.unnormalize(energy_inputs)
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)
                if has_energy:
                    denoised_prime_over_sigma += self.energy_guidance(env, state_energy, transition_energy,
                                                                      policy_energy, energy_inputs, clip, timestep)

                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)
            inputs = inputs_next
            timestep += 1
        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    def sample_wo_guidance(
            self,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            denoise_step: int = 0,
            clamp: bool = True,
            cond=None,
            disable_tqdm: bool = False
    ):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas[:denoise_step], desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)

            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond)

                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)
            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            clamp: bool = False,
            cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, inputs, cond=None):
        inputs = self.normalizer.normalize(inputs)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, cond=cond)
        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


@gin.configurable
class MLPModel(nn.Module):
    def __init__(
            self,
            net,
            input_normalizer: BaseNormalizer=None,
            output_normalizer: BaseNormalizer=None,
            args=None
    ):
        super().__init__()
        self.net = net
        self.normalizer_type = 'standard'
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.clamp_samples = False
        # parameters
        self.args = args

    @property
    def device(self):
        return next(self.net.parameters()).device

    def sample(self, inputs, clamp=False):
        inputs = self.input_normalizer.normalize(inputs)
        outputs = self.net(inputs)
        if clamp:
            outputs = outputs.clamp(-1., 1.)
        return self.output_normalizer.unnormalize(outputs)

    def forward(self, inputs, outputs):
        inputs = self.input_normalizer.normalize(inputs)
        output_preds = self.net(inputs)
        outputs = self.output_normalizer.unnormalize(outputs)

        losses = F.mse_loss(output_preds, outputs, reduction='none')
        return losses.mean()
