import gin
from typing import Optional, Sequence, Tuple
import torch
from multiprocessing import cpu_count
import pathlib
import tqdm
import torch.nn as nn
import numpy as np

from accelerate import Accelerator
from ema_pytorch import EMA
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusion.train_diffuser import SimpleDiffusionGenerator
from common.buffer import calq_ReplayBuffer




def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def sample_schedule(num_sample_steps=None, device=None,
            epsilon = 1e-20, 
            rho = 7,
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
):
    num_sample_steps = default(num_sample_steps, num_sample_steps)

    N = num_sample_steps
    inv_rho = 1 / rho

    steps = torch.arange(num_sample_steps, device=device, dtype=torch.float32)
    sigmas = (sigma_max ** inv_rho + steps / (N - 1 + epsilon) * (
            sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

    sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
    return sigmas


@gin.configurable
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 16,
            small_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        if dataset is not None:
            # If dataset size is less than 800K use the small batch size
            if len(dataset) < int(8e5):
                self.batch_size = small_batch_size
            else:
                self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == 'linear':
            print('using linear learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using cosine learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = (next(self.dl)[0]).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({
                    'step': self.step,
                    'loss': total_loss,
                    'lr': self.opt.param_groups[0]['lr']
                })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('training complete')

    # Allow user to pass in external data.
    def train_on_batch(
            self,
            data: torch.Tensor,
            use_wandb=True,
            splits=1,  # number of splits to split the batch into
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)

        total_loss = 0.
        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(data, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and data.shape[0] % splits == 0
            split_data = torch.split(data, data.shape[0] // splits)

            for idx, d in enumerate(split_data):
                with self.accelerator.autocast():
                    # Split condition as well
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            new_kwargs[k] = torch.split(v, v.shape[0] // splits)[idx]
                        else:
                            new_kwargs[k] = v

                    loss = self.model(d, **new_kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log({
                'step': self.step,
                'loss': total_loss,
                'lr': self.opt.param_groups[0]['lr'],
            })

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss

@gin.configurable
class REDQTrainer(Trainer):
    def __init__(
            self,
            diffusion_model,
            ope_optim: Optional[torch.optim.Optimizer] = None,
            te_optim: Optional[torch.optim.Optimizer] = None,
            pe_optim: Optional[torch.optim.Optimizer] = None,
            state_energy: Optional[nn.Module] = None,
            transition_energy: Optional[nn.Module] = None,
            policy_energy: Optional[nn.Module] = None,
            ebm_batch_size: int = 256,
            train_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = None,
            train_num_steps: int = 100000,
            energy_train_epoch: int = 100,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 10000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            model_terminals: bool = False,
            args=None,
            rew_model=None,
            rew_model_optim=None
    ):
        super().__init__(
            diffusion_model,
            dataset=None,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            train_lr=train_lr,
            lr_scheduler=lr_scheduler,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            weight_decay=weight_decay,
            results_folder=results_folder,
            amp=amp,
            fp16=fp16,
            split_batches=split_batches,
        )

        self.model_terminals = model_terminals
        self.ebm_batch_size = ebm_batch_size

        self.ope_optim = ope_optim
        self.pe_optim = pe_optim
        self.te_optim = te_optim

        self.state_energy = state_energy
        self.policy_energy = policy_energy
        self.transition_energy = transition_energy
        self.energy_train_epoch = energy_train_epoch
        self.args = args
        self.rew_model = rew_model
        self.rew_model_optim = rew_model_optim


    def train_energy(self,
                     online_buffer: calq_ReplayBuffer,
                     actor,
                     num_negative_sample,
                     env):

        for _ in range(self.energy_train_epoch):
            online_batch = online_buffer.sample(self.ebm_batch_size)

            if self.args.state_guide:
                # ope_loss = self.train_state_energy(online_batch, replay_batch)
                ope_loss = self.train_state_energy(online_batch, num_negative_sample, env)
                print('ope_loss: {}'.format(ope_loss))
            if self.args.transition_guide:
                te_loss = self.train_transition_energy(online_batch, num_negative_sample, env)
                print('te_loss: {}'.format(te_loss))
            if self.args.policy_guide:
                pe_loss = self.train_policy_energy(online_batch, actor, num_negative_sample, env)
                print('pe_loss: {}'.format(pe_loss))


    def train_rew_model(self, buffer, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
            data = [states, actions]
            accelerator = self.accelerator
            device = accelerator.device
            data = torch.cat(data, dim=1).to(device)

            pred_rew = self.rew_model(data)
            rewards = rewards.to(device)
            assert (pred_rew.shape == rewards.shape)
            loss = F.mse_loss(pred_rew, rewards)

            self.rew_model_optim.zero_grad()
            loss.backward()
            self.rew_model_optim.step()
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')



    def train_state_energy(self, fast_batch, num_negative_sample, env, contrastive_step = 128):
        device = self.accelerator.device
        (
            fast_states,
            fast_actions,
            fast_rewards,
            fast_next_states,
            fast_dones,
            fast_mc_returns,
        ) = fast_batch

        batch_size = fast_states.shape[0]
        generator = SimpleDiffusionGenerator(env=env, ema_model=self.ema.ema_model, sample_batch_size=batch_size)

        sigma = sample_schedule(num_sample_steps=generator.num_sample_steps, device=device)

        fast_data = torch.tensor([], device=device)
        slow_data = torch.tensor([], device=device)
        for step in range(0, generator.num_sample_steps, contrastive_step):

            action_noise = sigma[step] * torch.randn(fast_actions.shape, device=device)
            state_noise = sigma[step] * torch.randn(fast_states.shape, device=device)
            reward_noise = sigma[step] * torch.randn(fast_rewards.shape, device=device)
            next_state_noise = sigma[step] * torch.randn(fast_next_states.shape, device=device)

            negative_samples = generator.sample_wo_guidance(num_samples=batch_size * num_negative_sample, denoise_step=step)
            (
                slow_states,
                slow_actions,
                slow_rewards,
                slow_next_states,
                slow_dones,
            ) = [torch.tensor(negative_samples[i]).to(fast_states) for i in range(len(negative_samples))]

            intermi_fast_data = torch.cat([fast_states + state_noise, fast_actions + action_noise, fast_rewards + reward_noise, fast_next_states + next_state_noise, torch.ones_like(fast_rewards).to(fast_rewards) * step], dim=-1)

            slow_rewards = slow_rewards.unsqueeze(dim=-1)
            intermi_slow_data = torch.cat([slow_states, slow_actions, slow_rewards, slow_next_states, torch.ones_like(slow_rewards).to(slow_rewards) * step], dim=-1)

            fast_data = torch.cat((fast_data, intermi_fast_data), dim=0)
            slow_data = torch.cat((slow_data, intermi_slow_data), dim=0)

        prediction = torch.cat([fast_data, slow_data])
        slow_pred = self.state_energy(slow_data)
        fast_pred = self.state_energy(fast_data)
        prediction = self.state_energy(prediction)

        ones = torch.ones_like(fast_pred)
        zeros = torch.zeros_like(slow_pred)

        slow_logsoftmax = - slow_pred - torch.logsumexp(- prediction, dim=-1, keepdim=True)
        fast_logsoftmax = - fast_pred - torch.logsumexp(- prediction, dim=-1, keepdim=True)

        loss = torch.nn.KLDivLoss(reduction="none")(fast_logsoftmax, ones).sum() + torch.nn.KLDivLoss(reduction="none")(
            slow_logsoftmax, zeros).sum()
        self.ope_optim.zero_grad()
        loss.backward()
        self.ope_optim.step()
        return loss


    def train_transition_energy(self, online_batch, negative_sample, env, contrastive_step=16):
        device = self.accelerator.device
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            mc_returns,
        ) = online_batch
        batch_size = states.shape[0]
        state_action = torch.repeat_interleave(torch.cat([states, actions], dim=-1), negative_sample, dim=0)

        sigma = sample_schedule(num_sample_steps=self.ema.ema_model.num_sample_steps, device=device)

        positive_samples = torch.tensor([], device=device)
        negative_samples = torch.tensor([], device=device)
        for step in range(0, self.ema.ema_model.num_sample_steps, contrastive_step):

            intermi_negative_samples = self.ema.ema_model.negative_generator(state_action, batch_size * negative_sample, num_sample_steps=step, env=env)
            intermi_negative_samples = torch.cat([torch.tensor(i).to(state_action) for i in list(intermi_negative_samples)[:-1]], dim=-1)

            intermi_negative_samples = torch.cat([intermi_negative_samples, torch.ones(intermi_negative_samples.shape[0],1).to(intermi_negative_samples) * step], dim = -1)

            action_noise = sigma[step] * torch.randn(actions.shape, device=device)
            state_noise = sigma[step] * torch.randn(states.shape, device=device)
            reward_noise = sigma[step] * torch.randn(rewards.shape, device=device)
            next_state_noise = sigma[step] * torch.randn(next_states.shape, device=device)

            intermi_positive_samples = torch.cat([states + state_noise, actions + action_noise, rewards + reward_noise, next_states + next_state_noise, torch.ones_like(dones).to(dones) * step], dim=-1)
            positive_samples = torch.cat((positive_samples, intermi_positive_samples), dim=0)
            negative_samples = torch.cat((negative_samples, intermi_negative_samples), dim=0)


        positive_pred = self.transition_energy(positive_samples)
        negative_pred = self.transition_energy(negative_samples)

        negative_pred_unsq = negative_pred.view(batch_size, negative_sample, -1)
        positive_pred_unsq = positive_pred.view(batch_size, 1, -1)

        pred_unsq = torch.cat([negative_pred_unsq, positive_pred_unsq], dim=1)

        logsumexp = torch.logsumexp(-pred_unsq, dim=1, keepdim=True)
        negative_logsumexp = torch.repeat_interleave(logsumexp, negative_sample, dim=1)

        positive_logsoftmax = (- positive_pred_unsq - logsumexp).view(-1, 1)
        negative_logsoftmax = (- negative_pred_unsq - negative_logsumexp).view(-1, negative_sample)

        ones = torch.ones_like(positive_logsoftmax)
        zeros = torch.zeros_like(negative_logsoftmax)

        loss = torch.nn.KLDivLoss(reduction="none")(positive_logsoftmax, ones).sum() + torch.nn.KLDivLoss(
            reduction="none")(negative_logsoftmax, zeros).sum()
        self.te_optim.zero_grad()
        loss.backward()
        self.te_optim.step()
        return loss

    def train_policy_energy(self, online_batch, actor, negative_sample, env, contrastive_step=16):
        device = self.accelerator.device
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            mc_returns,
        ) = online_batch

        batch_size = states.shape[0]
        negative_states = torch.repeat_interleave(states, negative_sample, dim=0)

        sigma = sample_schedule(num_sample_steps=self.ema.ema_model.num_sample_steps, device=device)

        positive_samples = torch.tensor([], device=device)
        negative_samples = torch.tensor([], device=device)
        for step in range(0, self.ema.ema_model.num_sample_steps, contrastive_step):

            intermi_negative_samples = self.ema.ema_model.negative_generator(negative_states, batch_size * negative_sample, num_sample_steps = step, env=env)
            intermi_negative_samples = torch.cat([torch.tensor(i).to(states) for i in list(intermi_negative_samples)[:-1]], dim=-1)
            intermi_negative_samples = torch.cat([intermi_negative_samples, torch.ones(intermi_negative_samples.shape[0],1).to(intermi_negative_samples) * step], dim = -1)

            if 'take_action' in dir(actor):
                policy_action = actor.take_action(states)
                policy_action = torch.as_tensor(policy_action, device=device, dtype=torch.float32)
            else:
                try:
                    policy_action, _ = actor(states)
                except:
                    policy_action = actor(states).rsample()
            action_noise = sigma[step] * torch.randn(actions.shape, device=device)
            state_noise = sigma[step] * torch.randn(states.shape, device=device)
            reward_noise = sigma[step] * torch.randn(rewards.shape, device=device)
            next_state_noise = sigma[step] * torch.randn(next_states.shape, device=device)

            intermi_positive_samples = torch.cat([states + state_noise, policy_action + action_noise, rewards + reward_noise, next_states + next_state_noise, torch.ones_like(dones).to(dones) * step], dim=-1)
            positive_samples = torch.cat((positive_samples, intermi_positive_samples), dim=0)
            negative_samples = torch.cat((negative_samples, intermi_negative_samples), dim=0)

        negative_pred = self.policy_energy(negative_samples)
        positive_pred = self.policy_energy(positive_samples)

        negative_pred_unsq = negative_pred.view(batch_size, negative_sample, -1)
        positive_pred_unsq = positive_pred.view(batch_size, 1, -1)
        pred_unsq = torch.cat([negative_pred_unsq, positive_pred_unsq], dim=1)

        logsumexp = torch.logsumexp(-pred_unsq, dim=1, keepdim=True)
        negative_logsumexp = torch.repeat_interleave(logsumexp, negative_sample, dim=1)

        positive_logsoftmax = (- positive_pred_unsq - logsumexp).view(-1, 1)
        negative_logsoftmax = (- negative_pred_unsq - negative_logsumexp).view(-1, negative_sample)

        ones = torch.ones_like(positive_logsoftmax)
        zeros = torch.zeros_like(negative_logsoftmax)

        loss = torch.nn.KLDivLoss(reduction="none")(positive_logsoftmax, ones).sum() + torch.nn.KLDivLoss(
            reduction="none")(negative_logsoftmax, zeros).sum()
        self.pe_optim.zero_grad()
        loss.backward()
        self.pe_optim.step()
        return loss

    def train_from_redq_buffer(self, buffer, online_buffer = None, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            if online_buffer is not None:
                states, actions, rewards, next_states, dones, mc_returns = online_buffer.sample(int(0.5*self.batch_size))
                states_off, actions_off, rewards_off, next_states_off, dones_off, mc_returns_off = buffer.sample(int(0.5*self.batch_size))
                states = torch.cat([states, states_off], dim=0)
                actions = torch.cat([actions, actions_off], dim=0)
                rewards = torch.cat([rewards, rewards_off], dim=0)
                next_states = torch.cat([next_states, next_states_off], dim=0)
            else:
                states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
            data = [states, actions, rewards, next_states]
            if self.model_terminals:
                data.append(dones)
            data = torch.cat(data, dim=1)

            loss = self.train_on_batch(data, use_wandb=False)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def train_transition_from_redq_buffer(self, buffer, num_steps: Optional[int] = None):
        num_steps = num_steps or self.train_num_steps
        for j in range(num_steps):
            states, actions, rewards, next_states, dones, mc_returns = buffer.sample(self.batch_size)
            state_actions = torch.cat([states, actions], dim=1)
            data = [rewards, next_states]
            if self.model_terminals:
                data.append(dones)
            data = torch.cat(data, dim=1)
            loss = self.train_on_batch(data, use_wandb=False, cond=state_actions)
            if j % 1000 == 0:
                print(f'[{j}/{num_steps}] loss: {loss:.4f}')

    def update_normalizer(self, buffer, device=None):
        data = make_inputs_from_replay_buffer(buffer, self.model_terminals)
        data = torch.from_numpy(data).float()
        self.model.normalizer.reset(data)
        self.ema.ema_model.normalizer.reset(data)
        if device:
            self.model.normalizer.to(device)
            self.ema.ema_model.normalizer.to(device)


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer,
        model_terminals: bool = False,
) -> np.ndarray:
    ptr_location = replay_buffer._pointer
    obs = replay_buffer._states[:ptr_location].cpu().detach().numpy()
    actions = replay_buffer._actions[:ptr_location].cpu().detach().numpy()
    next_obs = replay_buffer._next_states[:ptr_location].cpu().detach().numpy()
    rewards = replay_buffer._rewards[:ptr_location].cpu().detach().numpy()
    inputs = [obs, actions, rewards, next_obs]
    if model_terminals:
        terminals = replay_buffer._dones[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None].cpu().detach().numpy())
    return np.concatenate(inputs, axis=1)
