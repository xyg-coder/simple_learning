from __future__ import annotations

from dataclasses import dataclass

from typing import Tuple

import random
import tempfile
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from model import Actor
from model import Critic


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env_take_action_func: callable,
        env_reset_func: callable,
        memory_size: int = 1000000,
        warm_up_steps: int = 1e4,
        s3_snapshot_path: str = None,
        gamma: float = 0.99,
        soft_update_tau: float = 0.001,
        eval_every_n_steps: int = 1000,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_model = Actor(state_dim, action_dim).to(self.device)
        self.actor_target_model = Actor(state_dim, action_dim).to(self.device)
        self.actor_target_model.load_state_dict(self.actor_model.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=1e-4)
        self.critic_model = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_model = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_model.load_state_dict(self.critic_model.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=1e-3)
        self.memory_size = memory_size
        self.memory_pool = MemoryPool(memory_size)
        self.warm_up_steps = warm_up_steps
        self.s3_snapshot_path = s3_snapshot_path
        self.env_take_action_func = env_take_action_func
        self.env_reset_func = env_reset_func
        self.random_processor = OrnsteinUhlenbeckProcess(
            size=(1, self.action_dim), std=LinearSchedule(0.2)
        )
        self.sample_size = 1000
        self.gamma = gamma
        self.soft_update_tau = soft_update_tau
        self.eval_every_n_steps = eval_every_n_steps

        self._initialize_state()
        self.total_steps = 0
        self.eval_epochs = 0

    def _to_np(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().detach().numpy()

    def _to_tensor(self, np_array: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(np_array).type(dtype).to(self.device)

    def _soft_update(self) -> None:
        for target_params, local_params in zip(self.actor_target_model.parameters(), self.actor_model.parameters()):
            target_params.data.copy_(self.soft_update_tau * local_params.data + (1.0 - self.soft_update_tau) * target_params.data)
        for target_params, local_params in zip(self.critic_target_model.parameters(), self.critic_model.parameters()):
            target_params.data.copy_(self.soft_update_tau * local_params.data + (1.0 - self.soft_update_tau) * target_params.data)

    def _train_model(self):
        state_tensor, action_tensor, reward_tensor, next_state_tensor = self.memory_pool.get_samples(self.sample_size, self.device)
        action_next = self.actor_target_model(next_state_tensor)
        q_next = self.critic_target_model(next_state_tensor, action_next)
        q_to_compare = reward_tensor + self.gamma * q_next
        q_to_compare = q_to_compare.detach()

        q_current = self.critic_model(state_tensor, action_tensor)

        value_loss = (q_current - q_to_compare).pow(2).mul(0.5).sum(-1).mean()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        action_to_take = self.actor_model(state_tensor)
        value_of_actions = self.critic_model(state_tensor, action_to_take).mean()
        actor_loss = -value_of_actions
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._soft_update()

    def _initialize_state(self) -> None:
        self.state = self.env_reset_func()
        self.random_processor.reset_states()

    def _eval(self) -> float:
        with torch.no_grad():
            score_sum = 0
            while True:
                action_tensor = self.actor_target_model(self._to_tensor(self.state, dtype=torch.float32))
                action_np = self._to_np(action_tensor)
                next_state, reward, done = self.env_take_action_func(action_np)
                score_sum += reward
                if done:
                    break
                self.state = next_state
            return score_sum

    def step(self):
        assert self.state is not None, 'agent state is not set'
        if len(self.memory_pool) > self.warm_up_steps:
            action_tensor = self.actor_model(self._to_tensor(self.state, dtype=torch.float32))
            action_np = self._to_np(action_tensor)
            action_np += self.random_processor.sample()
            action_np = np.clip(action_np, -1, 1)
        else:
            action_np = np.random.uniform(-1, 1, size=(1, self.action_dim))

        # next_state: [1,33], reward: list of size 1; dones: list of size 1
        next_state, reward, done = self.env_take_action_func(action_np)

        if not done:
            self.memory_pool.add_to_memory(self.state, action_np, reward, next_state)

        if len(self.memory_pool) > self.warm_up_steps:
            self._train_model()

        if done and len(self.memory_pool) > self.warm_up_steps:
            self._initialize_state()
            eval_score = self._eval()
            print(f'eval_epoch={self.eval_epochs}, eval_score={eval_score}')
            self.eval_epochs += 1
            self._initialize_state()
        else:
            self.state = next_state
        self.total_steps += 1


@dataclass
class Sample:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray


class MemoryPool:
    def __init__(self, memory_size: int) -> None:
        self.queue = deque(maxlen=memory_size)

    def __len__(
        self,
    ) -> int:
        return len(self.queue)

    def add_to_memory(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.queue.append(Sample(state=state, action=action, reward=reward, next_state=next_state))

    def get_samples(self, n_sample: int, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """return numpy array of [state, action, reward, next_state]"""
        sample_items = random.sample(self.queue, n_sample)
        state_list = [item.state for item in sample_items]
        action_list = [item.action for item in sample_items]
        reward_list = [np.array(item.reward) for item in sample_items]
        next_state_list = [item.next_state for item in sample_items]
        state_tensor = torch.Tensor(np.concatenate(state_list, axis=0)).to(device)
        action_tensor = torch.Tensor(np.concatenate(action_list, axis=0)).to(device)
        reward_tensor = torch.Tensor(np.stack(reward_list, axis=0)).reshape(-1, 1).to(device)
        next_state_tensor = torch.Tensor(np.concatenate(next_state_list, axis=0)).to(device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor


class OrnsteinUhlenbeckProcess:
    def __init__(self, size: int, std: callable, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
