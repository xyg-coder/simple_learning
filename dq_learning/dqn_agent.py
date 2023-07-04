from __future__ import annotations

from typing import Tuple

import random
import tempfile
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from ml_common.boto_s3 import download_if_s3
from ml_common.boto_s3 import s3_put_file_outpath
from model import LinearDqn


class Agent:
    def __init__(self, n_state: int, n_action: int, memory_size: int = 1000000, s3_snapshot: str = None) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.memory_pool = MemoryPool(memory_size=memory_size)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"now using {self.device}")
        self.model = LinearDqn(n_state, n_action, hidden_sizes=[64, 64], print_out_structure=True).to(self.device)
        self.calculate_model = LinearDqn(n_state, n_action, hidden_sizes=[64, 64]).to(self.device)
        self.calculate_model.load_state_dict(self.model.state_dict())
        if s3_snapshot is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_snapshot = download_if_s3(s3_snapshot, outdir=temp_dir)
                print("loading snapshot")
                self.model.load_state_dict(torch.load(local_snapshot))
                self.calculate_model.load_state_dict(torch.load(local_snapshot))
        self.criteria = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.gamma = 0.99
        self.bacth_size = 32
        self.cur_step = 0
        self.update_every_n_steps = 4
        self.learning_start_size = 50000

    def train(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, rewards: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        # state_value: [n, n_action]
        state_value = self.model(state)
        # action_value: [n, 1]
        action_value = torch.gather(state_value, dim=1, index=action)
        # expected_value: [n , 1]
        max_action_value = self.calculate_model.eval_forward_and_pick_max(next_state)
        expected_value = rewards + self.gamma * max_action_value
        loss_value = self.criteria(action_value, expected_value)
        loss_value.backward()
        self.optimizer.step()

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.memory_pool.add_to_memory(state, action, reward, next_state)
        self.cur_step = (self.cur_step + 1) % self.update_every_n_steps
        if len(self.memory_pool) >= self.learning_start_size:
            state_array, action_array, reward_array, next_state_array = self.memory_pool.get_samples(
                self.bacth_size, self.device
            )
            self.train(state_array, action_array, next_state_array, reward_array)
            if self.cur_step == 0:
                self.update_calculate_model()

    def update_calculate_model(self) -> None:
        self.calculate_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray, eps: float) -> int:
        """given state and eps for eps-greedy, return the action we want to take"""
        if np.random.rand() < eps:
            return np.random.randint(self.n_action)
        else:
            state_array = torch.Tensor(state).to(self.device)
            state_array = state_array.view(1, -1)
            # max_action_tensor should have shape [1, 1]
            max_action_tensor = self.calculate_model.eval_forward_and_pick_max_action(state_array)
            return max_action_tensor[0][0].item()

    def save_model(self, s3_path: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(self.model.state_dict(), f.name)
            s3_put_file_outpath(f.name, s3_path)
            print(f"saved model snapshot to {s3_path}")


@dataclass
class Sample:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class MemoryPool:
    def __init__(self, memory_size: int) -> None:
        self.queue = deque(maxlen=memory_size)

    def __len__(
        self,
    ) -> int:
        return len(self.queue)

    def add_to_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.queue.append(Sample(state=state, action=action, reward=reward, next_state=next_state))

    def get_samples(self, n_sample: int, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """return numpy array of [state, action, reward, next_state]"""
        sample_items = random.sample(self.queue, n_sample)
        state_list = [item.state for item in sample_items]
        action_list = [np.array(item.action, dtype=np.int64) for item in sample_items]
        reward_list = [np.array(item.reward) for item in sample_items]
        next_state_list = [item.next_state for item in sample_items]
        state_tensor = torch.Tensor(np.stack(state_list, axis=0)).to(device)
        action_tensor = torch.tensor(
            np.concatenate([action[np.newaxis][np.newaxis] for action in action_list]), dtype=torch.int64
        ).to(device)
        reward_tensor = torch.Tensor(np.stack(reward_list, axis=0)).reshape(-1, 1).to(device)
        next_state_tensor = torch.Tensor(np.stack(next_state_list, axis=0)).to(device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor
