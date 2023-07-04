from __future__ import annotations

import os

import numpy as np
import torch
from model import Policy
from ml_common.boto_s3 import s3_put_file_outpath
from parallel_env import parallelEnv
import tempfile

"""env setup:
pip3 install gym
pip install "gym[atari, accept-rom-license]"
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RIGHT = 4
LEFT = 5


class Trainer:
    def __init__(
        self,
        n_parallel_env: int,
        save_s3_path: str,
        tmax: int = 300,
        n_rand: int = 5,
        ratio_diff_epsilon: int = 0.1,
        gamma: float = 0.995,
        n_iter: int = 100,
        n_optimize_per_epoch: int = 4,
        lr: float = 1e-4,
        log_per_iter: int = 1,
        save_per_iter: int = 100,
    ) -> None:
        self.model = Policy(n_channel=2).to(device)
        self.tmax = tmax
        self.n_env = n_parallel_env
        self.envs = parallelEnv(n=n_parallel_env)
        self.n_rand = n_rand
        self.ratio_diff_epsilon = ratio_diff_epsilon
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_optimizer_per_epoch = n_optimize_per_epoch
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.log_per_iter = log_per_iter
        self.save_per_iter = save_per_iter
        self.save_s3_path = save_s3_path

    def _preprocess_batch(self, images, bkg_color=np.array([144, 72, 17])) -> torch.Tensor:
        """preprocess batch and return torch tensor
        the final output will be [n_env, n_fr=2, 80, 80]
        """
        list_of_images = np.asarray(images)
        if len(list_of_images.shape) < 5:
            list_of_images = np.expand_dims(list_of_images, 1)
        # subtract bkg and crop
        list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color, axis=-1) / 255.0
        batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
        return torch.from_numpy(batch_input).float().to(device)

    def _collect_trajectories(self) -> tuple[list, list, list, list]:
        """_summary_

        Returns:
            tuple[list, list, list, list]:
                state_list: list[tensor(n_env,2,80,80)]
                reward_list: list[np_array(n_env)]
                prob_list: list[tensor_of_shape(n_env)]
                action_list: list[tensor(n_env)]
        """
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []
        self.envs.reset()
        self.envs.step([1] * self.n_env)

        for _ in range(self.n_rand):
            fr1, re1, _, _ = self.envs.step(np.random.choice([LEFT, RIGHT], self.n_env))
            fr2, re2, _, _ = self.envs.step([0] * self.n_env)

        for t in range(self.tmax):
            batch_input = self._preprocess_batch([fr1, fr2])
            probs = self.model(batch_input)
            probs = probs.view(-1)
            random_indices = torch.multinomial(torch.stack([probs, 1 - probs], dim=1), 1, replacement=False)
            random_indices = random_indices.view(-1)
            actions = torch.full((self.n_env,), LEFT)
            actions[random_indices == 1] = RIGHT
            probs = torch.where(random_indices == 0, probs, 1 - probs)

            fr1, reward1, is_done, _ = self.envs.step(torch.squeeze(actions).numpy())
            fr2, reward2, is_done, _ = self.envs.step([0] * self.n_env)

            reward = reward1 + reward2
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(actions)

            if is_done.any():
                break

        return state_list, reward_list, prob_list, action_list

    def _get_loss_tensor(self, old_prob: torch.Tensor, new_prob: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """use clipped surrogate to get loss tensor

        Args:
            old_prob (torch.Tensor): old probability of shape [n_env, n_time]
            new_prob (torch.Tensor): new probability of shape [n_env, n_time]
            reward (torch.Tensor): reward of shape [n_env, n_time].
                Each element means the future reward from that timepoint

        Returns:
            torch.Tensor: loss tensor
        """
        old_prob = old_prob.detach()
        ratio = new_prob / old_prob
        clip_ratio = torch.clamp(ratio, 1 - self.ratio_diff_epsilon, 1 + self.ratio_diff_epsilon)
        ratio_reward = torch.min(ratio * reward, clip_ratio * reward)
        # entropy: have the new probability closer to old_probability
        entropy = -(new_prob * torch.log(old_prob + 1.0e-10) + (1.0 - new_prob) * torch.log(1.0 - old_prob + 1.0e-10))
        return -torch.mean(ratio_reward + 0.01 * entropy)

    def _process_reward(self, reward_list: list[np.array]) -> tuple[np.array, np.array]:
        """process reward

        Args:
            reward_list (list[np.array]): a list of np.array[2]

        Returns:
            tuple[np.array, np.array]: normalized_reward, raw_reward
        """
        n_time = len(reward_list)
        # reward_array: [n_env, n_time]
        reward_array = np.stack(reward_list, axis=1)
        gamma_array = np.array([self.gamma**t for t in range(n_time)])
        raw_reward = reward_array * gamma_array
        raw_reward = np.cumsum(raw_reward[:, ::-1], axis=1)
        # use copy because pytorch doesn't support negative stride
        raw_reward = np.flip(raw_reward, axis=1).copy()
        mean = np.mean(raw_reward, axis=1, keepdims=True)
        std = np.std(raw_reward, axis=1, keepdims=True) + 1.0e-10
        normalized_reward = (raw_reward - mean) / std
        return normalized_reward, raw_reward

    def _output_temp_result(self, iter: int, reward: np.array, reward_list: list[np.array]) -> None:
        print(f"epoch={iter}, reward={np.mean(reward)}, total_rewards={np.mean(np.sum(reward_list, axis=0))}")

    def _save_model(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(self.model.state_dict(), f.name)
            s3_put_file_outpath(f.name, self.save_s3_path)
            print(f"saved model snapshot to {self.save_s3_path}")

    def train(self) -> None:
        for i in range(self.n_iter):
            state_list, reward_list, prob_list, action_list = self._collect_trajectories()
            n_time = len(state_list)
            # state_tensor: [self.n_env, time, 2, 80, 80]
            state_tensor = torch.stack(state_list, dim=1)
            state_tensor = state_tensor.view(-1, *state_tensor.shape[-3:])
            normalized_reward, raw_reward = self._process_reward(reward_list)
            # reward_tensor: [time]
            reward_tensor = torch.from_numpy(normalized_reward).to(device)
            # old_prob: [self.n_env, n_time]
            old_prob = torch.stack(prob_list, dim=1)
            action_tensor = torch.stack(action_list, dim=1).to(device)

            # use current prob to optimize once
            loss = self._get_loss_tensor(old_prob, old_prob, reward_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            del loss

            for _ in range(self.n_optimizer_per_epoch):
                # new_prob: [self.n_env * time, 1]
                new_prob = self.model(state_tensor)
                new_prob = new_prob.view(self.n_env, n_time)
                new_prob = torch.where(action_tensor == LEFT, new_prob, 1.0 - new_prob)
                loss = self._get_loss_tensor(old_prob, new_prob, reward_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del loss

            if (i + 1) % self.log_per_iter == 0:
                self._output_temp_result(i, raw_reward[:, 0], reward_list)

            if (i + 1) % self.save_per_iter == 0:
                self._save_model()


if __name__ == "__main__":
    cpu_count = os.cpu_count()
    snapshot_path = "s3://datausers/xgui/enforcement_learning/pong/policy_ppo_snapshot_no_entropy.pt"
    assert cpu_count > 40, "better to run under more than 40 cpus"
    trainer = Trainer(20, n_iter=800, save_s3_path=snapshot_path)
    trainer.train()
