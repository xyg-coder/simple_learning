from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from ml_common.boto_s3 import s3_put_file_outpath
from model import ActorCrit
from parallel_env import parallelEnv

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
        save_s3_path: str,
        n_parallel_env: int,
        n_bootstrap: int = 5,
        n_rand: int = 5,
        gamma: float = 0.995,
        value_loss_weight: float = 1.0,
        entropy_loss_weight: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        self.model = ActorCrit(n_channel=2).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.save_s3_path = save_s3_path
        self.n_bootstrap = n_bootstrap
        self.n_env = n_parallel_env
        self.envs = parallelEnv(n=self.n_env)
        self.n_rand = n_rand
        self.gamma = gamma
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.cur_env_done = 0
        self._init_states()

    def _init_states(self) -> tuple:
        """init the states and return the first 2 frames
        Returns:
            tuple: [fr1, fr2]
        """
        self.envs.reset()
        self.envs.step([1] * self.n_env)
        for _ in range(self.n_rand):
            fr1, re1, _, _ = self.envs.step(np.random.choice([LEFT, RIGHT], self.n_env))
            fr2, re2, _, _ = self.envs.step([0] * self.n_env)
        self.state = self._preprocess_batch([fr1, fr2])

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

    def _collect_trajectories(self) -> tuple[list, list, list, list]:
        """run one episode
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

        for t in range(300):
            batch_input = self._preprocess_batch([fr1, fr2])
            probs = self.model(batch_input)["actor_output"]
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

    def _eval(self, iter: int):
        with torch.no_grad():
            state_list, reward_list, prob_list, action_list = self._collect_trajectories()
            _, raw_reward = self._process_reward(reward_list)
            print(
                f"epoch={iter}, reward={np.mean(raw_reward[:, 0])}, total_rewards={np.mean(np.sum(reward_list, axis=0))}"
            )
        self._init_states()

    def _save_model(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(self.model.state_dict(), f.name)
            s3_put_file_outpath(f.name, self.save_s3_path)
            print(f"saved model snapshot to {self.save_s3_path}")

    def _step(self):
        """collect n_bootstraps data and train. If any is_done, reset the env"""
        reward_list = []
        prob_list = []
        value_list = []
        is_done_list = []
        action_list = []
        state = self.state

        for i in range(self.n_bootstrap):
            network_output = self.model(state)
            probs = network_output["actor_output"]
            random_indices = torch.multinomial(torch.cat((probs, 1 - probs), dim=1), 1, replacement=False)
            random_indices = random_indices.view(-1)
            actions = torch.full((self.n_env,), LEFT)
            actions[random_indices == 1] = RIGHT
            values = network_output["crit_output"]
            # after processing, probs, values: [n_env, 1]

            # is_done: numpy_array[20,]
            fr1, reward1, is_done, _ = self.envs.step(torch.squeeze(actions).numpy())
            fr2, reward2, is_done, _ = self.envs.step([0] * self.n_env)

            reward = reward1 + reward2
            reward_list.append(torch.from_numpy(reward).to(device))
            prob_list.append(probs)
            value_list.append(values)
            is_done_list.append(torch.from_numpy(is_done).to(device))
            action_list.append(actions)

            state = self._preprocess_batch([fr1, fr2])

        self.state = state
        with torch.no_grad():
            # train
            return_value = torch.squeeze(self.model(self.state)["crit_output"])
            return_value_list = []
            for i in reversed(range(self.n_bootstrap)):
                # cur_return: [n_env,]
                cur_return = (~is_done_list[i]) * return_value * self.gamma + reward_list[i]
                return_value_list.insert(0, cur_return.detach())
                return_value = cur_return

        probs_tensor = torch.cat(prob_list, dim=1)
        actions_tensor = torch.stack(action_list, dim=1).to(device)
        probs_tensor = torch.where(actions_tensor == LEFT, probs_tensor, 1.0 - probs_tensor)

        advantages = torch.stack(return_value_list, dim=1) - torch.cat(value_list, dim=1)
        policy_loss = torch.log(probs_tensor) * advantages.detach()
        value_loss = -0.5 * advantages.pow(2)
        entropy_loss = -(
            probs_tensor * torch.log(probs_tensor + 1.0e-10)
            + (1.0 - probs_tensor) * torch.log(1.0 - probs_tensor + 1.0e-10)
        )
        loss = -torch.mean(policy_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del loss

        if is_done.any():
            self.cur_env_done += 1
            self._init_states()


if __name__ == "__main__":
    cpu_count = os.cpu_count()
    snapshot_path = "s3://datausers/xgui/enforcement_learning/pong/policy_ppo_actor_critor.pt"
    assert cpu_count > 40, "better to run under more than 40 cpus"
    trainer = Trainer(snapshot_path, 20)
    trainer._eval(0)
    for i in range(800 * 100):
        trainer._step()
        if (i + 1) % 200 == 0:
            trainer._eval(i)

        if (i + 1) % 10000 == 0:
            trainer._save_model()
