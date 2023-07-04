from __future__ import annotations

from datetime import datetime

import gym
from dqn_agent import Agent
from torch.utils.tensorboard import SummaryWriter

"""
to install libraries:
pip3 install box2d
pip3 install gym
pip3 install pygame
"""

env = gym.make("LunarLander-v2")
time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
writer = SummaryWriter(f"/var/log/tf_logs/landing-training-{time_str}")
snapshot_path = "s3://datausers/xgui/enforcement_learning/landing_model/model_snapshot.pt"


def eval(env, agent, i_episode):
    """do eval and print the score"""
    state = env.reset()[0]
    score = 0
    for j in range(2000):
        action = agent.act(state, 0)
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        state = next_state
        if done:
            break
    print(f"eval done, i_episode={i_episode}, step={j}, score={score}")
    writer.add_scalar("eval_score/total_score", score, i_episode)
    writer.flush()


def train():
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(state_shape[0], n_actions, s3_snapshot=snapshot_path)
    n_episodes = 4000
    eps = 1
    min_eps = 0.1
    eps_decay = 0.995
    for i in range(n_episodes):
        state = env.reset()[0]
        score = 0
        for j in range(2000):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state)
            score += reward
            state = next_state
            if done:
                break
        eps = max(min_eps, eps * eps_decay)
        print(f"episode={i}, score={score}")
        writer.add_scalar("train_loss/total_score", score, i)
        writer.flush()
        if i % 100 == 99:
            agent.save_model(snapshot_path)
            eval(env, agent, i)


if __name__ == "__main__":
    train()
