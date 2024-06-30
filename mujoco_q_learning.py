from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import gymnasium as gym
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (10, 5)


class QNetwork(nn.Module):
    """Q-Network to approximate Q-values."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates Q-values for each action.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        self.q_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
        )

    def forward(self, x):
        return self.q_net(x)


def select_action(q_network, state, epsilon, action_space_dims):
    """Selects an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randint(0, action_space_dims - 1)
    else:
        with torch.no_grad():
            return q_network(state).argmax().item()


def q_learning(
    env,
    q_network,
    num_episodes,
    learning_rate,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
):
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_start
    action_space_dims = env.action_space.shape[0]

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        total_reward = 0
        done = False

        while not done:
            action = select_action(q_network, state, epsilon, action_space_dims)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            with torch.no_grad():
                target = (
                    reward + (1 - done) * gamma * q_network(next_state).max().item()
                )

            current_q_value = q_network(state)[0, action]
            loss = loss_fn(current_q_value, torch.tensor(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            avg_reward = int(np.mean(env.return_queue))
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {avg_reward}")

    return q_network


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(
        env, 50
    )  # Records episode-reward

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    q_network = QNetwork(obs_space_dims, action_space_dims)
    num_episodes = 5000
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    trained_q_network = q_learning(
        wrapped_env,
        q_network,
        num_episodes,
        learning_rate,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
    )

    # Save the trained Q-network
    torch.save(trained_q_network.state_dict(), "q_network.pth")
