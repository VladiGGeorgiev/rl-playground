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
        """Initializes a neural network that estimates Q-values for each action."""
        super().__init__()

        hidden_space1 = 512  # Nothing special with 16, feel free to change
        hidden_space2 = 1024  # Nothing special with 32, feel free to change

        self.q_net = nn.Sequential(
            nn.Linear(obs_space_dims + action_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q_net(x)


def select_action(q_network, state, action_space, epsilon):
    """Selects an action using epsilon-greedy policy for continuous action spaces."""
    if random.random() < epsilon:
        return torch.tensor(action_space.sample(), dtype=torch.float32).unsqueeze(0)
    else:
        action = torch.tensor(
            np.linspace(action_space.low, action_space.high, 10)
        ).float()
        q_values = q_network(state.repeat(action.size(0), 1), action)
        return action[q_values.argmax()].unsqueeze(0)


def sarsa(
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
    action_space = env.action_space

    rewards = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action = select_action(q_network, state, action_space, epsilon)

        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action.numpy().flatten())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_action = select_action(q_network, next_state, action_space, epsilon)

            with torch.no_grad():
                target = torch.tensor(
                    [
                        reward
                        + (1 - done) * gamma * q_network(next_state, next_action).item()
                    ],
                    dtype=torch.float32,
                )

            current_q_value = q_network(state, action)
            loss = loss_fn(current_q_value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            action = next_action

            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards.append(total_reward)
        if episode % 100 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, Avg Rewards: {sum(rewards[-100:])/100}"
            )

    return q_network, rewards


if __name__ == "__main__":
    env = gym.make(
        "InvertedDoublePendulum-v4"
    )  # Example environment, replace with the specific MuJoCo environment
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    q_network = QNetwork(obs_space_dims, action_space_dims)
    num_episodes = 20000
    learning_rate = 0.002
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.999

    trained_q_network, rewards = sarsa(
        env,
        q_network,
        num_episodes,
        learning_rate,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
    )

    # Save the trained Q-network
    torch.save(trained_q_network.state_dict(), "sarsa_q_network.pth")

    # Plot the learning curve
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.show()
