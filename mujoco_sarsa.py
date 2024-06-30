import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SarsaAgent:
    def __init__(
        self, state_dim, action_space_size, alpha=0.001, gamma=0.99, epsilon=0.1
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

        self.q_network = QNetwork(state_dim, action_space_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.uniform(-1, 1, self.action_space_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return q_values.detach().numpy()

    def update_q_values(self, state, action, reward, next_state, next_action):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Compute the target for the selected action
        target = q_values.clone().detach()
        target_value = (
            reward + self.gamma * next_q_values.max().item()
        )  # Use the max Q-value for the next state
        target[action.argmax()] = (
            target_value  # Assign the target value to the selected action
        )

        loss = self.loss_fn(q_values, target)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_sarsa(env_name="InvertedPendulum-v4", episodes=5000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]
    agent = SarsaAgent(state_dim, action_space_size)

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        action = agent.choose_action(state)

        while True:
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update_q_values(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            if done:
                break


if __name__ == "__main__":
    run_sarsa()
