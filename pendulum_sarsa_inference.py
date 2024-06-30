import torch
import gymnasium as gym
from pendulum_sarsa import QNetwork
import numpy as np
from torch.distributions.normal import Normal
import random


def load_q_network(q_network, path):
    """Loads the weights from the saved model file into the Q-network."""
    q_network.load_state_dict(torch.load(path))
    q_network.eval()  # Set the network to evaluation mode


def select_action(q_network, state, action_space, epsilon):
    """Selects an action using epsilon-greedy policy for continuous action spaces."""
    if random.random() < epsilon:
        a = torch.tensor(action_space.sample(), dtype=torch.float32).unsqueeze(0)
        # print(a)
        return a
    else:
        action = torch.tensor(
            np.linspace(action_space.low, action_space.high, 10)
        ).float()
        q_values = q_network(state.repeat(action.size(0), 1), action)
        a = action[q_values.argmax()].unsqueeze(0)
        # print(a)
        return a


def run_inference(env, q_network, num_episodes, render=False):
    """Runs the environment using the trained Q-network for inference."""
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        total_reward = 0
        done = False
        total_steps = 0
        while not done:
            if render:
                env.render()  # Specify the render mode as 'human'

            with torch.no_grad():
                action = select_action(
                    q_network, state, env.action_space, epsilon=0
                )  # No exploration during inference

            next_state, reward, done, _, _ = env.step(action.numpy().flatten())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            total_steps += 1
            state = next_state

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Total Steps: {total_steps}"
        )


env = gym.make(
    "InvertedDoublePendulum-v4", render_mode="human"
)  # Example environment, replace with the specific MuJoCo environment
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

q_network = QNetwork(obs_space_dims, action_space_dims)

# Load the trained Q-network
load_q_network(q_network, "sarsa_q_network.pth")

# Run inference
run_inference(env, q_network, num_episodes=10, render=True)
