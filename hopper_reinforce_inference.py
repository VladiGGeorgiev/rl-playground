import torch
import gymnasium as gym
from mujoco_reinforce import Policy_Network
import numpy as np
from torch.distributions.normal import Normal


def load_policy_network(weights_path, obs_space_dims, action_space_dims):
    policy_net = Policy_Network(obs_space_dims, action_space_dims)
    policy_net.load_state_dict(torch.load(weights_path))
    policy_net.eval()
    return policy_net


def make_inference(policy_net, env_name, render=False):
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset(seed=42)
    done = False
    total_reward = 0
    total_steps = 0

    while not done:
        if render:
            env.render()

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_means, action_stddevs = policy_net(obs_tensor)
        action_distribution = Normal(action_means, action_stddevs)
        action = action_distribution.sample()
        action = action.detach().numpy().flatten()
        # action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        done = terminated or truncated

    env.close()
    return total_reward, total_steps


# Load the trained policy network
env_name = "Hopper-v4"  # Replace with the actual environment name
weights_path = "policy_network_weights_hopper.pth"
env = gym.make(env_name, render_mode="human")

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

policy_net = load_policy_network(weights_path, obs_space_dims, action_space_dims)

# Make an inference in the environment
total_reward, total_steps = make_inference(policy_net, env_name, render=True)

print(f"Total reward from inference: {total_reward}, Steps: {total_steps}")
