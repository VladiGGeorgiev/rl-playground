import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from multiprocessing import Pool, cpu_count, Manager
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        hidden_space1 = 16  # Hidden layer sizes
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 1e-4  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # Small number for stability
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        action_means, action_stddevs = self.net(state)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        return action.numpy(), prob

    def update(self, log_probs, rewards):
        discounted_rewards = []
        running_g = 0
        for R in rewards[::-1]:
            running_g = R + self.gamma * running_g
            discounted_rewards.insert(0, running_g)
        deltas = torch.tensor(discounted_rewards, dtype=torch.float32)
        loss = 0
        for log_prob, delta in zip(log_probs, deltas):
            loss += -log_prob * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_single_episode(seed, shared_net_params, obs_space_dims, action_space_dims):
    print(f"{seed=}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    local_env = gym.make("InvertedPendulum-v4")
    local_agent = REINFORCE(obs_space_dims, action_space_dims)
    local_agent.net.load_state_dict(dict(shared_net_params))

    state, _ = local_env.reset(seed=seed)
    done = False
    log_probs = []
    rewards = []

    while not done:
        action, prob = local_agent.sample_action(state)
        state, reward, terminated, truncated, _ = local_env.step(action)
        log_probs.append(prob)
        rewards.append(reward)
        done = terminated or truncated

    return [p.detach().numpy() for p in log_probs], rewards


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    total_num_episodes = int(100)
    num_processes = cpu_count()
    manager = Manager()
    shared_net_params = manager.dict()
    rewards_per_episode = manager.list()

    agent = REINFORCE(obs_space_dims, action_space_dims)
    shared_net_params.update(agent.net.state_dict())

    with Pool(processes=num_processes) as pool:
        for episode in tqdm(range(total_num_episodes)):
            results = pool.starmap(
                train_single_episode,
                [
                    (seed, shared_net_params, obs_space_dims, action_space_dims)
                    for seed in range(num_processes)
                ],
            )
            all_log_probs = []
            all_rewards = []
            for log_probs, rewards in results:
                log_probs = [
                    torch.tensor(p, requires_grad=True) for p in log_probs
                ]  # Convert back to tensor with grad
                all_log_probs.extend(log_probs)
                all_rewards.extend(rewards)
            agent.update(all_log_probs, all_rewards)
            rewards_per_episode.append(np.sum(all_rewards))
            shared_net_params.update(agent.net.state_dict())

    # Save the policy network weights
    torch.save(agent.net.state_dict(), "policy_network_weights.pth")
    print("Training finished. Weights saved to 'policy_network_weights.pth'.")

    # Plot the learning curve
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.show()
