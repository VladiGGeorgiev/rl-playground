import gymnasium as gym
import numpy as np
from collections import defaultdict


def epsilon_greedy_policy(Q, state, nA, epsilon=0.1):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q (dict): A dictionary that maps from state -> action-values.
        state (int): The current state.
        nA (int): Number of actions in the environment.
        epsilon (float): The probability to select a random action. Float between 0 and 1.

    Returns:
        A numpy array of action probabilities
    """
    action_probabilities = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    action_probabilities[best_action] += 1.0 - epsilon
    return action_probabilities


def generate_episode(env, policy):
    """
    Generates an episode using the given policy.

    Args:
        env: The environment.
        policy: A function that maps state to action probabilities.

    Returns:
        A list of (state, action, reward) tuples
    """
    episode = []
    state, _ = env.reset()
    done = False

    while not done:
        action_probabilities = policy(state)
        action = np.random.choice(
            np.arange(len(action_probabilities)), p=action_probabilities
        )
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated

    return episode


def mc_control(env, num_episodes, gamma=1.0, epsilon=0.1):
    """
    Monte Carlo control using epsilon-greedy policies.

    Args:
        env: The environment.
        num_episodes (int): Number of episodes to sample.
        gamma (float): The discount factor.
        epsilon (float): The probability to select a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function mapping state to action probabilities.
    """
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Generate an episode
        policy = lambda s: epsilon_greedy_policy(Q, s, nA, epsilon)
        episode = generate_episode(env, policy)

        # Find all state-action pairs we've visited in this episode
        state_action_pairs = [(x[0], x[1]) for x in episode]
        unique_state_action_pairs = set(state_action_pairs)

        for state, action in unique_state_action_pairs:
            # Find the first occurrence of the state-action pair in the episode
            first_occurrence_idx = next(
                i for i, x in enumerate(state_action_pairs) if x == (state, action)
            )
            # Sum up all rewards since the first occurrence
            G = sum(
                [
                    x[2] * (gamma**i)
                    for i, x in enumerate(episode[first_occurrence_idx:])
                ]
            )
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1.0
            Q[state][action] = (
                returns_sum[(state, action)] / returns_count[(state, action)]
            )
            if i_episode > 5000:
                env.render()

        # Print out progress
        if i_episode % 1000 == 0:
            print(f"Episode {i_episode}/{num_episodes}")

    # Create the final policy
    policy = lambda s: epsilon_greedy_policy(Q, s, nA, 0.0)  # deterministic policy
    return Q, policy


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    num_episodes = 50000
    Q, policy = mc_control(env, num_episodes)

    # Test the policy
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_probabilities = policy(state)
        action = np.argmax(action_probabilities)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Total reward: {total_reward}")
    env.close()
