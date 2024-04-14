import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    desc=["FFFF", "SHFH", "FFFH", "HFFG"],
    is_slippery=False,
    render_mode="human",
)

observation, info = env.reset()
current_row, current_col = divmod(observation, 4)

print(f"{observation=}")
print(f"{current_row=}")
print(f"{current_col=}")

env.render()

VS = dict()
for i in range(16):
    VS[i] = 0
print(f"{VS=}")
# ReturnsS = dict()
best_apisode = None
best_episode_G = None


def generate_episode():
    episode = []
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode.append({"state": observation, "action": action, "reward": reward})

        if terminated or truncated:
            observation, info = env.reset()
            break

    return episode


for _ in range(1000):
    episode = generate_episode()

    for step in reversed(episode):
        v = VS[step["state"]]
        VS[step["state"]] = 0  # Sum_a (p(a|s))
