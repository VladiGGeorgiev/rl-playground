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
ReturnsS = dict()
best_apisode = None
best_episode_G = None


for _ in range(1000):
    episode = []
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode.append({"state": observation, "action": action, "reward": reward})

        if terminated or truncated:
            observation, info = env.reset()
            break

    G = 0
    y = 0.9  # The discount rate determines the present value of future rewards. 0 - immediate rewards only, 1 - weight all
    for step in reversed(episode):
        G = y * G + step["reward"]

        if step["state"] not in ReturnsS:
            ReturnsS[step["state"]] = []

        ReturnsS[step["state"]].append(G)
        VS[step["state"]] = sum(ReturnsS[step["state"]]) / len(ReturnsS[step["state"]])

    if best_episode_G is None or best_episode_G < G:
        best_apisode = episode
        best_episode_G = G

best_apisode = [step["state"] for step in best_apisode]
print(f"{best_apisode=}")
