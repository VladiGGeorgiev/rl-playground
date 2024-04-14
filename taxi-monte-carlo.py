import gymnasium as gym

env = gym.make("Taxi-v3")

observation, info = env.reset()
quotient, destination = divmod(observation, 4)
quotient, passenger_location = divmod(quotient, 5)
taxi_row, taxi_col = divmod(quotient, 5)

print(f"{observation=}")
print(f"{taxi_row=}")
print(f"{taxi_col=}")
print(f"{passenger_location=}")
print(f"{destination=}")

VS = dict()
ReturnsS = dict()

for _ in range(10000):
    episode = []
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode.append({"state": observation, "action": action, "reward": reward})

        if terminated or truncated:
            observation, info = env.reset()
            break

    G = 0
    y = 0.5  # The discount rate determines the present value of future rewards. 0 - immediate rewards only, 1 - weight all
    for step in reversed(episode):
        G = y * G + step["reward"]

        if step["state"] not in ReturnsS:
            ReturnsS[step["state"]] = []

        ReturnsS[step["state"]].append(G)
        VS[step["state"]] = sum(ReturnsS[step["state"]]) / len(ReturnsS[step["state"]])


print(f"{VS=}")
print(f"{len(VS)=}")
print(f"{min(VS.values())=}")
print(f"{max(VS.values())=}")

# Input: a policy ⇡ to be evaluated
# Initialize:
#     V (s) 2 R, arbitrarily, for all s 2 S
#     Returns(s) # an empty list, for all s 2 S
# Loop forever (for each episode):
#    Generate an episode following ⇡: S0 , A0 , R1 , S1 , A1 , R2 , . . . , ST
#   G
# 0
# Loop for each step of episode, t = T 1, T 2, . . . , 0:
# G
# G + Rt+1
# Unless St appears in S0 , S1 , . . . , St 1 :
# Append G to Returns(St )
# V (St )
# average(Returns(St ))
