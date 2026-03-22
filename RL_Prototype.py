import numpy as np
import random
import matplotlib.pyplot as plt

# States and actions
num_states = 4
num_actions = 4

# Q-table
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 200

# Reward function
def get_reward(state, action):
    return 1 if state == action else -1

# Training loop
rewards_per_episode = []

for episode in range(episodes):
    state = random.randint(0, 3)
    total_reward = 0

    for step in range(10):  # steps per episode
        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        reward = get_reward(state, action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[state]) - Q[state, action]
        )

        total_reward += reward

        # Next state (random for simplicity)
        state = random.randint(0, 3)

    rewards_per_episode.append(total_reward)

# Print learned Q-table
print("Q-table:")
print(Q)

# Plot learning curve
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Learning Curve")
plt.show()
