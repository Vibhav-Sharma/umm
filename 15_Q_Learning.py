# ============================================
# Mini Project: Q-Learning
# ============================================

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Create synthetic environment (Grid World)
# 5x5 Grid World
# Agent starts at (0,0) and goal is at (4,4)
# Actions: 0=Up, 1=Down, 2=Left, 3=Right

grid_size = 5
n_states = grid_size * grid_size
n_actions = 4

# Reward matrix
rewards = np.full((grid_size, grid_size), -1)   # -1 for each step
rewards[4, 4] = 100                               # Goal reward
rewards[2, 2] = -10                                # Obstacle penalty
rewards[1, 3] = -10                                # Obstacle penalty

print("Reward Grid:")
reward_df = pd.DataFrame(rewards, columns=[f'Col_{i}' for i in range(grid_size)],
                          index=[f'Row_{i}' for i in range(grid_size)])
print(reward_df)

# Step 3: Initialize Q-Table
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# Helper functions
def state_to_pos(state):
    return state // grid_size, state % grid_size

def pos_to_state(row, col):
    return row * grid_size + col

def get_next_state(state, action):
    row, col = state_to_pos(state)
    if action == 0: row = max(0, row - 1)          # Up
    elif action == 1: row = min(grid_size - 1, row + 1)  # Down
    elif action == 2: col = max(0, col - 1)          # Left
    elif action == 3: col = min(grid_size - 1, col + 1)  # Right
    return pos_to_state(row, col)

# Step 4: Train Q-Learning Agent
total_rewards = []

for episode in range(episodes):
    state = pos_to_state(0, 0)  # Start at (0,0)
    done = False
    ep_reward = 0

    for step in range(100):  # Max steps per episode
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        # Take action
        next_state = get_next_state(state, action)
        row, col = state_to_pos(next_state)
        reward = rewards[row, col]

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state
        ep_reward += reward

        # Check if goal reached
        if state == pos_to_state(4, 4):
            break

    total_rewards.append(ep_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Step 5: Extract learned policy
policy = np.argmax(Q, axis=1)
action_symbols = ['↑', '↓', '←', '→']
policy_grid = np.array([action_symbols[a] for a in policy]).reshape(grid_size, grid_size)
policy_grid[4, 4] = '★'  # Goal

# Step 6: Evaluation
print("\n--- Q-Learning Results ---")
print(f"Episodes: {episodes}")
print(f"Final Epsilon: {epsilon:.4f}")
print(f"Average Reward (last 100 episodes): {np.mean(total_rewards[-100:]):.2f}")

print("\nLearned Policy:")
policy_df = pd.DataFrame(policy_grid, columns=[f'Col_{i}' for i in range(grid_size)],
                           index=[f'Row_{i}' for i in range(grid_size)])
print(policy_df)

print("\nQ-Table (sample first 5 states):")
q_df = pd.DataFrame(Q[:5], columns=['Up', 'Down', 'Left', 'Right'],
                      index=[f'State_{i}' for i in range(5)])
print(q_df.round(2))

# Step 7: Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rewards over episodes (smoothed)
window = 50
smoothed = [np.mean(total_rewards[max(0, i - window):i + 1]) for i in range(len(total_rewards))]
axes[0].plot(smoothed, color='steelblue', linewidth=1.5)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Q-Learning: Reward per Episode (Smoothed)')

# Heatmap of max Q-values
max_q = np.max(Q, axis=1).reshape(grid_size, grid_size)
im = axes[1].imshow(max_q, cmap='YlOrRd', origin='upper')
axes[1].set_title('Max Q-Value Heatmap')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
for i in range(grid_size):
    for j in range(grid_size):
        axes[1].text(j, i, f'{max_q[i, j]:.1f}', ha='center', va='center', fontsize=9, fontweight='bold')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
