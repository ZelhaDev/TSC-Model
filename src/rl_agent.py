"""
Traffic Sign Classifier — RL Agent (Q-Learning Grid World)
============================================================
A 5×5 grid world where an agent navigates toward a goal.
Traffic signs are placed on cells and modify the reward signal,
simulating a CNN-classified sign feeding into RL behavior.

Components:
  - TrafficSignGridEnv: Gymnasium-style grid environment
  - QLearningAgent: Tabular Q-learning with ε-greedy
  - Reward design: goal bonus, step penalty, sign-specific bonuses/penalties
  - Learning curves saved to experiments/results/

Run:
    python src/rl_agent.py
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline import load_config


# ==============================================================
# Traffic Sign Grid Environment
# ==============================================================

# Sign types placed in the grid (simulating CNN output)
SIGN_TYPES = {
    "stop":        {"reward_mod": -5.0, "desc": "Stop — must stop, heavy penalty if ignored"},
    "speed_limit": {"reward_mod":  2.0, "desc": "Speed limit — bonus for compliant slowing"},
    "yield":       {"reward_mod": -1.0, "desc": "Yield — small penalty for not yielding"},
    "priority":    {"reward_mod":  1.5, "desc": "Priority road — slight bonus"},
    "no_entry":    {"reward_mod": -8.0, "desc": "No entry — severe penalty for entering"},
}


class TrafficSignGridEnv:
    """
    5×5 grid world with traffic signs.

    State : (row, col) flattened to integer 0..grid_size²-1
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, grid_size=5, goal_reward=10.0, step_penalty=-0.2,
                 seed=42):
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.rng = random.Random(seed)

        # Goal at bottom-right
        self.goal = (grid_size - 1, grid_size - 1)

        # Place signs on the grid
        self.signs = self._place_signs()

        self.agent_pos = None
        self.reset()

    def _place_signs(self):
        """Place traffic signs at fixed positions."""
        signs = {}
        sign_placements = [
            ((1, 2), "stop"),
            ((2, 1), "speed_limit"),
            ((0, 3), "yield"),
            ((3, 2), "no_entry"),
            ((2, 4), "priority"),
            ((4, 1), "speed_limit"),
            ((1, 0), "yield"),
        ]
        for pos, stype in sign_placements:
            if pos != self.goal:
                signs[pos] = stype
        return signs

    def _pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def reset(self):
        """Reset agent to top-left corner."""
        self.agent_pos = (0, 0)
        return self._pos_to_state(self.agent_pos)

    def step(self, action):
        """Execute action, return (next_state, reward, done, info)."""
        r, c = self.agent_pos

        if action == 0:    # up
            r = max(0, r - 1)
        elif action == 1:  # down
            r = min(self.grid_size - 1, r + 1)
        elif action == 2:  # left
            c = max(0, c - 1)
        elif action == 3:  # right
            c = min(self.grid_size - 1, c + 1)

        self.agent_pos = (r, c)
        state = self._pos_to_state(self.agent_pos)
        info = {}

        # Base reward
        reward = self.step_penalty

        # Check goal
        done = (self.agent_pos == self.goal)
        if done:
            reward += self.goal_reward
            info["outcome"] = "goal_reached"

        # Check sign
        if self.agent_pos in self.signs:
            sign_type = self.signs[self.agent_pos]
            sign_mod = SIGN_TYPES[sign_type]["reward_mod"]
            reward += sign_mod
            info["sign"] = sign_type
            info["sign_reward"] = sign_mod

        return state, reward, done, info

    def render_text(self):
        """Return a string representation of the grid."""
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row.append(" A ")
                elif (r, c) == self.goal:
                    row.append(" G ")
                elif (r, c) in self.signs:
                    row.append(f" {self.signs[(r, c)][0].upper()} ")
                else:
                    row.append(" . ")
            lines.append("|".join(row))
        return "\n" + "\n".join(lines) + "\n"


# ==============================================================
# Q-Learning Agent
# ==============================================================

class QLearningAgent:
    """Tabular Q-learning with ε-greedy exploration."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            target - self.q_table[state, action]
        )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)


# ==============================================================
# Training Loop
# ==============================================================

def train_rl_agent(cfg):
    """Train Q-learning agent and return training log."""
    rl_cfg = cfg["rl"]
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  RL AGENT — Q-Learning on Traffic Sign Grid World")
    print("=" * 60)

    env = TrafficSignGridEnv(
        grid_size=rl_cfg["grid_size"],
        goal_reward=rl_cfg["goal_reward"],
        step_penalty=rl_cfg["step_penalty"],
        seed=seed,
    )

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=rl_cfg["alpha"],
        gamma=rl_cfg["gamma"],
        epsilon_start=rl_cfg["epsilon_start"],
        epsilon_end=rl_cfg["epsilon_end"],
        epsilon_decay=rl_cfg["epsilon_decay"],
    )

    episodes = rl_cfg["episodes"]
    max_steps = rl_cfg["max_steps"]

    # Logging
    rewards_per_ep = []
    success_per_ep = []
    epsilon_log = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        rewards_per_ep.append(total_reward)
        success_per_ep.append(1 if done else 0)
        epsilon_log.append(agent.epsilon)
        agent.decay_epsilon()

        if (ep + 1) % 100 == 0:
            recent_reward = np.mean(rewards_per_ep[-100:])
            recent_success = np.mean(success_per_ep[-100:])
            print(f"  Episode {ep+1:4d}/{episodes}  "
                  f"avg_reward={recent_reward:.2f}  "
                  f"success_rate={recent_success:.2f}  "
                  f"epsilon={agent.epsilon:.3f}")

    # Final stats
    final_reward = np.mean(rewards_per_ep[-100:])
    final_success = np.mean(success_per_ep[-100:])
    print(f"\n  Final 100-ep avg reward  : {final_reward:.2f}")
    print(f"  Final 100-ep success rate: {final_success:.2f}")

    # Save training log
    log_path = os.path.join(cfg["paths"]["logs"], "rl_training_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "rewards": [round(r, 3) for r in rewards_per_ep],
            "success": success_per_ep,
            "epsilon": [round(e, 4) for e in epsilon_log],
            "final_avg_reward": round(final_reward, 3),
            "final_success_rate": round(final_success, 3),
        }, f, indent=2)
    print(f"  RL training log saved → {log_path}")

    # Plot learning curve
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    plot_rl_curves(rewards_per_ep, success_per_ep, results_dir)

    # Print final grid and Q-table
    print("\n  Grid layout:")
    print(env.render_text())
    print("  Sign legend:")
    for stype, info in SIGN_TYPES.items():
        print(f"    {stype:12s} → reward_mod={info['reward_mod']:+.1f}  "
              f"({info['desc']})")

    return agent, env


def plot_rl_curves(rewards, successes, results_dir):
    """Plot and save RL learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Smoothed reward
    window = 25
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    axes[0].plot(rewards, alpha=0.2, color="blue", label="Raw")
    axes[0].plot(range(window-1, len(rewards)), smoothed,
                 color="blue", linewidth=2, label=f"Smoothed ({window}-ep)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("RL Learning Curve — Cumulative Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Success rate (rolling)
    rolling_success = np.convolve(successes, np.ones(window)/window,
                                   mode="valid")
    axes[1].plot(range(window-1, len(successes)), rolling_success,
                 color="green", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title(f"RL Success Rate (rolling {window}-ep avg)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Q-Learning Agent — Traffic Sign Grid World",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "rl_learning_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  RL learning curves saved → {save_path}")


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    cfg = load_config()
    agent, env = train_rl_agent(cfg)
    print("\n✓ RL agent training complete.")
