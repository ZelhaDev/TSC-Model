"""
Traffic Sign Classifier — RL Agent (Q-Learning Grid World + CNN Integration)
==============================================================================
A 5×5 grid world where an agent navigates toward a goal.
Traffic signs are placed on cells and modify the reward signal.

Key features (Week 3):
  - CNN integration: loads trained CNN to classify sign images and map
    predictions to grid-cell reward modifiers
  - Multi-seed evaluation (≥3 seeds) with variance reporting
  - Enhanced learning curves with confidence bands

Components:
  - TrafficSignGridEnv: Gymnasium-style grid environment
  - QLearningAgent: Tabular Q-learning with ε-greedy
  - CNN-RL bridge: CNN predictions feed sign types into the environment
  - Reward design: goal bonus, step penalty, sign-specific bonuses/penalties

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
import torch

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

# Map GTSRB class IDs → sign types for CNN integration
GTSRB_TO_SIGN_TYPE = {
    0: "speed_limit", 1: "speed_limit", 2: "speed_limit",
    3: "speed_limit", 4: "speed_limit", 5: "speed_limit",
    7: "speed_limit", 8: "speed_limit",
    9: "no_entry", 10: "no_entry",
    11: "priority", 12: "priority",
    13: "yield",
    14: "stop",
    15: "no_entry", 16: "no_entry", 17: "no_entry",
}


class TrafficSignGridEnv:
    """
    5×5 grid world with traffic signs.

    State : (row, col) flattened to integer 0..grid_size²-1
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, grid_size=5, goal_reward=10.0, step_penalty=-0.2,
                 seed=42, sign_placements=None):
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.rng = random.Random(seed)

        # Goal at bottom-right
        self.goal = (grid_size - 1, grid_size - 1)

        # Place signs on the grid
        self.signs = self._place_signs(sign_placements)

        self.agent_pos = None
        self.reset()

    def _place_signs(self, custom_placements=None):
        """Place traffic signs at fixed or custom positions."""
        signs = {}
        if custom_placements:
            placements = custom_placements
        else:
            placements = [
                ((1, 2), "stop"),
                ((2, 1), "speed_limit"),
                ((0, 3), "yield"),
                ((3, 2), "no_entry"),
                ((2, 4), "priority"),
                ((4, 1), "speed_limit"),
                ((1, 0), "yield"),
            ]
        for pos, stype in placements:
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
# CNN → RL Integration
# ==============================================================

def cnn_classify_signs(cfg):
    """
    Load trained CNN and classify representative sign images.
    Maps CNN predictions to RL sign types for the grid.

    Returns a list of (pos, sign_type) tuples for the environment,
    or None if CNN model is not available.
    """
    try:
        from models.cnn import TrafficSignCNN
        from data_pipeline import get_dataloaders

        ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
        if not os.path.exists(ckpt):
            print("  [CNN-RL] No CNN checkpoint found, using default signs.")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrafficSignCNN(
            num_classes=cfg["cnn"]["num_classes"],
            dropout=cfg["cnn"]["dropout"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        model.eval()

        # Get a batch from test set to sample sign images
        _, _, test_loader, _ = get_dataloaders(cfg)
        imgs, labels = next(iter(test_loader))

        # Sample diverse classes and classify with CNN
        cnn_sign_placements = []
        positions = [(1, 2), (2, 1), (0, 3), (3, 2), (2, 4), (4, 1), (1, 0)]
        seen_classes = set()

        with torch.no_grad():
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()

        for i, (pred, label) in enumerate(zip(preds, labels.numpy())):
            if len(cnn_sign_placements) >= len(positions):
                break
            pred_int = int(pred)
            if pred_int in GTSRB_TO_SIGN_TYPE and pred_int not in seen_classes:
                sign_type = GTSRB_TO_SIGN_TYPE[pred_int]
                pos = positions[len(cnn_sign_placements)]
                cnn_sign_placements.append((pos, sign_type))
                seen_classes.add(pred_int)

        # Fill remaining positions with defaults if needed
        default_signs = ["stop", "speed_limit", "yield", "no_entry",
                         "priority", "speed_limit", "yield"]
        while len(cnn_sign_placements) < len(positions):
            idx = len(cnn_sign_placements)
            cnn_sign_placements.append(
                (positions[idx], default_signs[idx])
            )

        print(f"  [CNN-RL] CNN classified {len(seen_classes)} sign types "
              f"for grid placement:")
        for pos, stype in cnn_sign_placements:
            print(f"    Grid {pos} → {stype}")

        return cnn_sign_placements

    except Exception as e:
        print(f"  [CNN-RL] Could not load CNN: {e}")
        print("  [CNN-RL] Falling back to default signs.")
        return None


# ==============================================================
# Single-seed Training
# ==============================================================

def train_single_seed(cfg, seed, sign_placements=None):
    """Train Q-learning agent with a single seed. Return training log."""
    rl_cfg = cfg["rl"]
    random.seed(seed)
    np.random.seed(seed)

    env = TrafficSignGridEnv(
        grid_size=rl_cfg["grid_size"],
        goal_reward=rl_cfg["goal_reward"],
        step_penalty=rl_cfg["step_penalty"],
        seed=seed,
        sign_placements=sign_placements,
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

    rewards_per_ep = []
    success_per_ep = []

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
        agent.decay_epsilon()

    return {
        "rewards": rewards_per_ep,
        "success": success_per_ep,
        "final_avg_reward": float(np.mean(rewards_per_ep[-100:])),
        "final_success_rate": float(np.mean(success_per_ep[-100:])),
    }, agent, env


# ==============================================================
# Multi-Seed Training
# ==============================================================

def train_rl_agent(cfg):
    """Train Q-learning agent across multiple seeds and report variance."""
    rl_cfg = cfg["rl"]
    num_seeds = rl_cfg.get("num_seeds", 3)
    base_seed = cfg["seed"]

    print("=" * 60)
    print("  RL AGENT — Q-Learning on Traffic Sign Grid World")
    print(f"  Multi-seed evaluation: {num_seeds} seeds")
    print("=" * 60)

    # Try CNN integration
    sign_placements = cnn_classify_signs(cfg)

    seeds = [base_seed + i for i in range(num_seeds)]
    all_logs = {}
    all_rewards = []
    all_success = []

    for i, seed in enumerate(seeds):
        print(f"\n  --- Seed {seed} ({i+1}/{num_seeds}) ---")
        log, agent, env = train_single_seed(cfg, seed, sign_placements)
        all_logs[f"seed_{seed}"] = log
        all_rewards.append(log["rewards"])
        all_success.append(log["success"])

        print(f"    Final 100-ep avg reward  : {log['final_avg_reward']:.2f}")
        print(f"    Final 100-ep success rate: {log['final_success_rate']:.2f}")

    # Aggregate statistics
    rewards_array = np.array(all_rewards)  # (num_seeds, episodes)
    success_array = np.array(all_success)

    mean_reward = np.mean([l["final_avg_reward"] for l in all_logs.values()])
    std_reward = np.std([l["final_avg_reward"] for l in all_logs.values()])
    mean_success = np.mean([l["final_success_rate"] for l in all_logs.values()])
    std_success = np.std([l["final_success_rate"] for l in all_logs.values()])

    print(f"\n  === Multi-Seed Summary ({num_seeds} seeds) ===")
    print(f"  Avg Reward  : {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Success Rate: {mean_success:.2f} ± {std_success:.2f}")

    # Save full training log
    log_data = {
        "seeds": seeds,
        "num_seeds": num_seeds,
        "summary": {
            "mean_reward": round(mean_reward, 3),
            "std_reward": round(std_reward, 3),
            "mean_success_rate": round(mean_success, 3),
            "std_success_rate": round(std_success, 3),
        },
        "per_seed": {
            k: {
                "rewards": [round(r, 3) for r in v["rewards"]],
                "success": v["success"],
                "final_avg_reward": round(v["final_avg_reward"], 3),
                "final_success_rate": round(v["final_success_rate"], 3),
            }
            for k, v in all_logs.items()
        },
        "cnn_integrated": sign_placements is not None,
    }
    log_path = os.path.join(cfg["paths"]["logs"], "rl_training_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n  RL training log saved → {log_path}")

    # Plot learning curves with confidence bands
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    plot_rl_curves_multiseed(rewards_array, success_array, seeds, results_dir)

    # Print grid layout (from last run)
    print("\n  Grid layout:")
    print(env.render_text())
    print("  Sign legend:")
    for stype, info in SIGN_TYPES.items():
        print(f"    {stype:12s} → reward_mod={info['reward_mod']:+.1f}  "
              f"({info['desc']})")

    return all_logs


def plot_rl_curves_multiseed(rewards_array, success_array, seeds, results_dir):
    """Plot RL learning curves with mean ± std confidence bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 25
    episodes = rewards_array.shape[1]

    # Smooth each seed
    def smooth(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode="valid")

    # --- Reward curves ---
    smoothed_rewards = np.array([smooth(r, window) for r in rewards_array])
    mean_r = smoothed_rewards.mean(axis=0)
    std_r = smoothed_rewards.std(axis=0)
    x = range(window - 1, episodes)

    for i, seed in enumerate(seeds):
        axes[0].plot(x, smoothed_rewards[i], alpha=0.25, linewidth=1)
    axes[0].plot(x, mean_r, color="blue", linewidth=2, label="Mean")
    axes[0].fill_between(x, mean_r - std_r, mean_r + std_r,
                         alpha=0.2, color="blue", label="±1 Std")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title(f"RL Reward (smoothed {window}-ep, {len(seeds)} seeds)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Success rate curves ---
    smoothed_success = np.array([smooth(s, window) for s in success_array])
    mean_s = smoothed_success.mean(axis=0)
    std_s = smoothed_success.std(axis=0)

    for i, seed in enumerate(seeds):
        axes[1].plot(x, smoothed_success[i], alpha=0.25, linewidth=1)
    axes[1].plot(x, mean_s, color="green", linewidth=2, label="Mean")
    axes[1].fill_between(x, mean_s - std_s, mean_s + std_s,
                         alpha=0.2, color="green", label="±1 Std")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title(f"RL Success Rate (rolling {window}-ep, {len(seeds)} seeds)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Q-Learning Agent — Traffic Sign Grid World (Multi-Seed)",
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
    all_logs = train_rl_agent(cfg)
    print("\n✓ RL agent training complete.")
