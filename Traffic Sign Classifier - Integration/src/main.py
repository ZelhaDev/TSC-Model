import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CNN
from src.models.cnn import TrafficSignCNN

# NLP
from src.nlp.nlp_component import describe_sign

# RL
from src.rl.rl_agent import TrafficSignGridEnv, QLearningAgent


def load_cnn_model():
    model = TrafficSignCNN(num_classes=43, dropout=0.3)
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
    except (FileNotFoundError, EOFError, RuntimeError):
        print("Warning: Could not load checkpoint. Using untrained model.")
    model.eval()
    return model


def run_pipeline():
    print("=" * 60)
    print("  INTEGRATED TRAFFIC SIGN SYSTEM")
    print("=" * 60)

    # -------------------------------
    # 1. CNN (Image → Class)
    # -------------------------------
    model = load_cnn_model()

    # Dummy input (replace later with real image)
    dummy_input = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input)
        pred_class = output.argmax(1).item()

    print(f"\n[ CNN ] Predicted Class ID: {pred_class}")

    # -------------------------------
    # 2. NLP (Class → Meaning)
    # -------------------------------
    description = describe_sign(pred_class)
    print(f"[ NLP ] Description: {description}")

    # -------------------------------
    # 3. RL (Decision Simulation)
    # -------------------------------
    env = TrafficSignGridEnv()
    agent = QLearningAgent(env.n_states, env.n_actions)

    state = env.reset()

    print("\n[ RL ] Simulating decision...")

    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)

    print(f"Action Taken : {action}")
    print(f"Reward       : {reward}")
    print(f"Environment  : {info}")

    print("\n✓ Integration successful.")


if __name__ == "__main__":
    run_pipeline()