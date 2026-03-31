import torch
import sys
import os
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CNN
from src.models.cnn import TrafficSignCNN, load_cnn_model

# NLP
from src.nlp.nlp_component import describe_sign

# RL
from src.rl.rl_agent import TrafficSignGridEnv, QLearningAgent, GTSRB_TO_SIGN_TYPE

# Data
from src.data.data_pipeline import get_transforms


def load_image(image_path):
    """Load and preprocess image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((32, 32), Image.Resampling.LANCZOS)
        print(f"✓ Image loaded: {image_path}")
        print(f"  Size: {img.size} → resized to 32×32")
        return img_resized
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


def run_pipeline(use_custom_image=False):
    print("\n" + "=" * 60)
    print("  INTEGRATED TRAFFIC SIGN SYSTEM")
    print("=" * 60)

    # -------------------------------
    # 0. GET IMAGE INPUT
    # -------------------------------
    if use_custom_image:
        print("\n[INPUT] Provide Your Traffic Sign Image")
        print("-" * 60)
        image_path = input("Enter path to your traffic sign image: ").strip().strip('"\'')
        
        if not image_path:
            print("❌ No image provided. Using random input instead.")
            img_tensor = torch.randn(1, 3, 32, 32)
        else:
            img_pil = load_image(image_path)
            if img_pil is None:
                print("Using random input instead.")
                img_tensor = torch.randn(1, 3, 32, 32)
            else:
                transforms = get_transforms(image_size=32, train=False)
                img_tensor = transforms(img_pil).unsqueeze(0)
    else:
        # Use random input (quick demo)
        img_tensor = torch.randn(1, 3, 32, 32)

    # -------------------------------
    # 1. CNN (Image → Class)
    # -------------------------------
    print("\n[ CNN ] Classification")
    print("-" * 60)
    
    model = load_cnn_model()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(1).item()
        confidence = probs[0, pred_class].item()

    print(f"Predicted Class ID: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Top-5
    top5_probs, top5_classes = torch.topk(probs[0], k=5)
    print(f"Top-5: ", end="")
    for cls, prob in zip(top5_classes.tolist()[:3], top5_probs.tolist()[:3]):
        print(f"Class {cls} ({prob:.1%}), ", end="")
    print("...")

    # -------------------------------
    # 2. NLP (Class → Meaning)
    
    # -------------------------------
    # 2. NLP (Class → Description)
    # -------------------------------
    print("\n[ NLP ] Generate Description")
    print("-" * 60)
    
    description = describe_sign(pred_class)
    print(f"Description: {description}")

    # -------------------------------
    # 3. RL (Sign Type → Action)
    # -------------------------------
    print("\n[ RL ] Agent Decision Making")
    print("-" * 60)
    
    # Map CNN class to RL sign type
    sign_type = GTSRB_TO_SIGN_TYPE.get(pred_class, "unknown")
    print(f"Sign Type: {sign_type}")

    # Initialize RL agent
    agent = QLearningAgent(n_states=25, n_actions=4)
    state = 12  # center of 5x5 grid
    action = agent.choose_action(state)
    
    action_names = ["Up", "Down", "Left", "Right"]
    print(f"Agent Action: {action_names[action]}")
    print(f"Q-Values: [Up={agent.q_table[state, 0]:.2f}, Down={agent.q_table[state, 1]:.2f}, "
          f"Left={agent.q_table[state, 2]:.2f}, Right={agent.q_table[state, 3]:.2f}]")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🔧 SETUP")
    print("Choose mode:")
    print("  1 = Quick demo (random image)")
    print("  2 = Your own image")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        run_pipeline(use_custom_image=True)
    else:
        run_pipeline(use_custom_image=False)