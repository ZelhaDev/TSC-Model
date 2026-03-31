"""
Traffic Sign Classifier -- Integrated AI System
=================================================
Unified inference pipeline combining CNN, RL, and NLP components.

System Flow:
    Input Image -> CNN Classifier -> Predicted Traffic Sign
                -> RL Agent -> Action Decision
                -> NLP -> Human-Readable Output
Run:
    python -m src.main                            # demo mode (GTSRB sample)
    python -m src.main --image path/to/sign.png   # real image inference
    python -m src.main --class-id 14              # skip CNN, test with class ID

Components Used (all existing -- no retraining):
    - src.models.cnn.TrafficSignCNN   -- 3-block CNN trained on GTSRB
    - src.rl.rl_agent                 -- Q-Learning agent on traffic sign grid
    - src.nlp.nlp_component           -- NLP sign descriptions & labeling
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Path setup -- ensure project root is on sys.path for package imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Local imports (existing modules -- not redefined)
# ---------------------------------------------------------------------------
from src.models.cnn import TrafficSignCNN
from src.nlp.nlp_component import describe_sign, GTSRB_DESCRIPTIONS
from src.rl.rl_agent import (
    TrafficSignGridEnv,
    QLearningAgent,
    SIGN_TYPES,
    GTSRB_TO_SIGN_TYPE,
)
from src.data.data_pipeline import load_config

# ===========================================================================
# Constants & Mappings
# ===========================================================================

# Extended mapping: ALL 43 GTSRB class IDs -> RL sign types
# (extends the existing GTSRB_TO_SIGN_TYPE to cover warning & mandatory signs)
EXTENDED_CLASS_TO_SIGN = {
    # Speed limits (classes 0-8)
    0: "speed_limit", 1: "speed_limit", 2: "speed_limit",
    3: "speed_limit", 4: "speed_limit", 5: "speed_limit",
    6: "speed_limit", 7: "speed_limit", 8: "speed_limit",
    # No passing / prohibitory (9-10, 15-17)
    9: "no_entry", 10: "no_entry",
    15: "no_entry", 16: "no_entry", 17: "no_entry",
    # Priority (11-12)
    11: "priority", 12: "priority",
    # Yield & Stop
    13: "yield", 14: "stop",
    # Warning signs (18-31) -> treated as yield / caution
    18: "yield", 19: "yield", 20: "yield", 21: "yield",
    22: "yield", 23: "yield", 24: "yield", 25: "yield",
    26: "yield", 27: "yield", 28: "yield", 29: "yield",
    30: "yield", 31: "yield",
    # End of restrictions (32)
    32: "speed_limit",
    # Mandatory direction signs (33-40) -> treated as priority
    33: "priority", 34: "priority", 35: "priority",
    36: "priority", 37: "priority", 38: "priority",
    39: "priority", 40: "priority",
    # End of no-passing zones (41-42)
    41: "speed_limit", 42: "speed_limit",
}

# Driving actions derived from RL reward signals
SIGN_TYPE_TO_DRIVING_ACTION = {
    "stop":        {"action": "STOP",      "detail": "Vehicle must come to a full stop"},
    "no_entry":    {"action": "TURN",      "detail": "Do not enter -- find alternative route"},
    "yield":       {"action": "SLOW DOWN", "detail": "Reduce speed and proceed with caution"},
    "speed_limit": {"action": "GO",        "detail": "Maintain posted speed and proceed"},
    "priority":    {"action": "GO",        "detail": "You have right of way -- proceed normally"},
}

# GTSRB short names for clean display
GTSRB_SHORT_NAMES = {
    0: "Speed limit (20 km/h)", 1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)", 3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)", 5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)", 7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)", 9: "No passing",
    10: "No passing (>3.5t)", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "No vehicles (>3.5t)", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road",
    24: "Road narrows on right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing",
    30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing (>3.5t)",
}


# ===========================================================================
# Pipeline Helpers
# ===========================================================================

def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def load_cnn_model(cfg, device):
    """
    Load the trained CNN model from checkpoint.

    Falls back to an untrained model if checkpoint is missing or corrupt,
    printing a clear warning rather than crashing.
    """
    num_classes = cfg["cnn"]["num_classes"]
    dropout = cfg["cnn"]["dropout"]

    model = TrafficSignCNN(num_classes=num_classes, dropout=dropout).to(device)

    # Resolve checkpoint path (relative to project root)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

    if not os.path.exists(ckpt_path):
        # Also check config-defined path
        ckpt_alt = os.path.join(PROJECT_ROOT,
                                cfg["paths"]["checkpoints"], "best_model.pth")
        if os.path.exists(ckpt_alt):
            ckpt_path = ckpt_alt

    if os.path.exists(ckpt_path):
        try:
            state_dict = torch.load(ckpt_path, map_location=device,
                                    weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  Checkpoint : {os.path.basename(ckpt_path)} [OK]")
        except (RuntimeError, EOFError) as e:
            print(f"  Checkpoint : LOAD FAILED ({e})")
            print(f"               Using untrained model as fallback.")
    else:
        print(f"  Checkpoint : NOT FOUND -- using untrained model")
        print(f"               (Looked at: {ckpt_path})")

    model.eval()
    return model


def get_inference_transform(image_size=32):
    """
    Build the image preprocessing pipeline (matches training normalization).
    Uses the same mean/std as data_pipeline.py for consistency.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3401, 0.3120, 0.3212],
            std=[0.2725, 0.2609, 0.2669],
        ),
    ])


def load_image(image_path, image_size=32):
    """
    Load and preprocess a single image for CNN inference.

    Parameters
    ----------
    image_path : str
        Path to a traffic sign image (JPG, PNG, etc.).
    image_size : int
        Target size for resizing.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor of shape (1, 3, H, W).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    tfm = get_inference_transform(image_size)
    tensor = tfm(img).unsqueeze(0)  # add batch dimension
    return tensor


def load_sample_from_gtsrb(cfg):
    """
    Try to load a random sample from the GTSRB test set (if downloaded).

    Returns
    -------
    (tensor, label) or (None, None) if dataset is not available.
    """
    try:
        from torchvision import datasets

        data_root = os.path.join(PROJECT_ROOT, cfg["data"]["root"])
        img_size = cfg["data"]["image_size"]

        test_dataset = datasets.GTSRB(
            root=data_root, split="test", download=False,
            transform=get_inference_transform(img_size),
        )

        # Pick a random sample
        idx = random.randint(0, len(test_dataset) - 1)
        img, label = test_dataset[idx]
        return img.unsqueeze(0), label

    except Exception:
        return None, None


def generate_synthetic_input(image_size=32):
    """Generate a synthetic random tensor as fallback input."""
    return torch.randn(1, 3, image_size, image_size)


# ===========================================================================
# CNN Classification Step
# ===========================================================================

def classify_image(model, image_tensor, device):
    """
    Run CNN inference on a preprocessed image tensor.

    Returns
    -------
    dict with keys: class_id, confidence, class_name, description, top_3
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        # Top prediction
        conf, pred = probs.max(dim=1)
        class_id = pred.item()
        confidence = conf.item()

        # Top-3 predictions
        top3_conf, top3_idx = probs.topk(3, dim=1)
        top_3 = [
            {
                "class_id": top3_idx[0, i].item(),
                "confidence": top3_conf[0, i].item(),
                "name": GTSRB_SHORT_NAMES.get(top3_idx[0, i].item(),
                                               f"Class {top3_idx[0, i].item()}"),
            }
            for i in range(3)
        ]

    return {
        "class_id": class_id,
        "confidence": confidence,
        "class_name": GTSRB_SHORT_NAMES.get(class_id, f"Class {class_id}"),
        "description": describe_sign(class_id),
        "top_3": top_3,
    }


# ===========================================================================
# RL Decision Step
# ===========================================================================

def rl_decide_action(class_id, cfg):
    """
    Use the RL agent to decide an action based on the detected sign.

    Process:
    1. Map CNN class ID -> RL sign type
    2. Create environment with the detected sign
    3. Quick-train Q-learning agent (learns sign-aware navigation)
    4. Extract the learned policy at the sign position
    5. Map to a semantic driving action

    Returns
    -------
    dict with keys: sign_type, reward_modifier, grid_action, driving_action,
                    driving_detail, q_values, episodes_trained, success_rate
    """
    # Step 1: Map class -> sign type
    sign_type = EXTENDED_CLASS_TO_SIGN.get(class_id, "yield")
    reward_mod = SIGN_TYPES[sign_type]["reward_mod"]
    sign_desc = SIGN_TYPES[sign_type]["desc"]

    # Step 2: Create environment with detected sign placed on grid
    sign_position = (2, 2)  # center of 5x5 grid
    sign_placements = [
        (sign_position, sign_type),
        ((1, 0), "yield"),          # additional context signs
        ((3, 4), "speed_limit"),
    ]

    rl_cfg = cfg.get("rl", {})
    grid_size = rl_cfg.get("grid_size", 5)
    episodes = min(rl_cfg.get("episodes", 500), 200)  # cap for quick demo
    max_steps = rl_cfg.get("max_steps", 50)

    env = TrafficSignGridEnv(
        grid_size=grid_size,
        goal_reward=rl_cfg.get("goal_reward", 10.0),
        step_penalty=rl_cfg.get("step_penalty", -0.2),
        seed=cfg.get("seed", 42),
        sign_placements=sign_placements,
    )

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=rl_cfg.get("alpha", 0.1),
        gamma=rl_cfg.get("gamma", 0.95),
        epsilon_start=rl_cfg.get("epsilon_start", 1.0),
        epsilon_end=rl_cfg.get("epsilon_end", 0.05),
        epsilon_decay=rl_cfg.get("epsilon_decay", 0.995),
    )

    # Step 3: Quick Q-learning training
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    successes = 0
    for ep in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                successes += 1
                break
        agent.decay_epsilon()

    success_rate = successes / episodes

    # Step 4: Extract learned Q-values at the sign position
    sign_state = env._pos_to_state(sign_position)
    q_values = agent.q_table[sign_state].copy()
    best_action_idx = int(np.argmax(q_values))
    grid_action = env.ACTION_NAMES[best_action_idx]

    # Step 5: Map to driving action
    driving = SIGN_TYPE_TO_DRIVING_ACTION.get(
        sign_type,
        {"action": "PROCEED WITH CAUTION", "detail": "Unknown sign -- slow down"}
    )

    return {
        "sign_type": sign_type,
        "sign_desc": sign_desc,
        "reward_modifier": reward_mod,
        "grid_action": grid_action,
        "grid_q_values": {env.ACTION_NAMES[i]: round(q_values[i], 3)
                          for i in range(len(q_values))},
        "driving_action": driving["action"],
        "driving_detail": driving["detail"],
        "episodes_trained": episodes,
        "success_rate": round(success_rate, 4),
        "grid_layout": env.render_text(),
    }


# ===========================================================================
# NLP Output Generation
# ===========================================================================

def generate_nlp_sentence(cnn_result, rl_result):
    """
    Generate a human-readable natural language sentence summarizing the
    full pipeline output.

    Combines CNN prediction + RL decision into a coherent explanation.
    """
    sign_name = cnn_result["class_name"]
    confidence = cnn_result["confidence"]
    action = rl_result["driving_action"]
    detail = rl_result["driving_detail"]

    # Build natural language output
    if confidence >= 0.8:
        conf_phrase = f"with high confidence ({confidence:.1%})"
    elif confidence >= 0.5:
        conf_phrase = f"with moderate confidence ({confidence:.1%})"
    else:
        conf_phrase = f"with low confidence ({confidence:.1%})"

    sentence = (
        f"Detected a {sign_name} sign {conf_phrase}. "
        f"{detail}. "
        f"The RL agent recommends: {action}."
    )

    return sentence


# ===========================================================================
# Console Formatting Helpers
# ===========================================================================

def print_header():
    """Print the system header banner."""
    print()
    print("=" * 65)
    print("  +===========================================================+")
    print("  |   TRAFFIC SIGN CLASSIFIER -- INTEGRATED AI SYSTEM        |")
    print("  |   CNN (Vision) + RL (Q-Learning) + NLP (Descriptions)   |")
    print("  +===========================================================+")
    print("=" * 65)


def print_section(number, total, title):
    """Print a section divider."""
    print(f"\n{'-' * 65}")
    print(f"  [{number}/{total}] {title}")
    print(f"{'-' * 65}")


def print_final_output(cnn_result, rl_result, nlp_sentence):
    """Print the final formatted output box."""
    sign_name = cnn_result["class_name"].upper()
    action = rl_result["driving_action"]
    conf = cnn_result["confidence"]

    print(f"\n{'-' * 65}")
    print(f"  [FINAL OUTPUT]")
    print(f"{'-' * 65}")
    print()
    print("  +-----------------------------------------------------------+")
    print(f"  |  [Sign]    : {sign_name:<44s}|")
    print(f"  |  [Action]  : {action:<44s}|")
    print(f"  |  [Conf.]   : {conf:<44.1%}|")
    print("  +-----------------------------------------------------------+")

    # Word-wrap the NLP sentence to fit the box
    words = nlp_sentence.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 57:
            current_line = f"{current_line} {word}" if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        print(f"  |  {line:<58s}|")

    print("  +-----------------------------------------------------------+")


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_pipeline(args):
    """
    Execute the full integrated pipeline:
        Input -> CNN -> RL -> NLP -> Output
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # 0. Header
    # ------------------------------------------------------------------
    print_header()

    # ------------------------------------------------------------------
    # 1. System Initialization
    # ------------------------------------------------------------------
    print_section(1, 5, "SYSTEM INITIALIZATION")

    device = get_device()
    print(f"  Device     : {device}")

    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    if os.path.exists(config_path):
        cfg = load_config(config_path)
        print(f"  Config     : config.yaml [OK]")
    else:
        print(f"  Config     : NOT FOUND at {config_path}")
        print(f"               Using default configuration.")
        cfg = {
            "seed": 42,
            "data": {"root": "data", "image_size": 32, "batch_size": 64,
                     "val_split": 0.2},
            "cnn": {"num_classes": 43, "dropout": 0.25},
            "rl": {"grid_size": 5, "episodes": 500, "max_steps": 50,
                   "alpha": 0.1, "gamma": 0.95, "epsilon_start": 1.0,
                   "epsilon_end": 0.05, "epsilon_decay": 0.995,
                   "goal_reward": 10.0, "step_penalty": -0.2},
            "paths": {"logs": "experiments/logs",
                      "results": "experiments/results",
                      "checkpoints": "experiments/logs"},
        }

    # Set seeds for reproducibility
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Load CNN model
    model = load_cnn_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  CNN Model  : TrafficSignCNN ({cfg['cnn']['num_classes']} classes, "
          f"{total_params:,} params)")

    # ------------------------------------------------------------------
    # 2. Image Input
    # ------------------------------------------------------------------
    print_section(2, 5, "IMAGE INPUT")

    image_tensor = None
    input_source = "unknown"
    true_label = None
    image_path = args.image

    if args.class_id is not None:
        # Direct class ID mode -- skip CNN inference
        print(f"  Mode       : Direct class ID (--class-id {args.class_id})")
        print(f"  Note       : Skipping CNN inference, using provided class.")
        input_source = "class_id"

    else:
        # If no --image was passed, prompt the user interactively
        if not image_path:
            print("  No image path provided.")
            print()
            user_input = input("  Enter image file path (or press Enter for demo): ").strip()
            if user_input:
                # Remove surrounding quotes if user pasted a quoted path
                image_path = user_input.strip('"').strip("'")

        # Try loading the image (from CLI arg or interactive input)
        if image_path:
            try:
                image_tensor = load_image(image_path, cfg["data"]["image_size"])
                input_source = "file"
                print(f"  Source     : {image_path}")
                print(f"  Tensor     : {image_tensor.shape}")
            except FileNotFoundError:
                print(f"  ERROR      : Image not found: {image_path}")
                print(f"  Falling back to demo mode.")
                input_source = "fallback"
            except Exception as e:
                print(f"  ERROR      : Could not load image: {e}")
                print(f"  Falling back to demo mode.")
                input_source = "fallback"

    if image_tensor is None and input_source != "class_id":
        # Try loading from GTSRB test set
        sample_tensor, sample_label = load_sample_from_gtsrb(cfg)
        if sample_tensor is not None:
            image_tensor = sample_tensor
            true_label = sample_label
            input_source = "gtsrb_sample"
            print(f"  Source     : GTSRB test set (random sample)")
            print(f"  True Label : Class {true_label} "
                  f"({GTSRB_SHORT_NAMES.get(true_label, '?')})")
            print(f"  Tensor     : {image_tensor.shape}")
        else:
            # Final fallback: synthetic input
            image_tensor = generate_synthetic_input(cfg["data"]["image_size"])
            input_source = "synthetic"
            print(f"  Source     : Synthetic random tensor (demo mode)")
            print(f"  Tensor     : {image_tensor.shape}")
            print(f"  Note       : For real results, provide an image path.")

    # ------------------------------------------------------------------
    # 3. CNN Classification
    # ------------------------------------------------------------------
    print_section(3, 5, "CNN CLASSIFICATION")

    if args.class_id is not None:
        # Skip CNN -- use provided class ID
        class_id = args.class_id
        if class_id < 0 or class_id >= 43:
            print(f"  WARNING    : Class ID {class_id} out of range [0, 42].")
            print(f"               Clamping to valid range.")
            class_id = max(0, min(42, class_id))

        cnn_result = {
            "class_id": class_id,
            "confidence": 1.0,
            "class_name": GTSRB_SHORT_NAMES.get(class_id, f"Class {class_id}"),
            "description": describe_sign(class_id),
            "top_3": [{"class_id": class_id, "confidence": 1.0,
                        "name": GTSRB_SHORT_NAMES.get(class_id, "?")}],
        }
        print(f"  (Using provided class ID -- CNN inference skipped)")
    else:
        cnn_result = classify_image(model, image_tensor, device)

    print(f"  Predicted  : Class {cnn_result['class_id']} -- "
          f"{cnn_result['class_name']}")
    print(f"  Confidence : {cnn_result['confidence']:.1%}")
    print(f"  Description: {cnn_result['description']}")

    if len(cnn_result["top_3"]) > 1 and args.class_id is None:
        print(f"  Top-3      :")
        for i, pred in enumerate(cnn_result["top_3"]):
            marker = ">>>" if i == 0 else "   "
            print(f"    {marker} {pred['confidence']:6.1%}  "
                  f"Class {pred['class_id']:2d} -- {pred['name']}")

    if true_label is not None:
        match = "CORRECT" if true_label == cnn_result["class_id"] else "WRONG"
        print(f"  Ground Truth: Class {true_label} "
              f"({GTSRB_SHORT_NAMES.get(true_label, '?')}) -- {match}")

    # ------------------------------------------------------------------
    # 4. RL Agent Decision
    # ------------------------------------------------------------------
    print_section(4, 5, "RL AGENT DECISION")

    rl_result = rl_decide_action(cnn_result["class_id"], cfg)

    print(f"  Sign Type  : {rl_result['sign_type']}")
    print(f"  Reward Mod : {rl_result['reward_modifier']:+.1f} "
          f"({rl_result['sign_desc']})")
    print(f"  RL Training: {rl_result['episodes_trained']} episodes, "
          f"success rate = {rl_result['success_rate']:.1%}")
    print(f"  Q-Values   : {rl_result['grid_q_values']}")
    print(f"  Grid Action: {rl_result['grid_action']} "
          f"(learned navigation at sign cell)")
    print()
    print(f"  +--------------------------------------+")
    print(f"  |  DRIVING ACTION  =>  {rl_result['driving_action']:<15s}|")
    print(f"  |  {rl_result['driving_detail']:<37s}|")
    print(f"  +--------------------------------------+")

    # Show grid layout
    print(f"\n  Grid World (A=Agent, G=Goal, sign initials on cells):")
    for line in rl_result["grid_layout"].strip().split("\n"):
        print(f"    {line}")

    # ------------------------------------------------------------------
    # 5. NLP Output (Human-Readable Summary)
    # ------------------------------------------------------------------
    print_section(5, 5, "NLP OUTPUT -- NATURAL LANGUAGE SUMMARY")

    nlp_sentence = generate_nlp_sentence(cnn_result, rl_result)
    print(f"  {nlp_sentence}")

    # ------------------------------------------------------------------
    # Final Combined Output
    # ------------------------------------------------------------------
    print_final_output(cnn_result, rl_result, nlp_sentence)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  [OK] Pipeline complete  ({elapsed:.2f}s)")
    print(f"  [OK] System: CNN (Vision) + RL (Q-Learning) + NLP (Labeling)")
    print(f"{'=' * 65}\n")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Traffic Sign Classifier -- Integrated AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                           # Demo mode (auto-picks input)
  python -m src.main --image test_stop.png     # Classify a real image
  python -m src.main --class-id 14             # Test with 'Stop' sign (class 14)
  python -m src.main --class-id 1              # Test with '30 km/h' sign

System Flow:
  Input Image -> CNN Classifier -> Predicted Sign -> RL Agent -> Action -> NLP Output
        """,
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a traffic sign image (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--class-id", type=int, default=None,
        help="Directly specify GTSRB class ID (0-42) to skip CNN inference",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n  [FAIL] Pipeline failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()