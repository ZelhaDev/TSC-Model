#!/bin/bash
# ==============================================================================
# Traffic Sign Classifier — One-Command Reproduce (Modeling Pipeline)
# ==============================================================================

echo "Starting reproducible training and evaluation pipeline..."

# 1. Install dependencies
echo ">> Installing dependencies..."
pip install -r requirements.txt

# 2. Run NLP Component Prototype
echo ">> Running NLP Component Prototype..."
python src/nlp_component.py

# 3. Run RL Agent Simulation
echo ">> Running RL Agent Simulation..."
python src/rl_agent.py

# 4. Run Core CNN & Baselines Pipeline (Includes GTSRB Data Download)
echo ">> Running Core CNN and SVM Training Pipeline..."
python src/train.py

# 5. Run Evaluation and Plot Generation
echo ">> Running Evaluation Pipeline..."
python src/eval.py

echo "Pipeline execution complete! Check experiments/logs/ and experiments/results/ for outputs."
