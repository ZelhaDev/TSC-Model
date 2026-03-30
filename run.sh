#!/bin/bash
# ==============================================================================
# Traffic Sign Classifier — One-Command Reproduce (Full Pipeline)
# ==============================================================================
# Reproduces all results: NLP, RL, CNN training, evaluation, ablations,
# error analysis, and Grad-CAM.
#
# Usage:
#   bash run.sh            # full pipeline
#   bash run.sh --quick    # skip ablations for faster run
# ==============================================================================

set -e  # Exit on error

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    echo "Running in QUICK mode (ablations skipped)"
fi

echo "=============================================="
echo " Traffic Sign Classifier — Reproduce Pipeline"
echo "=============================================="

# 0. Download datasets interactively
echo ">> Downloading datasets from Kaggle..."
python data/get_data.py

# 1. Install dependencies
echo ""
echo ">> [1/7] Installing dependencies..."
pip install -r requirements.txt

# 2. Run NLP Component
echo ""
echo ">> [2/7] Running NLP Component (TextCNN)..."
python src/nlp_component.py

# 3. Run RL Agent (multi-seed)
echo ""
echo ">> [3/7] Running RL Agent (Q-Learning, multi-seed)..."
python src/rl_agent.py

# 4. Run Core CNN & SVM Training
echo ""
echo ">> [4/7] Running CNN and SVM Training..."
python src/train.py

# 5. Run Evaluation (test metrics, confusion matrix, ROC/PR)
echo ""
echo ">> [5/7] Running Evaluation Pipeline..."
python src/eval.py

# 6. Run Ablation Studies (skip in quick mode)
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo ">> [6/7] Running Ablation Studies..."
    python src/ablation_runner.py
else
    echo ""
    echo ">> [6/7] Ablation studies SKIPPED (quick mode)"
fi

# 7. Run Error Analysis & Grad-CAM
echo ""
echo ">> [7/7] Running Error Analysis & Grad-CAM..."
python src/error_analysis.py
python src/grad_cam.py

echo ""
echo "=============================================="
echo " Pipeline complete!"
echo " Check experiments/logs/   for metric logs"
echo " Check experiments/results/ for plots & tables"
echo "=============================================="
