#!/bin/bash
# Train Advanced Attention-Based Classifiers
# This script trains both the AdvancedAttentionClassifier and MultiModalAttentionClassifier

set -e  # Exit on error

echo "============================================"
echo "Training Advanced Attention Classifiers"
echo "============================================"

# Default parameters
FEATURES="meta/acdc_features.csv"
FOLDS=5
EPOCHS=150
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=1e-4
HIDDEN_DIM=256
NUM_BLOCKS=4
NUM_HEADS=8
DROPOUT=0.3
PATIENCE=30
SEED=42
LOGDIR="logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num-blocks)
            NUM_BLOCKS="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --model)
            # Only train specific model (advanced or multimodal)
            ONLY_MODEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --features PATH       Path to features CSV (default: meta/acdc_features.csv)"
            echo "  --epochs N            Number of training epochs (default: 150)"
            echo "  --batch-size N        Batch size (default: 32)"
            echo "  --lr FLOAT            Learning rate (default: 1e-4)"
            echo "  --hidden-dim N        Hidden dimension (default: 256)"
            echo "  --num-blocks N        Number of attention blocks (default: 4)"
            echo "  --num-heads N         Number of attention heads (default: 8)"
            echo "  --dropout FLOAT       Dropout rate (default: 0.3)"
            echo "  --patience N          Early stopping patience (default: 30)"
            echo "  --seed N              Random seed (default: 42)"
            echo "  --logdir PATH         Output directory (default: logs)"
            echo "  --model TYPE          Train only specific model: 'advanced' or 'multimodal'"
            echo "  --verbose             Print detailed training progress"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Features:     $FEATURES"
echo "  Epochs:       $EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Hidden dim:   $HIDDEN_DIM"
echo "  Num blocks:   $NUM_BLOCKS"
echo "  Num heads:    $NUM_HEADS"
echo "  Dropout:      $DROPOUT"
echo "  Patience:     $PATIENCE"
echo "  Seed:         $SEED"
echo "  Log dir:      $LOGDIR"
echo ""

# Check if features file exists
if [ ! -f "$FEATURES" ]; then
    echo "ERROR: Features file not found: $FEATURES"
    exit 1
fi

# Create log directory
mkdir -p "$LOGDIR"

# Function to train a model
train_model() {
    local model_type=$1
    echo ""
    echo "============================================"
    echo "Training $model_type Attention Classifier"
    echo "============================================"
    echo ""
    
    python scripts/train_attention_classifier.py \
        --features "$FEATURES" \
        --model-type "$model_type" \
        --folds $FOLDS \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --hidden-dim $HIDDEN_DIM \
        --num-blocks $NUM_BLOCKS \
        --num-heads $NUM_HEADS \
        --dropout $DROPOUT \
        --patience $PATIENCE \
        --seed $SEED \
        --logdir "$LOGDIR" \
        $VERBOSE
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $model_type model training completed successfully"
    else
        echo ""
        echo "✗ $model_type model training failed"
        return 1
    fi
}

# Train models
if [ -z "$ONLY_MODEL" ]; then
    # Train both models
    train_model "advanced"
    train_model "multimodal"
else
    # Train only specified model
    train_model "$ONLY_MODEL"
fi

echo ""
echo "============================================"
echo "All Training Complete!"
echo "============================================"
echo ""
echo "Results saved to: $LOGDIR/"
echo ""
echo "Summary files:"
echo "  - $LOGDIR/attention_advanced_cv_summary.json"
echo "  - $LOGDIR/attention_multimodal_cv_summary.json"
echo ""
echo "Model checkpoints:"
echo "  - $LOGDIR/attention_advanced_fold*_best.pt"
echo "  - $LOGDIR/attention_multimodal_fold*_best.pt"
echo ""
echo "Out-of-fold predictions:"
echo "  - $LOGDIR/oof_preds/attention_advanced_oof_predictions.csv"
echo "  - $LOGDIR/oof_preds/attention_multimodal_oof_predictions.csv"
echo ""

# Compare with existing results if available
if [ -f "$LOGDIR/fusion_classifier_cv_summary.json" ]; then
    echo "============================================"
    echo "Quick Comparison with RAP Fusion"
    echo "============================================"
    echo ""
    
    python - <<EOF
import json
from pathlib import Path

logdir = Path("$LOGDIR")

# Load results
rap_results = json.load(open(logdir / "fusion_classifier_cv_summary.json"))
advanced_results = json.load(open(logdir / "attention_advanced_cv_summary.json"))
multimodal_results = json.load(open(logdir / "attention_multimodal_cv_summary.json"))

print("Model Comparison:")
print("=" * 80)
print(f"{'Model':<30s} {'Accuracy':<12s} {'F1-Macro':<12s} {'AUC':<12s} {'Params':<12s}")
print("-" * 80)

# RAP Fusion
rap_acc = rap_results['mean_metrics']['acc']
rap_f1 = rap_results['mean_metrics']['f1_macro']
rap_auc = rap_results['mean_metrics']['auc']
print(f"{'RAP Fusion (Baseline)':<30s} {rap_acc:>11.4f} {rap_f1:>11.4f} {rap_auc:>11.4f} {'~100K':<12s}")

# Advanced Attention
adv_acc = advanced_results['summary_stats']['acc']['mean']
adv_f1 = advanced_results['summary_stats']['f1_macro']['mean']
adv_auc = advanced_results['summary_stats']['auc']['mean']
adv_params = f"{advanced_results['num_params']:,}"
improvement_adv = (adv_acc - rap_acc) * 100
sign_adv = "↑" if improvement_adv > 0 else "↓"
print(f"{'Advanced Attention':<30s} {adv_acc:>11.4f} {adv_f1:>11.4f} {adv_auc:>11.4f} {adv_params:<12s} {sign_adv} {abs(improvement_adv):.2f}%")

# MultiModal Attention
mm_acc = multimodal_results['summary_stats']['acc']['mean']
mm_f1 = multimodal_results['summary_stats']['f1_macro']['mean']
mm_auc = multimodal_results['summary_stats']['auc']['mean']
mm_params = f"{multimodal_results['num_params']:,}"
improvement_mm = (mm_acc - rap_acc) * 100
sign_mm = "↑" if improvement_mm > 0 else "↓"
print(f"{'MultiModal Attention':<30s} {mm_acc:>11.4f} {mm_f1:>11.4f} {mm_auc:>11.4f} {mm_params:<12s} {sign_mm} {abs(improvement_mm):.2f}%")

print("=" * 80)
print("")
print("Best Model: ", end="")
best_acc = max(rap_acc, adv_acc, mm_acc)
if best_acc == rap_acc:
    print("RAP Fusion (Baseline)")
elif best_acc == adv_acc:
    print("Advanced Attention")
else:
    print("MultiModal Attention")
EOF
fi

echo ""
echo "✓ Training pipeline complete!"
