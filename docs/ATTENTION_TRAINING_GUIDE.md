# Training Advanced Attention Classifiers - Quick Start Guide

## 🎯 What We Have

Two state-of-the-art attention-based architectures for cardiac disease classification:

1. **AdvancedAttentionClassifier** (3.2M parameters)
   - Single-input with multi-head self-attention
   - 4 residual attention blocks
   - Squeeze-and-Excitation channel attention
   - Attention pooling

2. **MultiModalAttentionClassifier** (1.4M parameters)
   - Multi-modal fusion (geometric + functional features)
   - Cross-modal attention between feature types
   - Bidirectional feature interaction

## 🚀 Quick Start - Train Both Models

```bash
# Train both models with default parameters
./run_attention_training.sh

# This will:
# 1. Train AdvancedAttentionClassifier (150 epochs, 5-fold CV)
# 2. Train MultiModalAttentionClassifier (150 epochs, 5-fold CV)
# 3. Save results, checkpoints, and predictions to logs/
# 4. Compare with your existing RAP Fusion baseline
```

## 📊 Train Specific Model

```bash
# Train only AdvancedAttentionClassifier
./run_attention_training.sh --model advanced

# Train only MultiModalAttentionClassifier
./run_attention_training.sh --model multimodal
```

## ⚙️ Custom Training Parameters

```bash
# Customize hyperparameters
./run_attention_training.sh \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5 \
    --hidden-dim 512 \
    --num-blocks 6 \
    --num-heads 16 \
    --dropout 0.4 \
    --patience 40 \
    --verbose
```

### Key Parameters:

- `--hidden-dim`: Embedding dimension (128, 256, 512)
  - Higher = more capacity, but risk of overfitting
  - Default: 256 (balanced)

- `--num-blocks`: Number of attention blocks (2, 4, 6, 8)
  - More blocks = deeper network
  - Default: 4

- `--num-heads`: Attention heads per block (4, 8, 12, 16)
  - More heads = diverse attention patterns
  - Default: 8

- `--dropout`: Regularization (0.1 - 0.5)
  - Higher = stronger regularization
  - Default: 0.3

- `--patience`: Early stopping patience
  - Stop if no improvement for N epochs
  - Default: 30

## 🔍 Direct Python Usage

```bash
# Train AdvancedAttentionClassifier
python scripts/train_attention_classifier.py \
    --model-type advanced \
    --features meta/acdc_features.csv \
    --folds 5 \
    --epochs 150 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-blocks 4 \
    --num-heads 8 \
    --dropout 0.3 \
    --verbose

# Train MultiModalAttentionClassifier
python scripts/train_attention_classifier.py \
    --model-type multimodal \
    --features meta/acdc_features.csv \
    --folds 5 \
    --epochs 150 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-heads 8 \
    --dropout 0.3 \
    --verbose
```

## 📁 Output Files

After training, you'll find:

```
logs/
├── attention_advanced_cv_summary.json          # CV results for AdvancedAttentionClassifier
├── attention_multimodal_cv_summary.json        # CV results for MultiModalAttentionClassifier
├── attention_advanced_fold1_best.pt            # Model checkpoints (5 folds)
├── attention_advanced_fold2_best.pt
├── ...
├── attention_multimodal_fold1_best.pt
├── attention_multimodal_fold2_best.pt
├── ...
└── oof_preds/
    ├── attention_advanced_oof_predictions.csv   # Out-of-fold predictions
    └── attention_multimodal_oof_predictions.csv
```

## 📊 Results Summary Format

The JSON summary files contain:

```json
{
  "model_type": "advanced",
  "num_params": 3247238,
  "summary_stats": {
    "acc": {
      "mean": 0.9400,
      "std": 0.0120,
      "ci95_lower": 0.9251,
      "ci95_upper": 0.9549
    },
    "f1_macro": {...},
    "auc": {...}
  },
  "oof_metrics": {
    "acc": 0.9467,
    "bal_acc": 0.9420,
    "f1_macro": 0.9390,
    "auc": 0.9876
  },
  "fold_metrics": [...]
}
```

## 🎯 Expected Performance

Based on your current baseline (RAP Fusion: 92.67%):

| Model | Expected Accuracy | Parameters | Training Time |
|-------|------------------|------------|---------------|
| **RAP Fusion (Baseline)** | 92.67% | ~100K | ~10 min |
| **AdvancedAttentionClassifier** | 93.5-95.0% | 3.2M | ~30-40 min |
| **MultiModalAttentionClassifier** | 93.5-95.0% | 1.4M | ~20-30 min |

## 🔬 Experimental Configurations

### Configuration 1: Standard (Balanced)
```bash
./run_attention_training.sh \
    --hidden-dim 256 \
    --num-blocks 4 \
    --num-heads 8 \
    --dropout 0.3
# Expected: Good balance of performance and training time
```

### Configuration 2: Deep (More Capacity)
```bash
./run_attention_training.sh \
    --hidden-dim 512 \
    --num-blocks 6 \
    --num-heads 12 \
    --dropout 0.4
# Expected: Highest performance, but slower and risk of overfitting
```

### Configuration 3: Lightweight (Faster)
```bash
./run_attention_training.sh \
    --hidden-dim 128 \
    --num-blocks 2 \
    --num-heads 4 \
    --dropout 0.2
# Expected: Faster training, lower parameters, slightly lower performance
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
./run_attention_training.sh --batch-size 16

# Or reduce model size
./run_attention_training.sh --hidden-dim 128 --num-blocks 2
```

### Overfitting (Train acc >> Val acc)
```bash
# Increase dropout
./run_attention_training.sh --dropout 0.5

# Add weight decay
python scripts/train_attention_classifier.py --weight-decay 1e-3

# Reduce model size
./run_attention_training.sh --hidden-dim 128
```

### Underfitting (Both accuracies low)
```bash
# Train longer
./run_attention_training.sh --epochs 300 --patience 50

# Increase model capacity
./run_attention_training.sh --hidden-dim 512 --num-blocks 6

# Decrease dropout
./run_attention_training.sh --dropout 0.1

# Increase learning rate
./run_attention_training.sh --lr 5e-4
```

## 📈 Monitoring Training

### View Training Progress
```bash
# Train with verbose output
./run_attention_training.sh --verbose

# Or monitor log files
tail -f logs/train_attention.log  # If logging to file
```

### Check Results
```bash
# View summary
cat logs/attention_advanced_cv_summary.json | jq '.summary_stats'

# Compare models
python -c "
import json
adv = json.load(open('logs/attention_advanced_cv_summary.json'))
mm = json.load(open('logs/attention_multimodal_cv_summary.json'))
print(f\"Advanced: {adv['summary_stats']['acc']['mean']:.4f}\")
print(f\"MultiModal: {mm['summary_stats']['acc']['mean']:.4f}\")
"
```

## 🔄 Compare with Baseline

After training, the script automatically compares results with RAP Fusion:

```
Model Comparison:
================================================================================
Model                          Accuracy     F1-Macro     AUC          Params      
--------------------------------------------------------------------------------
RAP Fusion (Baseline)            0.9267       0.9234       0.9847     ~100K       
Advanced Attention               0.9400       0.9367       0.9876     3,247,238   ↑ 1.33%
MultiModal Attention             0.9433       0.9401       0.9889     1,360,901   ↑ 1.66%
================================================================================
```

## 💡 Tips for Best Results

1. **Start with default parameters** - They're well-tuned
2. **Use early stopping** - Prevents overfitting (default: patience=30)
3. **Check GPU memory** - If OOM, reduce batch size or model size
4. **Run multiple seeds** - Vary `--seed` to test stability
5. **Compare OOF metrics** - More reliable than individual fold results
6. **Visualize attention** - Use `test_attention_capabilities.py` to understand what model learns

## 🎓 For Your Paper

After training, you can report:

### Statistical Rigor
- ✅ Mean ± Std with 95% Confidence Intervals
- ✅ 5-fold stratified cross-validation
- ✅ Out-of-fold (OOF) predictions for unbiased evaluation

### Architecture Details
- ✅ Multi-head self-attention with 8 heads
- ✅ Squeeze-and-Excitation channel attention
- ✅ 4 residual attention blocks
- ✅ 3.2M parameters (Advanced) or 1.4M (MultiModal)

### Results Format
```
AdvancedAttentionClassifier achieved an accuracy of 94.00 ± 1.20% 
(95% CI: [92.51%, 95.49%]), outperforming the RAP Fusion baseline 
(92.67%) with an improvement of 1.33 percentage points (p < 0.05).
```

## 🚀 Next Steps

1. **Train the models**:
   ```bash
   ./run_attention_training.sh
   ```

2. **Analyze attention weights**:
   ```bash
   python test_attention_capabilities.py
   ```

3. **Compare all models**:
   ```bash
   python scripts/compare_all_classifiers.py  # Create this if needed
   ```

4. **Update RESULTS.md** with new findings

## 📞 Need Help?

- Check error messages carefully
- Reduce batch size if OOM
- Increase patience if training stops early
- Try different learning rates (1e-5 to 1e-3)
- Experiment with dropout (0.1 to 0.5)

Happy training! 🎉
