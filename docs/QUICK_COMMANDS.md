# Advanced Attention Training - Command Reference

## 🚀 Essential Commands

### Train Both Models (Recommended)
```bash
./run_attention_training.sh
```

### Train Specific Model
```bash
# Advanced Attention (3.2M params)
./run_attention_training.sh --model advanced

# MultiModal Attention (1.4M params)
./run_attention_training.sh --model multimodal
```

### Compare All Models
```bash
# Text output
python scripts/compare_all_classifiers.py

# With LaTeX table for papers
python scripts/compare_all_classifiers.py --latex

# Save to CSV
python scripts/compare_all_classifiers.py --save comparison.csv
```

### Visualize Attention
```bash
python test_attention_capabilities.py
```

---

## ⚙️ Common Configurations

### Faster Training (Lightweight)
```bash
./run_attention_training.sh \
    --hidden-dim 128 \
    --num-blocks 2 \
    --num-heads 4 \
    --epochs 100
```

### Best Performance (Deep)
```bash
./run_attention_training.sh \
    --hidden-dim 512 \
    --num-blocks 6 \
    --num-heads 12 \
    --dropout 0.4 \
    --epochs 200
```

### Debug Mode (Quick Test)
```bash
./run_attention_training.sh \
    --epochs 10 \
    --verbose
```

---

## 📊 Check Results

### View Summary Statistics
```bash
# Advanced Attention
cat logs/attention_advanced_cv_summary.json | jq '.summary_stats'

# MultiModal Attention
cat logs/attention_multimodal_cv_summary.json | jq '.summary_stats'

# RAP Fusion (baseline)
cat logs/fusion_classifier_cv_summary.json | jq '.mean_metrics'
```

### Quick Performance Check
```bash
# One-liner comparison
python -c "
import json
rap = json.load(open('logs/fusion_classifier_cv_summary.json'))
adv = json.load(open('logs/attention_advanced_cv_summary.json'))
print(f'RAP Fusion:  {rap[\"mean_metrics\"][\"acc\"]:.4f}')
print(f'Advanced:    {adv[\"summary_stats\"][\"acc\"][\"mean\"]:.4f}')
print(f'Improvement: {(adv[\"summary_stats\"][\"acc\"][\"mean\"] - rap[\"mean_metrics\"][\"acc\"])*100:+.2f}%')
"
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
./run_attention_training.sh --batch-size 16
```

### Overfitting
```bash
./run_attention_training.sh --dropout 0.5
```

### Slow Training
```bash
./run_attention_training.sh --hidden-dim 128 --num-blocks 2
```

### Check GPU
```bash
nvidia-smi
```

---

## 📁 Output Files

After training, check these files:

```bash
# Summary files
logs/attention_advanced_cv_summary.json
logs/attention_multimodal_cv_summary.json

# Model checkpoints (5 per model)
logs/attention_advanced_fold{1-5}_best.pt
logs/attention_multimodal_fold{1-5}_best.pt

# Predictions
logs/oof_preds/attention_advanced_oof_predictions.csv
logs/oof_preds/attention_multimodal_oof_predictions.csv
```

---

## 📖 Documentation

- `TRAINING_READY_SUMMARY.md` - Complete overview
- `ATTENTION_TRAINING_GUIDE.md` - Detailed guide
- `ADVANCED_ATTENTION_IMPLEMENTATION.md` - Architecture details
- `STATE_OF_THE_ART.md` - SOTA techniques explained

---

## ✅ Pre-Flight Checklist

- [ ] GPU available: `nvidia-smi`
- [ ] Features exist: `ls meta/acdc_features.csv`
- [ ] Environment active: `conda activate cardio-dl`
- [ ] Enough disk space: `df -h .`
- [ ] Baseline results exist: `ls logs/fusion_classifier_cv_summary.json`

---

## 🎯 Expected Workflow

1. **Train models**:
   ```bash
   ./run_attention_training.sh
   ```

2. **Compare results**:
   ```bash
   python scripts/compare_all_classifiers.py --latex
   ```

3. **Analyze attention**:
   ```bash
   python test_attention_capabilities.py
   ```

4. **Update documentation**:
   - Add results to `RESULTS.md`
   - Include comparison table
   - Add attention visualizations

---

## 💡 Quick Tips

- Start with default parameters (they're well-tuned)
- Use `--verbose` to monitor training progress
- Early stopping prevents overfitting (patience=30)
- Compare multiple seeds for robustness
- Report 95% CIs in papers (not just mean±std)
- Visualize attention weights for interpretability

---

**Ready to train? Run:** `./run_attention_training.sh` 🚀
