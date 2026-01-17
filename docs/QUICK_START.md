# Quick Start Guide - Advanced Cardiac Segmentation & Classification

## 🚀 What's New?

Your project now includes:
1. **3 Segmentation Architectures**: U-Net, U-Net+CRAM, UNETR (Transformer)
2. **Enhanced Metrics**: Accuracy + F1 scores added to Dice/IoU
3. **2 Fusion Classifiers**: RAP blocks and Gated fusion
4. **Complete Pipeline**: Segmentation → Feature Extraction → Classification

---

## 📋 Quick Commands

### 1. Run Segmentation (Choose Architecture)

```bash
# Activate environment
conda activate cardio-dl

# Standard U-Net (baseline)
python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass --epochs 15

# U-Net + CRAM Attention (Better feature learning)
python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass \
  --model unet3d_cram --epochs 15 --lr 1e-3

# UNETR - Transformer (State-of-the-art)
python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass \
  --model unetr --epochs 20 --lr 5e-4 --batch-size 2
```

###  2. Train Fusion Classifier

```bash
# RAP Fusion (Recommended)
python scripts/train_fusion_cv.py \
  --features meta/acdc_features.csv \
  --fusion-type rap \
  --use-cross-attention \
  --epochs 100 --hidden-dim 128

# Gated Fusion (Alternative)
python scripts/train_fusion_cv.py \
  --features meta/acdc_features.csv \
  --fusion-type gated \
  --epochs 100
```

### 3. Generate Results

```bash
python scripts/make_results_summary.py
cat results/RESULTS.md
```

---

## 📊 Expected Metrics

### Segmentation Output:
- **Dice Score**: Overlap between prediction and ground truth
- **IoU**: Intersection over Union
- **Accuracy**: Pixel-wise correctness
- **F1-macro**: Harmonic mean of precision & recall (averaged across classes)
- **F1-weighted**: F1 weighted by class frequency

### Classification Output:
- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Accounts for class imbalance
- **F1-macro/weighted**: Multi-class F1 scores
- **AUC**: Area Under ROC Curve
- **AP**: Average Precision

---

## 🔍 Model Comparison

| Model | Type | Parameters | Speed | Best For |
|-------|------|------------|-------|----------|
| **U-Net** | CNN | ~19M | Fast | Baseline, quick experiments |
| **U-Net+CRAM** | CNN+Attention | ~21M | Medium | Better boundaries, cardiac structures |
| **UNETR** | Transformer | ~90M | Slow | State-of-the-art accuracy |

| Fusion | Type | Parameters | Best For |
|--------|------|------------|----------|
| **RAP** | Attention | ~200K | Multi-modal fusion |
| **Gated** | Learned weights | ~150K | Simpler, faster |

---

## 📁 File Structure

```
cardiac_early_detection_repo/
├── models/
│   ├── unetr.py              # Transformer segmentation
│   ├── cram.py               # CRAM attention U-Net
│   ├── attention_modules.py  # RAP blocks
│   └── fusion_classifier.py  # Feature fusion classifiers
├── scripts/
│   ├── seg_cv.py             # Segmentation training (UPDATED)
│   ├── train_fusion_cv.py    # Fusion classifier training (NEW)
│   └── make_results_summary.py # Results aggregation (UPDATED)
└── QUICK_START.md            # This file
```

---

## 🎯 Recommended Workflow

1. **Baseline**: Run standard U-Net to establish baseline
   ```bash
   python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass --epochs 15
   ```

2. **Compare Architectures**: Test CRAM and UNETR
   ```bash
   python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass --model unet3d_cram --epochs 15
   python scripts/seg_cv.py --dataset acdc --phase ED --acdc-multiclass --model unetr --epochs 20 --batch-size 2
   ```

3. **Extract Features**: (Already done via your existing scripts)
   ```bash
   python scripts/extract_features_acdc.py
   python scripts/build_features_geom.py
   ```

4. **Train Classifier**: Use fusion classifier
   ```bash
   python scripts/train_fusion_cv.py --fusion-type rap --use-cross-attention --epochs 100
   ```

5. **Generate Report**:
   ```bash
   python scripts/make_results_summary.py
   cat results/RESULTS.md
   ```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'einops'"
```bash
conda activate cardio-dl
pip install einops
```

### CUDA Out of Memory (UNETR)
- Reduce batch size: `--batch-size 1`
- Reduce image size in `create_unetr()` call
- Use gradient checkpointing (advanced)

### Low Accuracy
- Increase epochs: `--epochs 30`
- Try different learning rates: `--lr 5e-4` or `--lr 2e-3`
- Use data augmentation: `--aug3d` for ACDC

---

## 📞 Support

Check these files for more details:
- `PIPELINE.md` - End-to-end walkthrough and outputs
- `scripts/seg_cv.py --help` - All segmentation options
- `scripts/train_fusion_cv.py --help` - All classifier options

---

**Happy Training! 🎉**
