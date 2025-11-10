# 🫀 Cardiac Early Detection - Quick Reference

## 🚀 Quick Start Commands

### Run Full Pipeline
```bash
./run_full_pipeline.sh
```

### Run From Specific Point
```bash
# Skip segmentation (already trained)
./run_full_pipeline.sh --skip-seg

# Skip segmentation + OOF
./run_full_pipeline.sh --skip-seg --skip-oof
```

### Test Pipeline
```bash
./test_pipeline.sh
```

---

## 📊 Current Performance

| Model | Accuracy | AUC |
|-------|----------|-----|
| **RAP Fusion + Cross-Attention** ⭐ | **92.67%** | **98.47%** |
| Logistic Regression | 92.00% | 99.11% |
| Random Forest | 88.00% | 98.11% |
| XGBoost | 86.00% | 96.92% |

**Segmentation**:
- CAMUS: 94.05% Dice
- ACDC: 67.40% Dice (multi-class), LV: 77.51%

---

## 📁 Key Files

- **Pipeline**: `run_full_pipeline.sh` - Automated workflow
- **Documentation**: `PIPELINE.md` - All commands
- **Results**: `results/RESULTS.md` - Complete analysis
- **Features**: `meta/acdc_features.csv` - Extracted features
- **Checkpoints**: `logs/*_best.pt` - Trained models

---

## 🔧 Individual Commands

### Segmentation
```bash
python scripts/seg_cv.py --dataset acdc --model unet3d --phase ED --acdc-multiclass --folds 5 --epochs 40 --aug3d
```

### OOF Inference
```bash
python scripts/oof_infer_acdc.py --phase ED --folds 5 --with-bg
```

### Feature Extraction
```bash
python scripts/build_features_geom.py
```

### Fusion Classifier
```bash
python scripts/train_fusion_classifier.py --features meta/acdc_features.csv --fusion-type rap --use-cross-attention --folds 5 --epochs 100 --batch-size 32 --lr 1e-3 --hidden-dim 128 --dropout 0.3 --seed 42
```

### Baselines
```bash
python scripts/classify_cv.py --features meta/acdc_features.csv --models logreg,rf,xgb --subset all --folds 5
```

### Results
```bash
python scripts/make_results_summary.py
```

---

## 📚 Documentation

- `README.md` - Project overview
- `PIPELINE.md` - Complete command reference
- `QUICK_START.md` - Setup guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `REPRODUCIBILITY.md` - Reproduction guide
- `results/RESULTS.md` - Latest results

---

## ✅ What's Fixed

1. ✅ File corruption (cram.py, train_fusion_classifier.py)
2. ✅ Script naming (train_fusion_cv.py → train_fusion_classifier.py)
3. ✅ Missing arguments (--fusion-type, --use-cross-attention)
4. ✅ Model signature (optional volume parameter)
5. ✅ PyTorch 2.6 (weights_only=False)
6. ✅ Baseline commands (--models plural)

---

**Last Updated**: November 5, 2025  
**Status**: ✅ All systems operational
