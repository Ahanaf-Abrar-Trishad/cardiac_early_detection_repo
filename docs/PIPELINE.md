# 🫀 ACDC Cardiac Early Detection - Full Pipeline Commands

## 🚀 Quick Start

### One-shot via Makefile
```bash
make seg3d-ed       # ACDC ED segmentation CV
make seg3d-es       # ACDC ES segmentation CV
make oof-all        # OOF inference for ED+ES
make features-geom  # Robust + geometric features
make diag-geom      # Diagnosis baseline (geom)
```

### Skip to a specific step
```bash
# Already have segmentation? build features + diagnosis only
make features-geom diag-geom

# Only re-run diagnosis with existing features
make diag-geom
```

---

## 📋 Individual Commands (Step-by-Step)

### 1️⃣ Segmentation Training

#### ACDC ED Phase (UNet3D)
```bash
python scripts/seg_cv.py \
  --dataset acdc \
  --model unet3d \
  --phase ED \
  --acdc-multiclass \
  --folds 5 \
  --epochs 40 \
  --aug3d \
  --class-weights auto
```

#### ACDC ES Phase (UNet3D)
```bash
python scripts/seg_cv.py \
  --dataset acdc \
  --model unet3d \
  --phase ES \
  --acdc-multiclass \
  --folds 5 \
  --epochs 40 \
  --aug3d \
  --class-weights auto
```

#### ACDC with CRAM (Advanced)
```bash
python scripts/seg_cv.py \
  --dataset acdc \
  --model unet3d_cram \
  --phase ED \
  --acdc-multiclass \
  --folds 5 \
  --epochs 40 \
  --aug3d \
  --class-weights auto
```

#### ACDC with UNETR (Transformer-based)
```bash
python scripts/seg_cv.py \
  --dataset acdc \
  --model unetr \
  --phase ED \
  --acdc-multiclass \
  --folds 5 \
  --epochs 50 \
  --batch-size 1 \
  --lr 5e-4 \
  --amp \
  --aug3d \
  --class-weights auto
```

#### CAMUS Segmentation
```bash
python scripts/seg_cv.py \
  --dataset camus \
  --model unet \
  --folds 5 \
  --epochs 50 \
  --aug
```

---

### 2️⃣ Out-of-Fold (OOF) Inference

#### Generate ED Phase Predictions
```bash
python scripts/oof_infer_acdc.py \
  --phase ED \
  --folds 5 \
  --with-bg
```

#### Generate ES Phase Predictions
```bash
python scripts/oof_infer_acdc.py \
  --phase ES \
  --folds 5 \
  --with-bg
```

**Alternative: Run Both Phases**
```bash
./run_feature_extraction.sh
```

---

### 3️⃣ Feature Extraction

#### Extract ACDC Features
```bash
python scripts/extract_features_acdc.py
```

#### Extract ACDC Ejection Fraction
```bash
python scripts/extract_acdc_ef.py
```

#### Extract CAMUS Ejection Fraction
```bash
python scripts/extract_camus_ef.py
```

#### Build Geometric Features
```bash
python scripts/build_features_geom.py
```

---

### 4️⃣ Fusion Classifier Training ⭐ NEW

#### RAP Fusion with Cross-Attention (Best Performance)
```bash
python scripts/train_fusion_classifier.py \
  --features meta/acdc_features.csv \
  --fusion-type rap \
  --use-cross-attention \
  --folds 5 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --hidden-dim 128 \
  --dropout 0.3 \
  --seed 42
```

#### RAP Fusion (Without Cross-Attention)
```bash
python scripts/train_fusion_classifier.py \
  --features meta/acdc_features.csv \
  --fusion-type rap \
  --folds 5 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --hidden-dim 128 \
  --dropout 0.3 \
  --seed 42
```

#### Simple Concatenation Baseline
```bash
python scripts/train_fusion_classifier.py \
  --features meta/acdc_features.csv \
  --fusion-type concat \
  --folds 5 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --hidden-dim 128 \
  --dropout 0.3 \
  --seed 42
```

---

### 5️⃣ Traditional Classification Baselines

#### All Baselines at Once (Recommended)

```bash
python scripts/classify_cv.py \
  --features meta/acdc_features.csv \
  --models logreg,rf,xgb \
  --subset all \
  --folds 5
```

#### Individual Models

Logistic Regression:
```bash
python scripts/classify_cv.py \
  --features meta/acdc_features.csv \
  --models logreg \
  --subset all \
  --folds 5
```

Random Forest:
```bash
python scripts/classify_cv.py \
  --features meta/acdc_features.csv \
  --models rf \
  --subset all \
  --folds 5
```

XGBoost:
```bash
python scripts/classify_cv.py \
  --features meta/acdc_features.csv \
  --models xgb \
  --subset all \
  --folds 5
```

---

### 6️⃣ Results Generation

#### Generate RESULTS.md Summary
```bash
python scripts/make_results_summary.py
```

#### Generate Markdown Report
```bash
python scripts/make_results_md.py
```

---

## 📊 Expected Results

### Current Performance (Based on Latest Run)

| Model | Accuracy | Balanced Acc | F1 Score | AUC |
|-------|----------|--------------|----------|-----|
| **RAP Fusion + Cross-Attention** | **92.67%** | **92.67%** | **92.63%** | **98.47%** |
| Logistic Regression | ~85% | ~85% | ~85% | ~92% |
| Random Forest | ~88% | ~88% | ~88% | ~95% |
| XGBoost | ~90% | ~90% | ~90% | ~96% |

### Segmentation Performance

| Dataset | Phase | Dice Score | HD95 |
|---------|-------|------------|------|
| ACDC | ED | ~0.92 | ~8mm |
| ACDC | ES | ~0.90 | ~9mm |
| CAMUS | Both | ~0.91 | ~7mm |

---

## 🔧 Troubleshooting

### Common Issues

1. **Missing checkpoints**: Run segmentation training first
   ```bash
   make seg3d-ed seg3d-es
   ```

2. **Missing features**: Run feature extraction
   ```bash
   make features-geom
   ```

3. **OOF predictions not found**: Make sure to use `--with-bg` flag
   ```bash
   python scripts/oof_infer_acdc.py --phase ED --folds 5 --with-bg
   ```

4. **PyTorch 2.6 checkpoint loading error**: Already fixed in latest code
   - Uses `weights_only=False` for sklearn scaler compatibility

---

## 📁 Output Files

### Checkpoints
- `logs/seg_acdc_fold{1-5}_best.pt` - ACDC segmentation models
- `logs/seg_camus_fold{1-5}_best.pt` - CAMUS segmentation models
- `logs/fusion_classifier_fold{1-5}_best.pt` - Fusion classifier models

### Predictions
- `logs/oof_preds/acdc/ED/*.nii.gz` - ED phase OOF predictions
- `logs/oof_preds/acdc/ES/*.nii.gz` - ES phase OOF predictions

### Features
- `meta/acdc_features.csv` - Extracted features (volumes, EF)
- `results/acdc_oof_features_geom.csv` - Geometric features

### Results
- `logs/cv_cls_summary.json` - Classification cross-validation results
- `logs/fusion_classifier_cv_summary.json` - Fusion classifier results
- `logs/cv_seg_acdc_summary.json` - Segmentation results
- `results/RESULTS.md` - Complete results report

---

## 🎯 Recommended Workflow

### For First-Time Users
```bash
# Complete pipeline from scratch
make seg3d-ed seg3d-es oof-all features-geom diag-geom
```

### For Iterative Development
```bash
# Train only segmentation
make seg3d-ed seg3d-es

# Train only classification
python scripts/train_fusion_classifier.py --features meta/acdc_features.csv --fusion-type rap --use-cross-attention

# Update results only
make results-md
```

### For Experimentation
```bash
# Try different fusion architectures
python scripts/train_fusion_classifier.py --fusion-type rap --use-cross-attention ...
python scripts/train_fusion_classifier.py --fusion-type rap ...
python scripts/train_fusion_classifier.py --fusion-type concat ...

# Compare with baselines
python scripts/classify_cv.py --model logreg ...
python scripts/classify_cv.py --model rf ...
python scripts/classify_cv.py --model xgb ...
```

---

## ⚡ Performance Tips

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` for single GPU
2. **Batch size**: Reduce if OOM (e.g., `--batch-size 16`)
3. **AMP**: Use `--amp` flag for faster training (segmentation only)
4. **Parallel runs**: Run different folds on different GPUs
5. **Skip completed steps**: Use `--skip-*` flags to save time

---

## 📊 Segmentation Outputs & Reporting

- Per-fold metrics: `logs/cv_seg_{dataset}_metrics.csv` with Dice/IoU/Accuracy/F1 and optional per-class ACDC columns.
- Summary stats: `logs/cv_seg_{dataset}_summary.json` (mean/std plus 95% CI from the CV folds).
- Per-class ACDC breakdown: `logs/cv_seg_acdc_multiclass_perclass.csv` when multiclass.
- Artifacts: best checkpoints `logs/seg_*_best.pt` and validation overlays in `logs/runs/*`.

## 📚 Additional Resources

- `QUICK_START.md` - Quick setup guide
- `REPRODUCIBILITY.md` - Reproduction instructions
- `ATTENTION_TRAINING_GUIDE.md` - Attention model commands
- `RESULTS.md` - Output format and latest metrics
