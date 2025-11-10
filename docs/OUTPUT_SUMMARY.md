# Segmentation Output Summary

## ✅ What You'll Get Now

After running segmentation training, you will get **comprehensive metrics per fold and overall summary**:

### **Per-Fold Metrics** (saved in CSV)

**File**: `logs/cv_seg_{dataset}_metrics.csv`

| fold | Dice | IoU | Accuracy | F1_macro | F1_weighted | artifact | best_ckpt |
|------|------|-----|----------|----------|-------------|----------|-----------|
| 1 | 0.8523 | 0.7654 | 0.9234 | 0.8123 | 0.8456 | path/to/img | path/to/model |
| 2 | 0.8612 | 0.7789 | 0.9312 | 0.8234 | 0.8567 | path/to/img | path/to/model |
| 3 | 0.8501 | 0.7623 | 0.9201 | 0.8098 | 0.8423 | path/to/img | path/to/model |
| 4 | 0.8567 | 0.7701 | 0.9267 | 0.8167 | 0.8501 | path/to/img | path/to/model |
| 5 | 0.8589 | 0.7734 | 0.9289 | 0.8189 | 0.8534 | path/to/img | path/to/model |

**For ACDC multiclass**, additional columns:
- `Dice_RV`, `Dice_MYO`, `Dice_LV`
- `IoU_RV`, `IoU_MYO`, `IoU_LV`

---

### **Summary Statistics** (saved in JSON)

**File**: `logs/cv_seg_{dataset}_summary.json`

```json
{
  "Dice_mean": 0.8558,
  "Dice_std": 0.0042,
  "IoU_mean": 0.7700,
  "IoU_std": 0.0063,
  "Accuracy_mean": 0.9261,
  "Accuracy_std": 0.0042,
  "F1_macro_mean": 0.8162,
  "F1_macro_std": 0.0054,
  "F1_weighted_mean": 0.8496,
  "F1_weighted_std": 0.0051,
  "folds": 5,
  "dataset": "acdc",
  "phase": "ED",
  "model": "unetr",
  "feat3d": "16,32,64,128",
  "amp": true,
  ...
}
```

**For ACDC multiclass**, additional fields:
- `Dice_RV_mean`, `Dice_RV_std`
- `Dice_MYO_mean`, `Dice_MYO_std`
- `Dice_LV_mean`, `Dice_LV_std`
- `IoU_RV_mean`, `IoU_RV_std`
- `IoU_MYO_mean`, `IoU_MYO_std`
- `IoU_LV_mean`, `IoU_LV_std`

---

### **Console Output Example**

```
=== ACDC Segmentation | Fold 1/5 ===
[info] Using UNETR (Transformer-based)
[Fold 1] Epoch 005/50 | train_loss=0.2134 | dice=0.8234 iou=0.7123 | acc=0.9123 f1=0.8012 | lr 4.50e-04
  ↳ best so far: 0.8234 @ seg_acdc_fold1_best.pt
[Fold 1] Epoch 010/50 | train_loss=0.1876 | dice=0.8456 iou=0.7345 | acc=0.9234 f1=0.8134 | lr 4.23e-04
  ↳ best so far: 0.8456 @ seg_acdc_fold1_best.pt
...
[Fold 1] Dice=0.8523 IoU=0.7654 Acc=0.9234 F1-macro=0.8123

=== ACDC Segmentation | Fold 2/5 ===
...

=== CV Summary ===
Dice      mean±std: 0.8558 ± 0.0042
IoU       mean±std: 0.7700 ± 0.0063
Accuracy  mean±std: 0.9261 ± 0.0042
F1-macro  mean±std: 0.8162 ± 0.0054
F1-weight mean±std: 0.8496 ± 0.0051
```

---

## 📊 Metric Definitions

| Metric | Description | Range |
|--------|-------------|-------|
| **Dice** | Overlap between prediction and ground truth | 0-1 (higher better) |
| **IoU** | Intersection over Union | 0-1 (higher better) |
| **Accuracy** | Pixel-wise correctness | 0-1 (higher better) |
| **F1-macro** | Harmonic mean of precision & recall (unweighted average across classes) | 0-1 (higher better) |
| **F1-weighted** | F1 score weighted by class frequency | 0-1 (higher better) |

---

## 📁 File Structure After Training

```
logs/
├── cv_seg_camus_metrics.csv          # Per-fold metrics (CAMUS)
├── cv_seg_camus_summary.json         # Summary statistics (CAMUS)
├── cv_seg_acdc_metrics.csv           # Per-fold metrics (ACDC)
├── cv_seg_acdc_summary.json          # Summary statistics (ACDC)
├── cv_seg_acdc_multiclass_perclass.csv  # ACDC per-class breakdown
├── seg_camus_fold1_best.pt           # Best model checkpoint
├── seg_camus_fold2_best.pt
├── ...
└── runs/
    └── acdc_val_previews/            # Validation overlay images
        ├── fold1/
        │   ├── epoch1/
        │   ├── epoch10/
        │   └── ...
        └── fold2/
            └── ...
```

---

## 🎯 Using the Metrics

### Compare Models

```bash
# Compare different architectures
python scripts/seg_cv.py --dataset acdc --model unet --epochs 30
python scripts/seg_cv.py --dataset acdc --model unet3d_cram --epochs 30
python scripts/seg_cv.py --dataset acdc --model unetr --epochs 30

# Compare results
cat logs/cv_seg_acdc_summary.json | grep "Dice_mean\|Accuracy_mean\|F1_macro_mean"
```

### Generate Report

```bash
# Aggregate all results
python scripts/make_results_summary.py

# View comprehensive report
cat results/RESULTS.md
```

---

## 💡 Tips

1. **Check per-fold variance**: High std means unstable training
2. **Monitor F1-macro vs F1-weighted**: 
   - F1-macro: All classes equally important
   - F1-weighted: Larger classes have more weight
3. **Use Accuracy + Dice together**: 
   - Accuracy: Overall pixel correctness
   - Dice: Segmentation quality (handles class imbalance better)
4. **For ACDC multiclass**: Check per-class Dice to find weak classes

---

**All metrics are now tracked, stored, and summarized!** 🎉
