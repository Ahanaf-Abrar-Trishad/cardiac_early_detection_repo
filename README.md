# Cardiac Early Detection — CAMUS + ACDC

Deep-learning + classical ML pipeline for early cardiac condition detection from **echocardiography (CAMUS)** and **cardiac MRI (ACDC)**.

## What’s new (polished pipeline)
- **ACDC 3D U‑Net multiclass (RV/MYO/LV)** with **class‑weighted CE** (`--class-weights auto|w1,w2,w3`) + **multiclass Dice** (`--dice-weight`).
- **3D augmentations** (random flips, in‑plane rotations, gamma/brightness jitter) with simple flags (`--aug3d ...`).
- **Per‑epoch per‑class CSV logging** for ACDC (`--perclass-csv results/acdc_per_class_dice.csv`) and validation preview overlays (`--save-val-previews`).
- **OOF inference** script to generate fold‑wise predictions and indexes.
- **Robust volume/EF & geometry features** from OOF segmentations + a compact **diagnosis** baseline (NOR/DCM/HCM/MINF/RV).
- **Makefile** targets for one‑command execution.

---

## 1) Environment

```bash
# Create / activate the environment
conda env create -f environment.yml
conda activate cardio-dl

# Install PyTorch that matches your CUDA (examples)
# CUDA 12.1 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Project deps if not installed via environment.yml
pip install -r requirements.txt
```
> Tip: set `export PYTHONPATH=$(pwd):$PYTHONPATH` before running any repo script.

**Hardware notes**
- ACDC 3D: `--batch-size 1` on a 16GB GPU is a good default.
- CAMUS 2D: `--batch-size 8` usually fits on 16GB for 256×256 inputs.

---

## 2) Data layout

Place the original datasets (unmodified) under:
```
cardio_data/
  raw/
    camus/
      patientXXXX/
        patientXXXX_2CH_ED.nii.gz
        patientXXXX_4CH_ED.nii.gz
        ...
    acdc/
      patientXXX/
        # ED/ES short-axis volumes (+ ground truths if available)
        patientXXX_ED.nii.gz
        patientXXX_ES.nii.gz
        ...
```

---

## 3) Build processed datasets + splits

```bash
# CAMUS (resized to 256×256; ED/ES pairs and masks)
python scripts/camus_process.py --raw cardio_data/raw/camus --out cardio_data/processed/camus --size 256

# ACDC (SAX resample to ~1.25×1.25×10.0 mm; ED/ES)
python scripts/acdc_process.py --raw cardio_data/raw/acdc --out cardio_data/processed/acdc --target_spacing 1.25 1.25 10.0

# Create patient-level splits and write meta
python scripts/make_splits.py --meta meta/master_metadata.csv --seed 42
```

---

## 4) Segmentation CV (patient-level, no leakage)

Always export PYTHONPATH:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### CAMUS — 2D U‑Net (binary LV)
```bash
python scripts/seg_cv.py --dataset camus --view 4CH --phase ED --folds 5 \
  --epochs 30 --batch-size 8 --lr 1e-3 --logdir logs \
  --amp --feat2d 32,64,128,256 \
  --grad-clip 1.0 --accum 1 --num-workers 4
```

### ACDC — 3D U‑Net (multiclass RV/MYO/LV)
```bash
python scripts/seg_cv.py --dataset acdc --phase ED --folds 5 \
  --epochs 60 --batch-size 1 --lr 1e-3 --logdir logs \
  --acdc-multiclass --amp --feat3d 16,32,64,128 \
  --grad-clip 1.0 --accum 2 --num-workers 4 \
  --save-val-previews --preview-batches 3 --perclass-csv results/acdc_per_class_dice.csv \
  --class-weights auto --ce-weight 1.0 --dice-weight 0.7 \
  --aug3d --p-flip 0.5 --p-rot 0.5 --rot-deg 12 --p-gamma 0.5 --gamma-min 0.9 --gamma-max 1.15 \
  --p-bright 0.5 --bright-min 0.9 --bright-max 1.1
```

**Key outputs**
- `logs/seg_acdc_fold{1..5}_best.pt` — best checkpoints per fold.
- `logs/cv_seg_acdc_metrics.csv`, `logs/cv_seg_acdc_summary.json` — CV metrics.
- `results/acdc_per_class_dice.csv` — per‑epoch per‑class Dice/IoU (when multiclass).
- `logs/runs/acdc_val_previews/.../*.png` — ED/ES preview overlays.

---

## 5) OOF inference (ACDC multiclass)

Run out‑of‑fold inference for both phases to produce NIfTIs + an index CSV for each phase.
```bash
# ED
python scripts/oof_infer_acdc.py --phase ED --folds 5 --amp \
  --ckpt-pattern logs/seg_acdc_fold{fold}_best.pt \
  --oof-dir logs/oof_preds/acdc

# ES
python scripts/oof_infer_acdc.py --phase ES --folds 5 --amp \
  --ckpt-pattern logs/seg_acdc_fold{fold}_best.pt \
  --oof-dir logs/oof_preds/acdc
```
**Outputs**
- `results/acdc_oof_index_ED.csv`, `results/acdc_oof_index_ES.csv`.
- NIfTIs: `logs/oof_preds/acdc/ED/*.nii.gz`, `logs/oof_preds/acdc/ES/*.nii.gz`.

---

## 6) Build robust volumes/EF & geometry features

From the OOF segmentations we assemble robust LV/RV ED/ES volumes, robust EF estimates, and simple geometry features.

```bash
python scripts/build_features_geom.py
```
**Outputs (under `results/`)**
- `acdc_oof_features_robust.csv`  — robust volumes + EF (with NaN counts reported).
- `acdc_oof_features_geom.csv`    — adds geometric features (myocardial thickness stats, axis ratios, etc.).

Export labels (if needed):
```bash
python - << 'PY'
import pandas as pd, pathlib
df = pd.read_csv("meta/master_metadata.csv")
lab = (df[df["dataset"]=="acdc"][["patient_id","diagnosis"]]
         .drop_duplicates().sort_values("patient_id"))
pathlib.Path("results").mkdir(parents=True, exist_ok=True)
lab.to_csv("results/acdc_labels.csv", index=False)
print("Saved results/acdc_labels.csv")
PY
```

---

## 7) Diagnosis baseline (NOR/DCM/HCM/MINF/RV)

Train a compact baseline on the robust+geom features with group‑aware CV.
```bash
python scripts/train_diag_geom.py
```
**Outputs (under `results/`)**
- `acdc_diag_cm_geom.csv` — confusion matrix.
- `acdc_diag_feature_importance.csv` — simple feature importances.
- Console shows **Acc**, **Balanced Acc**, **Macro‑F1**.

---

## 8) Makefile quickstart

If you prefer one‑liners, use the included `Makefile`:

```bash
make help                  # list targets
make acdc                  # preprocess ACDC
make splits                # write meta/master_metadata.csv
make seg3d-ed              # ACDC CV (ED)
make seg3d-es              # ACDC CV (ES)
make oof-all               # OOF inference for ED + ES
make features-geom         # build robust + geometric features
make labels                # export labels to results/
make diag-geom             # diagnosis CV with robust+geom features
make results-md            # generate RESULTS.md
```

You can also override defaults, e.g.:
```bash
make seg3d PHASE=ED EPOCHS=80 CLASS_WEIGHTS=1.0,1.4,1.0 DICE_W=1.0
```

---

## 9) Troubleshooting

- **`ModuleNotFoundError: datasets`** → You forgot PYTHONPATH. Run:
  ```bash
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```
- **CUDA OOM** → Reduce `--batch-size`, increase `--accum`, or switch off some augs.
- **Weird/negative EF** → Ensure ED/ES are correctly paired and that ACDC label ids were remapped (1→RV, 2→MYO, 3→LV → 0/1/2). The pipeline does this internally for training/eval; robust EF further mitigates outliers.
- **Slow dataloading** → Bump `--num-workers` and ensure your data is on fast storage (SSD/NVMe).

---

## 10) Reproducibility & leakage prevention

- **Patient-level CV** (GroupKFold) everywhere.
- **No leakage**: any scaling/selection happens inside folds.
- **Seeds set** in scripts; deterministic helpers where practical.
- Documented env: `environment.yml` + `requirements.txt`.

---

## 11) Citations

- **CAMUS**: Leclerc, S. *et al.* “Deep Learning for Segmentation of the Left Ventricle from Echocardiographic Data.” (CAMUS dataset).
- **ACDC**: Bernard, O. *et al.* “Deep Learning Techniques for Cardiac MR Segmentation.” (ACDC challenge).  
Please follow the datasets’ official citation rules in publications.

---

## 12) License

For academic/research use. Check dataset licenses and terms.

---

## 13) Documentation

All detailed documentation is organized in the **`docs/`** folder:

- **[docs/README.md](docs/README.md)** - Complete documentation index
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Quick start guide
- **[docs/TRAINING_READY_SUMMARY.md](docs/TRAINING_READY_SUMMARY.md)** - Training setup
- **[docs/ATTENTION_TRAINING_GUIDE.md](docs/ATTENTION_TRAINING_GUIDE.md)** - Attention models guide
- **[docs/PIPELINE.md](docs/PIPELINE.md)** - Full pipeline documentation
- **[docs/RESULTS.md](docs/RESULTS.md)** - Experimental results

See **[docs/README.md](docs/README.md)** for the complete documentation index.

