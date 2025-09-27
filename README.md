# Cardiac Early Detection — CAMUS + ACDC

Deep-learning + classical ML pipeline for early cardiac condition detection from **echocardiography (CAMUS)** and **cardiac MRI (ACDC)**.

**What’s included**
- Data processing for CAMUS (2D echo) and ACDC (3D MRI)
- Patient-level, leakage-safe CV for **segmentation** (2D/3D U-Net)
- ACDC **feature extraction** (ED/ES volumes, EF, etc.) and **clinical classification** (NOR, DCM, HCM, MINF, RV)
- MLflow/W&B hooks, rich logs (CSV/JSON), confusion matrices, overlays & NIfTI preds
- Two report notebooks in `notebooks/`: **classification** and **segmentation** aggregators

---

## 1) Environment

```bash
# Create / activate environment from your files
conda env create -f environment.yml
conda activate cardio-dl

# Install the correct PyTorch for your CUDA (examples)
# CUDA 12.4 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Project Python deps (if not installed via environment.yml's pip step)
pip install -r requirements.txt
```
> Tip: A quick-start script is also provided: `setup_cuda_pytorch.sh`.

**Hardware notes** (example stable configs)
- ACDC 3D: `--batch-size 1` on a 16GB GPU (e.g., RTX 4080 Super) works well.
- CAMUS 2D: `--batch-size 8` typically fits comfortably on 16GB for 256×256 inputs.

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
      # Either the official "database/training/patientXXX" layout OR
      patientXXX/patientXXX_frame01.nii.gz, patientXXX_frame01_gt.nii.gz, ...
```

---

## 3) Build processed datasets

```bash
# CAMUS (resized to 256×256; ED/ES pairs and masks)
python scripts/camus_process.py --raw cardio_data/raw/camus     --out cardio_data/processed/camus --size 256

# ACDC (SAX resample to ~1.25×1.25×10.0 mm; ED/ES)
python scripts/acdc_process.py --raw cardio_data/raw/acdc     --out cardio_data/processed/acdc --target_spacing 1.25 1.25 10.0
```

Create **patient-level splits**:
```bash
python scripts/make_splits.py --meta meta/master_metadata.csv --seed 42
```

> You can also use `make` shortcuts if present:
> ```bash
> make camus
> make acdc
> make splits
> ```

---

## 4) Segmentation CV (patient-level, no leakage)

### CAMUS — 2D U-Net
```bash
# 4CH, ED phase, 5-fold CV
export PYTHONPATH=$(pwd)
python scripts/seg_cv.py --dataset camus --view 4CH --phase ED   --folds 5 --epochs 30 --batch-size 8 --lr 1e-3 --logdir logs   --amp --feat2d 32,64,128,256
```

### ACDC — 3D U-Net
```bash
# Multiclass RV/MYO/LV, ED phase
export PYTHONPATH=$(pwd)
python scripts/seg_cv.py --dataset acdc --phase ED   --folds 5 --epochs 30 --batch-size 1 --lr 1e-3 --logdir logs   --acdc-multiclass --amp --feat3d 16,32,64,128
```

**Outputs**
- `logs/cv_seg_<dataset>_metrics.csv` (per-fold Dice/IoU), `cv_seg_<dataset>_summary.json`
- Example overlays (`*.png`) for CAMUS and example NIfTI preds (`*.nii.gz`) for ACDC
- Optional MLflow/W&B artifacts if `--mlflow/--wandb` are used

---

## 5) ACDC feature extraction (ED/ES volumes & EF)

```bash
export PYTHONPATH=$(pwd)
python scripts/extract_features_acdc.py   --meta meta/master_metadata.csv   --raw  cardio_data/raw/acdc   --out  meta/acdc_features.csv
```
This writes `meta/acdc_features.csv` with per-patient LV/RV ED/ES volumes, EF, and label.

---

## 6) Clinical classification CV (NOR/DCM/HCM/MINF/RV)

We train **LogReg**, **RandomForest**, **XGBoost** on the tabular features with **StratifiedGroupKFold** (patient-level), **OneHot/Scaling/SMOTE/FS inside folds** (no leakage).

**All features**
```bash
python scripts/classify_cv.py --features meta/acdc_features.csv   --folds 5 --seed 42 --logdir logs --mlflow --mlflow-experiment cls-cv
```

**EF-only**
```bash
python scripts/classify_cv.py --features meta/acdc_features.csv   --subset ef --folds 5 --seed 42 --logdir logs_ef
```

**Volumes-only + probability calibration**
```bash
python scripts/classify_cv.py --features meta/acdc_features.csv   --subset vol --calibrate --folds 5 --seed 42 --logdir logs_vol
```

**Outputs**
- Per-model per-fold metrics: `logs/cv_cls_<model>_metrics.csv`
- Aggregates: `logs/cv_cls_summary.csv`
- Confusion matrices: `logs/cv_cls_<model>_fold*_cm.png`
- Feature importances / coefficients: `logs/feature_importance/`

---

## 7) Reports / notebooks

Two ready-to-run notebooks live in `notebooks/`:

- `cardiac_cls_report.ipynb` — Aggregates **classification** logs from `logs/`, `logs_ef/`, `logs_vol/` and writes plots/tables to `reports/`.
- `cardiac_seg_report.ipynb` — Aggregates **segmentation** logs (CAMUS & ACDC) and writes plots/tables to `reports_seg/`.

```bash
# From repo root, open the notebooks and Run All
# The reports will be saved under reports/ and reports_seg/
```

---

## 8) Reproducibility & leakage prevention

- **Patient-level CV**: GroupKFold / StratifiedGroupKFold with groups = patient IDs.
- **No leakage**: All encoders/scalers/SMOTE/feature selection fit **inside** training folds.
- **Determinism**: seeds fixed (`--seed`), PyTorch seeds set in scripts.
- **Documented env**: `environment.yml` + `requirements.txt`. Keep PyTorch installed to match your CUDA.
- **Paths & data**: metadata (`meta/master_metadata.csv`) is updated by processing scripts and split maker.

---


---

## 9) Makefile quickstart (recommended)

You already have a **Makefile** — great! Use it to run the common pipelines without remembering long commands.

```bash
# List all available targets (if your Makefile has a 'help' target)
make help
```

Typical targets you should have (names may vary slightly — check `make help`):

```bash
# Data processing
make camus      # Process CAMUS into cardio_data/processed/camus
make acdc       # Process ACDC into cardio_data/processed/acdc
make splits     # Create patient-level splits into meta/master_metadata.csv

# Segmentation CV
make seg2d      # CAMUS 2D U-Net CV (e.g., 4CH/ED, 5 folds, AMP)
make seg3d      # ACDC 3D U-Net CV (ED, multiclass RV/MYO/LV, AMP)

# Clinical classification CV
make cls        # ACDC classification (all features); you may also have cls-ef, cls-vol
```

> If `make seg2d`, `make seg3d`, or `make cls` aren’t present yet, you can add them by mapping
> to the exact Python commands used above. A common pattern is:

```makefile
# Example help target to auto-generate usage
help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  [36m%-18s[0m %s
", $$1, $$2}'

camus:  ## Process CAMUS
	python scripts/camus_process.py --raw cardio_data/raw/camus --out cardio_data/processed/camus --size 256

acdc:   ## Process ACDC
	python scripts/acdc_process.py --raw cardio_data/raw/acdc --out cardio_data/processed/acdc --target_spacing 1.25 1.25 10.0

splits: ## Create patient-level splits
	python scripts/make_splits.py --meta meta/master_metadata.csv --seed 42

seg2d:  ## CAMUS 2D U-Net CV
	export PYTHONPATH=$(PWD); python scripts/seg_cv.py --dataset camus --view 4CH --phase ED --folds 5 --epochs 30 --batch-size 8 --lr 1e-3 --logdir logs --amp --feat2d 32,64,128,256

seg3d:  ## ACDC 3D U-Net CV (multiclass)
	export PYTHONPATH=$(PWD); python scripts/seg_cv.py --dataset acdc --phase ED --folds 5 --epochs 30 --batch-size 1 --lr 1e-3 --logdir logs --acdc-multiclass --amp --feat3d 16,32,64,128

cls:    ## ACDC classification (all features)
	python scripts/classify_cv.py --features meta/acdc_features.csv --folds 5 --seed 42 --logdir logs --mlflow --mlflow-experiment cls-cv
```

Use `make` wherever possible to keep your workflow reproducible and one-command simple.

## 10) Citations

- **CAMUS**: Leclerc, S. *et al.* “Deep Learning for Segmentation of the Left Ventricle from Echocardiographic Data.” (CAMUS dataset).
- **ACDC**: Bernard, O. *et al.* “Deep Learning Techniques for Cardiac MR Segmentation: A Short Review.” (ACDC challenge).

Please follow the datasets’ official citation rules in publications.

---

## License
For academic/research use. Check dataset licenses and terms.
