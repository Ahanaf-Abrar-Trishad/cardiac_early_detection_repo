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
# 4CH, ED phase, 5-fold CV, 30 epochs (optimized)
export PYTHONPATH=$(pwd)
python scripts/seg_cv.py --dataset camus --view 4CH --phase ED --folds 5 \
  --epochs 30 --batch-size 8 --lr 1e-3 --logdir logs_camus \
  --amp --feat2d 32,64,128,256 \
  --grad-clip 1.0 --accum 1 --num-workers 4
```

### ACDC — 3D U-Net
```bash
# Multiclass RV/MYO/LV, ED phase, 60 epochs (optimized)
export PYTHONPATH=$(pwd)
python scripts/seg_cv.py --dataset acdc --phase ED --folds 5 \
  --epochs 60 --batch-size 1 --lr 1e-3 --logdir logs_acdc \
  --acdc-multiclass --amp --feat3d 16,32,64,128 \
  --grad-clip 1.0 --accum 2 --num-workers 4

# Optional: Add experiment tracking
# --mlflow --mlflow-experiment seg-cv
# --wandb --wandb-project cardiac-seg
```
> Or use `make` shortcuts if present:
> ```bash
> make seg2d
> make seg3d
> ```

### Key Optimizations

- **Gradient clipping** (`--grad-clip 1.0`): Prevents gradient explosion during training
- **Gradient accumulation** (`--accum 1-2`): Simulates larger batch sizes on limited GPU memory
- **Multi-worker loading** (`--num-workers 4`): Accelerates data loading
- **Separate log directories**: `logs_camus/` and `logs_acdc/` for organized outputs
- **Epoch tuning**: 30 epochs for CAMUS (2D), 60 epochs for ACDC (3D) based on convergence


**Outputs**
- `logs_camus/` or `logs_acdc/`: Cross-validation metrics (`cv_seg_<dataset>_metrics.csv`, `cv_seg_<dataset>_summary.json`)
- Example overlays (`*.png`) for CAMUS and example NIfTI predictions (`*.nii.gz`) for ACDC
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

## 8) Additional utility scripts

The `scripts/` directory includes several additional tools for advanced analysis:

- **`ablate_classification.py`** — Ablation study for CAMUS image classification (augmentation, ImageNet init, sampling strategies)
- **`extract_acdc_ef.py`** — Extract ACDC EF values from segmentation masks (calculates EF from LV volumes at ED/ES frames)
- **`make_results_summary.py`** — Aggregates all logs into a comprehensive `RESULTS.md` markdown report
- **`qc_report.py`** — Quick quality control: EF histograms and data distribution summaries
- **`tabular_cv.py`** — Alternative tabular classification pipeline with anti-leakage safeguards
- **`torch_cv.py`** — Deep learning classification CV for CAMUS with Optuna hyperparameter optimization

```bash
# Always set PYTHONPATH and activate environment first
export PYTHONPATH=$(pwd)
conda activate cardio-dl  # or your environment name

# Extract CAMUS EF labels (run first if CAMUS data lacks EF values)
python scripts/extract_camus_ef.py

# Extract ACDC EF values from segmentation masks
python scripts/extract_acdc_ef.py --meta meta/master_metadata.csv --data_root cardio_data/raw/acdc

# Generate comprehensive results summary
python scripts/make_results_summary.py

# Run quality control check on metadata
python scripts/qc_report.py --meta meta/master_metadata.csv

# Run CAMUS classification ablation study (correct arguments)
python scripts/ablate_classification.py --meta meta/master_metadata.csv --labels three --view 4CH --phase ED --out logs/ablation_cls.csv

# Tabular classification with anti-leakage pipeline
python scripts/tabular_cv.py --csv meta/acdc_features.csv --target label --folds 3 --logdir logs_tabular --categoricals patient_id

# Deep learning classification with hyperparameter optimization
python scripts/torch_cv.py --meta meta/master_metadata.csv --labels three --view 4CH --phase ED --folds 2 --trials 2 --logdir logs
```

> **💡 Tip**: Use the Makefile targets instead for easier execution:
> ```bash
> make ablation      # CAMUS ablation study
> make tabular-cv    # Tabular classification
> make torch-cv      # Deep learning classification
> make qc-report     # Quality control report
> make results       # Results summary
> ```

---

## 9) Reproducibility & leakage prevention

- **Patient-level CV**: GroupKFold / StratifiedGroupKFold with groups = patient IDs.
- **No leakage**: All encoders/scalers/SMOTE/feature selection fit **inside** training folds.
- **Determinism**: seeds fixed (`--seed`), PyTorch seeds set in scripts.
- **Documented env**: `environment.yml` + `requirements.txt`. Keep PyTorch installed to match your CUDA.
- **Paths & data**: metadata (`meta/master_metadata.csv`) is updated by processing scripts and split maker.

> **📋 See `REPRODUCIBILITY.md`** for detailed notes on determinism, data sources, and exact reproduction steps.

---

## 10) Makefile quickstart (recommended)

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

Use `make` wherever possible to keep your workflow reproducible and one-command simple.

## 11) Citations

- **CAMUS**: Leclerc, S. *et al.* “Deep Learning for Segmentation of the Left Ventricle from Echocardiographic Data.” (CAMUS dataset).
- **ACDC**: Bernard, O. *et al.* “Deep Learning Techniques for Cardiac MR Segmentation: A Short Review.” (ACDC challenge).

Please follow the datasets’ official citation rules in publications.

---

## License

For academic/research use. Check dataset licenses and terms.
