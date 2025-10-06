# Makefile — Cardiac Early Detection (CAMUS + ACDC)
# Run `make help` to see available targets.

# ===== Variables =====
ENV            ?= cardio-dl
PYTHON         ?= python
PYTHONPATH     ?= $(PWD)

RAW_CAMUS      ?= cardio_data/raw/camus
RAW_ACDC       ?= cardio_data/raw/acdc
PROC_CAMUS     ?= cardio_data/processed/camus
PROC_ACDC      ?= cardio_data/processed/acdc
META           ?= meta/master_metadata.csv
LOGDIR         ?= logs

# Segmentation defaults (override on CLI if needed, e.g., `make seg3d PHASE=ES`)
PHASE          ?= ED
FOLDS          ?= 5
EPOCHS         ?= 60
BATCH          ?= 1
LR             ?= 1e-3
FEAT3D         ?= 16,32,64,128
ACCUM          ?= 2
NUM_WORKERS    ?= 4
AMP            ?= --amp

# Loss / class weights for ACDC multiclass (RV,MYO,LV)
CLASS_WEIGHTS  ?= auto
CE_W           ?= 1.0
DICE_W         ?= 0.7

# 3D augmentation knobs
AUG3D_FLAGS    ?= --aug3d --p-flip 0.5 --p-rot 0.5 --rot-deg 12 \
                  --p-gamma 0.5 --gamma-min 0.9 --gamma-max 1.15 \
                  --p-bright 0.5 --bright-min 0.9 --bright-max 1.1

# OOF / features
CKPT_PATTERN   ?= logs/seg_acdc_fold{fold}_best.pt
OOF_DIR        ?= logs/oof_preds/acdc
PERCLASS_CSV   ?= results/acdc_per_class_dice.csv

# ===== Helpers =====
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ===== Environment =====
.PHONY: init cuda torch check-setup
init: ## Create/Update conda env
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	@echo "Activate with: conda activate $(ENV)"
cuda: ## (Optional) Install PyTorch CUDA wheels via your setup script
	bash setup_cuda_pytorch.sh
torch: ## (Optional) Quick PyTorch install (edit the index URL to match your CUDA)
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
check-setup: ## Print torch / CUDA info
	$(PYTHON) scripts/check_setup.py

# ===== Data processing =====
.PHONY: camus acdc splits
camus: ## Process CAMUS -> $(PROC_CAMUS)
	$(PYTHON) scripts/camus_process.py --raw $(RAW_CAMUS) --out $(PROC_CAMUS) --size 256
acdc: ## Process ACDC -> $(PROC_ACDC)
	$(PYTHON) scripts/acdc_process.py --raw $(RAW_ACDC) --out $(PROC_ACDC) --target_spacing 1.25 1.25 10.0
splits: ## Create patient-level splits and write $(META)
	$(PYTHON) scripts/make_splits.py --meta $(META) --seed 42

# ===== Segmentation CV =====
.PHONY: seg2d seg3d seg3d-ed seg3d-es
seg2d: ## CAMUS 2D U-Net CV (example, 4CH/ED)
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/seg_cv.py --dataset camus --view 4CH --phase ED --folds 5 \
	  --epochs 30 --batch-size 8 --lr 1e-3 --logdir $(LOGDIR) \
	  --amp --feat2d 32,64,128,256 --grad-clip 1.0 --accum 1 --num-workers 4

seg3d: ## ACDC 3D U-Net CV (multiclass) with class-weights + 3D augs
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/seg_cv.py --dataset acdc --phase $(PHASE) --folds $(FOLDS) \
	  --epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR) --logdir $(LOGDIR) \
	  --acdc-multiclass $(AMP) --feat3d $(FEAT3D) --grad-clip 1.0 --accum $(ACCUM) --num-workers $(NUM_WORKERS) \
	  --save-val-previews --preview-batches 3 --perclass-csv $(PERCLASS_CSV) \
	  --class-weights $(CLASS_WEIGHTS) --ce-weight $(CE_W) --dice-weight $(DICE_W) \
	  $(AUG3D_FLAGS)

seg3d-ed: ## ACDC CV (ED phase)
	$(MAKE) seg3d PHASE=ED
seg3d-es: ## ACDC CV (ES phase)
	$(MAKE) seg3d PHASE=ES

# ===== OOF inference (ACDC, multiclass) =====
.PHONY: oof-ed oof-es oof-all
oof-ed: ## Write results/acdc_oof_index_ED.csv and NIfTIs under $(OOF_DIR)/ED
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/oof_infer_acdc.py --phase ED --folds $(FOLDS) --amp \
	  --ckpt-pattern $(CKPT_PATTERN) --oof-dir $(OOF_DIR)
oof-es: ## Write results/acdc_oof_index_ES.csv and NIfTIs under $(OOF_DIR)/ES
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/oof_infer_acdc.py --phase ES --folds $(FOLDS) --amp \
	  --ckpt-pattern $(CKPT_PATTERN) --oof-dir $(OOF_DIR)
oof-all: ## ED + ES
	$(MAKE) oof-ed
	$(MAKE) oof-es

# ===== Features from OOF (robust + geometric) =====
.PHONY: features-geom labels
features-geom: ## Build robust volumes/EF + geometry to results/acdc_oof_features_geom.csv
	$(PYTHON) scripts/build_features_geom.py
labels: ## Export labels (patient_id, diagnosis) to results/acdc_labels.csv
	$(PYTHON) scripts/extract_acdc_labels.py

# ===== Diagnosis CV (tabular) =====
.PHONY: diag-geom
diag-geom: ## HGB on robust+geom features (5x GroupKFold)
	$(PYTHON) scripts/train_diag_geom.py

# ===== Results summary =====
.PHONY: results-md
results-md: ## Generate RESULTS.md summary
	$(PYTHON) scripts/make_results_md.py

# ===== QA & Dev =====
.PHONY: fmt lint test qa
fmt: ## Format with black & isort
	black .
	isort .
lint: ## Lint with ruff
	ruff check .
test: ## Run unit tests
	pytest -q
qa: ## Lint + tests
	$(MAKE) lint
	$(MAKE) test

# ===== Reports / Notebooks =====
.PHONY: reports
reports: ## Run notebooks manually (open in Jupyter) and write to reports/ & reports_seg/
	@echo "Open notebooks/cardiac_cls_report.ipynb and notebooks/cardiac_seg_report.ipynb and Run All."

# ===== Convenience =====
.PHONY: all clean
all: camus acdc splits seg2d seg3d-ed oof-all features-geom labels diag-geom results-md ## End-to-end (heavy)
clean: ## Remove logs and reports
	rm -rf logs logs_ef logs_vol reports reports_seg
