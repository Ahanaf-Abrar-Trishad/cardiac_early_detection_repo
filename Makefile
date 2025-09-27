# Makefile — Cardiac Early Detection (CAMUS + ACDC)
# Run `make help` to see available targets.

# ===== Variables =====
ENV           ?= cardio-dl
PYTHON        ?= python
PYTHONPATH    ?= $(PWD)

RAW_CAMUS     ?= cardio_data/raw/camus
RAW_ACDC      ?= cardio_data/raw/acdc
PROC_CAMUS    ?= cardio_data/processed/camus
PROC_ACDC     ?= cardio_data/processed/acdc
META          ?= meta/master_metadata.csv
LOGDIR        ?= logs

# ===== Helpers =====
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ===== Environment =====
.PHONY: init cuda
init: ## Create/Update conda env
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	@echo "Activate with: conda activate $(ENV)"
cuda: ## Install PyTorch wheels via setup script
	bash setup_cuda_pytorch.sh

# ===== Data processing =====
.PHONY: camus acdc splits
camus: ## Process CAMUS -> $(PROC_CAMUS)
	$(PYTHON) scripts/camus_process.py --raw $(RAW_CAMUS) --out $(PROC_CAMUS) --size 256
acdc: ## Process ACDC -> $(PROC_ACDC)
	$(PYTHON) scripts/acdc_process.py --raw $(RAW_ACDC) --out $(PROC_ACDC) --target_spacing 1.25 1.25 10.0
splits: ## Create patient-level splits and write $(META)
	$(PYTHON) scripts/make_splits.py --meta $(META) --seed 42

# ===== Segmentation CV =====
.PHONY: seg2d seg3d
seg2d: ## CAMUS 2D U-Net CV (4CH/ED, 30 epochs, optimized)
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/seg_cv.py --dataset camus --view 4CH --phase ED --folds 5 \
	  --epochs 30 --batch-size 8 --lr 1e-3 --logdir logs_camus \
	  --amp --feat2d 32,64,128,256 \
	  --grad-clip 1.0 --accum 1 --num-workers 4
seg3d: ## ACDC 3D U-Net CV (ED, multiclass, 60 epochs, optimized)
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/seg_cv.py --dataset acdc --phase ED --folds 5 \
	  --epochs 60 --batch-size 1 --lr 1e-3 --logdir logs_acdc \
	  --acdc-multiclass --amp --feat3d 16,32,64,128 \
	  --grad-clip 1.0 --accum 2 --num-workers 4

# ===== Feature extraction =====
.PHONY: features_acdc
features_acdc: ## Extract ACDC volumetric/EF features -> meta/acdc_features.csv
	export PYTHONPATH=$(PYTHONPATH); \
	$(PYTHON) scripts/extract_features_acdc.py --meta $(META) --raw $(RAW_ACDC) --out meta/acdc_features.csv

# ===== Classification CV (ACDC tabular) =====
.PHONY: cls cls-ef cls-vol
cls: ## All features (logreg, rf, xgb) + MLflow logs
	$(PYTHON) scripts/classify_cv.py --features meta/acdc_features.csv --folds 5 --seed 42 --logdir $(LOGDIR) --mlflow --mlflow-experiment cls-cv
cls-ef: ## EF-only ablation
	$(PYTHON) scripts/classify_cv.py --features meta/acdc_features.csv --subset ef --folds 5 --seed 42 --logdir logs_ef
cls-vol: ## Volumes-only ablation + calibration
	$(PYTHON) scripts/classify_cv.py --features meta/acdc_features.csv --subset vol --calibrate --folds 5 --seed 42 --logdir logs_vol

# ===== Reports / Notebooks =====
.PHONY: reports
reports: ## Run notebooks manually (open in Jupyter) and write to reports/ & reports_seg/
	@echo "Open notebooks/cardiac_cls_report.ipynb and notebooks/cardiac_seg_report.ipynb and Run All."

# ===== Convenience =====
.PHONY: all clean
all: camus acdc splits seg2d seg3d features_acdc cls ## End-to-end (heavy)
clean: ## Remove logs and reports
	rm -rf logs logs_ef logs_vol reports reports_seg
