#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cardio-dl"
RAW_CAMUS="${RAW_CAMUS:-cardio_data/raw/camus}"
RAW_ACDC="${RAW_ACDC:-cardio_data/raw/acdc}"
SEED="${SEED:-42}"

echo ">> Creating/updating conda env: $ENV_NAME"
conda env create -f environment.yml || conda env update -f environment.yml
echo ">> Activate with: conda activate $ENV_NAME"
# Attempt auto-activation if running in an interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

if [ -f "setup_cuda_pytorch.sh" ]; then
  echo ">> Running CUDA/PyTorch setup"
  bash setup_cuda_pytorch.sh
else
  echo ">> setup_cuda_pytorch.sh not found; skipping CUDA-specific torch install"
fi

echo ">> Processing CAMUS from $RAW_CAMUS"
python scripts/camus_process.py --raw "$RAW_CAMUS" --out cardio_data/processed/camus --size 256 || true

echo ">> Processing ACDC from $RAW_ACDC"
python scripts/acdc_process.py  --raw "$RAW_ACDC" --out cardio_data/processed/acdc --target_spacing 1.25 1.25 10.0 || true

echo ">> Making splits (seed=$SEED)"
python scripts/make_splits.py --meta meta/master_metadata.csv --seed "$SEED"

echo ">> Starting CAMUS CV + HPO"
python scripts/torch_cv.py --meta meta/master_metadata.csv --view 4CH --phase ED --folds 5 --seed "$SEED" --trials 25

echo ">> Done."
