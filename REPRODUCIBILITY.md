# Reproducibility

- **Python** pinned in `environment.yml` (3.11). Use `conda env create -f environment.yml` then `conda activate <env>`.
- **Torch/CUDA**: install via `setup_cuda_pytorch.sh` to match your GPU (avoids wheel/driver mismatch).
- **Seeds**: all training scripts set `torch`, `numpy`, and `random` seeds (default 42), and set CuDNN to deterministic.
- **Exact splits**: patient-level CV splits are deterministic by `--seed`. Saved CSVs appear in `meta/` and metrics in `logs/`.
- **Scripts & Notebooks**: every pipeline step is runnable from `Makefile`/`run.sh` or `notebooks/`.
- **Data sources**: 
  - CAMUS: https://camus.creatis.insa-lyon.fr/challenge/  
  - ACDC:  https://www.creatis.insa-lyon.fr/Challenge/acdc/
- **Code**: This repository contains all processing, training, and evaluation code. Consider syncing to a public VCS (e.g., GitHub) and tagging a release for your thesis submission.

## Determinism notes
For GPU runs, full determinism can reduce speed and may still vary slightly across hardware/driver/library versions. We set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
to help stabilize results.
