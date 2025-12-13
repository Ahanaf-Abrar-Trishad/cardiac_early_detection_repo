# Cardiac Early Detection — Results

## Segmentation

**ACDC**
- Dice: **0.587 ± 0.029**
- IoU: **0.505 ± 0.023**

**CAMUS**
- Dice: **0.946 ± 0.002**
- IoU: **0.900 ± 0.004**

## Classification (Current)

### Geometric baseline (HistGB + geom features)
| fold | accuracy | balanced_accuracy | macro_f1 |
|---:|---:|---:|---:|
| 1 | 0.45 | 0.45 | 0.4442 |
| 2 | 0.45 | 0.45 | 0.4176 |
| 3 | 0.60 | 0.60 | 0.5492 |
| 4 | 0.50 | 0.50 | 0.4456 |
| 5 | 0.55 | 0.55 | 0.5190 |

Confusion matrix: `results/acdc_diag_cm_geom.csv`

## Classification (To Run / Compare)

The repository includes four attention-based classifiers:
- `advanced` (single-input attention)
- `multimodal` (geo + functional cross-attention)
- `tabular_transformer` (tokenized feature Transformer with phase token)
- `graph` (GAT over physiological tokens)

Train all attention models:
```bash
./run_attention_training.sh
```

Train a specific model (example: graph):
```bash
./run_attention_training.sh --model graph \
  --graph-hidden 128 --graph-heads 4 --graph-layers 2 --graph-dropout 0.2
```

Outputs to compare (after training):
- Summaries: `logs/attention_{model}_cv_summary.json`
- OOF predictions: `logs/oof_preds/attention_{model}_oof_predictions.csv`
- Checkpoints: `logs/attention_{model}_fold*_best.pt`

Update this doc by inserting the new metrics from the summary JSONs once runs complete.
