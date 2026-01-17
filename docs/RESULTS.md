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

### Attention-based models (latest runs, 5-fold CV)

| model                 | acc    | bal_acc | f1_macro | f1_weighted | auc   |
|-----------------------|-------:|--------:|---------:|------------:|------:|
| advanced              | 0.947  | 0.947   | 0.9463   | 0.9463      | 0.9908 |
| tabular_transformer   | 0.947  | 0.947   | 0.9457   | 0.9457      | 0.9911 |
| multimodal            | 0.920  | 0.920   | 0.9188   | 0.9188      | 0.9881 |
| graph (GAT)           | 0.787  | 0.787   | 0.7393   | 0.7393      | 0.9692 |

Source: `logs/attention_{model}_cv_summary.json`

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
