# Cardiac Early Detection — Results

## Segmentation (ACDC)
- Dice: **0.721 ± 0.032**
- IoU: **0.643 ± 0.034**

## Segmentation (CAMUS)
- Dice: **0.948 ± 0.002**
- IoU: **0.901 ± 0.003**

## Diagnosis (tabular, HGB + geometric features)
|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |   0.466667 |            0.52     |   0.45326  |
|      2 |   0.6      |            0.584286 |   0.59177  |
|      3 |   0.433333 |            0.466667 |   0.398442 |
|      4 |   0.533333 |            0.513333 |   0.508333 |
|      5 |   0.566667 |            0.55     |   0.547233 |

Confusion matrix saved to `results/acdc_diag_cm_geom.csv`.
