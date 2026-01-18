# Cardiac Early Detection — Results\n\n## Segmentation (ACDC)\n- Dice: **0.721 ± 0.032**\n- IoU: **0.643 ± 0.034**\n\n## Segmentation (CAMUS)\n- Dice: **0.948 ± 0.002**\n- IoU: **0.901 ± 0.003**\n\n## Diagnosis (tabular, HGB + geometric features)\n|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |   0.466667 |            0.52     |   0.45326  |
|      2 |   0.6      |            0.584286 |   0.59177  |
|      3 |   0.433333 |            0.466667 |   0.398442 |
|      4 |   0.533333 |            0.513333 |   0.508333 |
|      5 |   0.566667 |            0.55     |   0.547233 |\n\nConfusion matrix saved to `results/acdc_diag_cm_geom.csv`.\n