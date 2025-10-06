# Cardiac Early Detection — Results\n\n## Segmentation (ACDC)\n- Dice: **0.587 ± 0.029**\n- IoU: **0.505 ± 0.023**\n\n## Segmentation (CAMUS)\n- Dice: **0.946 ± 0.002**\n- IoU: **0.900 ± 0.004**\n\n## Diagnosis (tabular, HGB + geometric features)\n|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |       0.45 |                0.45 |   0.444156 |
|      2 |       0.45 |                0.45 |   0.417619 |
|      3 |       0.6  |                0.6  |   0.549206 |
|      4 |       0.5  |                0.5  |   0.445599 |
|      5 |       0.55 |                0.55 |   0.519048 |\n\nConfusion matrix saved to `results/acdc_diag_cm_geom.csv`.\n