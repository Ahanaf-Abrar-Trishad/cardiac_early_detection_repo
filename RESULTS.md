# Cardiac Early Detection — Results\n\n## Segmentation (ACDC)\n- Dice: **0.598 ± 0.025**\n- IoU: **0.516 ± 0.016**\n\n## Segmentation (CAMUS)\n- Dice: **0.946 ± 0.002**\n- IoU: **0.900 ± 0.004**\n\n## Diagnosis (tabular, HGB + geometric features)\n|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |       0.45 |                0.45 |   0.430952 |
|      2 |       0.6  |                0.6  |   0.601429 |
|      3 |       0.45 |                0.45 |   0.370952 |
|      4 |       0.4  |                0.4  |   0.34     |
|      5 |       0.5  |                0.5  |   0.511746 |\n\nConfusion matrix saved to `results/acdc_diag_cm_geom.csv`.\n