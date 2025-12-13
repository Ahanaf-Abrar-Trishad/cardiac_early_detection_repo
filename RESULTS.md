# Cardiac Early Detection — Results\n\n## Segmentation (ACDC)\n- Dice: **0.718 ± 0.030**\n- IoU: **0.639 ± 0.030**\n\n## Segmentation (CAMUS)\n- Dice: **0.948 ± 0.002**\n- IoU: **0.902 ± 0.003**\n\n## Diagnosis (tabular, HGB + geometric features)\n|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |   0.433333 |            0.48     |   0.435058 |
|      2 |   0.633333 |            0.629048 |   0.616934 |
|      3 |   0.6      |            0.633333 |   0.582018 |
|      4 |   0.5      |            0.484762 |   0.4929   |
|      5 |   0.666667 |            0.675    |   0.645696 |\n\nConfusion matrix saved to `results/acdc_diag_cm_geom.csv`.\n