# Cardiac Early Detection — Results\n\n## Segmentation (ACDC)\n- Dice: **0.652 ± 0.043**\n- IoU: **0.567 ± 0.044**\n\n## Segmentation (CAMUS)\n- Dice: **0.947 ± 0.002**\n- IoU: **0.900 ± 0.004**\n\n## Diagnosis (tabular, HGB + geometric features)\n|   fold |   accuracy |   balanced_accuracy |   macro_f1 |
|-------:|-----------:|--------------------:|-----------:|
|      1 |   0.4      |            0.47     |   0.368283 |
|      2 |   0.6      |            0.590952 |   0.603232 |
|      3 |   0.533333 |            0.558333 |   0.508333 |
|      4 |   0.533333 |            0.513333 |   0.490714 |
|      5 |   0.566667 |            0.592857 |   0.544505 |\n\nConfusion matrix saved to `results/acdc_diag_cm_geom.csv`.\n