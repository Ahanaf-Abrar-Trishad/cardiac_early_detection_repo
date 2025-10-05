#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import json

def main():
    out = Path("RESULTS.md")
    lines = []
    lines += ["# Cardiac Early Detection — Results\\n"]

    # Seg CV summary
    for ds in ["acdc","camus"]:
        p = Path(f"logs/cv_seg_{ds}_summary.json")
        if p.exists():
            s = json.loads(p.read_text())
            lines += [f"## Segmentation ({ds.upper()})",
                      f"- Dice: **{s['Dice_mean']:.3f} ± {s['Dice_std']:.3f}**",
                      f"- IoU: **{s['IoU_mean']:.3f} ± {s['IoU_std']:.3f}**\\n"]

    # Diagnosis CV
    p = Path("results/acdc_diag_cv_metrics_geom.csv")
    if p.exists():
        tab = pd.read_csv(p)
        lines += ["## Diagnosis (tabular, HGB + geometric features)"]
        lines += [tab.to_markdown(index=False), ""]
    if Path("results/acdc_diag_cm_geom.csv").exists():
        lines += ["Confusion matrix saved to `results/acdc_diag_cm_geom.csv`.\\n"]

    out.write_text("\\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
