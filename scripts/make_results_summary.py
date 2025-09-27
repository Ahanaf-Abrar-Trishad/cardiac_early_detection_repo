#!/usr/bin/env python3
"""
Aggregate logs to RESULTS.md: classification (binary & three-class) and segmentation (CAMUS/ACDC, incl. per-class Dice).
"""
import json, pandas as pd
from pathlib import Path

def fmt(m, s): 
    def f(x): 
        try: return f"{float(x):.3f}"
        except: return "nan"
    return f"{f(m)} ± {f(s)}"

def section_title(t): return f"# {t}\n\n"

def main():
    logs = Path("logs"); results = Path("results"); results.mkdir(parents=True, exist_ok=True)
    md = []

    # Classification summaries
    md.append(section_title("Classification"))
    for lab in ["three","binary"]:
        fp = logs / f"cv_image_{lab}_summary.json"
        if fp.exists():
            s = json.loads(fp.read_text())
            md.append(f"**{lab}**  ")
            md.append(f"- AUC (outer-CV): {fmt(s.get('AUC_mean'), s.get('AUC_std'))}")
            md.append(f"- ACC: {fmt(s.get('ACC_mean'), s.get('ACC_std'))}")
            md.append(f"- macro-F1: {fmt(s.get('macroF1_mean'), s.get('macroF1_std'))}")
            if s.get('AUC_macro_mean') is not None:
                md.append(f"- AUC (macro): {fmt(s.get('AUC_macro_mean'), s.get('AUC_macro_std'))}")
            if s.get('AUC_weighted_mean') is not None:
                md.append(f"- AUC (weighted): {fmt(s.get('AUC_weighted_mean'), s.get('AUC_weighted_std'))}")
            if s.get('AP_macro_mean') is not None:
                md.append(f"- AP (macro): {fmt(s.get('AP_macro_mean'), s.get('AP_macro_std'))}")
            if s.get('AP_weighted_mean') is not None:
                md.append(f"- AP (weighted): {fmt(s.get('AP_weighted_mean'), s.get('AP_weighted_std'))}")
            md.append("")
    md.append("")

    # Segmentation summaries
    md.append(section_title("Segmentation"))
    for ds in ["camus","acdc"]:
        fp = logs / f"cv_seg_{ds}_summary.json"
        if fp.exists():
            s = json.loads(fp.read_text())
            md.append(f"**{ds.upper()}**  ")
            md.append(f"- Dice: {fmt(s.get('Dice_mean'), s.get('Dice_std'))}")
            if s.get('IoU_mean') is not None and str(s.get('IoU_mean'))!='nan':
                md.append(f"- IoU: {fmt(s.get('IoU_mean'), s.get('IoU_std'))}")
            # per-class for ACDC multiclass
            if ds=='acdc' and s.get('Dice_RV_mean') is not None:
                md.append(f"- Dice (RV): {fmt(s.get('Dice_RV_mean'), s.get('Dice_RV_std'))}")
                md.append(f"- Dice (MYO): {fmt(s.get('Dice_MYO_mean'), s.get('Dice_MYO_std'))}")
                md.append(f"- Dice (LV): {fmt(s.get('Dice_LV_mean'), s.get('Dice_LV_std'))}")
            md.append("")
    md.append("")

    # Ablation table (if exists)
    abl = Path("logs/ablation_cls.csv")
    if abl.exists():
        md.append(section_title("Ablation (Classification)"))
        md.append("See `logs/ablation_cls.csv` for full table.")

    out = results / "RESULTS.md"
    out.write_text("\n".join(md))
    print("Wrote", out)

if __name__ == "__main__":
    main()
