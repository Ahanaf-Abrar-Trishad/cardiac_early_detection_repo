#!/usr/bin/env python3
"""
Aggregate logs to RESULTS.md: classification (binary & three-class), fusion classifiers,
and segmentation (CAMUS/ACDC, incl. per-class Dice + Accuracy & F1).
"""
import json
import pandas as pd
from pathlib import Path

def fmt(m, s): 
    def f(x): 
        try: return f"{float(x):.3f}"
        except Exception: return "nan"
    return f"{f(m)} ± {f(s)}"

def section_title(t): 
    return f"## {t}\n\n"

def main():
    logs = Path("logs")
    results = Path("results")
    results.mkdir(parents=True, exist_ok=True)
    md = []
    
    md.append("# Cardiac Early Detection - Results Summary\n\n")
    
    # ==================== Classification ====================
    md.append(section_title("Classification Results"))
    
    # Standard classification
    for lab in ["three", "binary"]:
        fp = logs / f"cv_image_{lab}_summary.json"
        if fp.exists():
            s = json.loads(fp.read_text())
            md.append(f"### {lab.capitalize()} Class Classification\n")
            md.append(f"- **Accuracy**: {fmt(s.get('ACC_mean'), s.get('ACC_std'))}\n")
            md.append(f"- **Balanced Accuracy**: {fmt(s.get('bal_ACC_mean', s.get('ACC_mean')), s.get('bal_ACC_std', s.get('ACC_std')))}\n")
            md.append(f"- **F1 (macro)**: {fmt(s.get('macroF1_mean'), s.get('macroF1_std'))}\n")
            md.append(f"- **AUC**: {fmt(s.get('AUC_mean'), s.get('AUC_std'))}\n")
            if s.get('AUC_macro_mean') is not None:
                md.append(f"- **AUC (macro)**: {fmt(s.get('AUC_macro_mean'), s.get('AUC_macro_std'))}\n")
            if s.get('AP_macro_mean') is not None:
                md.append(f"- **AP (macro)**: {fmt(s.get('AP_macro_mean'), s.get('AP_macro_std'))}\n")
            md.append("\n")
    
    # Fusion classifiers
    md.append(f"### Feature Fusion Classifiers\n\n")
    for fusion_type in ["rap", "gated"]:
        fp = logs / f"cv_fusion_{fusion_type}_summary.json"
        if fp.exists():
            s = json.loads(fp.read_text())
            md.append(f"**{fusion_type.upper()} Fusion**\n")
            md.append(f"- **Accuracy**: {fmt(s.get('acc_mean'), s.get('acc_std'))}\n")
            md.append(f"- **Balanced Accuracy**: {fmt(s.get('bal_acc_mean'), s.get('bal_acc_std'))}\n")
            md.append(f"- **F1 (macro)**: {fmt(s.get('f1_macro_mean'), s.get('f1_macro_std'))}\n")
            md.append(f"- **AUC**: {fmt(s.get('auc_mean'), s.get('auc_std'))}\n")
            md.append(f"- **AP**: {fmt(s.get('ap_mean'), s.get('ap_std'))}\n")
            md.append("\n")
    
    # ==================== Segmentation ====================
    md.append(section_title("Segmentation Results"))
    
    for ds in ["camus", "acdc"]:
        fp = logs / f"cv_seg_{ds}_summary.json"
        if fp.exists():
            s = json.loads(fp.read_text())
            md.append(f"### {ds.upper()} Segmentation\n")
            
            # Overall metrics
            md.append(f"**Overall Metrics:**\n")
            md.append(f"- **Dice Score**: {fmt(s.get('Dice_mean'), s.get('Dice_std'))}\n")
            md.append(f"- **IoU**: {fmt(s.get('IoU_mean'), s.get('IoU_std'))}\n")
            
            # Add Accuracy and F1 if available
            if s.get('Accuracy_mean') is not None:
                md.append(f"- **Accuracy**: {fmt(s.get('Accuracy_mean'), s.get('Accuracy_std'))}\n")
            if s.get('F1_mean') is not None:
                md.append(f"- **F1 Score**: {fmt(s.get('F1_mean'), s.get('F1_std'))}\n")
            
            # Per-class metrics for ACDC
            if ds == 'acdc':
                md.append(f"\n**Per-Class Dice Scores:**\n")
                if s.get('Dice_RV_mean') is not None:
                    md.append(f"- **Right Ventricle (RV)**: {fmt(s.get('Dice_RV_mean'), s.get('Dice_RV_std'))}\n")
                if s.get('Dice_MYO_mean') is not None:
                    md.append(f"- **Myocardium (MYO)**: {fmt(s.get('Dice_MYO_mean'), s.get('Dice_MYO_std'))}\n")
                if s.get('Dice_LV_mean') is not None:
                    md.append(f"- **Left Ventricle (LV)**: {fmt(s.get('Dice_LV_mean'), s.get('Dice_LV_std'))}\n")
                
                # Per-class IoU if available
                if s.get('IoU_RV_mean') is not None:
                    md.append(f"\n**Per-Class IoU:**\n")
                    md.append(f"- **RV**: {fmt(s.get('IoU_RV_mean'), s.get('IoU_RV_std'))}\n")
                    md.append(f"- **MYO**: {fmt(s.get('IoU_MYO_mean'), s.get('IoU_MYO_std'))}\n")
                    md.append(f"- **LV**: {fmt(s.get('IoU_LV_mean'), s.get('IoU_LV_std'))}\n")
            
            md.append("\n")
    
    # ==================== Model Architectures ====================
    md.append(section_title("Model Architectures Used"))
    md.append("### Segmentation Models\n")
    md.append("- **2D U-Net**: CAMUS dataset (binary segmentation)\n")
    md.append("- **3D U-Net**: ACDC dataset (multi-class segmentation)\n")
    md.append("- **3D U-Net + CRAM**: U-Net with Context Recalibration Attention Module\n")
    md.append("- **UNETR**: Transformer-based 3D segmentation (Vision Transformer encoder + CNN decoder)\n\n")
    
    md.append("### Classification Models\n")
    md.append("- **Traditional ML**: Logistic Regression, Random Forest, XGBoost\n")
    md.append("- **RAP Fusion**: Feature fusion with Residual Attention Pooling blocks\n")
    md.append("- **Gated Fusion**: Learned gating mechanism for multi-modal feature weighting\n")
    md.append("- **Cross-Modal Attention**: Query-key-value attention for feature fusion\n\n")
    
    # ==================== Ablation Studies ====================
    abl = logs / "ablation_cls.csv"
    if abl.exists():
        md.append(section_title("Ablation Studies"))
        md.append("See `logs/ablation_cls.csv` for detailed ablation results.\n\n")
    
    # Write output
    out = results / "RESULTS.md"
    out.write_text("\n".join(md))
    print(f"✓ Results summary written to {out}")
    print(f"  - Classification: binary, three-class, fusion (RAP, gated)")
    print(f"  - Segmentation: CAMUS, ACDC (with per-class metrics)")
    print(f"  - Architectures: U-Net, U-Net+CRAM, UNETR, RAP/Gated fusion")


if __name__ == "__main__":
    main()
