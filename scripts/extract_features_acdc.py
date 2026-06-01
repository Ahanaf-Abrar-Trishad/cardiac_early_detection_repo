#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import re

LABEL_MAP = {"RV":1, "MYO":2, "LV":3}  # ACDC label ids

def vol_ml(mask_path, label=None):
    nii = nib.load(str(mask_path))
    data = nii.get_fdata()
    data = (data == label).astype(np.float32) if label is not None else (data > 0).astype(np.float32)
    pixdim = nii.header.get_zooms()[:3]
    vxl_mm3 = float(pixdim[0] * pixdim[1] * pixdim[2])
    return float(data.sum() * vxl_mm3 / 1000.0)  # mL

def parse_info_cfg(cfg_path: Path) -> str:
    """Return ACDC Group label from Info.cfg (NOR/DCM/HCM/MINF/RV), else ''."""
    if not cfg_path.exists():
        return ""
    txt = cfg_path.read_text(errors="ignore")
    # Common patterns: "Group: NOR" or "Group = NOR"
    m = re.search(r"Group\s*[:=]\s*([A-Za-z]+)", txt)
    if not m: 
        return ""
    group = m.group(1).upper()
    # normalize
    valid = {"NOR","DCM","HCM","MINF","RV"}
    return group if group in valid else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--raw",  default="cardio_data/raw/acdc", help="Root folder that contains patientXXX/Info.cfg")
    ap.add_argument("--out",  default="meta/acdc_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df[df["dataset"]=="acdc"].copy()

    rows = []
    patients = sorted(df["patient_id"].unique())
    raw_root = Path(args.raw)

    for pid in patients:
        sub = df[df["patient_id"]==pid].copy()
        rec = {"patient_id": pid}

        # collect ED/ES mask paths from meta->paths json
        paths_by_phase = {}
        for phase in ["ED","ES"]:
            r = sub[sub["phase"]==phase]
            if r.empty: 
                continue
            try:
                P = json.loads(r.iloc[0]["paths"])
            except Exception:
                P = {}
            mpath = P.get(f"nii_mask_{phase}", "")
            if mpath and Path(mpath).exists():
                paths_by_phase[phase] = Path(mpath)

        # compute volumes if we have masks
        if "ED" in paths_by_phase:
            rec["LV_ED_mL"]  = vol_ml(paths_by_phase["ED"],  LABEL_MAP["LV"])
            rec["RV_ED_mL"]  = vol_ml(paths_by_phase["ED"],  LABEL_MAP["RV"])
            rec["MYO_ED_mL"] = vol_ml(paths_by_phase["ED"],  LABEL_MAP["MYO"])
        if "ES" in paths_by_phase:
            rec["LV_ES_mL"]  = vol_ml(paths_by_phase["ES"],  LABEL_MAP["LV"])
            rec["RV_ES_mL"]  = vol_ml(paths_by_phase["ES"],  LABEL_MAP["RV"])
            rec["MYO_ES_mL"] = vol_ml(paths_by_phase["ES"],  LABEL_MAP["MYO"])

        # derived EF
        if all(k in rec for k in ["LV_ED_mL","LV_ES_mL"]) and rec["LV_ED_mL"] > 0:
            rec["LV_EF"] = float((rec["LV_ED_mL"] - rec["LV_ES_mL"]) / rec["LV_ED_mL"])
        if all(k in rec for k in ["RV_ED_mL","RV_ES_mL"]) and rec["RV_ED_mL"] > 0:
            rec["RV_EF"] = float((rec["RV_ED_mL"] - rec["RV_ES_mL"]) / rec["RV_ED_mL"])

        # label: prefer meta['diagnosis'], else parse Info.cfg
        diag = ""
        if "diagnosis" in sub.columns and sub["diagnosis"].notna().any():
            diag = str(sub["diagnosis"].dropna().iloc[0]).upper()
        if not diag:
            # find raw patient folder (support patientXX or patient0XX styles)
            cand = raw_root / f"{pid}"
            if not cand.exists():
                cand = raw_root / f"patient{str(pid).replace('patient','')}"
            if not cand.exists():
                cand = raw_root / f"patient{str(pid).zfill(3).replace('patient','')}"
            # final fallback: exact pid like 'patient012'
            if not cand.exists():
                cand = raw_root / pid
            cfg = cand / "Info.cfg"
            diag = parse_info_cfg(cfg)

        rec["label"] = diag
        # only keep if we have at least a label and some features
        if diag and any(k in rec for k in ["LV_ED_mL","LV_ES_mL","RV_ED_mL","RV_ES_mL","LV_EF","RV_EF","MYO_ED_mL","MYO_ES_mL"]):
            rows.append(rec)

    feat = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.out, index=False)
    print(f"[extract_features_acdc] wrote {args.out} | patients: {len(patients)} -> usable: {len(feat)}")
    if feat.empty:
        print("WARNING: No usable rows. Check that processed masks exist and Info.cfg labels are readable.")

if __name__ == "__main__":
    main()
