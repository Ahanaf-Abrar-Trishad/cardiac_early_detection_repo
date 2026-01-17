#!/usr/bin/env python3
"""
Extract geometric features from CAMUS segmentation masks.
Since CAMUS is 2D echocardiography, we extract areas (in pixels) rather than volumes.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

def area_pixels(mask_path):
    """Calculate area in pixels for LV (assuming binary mask with 255=LV)."""
    if not Path(mask_path).exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # CAMUS masks are binary: 0=background, 255=LV
    lv_pixels = np.sum(mask > 127)  # Count pixels > 127 as LV
    return float(lv_pixels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--out", default="meta/camus_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df[df["dataset"] == "camus"].copy()

    rows = []
    patients = sorted(df["patient_id"].unique())

    for pid in patients:
        sub = df[df["patient_id"] == pid].copy()
        rec = {"patient_id": pid}

        # Collect ED/ES mask paths from meta->paths json
        paths_by_phase = {}
        for phase in ["ED", "ES"]:
            r = sub[(sub["phase"] == phase) & (sub["view"] == "4CH")]  # Use 4CH view
            if r.empty:
                continue
            try:
                P = json.loads(r.iloc[0]["paths"])
            except Exception:
                P = {}
            mpath = P.get(f"mask_{phase}", "")
            if mpath and Path(mpath).exists():
                paths_by_phase[phase] = Path(mpath)

        # Compute areas if we have masks
        if "ED" in paths_by_phase:
            rec["LV_ED_pixels"] = area_pixels(paths_by_phase["ED"])
        if "ES" in paths_by_phase:
            rec["LV_ES_pixels"] = area_pixels(paths_by_phase["ES"])

        # Get EF from metadata if available
        ef_row = sub[(sub["phase"] == "ED") & (sub["view"] == "4CH")]
        if not ef_row.empty and "ef_lv" in ef_row.columns:
            ef_val = ef_row["ef_lv"].dropna()
            if not ef_val.empty:
                rec["LV_EF"] = float(ef_val.iloc[0])

        # For CAMUS, we don't have diagnosis labels in the same way as ACDC
        # We'll use this for feature fusion later
        rec["dataset"] = "camus"

        # Only keep if we have some features
        if any(k in rec for k in ["LV_ED_pixels", "LV_ES_pixels", "LV_EF"]):
            rows.append(rec)

    feat = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.out, index=False)
    print(f"[extract_features_camus] wrote {args.out} | patients: {len(patients)} -> usable: {len(feat)}")

    if not feat.empty:
        print(f"Features extracted: {list(feat.columns)}")
        print(f"Sample patient: {feat.iloc[0].to_dict()}")

if __name__ == "__main__":
    main()