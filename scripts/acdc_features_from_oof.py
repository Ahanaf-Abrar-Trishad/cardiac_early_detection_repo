#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

def vol_ml(mask_path, label_id):
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    z = img.header.get_zooms()[:3]
    v_mm3 = float(z[0] * z[1] * z[2])
    vox = (data == label_id).sum()
    return vox * v_mm3 / 1000.0  # mL

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_oof(csv_path):
    df = pd.read_csv(csv_path)
    df = _norm_cols(df)
    need = {"study_id","patient_id","phase","pred_path"}
    if not need.issubset(set(df.columns)):
        missing = need - set(df.columns)
        raise RuntimeError(f"{csv_path} missing columns: {missing}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-ed-csv", required=True, help="results/acdc_oof_index_ED.csv")
    ap.add_argument("--oof-es-csv", required=True, help="results/acdc_oof_index_ES.csv")
    ap.add_argument("--out-csv", default="results/acdc_oof_features.csv")
    ap.add_argument("--save-per-slice", action="store_true", help="(optional) also compute per-slice areas (slow)")
    args = ap.parse_args()

    ed = load_oof(args.oof_ed_csv)
    es = load_oof(args.oof_es_csv)

    # Try study_id join first
    ed_sid = ed[["study_id","patient_id","pred_path"]].rename(columns={"pred_path":"pred_ED"})
    es_sid = es[["study_id","pred_path"]].rename(columns={"pred_path":"pred_ES"})
    merged = ed_sid.merge(es_sid, on="study_id", how="inner")
    used_patient_join = False

    # Fallback: join on patient_id (keep both study_id columns)
    if merged.empty:
        ed_pid = ed[["patient_id","study_id","pred_path"]].rename(columns={"study_id":"study_id_ED","pred_path":"pred_ED"})
        es_pid = es[["patient_id","study_id","pred_path"]].rename(columns={"study_id":"study_id_ES","pred_path":"pred_ES"})
        merged = ed_pid.merge(es_pid, on="patient_id", how="inner")
        used_patient_join = True

    if merged.empty:
        raise RuntimeError("Could not pair ED and ES predictions. Check your OOF CSVs.")

    rows = []
    for _, r in merged.iterrows():
        if used_patient_join:
            pid = r["patient_id"]
            sid_ed = r["study_id_ED"]; sid_es = r["study_id_ES"]
            pED = r["pred_ED"];        pES = r["pred_ES"]
        else:
            pid = r["patient_id"]
            sid_ed = r["study_id"];    sid_es = r["study_id"]   # same id both phases
            pED = r["pred_ED"];        pES = r["pred_ES"]

        # labels: 0=RV, 1=MYO, 2=LV
        RVED = vol_ml(pED, 0); RVES = vol_ml(pES, 0)
        MYED = vol_ml(pED, 1); MYES = vol_ml(pES, 1)
        LVED = vol_ml(pED, 2); LVES = vol_ml(pES, 2)

        RVEF = 100.0 * (RVED - RVES) / RVED if RVED > 1e-6 else np.nan
        LVEF = 100.0 * (LVED - LVES) / LVED if LVED > 1e-6 else np.nan

        row = {
            "patient_id": pid,
            "RVEDV_ml": RVED, "RVESV_ml": RVES, "RVEF_pct": RVEF,
            "LVEDV_ml": LVED, "LVESV_ml": LVES, "LVEF_pct": LVEF,
            "MYO_ED_ml": MYED, "MYO_ES_ml": MYES,
            "MYO_ES_to_ED_ratio": (MYES / MYED) if MYED > 1e-6 else np.nan,
            "LVES_to_ED_ratio": (LVES / LVED) if LVED > 1e-6 else np.nan,
            "RVES_to_ED_ratio": (RVES / RVED) if RVED > 1e-6 else np.nan,
        }
        if used_patient_join:
            row["study_id_ED"] = sid_ed
            row["study_id_ES"] = sid_es
        else:
            row["study_id"] = sid_ed
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["study_id","patient_id","study_id_ED"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[done] wrote features → {args.out_csv}")
    print(out.head())

if __name__ == "__main__":
    main()
