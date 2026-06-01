#!/usr/bin/env python3
"""
Create patient-level train/val/test splits and update meta/master_metadata.csv.

- Splits are created independently per dataset (e.g., CAMUS vs ACDC).
- For CAMUS, if 'ef_bin' exists in meta (e.g., ['normal','mid','reduced']),
  we stratify by that label at the PATIENT level. Otherwise, random split.
- For ACDC, we fall back to random split (unless you add a patient-level label column).

Outputs:
- Updates 'split' column in meta/master_metadata.csv (dtype=string).
- Writes meta/splits_seed{SEED}.csv with per-dataset patient-level assignments.

Usage:
    python scripts/make_splits.py --meta meta/master_metadata.csv --seed 42 \
        --val-fraction 0.1 --test-fraction 0.2
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def choose_strat_label(df_patients: pd.DataFrame, dataset: str) -> pd.Series | None:
    """
    Return a patient-level label Series for stratification if available, else None.
    For CAMUS we try 'ef_bin' aggregated at patient level; for others we return None.
    """
    if dataset.lower() == "camus":
        if "ef_bin" in df_patients.columns:
            # If ef_bin exists and is non-null for many patients, use it.
            lab = df_patients["ef_bin"].astype("string")
            if lab.notna().sum() >= 3 and lab.nunique() >= 2:
                return lab
    # Add additional dataset-specific logic here if you add labels later (e.g., ACDC pathology).
    return None


def split_patients(pat_ids, labels, seed, val_fraction, test_fraction):
    """
    Split patient IDs into train/val/test.
    If labels is not None, use StratifiedShuffleSplit; else ShuffleSplit.
    """
    rng = np.random.RandomState(seed)
    pat_ids = np.array(pat_ids)

    # First carve out test
    if labels is not None:
        splitter_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=seed
        )
        (trainval_idx, test_idx) = next(splitter_test.split(pat_ids, labels))
    else:
        splitter_test = ShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        (trainval_idx, test_idx) = next(splitter_test.split(pat_ids, np.zeros(len(pat_ids))))

    trainval_ids = pat_ids[trainval_idx]
    test_ids = pat_ids[test_idx]
    labels_trainval = labels[trainval_idx] if labels is not None else None

    # Then carve val from the remaining trainval set
    if val_fraction > 0:
        val_size_rel = val_fraction / (1.0 - test_fraction)
        if labels_trainval is not None:
            splitter_val = StratifiedShuffleSplit(
                n_splits=1, test_size=val_size_rel, random_state=seed + 1
            )
            (train_idx, val_idx) = next(splitter_val.split(trainval_ids, labels_trainval))
        else:
            splitter_val = ShuffleSplit(
                n_splits=1, test_size=val_size_rel, random_state=seed + 1
            )
            (train_idx, val_idx) = next(splitter_val.split(trainval_ids, np.zeros(len(trainval_ids))))
        train_ids = trainval_ids[train_idx]
        val_ids = trainval_ids[val_idx]
    else:
        train_ids = trainval_ids
        val_ids = np.array([], dtype=trainval_ids.dtype)

    return set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv", help="Path to master metadata CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.20)
    args = ap.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # Ensure required columns exist
    for col in ["dataset", "patient_id"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {meta_path}")

    # We'll build a patient-level split table, then merge back to row-level.
    patient_rows = []

    # Split independently per dataset to avoid cross-dataset coupling
    for dataset, dsub in df.groupby("dataset", dropna=False):
        if pd.isna(dataset):
            continue
        # Build a patient-level frame for this dataset
        # Try to carry ef_bin to patient-level for CAMUS if present
        agg_cols = {"patient_id": "first"}
        if "ef_bin" in dsub.columns:
            # Take the most frequent ef_bin per patient (if mixed rows)
            ef_by_pat = (dsub.groupby("patient_id")["ef_bin"]
                         .agg(lambda x: x.dropna().value_counts().idxmax() if x.dropna().size > 0 else pd.NA))
            df_pat = pd.DataFrame({"patient_id": ef_by_pat.index, "ef_bin": ef_by_pat.values})
        else:
            df_pat = dsub[["patient_id"]].drop_duplicates().copy()
        df_pat["dataset"] = dataset

        # Decide stratification label (if any)
        strat_lab = choose_strat_label(df_pat, dataset)
        if strat_lab is not None:
            pat_ids = df_pat["patient_id"].tolist()
            labels = strat_lab.values
        else:
            pat_ids = df_pat["patient_id"].tolist()
            labels = None

        tr, va, te = split_patients(
            pat_ids=np.array(pat_ids),
            labels=np.array(labels) if labels is not None else None,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )

        for pid in pat_ids:
            if pid in tr:
                split = "train"
            elif pid in va:
                split = "val"
            elif pid in te:
                split = "test"
            else:
                split = "train"  # fallback; should not happen
            patient_rows.append({"dataset": dataset, "patient_id": pid, "split": split})

    splits_df = pd.DataFrame(patient_rows)
    splits_df = splits_df.sort_values(["dataset", "patient_id"]).reset_index(drop=True)

    # ---- Write patient-level split file
    meta_dir = meta_path.parent
    splits_out = meta_dir / f"splits_seed{args.seed}.csv"
    splits_out.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(splits_out, index=False)

    # ---- Merge splits back to master meta at row-level
    df = df.merge(splits_df, on=["dataset", "patient_id"], how="left", suffixes=("", "_new"))

    # Ensure 'split' exists and is string dtype (avoid pandas FutureWarning)
    if "split" not in df.columns:
        df["split"] = pd.Series(index=df.index, dtype="string")
    if df["split"].dtype != "string":
        df["split"] = df["split"].astype("string")

    # Fill/overwrite from merged column if present
    if "split_new" in df.columns:
        # Prefer new non-null assignments
        mask = df["split_new"].notna()
        df.loc[mask, "split"] = df.loc[mask, "split_new"].astype("string")
        df.drop(columns=["split_new"], inplace=True)

    # Save back
    df.to_csv(meta_path, index=False)

    print(f"Updated splits in {meta_path} and wrote {splits_out}")


if __name__ == "__main__":
    main()

