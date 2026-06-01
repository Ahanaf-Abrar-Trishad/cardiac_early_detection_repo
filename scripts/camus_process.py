#!/usr/bin/env python3
"""
Process CAMUS dataset (supports .nii.gz and .mhd/.png) into 2D ED/ES PNGs + masks.
Writes to {out}/images and {out}/masks and updates meta/master_metadata.csv.

Expected raw per-patient files (your case):
  patientXXXX_2CH_ED.nii.gz
  patientXXXX_2CH_ES.nii.gz
  patientXXXX_2CH_ED_gt.nii.gz  (optional)
  patientXXXX_2CH_ES_gt.nii.gz
(similar for 4CH)
"""
import argparse, os, json, re, csv
from pathlib import Path
import numpy as np
import cv2

try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

def load_nii(path):
    if not _HAS_NIB:
        raise RuntimeError("nibabel not available to read NIfTI; please install nibabel.")
    img = nib.load(str(path)).get_fdata()
    # CAMUS NIfTI are 2D; if 3D, take middle slice
    if img.ndim == 3:
        img = img[..., img.shape[-1]//2]
    return img.astype(np.float32)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm01(x):
    x = x.astype(np.float32)
    m, M = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - m) / (M - m + 1e-6), 0, 1)
    return (x*255).astype(np.uint8)

def save_png(img, out_path, size):
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img)

def save_mask(msk, out_path, size):
    msk = (msk > 0).astype(np.uint8)*255
    if size:
        msk = cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), msk)

def find_patients(raw_root: Path):
    for p in sorted(raw_root.iterdir()):
        if p.is_dir() and re.match(r"patient\d+", p.name):
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--phase_only", choices=["ED","ES","both"], default="both")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_images = out_root / "images"
    out_masks  = out_root / "masks"
    ensure_dir(out_images); ensure_dir(out_masks)

    meta_path = Path("meta/master_metadata.csv")
    ensure_dir(meta_path.parent)
    meta_rows = []

    patients = list(find_patients(raw_root))
    if not patients:
        print("No patient folders like 'patientXXXX' found under", raw_root)
        return

    found = 0
    for pf in patients:
        files = {f.name: f for f in pf.iterdir() if f.is_file()}
        for view in ["2CH","4CH"]:
            for phase in ["ED","ES"]:
                if args.phase_only != "both" and phase != args.phase_only:
                    continue
                # prefer .nii.gz, fall back to .mhd/.png
                img_key = f"{pf.name}_{view}_{phase}.nii.gz"
                mhd_key = f"{view}_{phase}.mhd"
                mask_key = f"{pf.name}_{view}_{phase}_gt.nii.gz"
                mask_mhd = f"{view}_{phase}_gt.mhd"
                img_path = files.get(img_key) or files.get(mhd_key)
                msk_path = files.get(mask_key) or files.get(mask_mhd)
                if img_path is None:
                    continue

                # image
                if str(img_path).endswith(".nii.gz"):
                    img = load_nii(img_path)
                else:
                    try:
                        import SimpleITK as sitk
                        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
                        if img.ndim == 3: img = img[img.shape[0]//2]
                        img = img.astype(np.float32)
                    except Exception:
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

                png_name = f"{pf.name}_{view}_{phase}.png"
                out_img = out_images / png_name
                save_png(norm01(img), out_img, args.size)

                # mask (optional)
                out_msk = None
                if msk_path is not None:
                    if str(msk_path).endswith(".nii.gz"):
                        msk = load_nii(msk_path)
                    else:
                        try:
                            import SimpleITK as sitk
                            msk = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_path)))
                            if msk.ndim == 3: msk = msk[msk.shape[0]//2]
                            msk = msk.astype(np.float32)
                        except Exception:
                            msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    out_msk = out_masks / png_name
                    save_mask(msk, out_msk, args.size)

                # meta
                paths = {f"image_{phase}": str(out_img)}
                if out_msk is not None:
                    paths[f"mask_{phase}"] = str(out_msk)
                meta_rows.append({
                    "dataset": "camus",
                    "patient_id": pf.name,
                    "study_id": f"{pf.name}_{view}",
                    "view": view,
                    "phase": phase,
                    "paths": json.dumps(paths),
                })
                found += 1

    if found == 0:
        print("No CAMUS ED/ES pairs found. Check your raw folder structure.")
        return

    import pandas as pd
    if meta_path.exists():
        df = pd.read_csv(meta_path)
        df = df[df["dataset"]!="camus"].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["dataset","patient_id","study_id","view","phase","paths"])
    df = pd.concat([df, pd.DataFrame(meta_rows)], ignore_index=True)
    df.to_csv(meta_path, index=False)
    print(f"Processed {found} CAMUS items -> {out_root} and updated {meta_path}")

if __name__ == "__main__":
    main()

