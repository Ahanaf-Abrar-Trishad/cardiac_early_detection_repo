#!/usr/bin/env python3
"""
Process ACDC to standardized ED/ES volumes + masks with resampling.
- Supports standard ACDC layout with patientXXX_frameNN.nii.gz (+ _gt) and Info.cfg.
- Also supports already-renamed *_ED.nii.gz / *_ES.nii.gz if present.

Outputs:
  {out}/images_3d/patientXXXX_ED.nii.gz
  {out}/images_3d/patientXXXX_ES.nii.gz
  {out}/masks_3d/patientXXXX_ED.nii.gz
  {out}/masks_3d/patientXXXX_ES.nii.gz
and appends rows to meta/master_metadata.csv with keys:
  paths["nii_image_ED"], paths["nii_mask_ED"], paths["nii_image_ES"], paths["nii_mask_ES"]
"""
import argparse, re, json
from pathlib import Path

try:
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

except Exception as e:
    raise SystemExit(
        "SimpleITK is required for ACDC processing. Install via "
        "`conda install -c conda-forge simpleitk` or `pip install SimpleITK`."
    ) from e

# Optional: silence ITK warnings like "has unexpected scales in sform"
# (uncomment one of the two lines depending on your SimpleITK version)
# sitk.ProcessObject_SetGlobalWarningDisplay(False)
# sitk.ProcessObject.SetGlobalWarningDisplay(False)


def parse_info_cfg(cfg_path: Path):
    """Robustly read ED/ES frame numbers from Info.cfg (formats vary)."""
    ed = es = None
    txt = cfg_path.read_text(errors="ignore")
    m = re.search(r"ED\s*[:=]\s*(\d+)", txt, re.I)
    if m: ed = int(m.group(1))
    m = re.search(r"ES\s*[:=]\s*(\d+)", txt, re.I)
    if m: es = int(m.group(1))
    return ed, es


def load_nii(p: Path):
    return sitk.ReadImage(str(p))


def resample_like(img, target_spacing, is_label=False):
    """
    Resample to target_spacing while preserving origin/direction.
    BSpline for images, NearestNeighbor for labels.
    """
    spacing = img.GetSpacing()
    size = img.GetSize()
    new_spacing = tuple(target_spacing)
    new_size = [
        int(round(size[i] * (spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputPixelType(img.GetPixelID())
    return resampler.Execute(img)


def save_nii(img, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def find_patient_dirs(root: Path):
    """Return patient directories named like patientXXX under root."""
    return [p for p in sorted(root.iterdir()) if p.is_dir() and re.match(r"patient\d+", p.name)]


def locate_frames(pdir: Path, pid: str):
    """
    Return dict with paths for ED/ES image & mask.
    Prefers standard frameNN per Info.cfg; falls back to *_ED/_ES if already present.
    """
    # Already-renamed?
    ed_img = pdir / f"{pid}_ED.nii.gz"
    es_img = pdir / f"{pid}_ES.nii.gz"
    ed_msk = pdir / f"{pid}_ED_gt.nii.gz"
    es_msk = pdir / f"{pid}_ES_gt.nii.gz"
    if ed_img.exists() and es_img.exists():
        return {
            "ED_img": ed_img, "ES_img": es_img,
            "ED_msk": ed_msk if ed_msk.exists() else None,
            "ES_msk": es_msk if es_msk.exists() else None
        }

    # Standard ACDC layout with frames + Info.cfg
    cfg = pdir / "Info.cfg"
    if not cfg.exists():
        return None
    ed, es = parse_info_cfg(cfg)
    if ed is None or es is None:
        return None

    def fpath(n): return pdir / f"{pid}_frame{int(n):02d}.nii.gz"
    def mpath(n): return pdir / f"{pid}_frame{int(n):02d}_gt.nii.gz"

    ED_img = fpath(ed); ES_img = fpath(es)
    ED_msk = mpath(ed) if mpath(ed).exists() else None
    ES_msk = mpath(es) if mpath(es).exists() else None

    if ED_img.exists() and ES_img.exists():
        return {"ED_img": ED_img, "ES_img": ES_img, "ED_msk": ED_msk, "ES_msk": ES_msk}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Folder containing patientXXX subfolders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_spacing", nargs=3, type=float,
                    default=[1.25, 1.25, 10.0], metavar=("SX", "SY", "SZ"))
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_img_dir = out_root / "images_3d"
    out_msk_dir = out_root / "masks_3d"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = Path("meta/master_metadata.csv")
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    pats = find_patient_dirs(raw_root)
    processed = 0
    rows = []

    for pdir in pats:
        pid = pdir.name  # patientXXXX
        paths = locate_frames(pdir, pid)
        if not paths:
            continue

        # Load
        ed_img = load_nii(paths["ED_img"])
        es_img = load_nii(paths["ES_img"])
        ed_msk = load_nii(paths["ED_msk"]) if paths["ED_msk"] else None
        es_msk = load_nii(paths["ES_msk"]) if paths["ES_msk"] else None

        # Resample
        tgt = tuple(args.target_spacing)
        ed_img_r = resample_like(ed_img, tgt, is_label=False)
        es_img_r = resample_like(es_img, tgt, is_label=False)
        if ed_msk:
            ed_msk_r = resample_like(ed_msk, tgt, is_label=True)
        if es_msk:
            es_msk_r = resample_like(es_msk, tgt, is_label=True)

        # Save standardized filenames
        ed_img_o = out_img_dir / f"{pid}_ED.nii.gz"
        es_img_o = out_img_dir / f"{pid}_ES.nii.gz"
        save_nii(ed_img_r, ed_img_o)
        save_nii(es_img_r, es_img_o)
        ed_msk_o = out_msk_dir / f"{pid}_ED.nii.gz" if ed_msk else None
        es_msk_o = out_msk_dir / f"{pid}_ES.nii.gz" if es_msk else None
        if ed_msk:
            save_nii(ed_msk_r, ed_msk_o)
        if es_msk:
            save_nii(es_msk_r, es_msk_o)

        # Meta rows (two rows, one per phase)
        rows.append({
            "dataset": "acdc", "patient_id": pid, "study_id": f"{pid}_ED",
            "view": "SAX", "phase": "ED",
            "paths": json.dumps({
                "nii_image_ED": str(ed_img_o),
                "nii_mask_ED":  (str(ed_msk_o) if ed_msk else "")
            })
        })
        rows.append({
            "dataset": "acdc", "patient_id": pid, "study_id": f"{pid}_ES",
            "view": "SAX", "phase": "ES",
            "paths": json.dumps({
                "nii_image_ES": str(es_img_o),
                "nii_mask_ES":  (str(es_msk_o) if es_msk else "")
            })
        })
        processed += 1

    if processed == 0:
        print("No ACDC patients processed. Check your raw folder structure.")
        return

    # Append/update meta
    import pandas as pd
    df_new = pd.DataFrame(rows, columns=["dataset","patient_id","study_id","view","phase","paths"])
    if meta_csv.exists():
        df = pd.read_csv(meta_csv)
        df = df[df["dataset"] != "acdc"].reset_index(drop=True)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(meta_csv, index=False)
    print(f"Processed {processed} ACDC patients -> {out_root} and updated {meta_csv}")


if __name__ == "__main__":
    main()

