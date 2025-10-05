#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import nibabel as nib

# --- import your repo code ---
from datasets.acdc_ed_es_3d import ACDC3DEDS
from scripts.seg_cv import UNet3D, set_all_seeds, loader_fast_kwargs, DEVICE

@torch.no_grad()
def save_nifti_pred(vol_pred, ref_path, out_path):
    ref = nib.load(ref_path)
    img = nib.Nifti1Image(vol_pred.astype(np.uint8), affine=ref.affine, header=ref.header)
    nib.save(img, str(out_path))

def run_oof(meta, phase, folds, feat3d, logdir, ckpt_pattern, oof_dir, batch_size=1, amp=False, with_bg=False):
    df = pd.read_csv(meta)
    g = df[df["dataset"] == "acdc"].reset_index(drop=True)

    # only rows with images available for the phase
    def has_image(row):
        try:
            p = json.loads(row["paths"])
            return bool(p.get(f"nii_image_{phase}", ""))
        except Exception:
            return False
    g = g[g.apply(has_image, axis=1)].reset_index(drop=True)
    groups = g["patient_id"].values

    base = ACDC3DEDS(meta, split="", phase=phase, aug=None)
    sid2idx = {row["study_id"]: i for i, row in base.df.reset_index().iterrows()}
    idxs = [sid2idx[sid] for sid in g["study_id"].values if sid in sid2idx]
    g = g.iloc[:len(idxs)].copy()

    gkf = GroupKFold(n_splits=folds)
    feat = tuple(int(x) for x in feat3d.split(",")) if feat3d else (16,32,64,128)
    out_ch = 4 if with_bg else 3  # 4-class (BG+RV+MYO+LV) or 3-class (RV+MYO+LV)

    oof_dir = Path(oof_dir); oof_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold, (_, va) in enumerate(gkf.split(g, groups=groups), start=1):
        ckpt_path = Path(ckpt_pattern.format(fold=fold))
        if not ckpt_path.exists():
            print(f"[warn] missing checkpoint for fold {fold}: {ckpt_path} — skipping fold.")
            continue

        print(f"== OOF infer {phase} | fold {fold} ==")
        va_idxs = np.array(idxs)[va]
        va_set  = Subset(base, list(va_idxs))
        loader  = DataLoader(va_set, batch_size=batch_size, shuffle=False, **loader_fast_kwargs(num_workers=2))

        # model
        model = UNet3D(in_ch=1, out_ch=out_ch, feat=feat).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        use_amp = bool(amp) and DEVICE == "cuda"
        amp_ctx = (torch.amp.autocast(device_type="cuda") if use_amp else torch.no_grad())

        # infer
        for b, batch in enumerate(loader):
            x = batch["image"].to(DEVICE)
            # Recover the dataset index for metadata
            ds_idx = va_idxs[b] if b < len(va_idxs) else None
            row_meta = base.df.iloc[ds_idx]
            study_id = row_meta["study_id"]; patient_id = row_meta["patient_id"]
            ref_path = json.loads(row_meta["paths"])[f"nii_image_{phase}"]

            with (amp_ctx if use_amp else torch.no_grad()):
                lg = model(x)  # [1,C,D,H,W]
                pred = torch.argmax(lg, dim=1, keepdim=False).cpu().numpy()[0]  # [D,H,W] in {0..C-1}

            out_path = oof_dir / f"{study_id}_{phase}_oof.nii.gz"
            save_nifti_pred(pred, ref_path, out_path)

            rows.append({
                "fold": fold, "phase": phase, "study_id": study_id, "patient_id": patient_id,
                "pred_path": str(out_path), "ref_img": ref_path
            })

    out_csv = Path("results") / f"acdc_oof_index_{phase}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[done] wrote OOF index: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--phase", choices=["ED","ES"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--feat3d", default="")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--ckpt-pattern", default="logs/seg_acdc_fold{fold}_best.pt")
    ap.add_argument("--oof-dir", default="logs/oof_preds/acdc")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--with-bg", action="store_true", help="Checkpoints were trained with BG class (C=4)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    run_oof(
        meta=args.meta, phase=args.phase, folds=args.folds,
        feat3d=args.feat3d, logdir=args.logdir,
        ckpt_pattern=args.ckpt_pattern,
        oof_dir=Path(args.oof_dir)/args.phase,
        batch_size=args.batch_size, amp=args.amp, with_bg=args.with_bg
    )

if __name__ == "__main__":
    main()
