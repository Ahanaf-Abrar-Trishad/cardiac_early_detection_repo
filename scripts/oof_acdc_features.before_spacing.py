#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
import nibabel as nib

from datasets.acdc_ed_es_3d import ACDC3DEDS
from utils.ef import volume_ml

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    # Matches seg_cv.py: anisotropic upsampling + 4-class head
    def __init__(self, in_ch=1, out_ch=4, feat=(16,32,64,128)):
        super().__init__()
        self.down1 = DoubleConv3D(in_ch,  feat[0])
        self.down2 = DoubleConv3D(feat[0], feat[1])
        self.down3 = DoubleConv3D(feat[1], feat[2])
        self.bott  = DoubleConv3D(feat[2], feat[3])
        self.up3   = nn.ConvTranspose3d(feat[3], feat[2], kernel_size=(1,2,2), stride=(1,2,2))
        self.conv3 = DoubleConv3D(feat[3], feat[2])
        self.up2   = nn.ConvTranspose3d(feat[2], feat[1], kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = DoubleConv3D(feat[2], feat[1])
        self.up1   = nn.ConvTranspose3d(feat[1], feat[0], kernel_size=(1,2,2), stride=(1,2,2))
        self.conv1 = DoubleConv3D(feat[1], feat[0])
        self.out   = nn.Conv3d(feat[0], out_ch, 1)

    @staticmethod
    def _adaptive_pool(x):
        import torch.nn.functional as F
        k = [1,2,2]
        D,H,W = x.shape[2], x.shape[3], x.shape[4]
        if H < 2: k[1] = 1
        if W < 2: k[2] = 1
        return F.max_pool3d(x, kernel_size=tuple(k), stride=tuple(k))

    def forward(self, x):
        d1 = self.down1(x); p1 = self._adaptive_pool(d1)
        d2 = self.down2(p1); p2 = self._adaptive_pool(d2)
        d3 = self.down3(p2); p3 = self._adaptive_pool(d3)
        b  = self.bott(p3)

        u3 = self.up3(b)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = nn.functional.interpolate(u3, size=d3.shape[2:], mode='trilinear', align_corners=False)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=d2.shape[2:], mode='trilinear', align_corners=False)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=d1.shape[2:], mode='trilinear', align_corners=False)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

def build_folds_from_valid(ds, valid_idx, n_splits=5):
    dfv = ds.df.iloc[valid_idx].reset_index(drop=True)
    groups = dfv['patient_id'].values
    idx = np.arange(len(dfv))
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for _, val_idx in gkf.split(idx, groups=groups):
        folds.append([valid_idx[i] for i in val_idx])  # map back to ds indices
    return folds

@torch.no_grad()
def infer_fold(model, loader, multiclass=True, device=DEVICE):
    model.eval(); outs = []
    for batch in loader:
        x = batch['image'].to(device)
        lg = model(x)
        if multiclass:
            pr = torch.argmax(lg, dim=1).cpu().numpy()[0]   # (Z,Y,X)
        else:
            pr = (torch.sigmoid(lg) >= 0.5).cpu().numpy()[0,0]  # (Z,Y,X)
        outs.append(pr)
    return outs

def _read_spacing_from_header(paths_dict, prefer_phase='ED'):
    """
    Try ED image header first; if missing, try ES.
    Returns (px, py, th) in mm or (nan, nan, nan) on failure.
    """
    def _try(phase):
        key = f"nii_image_{phase}"
        p = paths_dict.get(key)
        if not p: return None
        P = Path(p)
        if not P.exists():
            # try relative path from repo root
            P = Path.cwd() / p
            if not P.exists(): return None
        try:
            img = nib.load(str(P))
            z = img.header.get_zooms()  # (dx, dy, dz) in mm
            if len(z) >= 3:
                return float(z[0]), float(z[1]), float(z[2])
        except Exception:
            return None
        return None

    out = _try(prefer_phase)
    if out: return out
    other = 'ES' if prefer_phase == 'ED' else 'ED'
    out = _try(other)
    if out: return out
    return np.nan, np.nan, np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='meta/master_metadata_std.csv')
    ap.add_argument('--phase', default='ED', choices=['ED','ES'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--feat3d', default='16,32,64,128')
    ap.add_argument('--logdir', default='logs_acdc')
    ap.add_argument('--out-csv', default='meta/acdc_features_oof.csv')
    ap.add_argument('--save-pred-root', default='preds_acdc_oof')
    ap.add_argument('--multiclass', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = DEVICE
    feat3d = tuple(int(x) for x in args.feat3d.split(',')) if args.feat3d else (16,32,64,128)
    out_ch = 4 if args.multiclass else 1

    # Build dataset
    ds_all = ACDC3DEDS(args.meta, split="", phase=args.phase, aug=None)

    # Filter rows that actually have the expected keys
    want_img = f"nii_image_{args.phase}"
    want_msk = f"nii_mask_{args.phase}"
    valid_idx = []
    for i, r in ds_all.df.reset_index(drop=True).iterrows():
        try:
            d = json.loads(r['paths'])
        except Exception:
            continue
        if want_img in d and want_msk in d:
            valid_idx.append(i)

    if len(valid_idx) == 0:
        raise RuntimeError(f"No valid rows with keys {want_img} & {want_msk}. Check your metadata.")
    print(f"[INFO] Using {len(valid_idx)}/{len(ds_all.df)} rows with valid keys for phase {args.phase}.")

    # Folds on filtered rows
    folds = build_folds_from_valid(ds_all, valid_idx, n_splits=args.folds)

    save_root = Path(args.save_pred_root); save_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for f, val_ds_idxs in enumerate(folds):
        ckpt = Path(args.logdir) / f'seg_acdc_fold{f}_best.pt'
        if not ckpt.exists():
            print(f'[WARN] Missing checkpoint for fold {f}: {ckpt} — skipping this fold.')
            continue

        print(f'Fold {f}: loading {ckpt}')
        model = UNet3D(in_ch=1, out_ch=out_ch, feat=feat3d).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)

        va_set = Subset(ds_all, list(val_ds_idxs))
        loader = DataLoader(va_set, batch_size=1, shuffle=False, num_workers=0)

        preds = infer_fold(model, loader, multiclass=args.multiclass, device=device)
        for pr, ds_idx in zip(preds, val_ds_idxs):
            row = ds_all.df.iloc[int(ds_idx)]
            patient = row['patient_id']
            lv_pred = (pr==3).astype('uint8') if args.multiclass else (pr>0).astype('uint8')
            npy_path = save_root / f'{patient}_{args.phase}_fold{f}_lv.npy'
            np.save(npy_path, lv_pred)
            all_rows.append({'patient_id': patient, 'phase': args.phase, 'fold': f, 'pred_path': str(npy_path)})

    pred_index = pd.DataFrame(all_rows)
    pred_index.to_csv(save_root/f'index_{args.phase}.csv', index=False)
    print('Indexed predictions ->', save_root/f'index_{args.phase}.csv')

    # If both phases exist, compute EF/EDV/ESV
    ed_idx_path = save_root/'index_ED.csv'
    es_idx_path = save_root/'index_ES.csv'
    if ed_idx_path.exists() and es_idx_path.exists():
        ed_idx = pd.read_csv(ed_idx_path); es_idx = pd.read_csv(es_idx_path)
        merged = pd.merge(ed_idx, es_idx, on=['patient_id','fold'], suffixes=('_ED','_ES'))
        feats = []
        meta_df = ds_all.df  # already loaded/standardized

        for _, r in merged.iterrows():
            try:
                ed_mask = np.load(r['pred_path_ED'])
                es_mask = np.load(r['pred_path_ES'])

                # find a row for this patient and read spacing from NIfTI header
                mrows = meta_df[(meta_df['patient_id']==r['patient_id'])]
                px=py=th=np.nan
                if len(mrows)>0:
                    paths = json.loads(mrows.iloc[0]['paths'])
                    px, py, th = _read_spacing_from_header(paths, prefer_phase='ED')

                if np.isnan(px) or np.isnan(py) or np.isnan(th):
                    ef = np.nan; edv = np.nan; esv = np.nan
                else:
                    edv = volume_ml(ed_mask>0, px, py, th)
                    esv = volume_ml(es_mask>0, px, py, th)
                    ef  = (edv-esv)/edv if edv>0 else np.nan
                feats.append({'patient_id': r['patient_id'], 'fold': int(r['fold']),
                              'ef_pred': ef, 'edv_pred': edv, 'esv_pred': esv})
            except Exception as e:
                feats.append({'patient_id': r['patient_id'], 'fold': int(r['fold']), 'error': str(e)})

        out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feats).to_csv(out_csv, index=False)
        print('Wrote OOF features ->', out_csv)

if __name__ == '__main__':
    main()
