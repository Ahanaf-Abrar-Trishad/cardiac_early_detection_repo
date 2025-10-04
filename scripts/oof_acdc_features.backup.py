
#!/usr/bin/env python3
"""
Generate out-of-fold (OOF) ACDC features from predicted masks.
- Expects trained checkpoints per fold at: {logdir}/seg_acdc_fold{fold}_best.pt
- Reproduces GroupKFold splits on patient_id.
- Saves predicted masks (.npy) and computes EF/EDV/ESV from LV masks.
- Writes features CSV to --out-csv (default: meta/acdc_features_oof.csv)
"""
import argparse, os, json, numpy as np, pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

# ---- Minimal 3D UNet (copy of training arch) ----
import torch.nn as nn

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, feat=(16,32,64,128)):
        super().__init__()
        self.down1 = DoubleConv3D(in_ch, feat[0]); self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv3D(feat[0], feat[1]); self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv3D(feat[1], feat[2]); self.pool3 = nn.MaxPool3d(2)
        self.bott  = DoubleConv3D(feat[2], feat[3])
        self.up3   = nn.ConvTranspose3d(feat[3], feat[2], 2, stride=2); self.conv3 = DoubleConv3D(feat[3], feat[2])
        self.up2   = nn.ConvTranspose3d(feat[2], feat[1], 2, stride=2); self.conv2 = DoubleConv3D(feat[2], feat[1])
        self.up1   = nn.ConvTranspose3d(feat[1], feat[0], 2, stride=2); self.conv1 = DoubleConv3D(feat[1], feat[0])
        self.out   = nn.Conv3d(feat[0], out_ch, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        b  = self.bott(p3)
        u3 = self.up3(b); c3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3); c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2); c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

# ---- EF utilities ----
from utils.ef import ef_from_cavity

# ---- Dataset ----
from datasets.acdc_ed_es_3d import ACDC3DEDS

def build_folds(df, n_splits=5):
    # Reproduce patient-level GroupKFold
    groups = df['patient_id'].values
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(df))
    folds = []
    for f, (_, val_idx) in enumerate(gkf.split(idx, groups=groups)):
        folds.append(val_idx)
    return folds

@torch.no_grad()
def infer_fold(model, loader, multiclass=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval(); preds = []
    for batch in loader:
        x = batch['image'].to(device)
        lg = model(x)
        if multiclass:
            pr = torch.argmax(lg, dim=1)  # (B,Z,Y,X) in {0,1,2}
        else:
            pr = (torch.sigmoid(lg) >= 0.5).long().squeeze(1)
        preds.append(pr.cpu().numpy())
    return np.concatenate(preds, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='meta/master_metadata.csv')
    ap.add_argument('--phase', default='ED', choices=['ED','ES'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--feat3d', default='16,32,64,128')
    ap.add_argument('--logdir', default='logs')
    ap.add_argument('--out-csv', default='meta/acdc_features_oof.csv')
    ap.add_argument('--save-pred-root', default='preds_acdc_oof')
    ap.add_argument('--multiclass', action='store_true', help='use 3-class head (RV/MYO/LV)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat3d = tuple(int(x) for x in args.feat3d.split(',')) if args.feat3d else (16,32,64,128)
    out_ch = 3 if args.multiclass else 1

    # Dataframe for the requested phase
    df = pd.read_csv(args.meta)
    df = df[(df['dataset']=='acdc') & (df['phase']==args.phase)].reset_index(drop=True)

    folds = build_folds(df, n_splits=args.folds)
    save_root = Path(args.save_pred_root); save_root.mkdir(parents=True, exist_ok=True)

    # Run OOF inference per fold
    all_rows = []
    for f, val_idx in enumerate(folds):
        ckpt = Path(args.logdir) / f'seg_acdc_fold{f}_best.pt'
        if not ckpt.exists():
            print(f'[WARN] Missing checkpoint for fold {f}: {ckpt} — skipping this fold.')
            continue

        print(f'Fold {f}: loading {ckpt}')
        model = UNet3D(in_ch=1, out_ch=out_ch, feat=feat3d).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        ds = ACDC3DEDS(args.meta, split="", phase=args.phase, aug=None)
        va_set = Subset(ds, list(val_idx))
        loader = DataLoader(va_set, batch_size=1, shuffle=False, num_workers=2)

        preds = infer_fold(model, loader, multiclass=args.multiclass, device=device)
        # Save predictions and compute simple EF components if both phases will be supplied later
        for i, idx in enumerate(val_idx):
            row = df.iloc[idx]
            patient = row['patient_id']
            # Save predicted LV mask for EF (class 2 under our mapping)
            pr = preds[i]
            lv_pred = (pr==2).astype('uint8') if args.multiclass else (pr>0).astype('uint8')
            npy_path = save_root / f'{patient}_{args.phase}_fold{f}_lv.npy'
            np.save(npy_path, lv_pred)
            all_rows.append({'patient_id': patient, 'phase': args.phase, 'fold': f, 'pred_path': str(npy_path)})

    pred_index = pd.DataFrame(all_rows)
    pred_index.to_csv(Path(args.save_pred_root)/f'index_{args.phase}.csv', index=False)
    print('Indexed predictions ->', Path(args.save_pred_root)/f'index_{args.phase}.csv')

    # If both ED and ES have been run, compute EF (optional post-step)
    # Here we try to compute EF if both ED and ES files exist in the same folder.
    other_phase = 'ES' if args.phase=='ED' else 'ED'
    ed_idx_path = Path(args.save_pred_root)/f'index_ED.csv'
    es_idx_path = Path(args.save_pred_root)/f'index_ES.csv'
    if ed_idx_path.exists() and es_idx_path.exists():
        ed_idx = pd.read_csv(ed_idx_path); es_idx = pd.read_csv(es_idx_path)
        # Merge on patient + fold (OOF alignment)
        merged = pd.merge(ed_idx, es_idx, on=['patient_id','fold'], suffixes=('_ED','_ES'))
        feats = []
        meta_df = pd.read_csv(args.meta)
        for _, r in merged.iterrows():
            try:
                ed_mask = np.load(r['pred_path_ED'])
                es_mask = np.load(r['pred_path_ES'])
                # Find voxel size; prefer meta columns
                mrows = meta_df[(meta_df['patient_id']==r['patient_id'])]
                # Pick any row for spacing (ED/ES have same spacing)
                if len(mrows)>0:
                    mr = mrows.iloc[0]
                    px = float(mr.get('pixdim_x', np.nan)); py = float(mr.get('pixdim_y', np.nan))
                    th = float(mr.get('slice_thickness', np.nan))
                else:
                    px=py=th=np.nan
                if np.isnan(px) or np.isnan(py) or np.isnan(th):
                    # Unknown spacing — cannot compute volumes correctly
                    ef = np.nan; edv = np.nan; esv = np.nan
                else:
                    from utils.ef import volume_ml
                    edv = volume_ml(ed_mask>0, px, py, th)
                    esv = volume_ml(es_mask>0, px, py, th)
                    ef = (edv-esv)/edv if edv>0 else np.nan
                feats.append({'patient_id': r['patient_id'], 'fold': int(r['fold']), 'ef_pred': ef, 'edv_pred': edv, 'esv_pred': esv})
            except Exception as e:
                feats.append({'patient_id': r['patient_id'], 'fold': int(r['fold']), 'error': str(e)})
        feats = pd.DataFrame(feats)
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out_csv, index=False)
        print('Wrote OOF features ->', out_csv)

if __name__ == '__main__':
    main()
