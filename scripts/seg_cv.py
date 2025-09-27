#!/usr/bin/env python3
"""
Segmentation cross-validation (patient-level) for CAMUS (2D) and ACDC (3D).

- GroupKFold by patient_id (prevents leakage).
- CAMUS → 2D U-Net (binary).
- ACDC → 3D U-Net; --acdc-multiclass for RV/MYO/LV (3 classes).
- Logs per-fold metrics (Dice/IoU) and summary; saves example preds.
- Optional MLflow / W&B tracking.
- Flags:
  * --feat2d / --feat3d to choose UNet widths
  * --amp to enable mixed precision (torch.amp / cuda.amp via shim)
  * --grad-clip, --accum, --num-workers for stability/perf
"""
import argparse, json, random, math
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

# -------- AMP compat shim (new torch.amp API with fallback to cuda.amp) --------
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    def make_scaler(enabled: bool):
        # torch>=2.0
        return _GradScaler('cuda', enabled=enabled)
    def autocast_ctx(enabled: bool):
        return _autocast('cuda', enabled=enabled)
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    def make_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)
    def autocast_ctx(enabled: bool):
        return _autocast(enabled=enabled)

# Optional experiment tracking
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Local datasets
from datasets.camus_ed_es import CamusEDESDataset
from datasets.acdc_ed_es_3d import ACDC3DEDS

# Utils for saving examples
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Small utils ----------------
def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cosine_with_warmup_lambda(T, warm=5, min_ratio=0.1):
    def f(ep):
        if ep < warm:
            return (ep + 1) / max(1, warm)
        t = (ep - warm) / max(1, (T - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return f

def loader_fast_kwargs(num_workers: int):
    return dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2
    )

# ---------------- Models ----------------
class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feat=(32, 64, 128, 256)):
        super().__init__()
        self.down1 = DoubleConv2D(in_ch, feat[0]); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv2D(feat[0], feat[1]); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv2D(feat[1], feat[2]); self.pool3 = nn.MaxPool2d(2)
        self.bott  = DoubleConv2D(feat[2], feat[3])
        self.up3   = nn.ConvTranspose2d(feat[3], feat[2], 2, stride=2); self.conv3 = DoubleConv2D(feat[3], feat[2])
        self.up2   = nn.ConvTranspose2d(feat[2], feat[1], 2, stride=2); self.conv2 = DoubleConv2D(feat[2], feat[1])
        self.up1   = nn.ConvTranspose2d(feat[1], feat[0], 2, stride=2); self.conv1 = DoubleConv2D(feat[1], feat[0])
        self.out   = nn.Conv2d(feat[0], out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        b  = self.bott(p3)

        u3 = self.up3(b)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = F.interpolate(u3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(c3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(c1)

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    """
    Anisotropic + adaptive 3D U-Net:
    - Pools/upsamples mainly in-plane, skips pooling for dims <2 at runtime.
    - Aligns skip sizes via F.interpolate.
    """
    def __init__(self, in_ch=1, out_ch=1, feat=(16, 32, 64, 128)):
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
        k = [1, 2, 2]  # (D,H,W)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
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
            u3 = F.interpolate(u3, size=d3.shape[2:], mode='trilinear', align_corners=False)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(c3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='trilinear', align_corners=False)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode='trilinear', align_corners=False)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(c1)


# ---------------- Loss & Metrics ----------------
def to_one_hot(mask, classes):
    m = mask.squeeze(1).long()
    return torch.stack([(m == c).float() for c in classes], dim=1)

def dice_per_class(pred, target, eps=1e-6):
    dims = list(range(2, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    return (2 * inter / denom).mean(dim=0)  # (C,)

def iou_per_class(pred, target, eps=1e-6):
    dims = list(range(2, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter + eps
    return (inter / union).mean(dim=0)  # (C,)

def dice_coeff(pred, target, eps=1e-6):
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item() + eps
    return 2.0 * inter / denom

def iou_coeff(pred, target, eps=1e-6):
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter + eps
    return inter / union

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        red_dims = [1,2,3,4] if logits.dim()==5 else [1,2,3]
        num = 2 * (probs * targets).sum(dim=red_dims) + self.eps
        den = probs.sum(dim=red_dims) + targets.sum(dim=red_dims) + self.eps
        return 1 - (num / den).mean()


# ---------------- Helpers ----------------
def save_overlay_2d(img_t, m_true_t, m_pred_t, out_png):
    img = img_t.squeeze().cpu().numpy()
    mt  = m_true_t.squeeze().cpu().numpy()
    mp  = m_pred_t.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap='gray'); plt.imshow(mt, alpha=0.3); plt.imshow(mp, alpha=0.3)
    plt.axis('off'); plt.title(Path(out_png).stem)
    plt.savefig(out_png, bbox_inches='tight'); plt.close()

def save_nifti_pred(vol_pred, ref_path, out_path):
    ref = nib.load(ref_path)
    img = nib.Nifti1Image(vol_pred.astype(np.uint8), affine=ref.affine, header=ref.header)
    nib.save(img, out_path)

def maybe_init_tracking(args):
    run = None
    if args.mlflow and _HAS_MLFLOW:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"{args.dataset}-{args.phase}-cv")
        mlflow.log_params({
            "dataset": args.dataset, "phase": args.phase, "folds": args.folds,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed, "view": args.view, "feat2d": args.feat2d, "feat3d": args.feat3d, "amp": args.amp,
            "grad_clip": args.grad_clip, "accum": args.accum, "num_workers": args.num_workers
        })
    if args.wandb and _HAS_WANDB:
        run = wandb.init(
            project=args.wandb_project or "cardiac-seg",
            entity=(args.wandb_entity or None),
            name=f"{args.dataset}-{args.phase}-cv",
            config={
                "dataset": args.dataset, "phase": args.phase, "folds": args.folds,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "seed": args.seed, "view": args.view, "feat2d": args.feat2d, "feat3d": args.feat3d, "amp": args.amp,
                "grad_clip": args.grad_clip, "accum": args.accum, "num_workers": args.num_workers
            }
        )
    return run

def track_metric(step_name, metrics: dict, args, images=None):
    if args.mlflow and _HAS_MLFLOW:
        for k, v in metrics.items():
            try: mlflow.log_metric(k, float(v))
            except Exception: pass
        if images:
            for label, path in images.items():
                try: mlflow.log_artifact(str(path), artifact_path=label)
                except Exception: pass
    if args.wandb and _HAS_WANDB:
        try: wandb.log(metrics)
        except Exception: pass
        if images:
            for label, path in images.items():
                try: wandb.log({label: wandb.Image(str(path))})
                except Exception: pass

def end_tracking(args):
    if args.mlflow and _HAS_MLFLOW:
        try: mlflow.end_run()
        except Exception: pass
    if args.wandb and _HAS_WANDB:
        try: wandb.finish()
        except Exception: pass


# ---------------- Training Loops ----------------
def run_fold_seg_2d(tr_idxs, va_idxs, phase, logdir, args, fold_idx):
    ds = CamusEDESDataset(args.meta, split="", view=args.view, phase=phase, aug=None)
    tr_set = Subset(ds, list(tr_idxs))
    va_set = Subset(ds, list(va_idxs))
    k = loader_fast_kwargs(args.num_workers)
    loader_tr = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **k)
    loader_va = DataLoader(va_set, batch_size=args.batch_size, shuffle=False, **k)

    feat2d = tuple(int(x) for x in args.feat2d.split(",")) if args.feat2d else (32,64,128,256)
    model = UNet2D(in_ch=1, out_ch=1, feat=feat2d).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce   = nn.BCEWithLogitsLoss(); dice = DiceLoss()

    # AMP + Scheduler + Best checkpoint
    use_amp = bool(args.amp) and DEVICE == "cuda"
    scaler = make_scaler(use_amp)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup_lambda(args.epochs, warm=5, min_ratio=0.1))
    best_dice, best_path = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0; nseen = 0
        for step, batch in enumerate(loader_tr, 1):
            x = batch["image"].to(DEVICE)
            y = (batch["mask"] > 0).float().to(DEVICE)
            with autocast_ctx(use_amp):
                lg = model(x)
                loss = (0.5 * bce(lg, y) + 0.5 * dice(lg, y)) / max(1, args.accum)
            scaler.scale(loss).backward()
            if step % args.accum == 0:
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * x.size(0); nseen += x.size(0)

        # validation
        model.eval(); dice_v = 0.0; iou_v = 0.0; m = 0
        with torch.no_grad(), autocast_ctx(use_amp):
            for batch in loader_va:
                x = batch["image"].to(DEVICE)
                y = (batch["mask"] > 0).float().to(DEVICE)
                pr = (torch.sigmoid(model(x)) >= 0.5).float()
                for i in range(x.size(0)):
                    d = dice_coeff(pr[i], y[i]); j = iou_coeff(pr[i], y[i])
                    dice_v += d; iou_v += j; m += 1
        dice_v /= max(m, 1); iou_v /= max(m, 1)

        # save best
        if dice_v > best_dice:
            best_dice = dice_v
            best_path = logdir / f"seg_camus_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), best_path)

        sched.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"[Fold {fold_idx}] Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={(train_loss/max(1,nseen)):.4f} | dice={dice_v:.4f} iou={iou_v:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
            if best_path is not None:
                print(f"  ↳ best so far: {best_dice:.4f} @ {best_path.name}")
            track_metric("val", {"dice": dice_v, "iou": iou_v, "epoch": epoch}, args)

    # Use best checkpoint for the preview overlay if we have it
    if best_path is not None:
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state)

    # Save one overlay
    batch = next(iter(loader_va))
    x = batch["image"].to(DEVICE)
    y = (batch["mask"] > 0).float().to(DEVICE)
    with torch.no_grad(), autocast_ctx(use_amp):
        pr = (torch.sigmoid(model(x)) >= 0.5).float()
    out_png = logdir / f"seg_camus_fold{fold_idx}_example.png"
    save_overlay_2d(x[0].cpu(), y[0].cpu(), pr[0].cpu(), out_png)

    return float(dice_v), float(iou_v), str(out_png), None, (str(best_path) if best_path is not None else "")


def run_fold_seg_3d(tr_idxs, va_idxs, phase, logdir, args, fold_idx):
    base = ACDC3DEDS(args.meta, split="", phase=phase, aug=None)
    tr_set = Subset(base, list(tr_idxs))
    va_set = Subset(base, list(va_idxs))
    k = loader_fast_kwargs(args.num_workers)
    loader_tr = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **k)
    loader_va = DataLoader(va_set, batch_size=1, shuffle=False, **k)

    feat3d = tuple(int(x) for x in args.feat3d.split(",")) if args.feat3d else (16,32,64,128)
    out_ch = 3 if args.acdc_multiclass else 1
    model  = UNet3D(in_ch=1, out_ch=out_ch, feat=feat3d).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce     = nn.CrossEntropyLoss()
    bce    = nn.BCEWithLogitsLoss(); dice = DiceLoss()

    use_amp = bool(args.amp) and DEVICE == "cuda"
    scaler = make_scaler(use_amp)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup_lambda(args.epochs, warm=5, min_ratio=0.1))
    best_dice, best_path = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0; nseen = 0
        for step, batch in enumerate(loader_tr, 1):
            x   = batch["image"].to(DEVICE)
            msk = batch["mask"].to(DEVICE)
            with autocast_ctx(use_amp):
                lg = model(x)
                if args.acdc_multiclass:
                    # map labels {1,2,3} -> {0,1,2}
                    y_idx = torch.where(msk==3, torch.tensor(2, device=msk.device),
                            torch.where(msk==2, torch.tensor(1, device=msk.device),
                            torch.where(msk==1, torch.tensor(0, device=msk.device), torch.tensor(0, device=msk.device)))).long()
                    loss = ce(lg, y_idx) / max(1, args.accum)
                else:
                    y = (msk > 0).float()
                    loss = (0.5 * bce(lg, y) + 0.5 * dice(lg, y)) / max(1, args.accum)
            scaler.scale(loss).backward()
            if step % args.accum == 0:
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * x.size(0); nseen += x.size(0)

        # validation
        model.eval(); m = 0
        dice_v = 0.0; iou_v = 0.0
        dice_classes = None; iou_classes = None
        with torch.no_grad(), autocast_ctx(use_amp):
            for batch in loader_va:
                x   = batch["image"].to(DEVICE)
                msk = batch["mask"].to(DEVICE)
                lg  = model(x)
                if args.acdc_multiclass:
                    pr_idx = torch.argmax(lg, dim=1, keepdim=True)  # (B,1,Z,Y,X) in {0,1,2}
                    classes = [0,1,2]
                    pr_oh = to_one_hot(pr_idx, classes)
                    # target 1..3 -> 0..2
                    t_idx = torch.where(msk==3, torch.tensor(2, device=msk.device),
                            torch.where(msk==2, torch.tensor(1, device=msk.device),
                            torch.where(msk==1, torch.tensor(0, device=msk.device), torch.tensor(0, device=msk.device)))).unsqueeze(1)
                    t_oh = to_one_hot(t_idx, classes)
                    dpc = dice_per_class(pr_oh, t_oh)
                    ipc = iou_per_class(pr_oh, t_oh)
                    dice_classes = dpc if dice_classes is None else dice_classes + dpc
                    iou_classes  = ipc if iou_classes  is None else iou_classes  + ipc
                    m += 1
                else:
                    y  = (msk > 0).float()
                    pr = (torch.sigmoid(lg) >= 0.5).float()
                    d = dice_coeff(pr[0], y[0]); j = iou_coeff(pr[0], y[0])
                    dice_v += d; iou_v += j; m += 1

        if args.acdc_multiclass and m > 0:
            dice_classes = dice_classes / m  # (3,)
            iou_classes  = iou_classes  / m  # (3,)
            dice_v = float(dice_classes.mean().item())
            iou_v  = float(iou_classes.mean().item())
        else:
            dice_v /= max(m, 1); iou_v /= max(m, 1)

        # save best
        if dice_v > best_dice:
            best_dice = dice_v
            best_path = logdir / f"seg_acdc_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), best_path)

        sched.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            extra = ""
            if args.acdc_multiclass and m > 0:
                drv, dmyo, dlv = [float(x) for x in dice_classes.tolist()]
                irv, imy, ilv = [float(x) for x in iou_classes.tolist()]
                extra = f" | RV Dice/IoU {drv:.3f}/{irv:.3f} MYO {dmyo:.3f}/{imy:.3f} LV {dlv:.3f}/{ilv:.3f}"
            print(f"[Fold {fold_idx}] Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={(train_loss/max(1,nseen)):.4f} | dice={dice_v:.4f} iou={iou_v:.4f}{extra} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
            if best_path is not None:
                print(f"  ↳ best so far: {best_dice:.4f} @ {best_path.name}")
            track_metric("val", {"dice": dice_v, "iou": iou_v, "epoch": epoch}, args)

    # Use best checkpoint for the preview NIfTI if we have it
    if best_path is not None:
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state)

    # Save a predicted NIfTI for the first val sample (aligned to reference)
    out_pred = logdir / f"seg_acdc_fold{fold_idx}_example_pred.nii.gz"
    va_first = next(iter(loader_va))
    x = va_first["image"].to(DEVICE)
    with torch.no_grad(), autocast_ctx(use_amp):
        lg = model(x)
        if args.acdc_multiclass:
            pred = torch.argmax(lg, dim=1, keepdim=True).cpu().numpy()[0,0]
        else:
            pred = (torch.sigmoid(lg) >= 0.5).float().cpu().numpy()[0,0]
    ref_path = json.loads(base.df.iloc[int(va_idxs[0])]["paths"])[f"nii_image_{phase}"]
    save_nifti_pred(pred, ref_path, out_pred)

    # return per-class vectors so caller can log them
    dpc = dice_classes.cpu().numpy().tolist() if (args.acdc_multiclass and dice_classes is not None) else None
    ipc = iou_classes.cpu().numpy().tolist()  if (args.acdc_multiclass and iou_classes  is not None) else None
    return float(dice_v), float(iou_v), str(out_pred), (dpc, ipc), (str(best_path) if best_path is not None else "")


# ---------------- Main ----------------
def main():
    def _comma_list(s): return s if not s else s.replace(" ", "")

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--dataset", choices=["camus", "acdc"], required=True)
    ap.add_argument("--acdc-multiclass", action="store_true", help="3-class (RV/MYO/LV) segmentation for ACDC")
    ap.add_argument("--view", default="4CH", choices=["2CH", "4CH"], help="CAMUS only")
    ap.add_argument("--phase", default="ED", choices=["ED", "ES"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--feat2d", type=_comma_list, default="", help="e.g. '16,32,64,128' (CAMUS)")
    ap.add_argument("--feat3d", type=_comma_list, default="", help="e.g. '16,32,64,128' (ACDC)")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip-norm; set 0 to disable")
    ap.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    # tracking
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-uri", default="", dest="mlflow_uri")
    ap.add_argument("--mlflow-experiment", default="seg-cv", dest="mlflow_experiment")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="cardiac-seg", dest="wandb_project")
    ap.add_argument("--wandb-entity", default="", dest="wandb_entity")
    args = ap.parse_args()

    # Reproducibility + perf niceties
    set_all_seeds(args.seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # TF32 on Ada/Ampere
    except Exception:
        pass

    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)

    # Load meta
    df = pd.read_csv(args.meta)

    # Select dataset rows + build mapping study_id -> dataset index
    if args.dataset == "camus":
        g = df[df["dataset"] == "camus"].reset_index(drop=True)

        def has_mask(row):
            try:
                p = json.loads(row["paths"])
                return bool(p.get(f"mask_{args.phase}", "")) or bool(p.get(f"png_mask_{args.phase}","")) or bool(p.get(f"nii_mask_{args.phase}",""))
            except Exception:
                return False

        g = g[(g["view"] == args.view) & (g.apply(has_mask, axis=1))].reset_index(drop=True)
        groups = g["patient_id"].values

        ds = CamusEDESDataset(args.meta, split="", view=args.view, phase=args.phase, aug=None)
        sid2idx = {row["study_id"]: i for i, row in ds.df.reset_index().iterrows()}
        idxs = [sid2idx[sid] for sid in g["study_id"].values if sid in sid2idx]
        g = g.iloc[:len(idxs)].copy()

    else:  # ACDC
        g = df[df["dataset"] == "acdc"].reset_index(drop=True)

        def has_mask(row):
            try:
                p = json.loads(row["paths"])
                return bool(p.get(f"nii_mask_{args.phase}", ""))
            except Exception:
                return False

        g = g[g.apply(has_mask, axis=1)].reset_index(drop=True)
        groups = g["patient_id"].values

        base = ACDC3DEDS(args.meta, split="", phase=args.phase, aug=None)
        sid2idx = {row["study_id"]: i for i, row in base.df.reset_index().iterrows()}
        idxs = [sid2idx[sid] for sid in g["study_id"].values if sid in sid2idx]
        g = g.iloc[:len(idxs)].copy()

    # GroupKFold across patients (train/val split fixed)
    gkf = GroupKFold(n_splits=args.folds)
    metrics = []
    _ = maybe_init_tracking(args)

    for fold, (tr, va) in enumerate(gkf.split(g, groups=groups), start=1):
        print(f"=== {args.dataset.upper()} Segmentation | Fold {fold}/{args.folds} ===")
        tr_idxs = np.array(idxs)[tr]
        va_idxs = np.array(idxs)[va]

        if args.dataset == "camus":
            dice_v, iou_v, artifact, _, best_ckpt = run_fold_seg_2d(tr_idxs, va_idxs, args.phase, logdir, args, fold)
            dpc = ipc = None
        else:
            dice_v, iou_v, artifact, tpl, best_ckpt = run_fold_seg_3d(tr_idxs, va_idxs, args.phase, logdir, args, fold)
            dpc, ipc = tpl if tpl is not None else (None, None)

        print(f"[Fold {fold}] Dice={dice_v:.4f} IoU={iou_v:.4f}")
        row = {"fold": fold, "Dice": dice_v, "IoU": iou_v, "artifact": artifact}
        if best_ckpt:
            row["best_ckpt"] = best_ckpt
        if args.dataset == "acdc" and args.acdc_multiclass and dpc is not None and ipc is not None:
            # order: [RV, MYO, LV]
            row.update({
                "Dice_RV": float(dpc[0]), "Dice_MYO": float(dpc[1]), "Dice_LV": float(dpc[2]),
                "IoU_RV":  float(ipc[0]), "IoU_MYO":  float(ipc[1]), "IoU_LV":  float(ipc[2]),
            })
        metrics.append(row)

        track_metric("fold_end",
                     {"fold_dice": dice_v, "fold_iou": iou_v, "fold": fold},
                     args,
                     images={f"{args.dataset}_fold{fold}": artifact})

    # Summary
    d = np.array([m["Dice"] for m in metrics], dtype=float)
    j = np.array([m["IoU"]  for m in metrics if not np.isnan(m["IoU"])], dtype=float) if any(not np.isnan(m["IoU"]) for m in metrics) else np.array([])
    summary = {
        "Dice_mean": float(np.nanmean(d)), "Dice_std": float(np.nanstd(d)),
        "IoU_mean":  float(np.mean(j)) if j.size>0 else float('nan'),
        "IoU_std":   float(np.std(j))  if j.size>0 else float('nan'),
        "folds": len(metrics), "dataset": args.dataset, "phase": args.phase, "view": args.view,
        "feat2d": args.feat2d, "feat3d": args.feat3d, "amp": args.amp,
        "grad_clip": args.grad_clip, "accum": args.accum, "num_workers": args.num_workers
    }
    if args.dataset == "acdc" and args.acdc_multiclass:
        pdf = pd.DataFrame(metrics)
        for col in ["Dice_RV", "Dice_MYO", "Dice_LV", "IoU_RV", "IoU_MYO", "IoU_LV"]:
            if col in pdf.columns:
                summary[col + "_mean"] = float(pdf[col].mean())
                summary[col + "_std"]  = float(pdf[col].std())

    print("\n=== CV Summary ===")
    print(f"Dice mean±std: {summary['Dice_mean']:.4f} ± {summary['Dice_std']:.4f}")
    print(f"IoU  mean±std: {summary['IoU_mean']:.4f} ± {summary['IoU_std']:.4f}")

    # Save artifacts
    logdir = Path(args.logdir)
    pdf = pd.DataFrame(metrics).sort_values("fold")
    pdf.to_csv(logdir / f"cv_seg_{args.dataset}_metrics.csv", index=False)
    if args.dataset == "acdc" and args.acdc_multiclass:
        cols = [c for c in ["fold","Dice_RV","Dice_MYO","Dice_LV","IoU_RV","IoU_MYO","IoU_LV"] if c in pdf.columns]
        if cols:
            pdf[["fold"]+cols[1:]].to_csv(logdir / "cv_seg_acdc_multiclass_perclass.csv", index=False)
    with open(logdir / f"cv_seg_{args.dataset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    end_tracking(args)


if __name__ == "__main__":
    main()
