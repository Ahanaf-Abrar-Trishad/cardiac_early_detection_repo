#!/usr/bin/env python3
"""
Segmentation cross-validation (patient-level) for CAMUS (2D) and ACDC (3D).

- GroupKFold by patient_id (prevents leakage).
- CAMUS → 2D U-Net (binary).
- ACDC → 3D U-Net.
- Logs per-fold metrics (Dice/IoU) and summary; saves example preds.
- Optional MLflow / W&B tracking.

New in this update:
- Per-epoch per-class Dice/IoU CSV for ACDC (--perclass-csv).
- Validation overlay previews (--save-val-previews, --preview-batches, --preview-slice).
- Label mapping sanity checks for ACDC (prints unique GT ids, assert on bad ids).
- Class-weighted CE + multiclass Dice (--class-weights / --ce-weight / --dice-weight).
- 3D augmentations for ACDC (--aug3d + intensity/flip/rotation knobs).
- **4-class ACDC option (BG,RV,MYO,LV) via --acdc-with-bg**; metrics/CSV still reported over RV/MYO/LV only.
- **Advanced models**: UNETR (Transformer), U-Net+CRAM attention
- **New metrics**: Accuracy, F1-macro, F1-weighted for segmentation
"""
import argparse, json, random, math, time, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

# Import advanced models
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from models.unetr import create_unetr
    _HAS_UNETR = True
except Exception:
    _HAS_UNETR = False

try:
    from models.cram import UNet3DCRAM
    _HAS_CRAM = True
except Exception:
    _HAS_CRAM = False

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

        self.out   = nn.Conv3d(feat[0], out_ch, kernel_size=1)

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

def compute_classification_metrics(pred_mask, true_mask, classes):
    """
    Compute pixel-wise accuracy and F1 for segmentation.
    
    Args:
        pred_mask: predicted labels [N, D, H, W] or [N, H, W]
        true_mask: ground truth labels [N, D, H, W] or [N, H, W]
        classes: list of class indices (e.g., [0,1,2,3])
    
    Returns:
        dict with 'accuracy', 'f1_macro', 'f1_weighted'
    """
    pred_flat = pred_mask.cpu().numpy().flatten()
    true_flat = true_mask.cpu().numpy().flatten()
    
    acc = accuracy_score(true_flat, pred_flat)
    
    # F1 scores
    f1_macro = f1_score(true_flat, pred_flat, average='macro', labels=classes, zero_division=0)
    f1_weighted = f1_score(true_flat, pred_flat, average='weighted', labels=classes, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        red_dims = [1,2,3,4] if logits.dim()==5 else [1,2,3]
        num = 2 * (probs * targets).sum(dim=red_dims) + self.eps
        den = probs.sum(dim=red_dims) + targets.sum(dim=red_dims) + self.eps
        return 1 - (num / den).mean()

# ---------- NEW: class weights, multiclass Dice, and 3D augmentations ----------
def _parse_class_weights(arg_str, C):
    """Parse class-weights string into a tensor of length C, or return 'auto'/None."""
    if not arg_str:
        return None
    if arg_str.strip().lower() == "auto":
        return "auto"
    vals = [float(x) for x in arg_str.replace(" ", "").split(",") if x != ""]
    assert len(vals) == C, f"--class-weights needs {C} numbers (got {len(vals)}) or 'auto'"
    return torch.tensor(vals, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def _auto_class_weights_from_loader(loader, classes, max_batches=30):
    """
    Estimate class weights from label voxel frequencies.
    classes: sequence of target class indices as used by the loss (e.g., [0,1,2] or [0,1,2,3]).
    """
    hist = torch.zeros(len(classes), dtype=torch.float64)
    seen = 0
    for b, batch in enumerate(loader):
        msk = batch["mask"].long()  # original labels: BG=0,RV=1,MYO=2,LV=3 in ACDC
        for i, c in enumerate(classes):
            hist[i] += (msk == c).sum().double().item()
        seen += 1
        if seen >= max_batches: break
    hist = torch.clamp(hist, min=1.0)
    inv = 1.0 / hist
    inv = inv / inv.mean()
    return inv.to(torch.float32).to(DEVICE)

def soft_dice_loss_multiclass(logits, target_idx, class_weights=None, eps=1e-6):
    """
    logits: [N,C,D,H,W], target_idx: [N,1,D,H,W] in {0..C-1}
    returns: 1 - weighted mean Dice across classes
    """
    N, C = logits.shape[:2]
    probs = torch.softmax(logits, dim=1)                      # [N,C,D,H,W]
    tgt_oh = torch.nn.functional.one_hot(target_idx.squeeze(1).long(), num_classes=C)
    tgt_oh = tgt_oh.permute(0,4,1,2,3).float()                # [N,C,D,H,W]

    dims = (0,2,3,4)
    inter = (probs * tgt_oh).sum(dim=dims)                    # [C]
    denom = probs.sum(dim=dims) + tgt_oh.sum(dim=dims) + eps  # [C]
    dice_c = (2 * inter / denom)                              # [C]

    if class_weights is not None and not isinstance(class_weights, str):
        w = class_weights / class_weights.sum()
        loss = 1.0 - (w * dice_c).sum()
    else:
        loss = 1.0 - dice_c.mean()
    return loss

def _rotate_3d_z(x, angle_deg, mode="bilinear"):
    """Rotate the 3D volume around the Z axis (in-plane rotation). x: [N,1,D,H,W]"""
    angle = math.radians(float(angle_deg))
    cos, sin = math.cos(angle), math.sin(angle)
    theta = x.new_zeros((x.size(0), 3, 4))
    theta[:,0,0] =  cos; theta[:,0,1] = -sin
    theta[:,1,0] =  sin; theta[:,1,1] =  cos
    theta[:,2,2] =  1.0
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)

def augment3d(x, msk, *,
              p_flip=0.5, p_rot=0.5, rot_deg=12,
              p_gamma=0.5, gamma_min=0.9, gamma_max=1.15,
              p_bright=0.5, bright_min=0.9, bright_max=1.1):
    """
    Simple, fast 3D augs for ACDC:
      - random flips in H/W
      - in-plane rotation around Z
      - gamma + brightness jitter (image only)

    x:   [N,1,D,H,W] float
    msk: [N,D,H,W]   or [N,1,D,H,W] with int labels
    """
    import random, torch
    def _rand(p): return random.random() < p

    # ---- normalize mask shape to 5D for transforms ----
    msk_dtype = msk.dtype
    msk_was_4d = (msk.dim() == 4)
    if msk_was_4d:
        msk = msk.unsqueeze(1)  # -> [N,1,D,H,W]

    # flips (H/W)
    if _rand(p_flip):
        x   = torch.flip(x,  dims=[-2])     # H
        msk = torch.flip(msk, dims=[-2])
    if _rand(p_flip):
        x   = torch.flip(x,  dims=[-1])     # W
        msk = torch.flip(msk, dims=[-1])

    # rotation (shared for image & mask)
    if _rand(p_rot) and rot_deg > 0:
        ang = random.uniform(-rot_deg, rot_deg)
        x   = _rotate_3d_z(x, ang,  mode="bilinear")
        msk = _rotate_3d_z(msk.float(), ang, mode="nearest").round().clamp(min=0, max=3).to(msk_dtype)

    # intensity jitter (image only)
    if _rand(p_bright):
        s = random.uniform(bright_min, bright_max)
        x = x * s
    if _rand(p_gamma):
        g = random.uniform(gamma_min, gamma_max)
        x = torch.clamp(x, 0, 1) ** g

    if msk_was_4d:
        msk = msk.squeeze(1)

    return x, msk
# -------------------------------------------------------------------------------


# ---------------- Helpers ----------------
def collate_3d_batch(batch):
    """
    Custom collate for 3D volumes with different shapes.
    Pads all volumes to the max D/H/W in the batch.
    """
    # Find max dimensions
    max_d = max(b["image"].shape[1] for b in batch)
    max_h = max(b["image"].shape[2] for b in batch)
    max_w = max(b["image"].shape[3] for b in batch)
    
    images = []
    masks = []
    labels = []
    efs = []
    metas = []
    
    for b in batch:
        img = b["image"]  # [1, D, H, W]
        msk = b["mask"]   # [1, D, H, W] or [D, H, W]
        
        # Pad image
        d, h, w = img.shape[1], img.shape[2], img.shape[3]
        pad = (0, max_w - w, 0, max_h - h, 0, max_d - d)
        img_padded = torch.nn.functional.pad(img, pad, mode='constant', value=0)
        images.append(img_padded)
        
        # Pad mask (ensure it's 4D [1, D, H, W])
        if msk.dim() == 3:
            msk = msk.unsqueeze(0)  # [D, H, W] -> [1, D, H, W]
        msk_padded = torch.nn.functional.pad(msk, pad, mode='constant', value=0)
        masks.append(msk_padded)
        
        # Collect other fields
        labels.append(b.get("label", "NA"))
        efs.append(b.get("ef", np.nan))
        metas.append(b.get("meta", {}))
    
    return {
        "image": torch.stack(images, dim=0),  # [N, 1, D, H, W]
        "mask": torch.stack(masks, dim=0),     # [N, 1, D, H, W]
        "label": labels,
        "ef": efs,
        "meta": metas
    }

def save_overlay_2d(img_t, m_true_t, m_pred_t, out_png):
    img = img_t.squeeze().cpu().numpy()
    mt  = m_true_t.squeeze().cpu().numpy()
    mp  = m_pred_t.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap='gray'); plt.imshow(mt, alpha=0.3); plt.imshow(mp, alpha=0.3)
    plt.axis('off'); plt.title(Path(out_png).stem)
    plt.savefig(out_png, bbox_inches='tight'); plt.close()

def save_overlay_3d_mid(x, y_idx, pr_idx, out_png, z=None, margin=10):
    """
    Robust overlay:
      - picks mid-slice (or --preview-slice),
      - crops around the union of GT/PRED masks with a small margin,
      - smooth image display so it isn't blocky.
    Handles both 3-class (0..2) and 4-class (0..3 with BG=0) inputs.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    x = x.detach().cpu().numpy()[0, 0]      # [D,H,W]
    y = y_idx.detach().cpu().numpy()[0, 0]  # [D,H,W]
    p = pr_idx.detach().cpu().numpy()[0, 0] # [D,H,W]

    D, H, W = x.shape
    z = (D // 2) if z is None else int(np.clip(z, 0, D-1))

    # which classes to draw?
    maxlab = max(int(y.max()), int(p.max()))
    draw_classes = [1,2,3] if maxlab >= 3 else [0,1,2]

    # crop ROI from union of GT/PRED masks on this slice (non-zero labels)
    mask_union = ((y[z] > 0) | (p[z] > 0))
    if mask_union.any():
        ys, xs = np.where(mask_union)
        y0, y1 = max(0, ys.min()-margin), min(H, ys.max()+margin+1)
        x0, x1 = max(0, xs.min()-margin), min(W, xs.max()+margin+1)
    else:
        cy, cx = H//2, W//2
        y0, y1 = max(0, cy-80), min(H, cy+80)
        x0, x1 = max(0, cx-80), min(W, cx+80)

    sl_img = x[z, y0:y1, x0:x1]
    sl_gt  = y[z, y0:y1, x0:x1]
    sl_pr  = p[z, y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(5,5), dpi=150)
    vmin, vmax = np.percentile(sl_img, [1, 99])
    ax.imshow(sl_img, cmap="gray", aspect="auto", interpolation="bilinear", vmin=vmin, vmax=vmax)
    for cls in draw_classes: ax.contour((sl_gt==cls), levels=[0.5], linewidths=1.2)
    for cls in draw_classes: ax.contour((sl_pr==cls), levels=[0.5], linestyles="--", linewidths=1.2)
    ax.set_axis_off(); ax.set_title(f"z={z} (solid=GT, dashed=PRED)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

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
            try:
                mlflow.log_metric(k, v)
            except Exception:
                pass
        if images:
            for label, path in images.items():
                try:
                    mlflow.log_artifact(str(path), artifact_path=label)
                except Exception:
                    pass
    if args.wandb and _HAS_WANDB:
        try:
            wandb.log(metrics)
        except Exception:
            pass
        if images:
            for label, path in images.items():
                try:
                    wandb.log({label: wandb.Image(str(path))})
                except Exception:
                    pass

def end_tracking(args):
    if args.mlflow and _HAS_MLFLOW:
        try:
            mlflow.end_run()
        except Exception:
            pass
    if args.wandb and _HAS_WANDB:
        try:
            wandb.finish()
        except Exception:
            pass

# ---- CSV logging for per-epoch per-class metrics (ACDC multiclass) ----
def append_epoch_csv(csv_path, fold, epoch, val_loss, dice_list, iou_list, mean_dice_no_bg):
    header = ["timestamp","fold","epoch","val_loss",
              "dice_rv","dice_myo","dice_lv",
              "iou_rv","iou_myo","iou_lv",
              "mean_dice_no_bg"]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = [int(time.time()), fold, epoch,
           (float(val_loss) if val_loss is not None else np.nan)] + \
          [float(x) for x in dice_list] + [float(x) for x in iou_list] + [float(mean_dice_no_bg)]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

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
        val_accs = []
        val_f1s_macro = []
        val_f1s_weighted = []
        with torch.no_grad(), autocast_ctx(use_amp):
            for batch in loader_va:
                x = batch["image"].to(DEVICE)
                y = (batch["mask"] > 0).float().to(DEVICE)
                pr = (torch.sigmoid(model(x)) >= 0.5).float()
                for i in range(x.size(0)):
                    d = dice_coeff(pr[i], y[i]); j = iou_coeff(pr[i], y[i])
                    dice_v += d; iou_v += j; m += 1

                # --- NEW: accuracy and F1 score ---
                cls_metrics = compute_classification_metrics(
                    pred_mask=(pr > 0.5).long(),
                    true_mask=y,
                    classes=[0, 1]  # binary for CAMUS
                )
                val_accs.append(cls_metrics['accuracy'])
                val_f1s_macro.append(cls_metrics['f1_macro'])
                val_f1s_weighted.append(cls_metrics['f1_weighted'])

        dice_v /= max(m, 1); iou_v /= max(m, 1)

        # save best
        if dice_v > best_dice:
            best_dice = dice_v
            best_path = logdir / f"seg_camus_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), best_path)

        # --- NEW: log accuracy and F1 ---
        mean_acc = np.mean(val_accs)
        mean_f1_macro = np.mean(val_f1s_macro)
        mean_f1_weighted = np.mean(val_f1s_weighted)

        sched.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"[Fold {fold_idx}] Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={(train_loss/max(1,nseen)):.4f} | dice={dice_v:.4f} iou={iou_v:.4f} | "
                  f"acc={mean_acc:.4f} f1={mean_f1_macro:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
            if best_path is not None:
                print(f"  ↳ best so far: {best_dice:.4f} @ {best_path.name}")
            track_metric("val", {"dice": dice_v, "iou": iou_v, "accuracy": mean_acc,
                                "f1_macro": mean_f1_macro, "f1_weighted": mean_f1_weighted,
                                "epoch": epoch}, args)

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

    return float(dice_v), float(iou_v), str(out_png), mean_acc, mean_f1_macro, mean_f1_weighted, (str(best_path) if best_path is not None else "")


def run_fold_seg_3d(tr_idxs, va_idxs, phase, logdir, args, fold_idx):
    base = ACDC3DEDS(args.meta, split="", phase=phase, aug=None)
    tr_set = Subset(base, list(tr_idxs))
    va_set = Subset(base, list(va_idxs))
    k = loader_fast_kwargs(args.num_workers)
    loader_tr = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_3d_batch, **k)
    loader_va = DataLoader(va_set, batch_size=1, shuffle=False, collate_fn=collate_3d_batch, **k)

    feat3d = tuple(int(x) for x in args.feat3d.split(",")) if args.feat3d else (16,32,64,128)
    # output channels (3 or 4)
    C = 4 if (args.acdc_multiclass and args.acdc_with_bg) else (3 if args.acdc_multiclass else 1)
    out_ch = C
    
    # Create model based on --model argument
    if args.model == "unet3d_cram" and _HAS_CRAM:
        model = UNet3DCRAM(in_ch=1, out_ch=out_ch, feat=feat3d).to(DEVICE)
        print(f"[info] Using U-Net 3D with CRAM attention")
    elif args.model == "unetr" and _HAS_UNETR:
        # UNETR needs image size - use typical ACDC dimensions (small depth, large spatial)
        model = create_unetr(in_channels=1, out_channels=out_ch, img_size=(10, 280, 280))
        model = model.to(DEVICE)
        print(f"[info] Using UNETR (Transformer-based)")
    else:
        if args.model != "unet":
            print(f"[warning] Model '{args.model}' not available, using standard U-Net 3D")
        model = UNet3D(in_ch=1, out_ch=out_ch, feat=feat3d).to(DEVICE)
        print(f"[info] Using standard U-Net 3D")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce    = nn.BCEWithLogitsLoss(); dice = DiceLoss()

    # ---- class weights ----
    if args.acdc_multiclass:
        classes_for_loss = list(range(C))  # [0,1,2] or [0,1,2,3]
        cw = _parse_class_weights(getattr(args, "class_weights", ""), C)
        if isinstance(cw, str) and cw == "auto":
            print("[info] computing auto class weights from training masks...")
            cw = _auto_class_weights_from_loader(loader_tr, classes_for_loss)
            print(f"[info] auto class weights (len={C}): {cw.tolist()}")
        ce = nn.CrossEntropyLoss(weight=cw)
    else:
        cw = None
        ce = None

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
            msk = batch["mask"].to(DEVICE).squeeze(1)  # [N, 1, D, H, W] -> [N, D, H, W]

            # ---- 3D augmentations ----
            if args.acdc_multiclass and args.aug3d:
                x, msk = augment3d(
                    x, msk,
                    p_flip=args.p_flip, p_rot=args.p_rot, rot_deg=args.rot_deg,
                    p_gamma=args.p_gamma, gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                    p_bright=args.p_bright, bright_min=args.bright_min, bright_max=args.bright_max
                )

            with autocast_ctx(use_amp):
                lg = model(x)
                if args.acdc_multiclass:
                    if args.acdc_with_bg:
                        y_idx = msk.long().clamp(min=0, max=3)   # BG=0,RV=1,MYO=2,LV=3
                    else:
                        # map labels {1,2,3} -> {0,1,2}; treat BG as 0 (RV)
                        y_idx = torch.where(msk==3, torch.tensor(2, device=msk.device),
                                torch.where(msk==2, torch.tensor(1, device=msk.device),
                                torch.where(msk==1, torch.tensor(0, device=msk.device), torch.tensor(0, device=msk.device)))).long()
                    ce_loss   = ce(lg, y_idx)
                    dice_loss = soft_dice_loss_multiclass(lg, y_idx.unsqueeze(1), class_weights=cw)
                    loss = (args.ce_weight * ce_loss + args.dice_weight * dice_loss) / max(1, args.accum)
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
        model.eval()
        m = 0
        dice_v = 0.0; iou_v = 0.0
        dice_classes = None; iou_classes = None
        val_loss_sum = 0.0; val_count = 0

        val_accs = []
        val_f1s_macro = []
        val_f1s_weighted = []

        with torch.no_grad(), autocast_ctx(use_amp):
            for batch_idx, batch in enumerate(loader_va):
                x   = batch["image"].to(DEVICE)
                msk = batch["mask"].to(DEVICE).squeeze(1)  # [N, 1, D, H, W] -> [N, D, H, W]

                # ---- one-time sanity on GT labels ----
                if epoch == 1 and batch_idx == 0:
                    uniq = torch.unique(msk).tolist()
                    print(f"[SANITY] unique GT labels in first val batch: {uniq} (expected subset of [0,1,2,3])")
                    assert all(int(v) in (0,1,2,3) for v in uniq), "Unexpected label id(s) in mask!"

                lg  = model(x)

                if args.acdc_multiclass:
                    pr_idx = torch.argmax(lg, dim=1, keepdim=True)  # (B,1,D,H,W)

                    # build target index tensor
                    if args.acdc_with_bg:
                        t_idx = msk.long().clamp(0,3).unsqueeze(1)  # includes BG
                        score_classes = [1,2,3]                     # RV,MYO,LV
                        val_ce_target = t_idx.squeeze(1)
                    else:
                        # 3-class model targets 0,1,2 (RV,MYO,LV)
                        t_idx = torch.where(msk==3, torch.tensor(2, device=msk.device),
                                torch.where(msk==2, torch.tensor(1, device=msk.device),
                                torch.where(msk==1, torch.tensor(0, device=msk.device), torch.tensor(0, device=msk.device)))).unsqueeze(1)
                        score_classes = [0,1,2]
                        val_ce_target = t_idx.squeeze(1)

                    # optional previews
                    if args.save_val_previews and batch_idx < args.preview_batches:
                        out_png = Path(args.logdir)/f"runs/acdc_val_previews/fold{fold_idx}/epoch{epoch}/b{batch_idx}.png"
                        save_overlay_3d_mid(x, t_idx, pr_idx, out_png, z=(None if args.preview_slice<0 else args.preview_slice))

                    # metrics (RV/MYO/LV only)
                    pr_oh = to_one_hot(pr_idx, score_classes)
                    t_oh  = to_one_hot(t_idx, score_classes)
                    dpc   = dice_per_class(pr_oh, t_oh)   # (3,)
                    ipc   = iou_per_class(pr_oh, t_oh)    # (3,)
                    dice_classes = dpc if dice_classes is None else dice_classes + dpc
                    iou_classes  = ipc if iou_classes  is None else iou_classes  + ipc

                    # val CE loss for CSV
                    val_loss_sum += ce(lg, val_ce_target.long()).item()
                    val_count += 1
                    m += 1

                    # --- NEW: accuracy and F1 score ---
                    if args.acdc_with_bg:
                        class_list = [0, 1, 2, 3]  # BG, RV, MYO, LV
                    else:
                        class_list = [0, 1, 2]  # RV, MYO, LV (remapped)
                    
                    cls_metrics = compute_classification_metrics(
                        pred_mask=pr_idx.squeeze(1),
                        true_mask=t_idx.squeeze(1),
                        classes=class_list
                    )
                    val_accs.append(cls_metrics['accuracy'])
                    val_f1s_macro.append(cls_metrics['f1_macro'])
                    val_f1s_weighted.append(cls_metrics['f1_weighted'])

                else:
                    y  = (msk > 0).float()
                    pr = (torch.sigmoid(lg) >= 0.5).float()
                    d = dice_coeff(pr[0], y[0]); j = iou_coeff(pr[0], y[0])
                    dice_v += d; iou_v += j; m += 1

        if args.acdc_multiclass and m > 0:
            dice_classes = dice_classes / m  # (3,)
            iou_classes  = iou_classes  / m  # (3,)
            drv, dmyo, dlv = [float(x) for x in dice_classes.tolist()]
            irv, imy, ilv = [float(x) for x in iou_classes.tolist()]
            dice_v = float(dice_classes.mean().item())
            iou_v  = float(iou_classes.mean().item())
            # ---- write per-epoch CSV ----
            mean_no_bg = dice_v  # already mean of (RV,MYO,LV)
            append_epoch_csv(args.perclass_csv, fold=fold_idx, epoch=epoch,
                             val_loss=(val_loss_sum/ max(1,val_count)),
                             dice_list=[drv, dmyo, dlv],
                             iou_list=[irv, imy, ilv],
                             mean_dice_no_bg=mean_no_bg)
        else:
            dice_v /= max(m, 1); iou_v /= max(m, 1)

        # save best
        if dice_v > best_dice:
            best_dice = dice_v
            best_path = logdir / f"seg_acdc_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), best_path)

        # --- NEW: log accuracy and F1 ---
        mean_acc = np.mean(val_accs) if val_accs else 0.0
        mean_f1_macro = np.mean(val_f1s_macro) if val_f1s_macro else 0.0
        mean_f1_weighted = np.mean(val_f1s_weighted) if val_f1s_weighted else 0.0

        sched.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            extra = ""
            if args.acdc_multiclass and m > 0:
                extra = f" | RV Dice/IoU {drv:.3f}/{irv:.3f} MYO {dmyo:.3f}/{imy:.3f} LV {dlv:.3f}/{ilv:.3f}"
            print(f"[Fold {fold_idx}] Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={(train_loss/max(1,nseen)):.4f} | dice={dice_v:.4f} iou={iou_v:.4f}{extra} | "
                  f"acc={mean_acc:.4f} f1={mean_f1_macro:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.6f}")
            if best_path is not None:
                print(f"  ↳ best so far: {best_dice:.4f} @ {best_path.name}")
            track_metric("val", {"dice": dice_v, "iou": iou_v, "accuracy": mean_acc,
                                "f1_macro": mean_f1_macro, "f1_weighted": mean_f1_weighted,
                                "epoch": epoch}, args)

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
    return float(dice_v), float(iou_v), str(out_pred), (dpc, ipc), mean_acc, mean_f1_macro, mean_f1_weighted, (str(best_path) if best_path is not None else "")


# ---------------- Main ----------------
def main():
    def _comma_list(s): return s if not s else s.replace(" ", "")

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--dataset", choices=["camus", "acdc"], required=True)
    ap.add_argument("--model", default="unet", 
                    choices=["unet", "unet3d_cram", "unetr"],
                    help="Model architecture: unet (default), unet3d_cram (U-Net+CRAM attention), unetr (Transformer)")
    ap.add_argument("--acdc-multiclass", action="store_true", help="Multiclass segmentation for ACDC (RV/MYO/LV; add --acdc-with-bg for BG too)")
    ap.add_argument("--acdc-with-bg", action="store_true", help="Use 4-class training for ACDC: BG=0, RV=1, MYO=2, LV=3")
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
    # NEW: previews + CSV
    ap.add_argument("--save-val-previews", action="store_true", help="Save contour overlays for first few val batches (ACDC)")
    ap.add_argument("--preview-batches", type=int, default=3, help="How many val batches to preview when --save-val-previews")
    ap.add_argument("--preview-slice", type=int, default=-1, help="Axial slice index to draw; <0 uses mid-slice")
    ap.add_argument("--perclass-csv", default="results/acdc_per_class_dice.csv", help="CSV to append per-epoch per-class Dice/IoU (ACDC)")
    # NEW: loss weighting & multiclass Dice
    ap.add_argument("--class-weights", default="", help="3 or 4 numbers (depending on classes) or 'auto'")
    ap.add_argument("--ce-weight", type=float, default=1.0, help="Weight for CrossEntropy in loss")
    ap.add_argument("--dice-weight", type=float, default=0.5, help="Weight for multiclass Dice in loss")
    # NEW: 3D augmentations
    ap.add_argument("--aug3d", action="store_true", help="Enable 3D flips/rotations/intensity jitter (ACDC)")
    ap.add_argument("--p-flip", type=float, default=0.5)
    ap.add_argument("--p-rot", type=float, default=0.5)
    ap.add_argument("--rot-deg", type=float, default=12.0, help="Max rotation angle in degrees")
    ap.add_argument("--p-gamma", type=float, default=0.5)
    ap.add_argument("--gamma-min", type=float, default=0.9)
    ap.add_argument("--gamma-max", type=float, default=1.15)
    ap.add_argument("--p-bright", type=float, default=0.5)
    ap.add_argument("--bright-min", type=float, default=0.9)
    ap.add_argument("--bright-max", type=float, default=1.1)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
            dice_v, iou_v, artifact, acc_v, f1_macro_v, f1_weighted_v, best_ckpt = run_fold_seg_2d(tr_idxs, va_idxs, args.phase, logdir, args, fold)
            dpc = ipc = None
        else:
            dice_v, iou_v, artifact, tpl, acc_v, f1_macro_v, f1_weighted_v, best_ckpt = run_fold_seg_3d(tr_idxs, va_idxs, args.phase, logdir, args, fold)
            dpc, ipc = tpl if tpl is not None else (None, None)

        print(f"[Fold {fold}] Dice={dice_v:.4f} IoU={iou_v:.4f} Acc={acc_v:.4f} F1-macro={f1_macro_v:.4f}")
        row = {
            "fold": fold, 
            "Dice": dice_v, 
            "IoU": iou_v, 
            "Accuracy": acc_v,
            "F1_macro": f1_macro_v,
            "F1_weighted": f1_weighted_v,
            "artifact": artifact
        }
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
                     {"fold_dice": dice_v, "fold_iou": iou_v, "fold_accuracy": acc_v,
                      "fold_f1_macro": f1_macro_v, "fold_f1_weighted": f1_weighted_v, "fold": fold},
                     args,
                     images={f"{args.dataset}_fold{fold}": artifact})

    # Summary
    d = np.array([m["Dice"] for m in metrics], dtype=float)
    j = np.array([m["IoU"]  for m in metrics if not np.isnan(m["IoU"])], dtype=float) if any(not np.isnan(m["IoU"]) for m in metrics) else np.array([])
    a = np.array([m["Accuracy"] for m in metrics], dtype=float)
    f1m = np.array([m["F1_macro"] for m in metrics], dtype=float)
    f1w = np.array([m["F1_weighted"] for m in metrics], dtype=float)
    
    summary = {
        "Dice_mean": float(np.nanmean(d)), "Dice_std": float(np.nanstd(d)),
        "IoU_mean":  float(np.mean(j)) if j.size>0 else float('nan'),
        "IoU_std":   float(np.std(j))  if j.size>0 else float('nan'),
        "Accuracy_mean": float(np.nanmean(a)), "Accuracy_std": float(np.nanstd(a)),
        "F1_macro_mean": float(np.nanmean(f1m)), "F1_macro_std": float(np.nanstd(f1m)),
        "F1_weighted_mean": float(np.nanmean(f1w)), "F1_weighted_std": float(np.nanstd(f1w)),
        "folds": len(metrics), "dataset": args.dataset, "phase": args.phase, "view": args.view,
        "feat2d": args.feat2d, "feat3d": args.feat3d, "amp": args.amp,
        "grad_clip": args.grad_clip, "accum": args.accum, "num_workers": args.num_workers,
        "model": args.model
    }
    if args.dataset == "acdc" and args.acdc_multiclass:
        pdf = pd.DataFrame(metrics)
        for col in ["Dice_RV", "Dice_MYO", "Dice_LV", "IoU_RV", "IoU_MYO", "IoU_LV"]:
            if col in pdf.columns:
                summary[col + "_mean"] = float(pdf[col].mean())
                summary[col + "_std"]  = float(pdf[col].std())

    print("\n=== CV Summary ===")
    print(f"Dice      mean±std: {summary['Dice_mean']:.4f} ± {summary['Dice_std']:.4f}")
    print(f"IoU       mean±std: {summary['IoU_mean']:.4f} ± {summary['IoU_std']:.4f}")
    print(f"Accuracy  mean±std: {summary['Accuracy_mean']:.4f} ± {summary['Accuracy_std']:.4f}")
    print(f"F1-macro  mean±std: {summary['F1_macro_mean']:.4f} ± {summary['F1_macro_std']:.4f}")
    print(f"F1-weight mean±std: {summary['F1_weighted_mean']:.4f} ± {summary['F1_weighted_std']:.4f}")

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
