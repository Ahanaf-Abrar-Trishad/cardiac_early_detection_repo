import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
import nibabel as nib


def _first_present(d: dict, candidates):
    """Return the first existing non-empty key from candidates, else None."""
    for k in candidates:
        v = d.get(k, "")
        if isinstance(v, str) and v.strip():
            p = Path(v)
            if p.exists():
                return str(p)
    return None


def _load_2d(path: str):
    """Load a single 2D image/mask from PNG/JPG or a 2D NIfTI slice."""
    p = Path(path)
    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        arr = np.array(Image.open(p))
        # normalize grayscale if RGB accidentally present
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)
    else:
        # assume NIfTI
        vol = nib.load(str(p)).get_fdata()
        # If it is 2D already: (H,W) -> return as is; if 3D, take middle slice in Z
        if vol.ndim == 2:
            arr = vol
        elif vol.ndim == 3:
            z = vol.shape[2] // 2
            arr = vol[:, :, z]
        else:
            raise ValueError(f"Unsupported NIfTI dims for 2D CAMUS file: {vol.shape} at {path}")
        return arr.astype(np.float32)


class CamusEDESDataset(Dataset):
    """
    Robust CAMUS ED/ES dataset for 2D segmentation.

    Expects rows in meta with:
      - dataset == 'camus'
      - view in {'2CH','4CH'}
      - paths (JSON) containing image/mask paths.

    Accepts multiple key variants inside 'paths':
      image keys tried (in order):
        image_{PHASE}, image, png_image_{PHASE}, png_image, nii_image_{PHASE}, nii_image, image_{VIEW}_{PHASE}
      mask keys tried (in order):
        mask_{PHASE}, mask, png_mask_{PHASE}, png_mask, nii_mask_{PHASE}, nii_mask, mask_{VIEW}_{PHASE}
    """
    def __init__(self, meta_csv: str, split: str = "", view: str = "4CH", phase: str = "ED", aug=None):
        super().__init__()
        self.view = view
        self.phase = phase
        self.aug = aug

        df = pd.read_csv(meta_csv)
        df = df[df["dataset"] == "camus"].reset_index(drop=True)
        if "view" in df.columns:
            df = df[df["view"] == view].reset_index(drop=True)

        # filter rows that actually have a mask for this phase
        keep_rows = []
        for _, r in df.iterrows():
            try:
                paths = json.loads(r["paths"])
            except Exception:
                continue
            img_candidates = [
                f"image_{phase}", "image",
                f"png_image_{phase}", "png_image",
                f"nii_image_{phase}", "nii_image",
                f"image_{view}_{phase}",
            ]
            msk_candidates = [
                f"mask_{phase}", "mask",
                f"png_mask_{phase}", "png_mask",
                f"nii_mask_{phase}", "nii_mask",
                f"mask_{view}_{phase}",
            ]
            img = _first_present(paths, img_candidates)
            msk = _first_present(paths, msk_candidates)
            if img and msk:
                keep_rows.append(True)
            else:
                keep_rows.append(False)

        self.df = df.loc[keep_rows].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "CAMUS dataset: no usable rows found for "
                f"view={view}, phase={phase}. "
                "Check that meta/master_metadata.csv contains valid 'paths' for CAMUS with image/mask files."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        paths = json.loads(row["paths"])

        img_candidates = [
            f"image_{self.phase}", "image",
            f"png_image_{self.phase}", "png_image",
            f"nii_image_{self.phase}", "nii_image",
            f"image_{self.view}_{self.phase}",
        ]
        msk_candidates = [
            f"mask_{self.phase}", "mask",
            f"png_mask_{self.phase}", "png_mask",
            f"nii_mask_{self.phase}", "nii_mask",
            f"mask_{self.view}_{self.phase}",
        ]

        img_path = _first_present(paths, img_candidates)
        msk_path = _first_present(paths, msk_candidates)

        if img_path is None or msk_path is None:
            raise KeyError(
                "CAMUS sample missing expected image/mask keys.\n"
                f" Tried image keys: {img_candidates}\n"
                f" Tried mask  keys: {msk_candidates}\n"
                f" Available keys: {list(paths.keys())}\n"
                f" Row: study_id={row.get('study_id','?')} patient_id={row.get('patient_id','?')}"
            )

        img = _load_2d(img_path)
        msk = _load_2d(msk_path)

        # normalize image to [0,1] robustly
        if np.ptp(img) > 0:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img, dtype=np.float32)

        # binarize mask (>0)
        msk = (msk > 0).astype(np.float32)

        # to torch tensors (1,H,W)
        img_t = torch.from_numpy(img)[None, ...].float()
        msk_t = torch.from_numpy(msk)[None, ...].float()

        if self.aug is not None:
            img_t, msk_t = self.aug(img_t, msk_t)

        return {"image": img_t, "mask": msk_t, "image_path": img_path, "mask_path": msk_path, "row": row.to_dict()}

