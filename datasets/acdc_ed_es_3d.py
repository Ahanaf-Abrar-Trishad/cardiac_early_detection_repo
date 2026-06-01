# datasets/acdc_ed_es_3d.py
import json, nibabel as nib, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class ACDC3DEDS(Dataset):
    """
    Loads ACDC ED/ES 3D volumes and masks (NIfTI) for a given split.
    """
    def __init__(self, meta_csv, split="train", phase="ED", aug=None):
        self.df = pd.read_csv(meta_csv)
        self.df = self.df[(self.df.dataset=="acdc")]
        if split:
            self.df = self.df[self.df.split==split]
        self.phase = phase  # "ED" or "ES"
        self.aug = aug

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        paths = json.loads(row["paths"])
        img_path = paths[f"nii_image_{self.phase}"]
        msk_path = paths[f"nii_mask_{self.phase}"]

        img = nib.load(img_path).get_fdata().astype(np.float32)
        msk = nib.load(msk_path).get_fdata().astype(np.uint8)

        v = img
        v = (v - np.mean(v)) / (np.std(v) + 1e-6)

        if self.aug:
            v, msk = self.aug(v, msk)

        v = torch.from_numpy(v[None, ...])  # (1,Z,Y,X)
        msk = torch.from_numpy(msk.astype(np.int64))
        return {"image": v, "mask": msk, "label": row.get("acdc_label", "NA"), "ef": row.get("ef_lv", np.nan), "meta": row.to_dict()}
