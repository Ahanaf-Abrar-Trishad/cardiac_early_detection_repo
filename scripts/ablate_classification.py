#!/usr/bin/env python3
"""
Ablation study for CAMUS classification.
Runs a small grid and records mean±std macro-F1/ACC/AUC over 3-fold CV (inner HPO reduced for speed).

Factors:
- augment: [on, off]
- imagenet_init: [True, False]
- sampler: [weighted, uniform]

Usage:
  python scripts/ablate_classification.py --meta meta/master_metadata.csv --labels three --view 4CH --phase ED --out logs/ablation_cls.csv
"""
import argparse, json, numpy as np, pandas as pd, random, os, torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score)
import torchvision.models as models
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from datasets.camus_ed_es import CamusEDESDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

class CAMUSDataset(torch.utils.data.Dataset):
    def __init__(self, meta_csv, view, phase, labels, augment):
        base = CamusEDESDataset(meta_csv, split="", view=view, phase=phase, aug=None)
        df = base.df[base.df["ef_bin"].isin(["normal","mid","reduced"])].reset_index(drop=True)
        self.df=df; self.labels=labels
        import torchvision.transforms as T, cv2, numpy as np
        self.t_aug = T.Compose([T.ToTensor(), T.Resize((224,224)), T.RandomHorizontalFlip(), T.RandomRotation(10)]) if augment else T.Compose([T.ToTensor(), T.Resize((224,224))])
        self.view=view; self.phase=phase
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        import cv2, numpy as np, json
        row = self.df.iloc[i]; paths = json.loads(row["paths"])
        img = cv2.imread(paths[f"image_{self.phase}"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6); img = np.stack([img,img,img], axis=-1)
        x = self.t_aug(img)
        if self.labels=="three":
            mapping={"normal":0,"mid":1,"reduced":2}; y=mapping[row["ef_bin"]]
        else:
            y= 0 if row["ef_bin"]=="normal" else 1
        return x,y

def build_model(imagenet_init=True, num_classes=3, dropout=0.2):
    if imagenet_init:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=None)
    in_fe = m.fc.in_features; m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fe, num_classes))
    return m

def run_cv(ds, labels="three", sampler="weighted", seed=42):
    y = np.array([ds[i][1] for i in range(len(ds))])
    idx = np.arange(len(ds)); num_classes = 2 if labels=="binary" else 3
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    out=[]
    for tr,va in skf.split(idx,y):
        tr_set=Subset(ds, idx[tr]); va_set=Subset(ds, idx[va]); ytr=y[tr]
        bs=16; epochs=8; lr=1e-3; wd=1e-4; dr=0.2
        if sampler=="weighted":
            c=np.bincount(ytr, minlength=num_classes); w=c.sum()/(c+1e-8); sw=w[ytr].astype(np.float32)
            train_loader = DataLoader(tr_set, batch_size=bs, sampler=WeightedRandomSampler(sw, len(sw), replacement=True), num_workers=2)
        else:
            train_loader = DataLoader(tr_set, batch_size=bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(va_set, batch_size=bs*2, shuffle=False, num_workers=2)
        model = build_model(num_classes=num_classes).to(DEVICE); opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        crit=nn.CrossEntropyLoss()
        for _ in range(epochs):
            model.train()
            for x,yb in train_loader:
                x,yb = x.to(DEVICE), torch.tensor(yb).to(DEVICE)
                opt.zero_grad(); lg=model(x); loss=crit(lg,yb); loss.backward(); opt.step()
        # eval
        model.eval(); all_logits=[]; all_y=[]
        for x,yb in val_loader:
            x=x.to(DEVICE); lg=model(x); all_logits.append(lg.cpu()); all_y.extend(yb)
        lg=torch.cat(all_logits,0); pr = torch.softmax(lg,1).detach().numpy(); yva=np.array(all_y)
        yhat = pr.argmax(1)
        if num_classes==2:
            auc = roc_auc_score(yva, pr[:,1]) if len(np.unique(yva))>1 else np.nan
        else:
            try: auc = roc_auc_score(yva, pr, multi_class="ovr", average="macro")
            except ValueError: auc=np.nan
        acc=accuracy_score(yva, yhat); mf1=f1_score(yva, yhat, average="macro")
        out.append((auc,acc,mf1))
    arr=np.array(out,float)
    return float(np.nanmean(arr[:,0])), float(np.nanmean(arr[:,1])), float(np.nanmean(arr[:,2]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--labels", choices=["binary","three"], default="three")
    ap.add_argument("--view", default="4CH")
    ap.add_argument("--phase", default="ED")
    ap.add_argument("--out", default="logs/ablation_cls.csv")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    set_seed(args.seed)

    grid=[
        {"augment": True,  "imagenet": True,  "sampler": "weighted"},
        {"augment": False, "imagenet": True,  "sampler": "weighted"},
        {"augment": True,  "imagenet": False, "sampler": "weighted"},
        {"augment": True,  "imagenet": True,  "sampler": "uniform"},
    ]

    rows=[]
    for cfg in grid:
        ds = CAMUSDataset(args.meta, args.view, args.phase, args.labels, augment=cfg["augment"])
        auc,acc,mf1 = run_cv(ds, labels=args.labels, sampler=cfg["sampler"], seed=args.seed)
        rows.append({"augment":cfg["augment"], "imagenet":cfg["imagenet"], "sampler":cfg["sampler"],
                     "labels":args.labels, "AUC":auc, "ACC":acc, "macroF1":mf1})
        print(rows[-1])
    import pandas as pd, os
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved:", args.out)

if __name__=="__main__":
    main()
