#!/usr/bin/env python3
"""
Image classification CV (CAMUS) with nested HPO via Optuna.
- Supports binary ('normal' vs 'abnormal') and 3-class ('normal','mid','reduced') labels.
- Logs per-fold: AUC (binary or macro-OVR for 3-class), ACC, macro-F1; saves confusion matrix & per-class report.
- Saves per-class ROC curves (PNG/JSON) when possible.

Usage examples:
  # Binary (normal vs {mid,reduced})
  python scripts/torch_cv.py --meta meta/master_metadata.csv --labels binary --view 4CH --phase ED --folds 5 --seed 42 --trials 25 --logdir logs

  # 3-class
  python scripts/torch_cv.py --meta meta/master_metadata.csv --labels three --view 4CH --phase ED --folds 5 --seed 42 --trials 25 --logdir logs
"""
import argparse, numpy as np, pandas as pd, json, random, os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve, average_precision_score)
from datasets.camus_ed_es import CamusEDESDataset
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CAMUSClassifier(torch.utils.data.Dataset):
    def __init__(self, meta_csv, view="4CH", phase="ED", aug=False, labels="binary"):
        base = CamusEDESDataset(meta_csv, split="", view=view, phase=phase, aug=None)
        df = base.df[base.df["ef_bin"].isin(["normal","mid","reduced"])].reset_index(drop=True)
        self.df = df
        self.view=view; self.phase=phase; self.aug=aug; self.labels=labels
        self._trans_train = T.Compose([T.ToTensor(), T.Resize((224,224)), T.RandomHorizontalFlip(), T.RandomRotation(10)])
        self._trans_val   = T.Compose([T.ToTensor(), T.Resize((224,224))])

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]; paths = json.loads(row["paths"])
        import cv2, numpy as np
        img = cv2.imread(paths[f"image_{self.phase}"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.stack([img, img, img], axis=-1)
        img = self._trans_train(img) if self.aug else self._trans_val(img)
        if self.labels == "three":
            # normal=0, mid=1, reduced=2
            mapping = {"normal":0, "mid":1, "reduced":2}
            y = mapping.get(row["ef_bin"], 0)
        else:
            y = 0 if row["ef_bin"]=="normal" else 1
        return img, y

def build_model(num_classes=2, dropout=0.2):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_fe = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fe, num_classes))
    return m

def train_one_epoch(model, loader, optimizer, criterion):
    model.train(); total=0.0
    for x,y in loader:
        x,y = x.to(DEVICE), torch.tensor(y).to(DEVICE)
        optimizer.zero_grad(); logits = model(x); loss = criterion(logits,y); loss.backward(); optimizer.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, num_classes):
    model.eval(); all_logits=[]; all_y=[]
    for x,y in loader:
        x=x.to(DEVICE); logits = model(x); all_logits.append(logits.cpu()); all_y.extend(y)
    logits = torch.cat(all_logits, dim=0); probs = torch.softmax(logits, dim=1).numpy()
    import numpy as np
    y = np.array(all_y)
    y_pred = probs.argmax(axis=1)
    # AUC
    if num_classes == 2:
        auc = roc_auc_score(y, probs[:,1]) if len(np.unique(y))>1 else np.nan
    else:
        try:
            auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    # per-class report
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=list(range(num_classes)))
    return auc, acc, macro_f1, y, probs, y_pred, report, cm

def save_roc_curves(y, probs, out_prefix, num_classes):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np, json
        if num_classes == 2:
            fpr, tpr, thr = roc_curve(y, probs[:,1])
            plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
            plt.savefig(out_prefix.with_suffix('.roc.png')); plt.close()
            with open(out_prefix.with_suffix('.roc.json'),'w') as f: json.dump({'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thr':thr.tolist()}, f, indent=2)
        else:
            # one-vs-rest curves
            for k in range(num_classes):
                y_k = (y==k).astype(int)
                if len(np.unique(y_k))<2: continue
                fpr, tpr, thr = roc_curve(y_k, probs[:,k])
                plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC class {k}')
                plt.savefig(out_prefix.with_suffix(f'.class{k}.roc.png')); plt.close()
                with open(out_prefix.with_suffix(f'.class{k}.roc.json'),'w') as f: json.dump({'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thr':thr.tolist()}, f, indent=2)
    except Exception as e:
        print("ROC save failed:", e)

def save_cm(cm, out_prefix, class_names):
    import json, numpy as np
    plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.colorbar(); plt.savefig(out_prefix.with_suffix('.cm.png')); plt.close()
    with open(out_prefix.with_suffix('.cm.json'),'w') as f:
        json.dump({'labels': class_names, 'matrix': cm.tolist()}, f, indent=2)

def objective_factory(idx, y, ds, seed, num_classes):
    def objective(trial):
        set_seed(seed+trial.number)
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [8,16,32])
        dr = trial.suggest_float("dropout", 0.0, 0.5)
        epochs = trial.suggest_int("epochs", 5, 12)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores=[]
        for tr,va in skf.split(idx, y):
            from torch.utils.data import Subset
            tr_subset = Subset(ds, idx[tr]); va_subset = Subset(ds, idx[va])
            ytr=y[tr]; import numpy as np, torch
            counts = np.bincount(ytr, minlength=num_classes)
            weights = counts.sum()/(counts+1e-8)
            sw = weights[ytr].astype(np.float32)
            train_loader = DataLoader(tr_subset, batch_size=bs, sampler=WeightedRandomSampler(sw, len(sw), replacement=True), num_workers=2)
            val_loader   = DataLoader(va_subset, batch_size=bs*2, shuffle=False, num_workers=2)
            model = build_model(num_classes=num_classes, dropout=dr).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit = nn.CrossEntropyLoss()
            for _ in range(epochs): train_one_epoch(model, train_loader, opt, crit)
            auc, acc, macro_f1, *_ = eval_model(model, val_loader, num_classes); scores.append(macro_f1 if not np.isnan(macro_f1) else acc)
        return float(np.nanmean(scores))
    return objective

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="meta/master_metadata.csv")
    ap.add_argument("--labels", choices=["binary","three"], default="binary")
    ap.add_argument("--view", default="4CH", choices=["2CH","4CH"])
    ap.add_argument("--phase", default="ED", choices=["ED","ES"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--logdir", default="logs", help="Directory to save CSV/JSON metrics")
    args = ap.parse_args()

    set_seed(args.seed)
    num_classes = 2 if args.labels=="binary" else 3
    ds = CAMUSClassifier(args.meta, view=args.view, phase=args.phase, aug=True, labels=args.labels)
    y=[]; 
    for i in range(len(ds)): _,yi = ds[i]; y.append(yi)
    import numpy as np
    y = np.array(y); idx = np.arange(len(ds))

    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results=[]; reports_all=[]

    for fold,(tr,va) in enumerate(skf.split(idx,y), start=1):
        print(f"=== Fold {fold}/{args.folds} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_factory(idx[tr], y[tr], ds, args.seed+fold*100, num_classes), n_trials=args.trials, show_progress_bar=False)
        best = study.best_params; print("Best:", best)
        bs=best["batch_size"]; lr=best["lr"]; wd=best["weight_decay"]; dr=best["dropout"]; epochs=best["epochs"]
        from torch.utils.data import Subset
        tr_subset = Subset(ds, idx[tr]); va_subset = Subset(ds, idx[va])
        ytr=y[tr]; import numpy as np, torch
        counts = np.bincount(ytr, minlength=num_classes)
        weights = counts.sum()/(counts+1e-8)
        sw = weights[ytr].astype(np.float32)
        train_loader = DataLoader(tr_subset, batch_size=bs, sampler=WeightedRandomSampler(sw, len(sw), replacement=True), num_workers=2)
        val_loader   = DataLoader(va_subset, batch_size=bs*2, shuffle=False, num_workers=2)
        model = build_model(num_classes=num_classes, dropout=dr).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs): train_one_epoch(model, train_loader, opt, crit)
        auc, acc, macro_f1, y_true, probs, y_pred, report, cm = eval_model(model, val_loader, num_classes)
        # Extra metrics
        try:
            if num_classes==2:
                auc_macro = auc
                auc_weighted = auc
                ap_macro = average_precision_score(y_true, probs[:,1])
                ap_weighted = ap_macro
            else:
                from sklearn.preprocessing import label_binarize
                Y = label_binarize(y_true, classes=list(range(num_classes)))
                auc_macro = roc_auc_score(Y, probs, average='macro')
                auc_weighted = roc_auc_score(Y, probs, average='weighted')
                ap_macro = average_precision_score(Y, probs, average='macro')
                ap_weighted = average_precision_score(Y, probs, average='weighted')
        except Exception:
            auc_macro = float('nan'); auc_weighted=float('nan'); ap_macro=float('nan'); ap_weighted=float('nan')
        print(f"[Fold {fold}] AUC={auc:.4f} ACC={acc:.4f} macroF1={macro_f1:.4f} | AUC_macro={auc_macro:.4f} AP_macro={ap_macro:.4f}")
        # Save confusion & ROC
        out_prefix = logdir / f"cv_image_{args.labels}_fold{fold}"
        save_cm(cm, out_prefix, class_names=(["normal","abnormal"] if num_classes==2 else ["normal","mid","reduced"]))
        save_roc_curves(y_true, probs, out_prefix, num_classes)
        save_pr_curves(y_true, probs, out_prefix, num_classes)
        # Save report
        with open(logdir / f"cv_image_{args.labels}_fold{fold}_report.json","w") as f: json.dump(report, f, indent=2)
        results.append({"fold": fold, "AUC": float(auc) if not np.isnan(auc) else float('nan'),
                        "ACC": float(acc), "macroF1": float(macro_f1),
                        "AUC_macro": float(auc_macro), "AUC_weighted": float(auc_weighted),
                        "AP_macro": float(ap_macro), "AP_weighted": float(ap_weighted)})

    m_auc = float(np.nanmean([r["AUC"] for r in results])) if results else float('nan')
    s_auc = float(np.nanstd([r["AUC"] for r in results])) if results else float('nan')
    m_acc = float(np.mean([r["ACC"] for r in results])) if results else float('nan')
    s_acc = float(np.std([r["ACC"] for r in results])) if results else float('nan')
    m_f1  = float(np.mean([r["macroF1"] for r in results])) if results else float('nan')
    s_f1  = float(np.std([r["macroF1"] for r in results])) if results else float('nan')

    print("\\n=== Outer-CV Summary ===")
    print(f"AUC mean±std: {m_auc:.4f} ± {s_auc:.4f}")
    print(f"ACC mean±std: {m_acc:.4f} ± {s_acc:.4f}")
    print(f"macroF1 mean±std: {m_f1:.4f} ± {s_f1:.4f}")

    # --- Save metrics ---
    df = pd.DataFrame(results).sort_values("fold")
    df.to_csv(logdir / f"cv_image_{args.labels}_metrics.csv", index=False)
    with open(logdir / f"cv_image_{args.labels}_summary.json", "w") as f: json.dump({
        "AUC_mean": m_auc, "AUC_std": s_auc, "ACC_mean": m_acc, "ACC_std": s_acc, "macroF1_mean": m_f1, "macroF1_std": s_f1,
        "folds": len(results), "view": args.view, "phase": args.phase, "trials": args.trials, "seed": args.seed, "labels": args.labels
    }, f, indent=2)

def save_pr_curves(y, probs, out_prefix, num_classes):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np, json
        if num_classes == 2:
            p, r, thr = precision_recall_curve(y, probs[:,1])
            plt.figure(); plt.plot(r, p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR')
            plt.savefig(out_prefix.with_suffix('.pr.png')); plt.close()
            with open(out_prefix.with_suffix('.pr.json'),'w') as f: json.dump({'precision':p.tolist(),'recall':r.tolist(),'thr':thr.tolist()}, f, indent=2)
        else:
            for k in range(num_classes):
                y_k = (y==k).astype(int)
                if len(np.unique(y_k))<2: continue
                p, r, thr = precision_recall_curve(y_k, probs[:,k])
                plt.figure(); plt.plot(r, p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR class {k}')
                plt.savefig(out_prefix.with_suffix(f'.class{k}.pr.png')); plt.close()
                with open(out_prefix.with_suffix(f'.class{k}.pr.json'),'w') as f: json.dump({'precision':p.tolist(),'recall':r.tolist(),'thr':thr.tolist()}, f, indent=2)
    except Exception as e:
        print("PR save failed:", e)

if __name__ == "__main__":
    main()
