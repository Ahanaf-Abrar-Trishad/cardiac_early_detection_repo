#!/usr/bin/env python3
"""
Train the feature fusion classifier with RAP blocks.
Uses extracted features from segmentation + EF.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.fusion_classifier import FeatureFusionClassifier


class FeatureDataset(Dataset):
    """Dataset for tabular features."""
    def __init__(self, features_df, geo_cols, ef_cols, scaler=None, fit_scaler=False):
        self.patient_ids = features_df["patient_id"].values
        self.labels = features_df["label_enc"].values
        
        # Extract feature groups
        geo_data = features_df[geo_cols].values.astype(np.float32)
        ef_data = features_df[ef_cols].values.astype(np.float32)
        
        # Standardize
        if fit_scaler:
            self.scaler_geo = StandardScaler().fit(geo_data)
            self.scaler_ef = StandardScaler().fit(ef_data)
        else:
            self.scaler_geo = scaler['geo']
            self.scaler_ef = scaler['ef']
        
        self.geo_feat = self.scaler_geo.transform(geo_data)
        self.ef_feat = self.scaler_ef.transform(ef_data)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'geometric': torch.from_numpy(self.geo_feat[idx]),
            'ef': torch.from_numpy(self.ef_feat[idx]),
            'label': self.labels[idx]
        }
    
    def get_scalers(self):
        return {'geo': self.scaler_geo, 'ef': self.scaler_ef}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        geo = batch['geometric'].to(device)
        ef = batch['ef'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(geo, ef)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        geo = batch['geometric'].to(device)
        ef = batch['ef'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(geo, ef)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # AUC (handle binary/multiclass)
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / len(loader),
        'acc': acc,
        'bal_acc': bal_acc,
        'f1_macro': f1_macro,
        'auc': auc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="meta/acdc_features.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-rap-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion-type", type=str, default="rap")
    parser.add_argument("--use-cross-attention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load features
    df = pd.read_csv(args.features)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    # Define feature groups
    ef_cols = [c for c in df.columns if 'EF' in c]
    geo_cols = [c for c in df.columns if c.endswith('_mL') or c.endswith('_mm')]
    
    if not ef_cols or not geo_cols:
        raise ValueError("No EF or geometric features found!")
    
    print(f"Geometric features ({len(geo_cols)}): {geo_cols}")
    print(f"EF features ({len(ef_cols)}): {ef_cols}")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(df["label"].astype(str))
    df["label_enc"] = labels_encoded
    
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = []
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_enc"]), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{args.folds}")
        print(f"{'='*60}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = FeatureDataset(train_df, geo_cols, ef_cols, fit_scaler=True)
        scaler = train_dataset.get_scalers()
        val_dataset = FeatureDataset(val_df, geo_cols, ef_cols, scaler=scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = FeatureFusionClassifier(
            geometric_dim=len(geo_cols),
            ef_dim=len(ef_cols),
            vol_dim=0,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            use_cross_attention=args.use_cross_attention,
            dropout=args.dropout
        ).to(args.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
            val_metrics = eval_epoch(model, val_loader, criterion, args.device, num_classes)
            scheduler.step()
            
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_epoch = epoch
                checkpoint = {
                    'model': model.state_dict(),
                    'scaler': scaler,
                    'label_encoder': le,
                    'args': vars(args)
                }
                torch.save(checkpoint, logdir / f"fusion_classifier_fold{fold}_best.pt")
            
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:3d}/{args.epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                      f"BalAcc: {val_metrics['bal_acc']:.4f} F1: {val_metrics['f1_macro']:.4f} "
                      f"AUC: {val_metrics['auc']:.4f}")
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Load best model and evaluate
        checkpoint = torch.load(logdir / f"fusion_classifier_fold{fold}_best.pt", weights_only=False)
        model.load_state_dict(checkpoint['model'])
        val_metrics = eval_epoch(model, val_loader, criterion, args.device, num_classes)
        val_metrics['fold'] = fold
        fold_metrics.append(val_metrics)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    
    metrics_df = pd.DataFrame(fold_metrics)
    for metric in ['acc', 'bal_acc', 'f1_macro', 'auc']:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std()
        print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    # Save summary
    summary = {
        'args': vars(args),
        'fold_metrics': fold_metrics,
        'mean_metrics': {
            'acc': float(metrics_df['acc'].mean()),
            'bal_acc': float(metrics_df['bal_acc'].mean()),
            'f1_macro': float(metrics_df['f1_macro'].mean()),
            'auc': float(metrics_df['auc'].mean())
        }
    }
    
    with open(logdir / "fusion_classifier_cv_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {logdir}")


if __name__ == "__main__":
    main()
