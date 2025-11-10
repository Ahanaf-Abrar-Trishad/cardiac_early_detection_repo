#!/usr/bin/env python3
"""
Train Advanced Attention-Based Classifiers for Cardiac Disease Classification

Supports two architectures:
1. AdvancedAttentionClassifier - Single-input with multi-head self-attention
2. MultiModalAttentionClassifier - Multi-modal with cross-attention

Implements state-of-the-art attention mechanisms:
- Multi-Head Self-Attention (MHSA)
- Squeeze-and-Excitation (SE) channel attention
- Residual Attention Blocks
- Cross-Modal Attention (for multi-modal variant)
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
from scipy import stats

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.advanced_attention_classifier import (
    AdvancedAttentionClassifier,
    MultiModalAttentionClassifier
)


class SingleInputDataset(Dataset):
    """Dataset for single-input attention classifier (all features combined)."""
    def __init__(self, features_df, feature_cols, scaler=None, fit_scaler=False):
        self.patient_ids = features_df["patient_id"].values
        self.labels = features_df["label_enc"].values
        
        # Extract all features
        feat_data = features_df[feature_cols].values.astype(np.float32)
        
        # Standardize
        if fit_scaler:
            self.scaler = StandardScaler().fit(feat_data)
        else:
            self.scaler = scaler
        
        self.features = self.scaler.transform(feat_data)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.features[idx]),
            'label': self.labels[idx]
        }
    
    def get_scaler(self):
        return self.scaler


class MultiModalDataset(Dataset):
    """Dataset for multi-modal attention classifier (geometric + functional)."""
    def __init__(self, features_df, geo_cols, func_cols, scaler=None, fit_scaler=False):
        self.patient_ids = features_df["patient_id"].values
        self.labels = features_df["label_enc"].values
        
        # Extract feature groups
        geo_data = features_df[geo_cols].values.astype(np.float32)
        func_data = features_df[func_cols].values.astype(np.float32)
        
        # Standardize
        if fit_scaler:
            self.scaler_geo = StandardScaler().fit(geo_data)
            self.scaler_func = StandardScaler().fit(func_data)
        else:
            self.scaler_geo = scaler['geo']
            self.scaler_func = scaler['func']
        
        self.geo_feat = self.scaler_geo.transform(geo_data)
        self.func_feat = self.scaler_func.transform(func_data)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'geometric': torch.from_numpy(self.geo_feat[idx]),
            'functional': torch.from_numpy(self.func_feat[idx]),
            'label': self.labels[idx]
        }
    
    def get_scalers(self):
        return {'geo': self.scaler_geo, 'func': self.scaler_func}


def train_epoch(model, loader, criterion, optimizer, device, model_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        labels = batch['label'].to(device)
        
        # Forward pass depends on model type
        if model_type == 'advanced':
            features = batch['features'].to(device)
            logits = model(features)
        else:  # multimodal
            geo = batch['geometric'].to(device)
            func = batch['functional'].to(device)
            logits = model(geo, func)
        
        # Backward pass
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes, model_type):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        labels = batch['label'].to(device)
        
        # Forward pass depends on model type
        if model_type == 'advanced':
            features = batch['features'].to(device)
            logits = model(features)
        else:  # multimodal
            geo = batch['geometric'].to(device)
            func = batch['functional'].to(device)
            logits = model(geo, func)
        
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
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
        'f1_weighted': f1_weighted,
        'auc': auc,
        'probs': all_probs,
        'preds': all_preds,
        'labels': all_labels
    }


def compute_ci95(arr):
    """Compute 95% confidence interval using t-distribution."""
    n = len(arr)
    if n < 2:
        return float('nan'), float('nan')
    mean = np.nanmean(arr)
    sem = stats.sem(arr, nan_policy='omit')
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    return float(ci[0]), float(ci[1])


def main():
    parser = argparse.ArgumentParser(description="Train Advanced Attention Classifiers")
    
    # Data arguments
    parser.add_argument("--features", default="meta/acdc_features.csv", help="Path to features CSV")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="advanced", 
                        choices=["advanced", "multimodal"],
                        help="Model architecture: 'advanced' (single input) or 'multimodal' (geo+func)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of attention blocks")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    
    # System arguments
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", default="logs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed training progress")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 80)
    print(f"Training {args.model_type.upper()} Attention Classifier")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_type}")
    print(f"Hidden dim: {args.hidden_dim}, Blocks: {args.num_blocks}, Heads: {args.num_heads}")
    print(f"Dropout: {args.dropout}, LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Folds: {args.folds}")
    
    # Load features
    df = pd.read_csv(args.features)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    print(f"\nDataset: {len(df)} patients")
    
    # Define feature groups
    ef_cols = [c for c in df.columns if 'EF' in c]
    geo_cols = [c for c in df.columns if c.endswith('_mL') or c.endswith('_mm')]
    all_feat_cols = geo_cols + ef_cols
    
    if args.model_type == 'advanced':
        print(f"Features ({len(all_feat_cols)}): {all_feat_cols}")
    else:
        print(f"Geometric features ({len(geo_cols)}): {geo_cols}")
        print(f"Functional features ({len(ef_cols)}): {ef_cols}")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(df["label"].astype(str))
    df["label_enc"] = labels_encoded
    
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    fold_metrics = []
    oof_probs = np.zeros((len(df), num_classes))  # Out-of-fold predictions
    oof_labels = np.zeros(len(df))
    
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)
    
    # Save out-of-fold predictions directory
    oof_dir = logdir / "oof_preds"
    oof_dir.mkdir(exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_enc"]), 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold}/{args.folds}")
        print(f"{'='*80}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train: {len(train_df)} patients | Val: {len(val_df)} patients")
        
        # Create datasets and loaders
        if args.model_type == 'advanced':
            train_dataset = SingleInputDataset(train_df, all_feat_cols, fit_scaler=True)
            scaler = train_dataset.get_scaler()
            val_dataset = SingleInputDataset(val_df, all_feat_cols, scaler=scaler)
        else:  # multimodal
            train_dataset = MultiModalDataset(train_df, geo_cols, ef_cols, fit_scaler=True)
            scaler = train_dataset.get_scalers()
            val_dataset = MultiModalDataset(val_df, geo_cols, ef_cols, scaler=scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Create model
        if args.model_type == 'advanced':
            model = AdvancedAttentionClassifier(
                input_features=len(all_feat_cols),
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout
            ).to(args.device)
        else:  # multimodal
            model = MultiModalAttentionClassifier(
                num_geometric=len(geo_cols),
                num_functional=len(ef_cols),
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                dropout=args.dropout
            ).to(args.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, args.device, args.model_type
            )
            val_metrics = eval_epoch(
                model, val_loader, criterion, args.device, num_classes, args.model_type
            )
            scheduler.step()
            
            # Early stopping and checkpointing
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'scaler': scaler,
                    'label_encoder': le,
                    'args': vars(args),
                    'num_params': num_params
                }
                torch.save(checkpoint, logdir / f"attention_{args.model_type}_fold{fold}_best.pt")
            else:
                patience_counter += 1
            
            # Print progress
            if args.verbose or epoch % 10 == 0 or epoch == args.epochs:
                lr_current = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{args.epochs} | LR: {lr_current:.2e} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                      f"BalAcc: {val_metrics['bal_acc']:.4f} F1: {val_metrics['f1_macro']:.4f} "
                      f"AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break
        
        print(f"\n✓ Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Load best model and evaluate
        checkpoint = torch.load(logdir / f"attention_{args.model_type}_fold{fold}_best.pt", 
                               weights_only=False)
        model.load_state_dict(checkpoint['model'])
        val_metrics = eval_epoch(
            model, val_loader, criterion, args.device, num_classes, args.model_type
        )
        
        # Save fold metrics
        val_metrics['fold'] = fold
        val_metrics['best_epoch'] = best_epoch
        val_metrics['num_params'] = num_params
        fold_metrics.append({
            'fold': fold,
            'acc': val_metrics['acc'],
            'bal_acc': val_metrics['bal_acc'],
            'f1_macro': val_metrics['f1_macro'],
            'f1_weighted': val_metrics['f1_weighted'],
            'auc': val_metrics['auc'],
            'best_epoch': best_epoch
        })
        
        # Store out-of-fold predictions
        oof_probs[val_idx] = val_metrics['probs']
        oof_labels[val_idx] = val_metrics['labels']
        
        # Print detailed classification report
        print(f"\nClassification Report (Fold {fold}):")
        print(classification_report(
            val_metrics['labels'], 
            val_metrics['preds'], 
            target_names=le.classes_,
            digits=4
        ))
    
    # Aggregate results across folds
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Compute mean, std, and 95% CI for each metric
    summary_stats = {}
    for metric in ['acc', 'bal_acc', 'f1_macro', 'f1_weighted', 'auc']:
        values = metrics_df[metric].values
        mean = np.mean(values)
        std = np.std(values)
        ci_lower, ci_upper = compute_ci95(values)
        
        summary_stats[metric] = {
            'mean': float(mean),
            'std': float(std),
            'ci95_lower': ci_lower,
            'ci95_upper': ci_upper
        }
        
        print(f"{metric.upper():15s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Overall out-of-fold metrics
    oof_preds = oof_probs.argmax(axis=1)
    oof_acc = accuracy_score(oof_labels, oof_preds)
    oof_bal_acc = balanced_accuracy_score(oof_labels, oof_preds)
    oof_f1_macro = f1_score(oof_labels, oof_preds, average='macro')
    oof_f1_weighted = f1_score(oof_labels, oof_preds, average='weighted')
    try:
        if num_classes == 2:
            oof_auc = roc_auc_score(oof_labels, oof_probs[:, 1])
        else:
            oof_auc = roc_auc_score(oof_labels, oof_probs, multi_class='ovr', average='macro')
    except:
        oof_auc = 0.0
    
    print(f"\n{'='*80}")
    print("OUT-OF-FOLD (OVERALL) METRICS")
    print(f"{'='*80}")
    print(f"{'ACCURACY':<15s}: {oof_acc:.4f}")
    print(f"{'BAL_ACCURACY':<15s}: {oof_bal_acc:.4f}")
    print(f"{'F1_MACRO':<15s}: {oof_f1_macro:.4f}")
    print(f"{'F1_WEIGHTED':<15s}: {oof_f1_weighted:.4f}")
    print(f"{'AUC':<15s}: {oof_auc:.4f}")
    
    print(f"\nOverall Classification Report:")
    print(classification_report(oof_labels, oof_preds, target_names=le.classes_, digits=4))
    
    # Save summary results
    summary = {
        'model_type': args.model_type,
        'args': vars(args),
        'num_params': num_params,
        'fold_metrics': fold_metrics,
        'summary_stats': summary_stats,
        'oof_metrics': {
            'acc': float(oof_acc),
            'bal_acc': float(oof_bal_acc),
            'f1_macro': float(oof_f1_macro),
            'f1_weighted': float(oof_f1_weighted),
            'auc': float(oof_auc)
        },
        'classes': list(le.classes_)
    }
    
    summary_file = logdir / f"attention_{args.model_type}_cv_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'patient_id': df['patient_id'].values,
        'true_label': le.inverse_transform(oof_labels.astype(int)),
        'pred_label': le.inverse_transform(oof_preds.astype(int)),
        **{f'prob_{cls}': oof_probs[:, i] for i, cls in enumerate(le.classes_)}
    })
    oof_file = oof_dir / f"attention_{args.model_type}_oof_predictions.csv"
    oof_df.to_csv(oof_file, index=False)
    
    print(f"\n✓ Results saved to {logdir}")
    print(f"  - Summary: {summary_file}")
    print(f"  - OOF predictions: {oof_file}")
    print(f"  - Model checkpoints: attention_{args.model_type}_fold*_best.pt")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
