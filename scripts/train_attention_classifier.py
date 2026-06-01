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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.advanced_attention_classifier import (
    AdvancedAttentionClassifier,
    MultiModalAttentionClassifier,
)
from models.tabular_transformer import TabularTransformerClassifier
from models.graph_classifier import GraphClassifier


def build_argparser():
    parser = argparse.ArgumentParser(description="Train Attention Classifiers")

    # Data arguments
    parser.add_argument("--features", default="meta/acdc_features.csv", help="Path to features CSV")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="advanced",
        choices=["advanced", "multimodal", "tabular_transformer", "graph"],
        help="Model architecture: 'advanced' (single input), 'multimodal' (geo+func), 'tabular_transformer' (grouped tokens), or 'graph' (GAT over tokens)",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of attention blocks")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--tt-d-model", type=int, default=256, help="Transformer width for tabular model")
    parser.add_argument("--tt-depth", type=int, default=4, help="Transformer layers for tabular model")
    parser.add_argument("--tt-heads", type=int, default=8, help="Attention heads for tabular model")
    parser.add_argument("--tt-dropout", type=float, default=0.3, help="Dropout for tabular model")
    # Graph model
    parser.add_argument("--graph-hidden", type=int, default=128, help="Hidden dim for graph classifier")
    parser.add_argument("--graph-heads", type=int, default=4, help="Attention heads for graph classifier")
    parser.add_argument("--graph-layers", type=int, default=2, help="GAT layers for graph classifier")
    parser.add_argument("--graph-dropout", type=float, default=0.2, help="Dropout for graph classifier")

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

    return parser


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


def build_feature_groups(features_df, valid_features=None):
    """
    Build feature groups (tokens) from a DataFrame.
    Group heuristics: LV*, RV*, EF*, MYO*, axis/ratio, and remaining numeric columns.
    If valid_features is provided, only use those features.
    """
    if features_df is None or features_df.empty:
        raise ValueError("features_df is empty; cannot build feature groups.")

    numeric_cols = [
        c for c in features_df.columns
        if c not in {"patient_id", "label", "label_enc"} and
        pd.api.types.is_numeric_dtype(features_df[c])
    ]
    if valid_features:
        numeric_cols = [c for c in numeric_cols if c in valid_features]
    
    if not numeric_cols:
        raise ValueError("No numeric feature columns found to build feature groups.")
    remaining = set(numeric_cols)
    groups = {}

    def take(name, predicate):
        cols = sorted([c for c in list(remaining) if predicate(c)])
        if cols:
            groups[name] = cols
            remaining.difference_update(cols)

    take("lv", lambda c: c.upper().startswith("LV"))
    take("rv", lambda c: c.upper().startswith("RV"))
    take("ef", lambda c: "EF" in c.upper())
    take("myo", lambda c: c.upper().startswith("MYO"))
    take("shape", lambda c: ("axis" in c.lower()) or ("ratio" in c.lower()))
    if remaining:
        groups["other"] = sorted(remaining)
    if not groups:
        raise ValueError("No numeric feature groups found for tokenization.")
    return groups


class TokenizedFeatureDataset(Dataset):
    """
    Dataset that packs feature groups into fixed-length tokens for transformer/GAT models.
    """

    def __init__(self, features_df, group_map, stats=None, fit_stats=False):
        self.group_map = group_map
        self.group_order = list(group_map.keys())
        self.max_dim = max(len(cols) for cols in group_map.values())
        self.labels = features_df["label_enc"].values

        # Prepare feature arrays
        self._raw = {
            name: features_df[cols].to_numpy(dtype=np.float32)
            for name, cols in group_map.items()
        }

        if fit_stats or stats is None:
            self.stats = {name: self._compute_stats(arr) for name, arr in self._raw.items()}
        else:
            self.stats = stats

        self.features = {
            name: self._standardize(self._raw[name], self.stats[name])
            for name in self.group_order
        }

    @staticmethod
    def _compute_stats(arr):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where(std < 1e-6, 1.0, std)
        std = np.where(np.isnan(std), 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _standardize(arr, stats):
        mean, std = stats
        arr = np.where(np.isnan(arr), mean, arr)
        return (arr - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = []
        for name in self.group_order:
            vec = self.features[name][idx]
            pad = np.zeros(self.max_dim, dtype=np.float32)
            pad[: vec.shape[0]] = vec
            tokens.append(pad)
        tokens = torch.from_numpy(np.stack(tokens, axis=0))
        return {"tokens": tokens, "label": int(self.labels[idx])}

    def get_stats(self):
        return self.stats

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
        elif model_type == 'multimodal':
            geo = batch['geometric'].to(device)
            func = batch['functional'].to(device)
            logits = model(geo, func)
        elif model_type == 'tabular_transformer':
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
        elif model_type == 'graph':
            tokens = batch['tokens'].to(device)  # (B, T, D)
            logits = model(tokens)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
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
        elif model_type == 'multimodal':
            geo = batch['geometric'].to(device)
            func = batch['functional'].to(device)
            logits = model(geo, func)
        elif model_type == 'tabular_transformer':
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
        elif model_type == 'graph':
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
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


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)
    
    # Set random seeds
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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

    # If label column is missing, try merging from a labels CSV
    if "label" not in df.columns:
        default_labels_path = Path("results") / "acdc_labels.csv"
        labels_path = default_labels_path if default_labels_path.exists() else None
        # Allow override via --labels argument if added; fallback to default path
        # Backward-compatible: infer labels if available
        try:
            # Try to find labels file near features
            if labels_path is None:
                candidate = Path(args.features).with_name("acdc_labels.csv")
                if candidate.exists():
                    labels_path = candidate
            if labels_path is not None:
                labels_df = pd.read_csv(labels_path)
                # Expect columns: patient_id, diagnosis
                if "patient_id" in df.columns and {"patient_id","diagnosis"}.issubset(labels_df.columns):
                    df = df.merge(labels_df[["patient_id","diagnosis"]], on="patient_id", how="left")
                    df = df.rename(columns={"diagnosis":"label"})
                else:
                    print("Warning: Could not merge labels; expected columns 'patient_id' and 'diagnosis'.")
            else:
                print("Warning: 'label' column not found and no labels CSV detected.")
        except Exception as e:
            print(f"Warning: Failed to merge labels automatically: {e}")

    # Final check for labels
    if "label" not in df.columns:
        raise KeyError("'label' column missing. Provide a labels CSV (results/acdc_labels.csv) with columns ['patient_id','diagnosis'] or include 'label' in features CSV.")

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    print(f"\nDataset: {len(df)} patients")
    
    # Define feature groups - filter out features with too many NaNs
    ef_cols = [c for c in df.columns if 'EF' in c and df[c].notna().sum() > len(df) * 0.3]  # At least 30% non-NaN
    geo_cols = [c for c in df.columns if (c.endswith('_ml') or c.endswith('_mL') or c.endswith('_mm') or 'axis' in c or 'ratio' in c) and df[c].notna().sum() > len(df) * 0.3]
    all_feat_cols = geo_cols + ef_cols
    
    group_map = None
    if args.model_type == 'advanced':
        print(f"Features ({len(all_feat_cols)}): {all_feat_cols}")
    elif args.model_type == 'multimodal':
        print(f"Geometric features ({len(geo_cols)}): {geo_cols}")
        print(f"Functional features ({len(ef_cols)}): {ef_cols}")
    else:
        # group_map will be built per fold from valid features
        pass
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(df["label"].astype(str))
    df["label_enc"] = labels_encoded
    
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")
    
    # Cross-validation setup — GroupKFold ensures no patient leaks across folds
    gkf = GroupKFold(n_splits=args.folds)
    groups = df["patient_id"].values

    fold_metrics = []
    oof_probs = np.zeros((len(df), num_classes))  # Out-of-fold predictions
    oof_labels = np.zeros(len(df))

    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)

    # Save out-of-fold predictions directory
    oof_dir = logdir / "oof_preds"
    oof_dir.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["label_enc"], groups), 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold}/{args.folds}")
        print(f"{'='*80}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Apply SMOTE for class balancing on training data
        print(f"Original training class distribution: {Counter(train_df['label_enc'])}")
        
        # Prepare features for SMOTE (use only valid features)
        valid_feat_cols = [col for col in all_feat_cols if col in train_df.columns and train_df[col].notna().any()]
        print(f"Using {len(valid_feat_cols)} valid features for SMOTE: {valid_feat_cols}")
        
        # Build group_map from valid features only
        if args.model_type in ['tabular_transformer', 'graph']:
            group_map = build_feature_groups(train_df, valid_features=valid_feat_cols)
            print(f"Grouped tokens ({sum(len(v) for v in group_map.values())} features total):")
            for name, cols in group_map.items():
                print(f"  - {name}: {cols}")
        
        X_train = train_df[valid_feat_cols].values
        y_train = train_df['label_enc'].values
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)

        # Fit scaler on real patients BEFORE SMOTE so synthetic samples don't bias statistics
        pre_smote_scaler = StandardScaler().fit(X_train_imputed)

        # Apply SMOTE
        min_samples = min(Counter(y_train).values())
        k_neighbors = min(5, max(1, min_samples - 1))
        smote = SMOTE(random_state=args.seed, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_imputed, y_train)

        # Create balanced training dataframe
        train_df_balanced = pd.DataFrame(X_train_balanced, columns=valid_feat_cols)
        train_df_balanced['label_enc'] = y_train_balanced
        train_df_balanced['patient_id'] = [f"smote_{i}" for i in range(len(train_df_balanced))]

        print(f"Balanced training class distribution: {Counter(y_train_balanced)}")
        print(f"Train: {len(train_df_balanced)} patients (after SMOTE) | Val: {len(val_df)} patients")

        # Create datasets and loaders — pass pre-SMOTE scaler so transform uses real-patient stats
        if args.model_type == 'advanced':
            train_dataset = SingleInputDataset(train_df_balanced, all_feat_cols,
                                               scaler=pre_smote_scaler, fit_scaler=False)
            scaler = train_dataset.get_scaler()
            val_dataset = SingleInputDataset(val_df, all_feat_cols, scaler=scaler)
            model = AdvancedAttentionClassifier(
                input_features=len(all_feat_cols),
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout
            ).to(args.device)
        elif args.model_type == 'multimodal':
            # Fit separate geo/func scalers on pre-SMOTE data
            pre_smote_scaler_mm = {
                'geo':  StandardScaler().fit(train_df[geo_cols].values.astype(np.float32)),
                'func': StandardScaler().fit(train_df[ef_cols].values.astype(np.float32)),
            }
            train_dataset = MultiModalDataset(train_df_balanced, geo_cols, ef_cols,
                                              scaler=pre_smote_scaler_mm, fit_scaler=False)
            scaler = train_dataset.get_scalers()
            val_dataset = MultiModalDataset(val_df, geo_cols, ef_cols, scaler=scaler)
            model = MultiModalAttentionClassifier(
                num_geometric=len(geo_cols),
                num_functional=len(ef_cols),
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                dropout=args.dropout
            ).to(args.device)
        elif args.model_type == 'tabular_transformer':
            train_dataset = TokenizedFeatureDataset(train_df_balanced, group_map, fit_stats=True)
            stats = train_dataset.get_stats()
            val_dataset = TokenizedFeatureDataset(val_df, group_map, stats=stats)
            scaler = stats  # keep naming consistent in checkpoint
            model = TabularTransformerClassifier(
                token_dim=train_dataset.max_dim,
                num_tokens=len(train_dataset.group_order),
                num_classes=num_classes,
                d_model=args.tt_d_model,
                nhead=args.tt_heads,
                depth=args.tt_depth,
                dropout=args.tt_dropout
            ).to(args.device)
        else:  # graph
            train_dataset = TokenizedFeatureDataset(train_df_balanced, group_map, fit_stats=True)
            stats = train_dataset.get_stats()
            val_dataset = TokenizedFeatureDataset(val_df, group_map, stats=stats)
            scaler = stats
            # Flatten tokens to a single vector per sample for the baseline GraphClassifier
            # We will wrap the DataLoader to reshape tokens before model forward.
            model = GraphClassifier(
                token_dim=train_dataset.max_dim,
                num_tokens=len(train_dataset.group_order),
                num_classes=num_classes,
                hidden_dim=args.graph_hidden,
                heads=args.graph_heads,
                layers=args.graph_layers,
                dropout=args.graph_dropout
            ).to(args.device)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
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
