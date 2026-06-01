#!/usr/bin/env python3
"""
Training script for Cross-Modal Attention Fusion Classifier.
Trains the model to fuse MRI and Echo features using multi-head attention.
Uses 5-fold GroupKFold cross-validation keyed on patient_id to prevent leakage.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from models.cross_modal_fusion import CrossModalFusionClassifier, CrossModalDataset, create_cross_modal_dataset

ACDC_CLASSES = ['NOR', 'DCM', 'HCM', 'MINF', 'RV']


def _train_one_fold(model, train_loader, val_loader, device, num_epochs, learning_rate,
                    weight_decay, patience, save_path, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            mri_f = batch['mri_features'].to(device)
            echo_f = batch['echo_features'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            out = model(mri_f, echo_f)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += out.argmax(1).eq(labels).sum().item()
            train_total += labels.size(0)
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                mri_f = batch['mri_features'].to(device)
                echo_f = batch['echo_features'].to(device)
                labels = batch['label'].to(device)
                out = model(mri_f, echo_f)
                val_loss += criterion(out, labels).item()
                val_correct += out.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


def train_cross_modal_fusion(
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=10,
    n_folds=5,
    save_path='logs/cross_modal_fusion'
):
    """
    Train the cross-modal attention fusion classifier with n-fold GroupKFold CV.
    GroupKFold ensures no patient leaks across train/val splits.
    """
    print("Training Cross-Modal Attention Fusion Classifier")
    print("=" * 60)

    import random as _random
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Loading cross-modal datasets...")
    mri_df, echo_df, mri_cols, echo_cols = create_cross_modal_dataset()

    if 'patient_id' not in mri_df.columns:
        raise KeyError("mri_df must have a 'patient_id' column for GroupKFold.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_classes = mri_df['label_enc'].nunique()
    groups = mri_df['patient_id'].values
    y = mri_df['label_enc'].values

    gkf = GroupKFold(n_splits=n_folds)

    fold_metrics = []
    oof_probs = np.zeros((len(mri_df), num_classes))
    oof_labels = np.zeros(len(mri_df), dtype=int)
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(mri_df, y, groups), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds} | train={len(train_idx)} val={len(val_idx)}")
        assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0, \
            "Patient leakage detected — patient appears in both train and val!"

        train_mri = mri_df.iloc[train_idx].reset_index(drop=True)
        val_mri   = mri_df.iloc[val_idx].reset_index(drop=True)

        train_ds = CrossModalDataset(train_mri, echo_df, mri_cols, echo_cols, fit_scaler=True)
        val_ds   = CrossModalDataset(val_mri,   echo_df, mri_cols, echo_cols,
                                     scaler=train_ds.get_scalers(), fit_scaler=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        model = CrossModalFusionClassifier(
            mri_dim=len(mri_cols),
            echo_dim=len(echo_cols),
            num_classes=num_classes,
            hidden_dim=128,
            num_heads=8,
            dropout=0.3
        ).to(device)

        model, history = _train_one_fold(
            model, train_loader, val_loader, device,
            num_epochs, learning_rate, weight_decay, patience,
            save_path, fold
        )
        all_histories.append(history)

        torch.save({
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'scalers': train_ds.get_scalers(),
        }, save_path / f'fold{fold}_best.pt')

        # Collect OOF predictions
        model.eval()
        fold_preds, fold_labels, fold_probs_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                mri_f  = batch['mri_features'].to(device)
                echo_f = batch['echo_features'].to(device)
                lbls   = batch['label'].to(device)
                out    = model(mri_f, echo_f)
                probs  = torch.softmax(out, dim=1).cpu().numpy()
                fold_probs_list.append(probs)
                fold_preds.extend(out.argmax(1).cpu().numpy())
                fold_labels.extend(lbls.cpu().numpy())

        fold_probs_arr = np.vstack(fold_probs_list)
        oof_probs[val_idx]  = fold_probs_arr
        oof_labels[val_idx] = fold_labels

        acc  = accuracy_score(fold_labels, fold_preds)
        bacc = balanced_accuracy_score(fold_labels, fold_preds)
        mf1  = f1_score(fold_labels, fold_preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(fold_labels, fold_probs_arr,
                                multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')
        fold_metrics.append({'fold': fold, 'acc': acc, 'bacc': bacc, 'f1_macro': mf1, 'auc': auc})
        print(f"  Fold {fold} | Acc={acc:.3f} BalAcc={bacc:.3f} F1={mf1:.3f} AUC={auc:.3f}")

    # Aggregate OOF results
    print(f"\n{'='*60}")
    print("Cross-Validation Summary (OOF):")
    metrics_df = pd.DataFrame(fold_metrics)
    for col in ['acc', 'bacc', 'f1_macro', 'auc']:
        print(f"  {col:10s}: {metrics_df[col].mean():.3f} ± {metrics_df[col].std():.3f}")

    metrics_df.to_csv(save_path / 'cv_metrics.csv', index=False)

    # Full OOF classification report with correct ACDC class names
    oof_preds = oof_probs.argmax(axis=1)
    print("\nOOF Classification Report:")
    print(classification_report(oof_labels, oof_preds,
                                target_names=ACDC_CLASSES[:num_classes]))

    # OOF confusion matrix
    cm = confusion_matrix(oof_labels, oof_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACDC_CLASSES[:num_classes],
                yticklabels=ACDC_CLASSES[:num_classes])
    plt.title('Cross-Modal Fusion — OOF Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'oof_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    return metrics_df, all_histories


if __name__ == "__main__":
    metrics_df, histories = train_cross_modal_fusion(
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        n_folds=5,
        save_path='logs/cross_modal_fusion'
    )
    print("\nCross-Modal Attention Fusion training completed.")
    print("Fold checkpoints: logs/cross_modal_fusion/fold*_best.pt")
    print("CV metrics:       logs/cross_modal_fusion/cv_metrics.csv")
    print("OOF confusion:    logs/cross_modal_fusion/oof_confusion_matrix.png")
