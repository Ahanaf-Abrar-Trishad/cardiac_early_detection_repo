# filepath: /home/ahanaf/cardiac_early_detection_repo/scripts/train_fusion_classifier.py
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
from torch.utils.data import Dataset, DataLoader, Subset

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.fusion_classifier import FeatureFusionClassifier


class FeatureDataset(Dataset):
    """Dataset for tabular features."""
    def __init__(self, features_df, geo_cols, ef_cols, scaler=None, fit_scaler=False):
        self.patient_ids = features_df["patient_id"].values
        self.labels = features_df["label"].values
        
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
    skf = StratifiedKFol# filepath: /home/ahanaf/cardiac_early_detection_repo/scripts/train_fusion_classifier.py
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
from torch.utils.data import Dataset, DataLoader, Subset

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.fusion_classifier import FeatureFusionClassifier


class FeatureDataset(Dataset):
    """Dataset for tabular features."""
    def __init__(self, features_df, geo_cols, ef_cols, scaler=None, fit_scaler=False):
        self.patient_ids = features_df["patient_id"].values
        self.labels = features_df["label"].values
        
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
    skf = StratifiedKFol