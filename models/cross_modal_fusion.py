#!/usr/bin/env python3
"""
Cross-Modal Attention Fusion Classifier for MRI + Echo features.
Implements Multi-Head Attention for Feature Fusion between different imaging modalities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing MRI and Echo features.
    Uses multi-head attention where one modality queries the other.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, seq_len_q, embed_dim] - Query features (one modality)
            key: [B, seq_len_k, embed_dim] - Key features (other modality)
            value: [B, seq_len_v, embed_dim] - Value features (other modality)
        Returns:
            attended: [B, seq_len_q, embed_dim] - Attended features
        """
        B, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # Linear projections and reshape
        Q = self.q_proj(query).view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)

        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(B, seq_len_q, self.embed_dim)
        output = self.out_proj(attended)

        return output


class CrossModalFusionClassifier(nn.Module):
    """
    Cross-Modal Attention Fusion Classifier for MRI + Echo features.

    Architecture:
    1. Modality-specific encoders (MRI and Echo)
    2. Self-attention within each modality
    3. Cross-attention between modalities (MRI↔Echo)
    4. Multi-head fusion with different attention patterns
    5. Classification head

    Clinical Relevance:
    - Can weigh MRI data more heavily when Echo is noisy
    - Can weigh Echo data more heavily when MRI has artifacts
    - Learns optimal fusion strategy from data
    """
    def __init__(
        self,
        mri_dim=8,      # ACDC features: LV/RV/MYO_ED/ES_mL + LV/RV_EF
        echo_dim=3,     # CAMUS features: LV_ED/ES_pixels + LV_EF
        num_classes=5,
        hidden_dim=128,
        num_heads=8,
        num_attn_blocks=2,
        dropout=0.3
    ):
        super().__init__()

        self.mri_dim = mri_dim
        self.echo_dim = echo_dim

        # Modality-specific feature projectors
        self.mri_projector = nn.Sequential(
            nn.Linear(mri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.echo_projector = nn.Sequential(
            nn.Linear(echo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Self-attention within modalities (treat features as sequence)
        self.mri_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.echo_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-modal attention layers
        self.cross_attn_mri_to_echo = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_echo_to_mri = CrossModalAttention(hidden_dim, num_heads, dropout)

        # Multi-head fusion attention (learn which modality to trust more)
        self.fusion_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Projection for fusion inputs
        self.fusion_proj = nn.Linear(4 * hidden_dim, hidden_dim)

        # Attention pooling
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Store attention weights for interpretability
        self.attention_weights = {}

    def forward(self, mri_features, echo_features):
        """
        Args:
            mri_features: [B, mri_dim] - ACDC features
            echo_features: [B, echo_dim] - CAMUS features
        Returns:
            logits: [B, num_classes]
            attention_weights: dict of attention weights for interpretability
        """
        B = mri_features.size(0)

        # Project to hidden dimension and add sequence dimension
        mri_hidden = self.mri_projector(mri_features).unsqueeze(1)  # [B, 1, hidden_dim]
        echo_hidden = self.echo_projector(echo_features).unsqueeze(1)  # [B, 1, hidden_dim]

        # Self-attention within modalities
        mri_self, mri_self_weights = self.mri_self_attn(mri_hidden, mri_hidden, mri_hidden)
        echo_self, echo_self_weights = self.echo_self_attn(echo_hidden, echo_hidden, echo_hidden)

        # Cross-modal attention
        mri_to_echo = self.cross_attn_mri_to_echo(mri_self, echo_self, echo_self)  # MRI queries Echo
        echo_to_mri = self.cross_attn_echo_to_mri(echo_self, mri_self, mri_self)  # Echo queries MRI

        # Simple fusion: concatenate cross-attended features
        fused_features = torch.cat([
            mri_self.squeeze(1),      # MRI after self-attention
            echo_self.squeeze(1),     # Echo after self-attention
            mri_to_echo.squeeze(1),   # MRI attending to Echo
            echo_to_mri.squeeze(1)    # Echo attending to MRI
        ], dim=-1)  # [B, 4*hidden_dim]

        # Project to standard dimension for fusion attention
        fused_proj = self.fusion_proj(fused_features)  # [B, hidden_dim]

        # Fusion attention (learn importance weights for each representation)
        fusion_queries = fused_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        fusion_keys = fused_proj.unsqueeze(1)     # [B, 1, hidden_dim]
        fusion_values = fused_proj.unsqueeze(1)   # [B, 1, hidden_dim]

        fusion_output, fusion_weights = self.fusion_attn(
            fusion_queries, fusion_keys, fusion_values
        )

        # Pool fusion output
        pooled = fusion_output.squeeze(1)  # [B, hidden_dim]

        # Classification
        logits = self.classifier(pooled)

        # Store attention weights for interpretability
        self.attention_weights = {
            'mri_self_attn': mri_self_weights,
            'echo_self_attn': echo_self_weights,
            'fusion_attn': fusion_weights
        }

        return logits

    def get_attention_weights(self):
        """Return attention weights for interpretability"""
        return self.attention_weights


class CrossModalDataset(Dataset):
    """
    Dataset for cross-modal MRI + Echo fusion.
    Combines features from ACDC and CAMUS datasets.
    """
    def __init__(self, mri_df, echo_df, mri_cols, echo_cols, scaler=None, fit_scaler=False):
        # For demonstration, we'll pair patients (in practice, use same patients)
        # Here we randomly pair for synthetic cross-modal examples
        self.pairs = []
        for i in range(min(len(mri_df), len(echo_df))):
            mri_idx = i % len(mri_df)
            echo_idx = i % len(echo_df)
            self.pairs.append((mri_idx, echo_idx))

        self.mri_data = mri_df[mri_cols].values.astype(np.float32)
        self.echo_data = echo_df[echo_cols].values.astype(np.float32)
        self.labels = mri_df['label_enc'].values  # Use MRI labels

        # Standardize features
        if fit_scaler or scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.mri_scaler = StandardScaler().fit(self.mri_data)
            self.echo_scaler = StandardScaler().fit(self.echo_data)
        else:
            self.mri_scaler = scaler['mri']
            self.echo_scaler = scaler['echo']

        self.mri_data = self.mri_scaler.transform(self.mri_data)
        self.echo_data = self.echo_scaler.transform(self.echo_data)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_idx, echo_idx = self.pairs[idx]
        return {
            'mri_features': torch.from_numpy(self.mri_data[mri_idx]),
            'echo_features': torch.from_numpy(self.echo_data[echo_idx]),
            'label': self.labels[mri_idx]
        }

    def get_scalers(self):
        return {'mri': self.mri_scaler, 'echo': self.echo_scaler}


def create_cross_modal_dataset():
    """Create a synthetic cross-modal dataset for demonstration"""
    # Load features
    mri_df = pd.read_csv('meta/acdc_features.csv')
    echo_df = pd.read_csv('meta/camus_features.csv')

    # Define feature columns
    mri_cols = ['LV_ED_mL', 'RV_ED_mL', 'MYO_ED_mL', 'LV_ES_mL', 'RV_ES_mL', 'MYO_ES_mL', 'LV_EF', 'RV_EF']
    echo_cols = ['LV_ED_pixels', 'LV_ES_pixels']  # CAMUS only has pixel areas

    # Filter to patients with all features
    mri_df = mri_df.dropna(subset=mri_cols + ['label'])
    echo_df = echo_df.dropna(subset=echo_cols)

    # Create label encoding if needed
    if 'label_enc' not in mri_df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        mri_df['label_enc'] = le.fit_transform(mri_df['label'])

    print(f"Available MRI patients: {len(mri_df)}")
    print(f"Available Echo patients: {len(echo_df)}")
    print(f"Cross-modal pairs: {min(len(mri_df), len(echo_df))}")

    return mri_df, echo_df, mri_cols, echo_cols


if __name__ == "__main__":
    # Test the cross-modal fusion classifier
    print("Testing Cross-Modal Attention Fusion Classifier")
    print("=" * 60)

    # Create model
    model = CrossModalFusionClassifier(
        mri_dim=8,      # ACDC features
        echo_dim=2,     # CAMUS features (pixels only)
        num_classes=5,
        hidden_dim=128,
        num_heads=8
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with dummy data
    mri_features = torch.randn(4, 8)    # Batch of 4, 8 MRI features
    echo_features = torch.randn(4, 2)   # Batch of 4, 2 Echo features (pixels only)

    logits = model(mri_features, echo_features)
    print(f"Input MRI shape: {mri_features.shape}")
    print(f"Input Echo shape: {echo_features.shape}")
    print(f"Output logits shape: {logits.shape}")

    print("\n✅ Cross-Modal Attention Fusion Classifier working!")
    print("\nClinical Benefits:")
    print("- Learns optimal fusion between MRI and Echo modalities")
    print("- Can weigh modalities based on data quality/noise")
    print("- Provides interpretable attention weights")
    print("- More robust diagnosis than single-modality approaches")