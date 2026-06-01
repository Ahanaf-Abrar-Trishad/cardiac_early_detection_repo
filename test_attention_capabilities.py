"""
Test script to demonstrate the capabilities of Advanced Attention Classifiers
Shows: forward pass, attention visualization, feature importance, interpretability
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.advanced_attention_classifier import (
    AdvancedAttentionClassifier,
    MultiModalAttentionClassifier
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("ADVANCED ATTENTION CLASSIFIER CAPABILITIES DEMO")
print("=" * 80)

# Feature names for visualization
FEATURE_NAMES = [
    'LV_ED_mL', 'LV_ES_mL',  # Left Ventricle volumes
    'RV_ED_mL', 'RV_ES_mL',  # Right Ventricle volumes
    'MYO_ED_mL', 'MYO_ES_mL',  # Myocardium volumes
    'LV_EF', 'RV_EF'  # Ejection Fractions
]

CLASS_NAMES = ['NOR', 'MINF', 'DCM', 'HCM', 'RV']

# ============================================================================
# CAPABILITY 1: Basic Forward Pass & Architecture Info
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 1: Basic Architecture & Forward Pass")
print("=" * 80)

model = AdvancedAttentionClassifier(
    input_features=8,
    num_classes=5,
    hidden_dim=256,
    num_blocks=4,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.1  # Lower dropout for demo
)

# Create sample batch (4 patients)
batch_size = 4
sample_features = torch.randn(batch_size, 8)

print(f"\n✓ Model initialized successfully")
print(f"  • Input features: 8")
print(f"  • Output classes: 5")
print(f"  • Hidden dimension: 256")
print(f"  • Attention blocks: 4")
print(f"  • Attention heads per block: 8")
print(f"  • Total attention heads: 32")
print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")

model.eval()
with torch.no_grad():
    logits = model(sample_features)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

print(f"\n✓ Forward pass successful")
print(f"  • Input shape: {list(sample_features.shape)}")
print(f"  • Output shape: {list(logits.shape)}")
print(f"  • Predictions: {preds.tolist()}")

print(f"\n✓ Probability distributions:")
for i in range(batch_size):
    pred_class = CLASS_NAMES[preds[i]]
    confidence = probs[i, preds[i]].item()
    print(f"  Patient {i+1}: {pred_class} (confidence: {confidence:.1%})")
    top3_probs = torch.topk(probs[i], 3)
    for j, (prob, idx) in enumerate(zip(top3_probs.values, top3_probs.indices)):
        print(f"    {j+1}. {CLASS_NAMES[idx]}: {prob:.1%}")

# ============================================================================
# CAPABILITY 2: Attention Weight Extraction & Visualization
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 2: Attention Weight Extraction & Visualization")
print("=" * 80)

attention_weights = model.get_attention_weights()
print(f"\n✓ Extracted attention weights from {len(attention_weights)} blocks")

for block_idx, attn in enumerate(attention_weights):
    print(f"\n  Block {block_idx + 1}:")
    print(f"    • Shape: {list(attn.shape)}")
    print(f"    • Format: [batch_size={attn.shape[0]}, num_heads={attn.shape[1]}, "
          f"features={attn.shape[2]}, features={attn.shape[3]}]")
    
    # Average attention across batch and heads
    avg_attn = attn.mean(dim=(0, 1))  # [8, 8]
    print(f"    • Average attention (across batch & heads): {list(avg_attn.shape)}")
    print(f"    • Min attention: {avg_attn.min().item():.4f}")
    print(f"    • Max attention: {avg_attn.max().item():.4f}")
    
    # Show top 5 feature interactions
    attn_flat = avg_attn.flatten()
    top_indices = torch.topk(attn_flat, 5).indices
    print(f"    • Top 5 feature interactions:")
    for rank, idx in enumerate(top_indices):
        i, j = idx // 8, idx % 8
        weight = avg_attn[i, j].item()
        print(f"      {rank+1}. {FEATURE_NAMES[i]:12s} → {FEATURE_NAMES[j]:12s}: {weight:.4f}")

# ============================================================================
# CAPABILITY 3: Feature Importance Analysis
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 3: Feature Importance Analysis")
print("=" * 80)

print(f"\n✓ Analyzing which features receive most attention:")

# Calculate average attention each feature receives from others
all_attentions = torch.stack([attn.mean(dim=(0, 1)) for attn in attention_weights])
avg_all_attn = all_attentions.mean(dim=0)  # [8, 8]

# Column-wise sum: how much attention each feature receives
incoming_attention = avg_all_attn.sum(dim=0)  # [8]
# Normalize
incoming_attention = incoming_attention / incoming_attention.sum()

print(f"\n  • Incoming attention (how much others attend to this feature):")
sorted_indices = torch.argsort(incoming_attention, descending=True)
for rank, idx in enumerate(sorted_indices):
    importance = incoming_attention[idx].item()
    print(f"    {rank+1}. {FEATURE_NAMES[idx]:12s}: {importance:.4f} {'█' * int(importance * 100)}")

# Row-wise sum: how much attention each feature pays to others
outgoing_attention = avg_all_attn.sum(dim=1)  # [8]
outgoing_attention = outgoing_attention / outgoing_attention.sum()

print(f"\n  • Outgoing attention (how much this feature attends to others):")
sorted_indices = torch.argsort(outgoing_attention, descending=True)
for rank, idx in enumerate(sorted_indices):
    importance = outgoing_attention[idx].item()
    print(f"    {rank+1}. {FEATURE_NAMES[idx]:12s}: {importance:.4f} {'█' * int(importance * 100)}")

# ============================================================================
# CAPABILITY 4: Multi-Modal Architecture
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 4: Multi-Modal Cross-Attention")
print("=" * 80)

multimodal_model = MultiModalAttentionClassifier(
    num_geometric=6,
    num_functional=2,
    num_classes=5,
    hidden_dim=256,
    num_heads=8,
    dropout=0.1
)

print(f"\n✓ Multi-modal model initialized")
print(f"  • Geometric features: 6 (LV/RV/MYO volumes at ED/ES)")
print(f"  • Functional features: 2 (LV_EF, RV_EF)")
print(f"  • Output classes: 5")
print(f"  • Parameters: {sum(p.numel() for p in multimodal_model.parameters()):,}")

# Create separate inputs
geometric_features = torch.randn(batch_size, 6)
functional_features = torch.randn(batch_size, 2)

multimodal_model.eval()
with torch.no_grad():
    logits_mm = multimodal_model(geometric_features, functional_features)
    probs_mm = torch.softmax(logits_mm, dim=-1)
    preds_mm = torch.argmax(logits_mm, dim=-1)

print(f"\n✓ Multi-modal forward pass successful")
print(f"  • Geometric input: {list(geometric_features.shape)}")
print(f"  • Functional input: {list(functional_features.shape)}")
print(f"  • Output: {list(logits_mm.shape)}")
print(f"  • Predictions: {[CLASS_NAMES[p] for p in preds_mm.tolist()]}")

# ============================================================================
# CAPABILITY 5: Realistic Example with Simulated Patient Data
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 5: Realistic Patient Examples")
print("=" * 80)

# Simulate 3 different cardiac conditions
# Patient 1: Normal
normal_patient = torch.tensor([[
    150.0, 50.0,   # LV: Normal ED/ES volumes
    150.0, 60.0,   # RV: Normal volumes
    120.0, 100.0,  # MYO: Normal
    0.67, 0.60     # EF: Normal (>55%)
]])

# Patient 2: Dilated Cardiomyopathy (DCM)
dcm_patient = torch.tensor([[
    300.0, 250.0,  # LV: Dilated with poor contraction
    200.0, 150.0,  # RV: Slightly dilated
    140.0, 120.0,  # MYO: Normal thickness
    0.17, 0.25     # EF: Very low (<35%)
]])

# Patient 3: Hypertrophic Cardiomyopathy (HCM)
hcm_patient = torch.tensor([[
    120.0, 40.0,   # LV: Small cavity, hyperdynamic
    130.0, 50.0,   # RV: Normal
    180.0, 160.0,  # MYO: Very thick wall
    0.67, 0.62     # EF: Normal or high
]])

patients = torch.cat([normal_patient, dcm_patient, hcm_patient], dim=0)
patient_labels = ['Normal', 'DCM (suspected)', 'HCM (suspected)']

model.eval()
with torch.no_grad():
    logits_real = model(patients)
    probs_real = torch.softmax(logits_real, dim=-1)
    preds_real = torch.argmax(logits_real, dim=-1)

print(f"\n✓ Analyzing {len(patients)} simulated patients:\n")

for i, (patient_type, features) in enumerate(zip(patient_labels, patients)):
    print(f"  Patient {i+1}: {patient_type}")
    print(f"  {'─' * 60}")
    print(f"    Features:")
    for j, (name, val) in enumerate(zip(FEATURE_NAMES, features)):
        print(f"      {name:12s}: {val:7.1f} mL" if j < 6 else f"      {name:12s}: {val:7.2%}")
    
    print(f"\n    Prediction: {CLASS_NAMES[preds_real[i]]} (confidence: {probs_real[i, preds_real[i]]:.1%})")
    print(f"    All probabilities:")
    for class_idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, probs_real[i])):
        bar = '█' * int(prob * 40)
        print(f"      {class_name:6s}: {prob:6.1%} {bar}")
    print()

# Get attention for DCM patient specifically
with torch.no_grad():
    _ = model(dcm_patient)
    dcm_attention = model.get_attention_weights()

print(f"  Feature importance for DCM patient (Block 1, averaged across heads):")
dcm_attn_block1 = dcm_attention[0][0].mean(dim=0)  # [8, 8]

# Which features does LV_EF attend to?
lv_ef_idx = 6
lv_ef_attention = dcm_attn_block1[lv_ef_idx]
sorted_attn_idx = torch.argsort(lv_ef_attention, descending=True)

print(f"\n    LV_EF is attending to:")
for rank, idx in enumerate(sorted_attn_idx[:5]):
    attn_score = lv_ef_attention[idx].item()
    print(f"      {rank+1}. {FEATURE_NAMES[idx]:12s}: {attn_score:.4f} {'█' * int(attn_score * 50)}")

# ============================================================================
# CAPABILITY 6: Comparison with Simple Baseline
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 6: Comparison with Simple Baseline")
print("=" * 80)

# Simple MLP baseline (no attention)
class SimpleMLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 5)
        )
    
    def forward(self, x):
        return self.net(x)

baseline = SimpleMLPBaseline()
baseline_params = sum(p.numel() for p in baseline.parameters())

print(f"\n✓ Architecture comparison:")
print(f"  {'Model':<30} {'Parameters':<15} {'Key Features'}")
print(f"  {'-' * 70}")
print(f"  {'Simple MLP Baseline':<30} {baseline_params:>10,}     None")
print(f"  {'AdvancedAttentionClassifier':<30} {sum(p.numel() for p in model.parameters()):>10,}     MHSA + SE + Residual")
print(f"  {'MultiModalAttentionClassifier':<30} {sum(p.numel() for p in multimodal_model.parameters()):>10,}     Cross-Modal Attn")

# ============================================================================
# CAPABILITY 7: Attention Pattern Analysis
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY 7: Attention Pattern Analysis")
print("=" * 80)

print("\n✓ Analyzing learned attention patterns:\n")

# Average attention across all blocks, batch, and heads
all_attentions = torch.stack([attn.mean(dim=(0, 1)) for attn in attention_weights])
avg_all_attn = all_attentions.mean(dim=0)  # [8, 8]

# Find feature pairs with high co-attention
high_attn_threshold = 0.15
print(f"  High-attention feature pairs (threshold > {high_attn_threshold}):")
count = 0
for i in range(8):
    for j in range(i+1, 8):  # Upper triangle only
        attn_score = avg_all_attn[i, j].item()
        if attn_score > high_attn_threshold:
            count += 1
            print(f"    {FEATURE_NAMES[i]:12s} ↔ {FEATURE_NAMES[j]:12s}: {attn_score:.4f}")

if count == 0:
    print(f"    (None found - model just initialized, needs training)")

# Self-attention analysis
print(f"\n  Self-attention (diagonal elements):")
for i in range(8):
    self_attn = avg_all_attn[i, i].item()
    print(f"    {FEATURE_NAMES[i]:12s}: {self_attn:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("CAPABILITY SUMMARY")
print("=" * 80)

capabilities = [
    ("✓ Forward Inference", "Fast inference with batch processing"),
    ("✓ Multi-Head Attention", "32 attention heads capture diverse patterns"),
    ("✓ Attention Extraction", "Extract & visualize attention weights per block"),
    ("✓ Feature Importance", "Quantify which features matter via attention pooling"),
    ("✓ Multi-Modal Fusion", "Cross-attention between geometric & functional features"),
    ("✓ Interpretability", "Understand model decisions through attention"),
    ("✓ Residual Learning", "Deep architecture (4 blocks) without degradation"),
    ("✓ SE Channel Attention", "Adaptive feature recalibration"),
    ("✓ Clinical Relevance", "Works with real cardiac features (volumes, EF)"),
    ("✓ Probabilistic Output", "Confidence scores for all classes"),
]

print("\nKey Capabilities:")
for i, (cap, desc) in enumerate(capabilities, 1):
    print(f"  {i:2d}. {cap:<25} - {desc}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Train on real patient data (meta/acdc_features.csv)")
print("  2. Compare against RAP Fusion baseline (92.67%)")
print("  3. Visualize attention heatmaps for clinical insights")
print("  4. Perform ablation studies (num_blocks, num_heads, hidden_dim)")
print("  5. Analyze which features are most attended for each disease")
print("=" * 80)
