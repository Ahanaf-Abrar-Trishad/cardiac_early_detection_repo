#!/usr/bin/env python3
"""
Training script for Cross-Modal Attention Fusion Classifier.
Trains the model to fuse MRI and Echo features using multi-head attention.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from models.cross_modal_fusion import CrossModalFusionClassifier, CrossModalDataset, create_cross_modal_dataset

def train_cross_modal_fusion(
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=10,
    save_path='logs/cross_modal_fusion'
):
    """
    Train the cross-modal attention fusion classifier.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        save_path: Path to save model and logs
    """
    print("🔄 Training Cross-Modal Attention Fusion Classifier")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("📊 Loading cross-modal datasets...")
    mri_df, echo_df, mri_cols, echo_cols = create_cross_modal_dataset()

    # Create train/val splits (stratified by MRI labels)
    from sklearn.model_selection import train_test_split
    train_mri, val_mri = train_test_split(
        mri_df, test_size=0.2, random_state=42, stratify=mri_df['label_enc']
    )

    # Create datasets
    train_dataset = CrossModalDataset(train_mri, echo_df, mri_cols, echo_cols, fit_scaler=True)
    val_dataset = CrossModalDataset(val_mri, echo_df, mri_cols, echo_cols,
                                   scaler=train_dataset.get_scalers(), fit_scaler=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = CrossModalFusionClassifier(
        mri_dim=len(mri_cols),
        echo_dim=len(echo_cols),
        num_classes=5,
        hidden_dim=128,
        num_heads=8,
        dropout=0.3
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            mri_features = batch['mri_features'].to(device)
            echo_features = batch['echo_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(mri_features, echo_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                mri_features = batch['mri_features'].to(device)
                echo_features = batch['echo_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(mri_features, echo_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'scalers': train_dataset.get_scalers()
            }, save_path / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break

    print(f"\n✅ Training completed! Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")

    # Load best model for evaluation
    checkpoint = torch.load(save_path / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    print("\n📊 Final Evaluation:")
    evaluate_model(model, val_loader, device, save_path)

    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_history(history, save_path)

    return model, history


def evaluate_model(model, val_loader, device, save_path):
    """Evaluate model and generate detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            mri_features = batch['mri_features'].to(device)
            echo_features = batch['echo_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(mri_features, echo_features)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    target_names = ['NOR', 'MIN', 'MR', 'MS', 'AR']
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Cross-Modal Fusion Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed metrics
    with open(save_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


def plot_training_history(history, save_path):
    """Plot training and validation curves"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Train the cross-modal fusion model
    model, history = train_cross_modal_fusion(
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        save_path='logs/cross_modal_fusion'
    )

    print("\n🎉 Cross-Modal Attention Fusion training completed!")
    print("Model saved to: logs/cross_modal_fusion/best_model.pt")
    print("Training history: logs/cross_modal_fusion/training_history.json")
    print("Evaluation metrics: logs/cross_modal_fusion/evaluation_metrics.json")