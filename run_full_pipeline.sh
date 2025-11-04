#!/bin/bash
# ============================================================================
# ACDC Cardiac Early Detection - Complete Pipeline
# ============================================================================
# This script runs the entire pipeline from start to finish:
# 1. Segmentation training (ACDC + CAMUS)
# 2. Out-of-fold inference
# 3. Feature extraction
# 4. Fusion classifier training
# 5. Traditional baseline classifiers
# 6. Results generation
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "🫀 ACDC Cardiac Early Detection - Full Pipeline"
echo "============================================================================"
echo ""

# Parse command line arguments
SKIP_SEG_TRAINING=false
SKIP_OOF=false
SKIP_FEATURES=false
SKIP_FUSION=false
SKIP_BASELINES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-seg) SKIP_SEG_TRAINING=true; shift ;;
        --skip-oof) SKIP_OOF=true; shift ;;
        --skip-features) SKIP_FEATURES=true; shift ;;
        --skip-fusion) SKIP_FUSION=true; shift ;;
        --skip-baselines) SKIP_BASELINES=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-seg         Skip segmentation training (use existing checkpoints)"
            echo "  --skip-oof         Skip OOF inference (use existing predictions)"
            echo "  --skip-features    Skip feature extraction (use existing features)"
            echo "  --skip-fusion      Skip fusion classifier training"
            echo "  --skip-baselines   Skip traditional baseline classifiers"
            echo "  --help             Show this help message"
            echo ""
            echo "Example: $0 --skip-seg --skip-oof  # Start from feature extraction"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# STEP 1: Segmentation Training (ACDC + CAMUS)
# ============================================================================
if [ "$SKIP_SEG_TRAINING" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 STEP 1: Segmentation Training"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "1.1 Training ACDC ED Segmentation (UNet3D)..."
    python scripts/seg_cv.py \
        --dataset acdc \
        --model unet3d \
        --phase ED \
        --acdc-multiclass \
        --folds 5 \
        --epochs 40 \
        --aug3d \
        --class-weights auto
    echo "✓ ACDC ED segmentation complete"
    
    echo ""
    echo "1.2 Training ACDC ES Segmentation (UNet3D)..."
    python scripts/seg_cv.py \
        --dataset acdc \
        --model unet3d \
        --phase ES \
        --acdc-multiclass \
        --folds 5 \
        --epochs 40 \
        --aug3d \
        --class-weights auto
    echo "✓ ACDC ES segmentation complete"
    
    echo ""
    echo "1.3 Training CAMUS Segmentation..."
    python scripts/seg_cv.py \
        --dataset camus \
        --model unet \
        --folds 5 \
        --epochs 50 \
        --aug
    echo "✓ CAMUS segmentation complete"
    
else
    echo "⏭️  Skipping segmentation training (using existing checkpoints)"
fi

# ============================================================================
# STEP 2: Out-of-Fold Inference
# ============================================================================
if [ "$SKIP_OOF" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔮 STEP 2: Out-of-Fold Inference"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "2.1 Generating OOF predictions for ED phase..."
    python scripts/oof_infer_acdc.py --phase ED --folds 5 --with-bg
    echo "✓ ED predictions complete"
    
    echo ""
    echo "2.2 Generating OOF predictions for ES phase..."
    python scripts/oof_infer_acdc.py --phase ES --folds 5 --with-bg
    echo "✓ ES predictions complete"
    
else
    echo "⏭️  Skipping OOF inference (using existing predictions)"
fi

# ============================================================================
# STEP 3: Feature Extraction
# ============================================================================
if [ "$SKIP_FEATURES" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 STEP 3: Feature Extraction"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "3.1 Extracting features from ACDC segmentation..."
    python scripts/extract_features_acdc.py
    echo "✓ ACDC features extracted"
    
    echo ""
    echo "3.2 Extracting ejection fraction from ACDC..."
    python scripts/extract_acdc_ef.py
    echo "✓ ACDC EF extracted"
    
    echo ""
    echo "3.3 Extracting ejection fraction from CAMUS..."
    python scripts/extract_camus_ef.py
    echo "✓ CAMUS EF extracted"
    
    echo ""
    echo "3.4 Building geometric features..."
    python scripts/build_features_geom.py
    echo "✓ Geometric features built"
    
else
    echo "⏭️  Skipping feature extraction (using existing features)"
fi

# ============================================================================
# STEP 4: Fusion Classifier Training
# ============================================================================
if [ "$SKIP_FUSION" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧠 STEP 4: Fusion Classifier Training"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "4.1 Training RAP Fusion Classifier (with Cross-Attention)..."
    python scripts/train_fusion_classifier.py \
        --features meta/acdc_features.csv \
        --fusion-type rap \
        --use-cross-attention \
        --folds 5 \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --hidden-dim 128 \
        --dropout 0.3 \
        --seed 42
    echo "✓ Fusion classifier complete"
    
else
    echo "⏭️  Skipping fusion classifier training"
fi

# ============================================================================
# STEP 5: Traditional Baseline Classifiers
# ============================================================================
if [ "$SKIP_BASELINES" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 STEP 5: Traditional Baseline Classifiers"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "5.1 Training All Baseline Classifiers (LogReg, RF, XGBoost)..."
    python scripts/classify_cv.py \
        --features meta/acdc_features.csv \
        --models logreg,rf,xgb \
        --subset all \
        --folds 5
    echo "✓ All baselines complete"
    
else
    echo "⏭️  Skipping traditional baselines"
fi

# ============================================================================
# STEP 6: Generate Results
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STEP 6: Generating Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Generating RESULTS.md and summary statistics..."
python scripts/make_results_summary.py
echo "✓ Results generated"

# ============================================================================
# Complete!
# ============================================================================
echo ""
echo "============================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "============================================================================"
echo ""
echo "📁 Output Files:"
echo "   Segmentation:"
echo "     - logs/seg_acdc_fold*_best.pt"
echo "     - logs/seg_camus_fold*_best.pt"
echo "     - logs/cv_seg_acdc_summary.json"
echo "     - logs/cv_seg_camus_summary.json"
echo ""
echo "   Features:"
echo "     - logs/oof_preds/acdc/ED/*.nii.gz"
echo "     - logs/oof_preds/acdc/ES/*.nii.gz"
echo "     - meta/acdc_features.csv"
echo "     - results/acdc_oof_features_geom.csv"
echo ""
echo "   Classifiers:"
echo "     - logs/fusion_classifier_fold*_best.pt"
echo "     - logs/fusion_classifier_cv_summary.json"
echo "     - logs/cv_cls_*.csv"
echo "     - logs/cv_cls_summary.json"
echo ""
echo "   Results:"
echo "     - results/RESULTS.md"
echo "     - logs/cv_cls_summary.csv"
echo ""
echo "🎉 All done! Check results/RESULTS.md for the complete report."
echo "============================================================================"
