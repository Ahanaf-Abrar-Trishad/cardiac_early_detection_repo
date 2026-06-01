#!/bin/bash
# Complete feature extraction workflow for ACDC cardiac analysis
# Run this after training segmentation models

set -e  # Exit on error

echo "========================================="
echo "ACDC Feature Extraction Pipeline"
echo "========================================="
echo ""

# Step 1: Generate OOF predictions for ED phase
echo "Step 1/6: Generating OOF predictions for ED phase..."
python scripts/oof_infer_acdc.py --phase ED --folds 5 --with-bg
echo "✓ ED predictions complete"
echo ""

# Step 2: Generate OOF predictions for ES phase  
echo "Step 2/6: Generating OOF predictions for ES phase..."
python scripts/oof_infer_acdc.py --phase ES --folds 5 --with-bg
echo "✓ ES predictions complete"
echo ""

# Step 3: Extract features from ACDC segmentation
echo "Step 3/6: Extracting features from ACDC segmentation..."
python scripts/extract_features_acdc.py
echo "✓ ACDC features extracted"
echo ""

# Step 4: Extract EF from ACDC
echo "Step 4/6: Extracting ejection fraction from ACDC..."
python scripts/extract_acdc_ef.py
echo "✓ ACDC EF extracted"
echo ""

# Step 5: Extract EF from CAMUS
echo "Step 5/6: Extracting ejection fraction from CAMUS..."
python scripts/extract_camus_ef.py
echo "✓ CAMUS EF extracted"
echo ""

# Step 6: Build geometric features
echo "Step 6/6: Building geometric features..."
python scripts/build_features_geom.py
echo "✓ Geometric features built"
echo ""

echo "========================================="
echo "✅ All feature extraction steps completed!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - logs/oof_preds/acdc/ED/*.nii.gz (OOF predictions)"
echo "  - logs/oof_preds/acdc/ES/*.nii.gz (OOF predictions)"
echo "  - results/acdc_oof_features_geom.csv (Geometric features)"
echo "  - cardio_data/processed/acdc_ef_extracted.csv (EF values)"
