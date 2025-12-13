#!/bin/bash
# Quick test of the full pipeline using existing checkpoints
# This validates feature extraction → classification → results workflow

set -e

echo "🧪 Testing pipeline with existing checkpoints..."
echo ""

# Test 1: Feature extraction (should be fast if already done)
echo "1/3 Testing feature extraction..."
python scripts/build_features_geom.py
echo "✓ Features OK"

# Test 2: Classification baselines
echo ""
echo "2/3 Testing baseline classifiers..."
python scripts/classify_cv.py \
  --features meta/acdc_features.csv \
  --models logreg \
  --subset all \
  --folds 5 2>&1 | tail -5
echo "✓ Baselines OK"

# Test 3: Results generation
echo ""
echo "3/3 Testing results generation..."
python scripts/make_results_summary.py 2>&1 | tail -5
echo "✓ Results OK"

echo ""
echo "✅ All pipeline components working!"
echo ""
echo "You can now run:"
echo "  make seg3d-ed seg3d-es oof-all features-geom diag-geom   # Complete from scratch"
echo "  make features-geom diag-geom                              # Classification-only refresh"
