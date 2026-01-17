# Cardiac Early Detection - Results Summary


## Classification Results


### Feature Fusion Classifiers


## Segmentation Results


### CAMUS Segmentation

**Overall Metrics:**

- **Dice Score**: 0.947 ± 0.001

- **IoU**: 0.901 ± 0.002

- **Accuracy**: 0.975 ± 0.000



### ACDC Segmentation

**Overall Metrics:**

- **Dice Score**: 0.713 ± 0.033

- **IoU**: 0.634 ± 0.038

- **Accuracy**: 0.975 ± 0.008


**Per-Class Dice Scores:**

- **Right Ventricle (RV)**: 0.988 ± 0.005

- **Myocardium (MYO)**: 0.461 ± 0.072

- **Left Ventricle (LV)**: 0.690 ± 0.045


**Per-Class IoU:**

- **RV**: 0.977 ± 0.009

- **MYO**: 0.328 ± 0.074

- **LV**: 0.597 ± 0.052



## Model Architectures Used


### Segmentation Models

- **2D U-Net**: CAMUS dataset (binary segmentation)

- **3D U-Net**: ACDC dataset (multi-class segmentation)

- **3D U-Net + CRAM**: U-Net with Context Recalibration Attention Module

- **UNETR**: Transformer-based 3D segmentation (Vision Transformer encoder + CNN decoder)


### Classification Models

- **Traditional ML**: Logistic Regression, Random Forest, XGBoost

- **RAP Fusion**: Feature fusion with Residual Attention Pooling blocks

- **Gated Fusion**: Learned gating mechanism for multi-modal feature weighting

- **Cross-Modal Attention**: Query-key-value attention for feature fusion

