#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

X_COLS = [
    "LVEDV_ml","LVESV_ml","LVEF_pct_robust",  # LV features (reliable)
    # "RVEDV_ml","RVESV_ml","RVEF_pct_robust",  # RV features (removed due to poor quality in ACDC)
    "MYO_ES_to_ED_ratio","MYO_th_mean_ED_mm","MYO_th_p95_ED_mm","MYO_th_max_ED_mm",
    "MYO_th_mean_ES_mm","MYO_th_p95_ES_mm","MYO_th_max_ES_mm","MYO_th_ratio_ES_ED",
    "LV_axis23_ED","LV_axis13_ED",
]

def main():
    geom = pd.read_csv("results/acdc_oof_features_geom.csv")
    lab  = pd.read_csv("results/acdc_labels.csv")
    df   = geom.merge(lab, on="patient_id", how="inner")
    df   = df[df['diagnosis']!=""].copy()

    X = df[X_COLS].values
    y = df["diagnosis"].values
    groups = df["patient_id"].values

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), list(range(len(X_COLS))))
    ])
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=700,
                                         l2_regularization=0.02, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    labels = sorted(df["diagnosis"].unique())
    gkf = GroupKFold(n_splits=5)
    rows=[]; cm_sum=None

    for fold,(tr,va) in enumerate(gkf.split(X,y,groups), start=1):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[va])
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[va], p),
            "balanced_accuracy": balanced_accuracy_score(y[va], p),
            "macro_f1": f1_score(y[va], p, average="macro")
        })
        cm = confusion_matrix(y[va], p, labels=labels)
        cm_sum = cm if cm_sum is None else cm_sum + cm

    Path("results").mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv("results/acdc_diag_cv_metrics_geom.csv", index=False)
    pd.DataFrame(cm_sum, index=labels, columns=labels).to_csv("results/acdc_diag_cm_geom.csv")

    print(pd.DataFrame(rows))
    print("Saved metrics+CM in results/")

if __name__ == "__main__":
    main()
