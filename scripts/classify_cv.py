#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# optional xgboost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Optional tracking
try:
    import mlflow; _HAS_MLFLOW=True
except Exception:
    _HAS_MLFLOW=False
try:
    import wandb; _HAS_WANDB=True
except Exception:
    _HAS_WANDB=False


def maybe_init_tracking(args, n_features, classes):
    if args.mlflow and _HAS_MLFLOW:
        if args.mlflow_uri: mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name="acdc-classification")
        mlflow.log_params({
            "seed": args.seed, "folds": args.folds,
            "n_features": n_features, "classes": ",".join(classes),
            "subset": args.subset, "models": args.models,
            "calibrate": args.calibrate
        })
    if args.wandb and _HAS_WANDB:
        wandb.init(
            project=args.wandb_project or "cardiac-cls",
            entity=args.wandb_entity or None,
            name="acdc-classification",
            config={
                "seed": args.seed, "folds": args.folds,
                "n_features": n_features, "classes": classes,
                "subset": args.subset, "models": args.models,
                "calibrate": args.calibrate
            }
        )

def track(metrics, args, cm_png=None):
    if args.mlflow and _HAS_MLFLOW:
        for k,v in metrics.items():
            try: mlflow.log_metric(k, float(v))
            except Exception: pass
        if cm_png and Path(cm_png).exists():
            mlflow.log_artifact(cm_png, artifact_path="confusion_matrix")
    if args.wandb and _HAS_WANDB:
        try: wandb.log(metrics)
        except Exception: pass
        if cm_png and Path(cm_png).exists():
            wandb.log({"confusion_matrix": wandb.Image(cm_png)})

def end_tracking(args):
    if args.mlflow and _HAS_MLFLOW:
        try: mlflow.end_run()
        except Exception: pass
    if args.wandb and _HAS_WANDB:
        try: wandb.finish()
        except Exception: pass

def save_confmat(cm, classes, out_png):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def build_pipeline(name, seed):
    """Return (scaler + classifier) pipeline. Calibration is added later if requested."""
    if name == "logreg":
        # lbfgs supports multinomial; no deprecated multi_class arg
        clf = LogisticRegression(max_iter=500, class_weight="balanced",
                                 solver="lbfgs", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                     random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "xgb":
        if not _HAS_XGB:
            print("[warn] xgboost not installed; skipping XGB.")
            return None
        clf = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            min_child_weight=2,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=4
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return None

def maybe_calibrate(pipe: Pipeline, method="sigmoid", cv=3):
    """Wrap the classifier step with CalibratedClassifierCV, keeping the scaler."""
    if "clf" not in pipe.named_steps:
        return pipe
    base = pipe.named_steps["clf"]

    # Skip if classifier cannot provide scores/probabilities
    has_proba = getattr(base, "predict_proba", None) is not None
    has_dec = getattr(base, "decision_function", None) is not None
    if not (has_proba or has_dec):
        return pipe

    # sklearn >= 1.6 uses 'estimator'; older versions used 'base_estimator'
    try:
        cal = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)

    return Pipeline([("scaler", pipe.named_steps["scaler"]), ("clf", cal)])

def pick_feature_columns(df: pd.DataFrame, subset: str):
    """Choose which features to use: 'all', 'ef', or 'vol'."""
    if subset == "ef":
        cols = [c for c in df.columns if c in ["LV_EF", "RV_EF"]]
    elif subset == "vol":
        cols = [c for c in df.columns if c.endswith("_mL")]
    else:
        cols = [c for c in df.columns if c not in ("patient_id","label")]
    return cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="meta/acdc_features.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--models", default="logreg,rf,xgb", help="comma list: logreg,rf,xgb")
    ap.add_argument("--subset", default="all", choices=["all","ef","vol"],
                    help="feature subset: all features, EF-only, or volumes-only")
    ap.add_argument("--calibrate", action="store_true", help="Platt scaling (sigmoid) inside each fold")
    # tracking
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-uri", default="", dest="mlflow_uri")
    ap.add_argument("--mlflow-experiment", default="cls-cv", dest="mlflow_experiment")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="cardiac-cls", dest="wandb_project")
    ap.add_argument("--wandb-entity", default="", dest="wandb_entity")
    args = ap.parse_args()

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    import torch as _torch
    _torch.manual_seed(args.seed)
    _torch.cuda.manual_seed_all(args.seed)

    feat = pd.read_csv(args.features)
    if feat.empty:
        raise SystemExit(f"[classify_cv] {args.features} is empty. Run extract_features_acdc.py first.")

    req = ["patient_id", "label"]
    miss = [c for c in req if c not in feat.columns]
    if miss:
        raise SystemExit(f"[classify_cv] Missing columns: {miss}")

    feat = feat.dropna(subset=["label"]).reset_index(drop=True)
    feat_cols = pick_feature_columns(feat, args.subset)
    if not feat_cols:
        raise SystemExit(f"[classify_cv] No feature columns selected for subset='{args.subset}'.")

    X = feat[feat_cols].values
    y_text = feat["label"].astype(str).values

    # encode labels → numeric
    le = LabelEncoder()
    y = le.fit_transform(y_text)              # 0..K-1
    class_names = list(le.classes_)           # e.g., ['DCM','HCM','MINF','NOR','RV']
    K = len(class_names)

    # folds: ensure feasibility
    counts = pd.Series(y).value_counts()
    eff_folds = max(2, min(args.folds, counts.min()))
    if eff_folds != args.folds:
        print(f"[classify_cv] Adjusting folds {args.folds} -> {eff_folds} based on min class count.")
    args.folds = eff_folds

    maybe_init_tracking(args, X.shape[1], class_names)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_rows = []
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        base_pipe = build_pipeline(model_name, args.seed)
        if base_pipe is None:
            continue
        pipe = maybe_calibrate(base_pipe, method="sigmoid", cv=3) if args.calibrate else base_pipe

        fold_rows = []
        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]

            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)

            # For reports and CMs use human-readable labels
            pred_names = le.inverse_transform(pred)
            yva_names  = le.inverse_transform(yva)

            # probabilities for AUC
            try:
                proba = pipe.predict_proba(Xva)  # shape (n, K)
            except Exception:
                proba = None

            acc  = accuracy_score(yva, pred)
            bacc = balanced_accuracy_score(yva, pred)
            f1m  = f1_score(yva, pred, average="macro")

            aucm = np.nan
            if proba is not None and len(np.unique(yva)) > 1:
                try:
                    from sklearn.preprocessing import label_binarize
                    yb = label_binarize(yva, classes=np.arange(K))
                    aucm = roc_auc_score(yb, proba, average="macro", multi_class="ovr")
                except Exception:
                    pass

            # confusion matrix (names)
            cm = confusion_matrix(yva_names, pred_names, labels=class_names)
            cm_png = Path(args.logdir) / f"cv_cls_{model_name}_fold{fold}_cm.png"
            save_confmat(cm, class_names, cm_png)

            # per-class report CSV
            rep = classification_report(
                yva_names, pred_names, labels=class_names,
                output_dict=True, zero_division=0
            )
            rep_df = pd.DataFrame(rep).T
            rep_path = Path(args.logdir) / f"cv_cls_{model_name}_fold{fold}_perclass.csv"
            rep_df.to_csv(rep_path)

            # feature importance / coefficients
            imp_dir = Path(args.logdir) / "feature_importance"
            imp_dir.mkdir(parents=True, exist_ok=True)
            clf = pipe.named_steps["clf"]

            # unwrap calibrated models across sklearn versions
            inner = clf
            try:
                if isinstance(clf, CalibratedClassifierCV):
                    inner = getattr(clf, "estimator", getattr(clf, "base_estimator", clf))
            except Exception:
                inner = clf

            # LR: class-wise coefficients (one-vs-rest or multinomial)
            try:
                if hasattr(inner, "coef_"):
                    coefs = inner.coef_
                    if isinstance(coefs, np.ndarray) and coefs.ndim == 1:
                        coefs = coefs[None, :]
                    coef_df = pd.DataFrame(coefs, columns=feat_cols)
                    if coef_df.shape[0] == K:
                        coef_df.index = class_names
                    coef_df.to_csv(imp_dir / f"coef_{model_name}_fold{fold}.csv")
            except Exception:
                pass

            # RF / XGB: feature importances
            try:
                if hasattr(inner, "feature_importances_"):
                    fi = pd.Series(inner.feature_importances_, index=feat_cols).sort_values(ascending=False)
                    fi.to_csv(imp_dir / f"fi_{model_name}_fold{fold}.csv")
            except Exception:
                pass

            row = {
                "model": model_name, "fold": fold,
                "acc": acc, "bacc": bacc,
                "f1_macro": f1m, "auc_macro": float(aucm)
            }
            fold_rows.append(row)
            track({
                f"{model_name}_acc": acc,
                f"{model_name}_bacc": bacc,
                f"{model_name}_f1m": f1m,
                f"{model_name}_aucm": float(aucm)
            }, args, str(cm_png))

        dfm = pd.DataFrame(fold_rows)
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
        dfm.to_csv(Path(args.logdir)/f"cv_cls_{model_name}_metrics.csv", index=False)
        all_rows.append({
            "model": model_name,
            "acc_mean": dfm["acc"].mean(),  "acc_std": dfm["acc"].std(),
            "bacc_mean": dfm["bacc"].mean(),"bacc_std": dfm["bacc"].std(),
            "f1m_mean": dfm["f1_macro"].mean(),"f1m_std": dfm["f1_macro"].std(),
            "aucm_mean": dfm["auc_macro"].mean(),"aucm_std": dfm["auc_macro"].std()
        })

    summ = pd.DataFrame(all_rows)
    summ.to_csv(Path(args.logdir)/"cv_cls_summary.csv", index=False)
    with open(Path(args.logdir)/"cv_cls_summary.json","w") as f:
        json.dump({"models": all_rows}, f, indent=2)

    end_tracking(args)
    print("Saved classification metrics to logs/ (per-model CSVs + summary).")

if __name__ == "__main__":
    main()
