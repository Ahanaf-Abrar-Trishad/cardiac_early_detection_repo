#!/usr/bin/env python3
"""
Tabular CV with proper anti-leakage pipeline and CSV/JSON logging.
OneHot -> Scale -> SelectKBest -> SMOTE -> Classifier.
"""
import argparse, numpy as np, pandas as pd, json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with features + target")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--categoricals", nargs="*", default=[], help="Categorical feature names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["logreg","rf"], default="logreg")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=30, help="RandomizedSearch iterations")
    ap.add_argument("--logdir", default="logs", help="Directory to save CSV/JSON metrics")
    return ap.parse_args()

def main():
    args = parse_args()
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df[args.target].values
    X = df.drop(columns=[args.target])
    cat_cols = [c for c in args.categoricals if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    col_proc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ], remainder="drop")

    if args.model == "logreg":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga")
        search_space = {
            "clf__C": np.logspace(-3, 2, 50),
            "fs__k": [10, 20, 50, 100, "all"],
            "smote__k_neighbors": [3, 5, 7]
        }
    else:
        estimator = RandomForestClassifier(class_weight="balanced", n_estimators=500, random_state=args.seed)
        search_space = {
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "fs__k": [10, 20, 50, 100, "all"],
            "smote__k_neighbors": [3, 5, 7]
        }

    pipe = ImbPipeline(steps=[
        ("prep", col_proc),
        ("fs", SelectKBest(score_func=mutual_info_classif, k="all")),
        ("smote", SMOTE(random_state=args.seed)),
        ("clf", estimator)
    ])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    metrics = []; params_per_fold=[]
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        # Determine if binary or multiclass
        n_classes = len(np.unique(ytr))
        scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
        
        search = RandomizedSearchCV(
            estimator=pipe, param_distributions=search_space, n_iter=args.n_iter,
            scoring=scoring, n_jobs=-1, cv=3, refit=True, random_state=args.seed
        )
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        pred = best.predict(Xva)
        
        # Handle AUC calculation for binary vs multiclass
        if hasattr(best, "predict_proba"):
            proba = best.predict_proba(Xva)
            n_classes = len(np.unique(yva))
            if n_classes == 2:
                auc = roc_auc_score(yva, proba[:, 1])
            else:
                auc = roc_auc_score(yva, proba, multi_class='ovr', average='macro')
        else:
            auc = np.nan
        acc = accuracy_score(yva, pred)
        n_classes = len(np.unique(yva))
        f1 = f1_score(yva, pred, average='macro' if n_classes > 2 else 'binary')
        metrics.append({"fold": fold, "AUC": float(auc) if proba is not None else float('nan'),
                        "ACC": float(acc), "F1": float(f1)})
        params_per_fold.append({"fold": fold, **search.best_params_})
        print(f"[Fold {fold}] AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f} | best={search.best_params_}")

    m = np.array([[d["AUC"], d["ACC"], d["F1"]] for d in metrics], dtype=float)
    m_auc = float(np.nanmean(m[:,0])); s_auc = float(np.nanstd(m[:,0]))
    m_acc = float(np.mean(m[:,1]));    s_acc = float(np.std(m[:,1]))
    m_f1  = float(np.mean(m[:,2]));    s_f1  = float(np.std(m[:,2]))

    print("\n=== CV Summary ===")
    print(f"AUC mean±std: {m_auc:.4f} ± {s_auc:.4f}")
    print(f"ACC mean±std: {m_acc:.4f} ± {s_acc:.4f}")
    print(f"F1  mean±std: {m_f1:.4f} ± {s_f1:.4f}")

    # Save
    pdf = pd.DataFrame(metrics).sort_values("fold")
    pdf.to_csv(logdir / "cv_tabular_metrics.csv", index=False)
    with open(logdir / "cv_tabular_metrics.json","w") as f: json.dump(metrics, f, indent=2)
    with open(logdir / "cv_tabular_best_params.json","w") as f: json.dump(params_per_fold, f, indent=2)
    with open(logdir / "cv_tabular_summary.json","w") as f: 
        json.dump({"AUC_mean": m_auc, "AUC_std": s_auc, "ACC_mean": m_acc, "ACC_std": s_acc, "F1_mean": m_f1, "F1_std": s_f1,
                   "folds": int(len(metrics)), "seed": int(args.seed), "n_iter": int(args.n_iter), "model": args.model}, f, indent=2)

if __name__ == "__main__":
    main()



def save_roc_and_cm(y, probs, preds, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # ROC
    try:
        if probs is not None and len(np.unique(y))>1:
            fpr, tpr, thr = roc_curve(y, probs)
            plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
            plt.savefig(out_prefix.with_suffix('.roc.png')); plt.close()
            with open(out_prefix.with_suffix('.roc.json'),'w') as f: json.dump({'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thresholds':thr.tolist()}, f, indent=2)
    except Exception as e:
        print('ROC save failed:', e)
    # CM
    try:
        cm = confusion_matrix(y, preds, labels=[0,1])
        plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i,j]), ha='center', va='center')
        plt.colorbar(); plt.savefig(out_prefix.with_suffix('.cm.png')); plt.close()
        with open(out_prefix.with_suffix('.cm.json'),'w') as f: json.dump({'labels':[0,1], 'matrix': cm.tolist()}, f, indent=2)
    except Exception as e:
        print('CM save failed:', e)