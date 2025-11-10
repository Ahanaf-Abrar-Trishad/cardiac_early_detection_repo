#!/usr/bin/env python3
"""
Compare all classification models: Traditional ML, RAP Fusion, and Attention models
Generates a comprehensive comparison report with tables and visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy import stats


def load_results(logdir):
    """Load results from all models."""
    results = {}
    logdir = Path(logdir)
    
    # Traditional ML results
    ml_files = {
        'Logistic Regression': 'cv_cls_logreg_metrics.csv',
        'Random Forest': 'cv_cls_rf_metrics.csv',
        'XGBoost': 'cv_cls_xgb_metrics.csv'
    }
    
    for name, filename in ml_files.items():
        filepath = logdir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            results[name] = {
                'type': 'traditional_ml',
                'acc': df['accuracy'].values,
                'bal_acc': df['balanced_accuracy'].values if 'balanced_accuracy' in df.columns else None,
                'f1_macro': df['f1_macro'].values,
                'auc': df['roc_auc_ovr'].values if 'roc_auc_ovr' in df.columns else None,
                'params': 'N/A'
            }
    
    # RAP Fusion results
    fusion_file = logdir / 'fusion_classifier_cv_summary.json'
    if fusion_file.exists():
        with open(fusion_file) as f:
            fusion_data = json.load(f)
        
        fold_metrics = fusion_data['fold_metrics']
        results['RAP Fusion'] = {
            'type': 'deep_learning',
            'acc': np.array([m['acc'] for m in fold_metrics]),
            'bal_acc': np.array([m['bal_acc'] for m in fold_metrics]),
            'f1_macro': np.array([m['f1_macro'] for m in fold_metrics]),
            'auc': np.array([m['auc'] for m in fold_metrics]),
            'params': '~100K'
        }
    
    # Advanced Attention results
    adv_file = logdir / 'attention_advanced_cv_summary.json'
    if adv_file.exists():
        with open(adv_file) as f:
            adv_data = json.load(f)
        
        fold_metrics = adv_data['fold_metrics']
        results['Advanced Attention'] = {
            'type': 'deep_learning',
            'acc': np.array([m['acc'] for m in fold_metrics]),
            'bal_acc': np.array([m['bal_acc'] for m in fold_metrics]),
            'f1_macro': np.array([m['f1_macro'] for m in fold_metrics]),
            'auc': np.array([m['auc'] for m in fold_metrics]),
            'params': f"{adv_data['num_params']:,}",
            'summary_stats': adv_data['summary_stats']
        }
    
    # MultiModal Attention results
    mm_file = logdir / 'attention_multimodal_cv_summary.json'
    if mm_file.exists():
        with open(mm_file) as f:
            mm_data = json.load(f)
        
        fold_metrics = mm_data['fold_metrics']
        results['MultiModal Attention'] = {
            'type': 'deep_learning',
            'acc': np.array([m['acc'] for m in fold_metrics]),
            'bal_acc': np.array([m['bal_acc'] for m in fold_metrics]),
            'f1_macro': np.array([m['f1_macro'] for m in fold_metrics]),
            'auc': np.array([m['auc'] for m in fold_metrics]),
            'params': f"{mm_data['num_params']:,}",
            'summary_stats': mm_data['summary_stats']
        }
    
    return results


def compute_stats(values):
    """Compute mean, std, and 95% CI."""
    if values is None:
        return None, None, None, None
    
    mean = np.mean(values)
    std = np.std(values)
    
    n = len(values)
    if n < 2:
        return mean, std, None, None
    
    sem = stats.sem(values)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    
    return mean, std, ci[0], ci[1]


def compare_models(results, baseline='RAP Fusion'):
    """Compare all models against baseline."""
    if baseline not in results:
        baseline = list(results.keys())[0]
    
    baseline_acc = results[baseline]['acc']
    baseline_mean = np.mean(baseline_acc)
    
    comparisons = []
    for name, data in results.items():
        acc = data['acc']
        mean, std, ci_lower, ci_upper = compute_stats(acc)
        
        # Compute improvement over baseline
        if name != baseline:
            improvement = (mean - baseline_mean) * 100
            
            # Statistical significance test (paired t-test)
            if len(acc) == len(baseline_acc):
                t_stat, p_value = stats.ttest_rel(acc, baseline_acc)
                significant = p_value < 0.05
            else:
                t_stat, p_value = None, None
                significant = False
        else:
            improvement = 0.0
            p_value = None
            significant = False
        
        comparisons.append({
            'model': name,
            'type': data['type'],
            'acc_mean': mean,
            'acc_std': std,
            'acc_ci_lower': ci_lower,
            'acc_ci_upper': ci_upper,
            'improvement': improvement,
            'p_value': p_value,
            'significant': significant,
            'params': data['params']
        })
    
    return pd.DataFrame(comparisons)


def print_comparison_table(results):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 120)
    
    # Header
    print(f"{'Model':<30s} {'Type':<15s} {'Accuracy':<25s} {'95% CI':<20s} {'F1-Macro':<12s} {'AUC':<12s} {'Params':<15s}")
    print("-" * 120)
    
    # Sort by accuracy (descending)
    sorted_models = sorted(results.items(), 
                          key=lambda x: np.mean(x[1]['acc']), 
                          reverse=True)
    
    for name, data in sorted_models:
        acc_mean, acc_std, acc_ci_l, acc_ci_u = compute_stats(data['acc'])
        f1_mean, f1_std, _, _ = compute_stats(data['f1_macro'])
        auc_mean, auc_std, _, _ = compute_stats(data['auc'])
        
        acc_str = f"{acc_mean:.4f} ± {acc_std:.4f}"
        ci_str = f"[{acc_ci_l:.4f}, {acc_ci_u:.4f}]" if acc_ci_l is not None else "N/A"
        f1_str = f"{f1_mean:.4f}" if f1_mean is not None else "N/A"
        auc_str = f"{auc_mean:.4f}" if auc_mean is not None else "N/A"
        
        print(f"{name:<30s} {data['type']:<15s} {acc_str:<25s} {ci_str:<20s} {f1_str:<12s} {auc_str:<12s} {data['params']:<15s}")
    
    print("=" * 120)


def print_statistical_comparison(comp_df, baseline='RAP Fusion'):
    """Print statistical comparison table."""
    print("\n" + "=" * 100)
    print(f"STATISTICAL COMPARISON (Baseline: {baseline})")
    print("=" * 100)
    
    print(f"{'Model':<30s} {'Improvement':<15s} {'P-value':<12s} {'Significant':<15s}")
    print("-" * 100)
    
    for _, row in comp_df.iterrows():
        if row['model'] == baseline:
            print(f"{row['model']:<30s} {'(baseline)':<15s} {'-':<12s} {'-':<15s}")
        else:
            improvement_str = f"{row['improvement']:+.2f}%"
            p_str = f"{row['p_value']:.4f}" if row['p_value'] is not None else "N/A"
            sig_str = "✓ Yes (p<0.05)" if row['significant'] else "✗ No"
            
            print(f"{row['model']:<30s} {improvement_str:<15s} {p_str:<12s} {sig_str:<15s}")
    
    print("=" * 100)


def print_best_model_summary(results):
    """Print summary of best performing model."""
    # Find best model by accuracy
    best_model = max(results.items(), key=lambda x: np.mean(x[1]['acc']))
    name, data = best_model
    
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODEL")
    print("=" * 80)
    print(f"Model: {name}")
    print(f"Type: {data['type']}")
    print(f"Parameters: {data['params']}")
    print()
    
    # Print all metrics with confidence intervals
    metrics = ['acc', 'bal_acc', 'f1_macro', 'auc']
    metric_names = ['Accuracy', 'Balanced Accuracy', 'F1-Macro', 'AUC-ROC']
    
    for metric, metric_name in zip(metrics, metric_names):
        if data[metric] is not None:
            mean, std, ci_l, ci_u = compute_stats(data[metric])
            print(f"{metric_name:<20s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_l:.4f}, {ci_u:.4f}]")
    
    print("=" * 80)


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    print("\n" + "=" * 80)
    print("LATEX TABLE (Copy-paste for paper)")
    print("=" * 80)
    
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of Classification Models for Cardiac Disease Detection}")
    print(r"\label{tab:model_comparison}")
    print(r"\begin{tabular}{lccccl}")
    print(r"\hline")
    print(r"Model & Accuracy (\%) & 95\% CI & F1-Macro & AUC & Parameters \\")
    print(r"\hline")
    
    # Sort by accuracy
    sorted_models = sorted(results.items(), 
                          key=lambda x: np.mean(x[1]['acc']), 
                          reverse=True)
    
    for name, data in sorted_models:
        acc_mean, acc_std, acc_ci_l, acc_ci_u = compute_stats(data['acc'])
        f1_mean, _, _, _ = compute_stats(data['f1_macro'])
        auc_mean, _, _, _ = compute_stats(data['auc'])
        
        # Format for LaTeX
        acc_str = f"{acc_mean*100:.2f} $\\pm$ {acc_std*100:.2f}"
        ci_str = f"[{acc_ci_l*100:.2f}, {acc_ci_u*100:.2f}]"
        f1_str = f"{f1_mean:.3f}" if f1_mean is not None else "N/A"
        auc_str = f"{auc_mean:.3f}" if auc_mean is not None else "N/A"
        
        # Add bold for best model (first in sorted list)
        if name == sorted_models[0][0]:
            name_str = f"\\textbf{{{name}}}"
            acc_str = f"\\textbf{{{acc_str}}}"
            f1_str = f"\\textbf{{{f1_str}}}"
            auc_str = f"\\textbf{{{auc_str}}}"
        else:
            name_str = name
        
        print(f"{name_str} & {acc_str} & {ci_str} & {f1_str} & {auc_str} & {data['params']} \\\\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare all classification models")
    parser.add_argument("--logdir", default="logs", help="Directory containing results")
    parser.add_argument("--baseline", default="RAP Fusion", help="Baseline model for comparison")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--save", help="Save comparison to CSV file")
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_results(args.logdir)
    
    if not results:
        print(f"ERROR: No results found in {args.logdir}")
        return
    
    print(f"Found {len(results)} models:")
    for name in results.keys():
        print(f"  - {name}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Statistical comparison
    comp_df = compare_models(results, baseline=args.baseline)
    print_statistical_comparison(comp_df, baseline=args.baseline)
    
    # Best model summary
    print_best_model_summary(results)
    
    # LaTeX table
    if args.latex:
        generate_latex_table(results)
    
    # Save to CSV
    if args.save:
        comp_df.to_csv(args.save, index=False)
        print(f"\n✓ Comparison saved to {args.save}")


if __name__ == "__main__":
    main()
