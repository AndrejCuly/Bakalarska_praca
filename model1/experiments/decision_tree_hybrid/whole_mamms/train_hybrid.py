"""
train_hybrid.py

Trains and evaluates classical ML classifiers on top of the 256-dim
features extracted from the frozen 3D CNN encoder.

Classifiers compared:
    - Decision Tree
    - Random Forest
    - Logistic Regression
    - Gradient Boosting (XGBoost-style via sklearn)

For each classifier, reports:
    - AUC-ROC
    - Sensitivity, Specificity
    - Confusion matrix

Train + val features are combined for training (since classical ML
doesn't need a separate val set for early stopping).
Test features are used for final evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

FEATURES_DIR = "features"
ROC_PLOT     = "roc_curve_hybrid.png"
RESULTS_FILE = "hybrid_results.txt"

# -------------------------------------------------------------------------

def load_split(split_name):
    path = os.path.join(FEATURES_DIR, f"{split_name}_features.npz")
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], data['patients']


def evaluate(name, clf, X_test, y_test, ax=None):
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"AUC-ROC:      {auc:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}  (TP={int(tp)}, FN={int(fn)})")
    print(f"Specificity:  {specificity:.4f}  (TN={int(tn)}, FP={int(fp)})")
    print(f"PPV:          {ppv:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"Confusion matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0       {int(tn):5d}         {int(fp):5d}")
    print(f"  Actual 1       {int(fn):5d}         {int(tp):5d}")

    if ax is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')

    return {
        'name': name, 'auc': auc, 'accuracy': accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def main():
    # Load features
    X_train, y_train, _ = load_split("train")
    X_val,   y_val,   _ = load_split("val")
    X_test,  y_test,  _ = load_split("test")

    # Combine train + val for classical ML training
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    print(f"Train+Val: {X_trainval.shape}  positives: {y_trainval.sum()}")
    print(f"Test:      {X_test.shape}      positives: {y_test.sum()}")

    # Scale features — important for logistic regression
    scaler     = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled     = scaler.transform(X_test)

    # Class weight for imbalance
    n_pos   = y_trainval.sum()
    n_neg   = len(y_trainval) - n_pos
    cw      = {0: 1.0, 1: n_neg / n_pos}
    print(f"Class weight for positive: {cw[1]:.1f}")

    # Define classifiers
    classifiers = [
        ("Decision Tree",
         DecisionTreeClassifier(
             max_depth=5,
             class_weight=cw,
             random_state=42
         ),
         False),   # needs scaling?

        ("Random Forest",
         RandomForestClassifier(
             n_estimators=200,
             max_depth=10,
             class_weight=cw,
             random_state=42,
             n_jobs=-1
         ),
         False),

        ("Logistic Regression",
         LogisticRegression(
             C=0.1,
             class_weight=cw,
             max_iter=1000,
             random_state=42
         ),
         True),    # needs scaling

        ("Gradient Boosting",
         GradientBoostingClassifier(
             n_estimators=200,
             max_depth=3,
             learning_rate=0.05,
             random_state=42
         ),
         False),
    ]

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    all_results = []
    results_text = []

    for name, clf, needs_scaling in classifiers:
        if needs_scaling:
            clf.fit(X_trainval_scaled, y_trainval)
            result = evaluate(name, clf, X_test_scaled, y_test, ax)
        else:
            clf.fit(X_trainval, y_trainval)
            result = evaluate(name, clf, X_test, y_test, ax)
        all_results.append(result)
        results_text.append(
            f"{name}: AUC={result['auc']:.4f}, "
            f"Sens={result['sensitivity']:.4f}, "
            f"Spec={result['specificity']:.4f}"
        )

    # Finalize ROC plot
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves — Hybrid 3D CNN + Classical ML')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150)
    print(f"\nROC curves saved to {ROC_PLOT}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Classifier':<25} {'AUC':>7} {'Sens':>7} {'Spec':>7}")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<25} {r['auc']:>7.4f} "
              f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")

    # Save results to text file
    with open(RESULTS_FILE, 'w') as f:
        f.write("HYBRID EXPERIMENT RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train+Val patients: {len(X_trainval)} "
                f"(positives: {int(y_trainval.sum())})\n")
        f.write(f"Test patients:      {len(X_test)} "
                f"(positives: {int(y_test.sum())})\n\n")
        for line in results_text:
            f.write(line + "\n")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()