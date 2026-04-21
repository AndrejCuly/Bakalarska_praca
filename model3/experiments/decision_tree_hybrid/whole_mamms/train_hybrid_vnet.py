"""
train_hybrid_vnet.py

Trains and evaluates classical ML classifiers on top of the 256-dim
features extracted from the frozen Model 3 VNet encoder.
Whole mammograms version.

Run extract_features_vnet.py first.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'features')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
ROC_PLOT     = os.path.join(RESULTS_DIR, 'roc_curve_hybrid.png')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'hybrid_results.txt')

IS_REGRESSION = {
    "Decision Tree":       False,
    "Random Forest":       False,
    "Logistic Regression": True,
    "Gradient Boosting":   True,
}

def load_split(split_name):
    path = os.path.join(FEATURES_DIR, f"{split_name}_features.npz")
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], data['patients']

def evaluate(name, clf, X_test, y_test, patients, ax=None):
    probs = clf.predict_proba(X_test)[:, 1]
    is_regression = IS_REGRESSION.get(name, False)
    preds = (probs >= 0.5).astype(int) if is_regression else clf.predict(X_test)
    output_col = 'predicted_prob' if is_regression else 'predicted_class'

    auc = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)

    print(f"\n{'='*50}")
    print(f"{name} ({'regression/scoring' if is_regression else 'classification'})")
    print(f"{'='*50}")
    print(f"AUC-ROC:      {auc:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}  (TP={int(tp)}, FN={int(fn)})")
    print(f"Specificity:  {specificity:.4f}  (TN={int(tn)}, FP={int(fp)})")
    print(f"PPV:          {ppv:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"Confusion matrix (threshold=0.5):")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0       {int(tn):5d}         {int(fp):5d}")
    print(f"  Actual 1       {int(fn):5d}         {int(tp):5d}")

    csv_name = name.lower().replace(' ', '_') + '_results.csv'
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'true_label', 'predicted_prob', output_col])
        for patient, label, prob, pred in zip(patients, y_test, probs, preds):
            if is_regression:
                writer.writerow([patient, int(label), f'{prob:.4f}', f'{prob:.4f}'])
            else:
                writer.writerow([patient, int(label), f'{prob:.4f}', int(pred)])
    print(f"  Per-patient results saved to {csv_path}")

    if ax is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')

    return {
        'name': name, 'auc': auc, 'accuracy': accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'is_regression': is_regression,
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, _ = load_split("train")
    X_val,   y_val,   _ = load_split("val")
    X_test,  y_test,  p_test = load_split("test")

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    print(f"Train+Val: {X_trainval.shape}  positives: {y_trainval.sum()}")
    print(f"Test:      {X_test.shape}      positives: {y_test.sum()}")

    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled     = scaler.transform(X_test)

    n_pos = y_trainval.sum()
    n_neg = len(y_trainval) - n_pos
    cw    = {0: 1.0, 1: n_neg / n_pos}
    print(f"Class weight for positive: {cw[1]:.1f}")

    classifiers = [
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=5, class_weight=cw, random_state=42), False),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight=cw, random_state=42, n_jobs=-1), False),
        ("Logistic Regression",
         LogisticRegression(C=0.1, class_weight=cw, max_iter=1000, random_state=42), True),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, random_state=42), False),
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    all_results, results_text = [], []

    for name, clf, needs_scaling in classifiers:
        if needs_scaling:
            clf.fit(X_trainval_scaled, y_trainval)
            result = evaluate(name, clf, X_test_scaled, y_test, p_test, ax)
        else:
            clf.fit(X_trainval, y_trainval)
            result = evaluate(name, clf, X_test, y_test, p_test, ax)
        all_results.append(result)
        results_text.append(
            f"{name} ({'scoring' if result['is_regression'] else 'classification'}): "
            f"AUC={result['auc']:.4f}, Sens={result['sensitivity']:.4f}, "
            f"Spec={result['specificity']:.4f}"
        )

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves — Model 3 VNet + Classical ML')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150)
    print(f"\nROC curves saved to {ROC_PLOT}")

    print("\n" + "="*60)
    print("SUMMARY — Model 3 VNet Hybrid (Whole Mammograms)")
    print("="*60)
    print(f"{'Classifier':<25} {'Type':<15} {'AUC':>7} {'Sens':>7} {'Spec':>7}")
    print("-"*60)
    for r in all_results:
        t = 'scoring' if r['is_regression'] else 'classif.'
        print(f"{r['name']:<25} {t:<15} {r['auc']:>7.4f} "
              f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")

    with open(RESULTS_FILE, 'w') as f:
        f.write("HYBRID EXPERIMENT RESULTS — Model 3 VNet (Whole Mammograms)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train+Val patients: {len(X_trainval)} (positives: {int(y_trainval.sum())})\n")
        f.write(f"Test patients:      {len(X_test)} (positives: {int(y_test.sum())})\n\n")
        for line in results_text:
            f.write(line + "\n")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()