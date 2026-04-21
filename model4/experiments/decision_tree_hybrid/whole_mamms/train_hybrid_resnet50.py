"""
train_hybrid_resnet50.py

Trains classical ML classifiers on top of frozen ResNet50 3D features
for whole mammogram classification (Model 4).

Classifiers:
    - Decision Tree        → binary (0/1)
    - Random Forest        → binary (0/1)
    - Logistic Regression  → probability (0-1)
    - Gradient Boosting    → probability (0-1)

All classifiers save per-patient CSV with probability scores.
Run extract_features_resnet50.py first.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# -------------------------------------------------------------------------
FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'features')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
ROC_PLOT     = os.path.join(RESULTS_DIR, 'roc_curves_hybrid_resnet50.png')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'hybrid_results_resnet50.txt')
# -------------------------------------------------------------------------


def load_features(split):
    feats    = np.load(os.path.join(FEATURES_DIR, f"{split}_features.npy"))
    labels   = np.load(os.path.join(FEATURES_DIR, f"{split}_labels.npy"))
    patients = np.load(os.path.join(FEATURES_DIR, f"{split}_patients.npy"), allow_pickle=True)
    return feats, labels, patients


def evaluate(name, clf, X_test, y_test, patients_test, ax, uses_proba):
    if uses_proba:
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = clf.predict(X_test)
        probs = preds.astype(float)

    auc = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    # Per-patient CSV
    csv_name = name.lower().replace(' ', '_')
    csv_path = os.path.join(RESULTS_DIR, f"test_results_{csv_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'true_label', 'predicted_prob', 'predicted_label'])
        for pid, true, prob, pred in zip(patients_test, y_test, probs, preds):
            writer.writerow([pid, int(true), f"{prob:.4f}", int(pred)])
    print(f"  CSV saved: {csv_path}")

    print(f"  {name}: AUC={auc:.4f}  Sens={sensitivity:.4f}  Spec={specificity:.4f}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    return dict(name=name, auc=auc, sensitivity=sensitivity, specificity=specificity,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, _          = load_features("train")
    X_val,   y_val,   _          = load_features("val")
    X_test,  y_test,  p_test     = load_features("test")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled     = scaler.transform(X_test)

    print(f"Train+Val: {len(X_trainval)} patients  (pos={int(y_trainval.sum())})")
    print(f"Test:      {len(X_test)} patients  (pos={int(y_test.sum())})")

    # (clf, needs_scaling, uses_proba)
    classifiers = [
        ("Decision Tree",       DecisionTreeClassifier(max_depth=5, random_state=42),            False, False),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42),        False, False),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0, random_state=42),        True,  True),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42),    False, True),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)

    all_results   = []
    results_text  = []

    for name, clf, needs_scaling, uses_proba in classifiers:
        print(f"\n--- {name} ---")
        if needs_scaling:
            clf.fit(X_trainval_scaled, y_trainval)
            result = evaluate(name, clf, X_test_scaled, y_test, p_test, ax, uses_proba)
        else:
            clf.fit(X_trainval, y_trainval)
            result = evaluate(name, clf, X_test, y_test, p_test, ax, uses_proba)
        all_results.append(result)
        results_text.append(
            f"{name}: AUC={result['auc']:.4f}, "
            f"Sens={result['sensitivity']:.4f}, "
            f"Spec={result['specificity']:.4f}"
        )

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves — Model 4 ResNet50 3D + Classical ML (Whole Mammograms)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150)
    print(f"\nROC curves saved to {ROC_PLOT}")

    print("\n" + "="*60)
    print("SUMMARY — Model 4 ResNet50 3D Hybrid (Whole Mammograms)")
    print("="*60)
    print(f"{'Classifier':<25} {'AUC':>7} {'Sens':>7} {'Spec':>7}")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<25} {r['auc']:>7.4f} "
              f"{r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")

    with open(RESULTS_FILE, 'w') as f:
        f.write("HYBRID EXPERIMENT RESULTS — Model 4 ResNet50 3D (Whole Mammograms)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train+Val patients: {len(X_trainval)} (positives: {int(y_trainval.sum())})\n")
        f.write(f"Test patients:      {len(X_test)} (positives: {int(y_test.sum())})\n\n")
        for line in results_text:
            f.write(line + "\n")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()