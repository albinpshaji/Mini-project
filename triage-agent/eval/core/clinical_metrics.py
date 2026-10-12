"""
Clinical Acuity & Safety Metrics for Emergency Triage.
Implements Quadratic Weighted Kappa (KTAS 1-5), Under-Triage, Over-Triage, and Red-Flag Recall.
"""

from typing import List, Dict, Any
import numpy as np


def compute_quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> float:
    """
    Computes Quadratic Weighted Kappa (Cohen's Kappa with quadratic penalty) across KTAS tiers 1 to 5.
    Scale: -1.0 to +1.0 (>= 0.85 indicates near-perfect concordance with emergency physicians).
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0

    # Weight matrix: w_ij = (i - j)^2 / (N - 1)^2
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # Observed confusion matrix
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for yt, yp in zip(y_true, y_pred):
        # Bound to [1, num_classes]
        c_true = max(1, min(num_classes, int(yt))) - 1
        c_pred = max(1, min(num_classes, int(yp))) - 1
        conf_mat[c_true, c_pred] += 1.0

    total_samples = len(y_true)
    hist_true = conf_mat.sum(axis=1)
    hist_pred = conf_mat.sum(axis=0)

    # Expected matrix under chance
    expected = np.outer(hist_true, hist_pred) / total_samples

    numerator = np.sum(w * conf_mat)
    denominator = np.sum(w * expected)

    if denominator == 0.0:
        return 1.0

    kappa = 1.0 - (numerator / denominator)
    return round(float(kappa), 4)


def evaluate_clinical_safety(
    y_true_ktas: List[int],
    y_pred_ktas: List[int],
    red_flags_true: List[bool] = None,
    red_flags_pred: List[bool] = None,
) -> Dict[str, Any]:
    """
    Evaluates clinical safety metrics:
      - Quadratic Weighted Kappa (KTAS 1-5 agreement)
      - Under-Triage Rate (% of KTAS 1/2 cases classified as KTAS 3/4/5) -- CRITICAL MUST BE 0.0%
      - Over-Triage Rate (% of KTAS 4/5 cases escalated to KTAS 1/2)
      - Red-Flag Recall / Sensitivity
    """
    y_true = np.array(y_true_ktas, dtype=int)
    y_pred = np.array(y_pred_ktas, dtype=int)

    # 1. Under-Triage: True High Acuity (Level 1 or 2) misdiagnosed as Low (3, 4, 5)
    high_mask = np.isin(y_true, [1, 2])
    high_total = int(np.sum(high_mask))
    under_count = int(np.sum(high_mask & np.isin(y_pred, [3, 4, 5])))
    under_rate = (under_count / high_total * 100.0) if high_total > 0 else 0.0

    # 2. Over-Triage: True Low Acuity (Level 4 or 5) misdiagnosed as Emergent (1, 2)
    low_mask = np.isin(y_true, [4, 5])
    low_total = int(np.sum(low_mask))
    over_count = int(np.sum(low_mask & np.isin(y_pred, [1, 2])))
    over_rate = (over_count / low_total * 100.0) if low_total > 0 else 0.0

    # 3. Kappa Score
    kappa = compute_quadratic_weighted_kappa(y_true_ktas, y_pred_ktas)

    # 4. Red-Flag Sensitivity / Specificity if flags provided
    red_flag_metrics = {}
    if red_flags_true is not None and red_flags_pred is not None:
        rf_true = np.array(red_flags_true, dtype=bool)
        rf_pred = np.array(red_flags_pred, dtype=bool)

        tp = int(np.sum(rf_true & rf_pred))
        fn = int(np.sum(rf_true & (~rf_pred)))
        fp = int(np.sum((~rf_true) & rf_pred))
        tn = int(np.sum((~rf_true) & (~rf_pred)))

        sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
        specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 100.0
        red_flag_metrics = {
            "red_flag_sensitivity_pct": round(sensitivity, 2),
            "red_flag_specificity_pct": round(specificity, 2),
            "red_flag_false_negatives": fn,
            "red_flag_false_positives": fp,
        }

    # 5. Build 5x5 confusion matrix
    conf_mat = np.zeros((5, 5), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        c_true = max(1, min(5, int(yt))) - 1
        c_pred = max(1, min(5, int(yp))) - 1
        conf_mat[c_true, c_pred] += 1

    return {
        "ktas_quadratic_weighted_kappa": kappa,
        "under_triage_rate_percent": round(under_rate, 2),
        "over_triage_rate_percent": round(over_rate, 2),
        "high_acuity_cases_total": high_total,
        "under_triaged_cases_count": under_count,
        "safety_invariant_passed": (under_rate == 0.0),
        "confusion_matrix": conf_mat.tolist(),
        **red_flag_metrics,
    }
