

from typing import Tuple
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np

from typing import Tuple
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score
)


def search_best_f1_threshold(y_true,y_probs, thresholds) -> Tuple[float, float, float, float]:

    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_probs = np.asarray(y_probs).reshape(-1)


    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)

    best_f1 = 0.0
    best_thr = 0.5

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred, average="binary",zero_division=0)

        # Update best threshold and F1
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thr = float(t)

    return float(best_f1),float(best_thr),float(auprc),float(auroc)
    