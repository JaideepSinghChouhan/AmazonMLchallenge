"""Three class-weighting variants selected by validation macro-F0.5."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from .decision import tune_threshold

log = logging.getLogger(__name__)
META = {"s1_id", "s2_id", "label"}


def _feature_cols(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in META]


def _weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "unweighted":
        return np.ones(len(y), dtype=np.float32)
    if mode == "moderate":
        return np.where(y == 1, 5.0, 1.0).astype(np.float32)
    if mode == "inverse_freq":
        pos = int(y.sum());  neg = len(y) - pos
        return np.where(y == 1, neg / max(pos, 1), 1.0).astype(np.float32)
    raise ValueError(mode)


def train_and_select(F_tr, y_tr, F_va, truth_va,
                     weight_modes=("unweighted", "moderate", "inverse_freq")):
    feats = _feature_cols(F_tr)
    X_tr = F_tr[feats].values
    X_va = F_va[feats].values

    results = {}
    best = (None, -1.0, None)
    for mode in weight_modes:
        log.info("[%s] training HGB on %d pairs (pos=%d)", mode, len(y_tr), int(y_tr.sum()))
        clf = HistGradientBoostingClassifier(**C.HGB_PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=_weights(y_tr, mode))
        scores_va = clf.predict_proba(X_va)[:, 1]

        val_df = pd.DataFrame({
            "s1_id": F_va["s1_id"].values,
            "s2_id": F_va["s2_id"].values,
            "score": scores_va,
        })
        t, sc = tune_threshold(val_df, truth_va)
        log.info("[%s] threshold=%.3f  val_macro_F0.5=%.4f", mode, t, sc)
        results[mode] = {"model": clf, "threshold": t, "score": sc,
                         "scores_va": scores_va}
        if sc > best[1]:
            best = (mode, sc, t)

    sel_mode, sel_score, sel_t = best
    log.info("Selected weight mode: %s (val=%.4f, threshold=%.3f)", sel_mode, sel_score, sel_t)
    return {
        "results": results,
        "selected_mode": sel_mode,
        "model": results[sel_mode]["model"],
        "threshold": results[sel_mode]["threshold"],
        "features": feats,
        "val_score": sel_score,
    }


def predict(model, F: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict_proba(F[feats].values)[:, 1]