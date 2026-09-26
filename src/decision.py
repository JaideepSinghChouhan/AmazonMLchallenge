"""Entity-level resolver. Threshold tuning with highest-tie-break (§7)."""
from __future__ import annotations
import logging
import pandas as pd
from . import config as C
from .evaluation import macro_f05

log = logging.getLogger(__name__)


def apply_threshold(scores_df: pd.DataFrame, threshold: float) -> dict[str, set[str]]:
    keep = scores_df[scores_df["score"] >= threshold]
    out: dict[str, set[str]] = {}
    for s1, grp in keep.groupby("s1_id", sort=False):
        out[s1] = set(grp["s2_id"].tolist())
    return out


def tune_threshold(scores_df: pd.DataFrame,
                   truth_map: dict[str, set[str]],
                   grid=None) -> tuple[float, float]:
    grid = grid if grid is not None else C.THRESHOLD_GRID
    best_score = -1.0
    best_ts: list[float] = []
    for t in grid:
        sc = macro_f05(apply_threshold(scores_df, t), truth_map)
        if sc > best_score + 1e-9:
            best_score, best_ts = sc, [t]
        elif abs(sc - best_score) <= 1e-9:
            best_ts.append(t)
    # Tie-break: highest threshold
    best_t = max(best_ts) if best_ts else 0.5
    log.info("Threshold tuned: t=%.3f, macro-F0.5=%.4f (ties: %s)",
             best_t, best_score, best_ts)
    return best_t, best_score