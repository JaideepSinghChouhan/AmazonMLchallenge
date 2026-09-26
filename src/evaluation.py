"""Four-case per-entity F0.5 evaluator. Mean-of-per-entity aggregation."""
from __future__ import annotations

BETA  = 0.5
BETA2 = 0.25


def entity_f05(pred: set, true: set) -> float:
    # Explicit four cases — no risk of 0/0
    if not pred and not true:  return 1.0
    if not pred and true:      return 0.0
    if pred and not true:      return 0.0
    # both non-empty
    tp = len(pred & true)
    if tp == 0: return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred_map: dict[str, set[str]], truth_map: dict[str, set[str]]) -> float:
    if not truth_map: return 0.0
    return sum(entity_f05(pred_map.get(s1, set()), true)
               for s1, true in truth_map.items()) / len(truth_map)


def candidate_recall(cand_map: dict[str, list[str]],
                     truth_map: dict[str, set[str]],
                     s1_ids: list[str]) -> tuple[float, int, int]:
    """Singletons (empty truth) excluded from both numerator and denominator."""
    hit, total = 0, 0
    for s1 in s1_ids:
        true = truth_map.get(s1, set())
        if not true:  continue
        cands = set(cand_map.get(s1, []))
        for g in true:
            total += 1
            if g in cands: hit += 1
    return (hit / total if total else float("nan")), hit, total