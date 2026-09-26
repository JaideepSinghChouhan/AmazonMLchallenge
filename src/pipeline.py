"""End-to-end orchestration."""
from __future__ import annotations
import json
import logging
import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data_loader as dl
from . import preprocessing as pp
from . import evaluation as ev
from . import submission as sub
from . import model as md
from . import decision as dec
from .blocking import CandidateGenerator, build_stoplist
from .features import compute_pair_features, precompute_side

log = logging.getLogger(__name__)


def _split_s1(s1: pd.DataFrame):
    ids = s1["entity_id"].tolist()
    n = len(ids)
    n_val  = int(n * C.VAL_FRACTION)
    n_hold = int(n * C.HOLDOUT_FRACTION)
    n_tr   = n - n_val - n_hold
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = rng.permutation(ids)
    return set(perm[:n_tr]), set(perm[n_tr:n_tr + n_val]), set(perm[n_tr + n_val:])


def _pairs(cand_map, keep_s1):
    rows = [(s, c) for s, cs in cand_map.items() if s in keep_s1 for c in cs]
    return pd.DataFrame(rows, columns=["s1_id", "s2_id"])


def _label(pairs_df, truth):
    pairs_df = pairs_df.copy()
    pairs_df["label"] = [
        1 if c in truth.get(s1, set()) else 0
        for s1, c in zip(pairs_df["s1_id"], pairs_df["s2_id"])
    ]
    return pairs_df


def run_train_val():
    log.info("=== LOAD TRAIN ===")
    s1, s2, s3, gt = dl.load_train()
    log.info("S1=%d S2=%d S3=%d GT=%d", len(s1), len(s2), len(s3), len(gt))
    dl.data_integrity_diagnostic(gt)

    log.info("=== PREPROCESS ===")
    s1 = pp.add_preprocessed_columns(s1)
    s2 = pp.add_preprocessed_columns(s2)
    s3 = pp.add_preprocessed_columns(s3)

    log.info("=== SPLIT S1 ===")
    tr_ids, va_ids, ho_ids = _split_s1(s1)
    assert not (tr_ids & va_ids) and not (tr_ids & ho_ids) and not (va_ids & ho_ids)
    log.info("split: train=%d val=%d holdout=%d", len(tr_ids), len(va_ids), len(ho_ids))

    log.info("=== STOPLIST (S1+S2+S3) ===")
    stoplist = build_stoplist(s1, s2, s3)
    with open(C.MODELS_DIR / "stoplist.json", "w") as f:
        json.dump({"size": len(stoplist), "cutoff": C.STOPLIST_DF_CUTOFF,
                   "tokens": sorted(stoplist)}, f)

    log.info("=== PRECOMPUTE SIDE FEATURES ===")
    s1 = precompute_side(s1, stoplist)
    s2 = precompute_side(s2, stoplist)
    s3 = precompute_side(s3, stoplist)

    log.info("=== FIT BLOCKER (separate S2/S3 pools) ===")
    blocker = CandidateGenerator()
    blocker.fit(s2, s3)

    log.info("=== CANDIDATES ===")
    s1_tr = s1[s1["entity_id"].isin(tr_ids)].reset_index(drop=True)
    s1_va = s1[s1["entity_id"].isin(va_ids)].reset_index(drop=True)
    cand_tr = blocker.query(s1_tr)
    cand_va = blocker.query(s1_va)

    truth = dl.gt_to_dict(gt)
    truth_tr = {k: v for k, v in truth.items() if k in tr_ids}
    truth_va = {k: v for k, v in truth.items() if k in va_ids}

    r_tr, h_tr, t_tr = ev.candidate_recall(cand_tr, truth_tr, list(tr_ids))
    r_va, h_va, t_va = ev.candidate_recall(cand_va, truth_va, list(va_ids))
    log.info("Candidate recall: train=%.4f (%d/%d)  val=%.4f (%d/%d)",
             r_tr, h_tr, t_tr, r_va, h_va, t_va)

    log.info("=== FEATURES ===")
    all_right = pd.concat([s2, s3], ignore_index=True)

    p_tr = _label(_pairs(cand_tr, tr_ids), truth_tr)
    log.info("train pairs=%d pos=%d", len(p_tr), int(p_tr["label"].sum()))
    F_tr = compute_pair_features(p_tr[["s1_id", "s2_id"]], s1, all_right)
    F_tr["label"] = p_tr["label"].values

    p_va = _label(_pairs(cand_va, va_ids), truth_va)
    log.info("val pairs=%d pos=%d", len(p_va), int(p_va["label"].sum()))
    F_va = compute_pair_features(p_va[["s1_id", "s2_id"]], s1, all_right)
    F_va["label"] = p_va["label"].values

    log.info("=== TRAIN (3 class-weighting variants) ===")
    result = md.train_and_select(F_tr, F_tr["label"].values,
                                 F_va, truth_va)

    log.info("=== VALIDATION DETAIL ===")
    val_scores = result["results"][result["selected_mode"]]["scores_va"]
    val_df = pd.DataFrame({
        "s1_id": F_va["s1_id"].values,
        "s2_id": F_va["s2_id"].values,
        "score": val_scores,
    })
    pred_va = dec.apply_threshold(val_df, result["threshold"])
    log.info("val macro-F0.5 = %.4f", ev.macro_f05(pred_va, truth_va))
    log.info("val GT singletons=%d, pred singletons=%d",
             sum(1 for v in truth_va.values() if not v),
             sum(1 for s1 in truth_va if not pred_va.get(s1)))
    log.info("val GT multi-match=%d, pred multi-match=%d",
             sum(1 for v in truth_va.values() if len(v) > 1),
             sum(1 for s1 in truth_va if len(pred_va.get(s1, set())) > 1))

    joblib.dump({
        "model": result["model"],
        "features": result["features"],
        "threshold": result["threshold"],
        "selected_mode": result["selected_mode"],
        "val_score": result["val_score"],
    }, C.MODELS_DIR / "ber_model.joblib")
    log.info("Model saved.")

    return result, blocker, stoplist


def run_test(blocker, stoplist, result):
    log.info("=== LOAD TEST ===")
    t1, t2, t3 = dl.load_test()
    log.info("test S1=%d S2=%d S3=%d", len(t1), len(t2), len(t3))

    t1 = pp.add_preprocessed_columns(t1)
    t2 = pp.add_preprocessed_columns(t2)
    t3 = pp.add_preprocessed_columns(t3)

    t1 = precompute_side(t1, stoplist)
    t2 = precompute_side(t2, stoplist)
    t3 = precompute_side(t3, stoplist)

    # ⚠️ CRITICAL: refit blocker on TEST S2/S3 — training blocker's index
    # returns TRAIN S2/S3 ids which don't exist in the test pool.
    log.info("=== FIT TEST BLOCKER (fresh on test S2/S3) ===")
    test_blocker = CandidateGenerator()
    test_blocker.fit(t2, t3)

    log.info("=== CANDIDATES (test) ===")
    cand_test = test_blocker.query(t1)
    log.info("test total candidates=%d", sum(len(v) for v in cand_test.values()))

    log.info("=== FEATURES (test) ===")
    all_right = pd.concat([t2, t3], ignore_index=True)
    pairs = _pairs(cand_test, set(t1["entity_id"]))
    log.info("test pairs=%d", len(pairs))
    F_test = compute_pair_features(pairs[["s1_id", "s2_id"]], t1, all_right)

    log.info("=== SCORE + DECIDE ===")
    F_test["score"] = md.predict(result["model"], F_test, result["features"])
    pred_map = dec.apply_threshold(F_test[["s1_id", "s2_id", "score"]],
                                   result["threshold"])

    all_s1 = t1["entity_id"].tolist()
    sub.write_matching_results(C.OUTPUT_DIR / "matching_results.tsv", all_s1, pred_map)
    sub.write_candidate_pairs(C.OUTPUT_DIR / "candidate_pairs.tsv", all_s1, cand_test)

    log.info("=== VALIDATE OUTPUTS ===")
    errs = sub.validate_outputs(
        C.OUTPUT_DIR / "matching_results.tsv",
        C.OUTPUT_DIR / "candidate_pairs.tsv",
        all_s1, set(t2["entity_id"]), set(t3["entity_id"]))
    if errs:
        for e in errs[:25]:
            log.error("  %s", e)
    else:
        log.info("Output validation PASSED")

    n_pred_sing = sum(1 for s in all_s1 if not pred_map.get(s))
    n_pred_multi = sum(1 for s in all_s1 if len(pred_map.get(s, set())) > 1)
    log.info("=== TEST SUMMARY ===")
    log.info("S1 total=%d | singletons predicted=%d | any-match=%d | multi-match=%d",
             len(all_s1), n_pred_sing, len(all_s1) - n_pred_sing, n_pred_multi)
    return {"errors": errs}