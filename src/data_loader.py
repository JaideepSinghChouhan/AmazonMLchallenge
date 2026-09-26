"""TSV loaders with strict validation and deterministic sampling."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from . import config as C

log = logging.getLogger(__name__)
REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS       = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")
    return df


def _read_and_sample(path: Path, required: list[str], n: int, seed: int, name: str) -> pd.DataFrame:
    """Read full file then deterministically sample. N<=0 means use all."""
    df = _read_tsv(path, required)
    if n <= 0 or len(df) <= n:
        log.info("[%s] using all %d rows", name, len(df))
        return df.reset_index(drop=True)
    log.warning("[%s] SAMPLING %d -> %d rows (seed=%d)", name, len(df), n, seed)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def _force_include_then_sample(df: pd.DataFrame, must_include: set,
                               n: int, seed: int, name: str) -> pd.DataFrame:
    """Keep every row whose entity_id is in must_include; sample the rest to reach n."""
    if not must_include:
        return _read_and_sample.__wrapped__ if False else df.sample(
            n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    df = df.reset_index(drop=True)
    forced = df[df["entity_id"].isin(must_include)]
    rest   = df[~df["entity_id"].isin(must_include)]

    n_rest = max(0, n - len(forced))
    if n_rest >= len(rest):
        sampled_rest = rest
    else:
        sampled_rest = rest.sample(n=n_rest, random_state=seed)

    combined = pd.concat([forced, sampled_rest], ignore_index=True)
    combined = combined.drop_duplicates(subset="entity_id").reset_index(drop=True)
    log.warning("[%s] forced=%d  sampled=%d  total=%d (target=%d)",
                name, len(forced), len(sampled_rest), len(combined), n)
    return combined
def load_train():
    # 1) Sample S1 first
    s1 = _read_and_sample(C.TRAIN_S1, REQUIRED_COLS, C.MAX_TRAIN_S1, C.RANDOM_SEED + 0, "train/S1")

    # 2) Read full GT, filter to sampled S1
    gt_full = _read_tsv(C.TRAIN_GT, GT_COLS)
    keep_s1 = set(s1["entity_id"])
    gt = gt_full[gt_full["source1_entity_id"].isin(keep_s1)].reset_index(drop=True)

    # 3) Collect every S2/S3 id referenced by this GT — force-include them
    referenced: set[str] = set()
    for raw in gt["matched_entity_ids"]:
        if isinstance(raw, str) and raw.strip():
            for mid in raw.split(","):
                mid = mid.strip()
                if mid:
                    referenced.add(mid)
    s2_ref = {m for m in referenced if m.startswith("S2-")}
    s3_ref = {m for m in referenced if m.startswith("S3-")}
    log.info("[train/GT] %d referenced S2 ids, %d referenced S3 ids",
             len(s2_ref), len(s3_ref))

    # 4) Read full S2/S3, force-include referenced ids, then sample the rest
    s2_full = _read_tsv(C.TRAIN_S2, REQUIRED_COLS)
    s3_full = _read_tsv(C.TRAIN_S3, REQUIRED_COLS)
    s2 = _force_include_then_sample(s2_full, s2_ref, C.MAX_TRAIN_S2,
                                    C.RANDOM_SEED + 1, "train/S2")
    s3 = _force_include_then_sample(s3_full, s3_ref, C.MAX_TRAIN_S3,
                                    C.RANDOM_SEED + 2, "train/S3")

    log.info("[train/GT] kept %d ground-truth rows for sampled S1", len(gt))
    return s1, s2, s3, gt


def load_test():
    s1 = _read_and_sample(C.TEST_S1, REQUIRED_COLS, C.MAX_TEST_S1, C.RANDOM_SEED + 10, "test/S1")
    s2 = _read_and_sample(C.TEST_S2, REQUIRED_COLS, C.MAX_TEST_S2, C.RANDOM_SEED + 11, "test/S2")
    s3 = _read_and_sample(C.TEST_S3, REQUIRED_COLS, C.MAX_TEST_S3, C.RANDOM_SEED + 12, "test/S3")
    return s1, s2, s3


def gt_to_dict(gt: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        s = set()
        if isinstance(raw, str) and raw.strip():
            s = {x.strip() for x in raw.split(",") if x.strip()}
        out[r["source1_entity_id"].strip()] = s
    return out


def data_integrity_diagnostic(gt: pd.DataFrame) -> dict:
    """Blueprint §2.5: does any S2/S3 id appear under >1 S1?"""
    from collections import defaultdict
    owner = defaultdict(set)
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        if not isinstance(raw, str) or not raw.strip():
            continue
        s1 = r["source1_entity_id"].strip()
        for mid in raw.split(","):
            mid = mid.strip()
            if mid:
                owner[mid].add(s1)
    duplicates = {k: v for k, v in owner.items() if len(v) > 1}
    log.info("[integrity] %d S2/S3 ids appear under >1 S1", len(duplicates))
    return duplicates