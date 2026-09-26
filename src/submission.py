"""Output writers + in-pipeline validator (mirrors official rules)."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def _fmt(ids) -> str:
    return ",".join(sorted(set(ids)))


def write_matching_results(path: Path, all_s1: list[str], pred: dict[str, set[str]]):
    rows = [(s, _fmt(pred.get(s, set()))) for s in all_s1]
    pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]) \
      .to_csv(path, sep="\t", index=False)
    log.info("Wrote %s (%d rows)", path, len(rows))


def write_candidate_pairs(path: Path, all_s1: list[str], cand: dict[str, list[str]]):
    rows = [(s, _fmt(cand.get(s, []))) for s in all_s1]
    pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]) \
      .to_csv(path, sep="\t", index=False)
    log.info("Wrote %s (%d rows)", path, len(rows))


def validate_outputs(matching_path: Path, candidate_path: Path,
                     test_s1: list[str], test_s2: set[str], test_s3: set[str]) -> list[str]:
    errs: list[str] = []
    m = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)

    if list(m.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errs.append(f"matching bad cols: {list(m.columns)}")
    if list(c.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        errs.append(f"candidate bad cols: {list(c.columns)}")

    valid_ids = test_s2 | test_s3
    expected_s1 = set(test_s1)

    for name, df, col in [("matching", m, "matched_entity_ids"),
                          ("candidate", c, "candidate_entity_ids")]:
        if df["source1_entity_id"].duplicated().any():
            errs.append(f"{name}: duplicate s1 rows")
        got = set(df["source1_entity_id"])
        if got != expected_s1:
            miss = expected_s1 - got;  extra = got - expected_s1
            if miss:  errs.append(f"{name}: missing {len(miss)} S1 ids")
            if extra: errs.append(f"{name}: extra {len(extra)} unknown S1 ids")
        for rid, val in zip(df["source1_entity_id"], df[col]):
            if not val: continue
            parts = val.split(",")
            if any(not p for p in parts):
                errs.append(f"{name}: empty id for {rid}")
            if len(set(parts)) != len(parts):
                errs.append(f"{name}: duplicate id for {rid}")
            for p in parts:
                if p.startswith("S1-"):
                    errs.append(f"{name}: S1 id inside list for {rid}")
                elif p not in valid_ids:
                    errs.append(f"{name}: unknown id {p} for {rid}")

    # matching subset of candidate
    cmap = {r: set(v.split(",")) if v else set()
            for r, v in zip(c["source1_entity_id"], c["candidate_entity_ids"])}
    for rid, v in zip(m["source1_entity_id"], m["matched_entity_ids"]):
        pred = set(v.split(",")) if v else set()
        if not pred.issubset(cmap.get(rid, set())):
            errs.append(f"matching not subset of candidate for {rid}")
    return errs