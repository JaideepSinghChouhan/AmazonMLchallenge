"""Vectorized pairwise feature computation. Missingness short-circuits."""
from __future__ import annotations
import logging
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"\d+")


def precompute_side(df: pd.DataFrame, stoplist: set) -> pd.DataFrame:
    out = df.copy()
    out["name_len"]    = out["name_norm"].str.len().astype(np.int32)
    out["addr_len"]    = out["addr_norm"].str.len().astype(np.int32)
    out["name_n_tok"]  = out["name_norm"].str.split().str.len().fillna(0).astype(np.int32)
    out["addr_n_tok"]  = out["addr_norm"].str.split().str.len().fillna(0).astype(np.int32)

    def _info(s: str) -> float:
        if not s: return 0.0
        toks = s.split()
        if not toks: return 0.0
        content = [t for t in toks if t not in stoplist]
        return len(content) / len(toks)

    out["name_info"] = out["name_norm"].map(_info).astype(np.float32)
    out["addr_nums"] = out["addr_norm"].map(lambda s: frozenset(_NUM_RE.findall(s)))
    return out


def compute_pair_features(pairs_df: pd.DataFrame,
                          left_df: pd.DataFrame,
                          right_df: pd.DataFrame) -> pd.DataFrame:
    """pairs_df columns: ['s1_id','s2_id']. Returns dense feature matrix."""
    n = len(pairs_df)
    log.info("Computing features for %d pairs", n)

    L = left_df.set_index("entity_id")
    R = right_df.set_index("entity_id")
    assert L.index.is_unique and R.index.is_unique

    s1_ids = pairs_df["s1_id"].values
    s2_ids = pairs_df["s2_id"].values

    # ---- per-record columns ----
    L_name  = L.loc[s1_ids, "name_norm"].values
    R_name  = R.loc[s2_ids, "name_norm"].values
    L_addr  = L.loc[s1_ids, "addr_norm"].values
    R_addr  = R.loc[s2_ids, "addr_norm"].values
    L_ctry  = L.loc[s1_ids, "country_norm"].values
    R_ctry  = R.loc[s2_ids, "country_norm"].values

    L_nm = L.loc[s1_ids, "name_missing"].values.astype(bool)
    R_nm = R.loc[s2_ids, "name_missing"].values.astype(bool)
    L_am = L.loc[s1_ids, "addr_missing"].values.astype(bool)
    R_am = R.loc[s2_ids, "addr_missing"].values.astype(bool)
    L_cm = L.loc[s1_ids, "country_missing"].values.astype(bool)
    R_cm = R.loc[s2_ids, "country_missing"].values.astype(bool)

    L_nlen = L.loc[s1_ids, "name_len"].values.astype(np.int32)
    R_nlen = R.loc[s2_ids, "name_len"].values.astype(np.int32)
    L_alen = L.loc[s1_ids, "addr_len"].values.astype(np.int32)
    R_alen = R.loc[s2_ids, "addr_len"].values.astype(np.int32)
    L_ntok = L.loc[s1_ids, "name_n_tok"].values.astype(np.int32)
    R_ntok = R.loc[s2_ids, "name_n_tok"].values.astype(np.int32)
    L_atok = L.loc[s1_ids, "addr_n_tok"].values.astype(np.int32)
    R_atok = R.loc[s2_ids, "addr_n_tok"].values.astype(np.int32)

    L_ninf = L.loc[s1_ids, "name_info"].values.astype(np.float32)
    R_ninf = R.loc[s2_ids, "name_info"].values.astype(np.float32)

    L_nums = L.loc[s1_ids, "addr_nums"].values
    R_nums = R.loc[s2_ids, "addr_nums"].values

    F: dict = {}

    # ---- Group E: missingness ----
    F["name_missing_L"]     = L_nm.astype(np.int8)
    F["name_missing_R"]     = R_nm.astype(np.int8)
    F["name_missing_both"]  = (L_nm & R_nm).astype(np.int8)
    F["name_missing_one"]   = (L_nm ^ R_nm).astype(np.int8)
    F["addr_missing_L"]     = L_am.astype(np.int8)
    F["addr_missing_R"]     = R_am.astype(np.int8)
    F["addr_missing_both"]  = (L_am & R_am).astype(np.int8)
    F["addr_missing_one"]   = (L_am ^ R_am).astype(np.int8)
    F["country_missing_L"]  = L_cm.astype(np.int8)
    F["country_missing_R"]  = R_cm.astype(np.int8)
    F["country_missing_both"] = (L_cm & R_cm).astype(np.int8)
    F["country_missing_one"]  = (L_cm ^ R_cm).astype(np.int8)

    # ---- Group E: lengths/tokens ----
    F["name_len_L"]       = L_nlen
    F["name_len_R"]       = R_nlen
    F["name_len_absdiff"] = np.abs(L_nlen - R_nlen).astype(np.int16)
    F["name_len_min"]     = np.minimum(L_nlen, R_nlen)
    F["name_len_max"]     = np.maximum(L_nlen, R_nlen)
    F["name_tok_L"]       = L_ntok
    F["name_tok_R"]       = R_ntok
    F["name_tok_absdiff"] = np.abs(L_ntok - R_ntok).astype(np.int16)
    F["name_tok_min"]     = np.minimum(L_ntok, R_ntok)
    F["name_tok_max"]     = np.maximum(L_ntok, R_ntok)
    F["addr_len_L"]       = L_alen
    F["addr_len_R"]       = R_alen
    F["addr_len_absdiff"] = np.abs(L_alen - R_alen).astype(np.int16)
    F["addr_len_min"]     = np.minimum(L_alen, R_alen)
    F["addr_len_max"]     = np.maximum(L_alen, R_alen)
    F["addr_tok_L"]       = L_atok
    F["addr_tok_R"]       = R_atok
    F["addr_tok_absdiff"] = np.abs(L_atok - R_atok).astype(np.int16)
    F["addr_tok_min"]     = np.minimum(L_atok, R_atok)
    F["addr_tok_max"]     = np.maximum(L_atok, R_atok)

    # ---- name similarity ----
    nm_miss = L_nm | R_nm
    F["name_exact"]      = ((L_name == R_name) & ~nm_miss).astype(np.int8)
    F["name_info_L"]     = L_ninf
    F["name_info_R"]     = R_ninf
    F["name_info_min"]   = np.minimum(L_ninf, R_ninf)

    log.info("  name ratio")
    name_ratio = np.fromiter(
        (fuzz.ratio(a, b) / 100.0 if (a and b) else np.nan
         for a, b in zip(L_name, R_name)),
        dtype=np.float32, count=n)
    name_ratio[nm_miss] = np.nan
    F["name_ratio"] = name_ratio

    log.info("  name token_set")
    name_ts = np.fromiter(
        (fuzz.token_set_ratio(a, b) / 100.0 if (a and b) else np.nan
         for a, b in zip(L_name, R_name)),
        dtype=np.float32, count=n)
    name_ts[nm_miss] = np.nan
    F["name_token_set"] = name_ts

    log.info("  name jaccard")
    name_jac = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if nm_miss[i]:  continue
        a, b = L_name[i], R_name[i]
        if not a or not b:  continue
        sa, sb = set(a.split()), set(b.split())
        if sa and sb:
            name_jac[i] = len(sa & sb) / len(sa | sb)
    F["name_jaccard"] = name_jac

    # ---- address similarity ----
    ad_miss = L_am | R_am
    F["addr_exact"] = ((L_addr == R_addr) & ~ad_miss).astype(np.int8)

    log.info("  addr ratio")
    addr_ratio = np.fromiter(
        (fuzz.ratio(a, b) / 100.0 if (a and b) else np.nan
         for a, b in zip(L_addr, R_addr)),
        dtype=np.float32, count=n)
    addr_ratio[ad_miss] = np.nan
    F["addr_ratio"] = addr_ratio

    log.info("  addr jaccard")
    addr_jac = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if ad_miss[i]:  continue
        a, b = L_addr[i], R_addr[i]
        if not a or not b:  continue
        sa, sb = set(a.split()), set(b.split())
        if sa and sb:
            addr_jac[i] = len(sa & sb) / len(sa | sb)
    F["addr_jaccard"] = addr_jac

    log.info("  addr numeric overlap")
    num_jac = np.full(n, np.nan, dtype=np.float32)
    num_conflict = np.zeros(n, dtype=np.int8)
    for i in range(n):
        a, b = L_nums[i], R_nums[i]
        if not a or not b:
            continue
        inter = len(a & b)
        num_jac[i] = inter / len(a | b)
        if inter == 0:
            num_conflict[i] = 1
    F["addr_num_jaccard"]  = num_jac
    F["addr_num_conflict"] = num_conflict

    # ---- country ----
    c_miss = L_cm | R_cm
    F["country_equal"]    = ((L_ctry == R_ctry) & ~c_miss).astype(np.int8)
    F["country_mismatch"] = ((L_ctry != R_ctry) & ~c_miss).astype(np.int8)

    # ---- source_pair ----
    F["is_s2"] = np.array([s.startswith("S2-") for s in s2_ids], dtype=np.int8)

    # ---- conflict interactions ----
    nr = np.nan_to_num(name_ratio, nan=0.0)
    aj = np.nan_to_num(addr_jac,   nan=0.0)
    F["conflict_name_high_addr_low"]  = (nr * (1.0 - aj)).astype(np.float32)
    F["conflict_addr_high_name_low"]  = (aj * (1.0 - nr)).astype(np.float32)
    F["conflict_name_high_num_conf"]  = (nr * num_conflict).astype(np.float32)
    F["conflict_country_high_name"]   = (F["country_mismatch"].astype(np.float32) * nr)

    F_df = pd.DataFrame(F)
    F_df["s1_id"] = s1_ids
    F_df["s2_id"] = s2_ids
    return F_df