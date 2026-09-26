"""Multi-channel candidate generation with separate S2/S3 pools (§4)."""
from __future__ import annotations
import logging
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config as C

log = logging.getLogger(__name__)

_BATCH = 64 # rows per query batch in dense top-k


class TFIDFChannel:
    """One (channel, source) index."""

    def __init__(self, name: str, vec_params: dict):
        self.name = name
        self.vectorizer = TfidfVectorizer(**vec_params)
        self.matrix = None   # csr, float32, L2-normalized
        self.ids: List[str] = []

    def fit(self, right_df: pd.DataFrame, text_col: str):
        corpus = right_df[text_col].fillna("").tolist()
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.matrix = normalize(self.matrix, copy=True).astype(np.float32)
        self.ids = right_df["entity_id"].tolist()
        log.info("[%s] fit on %d docs, vocab=%d, nnz=%d",
                 self.name, len(corpus), len(self.vectorizer.vocabulary_), self.matrix.nnz)

    def query(self, left_df: pd.DataFrame, text_col: str, top_k: int):
        corpus = left_df[text_col].fillna("").tolist()
        if not any(corpus):
            return [[] for _ in corpus]
        qmat = self.vectorizer.transform(corpus)
        qmat = normalize(qmat, copy=True).astype(np.float32)
        return self._top_k(qmat, top_k)

    def _top_k(self, qmat, top_k: int):
        """Returns List[List[Tuple[id, score]]] — top-k by score."""
        n_q = qmat.shape[0]
        out = [[] for _ in range(n_q)]
        _B = 64

        for i in range(0, n_q, _B):
            end = min(i + _B, n_q)
            sim = (qmat[i:end] @ self.matrix.T).tocsr()
            for r in range(sim.shape[0]):
                start, stop = sim.indptr[r], sim.indptr[r + 1]
                if stop == start:
                    continue
                idx  = sim.indices[start:stop]
                vals = sim.data[start:stop]
                if len(vals) <= top_k:
                    pairs = [(self.ids[j], float(v))
                             for j, v in zip(idx, vals) if v > 1e-6]
                else:
                    top_local = np.argpartition(-vals, top_k - 1)[:top_k]
                    pairs = [(self.ids[idx[j]], float(vals[j]))
                             for j in top_local if vals[j] > 1e-6]
                pairs.sort(key=lambda x: -x[1])
                out[i + r] = pairs
        return out

class CandidateGenerator:
    """Channels A (name word), B (name char), C-lite (addr word) per source."""

    def __init__(self,
                 top_k: int = C.RETRIEVAL_TOP_K_PER_CHANNEL,
                 max_per_s1: int = C.MAX_CANDIDATES_PER_S1):
        self.top_k = top_k
        self.max_per_s1 = max_per_s1
        self.channels: dict = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        for src, df in [("S2", s2_df), ("S3", s3_df)]:
            for ch, params, col in [
                ("name_word", C.NAME_WORD_VEC, "name_norm"),
                ("name_char", C.NAME_CHAR_VEC, "name_norm"),
                ("addr_word", C.ADDR_WORD_VEC, "addr_norm"),
            ]:
                idx = TFIDFChannel(f"{src}/{ch}", params)
                idx.fit(df, col)
                self.channels[(src, ch)] = idx

    def query(self, s1_df: pd.DataFrame) -> dict[str, list[str]]:
        s1_ids = s1_df["entity_id"].tolist()
        best: dict[str, dict[str, float]] = {s: {} for s in s1_ids}

        for src in ("S2", "S3"):
            for ch, col in [("name_word", "name_norm"),
                            ("name_char", "name_norm"),
                            ("addr_word", "addr_norm")]:
                idx = self.channels[(src, ch)]
                log.info("[%s/%s] querying %d S1 ...", src, ch, len(s1_df))
                res = idx.query(s1_df, col, self.top_k)
                for s1, cands in zip(s1_ids, res):
                    d = best[s1]
                    for cid, sc in cands:
                        if sc > d.get(cid, 0.0):
                            d[cid] = sc

        out: dict[str, list[str]] = {}
        for s1, d in best.items():
            if len(d) > self.max_per_s1:
                top = sorted(d.items(), key=lambda kv: -kv[1])[:self.max_per_s1]
                cands = {c for c, _ in top}
            else:
                cands = set(d.keys())
            out[s1] = sorted(cands)
        return out


def build_stoplist(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                   seed_words: list[str] | None = None,
                   df_cutoff: float | None = None) -> set[str]:
    """Stoplist from S1+S2+S3 corpus (§3.1). Unsupervised."""
    seed_words = seed_words or C.STOPLIST_SEED
    df_cutoff  = C.STOPLIST_DF_CUTOFF if df_cutoff is None else df_cutoff

    names = pd.concat([s1_df["name_norm"], s2_df["name_norm"], s3_df["name_norm"]],
                      ignore_index=True)
    counter: Counter = Counter()
    n_docs = 0
    for name in names:
        if not name:  continue
        n_docs += 1
        for tok in set(name.split()):
            counter[tok] += 1

    threshold = df_cutoff * max(n_docs, 1)
    stoplist  = set(seed_words)
    for tok, cnt in counter.items():
        if cnt > threshold:
            stoplist.add(tok)
    log.info("Stoplist: %d tokens (seed=%d, +DF=%d), cutoff=%.3f, n_docs=%d",
             len(stoplist), len(seed_words), len(stoplist) - len(seed_words),
             df_cutoff, n_docs)
    return stoplist