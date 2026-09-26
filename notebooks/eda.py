"""Headline EDA printed to stdout. Fill the notebook by copying this logic."""
from __future__ import annotations
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import data_loader as dl
from src import preprocessing as pp

logging.basicConfig(level=logging.INFO)


def _summ(name: str, df: pd.DataFrame):
    print(f"\n===== {name} =====")
    print(f"rows: {len(df)}")
    print(df.dtypes)
    print("\nmissing %:")
    print((df == "").mean().round(4) * 100)
    print("\nsample:")
    print(df.head(3).to_string())


def main():
    s1, s2, s3, gt = dl.load_train()
    t1, t2, t3 = dl.load_test()
    for nm, d in [("train S1", s1), ("train S2", s2), ("train S3", s3),
                  ("test S1", t1), ("test S2", t2), ("test S3", t3)]:
        _summ(nm, d)

    truth = dl.gt_to_dict(gt)
    lens = pd.Series({k: len(v) for k, v in truth.items()})
    print("\n===== ground-truth distribution =====")
    print("S1 with 0 matches (singleton):", int((lens == 0).sum()))
    print("S1 with 1 match:",                int((lens == 1).sum()))
    print("S1 with >1 match:",               int((lens > 1).sum()))
    s2_hit = sum(1 for v in truth.values() if any(x.startswith("S2-") for x in v))
    s3_hit = sum(1 for v in truth.values() if any(x.startswith("S3-") for x in v))
    print(f"S1 → S2 at least one: {s2_hit}; S1 → S3 at least one: {s3_hit}")

    # countries
    for nm, d in [("train S1", s1), ("train S2", s2), ("train S3", s3),
                  ("test S1", t1), ("test S2", t2), ("test S3", t3)]:
        print(f"{nm} country value_counts:\n{d['country'].value_counts(dropna=False)}\n")


if __name__ == "__main__":
    main()