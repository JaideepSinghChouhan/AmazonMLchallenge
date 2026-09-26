"""Entry point. Run: python run.py --mode all"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as C
from src.pipeline import run_train_val, run_test
from src import data_loader as dl


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eda", "train", "all"], default="all")
    args = ap.parse_args()
    setup_logging()

    if args.mode == "eda":
        s1, s2, s3, gt = dl.load_train()
        print(f"Train S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  GT={len(gt)}")
        truth = dl.gt_to_dict(gt)
        n_sing  = sum(1 for v in truth.values() if not v)
        n_one   = sum(1 for v in truth.values() if len(v) == 1)
        n_multi = sum(1 for v in truth.values() if len(v) > 1)
        print(f"GT: singletons={n_sing}  single={n_one}  multi={n_multi}")
        t1, t2, t3 = dl.load_test()
        print(f"Test  S1={len(t1)}  S2={len(t2)}  S3={len(t3)}")
        return

    result, blocker, stoplist = run_train_val()
    if args.mode == "all":
        run_test(blocker, stoplist, result)


if __name__ == "__main__":
    main()