"""Central config. All overridable by env vars."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT    = PROJECT_ROOT.parent
DATA_DIR     = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "dataset"))
TRAIN_DIR    = DATA_DIR / "train"
TEST_DIR     = DATA_DIR / "test"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Tier 0 sample sizes (blueprint §0.6) ---
def _env_int(name: str, default: int) -> int:
    try:    return int(os.environ.get(name, str(default)))
    except ValueError: return default

MAX_TRAIN_S1 = _env_int("BER_MAX_TRAIN_S1", 50_000)
MAX_TRAIN_S2 = _env_int("BER_MAX_TRAIN_S2", 500_000)
MAX_TRAIN_S3 = _env_int("BER_MAX_TRAIN_S3", 500_000)
MAX_TEST_S1  = _env_int("BER_MAX_TEST_S1",  50_000)
MAX_TEST_S2  = _env_int("BER_MAX_TEST_S2",  500_000)
MAX_TEST_S3  = _env_int("BER_MAX_TEST_S3",  500_000)

# --- S1-level split of train ---
VAL_FRACTION     = 0.15
HOLDOUT_FRACTION = 0.10   # remaining is train

# --- Retrieval (§4) ---
RETRIEVAL_TOP_K_PER_CHANNEL = 50
MAX_CANDIDATES_PER_S1       = 150

NAME_WORD_VEC = dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
NAME_CHAR_VEC = dict(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True)
ADDR_WORD_VEC = dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True)

# --- Stoplist (§3.1) ---
STOPLIST_DF_CUTOFF = 0.02
STOPLIST_SEED = [
    "corp","corporation","inc","incorporated","llc","ltd","limited",
    "pvt","private","plc","llp","co","company",
    "traders","trading","services","service","enterprises","enterprise",
    "industries","industry","store","stores","shop","shops",
    "group","holdings","international","intl",
    "the","and","of","a","an",
]

# --- Model (§6) ---
HGB_PARAMS = dict(
    max_iter=300,
    learning_rate=0.06,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=RANDOM_SEED,
)

# --- Threshold grid (§7) ---
THRESHOLD_GRID = [round(x, 3) for x in
    [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,
     0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]]