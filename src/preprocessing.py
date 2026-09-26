"""Multi-representation text normalization. Missing -> "" for all normalize_*."""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown", "nil"}


def is_missing(v) -> bool:
    if v is None: return True
    if isinstance(v, float) and v != v: return True
    return str(v).strip().lower() in NULL_TOKENS


def safe_str(v) -> str:
    return "" if is_missing(v) else str(v)


# ---------- name ----------
NAME_ABBREV = {
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "ltd": "limited", "pvt": "private", "intl": "international",
    "mgmt": "management", "ent": "enterprises", "svc": "services",
    "svcs": "services", "bro": "brothers", "bros": "brothers",
}


def _clean(s) -> str:
    s = safe_str(s)
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^0-9a-z\u00c0-\u024f\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(v) -> str:
    s = _clean(v)
    if not s: return ""
    return " ".join(NAME_ABBREV.get(t, t) for t in s.split())


# ---------- address ----------
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway",
    "apt": "apartment", "fl": "floor", "bldg": "building", "bld": "building",
    "sec": "sector", "nr": "near", "opp": "opposite",
}


def normalize_address(v) -> str:
    s = _clean(v)
    if not s: return ""
    return " ".join(ADDR_ABBREV.get(t, t) for t in s.split())


# ---------- country ----------
def normalize_country(v) -> str:
    s = _clean(v)
    if not s: return ""
    aliases = {
        "united states": "usa", "united states of america": "usa",
        "u s a": "usa", "us": "usa",
        "in": "india", "bharat": "india",
        "fr": "france",
    }
    return aliases.get(s, s)


# ---------- dataframe helper ----------
def add_preprocessed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"]    = out["business_name"].map(normalize_name)
    out["addr_norm"]    = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_country)
    out["name_missing"]    = out["name_norm"].eq("")
    out["addr_missing"]    = out["addr_norm"].eq("")
    out["country_missing"] = out["country_norm"].eq("")
    return out