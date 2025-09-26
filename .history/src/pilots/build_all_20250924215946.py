from pathlib import Path
import pandas as pd
import re

PRICES_PARQUET = Path("data/processed/prices.parquet")

def _norm_cols(cols):
    # lower, strip, turn spaces & non-alnum to underscores, collapse repeats
    def norm(c):
        c = str(c).strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        return re.sub(r"_+", "_", c).strip("_")
    return [norm(c) for c in cols]

def load_prices() -> pd.DataFrame:
    if not PRICES_PARQUET.exists():
        raise FileNotFoundError(f"Missing {PRICES_PARQUET}. Seed prices first.")

    df = pd.read_parquet(PRICES_PARQUET)

    # If date is the index, move it to a column
    if "date" not in {str(getattr(df.index, "name", "")).lower()} and "date" not in [str(c).lower() for c in df.columns]:
        # try to reset anyway; if it’s a RangeIndex this is harmless
        df = df.reset_index()

    # Normalize column names
    df.columns = _norm_cols(df.columns)

    # Common aliases
    alias_map = {}
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            alias_map["adjclose"] = "adj_close"
        elif "adjusted_close" in df.columns:
            alias_map["adjusted_close"] = "adj_close"
        elif "close" in df.columns:
            # fallback to close if adj not present
            alias_map["close"] = "adj_close"

    if "ticker" not in df.columns:
        if "symbol" in df.columns:
            alias_map["symbol"] = "ticker"
        elif "tic" in df.columns:
            alias_map["tic"] = "ticker"
        elif "name" in df.columns:
            alias_map["name"] = "ticker"

    if "date" not in df.columns:
        # if the index name was date, after reset it becomes 'index'
        if "index" in df.columns:
            alias_map["index"] = "date"

    if alias_map:
        df = df.rename(columns=alias_map)

    # Final sanity
    required = {"date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"prices.parquet is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["date", "ticker", "adj_close"]].copy()
    # ensure dtypes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df
