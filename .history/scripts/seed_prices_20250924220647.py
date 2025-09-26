import pandas as pd
from pathlib import Path

# install first: pip install yfinance pandas pyarrow
import yfinance as yf

SYMS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","JPM","XOM","JNJ","PG"]
START, END = "2013-01-01", "2025-01-01"

rows = []
for s in SYMS:
    df = yf.download(s, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {s}. Check internet access/firewall.")
    df = df.rename(columns={"Adj Close": "adj_close"}).reset_index()
    df["ticker"] = s
    rows.append(df[["Date", "ticker", "adj_close"]].rename(columns={"Date": "date"}))

prices = pd.concat(rows, ignore_index=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Prefer parquet; fall back to CSV if pyarrow not present
try:
    prices.to_parquet("data/processed/prices.parquet", index=False)
    where = "data/processed/prices.parquet"
except Exception:
    prices.to_csv("data/processed/prices.csv", index=False)
    where = "data/processed/prices.csv"

print("Wrote", where, prices.shape)
