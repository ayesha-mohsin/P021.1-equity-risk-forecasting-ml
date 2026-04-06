# =============================================================================
# src/data_loader.py
# PURPOSE: Download and merge all market data from free public sources.
#
# SOURCES:
#   - Yahoo Finance (yfinance): 8 ETF prices + VIX
#   - Fama-French Data Library: 3-factor model data
#
# OUTPUT:
#   - data/processed/prices.parquet   — daily adjusted close prices
#   - data/processed/merged.parquet   — prices + returns + FF factors + VIX
#
# USAGE (standalone):
#   python src/data_loader.py
# =============================================================================

import os
import io
import zipfile
import requests
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS   = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "EFA", "EEM"]
START     = "2007-01-01"
END       = "2024-12-31"
FF_URL    = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
             "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
PROCESSED = "data/processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dirs():
    os.makedirs(PROCESSED, exist_ok=True)


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def load_yahoo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices for all tickers.

    Returns
    -------
    prices : DataFrame  shape (T, N)  — daily adjusted close
    returns : DataFrame shape (T, N)  — daily log returns (shifted 1 day)

    Why log returns?
    Log returns are additive across time and approximately normally
    distributed, making them better suited for ML features than simple
    percentage returns.
    """
    print(f"  Downloading {len(TICKERS)} tickers from Yahoo Finance...")
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when >1 ticker
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    # Drop days where ALL tickers are NaN (e.g. non-US holidays)
    prices = prices.dropna(how="all")

    # Forward-fill up to 2 days for minor gaps (e.g. single-ticker halts)
    prices = prices.ffill(limit=2)

    # Log returns: ln(P_t / P_{t-1})
    returns = prices.apply(lambda col: (col / col.shift(1)).apply(
        lambda x: x if pd.isna(x) else x.__class__.__mro__  # placeholder
    ))
    import numpy as np
    returns = np.log(prices / prices.shift(1))
    returns.columns = [f"ret_{t}" for t in TICKERS]

    print(f"  Prices: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    return prices, returns


# ---------------------------------------------------------------------------
# Fama-French 3-Factor Model
# ---------------------------------------------------------------------------
def load_fama_french() -> pd.DataFrame:
    """
    Download Fama-French daily 3-factor data from Ken French's data library.

    Factors returned:
      Mkt-RF : excess market return (market return minus risk-free rate)
      SMB    : Small Minus Big — size factor
      HML    : High Minus Low  — value factor
      RF     : daily risk-free rate (annualised ÷ 252)

    All values are in percentage points; we divide by 100 to get decimals.

    Why Fama-French?
    These factors capture systematic risk exposures that explain most of
    equity returns. If our ML model's 'alpha' disappears after controlling
    for these factors, it's not genuinely predictive — it's just a factor tilt.
    """
    print("  Downloading Fama-French factors...")
    try:
        resp = requests.get(FF_URL, timeout=30)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
        raw_text = zf.read(csv_name).decode("utf-8", errors="ignore")

        # The CSV has a text header before the data starts
        lines = raw_text.split("\n")
        data_start = next(i for i, l in enumerate(lines) if l.strip().startswith("2"))
        data_end = next(
            (i for i, l in enumerate(lines[data_start:], data_start)
             if l.strip() == "" and i > data_start + 5),
            len(lines)
        )
        csv_block = "\n".join(lines[data_start:data_end])
        ff = pd.read_csv(io.StringIO(csv_block), header=None,
                         names=["Date", "Mkt-RF", "SMB", "HML", "RF"])

        ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m%d", errors="coerce")
        ff = ff.dropna(subset=["Date"]).set_index("Date")
        ff = ff.apply(pd.to_numeric, errors="coerce") / 100  # convert % → decimal
        ff = ff[(ff.index >= START) & (ff.index <= END)]
        print(f"  Fama-French: {ff.shape[0]} rows")
        return ff

    except Exception as e:
        print(f"  WARNING: Fama-French download failed ({e}). Using zeros.")
        idx = pd.date_range(START, END, freq="B")
        return pd.DataFrame(0.0, index=idx, columns=["Mkt-RF", "SMB", "HML", "RF"])


# ---------------------------------------------------------------------------
# VIX
# ---------------------------------------------------------------------------
def load_vix() -> pd.DataFrame:
    """
    Download CBOE VIX index from Yahoo Finance.

    VIX measures implied volatility of S&P 500 options over the next 30 days.
    It is a forward-looking 'fear gauge' — spikes during market stress even
    before realized volatility rises. This gives the model early warning signal.
    """
    print("  Downloading VIX...")
    vix_raw = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix_raw["Close"].rename("VIX")
    vix.index = pd.to_datetime(vix.index)
    vix.index.name = "Date"
    vix = vix.ffill(limit=5)
    print(f"  VIX: {vix.shape[0]} rows")
    return vix.to_frame()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_all(prices: pd.DataFrame,
              returns: pd.DataFrame,
              ff: pd.DataFrame,
              vix: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join all data sources on their date index.

    Inner join ensures we only keep days present in ALL sources, preventing
    NaN-filled rows that would corrupt downstream feature engineering.
    """
    print("  Merging datasets...")
    merged = (prices
              .join(returns, how="inner")
              .join(ff,      how="inner")
              .join(vix,     how="inner"))
    merged = merged.sort_index()
    merged = merged.dropna(thresh=int(0.8 * merged.shape[1]))  # drop sparse rows
    print(f"  Merged: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(f"  Date range: {merged.index[0].date()} → {merged.index[-1].date()}")
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _ensure_dirs()
    prices, returns = load_yahoo()
    ff  = load_fama_french()
    vix = load_vix()
    merged = merge_all(prices, returns, ff, vix)

    prices.to_parquet(f"{PROCESSED}/prices.parquet")
    merged.to_parquet(f"{PROCESSED}/merged.parquet")
    print(f"\nSaved to {PROCESSED}/")
