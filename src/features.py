# =============================================================================
# src/features.py
# PURPOSE: Engineer 27 predictive features from raw market data.
#          Also creates regression and classification target variables.
#
# FEATURE GROUPS (27 total):
#   1. Lag returns        — momentum / mean-reversion signals
#   2. Rolling stats      — volatility, mean, skewness (regime detection)
#   3. VIX features       — fear gauge signals
#   4. Fama-French        — systematic factor exposures
#   5. 200-day MA signal  — trend regime (bull vs bear)
#   6. Momentum divergence — short vs long-term momentum breakdown
#   7. Volatility regime  — short vs long vol ratio (spike detection)
#   8. Flight-to-safety   — SPY/TLT rolling correlation (risk-off signal)
#   9. Peak drawdown      — distance from 252-day high (#1 SHAP feature)
#
# TARGET VARIABLES:
#   target_vol_fwd      — realized vol of SPY over next 5 trading days (regression)
#   target_crash        — 1 if SPY return next 5 days < -2%, else 0 (classification)
#
# USAGE (standalone):
#   python src/features.py
# =============================================================================

import os
import numpy as np
import pandas as pd

PROCESSED = "data/processed"
TICKERS   = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "EFA", "EEM"]
CRASH_THRESHOLD = -0.02   # -2% over 5 days = crash label
FWD_WINDOW      = 5       # days ahead for targets


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 27 engineered features and both target variables to the DataFrame.

    IMPORTANT — no look-ahead bias:
      All features use only past data (shift(1) or rolling on past).
      Targets use FORWARD data (shift(-FWD_WINDOW)) and are only used
      as labels, never as features during training.
    """
    out = df.copy()

    spy = df["SPY"]          # SPY adjusted close
    spy_ret = df["ret_SPY"] if "ret_SPY" in df.columns else np.log(spy / spy.shift(1))

    # ------------------------------------------------------------------
    # GROUP 1: Lag returns (momentum / reversal)
    # Why: Past returns predict future returns in both momentum
    # (1-12 month) and mean-reversion (1-5 day) regimes.
    # ------------------------------------------------------------------
    for lag in [1, 2, 5, 10, 21]:
        out[f"ret_{lag}d"] = spy_ret.shift(1).rolling(lag).sum()

    # ------------------------------------------------------------------
    # GROUP 2: Rolling statistics (distributional regime)
    # Why: Volatility clusters — high vol today → high vol tomorrow.
    # Negative skewness flags left-tail risk (crash-prone environment).
    # ------------------------------------------------------------------
    for window in [5, 21]:
        out[f"vol_{window}d"]  = spy_ret.shift(1).rolling(window).std()
        out[f"mean_{window}d"] = spy_ret.shift(1).rolling(window).mean()
    out["skew_21d"] = spy_ret.shift(1).rolling(21).skew()

    # ------------------------------------------------------------------
    # GROUP 3: VIX features (forward-looking fear)
    # Why: VIX is option-implied vol — a FORWARD-looking fear measure.
    # It spikes before realized vol rises, giving early warning signal.
    # VIX/vol ratio: when implied vol >> realized vol, fear premium is high.
    # ------------------------------------------------------------------
    vix = df["VIX"].shift(1)
    out["vix_level"]     = vix
    out["vix_change_5d"] = vix - vix.shift(5)
    out["vix_change_21d"] = vix - vix.shift(21)
    realized_vol = out["vol_21d"].replace(0, np.nan)
    out["vix_vol_ratio"] = vix / (realized_vol * np.sqrt(252))  # annualize realized vol

    # ------------------------------------------------------------------
    # GROUP 4: Fama-French factors (systematic risk exposures)
    # Why: These capture known risk premia. Including them lets the model
    # isolate genuine predictive signal vs factor-driven returns.
    # ------------------------------------------------------------------
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        if col in df.columns:
            out[col] = df[col].shift(1)

    # ------------------------------------------------------------------
    # GROUP 5: 200-day moving average signal (trend regime)
    # Why: Institutional investors widely use the 200-day MA as a
    # bull/bear regime filter. Below = bear regime = higher crash risk.
    # Binary feature: 1 (above MA, bullish) or 0 (below MA, bearish).
    # ------------------------------------------------------------------
    ma200 = spy.shift(1).rolling(200).mean()
    out["ma200_signal"] = (spy.shift(1) > ma200).astype(float)

    # ------------------------------------------------------------------
    # GROUP 6: Momentum divergence (breakdown signal)
    # Why: When short-term momentum breaks sharply from long-term trend,
    # it signals instability — either a reversal or an acceleration.
    # Formula: 5-day return / 63-day return (normalized divergence).
    # ------------------------------------------------------------------
    mom_short = spy_ret.shift(1).rolling(5).sum()
    mom_long  = spy_ret.shift(1).rolling(63).sum().replace(0, np.nan)
    out["mom_div"] = mom_short / mom_long

    # ------------------------------------------------------------------
    # GROUP 7: Volatility regime (vol spike detector)
    # Why: Ratio of short-term to long-term vol captures vol spikes.
    # Ratio > 1 = short-term vol elevated above baseline (regime shift).
    # ------------------------------------------------------------------
    vol5  = spy_ret.shift(1).rolling(5).std().replace(0, np.nan)
    vol21 = spy_ret.shift(1).rolling(21).std().replace(0, np.nan)
    out["vol_regime"] = vol5 / vol21

    # ------------------------------------------------------------------
    # GROUP 8: Flight-to-safety correlation (risk-off indicator)
    # Why: SPY and TLT (long treasuries) are normally negatively
    # correlated — bonds rise when stocks fall. When this correlation
    # becomes deeply negative, it signals a classic risk-off rotation:
    # institutional money fleeing equities for safe-haven assets.
    # ------------------------------------------------------------------
    tlt_ret = df["ret_TLT"] if "ret_TLT" in df.columns else np.log(df["TLT"] / df["TLT"].shift(1))
    out["flight_safety"] = (
        spy_ret.shift(1)
        .rolling(21)
        .corr(tlt_ret.shift(1))
    )

    # ------------------------------------------------------------------
    # GROUP 9: Peak drawdown from 252-day high (#1 SHAP feature)
    # Why: A market already 15% below its annual high is in a different
    # regime than one at all-time highs. Drawdown captures cumulative
    # stress — negative feedback loops (forced selling, margin calls)
    # make further declines more likely once drawdown is significant.
    # Formula: (current price / rolling 252-day max) - 1  → negative number
    # ------------------------------------------------------------------
    rolling_peak = spy.shift(1).rolling(252).max().replace(0, np.nan)
    out["peak_dd_252"] = (spy.shift(1) / rolling_peak) - 1

    # ------------------------------------------------------------------
    # TARGET VARIABLES (forward-looking — used as labels only)
    # ------------------------------------------------------------------
    # Regression target: realized volatility over next FWD_WINDOW days
    fwd_vol = spy_ret.rolling(FWD_WINDOW).std().shift(-FWD_WINDOW)
    out["target_vol_fwd"] = fwd_vol

    # Classification target: 1 if SPY drops >2% over next 5 days
    fwd_ret = spy_ret.rolling(FWD_WINDOW).sum().shift(-FWD_WINDOW)
    out["target_crash"] = (fwd_ret < CRASH_THRESHOLD).astype(float)

    # ------------------------------------------------------------------
    # Drop rows with NaN in either target (typically first ~252 rows
    # and last FWD_WINDOW rows due to rolling windows)
    # ------------------------------------------------------------------
    out = out.dropna(subset=["target_vol_fwd", "target_crash"])

    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excludes targets and raw prices).
    These are the exact columns passed to model training.
    """
    exclude_prefixes = ("target_", "SPY", "QQQ", "IWM", "TLT",
                        "HYG", "GLD", "EFA", "EEM")
    exclude_exact = set()
    feature_cols = [
        c for c in df.columns
        if not c.startswith(exclude_prefixes) and c not in exclude_exact
        and df[c].dtype in [float, "float64", "float32"]
    ]
    return feature_cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = pd.read_parquet(f"{PROCESSED}/merged.parquet")
    print(f"Raw data: {df_raw.shape}")

    df = add_features(df_raw)
    features = get_feature_cols(df)

    print(f"Features engineered: {len(features)}")
    print(f"Feature names: {features}")
    print(f"Final dataset: {df.shape}")
    print(f"Crash rate: {df['target_crash'].mean():.1%} of days")

    os.makedirs(PROCESSED, exist_ok=True)
    df.to_parquet(f"{PROCESSED}/features.parquet")
    print(f"Saved: {PROCESSED}/features.parquet")
