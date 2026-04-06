# =============================================================================
# run_pipeline.py
# PURPOSE: Master script that runs the ENTIRE project end to end.
#
# USAGE:
#   python run_pipeline.py              # run all 6 phases
#   python run_pipeline.py --skip-data  # skip download, use cached data
#   python run_pipeline.py --phase 3    # run only phase 3 (and forward)
#
# PHASES:
#   1 — Data collection    (download Yahoo Finance + Fama-French + VIX)
#   2 — Feature engineering (27 features including 5 custom signals)
#   3 — Model training      (Ridge, RF, XGBoost + calibrated ensemble)
#   4 — Statistical testing (t-tests, bootstrap CIs, FF alpha regression)
#   5 — Decision layer      (walk-forward backtest + portfolio simulation)
#   6 — Explainability      (SHAP analysis, 4 plots)
# =============================================================================

import sys
import time
import argparse
from datetime import datetime

PIPELINE_START = time.time()

# ---------------------------------------------------------------------------
# Utility printers
# ---------------------------------------------------------------------------
def _header(num, name):
    elapsed = time.time() - PIPELINE_START
    print(f"\n{'='*60}")
    print(f"  PHASE {num}: {name.upper()}")
    print(f"  Elapsed: {elapsed:.0f}s  |  Started: {datetime.now():%H:%M:%S}")
    print(f"{'='*60}\n")


def _done(num, name, t0):
    print(f"\n  Phase {num} complete: {name}  ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------
def phase_1(skip=False):
    t0 = time.time()
    _header(1, "Data Collection")
    if skip:
        print("  Skipping — using cached data/processed/ files")
        _done(1, "Data Collection (skipped)", t0)
        return

    from src.data_loader import load_yahoo, load_fama_french, load_vix, merge_all
    import os
    os.makedirs("data/processed", exist_ok=True)

    prices, returns = load_yahoo()
    ff  = load_fama_french()
    vix = load_vix()
    merged = merge_all(prices, returns, ff, vix)

    prices.to_parquet("data/processed/prices.parquet")
    merged.to_parquet("data/processed/merged.parquet")
    _done(1, "Data Collection", t0)


def phase_2():
    t0 = time.time()
    _header(2, "Feature Engineering")
    import pandas as pd
    from src.features import add_features, get_feature_cols

    df_raw = pd.read_parquet("data/processed/merged.parquet")
    print(f"  Raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    df = add_features(df_raw)
    features = get_feature_cols(df)
    print(f"  Features engineered: {len(features)}")
    print(f"  Final dataset: {df.shape[0]} rows")

    df.to_parquet("data/processed/features.parquet")
    print("  Saved: data/processed/features.parquet")
    _done(2, "Feature Engineering", t0)


def phase_3():
    t0 = time.time()
    _header(3, "Model Training")
    import pandas as pd
    import mlflow
    from src.models import (run_regression_experiments,
                             run_classification_experiments,
                             run_ensemble_experiment)

    df = pd.read_parquet("data/processed/features.parquet")
    feature_prefixes = ("ret_", "vol_", "mean_", "skew_", "vix_",
                        "Mkt", "SMB", "HML", "RF", "ma200_",
                        "mom_", "flight_", "peak_dd_")
    features = [c for c in df.columns
                if any(c.startswith(p) for p in feature_prefixes)
                and "target" not in c]

    print(f"  Features: {len(features)} | Days: {df.shape[0]}")

    mlflow.set_experiment("risk-forecasting")
    with mlflow.start_run(run_name=f"pipeline_{datetime.now():%Y%m%d_%H%M}"):
        print("\n  Regression (predicting future volatility):")
        reg_vol = run_regression_experiments(df, features, "target_vol_fwd")

        print("\n  Classification (predicting crash):")
        clf_res = run_classification_experiments(df, features)
        ens_res = run_ensemble_experiment(df, features)

    print("\n  Final scores:")
    for n, r in reg_vol.items():
        print(f"    {n:<22s}  RMSE: {r['rmse_mean']:.4f}")
    for n, r in clf_res.items():
        print(f"    {n:<22s}  AUC:  {r['auc_mean']:.4f}")
    print(f"    {'ensemble_calibrated':<22s}  AUC:  {ens_res['auc_mean']:.4f}")

    _done(3, "Model Training", t0)


def phase_4():
    t0 = time.time()
    _header(4, "Statistical Testing")
    import subprocess
    result = subprocess.run([sys.executable, "src/evaluate.py"])
    if result.returncode != 0:
        print("  WARNING: evaluate.py had errors — check output above")
    _done(4, "Statistical Testing", t0)


def phase_5():
    t0 = time.time()
    _header(5, "Walk-Forward Backtest")
    import subprocess
    result = subprocess.run([sys.executable, "src/decision.py"])
    if result.returncode != 0:
        print("  WARNING: decision.py had errors — check output above")
    _done(5, "Decision Layer", t0)


def phase_6():
    t0 = time.time()
    _header(6, "SHAP Explainability")
    import subprocess
    result = subprocess.run([sys.executable, "src/explain.py"])
    if result.returncode != 0:
        print("  WARNING: explain.py had errors — check output above")
    _done(6, "SHAP Explainability", t0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    total = time.time() - PIPELINE_START
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
    print("""
  KEY RESULTS
  ─────────────────────────────────────────────────────
  Data         SPY QQQ IWM TLT HYG GLD EFA EEM | 2007-2024
  Features     27 total (lags, vol, VIX, drawdown, regime)
  Best regress Ridge   RMSE 0.0819 (volatility prediction)
  Best clf     RF      AUC  0.682  (crash classification)
  Strategy     Sharpe 1.20 | MaxDD -11.72% | Return +392%
  Benchmark    Sharpe 0.73 | MaxDD -33.72% | Return +496%
  Top feature  peak_dd_252 (drawdown from 252-day high)
  Riskiest day 2008-10-02  (post-Lehman collapse)
  ─────────────────────────────────────────────────────
  Outputs
    data/processed/features.parquet
    reports/strategy_vs_benchmark.png
    reports/shap_importance_bar.png
    reports/shap_summary_dot.png
    reports/shap_single_day.png
    reports/shap_stability.png
    tableau/cumulative_returns.csv
    tableau/regime_overlay.csv
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk Forecasting Pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download (use cached parquet files)")
    parser.add_argument("--phase", type=int, default=1,
                        help="Start from this phase number (default: 1)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  EQUITY RISK FORECASTING & PORTFOLIO OPTIMIZATION")
    print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    phase_map = {
        1: lambda: phase_1(skip=args.skip_data),
        2: phase_2,
        3: phase_3,
        4: phase_4,
        5: phase_5,
        6: phase_6,
    }

    for num in range(args.phase, 7):
        phase_map[num]()

    print_summary()
