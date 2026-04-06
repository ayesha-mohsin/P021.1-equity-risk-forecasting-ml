# =============================================================================
# src/evaluate.py
# PURPOSE: Statistical validation of model performance and strategy returns.
#
# TESTS:
#   1. Paired t-test         — is strategy outperformance statistically significant?
#   2. Bootstrap CIs         — confidence intervals on Sharpe ratio
#   3. Fama-French regression — does strategy have genuine alpha?
#
# WHY THIS MATTERS:
#   A model can produce good backtest returns just from overfitting or luck.
#   Statistical testing asks: "Is this outperformance real, or noise?"
#   The Fama-French regression specifically asks: "Is there genuine alpha,
#   or is this just exposure to known risk factors?"
#
# USAGE:
#   python src/evaluate.py
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

PROCESSED = "data/processed"
TABLEAU   = "tableau"
REPORTS   = "reports"


# ---------------------------------------------------------------------------
# Load strategy returns (produced by decision.py)
# ---------------------------------------------------------------------------
def _load_returns():
    path = f"{TABLEAU}/cumulative_returns.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run decision.py first (Phase 5 must run before Phase 4 re-evaluation)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


# ---------------------------------------------------------------------------
# Test 1: Paired t-test
# ---------------------------------------------------------------------------
def test_outperformance(strategy_ret: pd.Series, benchmark_ret: pd.Series):
    """
    Paired t-test: H0 = mean(strategy - benchmark) = 0

    A statistically significant result (p < 0.05) means the strategy's
    outperformance is unlikely to be due to random chance.

    'Paired' because both series are measured on the same days — the
    pairing controls for shared market conditions.
    """
    diff = strategy_ret - benchmark_ret
    t_stat, p_value = stats.ttest_1samp(diff.dropna(), 0)
    print(f"\nPaired t-test (strategy vs benchmark):")
    print(f"  Mean daily excess return: {diff.mean()*100:.4f}%")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}  {'*** SIGNIFICANT' if p_value < 0.05 else '(not significant at 5%)'}")
    return {"t_stat": t_stat, "p_value": p_value, "mean_excess": diff.mean()}


# ---------------------------------------------------------------------------
# Test 2: Bootstrap confidence intervals on Sharpe ratio
# ---------------------------------------------------------------------------
def bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """
    Bootstrap 95% confidence interval on the annualized Sharpe ratio.

    Why bootstrap instead of parametric CI?
    Sharpe ratio is not normally distributed (especially with fat tails
    common in financial returns). Bootstrap makes no distribution assumption
    — it empirically resamples the actual return distribution.

    Annualized Sharpe = mean(daily returns) / std(daily returns) * sqrt(252)
    """
    ret = returns.dropna().values
    boot_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(ret, size=len(ret), replace=True)
        s = (sample.mean() / sample.std()) * np.sqrt(252) if sample.std() > 0 else 0
        boot_sharpes.append(s)

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_sharpes, alpha * 100)
    hi = np.percentile(boot_sharpes, (1 - alpha) * 100)
    point = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0

    print(f"\nBootstrap Sharpe CI ({int(ci*100)}%, {n_boot} resamples):")
    print(f"  Point estimate: {point:.3f}")
    print(f"  {int(ci*100)}% CI:        [{lo:.3f}, {hi:.3f}]")
    return {"sharpe": point, "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# Test 3: Fama-French alpha regression
# ---------------------------------------------------------------------------
def fama_french_alpha(strategy_ret: pd.Series, df_features: pd.DataFrame) -> dict:
    """
    Regress strategy daily excess returns on Fama-French 3 factors:
        R_strategy - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + e

    If alpha is positive and statistically significant, the strategy has
    genuine predictive power not explained by systematic factor exposures.

    If alpha ≈ 0 but beta_mkt ≈ 0.6, the strategy simply holds ~60%
    equities on average — no genuine alpha, just a factor tilt.

    This is the definitive test used in academic finance to validate
    whether any trading strategy has real edge.
    """
    factors = ["Mkt-RF", "SMB", "HML", "RF"]
    available = [f for f in factors if f in df_features.columns]
    if not available:
        print("\nFama-French regression: factor data not available, skipping.")
        return {}

    ff = df_features[available].copy()
    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0, index=ff.index)
    excess_strategy = strategy_ret - rf

    aligned = pd.concat([excess_strategy.rename("strategy"), ff], axis=1).dropna()
    if len(aligned) < 100:
        print("\nFama-French regression: insufficient data.")
        return {}

    y = aligned["strategy"]
    X_cols = [c for c in ["Mkt-RF", "SMB", "HML"] if c in aligned.columns]
    X = sm.add_constant(aligned[X_cols])

    model = sm.OLS(y, X).fit()

    print("\nFama-French Alpha Regression:")
    print(f"  Alpha (annualized): {model.params['const'] * 252 * 100:.3f}%")
    print(f"  Alpha p-value:      {model.pvalues['const']:.4f}  "
          f"{'*** SIGNIFICANT' if model.pvalues['const'] < 0.05 else '(not significant)'}")
    print(f"  R-squared:          {model.rsquared:.4f}")
    for col in X_cols:
        print(f"  beta_{col:<8s}:     {model.params[col]:.4f}")

    return {
        "alpha_annualized": model.params["const"] * 252,
        "alpha_pvalue": model.pvalues["const"],
        "r_squared": model.rsquared,
        "betas": {col: model.params[col] for col in X_cols}
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)

    try:
        ret_df = _load_returns()
        strategy_ret  = ret_df["strategy_return"]
        benchmark_ret = ret_df["benchmark_return"]
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Running a simplified evaluation on feature data instead.")
        # Fallback: load feature data and simulate dummy returns for testing
        df = pd.read_parquet(f"{PROCESSED}/features.parquet")
        spy_col = "ret_SPY" if "ret_SPY" in df.columns else [c for c in df.columns if "ret_" in c][0]
        strategy_ret  = df[spy_col] * 0.7  # placeholder
        benchmark_ret = df[spy_col]
        ret_df = df

    results = {}
    results["t_test"]   = test_outperformance(strategy_ret, benchmark_ret)
    results["sharpe_ci"] = bootstrap_sharpe(strategy_ret)

    # Load feature data for Fama-French factors
    try:
        df_features = pd.read_parquet(f"{PROCESSED}/features.parquet")
        ff_ret = strategy_ret.reindex(df_features.index).dropna()
        results["ff_alpha"] = fama_french_alpha(ff_ret, df_features)
    except Exception as e:
        print(f"Fama-French regression skipped: {e}")

    # Save results
    with open(f"{REPORTS}/eval_results.json", "w") as f:
        def _convert(obj):
            if isinstance(obj, (np.float64, np.float32, float)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        json.dump(_convert(results), f, indent=2)

    print(f"\nResults saved to {REPORTS}/eval_results.json")
