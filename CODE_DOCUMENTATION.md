# Code Documentation & Interview Prep Notes
## Equity Risk Forecasting ML — Line-by-Line Module Guide

---

> **How to use this document**: Each section covers one source file. For each file you'll find: what it does, how it works, key design decisions, and interview talking points.

---

## `run_pipeline.py` — The Master Orchestrator

### What it does
This is the entry point for the entire project. Running `python run_pipeline.py` executes all 6 phases in order, with timing and status output. It does not contain any ML logic itself — it imports and calls functions from the `src/` modules.

### How it works
- Uses Python's `argparse` module to accept a `--skip-data` flag, which skips the data download if you already have the parquet files cached
- Uses `subprocess.run()` to call `evaluate.py`, `decision.py`, and `explain.py` as separate processes (these scripts are self-contained and easier to run independently)
- Times each phase using `time.time()` and prints a summary with elapsed time
- Ends with a hardcoded results summary showing the final model metrics

### Key design decisions
- **Why subprocess for some phases?** The evaluate, decision, and explain scripts were originally designed as standalone files. Using `subprocess.run()` lets them stay independent while still being orchestrated by the master script. A cleaner alternative would be to import their main functions directly — this is worth mentioning as a known trade-off.
- **Why hardcoded results in the final summary?** The summary at the bottom shows results from a completed run. In production you'd write these to a JSON file and read them back dynamically.

### Interview talking points
- "I built a pipeline orchestrator that runs 6 sequential phases with error handling and timing. Each phase produces an artifact (parquet file, model, chart) that the next phase consumes."
- "I used argparse to make the pipeline configurable — `--skip-data` lets you iterate on the ML logic without re-downloading 17 years of data every run."

---

## `src/data_loader.py` — Data Collection

### What it does
Downloads and merges three external data sources into a single clean DataFrame that serves as the input to all downstream processing.

### Data sources
**Yahoo Finance (via `yfinance`):**
- 8 ETFs: SPY (large-cap US), QQQ (tech), IWM (small-cap), TLT (long treasuries), HYG (high-yield bonds), GLD (gold), EFA (international developed), EEM (emerging markets)
- Downloads adjusted closing prices 2007-01-01 to 2024-12-31
- Computes daily log returns: `log(price_t / price_t-1)`

**Fama-French 3-Factor Model (from Ken French's data library):**
- Market risk premium (Mkt-RF): the excess return of the market over the risk-free rate
- SMB (Small Minus Big): return differential between small-cap and large-cap stocks
- HML (High Minus Low): return differential between value and growth stocks
- RF: daily risk-free rate (typically ~T-bill rate / 252)
- These are loaded as daily percentages and merged on date

**VIX (CBOE Volatility Index):**
- Downloaded via Yahoo Finance ticker `^VIX`
- Represents the implied volatility of S&P 500 options over the next 30 days
- Acts as a real-time "fear gauge" — spikes sharply during market stress

### `merge_all()` function
Joins all three data sources on their date index using an inner join (only keeps trading days present in all sources). Saves to `data/processed/merged.parquet` using Apache Parquet format for fast columnar I/O.

### Interview talking points
- "I used yfinance for market data because it's free and covers the full period including the 2008 crisis. The data goes back to 2007 to capture the GFC as a training signal."
- "Fama-French factors are important because they let you decompose returns — if your model's alpha disappears after controlling for market/size/value factors, your edge might just be a known factor tilt, not genuine predictive power."
- "I used Parquet instead of CSV because it's ~10× faster to load for large DataFrames and preserves dtypes (datetime index, float64 columns) without manual conversion."
- **On VIX**: "VIX is forward-looking. It reflects option market pricing of future risk. A high VIX today doesn't just mean markets fell yesterday — it means options traders are pricing in continued uncertainty."

---

## `src/features.py` — Feature Engineering

### What it does
Takes the merged raw dataset and engineers 27 predictive features. Also defines the target variables that the models will predict.

### Feature groups and the reasoning behind each

**Lag returns** (`ret_1d`, `ret_2d`, `ret_5d`, `ret_10d`, `ret_21d`):
Returns from 1, 2, 5, 10, and 21 trading days ago. These capture momentum (recent winners tend to continue short-term) and mean-reversion (extreme moves sometimes snap back). Using multiple lags lets the model learn which horizon is most predictive for crash risk.

**Rolling statistics** (`vol_5d`, `vol_21d`, `mean_5d`, `mean_21d`, `skew_21d`):
Rolling standard deviation (volatility), mean return, and skewness of returns over 5 and 21 day windows. Volatility clustering is well-established in finance — high vol today predicts high vol tomorrow (GARCH-like behavior). Negative skewness means more left-tail risk. These are computed using pandas `.rolling().std()` etc.

**VIX features** (`vix_level`, `vix_change_5d`, `vix_change_21d`, `vix_vol_ratio`):
Raw VIX level plus rate of change over 5 and 21 days. The ratio of VIX to realized vol (`vix_vol_ratio`) captures when implied vol is high relative to realized — a signal of fear premium / uncertainty above and beyond recent actual volatility.

**Fama-French factors** (`Mkt-RF`, `SMB`, `HML`, `RF`):
Used directly as features. For example, a negative Mkt-RF on a given day is a direct signal of broad market weakness.

**200-day MA signal** (`ma200_signal`):
Binary feature: 1 if SPY's current price is above its 200-day moving average, 0 if below. This is a classic trend-following signal used by institutional traders. Being below the 200MA is associated with bear market regimes.

**Momentum divergence** (`mom_div`):
Ratio of short-term momentum (5-day return) to long-term momentum (63-day return). When short-term momentum sharply diverges from long-term trend, it signals instability — either a breakout or a reversal.

**Volatility regime** (`vol_regime`):
Ratio of 5-day vol to 21-day vol. A ratio > 1 means short-term volatility is elevated relative to the recent baseline — a vol spike. This is one of the fastest signals for detecting regime changes.

**Flight-to-safety** (`flight_safety`):
Rolling 21-day correlation between SPY and TLT (long treasury bonds). Normally this correlation is negative (bonds rise when stocks fall). When this correlation becomes deeply negative (e.g., -0.8), it signals a classic risk-off rotation — investors fleeing equities for safe-haven assets.

**Peak drawdown** (`peak_dd_252`) — THE MOST IMPORTANT FEATURE:
How far the current SPY price is below its 252-day (roughly 1-year) rolling high. Formula: `(current_price / rolling_max_price) - 1`. This captures whether we are in a drawdown regime. SHAP analysis confirmed this as the top predictor — a market that is already 15% below its annual peak is significantly more likely to crash further than one near all-time highs.

### Target variables
Two prediction targets are created:
- `target_vol_fwd`: Realized volatility of SPY returns over the next 5 trading days (regression target)
- `target_crash` (implied via labels): Binary — 1 if SPY return over next 5 days is worse than -2%, else 0 (classification target)

### Interview talking points
- "I engineered 27 features across 8 conceptual groups. Feature engineering is where most of the predictive signal comes from in tabular financial ML."
- "The peak drawdown feature turned out to be the most important by a wide margin in SHAP analysis. Intuitively, a market that's already fallen 20% from its high is in a different regime than a market at all-time highs."
- "I was careful about look-ahead bias. All features use only past data — rolling windows shift on historical data only. The target is future data, correctly aligned forward."
- **On flight-to-safety**: "The SPY-TLT correlation flipping negative is one of the most reliable institutional signals of risk-off mode. It shows up in the data during 2008, 2011, 2020, and every other major stress period."

---

## `src/models.py` — Model Training & MLflow Tracking

### What it does
Trains three model families for two different prediction tasks (regression and classification), evaluates them using proper time-series cross-validation, and logs everything to MLflow.

### Why time-series cross-validation (not regular k-fold)?
Standard k-fold cross-validation shuffles data randomly. In financial time series this creates **data leakage** — the model might train on data from 2015 and test on data from 2014, which is impossible in real deployment. Time-series CV uses `TimeSeriesSplit` from scikit-learn, which always trains on earlier data and tests on later data.

### Regression models — predicting future volatility
**Ridge Regression**: Linear model with L2 regularization. Ridge adds a penalty term (`alpha * sum(weights²)`) to the loss function, which shrinks coefficients toward zero and prevents overfitting on correlated financial features. Best RMSE: 0.0819.

**Random Forest**: Ensemble of decision trees, each trained on a bootstrap sample with a random feature subset. For time series, `bootstrap=True` and feature randomness provide regularization. Captures non-linear interactions between features (e.g., "high VIX AND below 200MA" being worse than either alone). RMSE: 0.0837.

**XGBoost**: Gradient-boosted trees. Sequentially builds trees where each tree corrects the errors of the previous ones. Generally the strongest tabular ML algorithm. RMSE: 0.0884 — slightly worse than Ridge here, which is common when the true relationship is relatively linear (Ridge dominates on linear problems, XGBoost dominates on complex non-linear ones).

### Classification models — predicting crash probability
Same three model families plus a calibrated ensemble:

**Logistic Regression**: Linear classifier. Predicts `P(crash) = sigmoid(w·x)`. AUC: 0.675.

**Random Forest**: AUC: 0.682 — best individual model. The ensemble nature helps reduce variance on the noisy crash prediction task.

**XGBoost**: AUC: 0.632. Underperformed here — possibly due to overfitting on the limited number of actual crash events in the training data (class imbalance).

**Calibrated Ensemble**: Averages the predicted probabilities from all three models, then applies isotonic regression calibration (`CalibratedClassifierCV`). Calibration ensures that when the model says 70% crash probability, crashes actually happen 70% of the time historically. AUC: 0.652 — lower than RF alone, but the probabilities are more reliable for position sizing.

### MLflow tracking
Every experiment run logs:
- **Parameters**: model hyperparameters, feature list, target variable
- **Metrics**: RMSE or AUC (mean and std across CV folds)
- **Artifacts**: serialized model (pkl), feature importance plot
- **Tags**: run name, timestamp, phase

Models are registered in the MLflow model registry at `mlruns/1/models/`.

### Interview talking points
- "I used TimeSeriesSplit instead of KFold to respect the temporal structure of financial data and avoid look-ahead bias."
- "Ridge won the regression task because the volatility prediction relationship is approximately linear — future vol is strongly correlated with recent vol. XGBoost's extra complexity didn't help here."
- "I built a calibrated ensemble for the classification task. Raw model probabilities are often poorly calibrated (e.g., a model saying 0.9 doesn't mean 90% crash chance). Isotonic calibration fixes this, which is critical when using the probability for position sizing."
- "MLflow lets me reproduce any experiment run. I can see exactly which hyperparameters produced which AUC, which is important for auditability in a financial context."

---

## `src/evaluate.py` — Statistical Validation

### What it does
Tests whether the model's predictions and the resulting strategy's performance are **statistically significant**, not just lucky.

### Tests performed

**Paired t-test on strategy vs benchmark returns**:
Tests H0: mean(strategy_returns - benchmark_returns) = 0. A low p-value (< 0.05) means the outperformance is unlikely to be due to chance. This is the most common statistical test in quantitative finance.

**Bootstrap confidence intervals**:
Resamples the return series 1,000 times to compute confidence intervals on the Sharpe ratio. If the lower bound of the 95% CI is still > 0.73 (benchmark Sharpe), the outperformance is robust.

**Fama-French alpha regression**:
Regresses the strategy's daily excess returns on the three Fama-French factors:
`R_strategy - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + error`
If alpha is positive and significant, the strategy has genuine predictive alpha — not just a factor tilt. If alpha is zero but beta_mkt is 0.6, it just means the strategy holds ~60% equities on average.

### Interview talking points
- "A model can have good backtest returns just from luck or data snooping. I validated statistical significance with t-tests and bootstrap CIs before claiming any performance edge."
- "The Fama-French regression is the finance-specific robustness check. If my 'alpha' disappears after controlling for market/size/value exposures, then I haven't found anything novel — I've just reinvented a factor tilt."
- "P-values alone are insufficient. I computed confidence intervals to show the range of plausible Sharpe ratios, not just a point estimate."

---

## `src/decision.py` — Walk-Forward Backtest

### What it does
Simulates how the ML strategy would have performed if deployed in real-time, using a walk-forward methodology that prevents any look-ahead bias.

### Walk-forward methodology
```
Train window: 3 years of daily data
Test window: 1 year
Step: 1 year
```
So the model trains on 2007–2009, tests on 2010. Then retrains on 2008–2010, tests on 2011. And so on through 2024. This produces a realistic out-of-sample performance estimate for the full period.

### Position sizing logic
```python
# Pseudocode
crash_prob = model.predict_proba(today_features)
equity_weight = max(0, 1 - crash_prob)  # Scale down equity when crash risk is high
cash_weight = 1 - equity_weight
portfolio_return = equity_weight * SPY_return + cash_weight * RF_rate
```
When crash probability is 0% → 100% SPY. When 100% → 100% cash. Probabilities in between linearly scale exposure.

### Transaction costs
A 10 basis point (0.10%) cost is deducted on any day where the equity weight changes by more than a threshold. This prevents the backtest from appearing unrealistically profitable by ignoring trading friction.

### Benchmark
SPY buy-and-hold with the same starting capital. The benchmark is fully invested every day.

### Key outputs
- `reports/strategy_vs_benchmark.png`: Cumulative wealth comparison chart
- `tableau/cumulative_returns.csv`: Daily returns for both strategy and benchmark
- `tableau/regime_overlay.csv`: Crash probability over time with market regimes labeled

### Interview talking points
- "Walk-forward backtesting is the gold standard for validating financial ML models. It mimics the real deployment scenario — you only ever use past data to make decisions."
- "I included transaction costs because without them, any strategy that trades frequently will look unrealistically good. 10bps per trade is a conservative institutional estimate."
- "The strategy returned 392% vs 496% for buy-and-hold, but with a Sharpe of 1.20 vs 0.73 and a max drawdown of -11.72% vs -33.72%. For an institutional investor managing other people's money, the drawdown constraint is paramount — losing 33% requires a 50% recovery just to break even."

---

## `src/explain.py` — SHAP Explainability

### What it does
Uses SHAP (SHapley Additive exPlanations) to explain why the Random Forest model makes each prediction. Produces 4 charts saved to `reports/`.

### What is SHAP?
SHAP is a game theory-based method for explaining ML model predictions. For each prediction, SHAP assigns each feature a contribution value (in units of prediction impact). The contributions sum to the final prediction. This makes the model interpretable even though Random Forest is typically a "black box."

For example, if the model predicts a 70% crash probability today, SHAP might say:
- `peak_dd_252`: +0.25 (contributes +25% to crash probability)
- `vix_level`: +0.15
- `ma200_signal`: +0.10
- `ret_5d`: -0.08 (recent positive return reduces crash probability)

### The 4 SHAP outputs

**`shap_importance_bar.png`** — Global importance ranking:
Mean absolute SHAP value for each feature across all test predictions. Shows which features matter most overall. Top result: `peak_dd_252` is far and away the most important feature.

**`shap_summary_dot.png`** — Beeswarm / dot plot:
Each dot is one day. X-axis is SHAP value (positive = pushes toward crash), Y-axis groups dots by feature. Color indicates feature value (red = high, blue = low). This shows both importance AND direction. For example: high VIX (red dots) consistently push right → higher crash probability.

**`shap_single_day.png`** — Waterfall for 2008-10-02:
Shows exactly what drove the prediction on the single riskiest day in the dataset (October 2nd, 2008 — two weeks after Lehman Brothers filed for bankruptcy on September 15, 2008). Each feature's contribution is shown as a bar extending right (increases crash prob) or left (decreases it).

**`shap_stability.png`** — Feature importance over time:
Shows whether feature importance is consistent or changes across market regimes. Stable importance = robust signal. Unstable = feature may be regime-dependent.

### Key SHAP findings
- `peak_dd_252` (drawdown from 252-day high): Top feature by a large margin. When the market is significantly below its annual high, the model assigns high crash probability.
- VIX features: Second most important group. High VIX consistently signals elevated risk.
- `flight_safety` (SPY-TLT correlation): Important during stress periods specifically.
- Most important day in the dataset: **2008-10-02** — which makes intuitive sense as one of the most volatile periods in market history.

### Interview talking points
- "SHAP gives me per-prediction explanations, not just global feature importance. This is crucial for a financial model — if I'm recommending reducing equity exposure by 40%, I need to be able to explain exactly why to a portfolio manager."
- "The most important feature turned out to be `peak_dd_252` — drawdown from the annual high. This makes economic sense: a market already in a significant drawdown is more fragile, liquidity is tighter, and momentum is negative."
- "I used SHAP specifically to check that the model isn't relying on spurious correlations. If VIX features dominate but VIX was only available from 1990, for instance, that would be a concern for earlier data. Understanding the model's reasoning also helps validate it against financial intuition."
- "The single-day waterfall for 2008-10-02 is a great demonstration — I can show exactly how each feature contributed to the model correctly flagging that day as extremely high risk."

---

## Things to Note / Potential Improvements (for interview discussion)

**1. Requirements.txt is too large**
The current `requirements.txt` contains 177 packages including the full Jupyter stack. For reproducibility in an interview context it's fine, but in production you'd use `pip-tools` or `poetry` to manage a leaner dependency set.

**2. `mlflow.db` committed to git**
The MLflow database file is committed to the repo to preserve experiment history. In a team environment you'd run an MLflow tracking server instead.

**3. `subprocess.run()` vs direct imports in `run_pipeline.py`**
Phases 4–6 are called via subprocess. A cleaner architecture would define `main()` functions in each script and import them directly, giving better error handling and shared memory.

**4. Class imbalance in classification**
Crashes (>-2% in 5 days) are relatively rare. The classification models could benefit from SMOTE oversampling or `class_weight='balanced'` in scikit-learn, which may improve recall on actual crash events.

**5. No hyperparameter tuning**
Models use default or manually-set hyperparameters. Adding `GridSearchCV` or `Optuna` with time-series CV would likely improve AUC further.

**6. No `__init__.py` visible in `src/`**
The `src/` directory should have an `__init__.py` to be a proper Python package. Without it, imports work via direct path but break if you move the working directory.
