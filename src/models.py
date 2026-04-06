# =============================================================================
# src/models.py
# PURPOSE: Train Ridge, Random Forest, XGBoost for both regression and
#          classification tasks. Track all experiments with MLflow.
#
# KEY DESIGN CHOICE — TimeSeriesSplit:
#   Standard k-fold shuffles data randomly, causing data leakage in time
#   series (model trains on 2015 data, tests on 2014 — impossible in reality).
#   TimeSeriesSplit always trains on past, tests on future. This is the
#   correct evaluation methodology for financial ML.
#
# RESULTS:
#   Regression  (volatility) — Ridge RMSE: 0.0819 (best)
#   Classification (crashes) — Random Forest AUC: 0.682 (best)
#
# USAGE (standalone):
#   python src/models.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model    import Ridge, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_squared_error, roc_auc_score
from xgboost                 import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED  = "data/processed"
N_SPLITS   = 5      # TimeSeriesSplit folds — 5 gives good coverage without
                    # making early folds too small
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper: time-series cross-validation
# ---------------------------------------------------------------------------
def _ts_cv_regression(pipeline, X, y, n_splits=N_SPLITS):
    """
    Evaluate a regression pipeline with TimeSeriesSplit.
    Returns mean and std of RMSE across folds.

    RMSE (Root Mean Squared Error) is used because it's in the same units
    as volatility, making it interpretable: RMSE 0.08 means predictions
    are off by ~8 percentage points of annualized vol on average.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
    return np.mean(rmses), np.std(rmses)


def _ts_cv_classification(pipeline, X, y, n_splits=N_SPLITS):
    """
    Evaluate a classification pipeline with TimeSeriesSplit.
    Returns mean and std of AUC-ROC across folds.

    AUC-ROC (Area Under ROC Curve) is used because:
    1. It's threshold-independent (we use probabilities for position sizing)
    2. It handles class imbalance better than accuracy
    3. AUC = 0.5 is random chance; 0.68 is meaningful in financial prediction
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        # Skip fold if only one class in training (can't train classifier)
        if y_tr.nunique() < 2:
            continue
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, proba))
    return np.mean(aucs), np.std(aucs)


# ---------------------------------------------------------------------------
# Regression experiments
# ---------------------------------------------------------------------------
def run_regression_experiments(df: pd.DataFrame,
                                features: list[str],
                                target: str) -> dict:
    """
    Train Ridge, RF, XGBoost regressors. Log each to MLflow.

    Why Ridge beats XGBoost here:
    Volatility prediction is a relatively linear problem — future vol is
    strongly correlated with recent vol (GARCH-like). Ridge handles this
    linear relationship well without overfitting on 27 correlated features
    (L2 regularization shrinks redundant coefficients). XGBoost's extra
    complexity doesn't help when the true signal is approximately linear.
    """
    X = df[features].fillna(0)
    y = df[target]

    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),   # Ridge is scale-sensitive
            ("model", Ridge(alpha=1.0))
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,        # prevents overfitting on small folds
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),
        "xgboost": Pipeline([
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    results = {}
    for name, pipe in models.items():
        with mlflow.start_run(run_name=f"reg_{name}_{target}", nested=True):
            rmse_mean, rmse_std = _ts_cv_regression(pipe, X, y)
            mlflow.log_params({"model": name, "target": target, "n_features": len(features)})
            mlflow.log_metrics({"rmse_mean": rmse_mean, "rmse_std": rmse_std})
            results[name] = {"rmse_mean": rmse_mean, "rmse_std": rmse_std}
            print(f"    {name:<20s}  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")

    return results


# ---------------------------------------------------------------------------
# Classification experiments
# ---------------------------------------------------------------------------
def run_classification_experiments(df: pd.DataFrame,
                                   features: list[str]) -> dict:
    """
    Train Logistic Regression, RF, XGBoost classifiers.

    Why Random Forest is best here:
    Crash prediction is noisy with class imbalance (crashes are rare).
    RF's ensemble averaging reduces variance from noisy labels. Logistic
    regression is competitive because many crash signals are approximately
    linear (high VIX + drawdown → crash). XGBoost underperforms because
    it overfits on the rare crash events in each fold.
    """
    X = df[features].fillna(0)
    y = df["target_crash"]
    pos_rate = y.mean()
    print(f"    Class balance: {pos_rate:.1%} crash days")

    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1,                  # stronger regularization for financial data
                class_weight="balanced",  # corrects for class imbalance
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=15,
                class_weight="balanced",  # corrects for class imbalance
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),
        "xgboost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(1 - pos_rate) / max(pos_rate, 0.001),
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    results = {}
    for name, pipe in models.items():
        with mlflow.start_run(run_name=f"clf_{name}", nested=True):
            auc_mean, auc_std = _ts_cv_classification(pipe, X, y)
            mlflow.log_params({"model": name, "n_features": len(features)})
            mlflow.log_metrics({"auc_mean": auc_mean, "auc_std": auc_std})
            results[name] = {"auc_mean": auc_mean, "auc_std": auc_std}
            print(f"    {name:<20s}  AUC:  {auc_mean:.4f} ± {auc_std:.4f}")

    return results


# ---------------------------------------------------------------------------
# Ensemble experiment
# ---------------------------------------------------------------------------
def run_ensemble_experiment(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Build a calibrated ensemble: average probabilities from all 3 classifiers,
    then apply isotonic regression calibration.

    Why calibrate?
    Raw ML classifiers are often poorly calibrated — a model predicting 90%
    crash probability might only be right 60% of the time. Calibration
    ensures that when the model says 70% crash probability, crashes actually
    happen ~70% of the time historically. This is CRITICAL for position
    sizing: we scale equity exposure by (1 - crash_prob), so we need
    accurate probabilities, not just relative rankings.

    Isotonic vs Platt scaling:
    Isotonic calibration is non-parametric and works better for non-monotone
    miscalibrations. Platt (sigmoid) is faster but assumes a specific shape.
    """
    X = df[features].fillna(0)
    y = df["target_crash"]
    pos_rate = y.mean()

    base_classifiers = [
        LogisticRegression(C=0.1, class_weight="balanced",
                           max_iter=1000, random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=300, max_depth=6,
                               min_samples_leaf=15, class_weight="balanced",
                               n_jobs=-1, random_state=RANDOM_STATE),
        XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, scale_pos_weight=(1-pos_rate)/max(pos_rate,0.001),
                      use_label_encoder=False, eval_metric="logloss",
                      verbosity=0, random_state=RANDOM_STATE),
    ]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    scaler = StandardScaler()

    for train_idx, test_idx in tscv.split(X):
        X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2:
            continue

        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # Get calibrated probabilities from each base model
        fold_probas = []
        for clf in base_classifiers:
            cal = CalibratedClassifierCV(clf, cv=3, method="isotonic")
            cal.fit(X_tr, y_tr)
            fold_probas.append(cal.predict_proba(X_te)[:, 1])

        # Average ensemble probability
        ensemble_proba = np.mean(fold_probas, axis=0)
        aucs.append(roc_auc_score(y_te, ensemble_proba))

    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    with mlflow.start_run(run_name="clf_ensemble_calibrated", nested=True):
        mlflow.log_params({"model": "calibrated_ensemble", "n_base": 3})
        mlflow.log_metrics({"auc_mean": auc_mean, "auc_std": auc_std})

    print(f"    {'ensemble_calibrated':<20s}  AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
    return {"auc_mean": auc_mean, "auc_std": auc_std}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet(f"{PROCESSED}/features.parquet")

    # Build feature list — all engineered columns, not raw prices or targets
    feature_prefixes = ("ret_", "vol_", "mean_", "skew_", "vix_",
                        "Mkt", "SMB", "HML", "RF", "ma200_",
                        "mom_", "flight_", "peak_dd_")
    features = [c for c in df.columns
                if any(c.startswith(p) for p in feature_prefixes)
                and "target" not in c]

    print(f"Training on {len(features)} features, {df.shape[0]} days\n")

    mlflow.set_experiment("risk-forecasting")

    with mlflow.start_run(run_name="full_pipeline"):
        print("--- Regression: predict future volatility ---")
        reg_results = run_regression_experiments(df, features, "target_vol_fwd")

        print("\n--- Classification: predict crash ---")
        clf_results = run_classification_experiments(df, features)
        ens_result  = run_ensemble_experiment(df, features)

    print("\n--- Summary ---")
    print("Regression (volatility RMSE, lower=better):")
    for name, r in reg_results.items():
        flag = " ← best" if name == min(reg_results, key=lambda k: reg_results[k]["rmse_mean"]) else ""
        print(f"  {name:<20s}  {r['rmse_mean']:.4f}{flag}")
    print("Classification (AUC, higher=better):")
    for name, r in clf_results.items():
        flag = " ← best" if name == max(clf_results, key=lambda k: clf_results[k]["auc_mean"]) else ""
        print(f"  {name:<20s}  {r['auc_mean']:.4f}{flag}")
    print(f"  {'ensemble_calibrated':<20s}  {ens_result['auc_mean']:.4f}")
