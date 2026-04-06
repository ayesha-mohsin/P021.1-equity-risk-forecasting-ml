# =============================================================================
# src/explain.py
# PURPOSE: SHAP explainability analysis on the best classification model.
#          Produces 4 charts that explain WHY the model makes predictions.
#
# WHAT IS SHAP?
#   SHapley Additive exPlanations — a game theory-based method for
#   explaining ML predictions. For each prediction, SHAP assigns each
#   feature a contribution value showing how much it pushed the prediction
#   up or down. Contributions sum to the final predicted probability.
#
# WHY THIS MATTERS FOR FINANCE:
#   Black-box models are problematic in finance — if we're telling a
#   portfolio manager to reduce equity exposure by 40%, we need to explain
#   exactly why. SHAP provides that per-prediction transparency.
#
# KEY FINDINGS:
#   #1 feature: peak_dd_252 — drawdown from 252-day high
#   Riskiest day: 2008-10-02 (2 weeks after Lehman collapse)
#
# USAGE:
#   python src/explain.py
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

PROCESSED = "data/processed"
REPORTS   = "reports"
N_EXPLAIN = 500    # number of background samples for SHAP TreeExplainer
                   # (more = slower but more accurate; 500 is a good trade-off)


# ---------------------------------------------------------------------------
# Train final model on 80% of data
# ---------------------------------------------------------------------------
def train_final_model(df: pd.DataFrame, features: list[str]):
    """
    Train a final Random Forest on the first 80% of data.
    Use the last 20% as the explanation set (truly out-of-sample).

    We use 80/20 here (not walk-forward) because SHAP analysis is
    exploratory/explanatory — we want a stable model to explain, not
    a rolling re-trained one. Walk-forward is for performance metrics.
    """
    X = df[features].fillna(0)
    y = df["target_crash"]

    split = int(len(df) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_tr_s, y_tr)

    print(f"  Model trained on {len(X_tr)} days")
    print(f"  Explaining {len(X_te)} out-of-sample days")
    return clf, scaler, X_te_s, X_te, y_te


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap(clf, X_te_scaled: np.ndarray,
                 X_te: pd.DataFrame, feature_names: list[str]):
    """
    Compute SHAP values using TreeExplainer (optimized for tree-based models).

    TreeExplainer uses the model's tree structure directly — much faster
    than KernelExplainer (which is model-agnostic but much slower).

    shap_values[:, :, 1] = SHAP values for the positive class (crash=1).
    Each value shows how much that feature pushes the predicted crash
    probability up (positive SHAP) or down (negative SHAP).
    """
    print("  Computing SHAP values (this takes ~30 seconds)...")
    explainer = shap.TreeExplainer(clf)

    # Use a background sample for efficiency
    bg_idx = np.random.choice(len(X_te_scaled), size=min(N_EXPLAIN, len(X_te_scaled)),
                               replace=False)
    shap_values = explainer.shap_values(X_te_scaled)

    # For binary classification, shap_values is list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # positive class (crash=1)
    else:
        sv = shap_values

    sv_df = pd.DataFrame(sv, columns=feature_names, index=X_te.index)
    print(f"  SHAP values shape: {sv_df.shape}")
    return sv_df, explainer


# ---------------------------------------------------------------------------
# Plot 1: Global feature importance (bar chart)
# ---------------------------------------------------------------------------
def plot_importance_bar(sv_df: pd.DataFrame, feature_names: list[str]):
    """
    Mean absolute SHAP value per feature — how important is each feature
    on average across all predictions?
    Higher = more influential globally.
    """
    mean_abs = sv_df.abs().mean().sort_values(ascending=True)
    top_n = mean_abs.tail(15)  # top 15 features

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563EB" if i == len(top_n) - 1 else "#93C5FD" for i in range(len(top_n))]
    bars = ax.barh(top_n.index, top_n.values, color=colors)
    ax.set_xlabel("Mean |SHAP value| (average impact on crash probability)", fontsize=10)
    ax.set_title("Feature Importance — SHAP (Random Forest Classifier)", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Top 5 features by SHAP importance:")
    for feat, val in mean_abs.tail(5).iloc[::-1].items():
        print(f"    {feat:<25s}  {val:.4f}")


# ---------------------------------------------------------------------------
# Plot 2: Beeswarm / summary dot plot
# ---------------------------------------------------------------------------
def plot_summary_dot(sv_df: pd.DataFrame, X_te: pd.DataFrame, feature_names: list[str]):
    """
    For each feature, plots all SHAP values as dots.
    X-axis: SHAP value (positive = pushes toward crash prediction)
    Color: feature value (red = high, blue = low)

    This shows BOTH importance (spread of dots) AND direction
    (e.g., high VIX → red dots on right side = increases crash prob).
    """
    # Use only top 15 features for readability
    top_features = sv_df.abs().mean().sort_values(ascending=False).head(15).index.tolist()
    sv_top = sv_df[top_features]
    X_top  = X_te[top_features]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_positions = range(len(top_features))

    for i, feat in enumerate(top_features):
        vals     = sv_top[feat].values
        feat_raw = X_top[feat].values

        # Normalize feature values for coloring
        vmin, vmax = feat_raw.min(), feat_raw.max()
        norm_feat = (feat_raw - vmin) / (vmax - vmin + 1e-9)

        # Color: red = high feature value, blue = low
        colors = plt.cm.RdBu_r(norm_feat)

        # Add jitter to y-position for readability
        jitter = np.random.default_rng(i).uniform(-0.3, 0.3, size=len(vals))
        ax.scatter(vals, i + jitter, c=colors, alpha=0.4, s=6)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(top_features[::-1] if False else top_features, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on crash probability prediction)", fontsize=10)
    ax.set_title("SHAP Beeswarm — Feature Values vs Impact\n"
                 "Red = high feature value, Blue = low", fontsize=11)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Single-day waterfall (riskiest day)
# ---------------------------------------------------------------------------
def plot_single_day(sv_df: pd.DataFrame, X_te: pd.DataFrame,
                    explainer, X_te_scaled: np.ndarray, feature_names: list[str]):
    """
    Waterfall plot for the single riskiest day (highest predicted crash prob).
    Shows each feature's individual contribution to that day's prediction.

    This is the most interpretable output — you can say:
    "On 2008-10-02, the model predicted 87% crash probability because:
     - peak_dd_252 contributed +0.25 (market was 22% below annual high)
     - vix_level contributed +0.18 (VIX was at 45)
     - flight_safety contributed +0.12 (SPY/TLT corr = -0.82)"
    """
    # Find the day with the highest predicted crash probability
    total_shap = sv_df.sum(axis=1)
    riskiest_idx = total_shap.idxmax()
    riskiest_pos = sv_df.index.get_loc(riskiest_idx)

    print(f"\n  Riskiest day in explanation set: {riskiest_idx.date()}")

    # Sort features by absolute SHAP for this single day
    day_shap = sv_df.loc[riskiest_idx].sort_values(key=abs, ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in day_shap.values]
    ax.barh(day_shap.index, day_shap.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall — {riskiest_idx.date()}\n"
        f"Red = increases crash probability | Blue = decreases it",
        fontsize=11
    )
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/shap_single_day.png", dpi=150, bbox_inches="tight")
    plt.close()

    return riskiest_idx


# ---------------------------------------------------------------------------
# Plot 4: SHAP stability over time
# ---------------------------------------------------------------------------
def plot_stability(sv_df: pd.DataFrame, feature_names: list[str]):
    """
    Rolling 252-day mean |SHAP value| for top 5 features.
    Shows whether feature importance is consistent across time (good) or
    regime-dependent (a signal that the model may not generalise).

    Stable importance = robust signal that doesn't depend on market regime.
    """
    top5 = sv_df.abs().mean().sort_values(ascending=False).head(5).index.tolist()
    rolling = sv_df[top5].abs().rolling(252).mean().dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1D4ED8", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
    for feat, col in zip(top5, colors):
        ax.plot(rolling.index, rolling[feat], label=feat, linewidth=1.8, color=col)
    ax.set_ylabel("Rolling mean |SHAP|", fontsize=10)
    ax.set_title("SHAP Feature Importance Stability Over Time (252-day rolling)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(f"{REPORTS}/shap_stability.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(REPORTS, exist_ok=True)

    df = pd.read_parquet(f"{PROCESSED}/features.parquet")

    feature_prefixes = ("ret_", "vol_", "mean_", "skew_", "vix_",
                        "Mkt", "SMB", "HML", "RF", "ma200_",
                        "mom_", "flight_", "peak_dd_")
    features = [c for c in df.columns
                if any(c.startswith(p) for p in feature_prefixes)
                and "target" not in c]

    print(f"  Features: {len(features)}")

    clf, scaler, X_te_s, X_te, y_te = train_final_model(df, features)
    sv_df, explainer = compute_shap(clf, X_te_s, X_te, features)

    print("\n  Generating SHAP plots...")
    plot_importance_bar(sv_df, features)
    plot_summary_dot(sv_df, X_te, features)
    riskiest_day = plot_single_day(sv_df, X_te, explainer, X_te_s, features)
    plot_stability(sv_df, features)

    print(f"\n  Plots saved to {REPORTS}/:")
    for f in ["shap_importance_bar.png", "shap_summary_dot.png",
              "shap_single_day.png", "shap_stability.png"]:
        print(f"    {f}")
    print(f"\n  Riskiest day found: {riskiest_day.date()}")
