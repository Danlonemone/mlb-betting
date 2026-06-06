"""
Phase 1: Train and evaluate the baseline moneyline model.

Walk-forward validation: train on seasons up to year Y, predict year Y+1.
This mirrors real deployment — you never see future data.

Models tried:
  1. Logistic regression (interpretable baseline)
  2. Gradient boosting (XGBoost)

Evaluation:
  - Accuracy
  - Log loss (lower = better calibrated)
  - Brier score (lower = better)
  - Calibration plot (do 60% picks win ~60% of the time?)
  - Predicted probability distribution
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.engineering import load_feature_matrix, FEATURE_COLS, load_f5_feature_matrix, F5_FEATURE_COLS

MODEL_DIR = Path(__file__).parent
DATA_DIR  = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------

SEASONS_IN_ORDER = [2019, 2021, 2022, 2023, 2024, 2025]

def walk_forward_splits():
    """
    Yield (train_seasons, test_season) pairs.
    Minimum 2 training seasons before first test.
    """
    for i in range(2, len(SEASONS_IN_ORDER)):
        yield SEASONS_IN_ORDER[:i], SEASONS_IN_ORDER[i]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def make_logistic() -> CalibratedClassifierCV:
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")),
    ])
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


def make_xgb() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, label: str) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model":      label,
        "n":          len(y_true),
        "accuracy":   accuracy_score(y_true, y_pred),
        "log_loss":   log_loss(y_true, y_prob),
        "brier":      brier_score_loss(y_true, y_prob),
        "auc":        roc_auc_score(y_true, y_prob),
        "mean_prob":  y_prob.mean(),
    }
    return metrics


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned calibration: how close is predicted prob to actual win rate per bin?"""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted": mean_pred, "actual": frac_pos,
                         "gap": frac_pos - mean_pred})


def plot_calibration(results: list[dict], save_path: Path):
    """Save calibration curves for all models to a single PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for r in results:
        frac, pred = calibration_curve(r["y_true"], r["y_prob"], n_bins=10, strategy="quantile")
        ax.plot(pred, frac, marker="o", label=r["label"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual win rate)")
    ax.set_title("Calibration curves (all walk-forward folds)")
    ax.legend()

    ax2 = axes[1]
    for r in results:
        ax2.hist(r["y_prob"], bins=30, alpha=0.5, label=r["label"], density=True)
    ax2.set_xlabel("Predicted home-win probability")
    ax2.set_title("Predicted probability distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Calibration plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def run_walk_forward():
    X_all, y_all, meta_all = load_feature_matrix()

    all_metrics = []
    cal_data_lr  = {"label": "Logistic", "y_true": [], "y_prob": []}
    cal_data_xgb = {"label": "XGBoost",  "y_true": [], "y_prob": []}

    print(f"\n{'='*60}")
    print("Walk-forward validation")
    print(f"{'='*60}")

    for train_seasons, test_season in walk_forward_splits():
        train_mask = meta_all["season"].isin(train_seasons)
        test_mask  = meta_all["season"] == test_season

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        print(f"\nTrain {train_seasons} → Test {test_season}  "
              f"({train_mask.sum()} train / {test_mask.sum()} test games)")

        for name, model in [("Logistic", make_logistic()), ("XGBoost", make_xgb())]:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]

            m = evaluate(y_test.values, prob, f"{name} {test_season}")
            m["train_seasons"] = str(train_seasons)
            m["test_season"]   = test_season
            all_metrics.append(m)

            print(f"  {name:<12} acc={m['accuracy']:.3f}  "
                  f"logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
                  f"auc={m['auc']:.3f}  mean_p={m['mean_prob']:.3f}")

            if name == "Logistic":
                cal_data_lr["y_true"].extend(y_test.values.tolist())
                cal_data_lr["y_prob"].extend(prob.tolist())
            else:
                cal_data_xgb["y_true"].extend(y_test.values.tolist())
                cal_data_xgb["y_prob"].extend(prob.tolist())

    # Aggregate across folds
    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n{'='*60}")
    print("Aggregate walk-forward metrics (mean across test seasons):")
    # Group by model name prefix
    for prefix in ["Logistic", "XGBoost"]:
        sub = metrics_df[metrics_df["model"].str.startswith(prefix)]
        print(f"\n  {prefix}:")
        print(f"    accuracy  = {sub['accuracy'].mean():.3f}  ± {sub['accuracy'].std():.3f}")
        print(f"    log_loss  = {sub['log_loss'].mean():.4f}  ± {sub['log_loss'].std():.4f}")
        print(f"    brier     = {sub['brier'].mean():.4f}  ± {sub['brier'].std():.4f}")
        print(f"    auc       = {sub['auc'].mean():.3f}  ± {sub['auc'].std():.3f}")

    # Calibration report on pooled out-of-sample predictions
    print(f"\n{'='*60}")
    print("Calibration (pooled out-of-sample predictions):")
    for cal in [cal_data_lr, cal_data_xgb]:
        yt = np.array(cal["y_true"])
        yp = np.array(cal["y_prob"])
        print(f"\n  {cal['label']}  (n={len(yt):,})")
        cal_df = calibration_report(yt, yp, n_bins=10)
        print(cal_df.to_string(index=False, float_format="{:.3f}".format))

    # Calibration plot
    cal_data_lr["y_true"]  = np.array(cal_data_lr["y_true"])
    cal_data_lr["y_prob"]  = np.array(cal_data_lr["y_prob"])
    cal_data_xgb["y_true"] = np.array(cal_data_xgb["y_true"])
    cal_data_xgb["y_prob"] = np.array(cal_data_xgb["y_prob"])
    plot_calibration([cal_data_lr, cal_data_xgb], DATA_DIR / "calibration.png")

    return metrics_df, cal_data_lr, cal_data_xgb


# ---------------------------------------------------------------------------
# Train final model on all data and save
# ---------------------------------------------------------------------------

def train_final_model(model_type: str = "logistic", before_date: str | None = None):
    """
    Train on all available historical data and save the model to disk.
    This is the model used for live predictions and paper trading.

    before_date excludes games on or after that ISO date. For same-day live
    picks, pass today's date so completed early games cannot leak into the
    model used for later games.
    """
    X, y, meta = load_feature_matrix(before_date=before_date)
    cutoff = f" before {before_date}" if before_date else ""
    print(f"\nTraining final {model_type} model on {len(X):,} games{cutoff} "
          f"({meta['season'].min()}–{meta['season'].max()})...")

    if model_type == "logistic":
        model = make_logistic()
    else:
        model = make_xgb()

    model.fit(X, y)

    save_path = MODEL_DIR / f"model_{model_type}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    print(f"  Saved to {save_path}")

    # Print logistic coefficients (only available on uncalibrated pipeline)
    if model_type == "logistic":
        try:
            # Access one calibrated estimator's inner pipeline to read coefficients
            inner = model.calibrated_classifiers_[0].estimator
            clf   = inner.named_steps["clf"]
            coefs = clf.coef_[0]
            print("\n  Logistic regression coefficients (scaled, one fold):")
            for feat, coef in sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {feat:<25} {coef:+.4f}")
        except Exception:
            pass

    return model


# ---------------------------------------------------------------------------
# F5 model
# ---------------------------------------------------------------------------

def train_f5_model(model_type: str = "logistic", before_date: str | None = None):
    """
    Train the first-5-innings moneyline model and save to model_f5_logistic.pkl.
    Uses the same logistic regression architecture as the full-game model but
    trained on home_win_f5 and without team bullpen ERA/FIP features.
    """
    X, y, meta = load_f5_feature_matrix(before_date=before_date)
    cutoff = f" before {before_date}" if before_date else ""
    print(f"\nTraining F5 {model_type} model on {len(X):,} games{cutoff} "
          f"({meta['season'].min()}–{meta['season'].max()})...")

    model = make_logistic() if model_type == "logistic" else make_xgb()
    model.fit(X, y)

    save_path = MODEL_DIR / f"model_f5_{model_type}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": F5_FEATURE_COLS}, f)
    print(f"  Saved to {save_path}")

    if model_type == "logistic":
        try:
            inner = model.calibrated_classifiers_[0].estimator
            coefs = inner.named_steps["clf"].coef_[0]
            print("\n  F5 logistic coefficients (scaled, one fold):")
            for feat, coef in sorted(zip(F5_FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {feat:<25} {coef:+.4f}")
        except Exception:
            pass

    return model


# ---------------------------------------------------------------------------
# Feature importance (XGBoost)
# ---------------------------------------------------------------------------

def print_xgb_importance(model):
    xgb_clf = model.named_steps["clf"]
    imp = pd.Series(
        xgb_clf.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\n  XGBoost feature importances:")
    for feat, val in imp.items():
        bar = "█" * int(val * 200)
        print(f"    {feat:<25} {val:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics_df, cal_lr, cal_xgb = run_walk_forward()

    print(f"\n{'='*60}")
    print("Training final models on all data...")

    lr_model  = train_final_model("logistic")
    xgb_model = train_final_model("xgboost")
    print_xgb_importance(xgb_model)

    print(f"\n{'='*60}")
    print("Phase 1 complete.")
    print("Deliverable: home-team win probability model.")
    print("Next step: Phase 2 — convert probabilities to value bets.")
