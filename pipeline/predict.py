"""
predict.py — Daily model training, evaluation, and prediction.

Strategy:
  - Train on all labeled announcements older than 4 weeks (return_30d available)
  - Evaluate on the most recent 4 weeks (return_30d available but recent)
  - Predict on announcements without return_30d (new, unresolved)
  - Log metrics to Supabase model_runs table + optionally MLflow
  - Save model artifact to pipeline/models/
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder

from pipeline.features import build_feature_matrix, build_labels, fit_ohe
from pipeline.ml_config import (
    SF_COLS, GBM_N_ESTIMATORS, GBM_MAX_DEPTH, GBM_LEARNING_RATE,
    RANDOM_STATE, P85_30D, P85_5D, DROP_FEATURES,
)

logger = logging.getLogger(__name__)

# Model output directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def run_daily_prediction(
    announcements_df: pd.DataFrame,
    horizon: str = "30d",
    test_weeks: int = 4,
) -> tuple[list[dict], dict]:
    """Train model, evaluate, and generate predictions for new announcements.

    Args:
        announcements_df: Full announcements DataFrame from Supabase.
            Required columns: ticker, source, event_type, published_at, raw_text,
            return_30d (nullable), return_5d (nullable).
        horizon: '30d' or '5d' (default '30d').
        test_weeks: How many weeks back to use as test set (default 4).

    Returns:
        (predictions, metrics)
        - predictions: list of prediction dicts for announcements without return labels
        - metrics: dict with accuracy, precision, recall, specificity, f1_score, etc.
    """
    if announcements_df.empty:
        logger.warning("Empty announcements DataFrame — skipping prediction")
        return [], {}

    # Normalize published_at
    announcements_df = announcements_df.copy()
    announcements_df["published_at"] = pd.to_datetime(announcements_df["published_at"], utc=False, errors="coerce")
    if announcements_df["published_at"].dt.tz is not None:
        announcements_df["published_at"] = announcements_df["published_at"].dt.tz_localize(None)

    cutoff_date = pd.Timestamp(datetime.utcnow()) - pd.DateOffset(weeks=test_weeks)
    return_col = f"return_{horizon}"

    # Labeled data: announcements where we know the return outcome
    labeled = announcements_df[announcements_df[return_col].notna()].copy()

    # Unlabeled data: recent announcements without outcome yet (predict on these)
    unlabeled = announcements_df[announcements_df[return_col].isna()].copy()

    if len(labeled) < 50:
        logger.warning(
            "Only %d labeled announcements — model may be unreliable (need ≥50)", len(labeled)
        )

    # Train/test split: everything before cutoff is train, recent is test
    train_df = labeled[labeled["published_at"] < cutoff_date].copy()
    test_df = labeled[labeled["published_at"] >= cutoff_date].copy()

    logger.info(
        "Data split: %d train, %d test, %d unlabeled (cutoff=%s)",
        len(train_df), len(test_df), len(unlabeled), cutoff_date.date()
    )

    if len(train_df) < 30:
        logger.warning("Insufficient training data (%d rows) — skipping", len(train_df))
        return [], {"error": "insufficient_data", "n_train": len(train_df)}

    # Fit OHE on training data only (not test/full dataset to avoid leakage)
    ohe = _fit_ohe_train(train_df)

    # Build features
    threshold = P85_30D if horizon == "30d" else P85_5D
    X_train, y_train = _build_Xy(train_df, ohe, horizon, threshold)

    if len(np.unique(y_train)) < 2:
        logger.warning("Training labels have only one class — skipping")
        return [], {"error": "single_class_labels"}

    # Train model
    clf = _train(X_train, y_train)

    # Evaluate on test set
    metrics = {}
    if len(test_df) >= 5:
        X_test, y_test = _build_Xy(test_df, ohe, horizon, threshold)
        metrics = _evaluate(clf, X_test, y_test, train_df, test_df, horizon)
    else:
        logger.warning("Test set too small (%d rows) — skipping evaluation", len(test_df))
        metrics = {
            "horizon": horizon,
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "n_positive_train": int(y_train.sum()),
            "warning": "test_set_too_small",
        }

    # Save model artifact
    model_version = _save_model(clf, ohe, horizon)
    metrics["model_version"] = model_version
    metrics["run_date"] = date.today().isoformat()

    # Generate predictions for unlabeled announcements
    predictions = []
    if len(unlabeled) > 0:
        try:
            X_unlabeled, _ = _build_Xy(unlabeled, ohe, horizon, threshold, has_labels=False)
            preds = clf.predict(X_unlabeled)
            probs = clf.predict_proba(X_unlabeled)[:, 1]

            for i, (_, row) in enumerate(unlabeled.iterrows()):
                predictions.append({
                    "announcement_id": row.get("id"),
                    "ticker": row.get("ticker"),
                    "published_at": row.get("published_at"),
                    "predicted_label": int(preds[i]),
                    "predicted_probability": float(probs[i]),
                    "model_version": model_version,
                })
            logger.info(
                "Generated %d predictions (%d BUY signals)",
                len(predictions), sum(p["predicted_label"] for p in predictions)
            )
        except Exception as e:
            logger.error("Prediction on unlabeled data failed: %s", e)

    # Log to MLflow if configured
    _log_mlflow(clf, metrics)

    return predictions, metrics


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_ohe_train(train_df: pd.DataFrame) -> OneHotEncoder:
    """Fit OHE on training data only."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ohe.fit(train_df[["source", "event_type"]])
    return ohe


def _build_Xy(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    horizon: str,
    threshold: float,
    has_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build feature matrix and labels from a DataFrame."""
    X = build_feature_matrix(ohe, df)

    # Drop sf_word_count feature (matches backtesting config)
    if DROP_FEATURES:
        n_ohe = X.shape[1] - len(SF_COLS)
        drop_idxs = [n_ohe + SF_COLS.index(f) for f in DROP_FEATURES if f in SF_COLS]
        if drop_idxs:
            X = np.delete(X, drop_idxs, axis=1)

    if has_labels:
        y = build_labels(df, horizon, threshold)
        return X, y
    return X, None


def _train(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """Train GradientBoostingClassifier (same params as backtesting config)."""
    clf = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    logger.info(
        "Model trained: %d samples, %d positive (%.1f%%)",
        len(y_train), int(y_train.sum()), 100 * y_train.mean()
    )
    return clf


def _evaluate(
    clf: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: str,
) -> dict:
    """Evaluate model and compute accuracy, precision, recall, specificity, F1, ROC AUC."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None  # only one class in test set

    metrics = {
        "horizon": horizon,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_score": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(auc) if auc is not None else None,
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "n_positive_train": int(y_test.sum()),  # label dist in test (for context)
        "n_positive_test": int(y_test.sum()),
    }

    logger.info(
        "Evaluation (horizon=%s): acc=%.3f prec=%.3f rec=%.3f spec=%.3f f1=%.3f auc=%s",
        horizon,
        metrics["accuracy"],
        metrics["precision_score"],
        metrics["recall"],
        metrics["specificity"],
        metrics["f1_score"],
        f"{auc:.3f}" if auc is not None else "n/a",
    )
    return metrics


def _save_model(
    clf: GradientBoostingClassifier,
    ohe: OneHotEncoder,
    horizon: str,
) -> str:
    """Save model and OHE to disk. Returns model version string."""
    import joblib

    today = date.today().strftime("%Y%m%d")
    model_version = f"{horizon}_{today}"

    clf_path = MODELS_DIR / f"gbm_{model_version}.joblib"
    ohe_path = MODELS_DIR / f"ohe_{model_version}.joblib"

    joblib.dump(clf, clf_path)
    joblib.dump(ohe, ohe_path)

    # Also save as "latest" for easy loading
    joblib.dump(clf, MODELS_DIR / f"gbm_{horizon}_latest.joblib")
    joblib.dump(ohe, MODELS_DIR / f"ohe_{horizon}_latest.joblib")

    logger.info("Model saved: %s", clf_path)
    return model_version


def _log_mlflow(clf, metrics: dict) -> None:
    """Log metrics and model to MLflow if MLFLOW_TRACKING_URI is configured."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)

        # DagsHub (and other hosted MLflow servers) require HTTP basic auth.
        # Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in env/secrets.
        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        experiment_name = "biotech-trading-daily"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"daily_{metrics.get('run_date', 'unknown')}_{metrics.get('horizon', '30d')}") as run:
            # Log parameters
            mlflow.log_param("horizon", metrics.get("horizon"))
            mlflow.log_param("n_train_samples", metrics.get("n_train_samples"))
            mlflow.log_param("n_test_samples", metrics.get("n_test_samples"))
            mlflow.log_param("gbm_n_estimators", GBM_N_ESTIMATORS)
            mlflow.log_param("gbm_max_depth", GBM_MAX_DEPTH)
            mlflow.log_param("gbm_learning_rate", GBM_LEARNING_RATE)

            # Log metrics
            for key in ["accuracy", "precision_score", "recall", "specificity", "f1_score", "roc_auc"]:
                if metrics.get(key) is not None:
                    mlflow.log_metric(key, metrics[key])

            # Log model
            mlflow.sklearn.log_model(clf, "model")

            # Store run ID and URL back into metrics for Supabase
            metrics["mlflow_run_id"] = run.info.run_id
            metrics["mlflow_experiment_url"] = f"{tracking_uri}/#/experiments/{run.info.experiment_id}"

            logger.info("MLflow run logged: %s", run.info.run_id)

    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
