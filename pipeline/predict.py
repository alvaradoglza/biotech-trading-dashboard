"""
predict.py — Daily model training, evaluation, and prediction.

Two-phase strategy:
  Phase 1 (eval):
    - Sort all labeled announcements by published_at
    - Train on first 80%, test on last 20%
    - Record accuracy / precision / recall / F1 / AUC → Supabase + MLflow

  Phase 2 (production):
    - Retrain on 100% of labeled data (maximises signal for live predictions)
    - Predict on new_announcements (freshly fetched, no return label yet)

Why two phases:
  - Eval phase gives an unbiased view of model quality on held-out data
  - Production phase gives the strongest possible model for live signals
"""

import logging
import os
from datetime import date
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

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def run_daily_prediction(
    labeled_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    horizon: str = "30d",
    eval_train_frac: float = 0.80,
) -> tuple[list[dict], dict]:
    """Train model (two-phase) and generate predictions for new announcements.

    Args:
        labeled_df: Announcements with a known return label (return_30d not null).
                    Required columns: ticker, source, event_type, published_at, raw_text,
                    return_30d, return_5d.
        predict_df: New announcements to generate signals for (no return label needed).
                    Required columns: id, ticker, source, event_type, published_at, raw_text.
        horizon:    '30d' or '5d'.
        eval_train_frac: Fraction of labeled data used for training in the eval phase
                    (the remaining fraction becomes the held-out test set). Default 0.80.

    Returns:
        (predictions, metrics)
        - predictions: list of prediction dicts for the new announcements
        - metrics: dict with eval-phase accuracy, precision, recall, etc.
    """
    if labeled_df.empty:
        logger.warning("No labeled data — skipping model training")
        return [], {}

    # ── Normalize timestamps ──────────────────────────────────────────────────
    labeled_df = labeled_df.copy()
    labeled_df["published_at"] = pd.to_datetime(labeled_df["published_at"], utc=False, errors="coerce")
    if labeled_df["published_at"].dt.tz is not None:
        labeled_df["published_at"] = labeled_df["published_at"].dt.tz_localize(None)
    labeled_df = labeled_df.dropna(subset=["published_at"])

    return_col = f"return_{horizon}"
    threshold = P85_30D if horizon == "30d" else P85_5D

    # ── Sort chronologically for temporal split ───────────────────────────────
    labeled_df = labeled_df.sort_values("published_at").reset_index(drop=True)
    n_total = len(labeled_df)
    split_idx = max(1, int(n_total * eval_train_frac))

    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()

    logger.info(
        "80/20 split: %d train | %d test (split at record %d/%d, cutoff date %s)",
        len(train_df), len(test_df), split_idx, n_total,
        test_df["published_at"].iloc[0].date() if len(test_df) > 0 else "—",
    )

    if len(train_df) < 30:
        logger.warning("Insufficient training data (%d rows) — skipping", len(train_df))
        return [], {"error": "insufficient_data", "n_train": len(train_df)}

    # ── Phase 1: Eval (80% train / 20% test) ─────────────────────────────────
    ohe_eval = _fit_ohe(train_df)
    X_train, y_train = _build_Xy(train_df, ohe_eval, horizon, threshold)

    if len(np.unique(y_train)) < 2:
        logger.warning("Training labels have only one class — skipping")
        return [], {"error": "single_class_labels"}

    clf_eval = _train(X_train, y_train, tag="eval")

    metrics: dict = {}
    if len(test_df) >= 5:
        X_test, y_test = _build_Xy(test_df, ohe_eval, horizon, threshold)
        metrics = _evaluate(clf_eval, X_test, y_test, train_df, test_df, horizon)
    else:
        logger.warning(
            "Test set only %d rows — evaluation skipped (need ≥5). "
            "Increase labeled data or adjust eval_train_frac.", len(test_df)
        )
        metrics = {
            "horizon": horizon,
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "n_positive_train": int(y_train.sum()),
            "warning": "test_set_too_small",
        }

    # ── Phase 2: Production (retrain on 100% of labeled data) ────────────────
    ohe_prod = _fit_ohe(labeled_df)
    X_all, y_all = _build_Xy(labeled_df, ohe_prod, horizon, threshold)
    clf_prod = _train(X_all, y_all, tag="prod")

    model_version = _save_model(clf_prod, ohe_prod, horizon)
    metrics["model_version"] = model_version
    metrics["run_date"] = date.today().isoformat()
    metrics["eval_train_frac"] = eval_train_frac

    # ── Predict on new announcements ─────────────────────────────────────────
    predictions: list[dict] = []
    if predict_df is not None and len(predict_df) > 0:
        predict_df = predict_df.copy()
        predict_df["published_at"] = pd.to_datetime(predict_df["published_at"], utc=False, errors="coerce")
        if predict_df["published_at"].dt.tz is not None:
            predict_df["published_at"] = predict_df["published_at"].dt.tz_localize(None)

        try:
            X_pred, _ = _build_Xy(predict_df, ohe_prod, horizon, threshold, has_labels=False)
            preds = clf_prod.predict(X_pred)
            probs = clf_prod.predict_proba(X_pred)[:, 1]

            for i, (_, row) in enumerate(predict_df.iterrows()):
                predictions.append({
                    "announcement_id": row.get("id"),
                    "ticker": row.get("ticker"),
                    "published_at": row.get("published_at"),
                    "predicted_label": int(preds[i]),
                    "predicted_probability": float(probs[i]),
                    "model_version": model_version,
                })

            logger.info(
                "Generated %d predictions on new announcements (%d BUY signals)",
                len(predictions), sum(p["predicted_label"] for p in predictions),
            )
        except Exception as e:
            logger.error("Prediction on new announcements failed: %s", e, exc_info=True)
    else:
        logger.info("No new announcements to predict on.")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    _log_mlflow(clf_prod, metrics)

    return predictions, metrics


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ohe.fit(df[["source", "event_type"]])
    return ohe


def _build_Xy(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    horizon: str,
    threshold: float,
    has_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    X = build_feature_matrix(ohe, df)

    if DROP_FEATURES:
        n_ohe = X.shape[1] - len(SF_COLS)
        drop_idxs = [n_ohe + SF_COLS.index(f) for f in DROP_FEATURES if f in SF_COLS]
        if drop_idxs:
            X = np.delete(X, drop_idxs, axis=1)

    if has_labels:
        y = build_labels(df, horizon, threshold)
        return X, y
    return X, None


def _train(X: np.ndarray, y: np.ndarray, tag: str = "") -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    logger.info(
        "Model trained [%s]: %d samples, %d positive (%.1f%%)",
        tag, len(y), int(y.sum()), 100 * y.mean(),
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
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None

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
        "n_positive_train": int((y_test >= 0).sum()),  # placeholder; y_train not passed here
        "n_positive_test": int(y_test.sum()),
    }

    logger.info(
        "Eval [horizon=%s]: acc=%.3f prec=%.3f rec=%.3f spec=%.3f f1=%.3f auc=%s",
        horizon,
        metrics["accuracy"], metrics["precision_score"], metrics["recall"],
        metrics["specificity"], metrics["f1_score"],
        f"{auc:.3f}" if auc is not None else "n/a",
    )
    return metrics


def _save_model(clf: GradientBoostingClassifier, ohe: OneHotEncoder, horizon: str) -> str:
    import joblib

    today = date.today().strftime("%Y%m%d")
    model_version = f"{horizon}_{today}"

    joblib.dump(clf, MODELS_DIR / f"gbm_{model_version}.joblib")
    joblib.dump(ohe, MODELS_DIR / f"ohe_{model_version}.joblib")
    joblib.dump(clf, MODELS_DIR / f"gbm_{horizon}_latest.joblib")
    joblib.dump(ohe, MODELS_DIR / f"ohe_{horizon}_latest.joblib")

    logger.info("Model saved: gbm_%s.joblib", model_version)
    return model_version


def _log_mlflow(clf, metrics: dict) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)

        username = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_experiment("biotech-trading-daily")

        with mlflow.start_run(
            run_name=f"daily_{metrics.get('run_date', 'unknown')}_{metrics.get('horizon', '30d')}"
        ) as run:
            mlflow.log_param("horizon", metrics.get("horizon"))
            mlflow.log_param("n_train_samples", metrics.get("n_train_samples"))
            mlflow.log_param("n_test_samples", metrics.get("n_test_samples"))
            mlflow.log_param("eval_train_frac", metrics.get("eval_train_frac", 0.80))
            mlflow.log_param("gbm_n_estimators", GBM_N_ESTIMATORS)
            mlflow.log_param("gbm_max_depth", GBM_MAX_DEPTH)
            mlflow.log_param("gbm_learning_rate", GBM_LEARNING_RATE)

            for key in ["accuracy", "precision_score", "recall", "specificity", "f1_score", "roc_auc"]:
                if metrics.get(key) is not None:
                    mlflow.log_metric(key, metrics[key])

            mlflow.sklearn.log_model(clf, "model")

            metrics["mlflow_run_id"] = run.info.run_id
            metrics["mlflow_experiment_url"] = (
                f"{tracking_uri}/#/experiments/{run.info.experiment_id}"
            )
            logger.info("MLflow run logged: %s", run.info.run_id)

    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
