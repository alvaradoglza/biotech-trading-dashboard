"""
compare_thresholds.py — Compare dynamic vs static P85 threshold strategies.

Trains the GBM model across 5 temporal splits for each of three threshold approaches:
  1. dynamic    — threshold = P85 of train_df["return_30d"] computed per split
  2. static_new — threshold = 18.0425% (true P85 of clean Supabase data)
  3. static_old — threshold = 7.2225%  (original hardcoded value, for reference)

Reports per-split metrics and cross-split averages.
No writes to Supabase.

Usage:
  python scripts/compare_thresholds.py
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.features import build_feature_matrix, build_labels
from pipeline.ml_config import (
    DROP_FEATURES, GBM_LEARNING_RATE, GBM_MAX_DEPTH, GBM_N_ESTIMATORS,
    RANDOM_STATE, SF_COLS,
)
from pipeline.supabase_writer import get_client

# ── Config ────────────────────────────────────────────────────────────────────

EVAL_FRACS   = [0.65, 0.70, 0.75, 0.80, 0.85]
HORIZON      = "30d"
PREDICT_DAYS = 14   # days of recent announcements used to count live signals

APPROACHES = {
    "dynamic":     None,      # computed from train_df each split
    "static_new":  18.0425,   # real P85 on clean data
    "static_old":  7.2225,    # original (bad) hardcoded value
}

EVENT_TYPE_CORRECTIONS = {
    "FDA_APPROVAL": "FDA_ORIG",
    "TRIAL_RESULTS": "CT_COMPLETED",
    "TRIAL_START":   "CT_RECRUITING",
    "TRIAL_UPDATE":  "CT_UNKNOWN",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def paginate(sb, build_query, page_size: int = 1000) -> list[dict]:
    rows, offset = [], 0
    while True:
        resp = build_query(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def load_labeled(sb) -> pd.DataFrame:
    rows = paginate(sb, lambda lo, hi: (
        sb.table("announcements")
        .select("id, ticker, source, event_type, published_at, raw_text, return_30d, return_5d")
        .not_.is_("return_30d", "null")
        .order("published_at", desc=False)
        .range(lo, hi)
    ))
    df = pd.DataFrame(rows)
    before = len(df)
    df = df[df["return_30d"] != 0.0].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=False, errors="coerce")
    df = df.dropna(subset=["published_at"])
    if df["published_at"].dt.tz is not None:
        df["published_at"] = df["published_at"].dt.tz_localize(None)
    df = df.sort_values("published_at").reset_index(drop=True)
    print(f"Labeled data: {before} total → {before - len(df)} zero-return dropped → {len(df)} clean rows")
    return df


def load_predict(sb, days_back: int) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    rows = paginate(sb, lambda lo, hi: (
        sb.table("announcements")
        .select("id, ticker, source, event_type, published_at, raw_text")
        .gte("published_at", cutoff)
        .order("published_at", desc=False)
        .range(lo, hi)
    ))
    df = pd.DataFrame(rows)
    df["event_type"] = df["event_type"].replace(EVENT_TYPE_CORRECTIONS)
    print(f"Predict data: {len(df)} announcements (last {days_back} days, event_types corrected)")
    return df


# ── Model helpers ─────────────────────────────────────────────────────────────

def fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ohe.fit(df[["source", "event_type"]])
    return ohe


def build_X(df: pd.DataFrame, ohe: OneHotEncoder) -> np.ndarray:
    X = build_feature_matrix(ohe, df)
    if DROP_FEATURES:
        n_ohe = X.shape[1] - len(SF_COLS)
        drop_idxs = [n_ohe + SF_COLS.index(f) for f in DROP_FEATURES if f in SF_COLS]
        if drop_idxs:
            X = np.delete(X, drop_idxs, axis=1)
    return X


def build_y(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return build_labels(df, HORIZON, threshold)


def train_clf(X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)
    return clf


def evaluate(clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy":     float(accuracy_score(y_test, y_pred)),
        "precision":    float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity":  float(specificity),
        "f1":           float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":          auc,
    }


def predict_signals(clf, ohe, predict_df) -> tuple[int, float, float]:
    """Returns (n_buy, median_prob, max_prob)."""
    if predict_df is None or predict_df.empty:
        return 0, 0.0, 0.0
    X = build_X(predict_df, ohe)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return int(preds.sum()), float(np.median(probs)), float(np.max(probs))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison(labeled_df: pd.DataFrame, predict_df: pd.DataFrame) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {name: [] for name in APPROACHES}

    for frac in EVAL_FRACS:
        n = len(labeled_df)
        split = max(1, int(n * frac))
        train_df = labeled_df.iloc[:split].copy()
        test_df  = labeled_df.iloc[split:].copy()

        if len(train_df) < 30 or len(test_df) < 5:
            print(f"  split {frac:.0%}: insufficient data — skipping")
            continue

        split_date = test_df["published_at"].iloc[0].date()
        print(f"\n  split {frac:.0%}  |  train={len(train_df)}  test={len(test_df)}  cutoff={split_date}")

        for approach_name, static_threshold in APPROACHES.items():
            # Determine threshold
            if approach_name == "dynamic":
                threshold = float(np.percentile(train_df[f"return_{HORIZON}"], 85))
            else:
                threshold = static_threshold

            # Eval phase: fit OHE + train on split, evaluate on test
            ohe_eval = fit_ohe(train_df)
            X_train = build_X(train_df, ohe_eval)
            y_train = build_y(train_df, threshold)

            if len(np.unique(y_train)) < 2:
                print(f"    {approach_name}: single-class train labels — skipping")
                continue

            clf_eval = train_clf(X_train, y_train)

            X_test = build_X(test_df, ohe_eval)
            y_test  = build_y(test_df, threshold)
            metrics = evaluate(clf_eval, X_test, y_test)

            # Production phase: retrain on 100% of labeled data, predict on recent
            ohe_prod = fit_ohe(labeled_df)
            X_all = build_X(labeled_df, ohe_prod)
            y_all = build_y(labeled_df, threshold)
            clf_prod = train_clf(X_all, y_all)
            n_buy, med_prob, max_prob = predict_signals(clf_prod, ohe_prod, predict_df)

            pos_train = float(y_train.mean())
            pos_test  = float(y_test.mean())

            row = {
                "frac":           frac,
                "threshold":      threshold,
                "pos_train":      pos_train,
                "pos_test":       pos_test,
                "n_buy":          n_buy,
                "med_prob":       med_prob,
                "max_prob":       max_prob,
                "n_labeled_rows": len(labeled_df),
                "n_predict_rows": len(predict_df) if predict_df is not None else 0,
                **metrics,
            }
            results[approach_name].append(row)

            print(
                f"    {approach_name:<12}  thresh={threshold:5.2f}%  "
                f"pos_train={pos_train:.1%}  pos_test={pos_test:.1%}  "
                f"spec={metrics['specificity']:.3f}  prec={metrics['precision']:.3f}  "
                f"rec={metrics['recall']:.3f}  auc={metrics['auc']:.3f}  "
                f"buy_signals={n_buy}  max_p={max_prob:.3f}"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — averaged across all splits")
    print("=" * 80)

    hdr = f"{'approach':<14} {'threshold':>10} {'pos_train':>10} {'pos_test':>9} {'specificity':>12} {'precision':>10} {'recall':>8} {'AUC':>7} {'buy_signals':>12} {'max_prob':>9}"
    print(hdr)
    print("-" * 90)

    for name, rows in results.items():
        if not rows:
            print(f"  {name}: no results")
            continue
        df_r = pd.DataFrame(rows)
        print(
            f"{name:<14} "
            f"{df_r['threshold'].mean():>9.2f}% "
            f"{df_r['pos_train'].mean():>10.1%} "
            f"{df_r['pos_test'].mean():>9.1%} "
            f"{df_r['specificity'].mean():>12.3f} "
            f"{df_r['precision'].mean():>10.3f} "
            f"{df_r['recall'].mean():>8.3f} "
            f"{df_r['auc'].mean():>7.3f} "
            f"{df_r['n_buy'].mean():>12.1f} "
            f"{df_r['max_prob'].mean():>9.3f}"
        )

    print()
    print("── Detail per split ─────────────────────────────────────────────────")
    for name, rows in results.items():
        if not rows:
            continue
        print(f"\n  {name}")
        df_r = pd.DataFrame(rows)
        print(df_r[["frac", "threshold", "pos_train", "pos_test", "specificity", "precision", "recall", "auc", "n_buy", "max_prob"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return results


def save_results(results: dict[str, list[dict]], sb) -> None:
    run_date = date.today().isoformat()
    rows = []
    for approach, splits in results.items():
        for row in splits:
            rows.append({
                "run_date":        run_date,
                "approach":        approach,
                "eval_frac":       float(row["frac"]),
                "threshold":       float(row["threshold"]),
                "pos_train":       float(row["pos_train"]),
                "pos_test":        float(row["pos_test"]),
                "specificity":     float(row["specificity"]),
                "precision_score": float(row["precision"]),
                "recall":          float(row["recall"]),
                "f1":              float(row["f1"]),
                "auc":             float(row["auc"]),
                "n_buy":           int(row["n_buy"]),
                "med_prob":        float(row["med_prob"]),
                "max_prob":        float(row["max_prob"]),
                "n_labeled_rows":  int(row["n_labeled_rows"]),
                "n_predict_rows":  int(row["n_predict_rows"]),
            })
    if not rows:
        return
    resp = (
        sb.table("threshold_experiments")
        .upsert(rows, on_conflict="run_date,approach,eval_frac")
        .execute()
    )
    n = len(resp.data) if resp.data else 0
    print(f"\nSaved {n} rows to threshold_experiments table in Supabase.")


def main() -> None:
    sb = get_client()
    print("Loading data from Supabase...\n")
    labeled_df  = load_labeled(sb)
    predict_df  = load_predict(sb, days_back=PREDICT_DAYS)

    print(f"\nRunning {len(EVAL_FRACS)} splits × {len(APPROACHES)} approaches = {len(EVAL_FRACS)*len(APPROACHES)} model trains...\n")
    results = run_comparison(labeled_df, predict_df)
    save_results(results, sb)


if __name__ == "__main__":
    main()
