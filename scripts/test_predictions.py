"""
test_predictions.py — Run the model locally against recent Supabase announcements.

Reads labeled data + last N days of announcements from Supabase.
Normalizes event_type names to the OHE-known canonical values (same mapping as the
corrected fetch functions) so rows already in the DB can be tested fairly.
Prints a full prediction summary. No writes to Supabase.

Usage:
  python scripts/test_predictions.py
  python scripts/test_predictions.py --days 14
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.predict import run_daily_prediction
from pipeline.supabase_writer import get_client

# Map old (wrong) event_type names → OHE-known canonical names.
# Mirrors the logic now in fetch_fda.py and fetch_clinical_trials.py.
EVENT_TYPE_CORRECTIONS = {
    "FDA_APPROVAL": "FDA_ORIG",
    "TRIAL_RESULTS": "CT_COMPLETED",
    "TRIAL_START":   "CT_RECRUITING",
    "TRIAL_UPDATE":  "CT_UNKNOWN",
}


def _paginate(sb, query_builder, page_size: int = 1000) -> list[dict]:
    all_rows: list[dict] = []
    offset = 0
    while True:
        resp = query_builder(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_rows


def main(days_back: int = 14) -> None:
    sb = get_client()

    # ── Labeled data ──────────────────────────────────────────────────────────
    print(f"Loading labeled announcements from Supabase...")
    labeled_rows = _paginate(
        sb,
        lambda lo, hi: (
            sb.table("announcements")
            .select("id, ticker, source, event_type, published_at, raw_text, return_30d, return_5d")
            .not_.is_("return_30d", "null")
            .order("published_at", desc=False)
            .range(lo, hi)
        ),
    )
    labeled_df = pd.DataFrame(labeled_rows)
    before = len(labeled_df)
    labeled_df = labeled_df[labeled_df["return_30d"] != 0.0].copy()
    n_dropped = before - len(labeled_df)
    print(f"  {before} rows loaded, {n_dropped} zero-return rows dropped → {len(labeled_df)} used for training")

    if labeled_df.empty:
        print("No labeled data — cannot run. Ensure Supabase is seeded.")
        sys.exit(1)

    pos_rate = (labeled_df["return_30d"] >= 7.2225).mean()
    print(f"  Positive rate in labeled data: {pos_rate:.1%}  (expected ~15% if labels are clean)")

    # ── Predict data (last N days) ────────────────────────────────────────────
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    print(f"\nLoading announcements since {cutoff} ({days_back} days back)...")
    predict_rows = _paginate(
        sb,
        lambda lo, hi: (
            sb.table("announcements")
            .select("id, ticker, source, event_type, published_at, raw_text")
            .gte("published_at", cutoff)
            .order("published_at", desc=False)
            .range(lo, hi)
        ),
    )
    predict_df = pd.DataFrame(predict_rows)
    print(f"  {len(predict_df)} announcements loaded")

    if predict_df.empty:
        print("No recent announcements to predict on.")
        sys.exit(0)

    # Show event_type distribution before and after correction
    print(f"\n  event_type distribution (before correction):")
    for evt, cnt in predict_df["event_type"].value_counts().items():
        tag = "  ← WRONG NAME" if evt in EVENT_TYPE_CORRECTIONS else ""
        print(f"    {evt:<35} {cnt:>5}{tag}")

    predict_df["event_type"] = predict_df["event_type"].replace(EVENT_TYPE_CORRECTIONS)
    n_corrected = predict_df["event_type"].isin(EVENT_TYPE_CORRECTIONS.values()).sum()
    print(f"\n  event_type distribution (after correction):")
    for evt, cnt in predict_df["event_type"].value_counts().items():
        print(f"    {evt:<35} {cnt:>5}")

    # ── Run model ─────────────────────────────────────────────────────────────
    print("\nTraining model and predicting...")
    predictions, metrics = run_daily_prediction(
        labeled_df=labeled_df,
        predict_df=predict_df,
        horizon="30d",
        eval_train_frac=0.80,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("MODEL METRICS")
    print("=" * 55)
    for k in ["n_train_samples", "n_test_samples", "n_positive_train",
              "accuracy", "precision_score", "recall", "f1_score", "roc_auc"]:
        v = metrics.get(k)
        if v is None:
            continue
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")
    if metrics.get("warning"):
        print(f"  warning: {metrics['warning']}")

    print("\n" + "=" * 55)
    print("PREDICTIONS")
    print("=" * 55)
    if not predictions:
        print("  No predictions generated.")
        return

    probs = np.array([p["predicted_probability"] for p in predictions])
    buy_preds = [p for p in predictions if p["predicted_label"] == 1]

    print(f"  Total scored:  {len(predictions)}")
    print(f"  BUY signals:   {len(buy_preds)}")
    print(f"\n  Probability distribution:")
    print(f"    median  {np.median(probs):.4f}")
    print(f"    mean    {np.mean(probs):.4f}")
    print(f"    p75     {np.percentile(probs, 75):.4f}")
    print(f"    p90     {np.percentile(probs, 90):.4f}")
    print(f"    p95     {np.percentile(probs, 95):.4f}")
    print(f"    max     {np.max(probs):.4f}")

    if buy_preds:
        buy_preds_sorted = sorted(buy_preds, key=lambda p: p["predicted_probability"], reverse=True)
        print(f"\n  Top BUY signals:")
        print(f"    {'ticker':<10} {'prob':>7}  published_at")
        print(f"    {'-'*8:<10} {'-'*7:>7}  {'-'*12}")
        for p in buy_preds_sorted[:20]:
            pub = str(p.get("published_at", ""))[:10]
            print(f"    {p['ticker']:<10} {p['predicted_probability']:>7.4f}  {pub}")
    else:
        # Still useful: show top-10 highest-probability predictions even if all label=0
        top10 = sorted(predictions, key=lambda p: p["predicted_probability"], reverse=True)[:10]
        print(f"\n  No BUY signals (all below 0.5). Top-10 by probability:")
        print(f"    {'ticker':<10} {'prob':>7}  published_at")
        print(f"    {'-'*8:<10} {'-'*7:>7}  {'-'*12}")
        for p in top10:
            pub = str(p.get("published_at", ""))[:10]
            print(f"    {p['ticker']:<10} {p['predicted_probability']:>7.4f}  {pub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="How many days back to pull announcements (default: 14)")
    args = parser.parse_args()
    main(days_back=args.days)
