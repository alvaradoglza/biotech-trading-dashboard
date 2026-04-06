-- Biotech Trading Dashboard — Supabase/Postgres Schema
-- Run this in the Supabase SQL editor to create all tables.
-- Enable UUID extension first.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── announcements ─────────────────────────────────────────────────────────────
-- Regulatory announcements from FDA and ClinicalTrials.gov
CREATE TABLE IF NOT EXISTS announcements (
    id              uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    source          varchar(50)  NOT NULL,          -- 'clinicaltrials' | 'openfda'
    ticker          varchar(20)  NOT NULL,
    company_name    varchar(255),
    event_type      varchar(100) NOT NULL,           -- e.g. TRIAL_UPDATE, FDA_APPROVAL
    title           text,
    announcement_url text,
    published_at    timestamptz  NOT NULL,
    fetched_at      timestamptz  DEFAULT now(),
    raw_text        text,                            -- full extracted text (for ML)
    external_id     varchar(255),                    -- source-specific ID (NCT ID, etc.)
    return_30d      float,                           -- % return 30d post-announcement (T+1)
    return_5d       float,                           -- % return 5d post-announcement (T+1)
    created_at      timestamptz  DEFAULT now(),
    UNIQUE (source, external_id)                     -- idempotent upserts
);

CREATE INDEX IF NOT EXISTS idx_announcements_ticker     ON announcements (ticker);
CREATE INDEX IF NOT EXISTS idx_announcements_published  ON announcements (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_announcements_source     ON announcements (source);

-- ── model_runs ────────────────────────────────────────────────────────────────
-- Daily ML model training results
CREATE TABLE IF NOT EXISTS model_runs (
    id               uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    run_date         date         NOT NULL,
    horizon          varchar(10)  NOT NULL DEFAULT '30d',
    mlflow_run_id    varchar(100),
    mlflow_experiment_url text,
    accuracy         float,
    precision_score  float,
    recall           float,
    specificity      float,
    f1_score         float,
    roc_auc          float,
    n_train_samples  int,
    n_test_samples   int,
    n_positive_train int,
    n_positive_test  int,
    model_version    varchar(50),
    created_at       timestamptz  DEFAULT now(),
    UNIQUE (run_date, horizon)
);

CREATE INDEX IF NOT EXISTS idx_model_runs_date ON model_runs (run_date DESC);

-- ── predictions ───────────────────────────────────────────────────────────────
-- ML model predictions for each announcement
CREATE TABLE IF NOT EXISTS predictions (
    id                   uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    announcement_id      uuid REFERENCES announcements(id) ON DELETE CASCADE,
    model_run_id         uuid REFERENCES model_runs(id),
    model_version        varchar(50),
    predicted_label      int          NOT NULL,   -- 0 or 1
    predicted_probability float       NOT NULL,   -- confidence score
    expected_return_30d  float,                   -- predicted return if positive
    created_at           timestamptz  DEFAULT now(),
    UNIQUE (announcement_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_announcement ON predictions (announcement_id);
CREATE INDEX IF NOT EXISTS idx_predictions_label        ON predictions (predicted_label);

-- ── signals ───────────────────────────────────────────────────────────────────
-- Actionable trading signals generated from predictions
CREATE TABLE IF NOT EXISTS signals (
    id              uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    prediction_id   uuid REFERENCES predictions(id) ON DELETE CASCADE,
    signal_date     date         NOT NULL,
    ticker          varchar(20)  NOT NULL,
    action          varchar(20)  NOT NULL,    -- BUY | SELL | HOLD
    reason          text,
    score           float,                   -- decision score from model
    created_at      timestamptz  DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_signals_ticker_date ON signals (signal_date, ticker, action);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals (ticker);
CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals (signal_date DESC);

-- ── trades ────────────────────────────────────────────────────────────────────
-- Executed (or simulated) trades
CREATE TABLE IF NOT EXISTS trades (
    id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_id   uuid REFERENCES signals(id),
    trade_date  date         NOT NULL,
    ticker      varchar(20)  NOT NULL,
    side        varchar(10)  NOT NULL,     -- BUY | SELL
    quantity    float        NOT NULL,
    price       float        NOT NULL,
    amount_usd  float        NOT NULL,
    status      varchar(20)  DEFAULT 'filled',  -- filled | pending | cancelled
    exit_reason varchar(50),               -- take_profit | stop_loss | horizon | manual
    created_at  timestamptz  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades (ticker);
CREATE INDEX IF NOT EXISTS idx_trades_date   ON trades (trade_date DESC);

-- ── positions ─────────────────────────────────────────────────────────────────
-- Current open positions (one row per ticker)
CREATE TABLE IF NOT EXISTS positions (
    ticker       varchar(20)  PRIMARY KEY,
    quantity     float        NOT NULL DEFAULT 0,
    avg_cost     float        NOT NULL DEFAULT 0,
    market_value float        DEFAULT 0,        -- updated by pipeline
    unrealized_pnl float      DEFAULT 0,        -- updated by pipeline
    entry_date   date,
    updated_at   timestamptz  DEFAULT now()
);

-- ── portfolio_snapshots ───────────────────────────────────────────────────────
-- Daily portfolio value snapshots for history chart
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id             uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    snapshot_date  date         NOT NULL UNIQUE,
    cash           float        NOT NULL DEFAULT 0,
    equity_value   float        NOT NULL DEFAULT 0,
    total_value    float        NOT NULL DEFAULT 0,
    n_positions    int          DEFAULT 0,
    created_at     timestamptz  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots (snapshot_date DESC);

-- ── portfolio_config ──────────────────────────────────────────────────────────
-- Single-row table for portfolio state (cash balance, etc.)
CREATE TABLE IF NOT EXISTS portfolio_config (
    id             int PRIMARY KEY DEFAULT 1,
    initial_capital float NOT NULL DEFAULT 1000000.0,
    cash           float NOT NULL DEFAULT 1000000.0,
    updated_at     timestamptz DEFAULT now(),
    CHECK (id = 1)   -- enforce single row
);

-- Insert initial portfolio config if not exists
INSERT INTO portfolio_config (id, initial_capital, cash)
VALUES (1, 1000000.0, 1000000.0)
ON CONFLICT (id) DO NOTHING;
