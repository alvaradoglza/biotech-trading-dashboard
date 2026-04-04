# Biotech Trading Dashboard

A live portfolio dashboard for a systematic biotech trading strategy based on FDA and ClinicalTrials.gov announcements.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  GitHub Actions (daily @ 7:07 AM Mexico City)       │
│  ┌────────────────────────────────────────────────┐  │
│  │  pipeline/run_daily.py                         │  │
│  │    ├── fetch_fda.py         (OpenFDA API)      │  │
│  │    ├── fetch_clinical_trials.py (CT.gov API)   │  │
│  │    ├── predict.py           (GBM model + ML)   │  │
│  │    ├── generate_trades.py   (portfolio logic)  │  │
│  │    └── supabase_writer.py   (write to DB)      │  │
│  └────────────────────────────────────────────────┘  │
│              │                                       │
│              ▼                                       │
│         Supabase Postgres                            │
│              │                                       │
│              ▼                                       │
│  Streamlit Dashboard (app.py)                        │
│    · Portfolio summary + equity curve                │
│    · Open positions + live prices (EODHD)            │
│    · Trades, announcements, predictions              │
│    · Model performance metrics                       │
└─────────────────────────────────────────────────────┘
```

## Quick Start (Local)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd biotech-trading-dashboard

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure secrets

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your actual values
```

Required variables:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key          # for pipeline writes
SUPABASE_ANON_KEY=your-anon-key             # for dashboard reads
EODHD_API_KEY=your-eodhd-api-key
SEC_CONTACT_EMAIL=your@email.com
```

Optional:
```
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
```

### 3. Set up Supabase

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run `sql/schema.sql`
3. Copy your project URL and keys from **Settings → API**

### 4. Run the dashboard

```bash
streamlit run app.py
```

### 5. Run the pipeline manually

```bash
# Full pipeline
python -m pipeline.run_daily

# Dry run (no writes to Supabase)
python -m pipeline.run_daily --dry-run

# Skip API fetching (use data already in Supabase)
python -m pipeline.run_daily --skip-fetch
```

## Seeding Historical Data

If you have the existing `backtesting-biotech/announcements2.parquet` file, the pipeline
will automatically use it as a fallback for model training on the first run.

To permanently seed historical data into Supabase:
```bash
python scripts/seed_supabase.py  # (create this script once Supabase is set up)
```

## Deployment

### Streamlit Community Cloud

1. Push your repo to GitHub (make sure `.gitignore` excludes `.env` and secrets)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and set `app.py` as the main file
4. In **Secrets**, add:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_ANON_KEY = "your-anon-key"
   EODHD_API_KEY = "your-eodhd-key"
   ```

### GitHub Actions (Daily Pipeline)

1. Go to your repo → **Settings → Secrets and variables → Actions**
2. Add repository secrets:
   - `SUPABASE_URL`
   - `SUPABASE_KEY` (service role key — NOT anon key)
   - `EODHD_API_KEY`
   - `SEC_CONTACT_EMAIL`
   - `MLFLOW_TRACKING_URI` (optional)
3. The workflow runs automatically at **7:07 AM America/Mexico_City** (13:07 UTC)
4. Trigger manually: **Actions → Daily Pipeline → Run workflow**

## ML Model

The model is a **GradientBoostingClassifier** (scikit-learn) that predicts whether
a regulatory announcement will lead to an outsized 30-day return (>85th percentile).

### Daily training strategy
- **Train set**: All labeled announcements older than 4 weeks
- **Test set**: Most recent 4 weeks of labeled announcements
- **Predict**: New announcements without return outcomes yet

### Features (14 structured + OHE)
- Clinical phase (1–4), patient count, endpoint keywords (positive/negative)
- Mechanism hits, disease hits, oncology/rare flags
- Source and event type (one-hot encoded)

### Metrics tracked daily
- Accuracy, Precision, Recall, Specificity, F1 Score, ROC AUC
- All saved to Supabase `model_runs` table
- Optionally logged to MLflow (set `MLFLOW_TRACKING_URI`)

## Project Structure

```
biotech-trading-dashboard/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt
├── CLAUDE.md                       # Architecture notes & pitfalls
├── LOGBOOK.md                      # Development progress log
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Dark theme + server config
├── sql/
│   └── schema.sql                  # Supabase/Postgres schema
├── dashboard/
│   ├── db.py                       # Supabase read queries (cached)
│   ├── prices.py                   # EODHD live prices (15-min cache)
│   ├── charts.py                   # Plotly chart builders
│   └── ui_helpers.py               # Formatting & display helpers
├── pipeline/
│   ├── run_daily.py                # Daily orchestrator
│   ├── fetch_fda.py                # OpenFDA API wrapper
│   ├── fetch_clinical_trials.py    # ClinicalTrials.gov wrapper
│   ├── predict.py                  # ML model training + prediction
│   ├── generate_trades.py          # Portfolio logic
│   └── supabase_writer.py          # DB write functions
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml      # GitHub Actions scheduler
├── biotech-monitor/                # Data pipeline (existing codebase)
└── backtesting-biotech/            # ML + backtesting (existing codebase)
```

## Environment Variables Reference

| Variable | Required By | Description |
|----------|------------|-------------|
| `SUPABASE_URL` | Pipeline + Dashboard | Supabase project URL |
| `SUPABASE_KEY` | Pipeline | Service role key (write access) |
| `SUPABASE_ANON_KEY` | Dashboard | Anon key (read-only access) |
| `EODHD_API_KEY` | Pipeline + Dashboard | Stock price API |
| `SEC_CONTACT_EMAIL` | Pipeline | Required by SEC EDGAR |
| `MLFLOW_TRACKING_URI` | Pipeline (optional) | Remote MLflow server |
| `INITIAL_CAPITAL` | Pipeline (optional) | Starting capital (default: 1000000) |
| `MAX_OPEN_POSITIONS` | Pipeline (optional) | Max concurrent positions (default: 20) |
| `MAX_WEIGHT` | Pipeline (optional) | Max position weight (default: 0.07) |
