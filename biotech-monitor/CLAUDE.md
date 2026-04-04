# CLAUDE.md - Project Context for Claude Code

## Project Overview

This is a **Biopharma Stock Announcement Monitoring System** that tracks regulatory announcements, clinical trials, and press releases for US-listed biotech and pharmaceutical small-cap stocks (<$2B market cap).

**Read `prompt.md` for full project specification, phases, and requirements.**

---

## Quick Reference

### Tech Stack
- **Language**: Python 3.11+
- **Database**: MySQL 8.0
- **HTTP Client**: httpx with tenacity for retries
- **Scraping**: BeautifulSoup4, Playwright (for JS-heavy pages)
- **Dashboard**: Streamlit
- **Platform**: macOS (local development)

### Key APIs
| API | Purpose | Docs Location |
|-----|---------|---------------|
| EODHD | Stock data, fundamentals | `prompt.md` Appendix A |
| SEC EDGAR | 8-K filings, CIK lookup | `https://www.sec.gov/developer` |
| ClinicalTrials.gov | Trial updates | `https://clinicaltrials.gov/api/v2` |
| OpenFDA | Drug approvals | `https://open.fda.gov/apis/` |

---

## Project Structure

```
biopharma-monitor/
├── config/           # YAML configuration files
├── data/             # CSV outputs, quality reports
├── logs/             # Pipeline logs
├── sql/              # Database schema and migrations
├── src/
│   ├── clients/      # API clients (eodhd, edgar, etc.)
│   ├── scrapers/     # Web scrapers
│   ├── pipeline/     # Data orchestration
│   ├── notifications/# Slack/Discord webhooks
│   ├── models/       # Data models
│   ├── db/           # Database connection, queries
│   └── utils/        # Shared utilities
├── scripts/          # Runnable scripts
└── tests/            # Unit tests
```

---

## Coding Conventions

### Python Style
- Use **type hints** for all function signatures
- Use **dataclasses** or **Pydantic** for data models
- Use **async/await** with httpx for API calls
- Follow **PEP 8** naming conventions
- Maximum line length: **100 characters**

### File Naming
- Use **snake_case** for all Python files
- Use **snake_case** for all functions and variables
- Use **PascalCase** for classes

### Imports
```python
# Standard library
import os
from datetime import datetime
from typing import Optional

# Third-party
import httpx
from tenacity import retry, stop_after_attempt

# Local
from src.utils.logging import get_logger
from src.models.stock import Stock
```

### Logging
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Processing stock", ticker=ticker)
logger.error("API request failed", error=str(e), ticker=ticker)
```

### Error Handling
```python
# Use specific exceptions
class EODHDAPIError(Exception):
    """EODHD API returned an error."""
    pass

class RateLimitError(Exception):
    """API rate limit exceeded."""
    pass

# Always log errors with context
try:
    response = await client.get(url)
except httpx.HTTPError as e:
    logger.error("HTTP error", url=url, error=str(e))
    raise
```

### Configuration
- Store secrets in `.env` file (never commit!)
- Store settings in `config/*.yaml` files
- Use environment variables for sensitive data:
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")
```

---

## Database Conventions

### MySQL
- Use **utf8mb4** charset for all tables
- Use **InnoDB** engine
- Always add **indexes** for frequently queried columns
- Use **foreign keys** for referential integrity

### Naming
- Tables: **plural, snake_case** (e.g., `stocks`, `announcements`)
- Columns: **singular, snake_case** (e.g., `ticker`, `market_cap`)
- Indexes: `idx_{table}_{column}` (e.g., `idx_stocks_ticker`)
- Foreign keys: `fk_{table}_{referenced_table}`

### Timestamps
- Always include `created_at` and `updated_at`
- Use `TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- Use `ON UPDATE CURRENT_TIMESTAMP` for `updated_at`

---

## API Client Conventions

### Base Client Pattern
All API clients should inherit from a base client:

```python
class BaseAPIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rate_limiter = RateLimiter(requests_per_second=10)
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        await self.rate_limiter.acquire()
        url = f"{self.base_url}{endpoint}"
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
```

### Rate Limiting
- Always implement rate limiting per API requirements
- SEC EDGAR: 10 requests/second
- EODHD: Based on plan (check daily limit)
- ClinicalTrials.gov: Be respectful, ~3 req/sec

### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((httpx.HTTPError, RateLimitError))
)
async def fetch_with_retry(self, url: str) -> dict:
    ...
```

### Caching
- Cache API responses to disk/database when appropriate
- Fundamentals data changes infrequently → cache for 24 hours
- Price data → don't cache

---

## Common Commands

### Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup database
mysql -u root -p -e "CREATE DATABASE biopharma_monitor;"
mysql -u root -p biopharma_monitor < sql/schema.sql
```

### Running Scripts
```bash
# Phase 1: Build stock universe
python scripts/fetch_stock_list.py

# Test API connections
python scripts/test_connections.py

# Run daily pipeline
python scripts/run_pipeline.py

# Launch dashboard
streamlit run scripts/dashboard.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_eodhd.py

# Run with coverage
pytest --cov=src
```

---

## Phase-Specific Notes

### Phase 1: Stock Universe
- Primary data source: EODHD `/exchange-symbol-list/US` + `/fundamentals/{ticker}.US`
- Filter: Industry contains "biotechnology" or "pharmaceutical", Market cap < $2B
- Enrich missing CIK from SEC EDGAR
- Output: `data/stocks.csv` initially, then MySQL `stocks` table

### Phase 2: API Connections & Testing
**CRITICAL: This phase requires rigorous testing with 95%+ coverage**

**APIs to implement**:
| API | Base URL | Rate Limit |
|-----|----------|------------|
| SEC EDGAR | `https://data.sec.gov` | 10 req/sec |
| ClinicalTrials.gov | `https://clinicaltrials.gov/api/v2` | ~10 req/sec |
| OpenFDA | `https://api.fda.gov` | 40 req/min (no key) |

**Testing requirements**:
- Use `pytest` + `pytest-asyncio` + `respx` for mocking httpx
- Create realistic fixtures in `tests/conftest.py`
- Test all edge cases: rate limits, timeouts, malformed responses
- Integration tests optional (gated by `RUN_INTEGRATION_TESTS=1`)
- Target: 95% code coverage

**Key files to create**:
```
src/clients/base.py          # Base HTTP client with retries
src/clients/edgar.py         # SEC EDGAR (8-K filings)
src/clients/clinicaltrials.py # Trial status updates
src/clients/openfda.py       # Drug approvals
src/utils/rate_limiter.py    # Token bucket rate limiter
tests/conftest.py            # Shared fixtures
tests/test_edgar.py          # ~20 tests
tests/test_clinicaltrials.py # ~15 tests
tests/test_openfda.py        # ~15 tests
tests/test_rate_limiter.py   # ~5 tests
tests/test_integration.py    # Live API tests (optional)
```

**SEC EDGAR specifics**:
- MUST include User-Agent header with contact email
- CIK must be zero-padded to 10 digits
- Parse 8-K items from comma-separated string
- Monitor ALL 8-K items (comprehensive)

**ClinicalTrials.gov specifics**:
- API v2 uses different response format than legacy
- Monitor ALL status changes
- Match sponsor names to our company list (fuzzy matching)

**Run tests with**:
```bash
pytest --cov=src --cov-fail-under=95
```

### Phase 2.5: Corrections, Debugging & Visualization
**Complete before Phase 3**

**Correction 1: Remove EODHD Fundamentals**
- Use SEC EDGAR as ONLY source for company data (CIK, SIC code, name)
- Filter by SIC codes: 2833, 2834, 2835, 2836 (biopharma)
- EODHD only used for: ticker list + price history

**Correction 2: Debug OpenFDA (0 announcements)**
- Run `scripts/debug_openfda.py` first
- Common issues: 404 = no results (not error), uppercase sponsor names, YYYYMMDD dates
- Fix client to handle these edge cases

**Correction 3: Visualization Charts**
- Generate price charts with announcement markers
- Use Plotly for interactive HTML charts
- Color-code by announcement category
- Show announcement ID for cross-reference
- Output to `charts/` directory

### Phase 3: Text Extraction & File Storage
**Extract full text from all announcements and store in files + CSV index**

**Key Decisions**:
- ID generation: `sha256(url + published_date)[:16]`
- SEC EDGAR: Extract 8-K + ALL exhibits (HTML, PDF, TXT)
- PDF extraction: `pymupdf` (fitz)
- ClinicalTrials.gov: All text fields concatenated
- OpenFDA: Preserve as structured JSON (not plain text)
- Failure handling: Retry 3x → FAILED → keep raw → empty text file

**Folder structure**:
```
data/
├── raw/{source}/{date}/{ticker}/{id}.{ext}    # Original content
├── text/{source}/{date}/{ticker}/{id}.txt     # Extracted text
└── index/announcements.csv                     # Master index
```

**CSV columns**: id, ticker, source, event_type, published_at, fetched_at, title, url, external_id, raw_path, raw_paths_extra, text_path, raw_mime, raw_size_bytes, text_size_bytes, parse_status, parse_attempts, error, extra_json

**Key files**:
```
src/storage/paths.py              # Path generation
src/storage/csv_index.py          # CSV index management
src/extraction/edgar_extractor.py # HTML/PDF extraction
src/extraction/clinicaltrials_extractor.py
src/extraction/openfda_extractor.py
src/extraction/pipeline.py        # Main pipeline
scripts/run_extraction.py
```

**Tests**: ~50 tests across all extraction modules, 95%+ coverage

### Phase 3.5: Post-Announcement Return Calculation + Parquet Dataset
**Calculate stock returns 30/60/90 days after each announcement + ML-ready dataset**

**Key Decisions**:
- Price point: Adjusted close (handles splits/dividends)
- **Start price: NEXT trading day close (T+1)** - accounts for after-hours announcements
- End price: Target day close (or closest trading day)
- Days: Calendar days (not trading days)
- Format: Decimal with 2 places (e.g., `15.43` for 15.43%)
- Recent announcements: Leave as NULL (calculate later)
- Missing data: Fail entire ticker batch

**New CSV columns**: `return_30d`, `return_60d`, `return_90d`

**Parquet Dataset** (`data/index/announcements.parquet`):
- Mirror of CSV with all columns
- Includes `raw_text` column (full extracted text content)
- Auto-syncs when CSV is updated
- ML-ready: filter by text length, require returns

**Key files**:
```
src/returns/price_lookup.py     # T+1 trading day handling
src/returns/calculator.py       # Return calculation
src/returns/pipeline.py         # Batch processing
src/storage/parquet_sync.py     # Parquet mirror with raw_text
scripts/calculate_returns.py
scripts/update_parquet.py
```

**Processing strategy**: Batch by ticker (fetch price history once per ticker)

**Tests**: ~40 tests for price lookup, calculator, pipeline, parquet sync

### Phase 4: Database Storage (MySQL)
- Load data from CSV index into MySQL
- Unified `announcements` table schema
- Deduplicate by (source, source_id)
- Run daily via cron at 6 AM ET

### Phase 5: Web Scraping
- FDA CDER for drug approvals
- Company IR pages for press releases
- Use Playwright for JavaScript-heavy pages
- Respect robots.txt, add delays

### Phase 6: AI (Deferred)
- Schema already includes `ai_summary` and `sentiment` columns
- Will implement later with Claude API or Ollama

### Phase 7: Dashboard & Notifications
- Streamlit dashboard for browsing announcements
- Slack/Discord webhooks for alerts

---

## Important Files

| File | Purpose |
|------|---------|
| `prompt.md` | Full project specification |
| `CLAUDE.md` | This file - project context |
| `.env` | Environment variables (secrets) |
| `config/filters.yaml` | Stock filter criteria |
| `config/api_config.yaml` | API rate limits, endpoints |
| `sql/schema.sql` | Database schema |
| `data/stocks.csv` | Stock universe output |
| `data/index/announcements.csv` | Master announcement index (Phase 3+) |
| `data/index/announcements.parquet` | ML-ready dataset with raw_text (Phase 3.5) |

---

## Data Flow & Master Pipeline

### Current Data Flow
```
data/stocks.csv              ← fetch_stock_list.py (Phase 1)
        ↓
run_extraction.py --all      (Phase 3)
        ↓
data/index/announcements.csv   ← Single source of truth
data/raw/{source}/{date}/{ticker}/   ← Raw files
data/text/{source}/{date}/{ticker}/  ← Extracted text
        ↓
calculate_returns.py         (Phase 3.5)
        ↓
data/index/announcements.csv   ← + return_30d, return_60d, return_90d
data/index/announcements.parquet ← + raw_text column (ML-ready)
        ↓
generate_announcement_charts.py (Phase 2.5)
        ↓
charts/index.html            ← Visualization output
```

**Note on Returns**: Start price uses T+1 (next trading day after announcement) to account for after-hours announcements.

### Master Pipeline Script
```bash
# Run everything (stocks → extraction → returns → parquet → charts)
python scripts/run_pipeline.py

# Skip stock fetch (use existing stocks.csv)
python scripts/run_pipeline.py --no-stocks

# Just fetch data, no charts
python scripts/run_pipeline.py --no-charts

# Limit announcements per source (for testing)
python scripts/run_pipeline.py --limit 10
```

### Deprecated Scripts (DO NOT USE)
- `scripts/fetch_all_announcements.py` → Use `run_extraction.py` instead
- `scripts/test_fetch_announcements.py` → Use `run_extraction.py` instead

These scripts show deprecation warnings and point to the new workflow.

---

## Do's and Don'ts

### Do
- ✅ Read `prompt.md` Appendix A for EODHD API details
- ✅ Use async/await for all HTTP requests
- ✅ Add proper logging with context
- ✅ Write unit tests for new functionality
- ✅ Handle API errors gracefully (don't crash on one bad ticker)
- ✅ Use type hints everywhere
- ✅ Cache expensive API calls when appropriate

### Don't
- ❌ Commit `.env` or API keys
- ❌ Exceed API rate limits
- ❌ Use synchronous HTTP in async contexts
- ❌ Skip error handling
- ❌ Hard-code configuration values
- ❌ Ignore the existing project structure

---

## Environment Variables

Required in `.env`:
```bash
EODHD_API_KEY=your_key_here
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=biopharma
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=biopharma_monitor
SEC_CONTACT_EMAIL=your_email@example.com  # Required by SEC
```

Optional (Phase 6):
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## Getting Help

1. **API Documentation**: See `prompt.md` Appendix A for EODHD reference
2. **Project Phases**: See `prompt.md` for detailed phase specifications
3. **Filter Criteria**: See `config/filters.yaml`
4. **Database Schema**: See `sql/schema.sql`

---

## Current Phase

**Phase 2.5: Corrections, Debugging & Visualization**

### Goals:
1. **Remove EODHD Fundamentals** - Use SEC EDGAR as sole source for company data
2. **Debug OpenFDA** - Fix 0 announcements issue, run diagnostics
3. **Generate Charts** - Price charts with announcement markers for pattern analysis

### Key Commands:
```bash
# 1. Run OpenFDA diagnostics first
python scripts/debug_openfda.py

# 2. Run correction tests
pytest tests/test_stock_universe_corrections.py -v
pytest tests/test_openfda_debug.py -v

# 3. After fixes, re-run stock universe
python scripts/fetch_stock_list.py

# 4. Re-run announcement pipeline
python scripts/run_pipeline.py

# 5. Generate visualization charts
python scripts/generate_announcement_charts.py

# 6. View charts
open charts/index.html
```

### Files to Create/Modify:
```
# Corrections (2.5.1)
src/clients/eodhd.py          # REMOVE fundamentals methods
scripts/fetch_stock_list.py   # USE SEC EDGAR only
src/utils/filters.py          # NEW: SIC code filtering

# OpenFDA Debug (2.5.2)  
scripts/debug_openfda.py      # NEW: Diagnostic script
src/clients/openfda.py        # FIX: 404 handling, uppercase, dates

# Visualization (2.5.3)
src/visualization/            # NEW: Directory
  announcement_charts.py      # NEW: Chart generator
scripts/generate_announcement_charts.py  # NEW: Runner script
charts/                       # OUTPUT: HTML + PNG files
```

### Success Criteria:
- [ ] No EODHD `/fundamentals/` API calls
- [ ] All stocks have CIK from SEC EDGAR
- [ ] OpenFDA returns > 0 announcements
- [ ] Charts generated for all stocks with announcements
- [ ] Tests pass with 95%+ coverage

---

## Next Phase: Phase 3 (Text Extraction & File Storage)

After completing Phase 2.5, proceed to Phase 3:

### Goals:
1. **Extract text** from all announcements (HTML, PDF, JSON)
2. **Store raw files** in `data/raw/{source}/{date}/{ticker}/`
3. **Store extracted text** in `data/text/{source}/{date}/{ticker}/`
4. **Maintain CSV index** at `data/index/announcements.csv`

### Key Commands:
```bash
# Run extraction pipeline
python scripts/run_extraction.py

# Check statistics
python scripts/run_extraction.py --stats

# Retry failed extractions
python scripts/run_extraction.py --retry-failed

# Run tests
pytest tests/test_paths.py tests/test_csv_index.py tests/test_*_extraction.py -v
```

### Files to Create:
```
src/storage/paths.py              # Path generation
src/storage/csv_index.py          # CSV index management
src/extraction/edgar_extractor.py # HTML/PDF extraction
src/extraction/clinicaltrials_extractor.py
src/extraction/openfda_extractor.py
src/extraction/pipeline.py        # Main pipeline
scripts/run_extraction.py
```

### Success Criteria:
- [ ] Path generation is deterministic (same input = same output)
- [ ] No duplicate IDs in CSV
- [ ] HTML extraction removes scripts/styles
- [ ] PDF extraction works or returns empty on failure
- [ ] Failed records have raw file + empty text file
- [ ] 95%+ test coverage

---

## Then: Phase 3.5 (Post-Announcement Returns + Parquet Dataset)

After Phase 3, calculate stock returns and generate ML-ready Parquet:

### Goals:
1. **Add return columns** to `data/index/announcements.csv`
2. **Calculate 30/60/90 day returns** using **T+1** (next trading day) as start
3. **Generate Parquet dataset** with full text content for ML
4. **Handle edge cases** (weekends, holidays, recent announcements)

### Key Commands:
```bash
# Check current state
python scripts/calculate_returns.py --stats

# Calculate returns (auto-syncs Parquet)
python scripts/calculate_returns.py

# Force recalculate all
python scripts/calculate_returns.py --force

# Update Parquet manually (if needed)
python scripts/update_parquet.py

# Check Parquet ML-ready stats
python scripts/update_parquet.py --stats --ml-ready

# Run tests
pytest tests/test_price_lookup.py tests/test_return_calculator.py tests/test_return_pipeline.py tests/test_parquet_sync.py -v
```

### Files to Create:
```
src/returns/price_lookup.py     # T+1 trading day handling
src/returns/calculator.py       # Return calculation logic
src/returns/pipeline.py         # Batch processing
src/storage/parquet_sync.py     # Parquet mirror with raw_text
scripts/calculate_returns.py    # Runner script
scripts/update_parquet.py       # Parquet update script
```

### Success Criteria:
- [ ] **Start price uses T+1** (next trading day after announcement)
- [ ] Weekend/holiday prices handled correctly
- [ ] Recent announcements have NULL (not error)
- [ ] Returns rounded to 2 decimal places
- [ ] Batch processing (1 API call per ticker)
- [ ] Parquet auto-syncs with CSV
- [ ] Parquet includes `raw_text` column
- [ ] 95%+ test coverage
