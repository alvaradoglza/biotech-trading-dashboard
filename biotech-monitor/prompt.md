# Biopharma Stock Announcement Monitoring System

## Project Overview

Build an automated system that monitors regulatory announcements, clinical trial updates, and press releases for US-listed biopharma and biotech small-cap stocks. The system will aggregate data from multiple sources (SEC EDGAR, ClinicalTrials.gov, FDA, company IR pages), summarize announcements using AI, and present them in a searchable database for investment research.

### Goals
- **Primary**: Never miss a material announcement from a biopharma small-cap stock
- **Secondary**: Reduce research time by auto-summarizing and categorizing announcements
- **Tertiary**: Build a historical database of announcements for pattern analysis

### Target User
Individual investor/trader focused on biopharma small-cap stocks who wants to identify catalysts early.

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | Best ecosystem for data/scraping |
| Database | MySQL 8.0+ | Robust, widely supported, good performance |
| ORM | SQLAlchemy 2.0 | Database abstraction, easy migrations |
| Scheduling | cron or APScheduler | Daily runs, keep it simple |
| AI Summarization | Deferred to Phase 5 | Add later when core pipeline stable |
| HTTP Client | httpx + tenacity | Async + retry logic |
| Scraping | BeautifulSoup4 + Playwright | BS4 for simple, Playwright for JS-heavy |

**Monitoring Schedule**: Once daily (recommended: 6 AM ET, before market open)

---

## Phase 1: Stock Universe Creation

### Objective
Create a comprehensive list of all US-listed biopharma and biotech stocks that meet our criteria.

### Data Requirements

For each stock, capture:

| Field | Description | Required |
|-------|-------------|----------|
| `ticker` | Stock symbol (e.g., MRNA) | ✅ |
| `company_name` | Full legal name | ✅ |
| `exchange` | NASDAQ, NYSE, NYSE_AMER, OTC | ✅ |
| `market_cap` | Market capitalization in USD | ✅ |
| `market_cap_category` | micro/small/mid based on thresholds | ✅ |
| `sector` | e.g., "Healthcare" | ✅ |
| `industry` | e.g., "Biotechnology" | ✅ |
| `cik` | SEC Central Index Key | ✅ |
| `cusip` | 9-digit CUSIP identifier | ⚪ Nice to have |
| `ir_url` | Investor Relations page URL | ✅ |
| `sec_filings_url` | Direct link to SEC filings | ⚪ Auto-generated |
| `website` | Company main website | ⚪ Nice to have |
| `country` | HQ country (filter for US-based) | ⚪ Nice to have |
| `ipo_date` | Date of IPO | ⚪ Nice to have |
| `is_active` | Still trading flag | ✅ |
| `last_updated` | Timestamp of last data refresh | ✅ |

### Filter Criteria

```python
FILTERS = {
    "exchanges": [
        "NASDAQ",      # Main tech/biotech exchange
        "NYSE",        # New York Stock Exchange
        "NYSE_AMER",   # NYSE American (formerly AMEX) - many small caps
        "OTCQX",       # OTC Best Market - highest tier
        "OTCQB",       # OTC Venture Market - mid tier
        # Note: Excluding Pink Sheets (OTCPK) due to lower reporting standards
    ],
    "industries": [
        "Biotechnology",
        "Pharmaceuticals",
    ],
    "market_cap": {
        "min": 0,                # Include all micro-caps
        "max": 2_000_000_000,    # $2B ceiling (micro + small cap)
    },
    "exclude_adr": True,         # Exclude foreign ADRs (focus on US-listed)
    "only_us_hq": False,         # Include foreign companies listed in US
}
```

**Expected Universe Size**: ~400-600 stocks based on these criteria.

### Data Sources (in order of preference)

#### Primary: EODHD API ✅ (You have API key)
- **URL**: `https://eodhd.com/api/`
- **Coverage**: NASDAQ, NYSE, NYSE AMER, OTC markets
- **Endpoints we'll use**:
  | Endpoint | Purpose |
  |----------|---------|
  | `/exchange-symbol-list/{exchange}` | Get all tickers for an exchange |
  | `/fundamentals/{ticker}` | Company profile, market cap, industry |
  | `/bulk-fundamentals/{exchange}` | Bulk fetch for efficiency |
  | `/screener` | Filter by market cap, sector (if available on your plan) |

- **Key fields from EODHD**:
  - `General.Name` → company_name
  - `General.Exchange` → exchange
  - `General.Sector` → sector
  - `General.Industry` → industry
  - `General.WebURL` → website (derive IR URL from this)
  - `Highlights.MarketCapitalization` → market_cap
  - `General.CIK` → cik (if available, otherwise enrich from SEC)
  - `General.CUSIP` → cusip

- **Rate Limits**: Depends on your plan (check your tier)
  - Free: 20 requests/day
  - Basic+: 100,000 requests/day

#### Secondary: SEC EDGAR (Free)
- **Purpose**: Authoritative source for CIK numbers, backup for missing data
- **URL**: `https://www.sec.gov/cgi-bin/browse-edgar`
- **Use when**: EODHD missing CIK, or need to verify company identity

#### Tertiary: NASDAQ Screener (Free, manual backup)
- **URL**: `https://www.nasdaq.com/market-activity/stocks/screener`
- **Purpose**: Cross-reference, catch any stocks EODHD might miss
- **Limitation**: NASDAQ exchange only, requires manual download

### Implementation Steps

#### Step 1.1: Fetch Stock Lists from EODHD
```python
"""
Fetch all tickers from target exchanges via EODHD API.

Exchanges to query:
- US (NASDAQ + NYSE combined)
- NYSE AMER (or AMEX)
- OTCQX
- OTCQB

Example API call:
GET https://eodhd.com/api/exchange-symbol-list/US?api_token={API_KEY}&fmt=json
"""

EXCHANGES = {
    "US": "NASDAQ and NYSE combined",
    "NYSE AMER": "NYSE American (formerly AMEX)", 
    "OTCQX": "OTC Best Market",
    "OTCQB": "OTC Venture Market",
}

# Returns: list of {Code, Name, Country, Exchange, Currency, Type}
# Filter by Type == "Common Stock" to exclude ETFs, ADRs, etc.
```

#### Step 1.2: Fetch Fundamentals for Each Stock
```python
"""
For each ticker, fetch fundamentals to get market cap and industry.

Option A: Individual requests (slower, but works on all plans)
GET https://eodhd.com/api/fundamentals/{TICKER}.US?api_token={API_KEY}

Option B: Bulk endpoint (faster, may require higher tier)
GET https://eodhd.com/api/bulk-fundamentals/US?api_token={API_KEY}&fmt=json

Extract:
- General.Industry → filter for Biotechnology/Pharmaceuticals
- Highlights.MarketCapitalization → filter for <$2B
- General.CIK → store for SEC EDGAR lookups
"""
```

#### Step 1.3: Apply Filters
```python
def filter_stocks(stocks: list[dict]) -> list[dict]:
    """
    Apply all filter criteria.
    
    1. Industry filter: Keep only Biotechnology, Pharmaceuticals
       - EODHD industry names may vary, use fuzzy matching:
         - "Biotechnology"
         - "Pharmaceuticals" 
         - "Drug Manufacturers"
         - "Pharmaceutical"
         - Check actual values and adjust
    
    2. Market cap filter: Keep only <$2B
       - Handle None/missing market cap (keep for manual review)
    
    3. Stock type filter: Exclude ETFs, ADRs, Preferred shares
       - Filter by Type == "Common Stock"
    
    4. Active filter: Exclude delisted stocks
    """
    
    INDUSTRY_KEYWORDS = [
        "biotechnology", "biotech",
        "pharmaceutical", "pharma",
        "drug manufacturer",
    ]
    
    filtered = []
    for stock in stocks:
        industry = (stock.get("industry") or "").lower()
        market_cap = stock.get("market_cap") or 0
        
        # Industry check (fuzzy)
        industry_match = any(kw in industry for kw in INDUSTRY_KEYWORDS)
        
        # Market cap check
        cap_ok = market_cap < 2_000_000_000 or market_cap == 0  # Keep if missing
        
        if industry_match and cap_ok:
            filtered.append(stock)
    
    return filtered
```

#### Step 1.4: Enrich Missing Data from SEC EDGAR
```python
"""
For stocks missing CIK, look up in SEC EDGAR.

SEC EDGAR company search:
GET https://www.sec.gov/cgi-bin/browse-edgar?company={NAME}&CIK=&type=&owner=include&count=10&action=getcompany&output=atom

Or use the full company tickers JSON:
GET https://www.sec.gov/files/company_tickers.json

This gives CIK for all SEC-registered companies.
Match by ticker symbol.
"""
```

#### Step 1.5: Find Investor Relations URLs
```python
"""
Derive IR page URL from company website.

Common patterns:
- {website}/investors
- {website}/investor-relations  
- {website}/ir
- investors.{domain}
- ir.{domain}

Strategy:
1. Start with EODHD's General.WebURL
2. Try common IR paths, check if they return 200
3. If none work, flag for manual review
4. Store the working URL
"""

IR_PATH_PATTERNS = [
    "/investors",
    "/investor-relations",
    "/ir", 
    "/investors/overview",
    "/about/investors",
]

def find_ir_url(website: str) -> str | None:
    for path in IR_PATH_PATTERNS:
        url = f"{website.rstrip('/')}{path}"
        if check_url_exists(url):
            return url
    return None
```

#### Step 1.6: Store Data
```
Phase 1 MVP: Save to stocks.csv for quick iteration
Production: Insert into MySQL stocks table

CREATE TABLE stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    market_cap DECIMAL(15,2),
    market_cap_category ENUM('micro', 'small', 'mid') DEFAULT 'small',
    sector VARCHAR(100),
    industry VARCHAR(100),
    cik VARCHAR(20),
    cusip VARCHAR(9),
    ir_url VARCHAR(500),
    website VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    data_quality_score INT DEFAULT 0,  -- 0-100, based on completeness
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_exchange (exchange),
    INDEX idx_industry (industry),
    INDEX idx_market_cap (market_cap),
    INDEX idx_cik (cik)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### Step 1.7: Generate Quality Report
```python
"""
Create a data quality report flagging issues:

- missing_cik: Can't monitor SEC filings
- missing_ir_url: Can't scrape press releases  
- missing_market_cap: Can't properly filter
- suspicious_industry: Industry didn't match exactly

Output: data/stocks_quality_report.csv
"""

def calculate_quality_score(stock: dict) -> int:
    score = 100
    if not stock.get("cik"): score -= 30
    if not stock.get("ir_url"): score -= 25
    if not stock.get("market_cap"): score -= 20
    if not stock.get("industry"): score -= 15
    if not stock.get("website"): score -= 10
    return max(0, score)
```

### Deliverables for Phase 1
- [ ] `src/clients/eodhd.py` - EODHD API client with rate limiting
- [ ] `scripts/fetch_stock_list.py` - Main script to build stock universe
- [ ] `scripts/enrich_stocks.py` - Adds missing CIK, IR URL from secondary sources
- [ ] `scripts/find_ir_urls.py` - Discovers IR page URLs for each company
- [ ] `data/stocks.csv` - Output file with all filtered stocks
- [ ] `data/stocks_quality_report.csv` - Data quality issues
- [ ] `config/filters.yaml` - Configurable filter criteria
- [ ] `config/eodhd.yaml` - EODHD API configuration
- [ ] Unit tests for EODHD client and filters

### Success Criteria
- [ ] Captures 90%+ of known biopharma small-caps (verify against manual research)
- [ ] CIK populated for 95%+ of stocks
- [ ] IR URL populated for 80%+ of stocks
- [ ] Script runs in under 10 minutes (depends on EODHD rate limits)
- [ ] Data refreshable on demand
- [ ] Quality score calculated for each stock

---

## Phase 2: API Connections & Testing

### Objective
Build robust, well-tested API clients for SEC EDGAR, ClinicalTrials.gov, and OpenFDA with proper rate limiting, error handling, retry logic, and comprehensive test coverage.

### Testing Strategy Overview

| Test Type | Purpose | Tools |
|-----------|---------|-------|
| **Unit Tests** | Test individual methods in isolation | pytest, respx |
| **Mock Tests** | Test with realistic canned responses | respx, pytest fixtures |
| **Integration Tests** | Test against live APIs (optional) | pytest, pytest-asyncio |
| **Edge Case Tests** | Test error handling, rate limits | respx with error responses |

**Coverage Target**: 95%+ for all client code

---

### APIs to Connect

#### 2.0 EODHD API (Stock Data) — Already specified in Phase 1

See Appendix A for full EODHD documentation.

---

#### 2.1 SEC EDGAR API

**Base URL**: `https://data.sec.gov`

**Authentication**: None required, but **must include User-Agent header with contact email** (SEC requirement)

**Rate Limit**: 10 requests/second (enforced by SEC)

**Headers Required**:
```python
HEADERS = {
    "User-Agent": "CompanyName contact@email.com",  # SEC requires this
    "Accept-Encoding": "gzip, deflate",
}
```

##### Endpoint 2.1.1: Company Submissions

**Purpose**: Get all filings for a company by CIK

**URL**: `GET /submissions/CIK{cik_padded}.json`

**Parameters**:
- `cik_padded`: 10-digit CIK with leading zeros (e.g., `0001682852`)

**Example Request**:
```bash
curl -H "User-Agent: MyApp contact@example.com" \
  "https://data.sec.gov/submissions/CIK0001682852.json"
```

**Example Response** (abbreviated):
```json
{
  "cik": "1682852",
  "entityType": "operating",
  "sic": "2836",
  "sicDescription": "Biological Products, Except Diagnostic Substances",
  "name": "Moderna, Inc.",
  "tickers": ["MRNA"],
  "exchanges": ["NASDAQ"],
  "ein": "811365880",
  "website": "https://www.modernatx.com",
  "filings": {
    "recent": {
      "accessionNumber": ["0001682852-24-000008", "0001682852-24-000007"],
      "filingDate": ["2024-01-15", "2024-01-10"],
      "reportDate": ["2024-01-15", "2024-01-10"],
      "form": ["8-K", "4"],
      "primaryDocument": ["mrna-20240115.htm", "xslForm4X01.xml"],
      "items": ["2.02, 7.01", ""],
      "size": [125000, 5000]
    },
    "files": [
      {"name": "CIK0001682852-submissions-001.json", "filingCount": 1000}
    ]
  }
}
```

**Key Fields**:
| Field | Description |
|-------|-------------|
| `filings.recent.accessionNumber` | Unique filing identifier |
| `filings.recent.filingDate` | Date filed with SEC |
| `filings.recent.form` | Filing type (8-K, 10-K, etc.) |
| `filings.recent.items` | 8-K item numbers (comma-separated) |
| `filings.files` | Pagination - additional files for older filings |

##### Endpoint 2.1.2: Filing Document Access

**Purpose**: Get the actual filing document

**URL**: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}`

**Example**:
```
https://www.sec.gov/Archives/edgar/data/1682852/000168285224000008/mrna-20240115.htm
```

##### 8-K Item Numbers Reference

| Item | Description | Category |
|------|-------------|----------|
| 1.01 | Entry into Material Definitive Agreement | `partnership` |
| 1.02 | Termination of Material Definitive Agreement | `partnership` |
| 1.03 | Bankruptcy or Receivership | `other` |
| 2.01 | Completion of Acquisition/Disposition of Assets | `other` |
| 2.02 | Results of Operations and Financial Condition | `earnings` |
| 2.03 | Creation of Direct Financial Obligation | `financing` |
| 2.04 | Triggering Events Accelerating Obligation | `financing` |
| 2.05 | Exit/Disposal Activities Costs | `other` |
| 2.06 | Material Impairments | `other` |
| 3.01 | Notice of Delisting | `other` |
| 3.02 | Unregistered Sales of Equity Securities | `financing` |
| 3.03 | Material Modification to Shareholder Rights | `other` |
| 4.01 | Changes in Registrant's Certifying Accountant | `other` |
| 4.02 | Non-Reliance on Financial Statements | `other` |
| 5.01 | Changes in Control of Registrant | `executive` |
| 5.02 | Departure/Election of Directors/Officers | `executive` |
| 5.03 | Amendments to Articles/Bylaws | `other` |
| 5.04 | Temporary Suspension of Trading | `other` |
| 5.05 | Amendment to Code of Ethics | `other` |
| 5.06 | Change in Shell Company Status | `other` |
| 5.07 | Submission of Matters to Shareholder Vote | `other` |
| 5.08 | Shareholder Nominations | `other` |
| 6.01 | ABS Informational and Computational Material | `other` |
| 6.02 | Change of Servicer or Trustee | `other` |
| 6.03 | Change in Credit Enhancement | `other` |
| 6.04 | Failure to Make Required Distribution | `other` |
| 6.05 | Securities Act Updating Disclosure | `other` |
| 7.01 | Regulation FD Disclosure | `other` (often press releases) |
| 8.01 | Other Events | `other` |
| 9.01 | Financial Statements and Exhibits | `other` |

##### SEC EDGAR Client Specification

```python
# src/clients/edgar.py

from dataclasses import dataclass
from datetime import date
from typing import Optional
import httpx

@dataclass
class SECFiling:
    """Represents a single SEC filing."""
    accession_number: str
    filing_date: date
    form_type: str
    items: list[str]  # For 8-K filings
    primary_document: str
    cik: str
    company_name: str
    url: str
    size: int

class EDGARClient:
    """
    SEC EDGAR API client.
    
    Attributes:
        base_url: API base URL
        contact_email: Required contact email for User-Agent
        rate_limiter: Rate limiter instance (10 req/sec)
    
    Example:
        client = EDGARClient(contact_email="me@example.com")
        filings = await client.get_8k_filings("0001682852", days_back=30)
    """
    
    BASE_URL = "https://data.sec.gov"
    RATE_LIMIT = 10  # requests per second
    
    def __init__(self, contact_email: str):
        """
        Initialize EDGAR client.
        
        Args:
            contact_email: Email for User-Agent header (SEC requirement)
        """
        pass
    
    async def get_company_submissions(self, cik: str) -> dict:
        """
        Get all submissions for a company.
        
        Args:
            cik: Central Index Key (will be zero-padded to 10 digits)
            
        Returns:
            Full submissions JSON response
            
        Raises:
            EDGARAPIError: If API returns error
            RateLimitError: If rate limit exceeded
        """
        pass
    
    async def get_8k_filings(
        self, 
        cik: str, 
        days_back: int = 365,
        items: Optional[list[str]] = None
    ) -> list[SECFiling]:
        """
        Get 8-K filings for a company.
        
        Args:
            cik: Central Index Key
            days_back: How many days of history to fetch
            items: Filter by specific 8-K items (e.g., ["2.02", "7.01"])
            
        Returns:
            List of SECFiling objects
        """
        pass
    
    async def get_filing_document(
        self, 
        cik: str, 
        accession_number: str, 
        document_name: str
    ) -> str:
        """
        Get the content of a filing document.
        
        Args:
            cik: Central Index Key
            accession_number: Filing accession number
            document_name: Name of document to fetch
            
        Returns:
            Document content as string (HTML or XML)
        """
        pass
    
    @staticmethod
    def pad_cik(cik: str) -> str:
        """Pad CIK to 10 digits with leading zeros."""
        return cik.zfill(10)
    
    @staticmethod
    def parse_items(items_str: str) -> list[str]:
        """Parse comma-separated items string into list."""
        if not items_str:
            return []
        return [item.strip() for item in items_str.split(",")]
```

---

#### 2.2 ClinicalTrials.gov API (v2)

**Base URL**: `https://clinicaltrials.gov/api/v2`

**Authentication**: None required

**Rate Limit**: ~10 requests/second (be respectful)

##### Endpoint 2.2.1: Search Studies

**Purpose**: Search for clinical trials by sponsor, condition, etc.

**URL**: `GET /studies`

**Key Parameters**:
| Parameter | Description | Example |
|-----------|-------------|---------|
| `query.spons` | Sponsor/collaborator name | `Moderna` |
| `query.cond` | Condition being studied | `COVID-19` |
| `query.term` | General search term | `mRNA vaccine` |
| `filter.overallStatus` | Trial status filter | `RECRUITING` |
| `pageSize` | Results per page (max 1000) | `100` |
| `pageToken` | Pagination token | `NF0...` |
| `countTotal` | Include total count | `true` |
| `format` | Response format | `json` |
| `fields` | Specific fields to return | `NCTId,OfficialTitle` |

**Example Request**:
```bash
curl "https://clinicaltrials.gov/api/v2/studies?query.spons=Moderna&pageSize=10&format=json"
```

**Example Response** (abbreviated):
```json
{
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT04470427",
          "orgStudyIdInfo": {"id": "mRNA-1273-P301"},
          "organization": {"fullName": "ModernaTX, Inc."},
          "officialTitle": "A Phase 3 Study to Evaluate the Efficacy and Safety of mRNA-1273",
          "acronym": "COVE"
        },
        "statusModule": {
          "overallStatus": "COMPLETED",
          "startDateStruct": {"date": "2020-07-27"},
          "completionDateStruct": {"date": "2023-06-30"},
          "lastUpdateSubmitDate": "2024-01-15"
        },
        "sponsorCollaboratorsModule": {
          "leadSponsor": {"name": "ModernaTX, Inc.", "class": "INDUSTRY"},
          "collaborators": [{"name": "BARDA", "class": "FED"}]
        },
        "descriptionModule": {
          "briefSummary": "This is a phase 3 study to evaluate efficacy..."
        },
        "designModule": {
          "phases": ["PHASE3"],
          "studyType": "INTERVENTIONAL",
          "enrollmentInfo": {"count": 30000, "type": "ACTUAL"}
        },
        "armsInterventionsModule": {
          "interventions": [
            {"type": "BIOLOGICAL", "name": "mRNA-1273", "description": "mRNA vaccine"}
          ]
        },
        "conditionsModule": {
          "conditions": ["COVID-19"]
        }
      },
      "hasResults": true
    }
  ],
  "nextPageToken": "NF0xNjg...",
  "totalCount": 45
}
```

##### ClinicalTrials.gov Status Values

All possible `overallStatus` values to monitor:

| Status | Category | Description |
|--------|----------|-------------|
| `NOT_YET_RECRUITING` | `trial_start` | Approved but not yet enrolling |
| `RECRUITING` | `trial_update` | Actively enrolling |
| `ENROLLING_BY_INVITATION` | `trial_update` | Enrolling by invitation only |
| `ACTIVE_NOT_RECRUITING` | `trial_update` | Ongoing but not enrolling |
| `COMPLETED` | `trial_results` | Study completed |
| `SUSPENDED` | `trial_terminated` | Temporarily halted |
| `TERMINATED` | `trial_terminated` | Stopped early |
| `WITHDRAWN` | `trial_terminated` | Withdrawn before enrollment |
| `AVAILABLE` | `other` | Expanded access available |
| `NO_LONGER_AVAILABLE` | `other` | Expanded access no longer available |
| `TEMPORARILY_NOT_AVAILABLE` | `other` | Expanded access temporarily unavailable |
| `APPROVED_FOR_MARKETING` | `fda_approval` | Product approved |
| `WITHHELD` | `other` | Results withheld |
| `UNKNOWN` | `other` | Status unknown |

##### ClinicalTrials.gov Client Specification

```python
# src/clients/clinicaltrials.py

from dataclasses import dataclass
from datetime import date
from typing import Optional
from enum import Enum

class TrialStatus(Enum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"

@dataclass
class ClinicalTrial:
    """Represents a clinical trial."""
    nct_id: str
    title: str
    sponsor: str
    status: TrialStatus
    phase: list[str]
    conditions: list[str]
    interventions: list[str]
    start_date: Optional[date]
    completion_date: Optional[date]
    last_update_date: date
    enrollment: Optional[int]
    has_results: bool
    url: str

class ClinicalTrialsClient:
    """
    ClinicalTrials.gov API v2 client.
    
    Example:
        client = ClinicalTrialsClient()
        trials = await client.search_by_sponsor("Moderna", days_back=30)
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    RATE_LIMIT = 10  # requests per second (be respectful)
    MAX_PAGE_SIZE = 1000
    
    async def search_studies(
        self,
        sponsor: Optional[str] = None,
        condition: Optional[str] = None,
        term: Optional[str] = None,
        status: Optional[list[TrialStatus]] = None,
        page_size: int = 100,
        page_token: Optional[str] = None,
    ) -> tuple[list[ClinicalTrial], Optional[str], int]:
        """
        Search for clinical trials.
        
        Args:
            sponsor: Sponsor name to search
            condition: Condition to search
            term: General search term
            status: Filter by status(es)
            page_size: Results per page (max 1000)
            page_token: Token for pagination
            
        Returns:
            Tuple of (trials, next_page_token, total_count)
            
        Raises:
            ClinicalTrialsAPIError: If API returns error
        """
        pass
    
    async def search_by_sponsor(
        self,
        sponsor_name: str,
        days_back: int = 365,
        include_all_statuses: bool = True,
    ) -> list[ClinicalTrial]:
        """
        Get all trials for a sponsor, handling pagination.
        
        Args:
            sponsor_name: Company/organization name
            days_back: Filter by last update within N days
            include_all_statuses: Include all statuses or just active
            
        Returns:
            List of ClinicalTrial objects
        """
        pass
    
    async def get_study(self, nct_id: str) -> ClinicalTrial:
        """
        Get a specific study by NCT ID.
        
        Args:
            nct_id: NCT identifier (e.g., "NCT04470427")
            
        Returns:
            ClinicalTrial object
        """
        pass
    
    async def get_recently_updated(
        self,
        sponsor_names: list[str],
        days_back: int = 7,
    ) -> list[ClinicalTrial]:
        """
        Get trials with recent updates for multiple sponsors.
        
        Args:
            sponsor_names: List of sponsor names to search
            days_back: How many days to look back
            
        Returns:
            List of recently updated trials
        """
        pass
    
    @staticmethod
    def match_sponsor_to_company(
        sponsor_name: str, 
        company_names: list[str]
    ) -> Optional[str]:
        """
        Fuzzy match sponsor name to our company list.
        
        Handles variations like:
        - "ModernaTX, Inc." -> "Moderna"
        - "Pfizer Inc" -> "Pfizer"
        """
        pass
```

---

#### 2.3 OpenFDA API

**Base URL**: `https://api.fda.gov`

**Authentication**: Optional API key (higher rate limits with key)
- Without key: 40 requests/minute, 1000/day
- With key: 240 requests/minute, 120000/day

**Rate Limit Header**: Check `X-RateLimit-Remaining` in responses

##### Endpoint 2.3.1: Drugs@FDA (Drug Approvals)

**Purpose**: Get FDA drug approval information

**URL**: `GET /drug/drugsfda.json`

**Key Parameters**:
| Parameter | Description | Example |
|-----------|-------------|---------|
| `search` | Search query | `sponsor_name:"Moderna"` |
| `limit` | Results per request (max 1000) | `100` |
| `skip` | Pagination offset | `100` |

**Example Request**:
```bash
curl "https://api.fda.gov/drug/drugsfda.json?search=sponsor_name:Moderna&limit=10"
```

**Example Response** (abbreviated):
```json
{
  "meta": {
    "disclaimer": "...",
    "terms": "...",
    "license": "...",
    "last_updated": "2024-01-15",
    "results": {
      "skip": 0,
      "limit": 10,
      "total": 5
    }
  },
  "results": [
    {
      "application_number": "BLA761222",
      "sponsor_name": "MODERNATX, INC.",
      "products": [
        {
          "brand_name": "SPIKEVAX",
          "active_ingredients": [
            {"name": "ELASOMERAN", "strength": "0.1 MG/ML"}
          ],
          "dosage_form": "INJECTION, SUSPENSION",
          "route": "INTRAMUSCULAR",
          "marketing_status": "Prescription"
        }
      ],
      "submissions": [
        {
          "submission_type": "BLA",
          "submission_number": "1",
          "submission_status": "AP",
          "submission_status_date": "20220131",
          "submission_class_code": "Priority",
          "submission_class_code_description": "Priority Review"
        }
      ],
      "openfda": {
        "application_number": ["BLA761222"],
        "brand_name": ["SPIKEVAX"],
        "generic_name": ["COVID-19 VACCINE, MRNA"],
        "manufacturer_name": ["ModernaTX, Inc."],
        "product_ndc": ["80777-0273"],
        "product_type": ["VACCINE"],
        "route": ["INTRAMUSCULAR"],
        "substance_name": ["ELASOMERAN"],
        "rxcui": ["2468231"],
        "spl_id": ["..."],
        "spl_set_id": ["..."],
        "package_ndc": ["80777-0273-99"],
        "nui": ["..."],
        "pharm_class_cs": ["..."],
        "pharm_class_epc": ["..."],
        "pharm_class_pe": ["..."],
        "pharm_class_moa": ["..."]
      }
    }
  ]
}
```

##### Endpoint 2.3.2: Drug Labels

**Purpose**: Get drug labeling/prescribing information

**URL**: `GET /drug/label.json`

##### Endpoint 2.3.3: Drug Adverse Events

**Purpose**: Get adverse event reports

**URL**: `GET /drug/event.json`

##### OpenFDA Client Specification

```python
# src/clients/openfda.py

from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class DrugApproval:
    """Represents an FDA drug approval."""
    application_number: str
    sponsor_name: str
    brand_name: str
    generic_name: Optional[str]
    approval_date: Optional[date]
    submission_type: str  # NDA, BLA, ANDA
    submission_status: str
    dosage_form: str
    route: str
    active_ingredients: list[str]

class OpenFDAClient:
    """
    OpenFDA API client.
    
    Example:
        client = OpenFDAClient(api_key="optional_key")
        approvals = await client.get_approvals_by_sponsor("Moderna")
    """
    
    BASE_URL = "https://api.fda.gov"
    RATE_LIMIT_WITH_KEY = 240  # per minute
    RATE_LIMIT_NO_KEY = 40     # per minute
    MAX_RESULTS = 1000
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenFDA client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        pass
    
    async def get_drug_approvals(
        self,
        sponsor_name: Optional[str] = None,
        application_number: Optional[str] = None,
        brand_name: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> tuple[list[DrugApproval], int]:
        """
        Search Drugs@FDA for approvals.
        
        Args:
            sponsor_name: Filter by sponsor
            application_number: Filter by NDA/BLA number
            brand_name: Filter by brand name
            limit: Results per request (max 1000)
            skip: Pagination offset
            
        Returns:
            Tuple of (approvals, total_count)
        """
        pass
    
    async def get_approvals_by_sponsor(
        self,
        sponsor_name: str,
        days_back: int = 365,
    ) -> list[DrugApproval]:
        """
        Get all approvals for a sponsor.
        
        Args:
            sponsor_name: Company name
            days_back: Filter by approval date within N days
            
        Returns:
            List of DrugApproval objects
        """
        pass
    
    async def get_recent_approvals(
        self,
        days_back: int = 30,
    ) -> list[DrugApproval]:
        """
        Get all recent FDA approvals.
        
        Args:
            days_back: How many days to look back
            
        Returns:
            List of recent approvals
        """
        pass
    
    async def get_drug_label(
        self,
        application_number: str,
    ) -> dict:
        """
        Get drug labeling information.
        
        Args:
            application_number: NDA/BLA number
            
        Returns:
            Label data dictionary
        """
        pass
```

---

### Shared Infrastructure

#### Base HTTP Client

```python
# src/clients/base.py

from abc import ABC, abstractmethod
from typing import Optional, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class RateLimitError(APIError):
    """Rate limit exceeded."""
    pass

class NotFoundError(APIError):
    """Resource not found."""
    pass

class BaseAPIClient(ABC):
    """
    Base class for all API clients.
    
    Provides:
    - Async HTTP client with connection pooling
    - Rate limiting
    - Retry logic with exponential backoff
    - Logging
    - Response caching (optional)
    """
    
    def __init__(
        self,
        base_url: str,
        rate_limit: float,  # requests per second
        timeout: float = 30.0,
        headers: Optional[dict[str, str]] = None,
    ):
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit)
        self.timeout = timeout
        self.headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, RateLimitError)),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Make an HTTP request with rate limiting and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            **kwargs: Additional arguments for httpx
            
        Returns:
            httpx.Response object
            
        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: If API returns error
        """
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                **kwargs,
            )
            
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded", 429)
            
            if response.status_code == 404:
                raise NotFoundError(f"Not found: {url}", 404)
            
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            raise APIError(str(e), e.response.status_code)
    
    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """GET request returning JSON."""
        response = await self._request("GET", endpoint, params=params)
        return response.json()
    
    async def _get_text(self, endpoint: str, params: Optional[dict] = None) -> str:
        """GET request returning text."""
        response = await self._request("GET", endpoint, params=params)
        return response.text
```

#### Rate Limiter

```python
# src/utils/rate_limiter.py

import asyncio
import time
from collections import deque

class RateLimiter:
    """
    Token bucket rate limiter for async operations.
    
    Example:
        limiter = RateLimiter(rate=10)  # 10 requests/second
        await limiter.acquire()  # Blocks if rate exceeded
    """
    
    def __init__(self, rate: float, burst: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst: Maximum burst size (default: same as rate)
        """
        self.rate = rate
        self.burst = burst or int(rate)
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire a token, waiting if necessary.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
    
    @property
    def available_tokens(self) -> float:
        """Return current available tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.burst, self.tokens + elapsed * self.rate)
```

---

### Comprehensive Test Plan

#### Test Framework Setup

**Dependencies** (add to requirements.txt):
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
respx>=0.20.0
pytest-timeout>=2.2.0
freezegun>=1.2.0
factory-boy>=3.3.0
```

**pytest configuration** (pyproject.toml):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "-v",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-fail-under=95",
]
markers = [
    "integration: marks tests as integration tests (deselect with '-m not integration')",
    "slow: marks tests as slow (deselect with '-m not slow')",
]
timeout = 30
```

**conftest.py**:
```python
# tests/conftest.py

import pytest
import respx
import httpx
from typing import Generator, AsyncGenerator

# ============================================================
# FIXTURES: API Response Data
# ============================================================

@pytest.fixture
def edgar_submissions_response() -> dict:
    """Real-world SEC EDGAR submissions response structure."""
    return {
        "cik": "1682852",
        "entityType": "operating",
        "sic": "2836",
        "sicDescription": "Biological Products, Except Diagnostic Substances",
        "name": "Moderna, Inc.",
        "tickers": ["MRNA"],
        "exchanges": ["NASDAQ"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0001682852-24-000008",
                    "0001682852-24-000007",
                    "0001682852-24-000006",
                ],
                "filingDate": ["2024-01-15", "2024-01-10", "2024-01-05"],
                "reportDate": ["2024-01-15", "2024-01-10", "2024-01-05"],
                "form": ["8-K", "4", "8-K"],
                "primaryDocument": [
                    "mrna-20240115.htm",
                    "xslForm4X01.xml",
                    "mrna-20240105.htm",
                ],
                "items": ["2.02, 7.01", "", "8.01"],
                "size": [125000, 5000, 85000],
            },
            "files": [],
        },
    }

@pytest.fixture
def edgar_empty_response() -> dict:
    """Empty EDGAR response for company with no filings."""
    return {
        "cik": "9999999",
        "entityType": "operating",
        "name": "Test Company",
        "tickers": [],
        "exchanges": [],
        "filings": {
            "recent": {
                "accessionNumber": [],
                "filingDate": [],
                "reportDate": [],
                "form": [],
                "primaryDocument": [],
                "items": [],
                "size": [],
            },
            "files": [],
        },
    }

@pytest.fixture
def clinicaltrials_search_response() -> dict:
    """Real-world ClinicalTrials.gov search response structure."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT04470427",
                        "organization": {"fullName": "ModernaTX, Inc."},
                        "officialTitle": "A Phase 3 Study to Evaluate mRNA-1273",
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                        "startDateStruct": {"date": "2020-07-27"},
                        "completionDateStruct": {"date": "2023-06-30"},
                        "lastUpdateSubmitDate": "2024-01-15",
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "ModernaTX, Inc.", "class": "INDUSTRY"},
                    },
                    "designModule": {
                        "phases": ["PHASE3"],
                        "enrollmentInfo": {"count": 30000, "type": "ACTUAL"},
                    },
                    "conditionsModule": {
                        "conditions": ["COVID-19"],
                    },
                    "armsInterventionsModule": {
                        "interventions": [
                            {"type": "BIOLOGICAL", "name": "mRNA-1273"},
                        ],
                    },
                },
                "hasResults": True,
            }
        ],
        "nextPageToken": None,
        "totalCount": 1,
    }

@pytest.fixture
def openfda_drugsfda_response() -> dict:
    """Real-world OpenFDA Drugs@FDA response structure."""
    return {
        "meta": {
            "results": {"skip": 0, "limit": 10, "total": 1},
        },
        "results": [
            {
                "application_number": "BLA761222",
                "sponsor_name": "MODERNATX, INC.",
                "products": [
                    {
                        "brand_name": "SPIKEVAX",
                        "active_ingredients": [
                            {"name": "ELASOMERAN", "strength": "0.1 MG/ML"}
                        ],
                        "dosage_form": "INJECTION, SUSPENSION",
                        "route": "INTRAMUSCULAR",
                        "marketing_status": "Prescription",
                    }
                ],
                "submissions": [
                    {
                        "submission_type": "BLA",
                        "submission_number": "1",
                        "submission_status": "AP",
                        "submission_status_date": "20220131",
                    }
                ],
                "openfda": {
                    "brand_name": ["SPIKEVAX"],
                    "generic_name": ["COVID-19 VACCINE, MRNA"],
                    "manufacturer_name": ["ModernaTX, Inc."],
                },
            }
        ],
    }

# ============================================================
# FIXTURES: Mock API Setup
# ============================================================

@pytest.fixture
def mock_edgar(edgar_submissions_response) -> Generator:
    """Mock SEC EDGAR API."""
    with respx.mock(base_url="https://data.sec.gov") as respx_mock:
        respx_mock.get("/submissions/CIK0001682852.json").mock(
            return_value=httpx.Response(200, json=edgar_submissions_response)
        )
        yield respx_mock

@pytest.fixture
def mock_clinicaltrials(clinicaltrials_search_response) -> Generator:
    """Mock ClinicalTrials.gov API."""
    with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
        respx_mock.get("/studies").mock(
            return_value=httpx.Response(200, json=clinicaltrials_search_response)
        )
        yield respx_mock

@pytest.fixture
def mock_openfda(openfda_drugsfda_response) -> Generator:
    """Mock OpenFDA API."""
    with respx.mock(base_url="https://api.fda.gov") as respx_mock:
        respx_mock.get("/drug/drugsfda.json").mock(
            return_value=httpx.Response(200, json=openfda_drugsfda_response)
        )
        yield respx_mock

# ============================================================
# FIXTURES: Error Responses
# ============================================================

@pytest.fixture
def mock_rate_limit_error():
    """Mock 429 rate limit response."""
    return httpx.Response(
        429,
        json={"error": "Rate limit exceeded"},
        headers={"Retry-After": "60"},
    )

@pytest.fixture
def mock_server_error():
    """Mock 500 server error response."""
    return httpx.Response(500, json={"error": "Internal server error"})

@pytest.fixture
def mock_not_found_error():
    """Mock 404 not found response."""
    return httpx.Response(404, json={"error": "Not found"})
```

#### Unit Tests: SEC EDGAR Client

```python
# tests/test_edgar.py

import pytest
import respx
import httpx
from datetime import date
from src.clients.edgar import EDGARClient, SECFiling, EDGARAPIError

class TestEDGARClient:
    """Unit tests for SEC EDGAR client."""
    
    # ============================================================
    # INITIALIZATION TESTS
    # ============================================================
    
    def test_init_with_valid_email(self):
        """Client initializes with valid contact email."""
        client = EDGARClient(contact_email="test@example.com")
        assert client.contact_email == "test@example.com"
        assert "User-Agent" in client.headers
        assert "test@example.com" in client.headers["User-Agent"]
    
    def test_init_without_email_raises_error(self):
        """Client raises error without contact email."""
        with pytest.raises(ValueError, match="contact_email is required"):
            EDGARClient(contact_email="")
    
    def test_init_with_invalid_email_raises_error(self):
        """Client raises error with invalid email format."""
        with pytest.raises(ValueError, match="Invalid email format"):
            EDGARClient(contact_email="not-an-email")
    
    # ============================================================
    # CIK PADDING TESTS
    # ============================================================
    
    def test_pad_cik_short(self):
        """CIK is padded to 10 digits."""
        assert EDGARClient.pad_cik("123") == "0000000123"
    
    def test_pad_cik_already_padded(self):
        """Already padded CIK is unchanged."""
        assert EDGARClient.pad_cik("0001682852") == "0001682852"
    
    def test_pad_cik_no_leading_zeros(self):
        """CIK without leading zeros is padded."""
        assert EDGARClient.pad_cik("1682852") == "0001682852"
    
    # ============================================================
    # ITEMS PARSING TESTS
    # ============================================================
    
    def test_parse_items_multiple(self):
        """Multiple items are parsed correctly."""
        items = EDGARClient.parse_items("2.02, 7.01, 8.01")
        assert items == ["2.02", "7.01", "8.01"]
    
    def test_parse_items_single(self):
        """Single item is parsed correctly."""
        items = EDGARClient.parse_items("8.01")
        assert items == ["8.01"]
    
    def test_parse_items_empty(self):
        """Empty string returns empty list."""
        items = EDGARClient.parse_items("")
        assert items == []
    
    def test_parse_items_none(self):
        """None returns empty list."""
        items = EDGARClient.parse_items(None)
        assert items == []
    
    def test_parse_items_with_extra_spaces(self):
        """Items with extra spaces are trimmed."""
        items = EDGARClient.parse_items("  2.02 ,  7.01  ")
        assert items == ["2.02", "7.01"]
    
    # ============================================================
    # API REQUEST TESTS (MOCKED)
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_get_company_submissions_success(self, mock_edgar):
        """Successfully fetches company submissions."""
        async with EDGARClient(contact_email="test@example.com") as client:
            result = await client.get_company_submissions("1682852")
        
        assert result["cik"] == "1682852"
        assert result["name"] == "Moderna, Inc."
        assert len(result["filings"]["recent"]["accessionNumber"]) == 3
        assert mock_edgar["GET /submissions/CIK0001682852.json"].called
    
    @pytest.mark.asyncio
    async def test_get_company_submissions_pads_cik(self, mock_edgar):
        """CIK is automatically padded."""
        async with EDGARClient(contact_email="test@example.com") as client:
            await client.get_company_submissions("1682852")  # Not padded
        
        # Verify the padded CIK was used in the request
        assert mock_edgar["GET /submissions/CIK0001682852.json"].called
    
    @pytest.mark.asyncio
    async def test_get_company_submissions_not_found(self):
        """Returns NotFoundError for invalid CIK."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            respx_mock.get("/submissions/CIK9999999999.json").mock(
                return_value=httpx.Response(404, json={"error": "Not found"})
            )
            
            async with EDGARClient(contact_email="test@example.com") as client:
                with pytest.raises(NotFoundError):
                    await client.get_company_submissions("9999999999")
    
    @pytest.mark.asyncio
    async def test_get_company_submissions_rate_limited(self, mock_rate_limit_error):
        """Raises RateLimitError on 429 response."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            respx_mock.get("/submissions/CIK0001682852.json").mock(
                return_value=mock_rate_limit_error
            )
            
            async with EDGARClient(contact_email="test@example.com") as client:
                with pytest.raises(RateLimitError):
                    await client.get_company_submissions("1682852")
    
    @pytest.mark.asyncio
    async def test_get_company_submissions_server_error_retries(self):
        """Retries on 500 error then succeeds."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            # First two calls fail, third succeeds
            route = respx_mock.get("/submissions/CIK0001682852.json")
            route.side_effect = [
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200, json={"cik": "1682852", "name": "Test"}),
            ]
            
            async with EDGARClient(contact_email="test@example.com") as client:
                result = await client.get_company_submissions("1682852")
            
            assert result["cik"] == "1682852"
            assert route.call_count == 3
    
    # ============================================================
    # 8-K FILTERING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_get_8k_filings_returns_only_8k(self, mock_edgar):
        """Only returns 8-K filings."""
        async with EDGARClient(contact_email="test@example.com") as client:
            filings = await client.get_8k_filings("1682852")
        
        assert all(f.form_type == "8-K" for f in filings)
        assert len(filings) == 2  # 2 out of 3 are 8-K
    
    @pytest.mark.asyncio
    async def test_get_8k_filings_with_item_filter(self, mock_edgar):
        """Filters by specific 8-K items."""
        async with EDGARClient(contact_email="test@example.com") as client:
            filings = await client.get_8k_filings("1682852", items=["2.02"])
        
        assert len(filings) == 1
        assert "2.02" in filings[0].items
    
    @pytest.mark.asyncio
    async def test_get_8k_filings_with_days_filter(self, mock_edgar):
        """Filters by date range."""
        async with EDGARClient(contact_email="test@example.com") as client:
            filings = await client.get_8k_filings("1682852", days_back=10)
        
        # Only filings within last 10 days
        for filing in filings:
            assert (date.today() - filing.filing_date).days <= 10
    
    @pytest.mark.asyncio
    async def test_get_8k_filings_builds_correct_url(self, mock_edgar):
        """Filing URL is correctly constructed."""
        async with EDGARClient(contact_email="test@example.com") as client:
            filings = await client.get_8k_filings("1682852")
        
        expected_url = (
            "https://www.sec.gov/Archives/edgar/data/1682852/"
            "000168285224000008/mrna-20240115.htm"
        )
        assert filings[0].url == expected_url
    
    # ============================================================
    # EDGE CASES
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_get_8k_filings_empty_response(self, edgar_empty_response):
        """Handles company with no filings."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            respx_mock.get("/submissions/CIK0009999999.json").mock(
                return_value=httpx.Response(200, json=edgar_empty_response)
            )
            
            async with EDGARClient(contact_email="test@example.com") as client:
                filings = await client.get_8k_filings("9999999")
            
            assert filings == []
    
    @pytest.mark.asyncio
    async def test_handles_malformed_response(self):
        """Handles unexpected response format gracefully."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            respx_mock.get("/submissions/CIK0001682852.json").mock(
                return_value=httpx.Response(200, json={"unexpected": "format"})
            )
            
            async with EDGARClient(contact_email="test@example.com") as client:
                with pytest.raises(EDGARAPIError, match="Invalid response format"):
                    await client.get_company_submissions("1682852")
    
    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        """Handles request timeout."""
        with respx.mock(base_url="https://data.sec.gov") as respx_mock:
            respx_mock.get("/submissions/CIK0001682852.json").mock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            
            async with EDGARClient(contact_email="test@example.com") as client:
                with pytest.raises(httpx.TimeoutException):
                    await client.get_company_submissions("1682852")

    # ============================================================
    # USER-AGENT HEADER TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_request_includes_user_agent(self, mock_edgar):
        """All requests include required User-Agent header."""
        async with EDGARClient(contact_email="test@example.com") as client:
            await client.get_company_submissions("1682852")
        
        request = mock_edgar.calls[0].request
        assert "User-Agent" in request.headers
        assert "test@example.com" in request.headers["User-Agent"]
```

#### Unit Tests: ClinicalTrials.gov Client

```python
# tests/test_clinicaltrials.py

import pytest
import respx
import httpx
from datetime import date
from src.clients.clinicaltrials import (
    ClinicalTrialsClient, 
    ClinicalTrial,
    TrialStatus,
    ClinicalTrialsAPIError,
)

class TestClinicalTrialsClient:
    """Unit tests for ClinicalTrials.gov client."""
    
    # ============================================================
    # INITIALIZATION TESTS
    # ============================================================
    
    def test_init_defaults(self):
        """Client initializes with default settings."""
        client = ClinicalTrialsClient()
        assert client.BASE_URL == "https://clinicaltrials.gov/api/v2"
        assert client.MAX_PAGE_SIZE == 1000
    
    # ============================================================
    # SEARCH TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_search_studies_by_sponsor(self, mock_clinicaltrials):
        """Searches studies by sponsor name."""
        async with ClinicalTrialsClient() as client:
            trials, next_token, total = await client.search_studies(
                sponsor="Moderna"
            )
        
        assert len(trials) == 1
        assert trials[0].sponsor == "ModernaTX, Inc."
        assert trials[0].nct_id == "NCT04470427"
    
    @pytest.mark.asyncio
    async def test_search_studies_builds_correct_query(self, mock_clinicaltrials):
        """Query parameters are correctly built."""
        async with ClinicalTrialsClient() as client:
            await client.search_studies(
                sponsor="Moderna",
                condition="COVID-19",
                status=[TrialStatus.COMPLETED],
            )
        
        request = mock_clinicaltrials.calls[0].request
        assert "query.spons=Moderna" in str(request.url)
        assert "query.cond=COVID-19" in str(request.url)
        assert "filter.overallStatus=COMPLETED" in str(request.url)
    
    @pytest.mark.asyncio
    async def test_search_studies_pagination(self):
        """Handles pagination correctly."""
        page1 = {
            "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT001"}}}],
            "nextPageToken": "token123",
            "totalCount": 2,
        }
        page2 = {
            "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT002"}}}],
            "nextPageToken": None,
            "totalCount": 2,
        }
        
        with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
            route = respx_mock.get("/studies")
            route.side_effect = [
                httpx.Response(200, json=page1),
                httpx.Response(200, json=page2),
            ]
            
            async with ClinicalTrialsClient() as client:
                all_trials = await client.search_by_sponsor("Test", days_back=365)
            
            assert len(all_trials) == 2
            assert route.call_count == 2
    
    # ============================================================
    # TRIAL STATUS TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_parses_all_status_values(self):
        """All trial status values are correctly parsed."""
        for status in TrialStatus:
            response = {
                "studies": [{
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT001"},
                        "statusModule": {"overallStatus": status.value},
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test"}},
                    }
                }],
                "totalCount": 1,
            }
            
            with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
                respx_mock.get("/studies").mock(
                    return_value=httpx.Response(200, json=response)
                )
                
                async with ClinicalTrialsClient() as client:
                    trials, _, _ = await client.search_studies()
                
                assert trials[0].status == status
    
    # ============================================================
    # DATA PARSING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_parses_trial_fields_correctly(self, mock_clinicaltrials):
        """All trial fields are correctly extracted."""
        async with ClinicalTrialsClient() as client:
            trials, _, _ = await client.search_studies()
        
        trial = trials[0]
        assert trial.nct_id == "NCT04470427"
        assert trial.title == "A Phase 3 Study to Evaluate mRNA-1273"
        assert trial.sponsor == "ModernaTX, Inc."
        assert trial.status == TrialStatus.COMPLETED
        assert trial.phase == ["PHASE3"]
        assert trial.conditions == ["COVID-19"]
        assert trial.interventions == ["mRNA-1273"]
        assert trial.enrollment == 30000
        assert trial.has_results == True
        assert trial.url == "https://clinicaltrials.gov/study/NCT04470427"
    
    @pytest.mark.asyncio
    async def test_handles_missing_optional_fields(self):
        """Handles trials with missing optional fields."""
        response = {
            "studies": [{
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT001"},
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test"}},
                    # Missing: descriptionModule, designModule, etc.
                }
            }],
            "totalCount": 1,
        }
        
        with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
            respx_mock.get("/studies").mock(
                return_value=httpx.Response(200, json=response)
            )
            
            async with ClinicalTrialsClient() as client:
                trials, _, _ = await client.search_studies()
            
            assert trials[0].phase == []
            assert trials[0].conditions == []
            assert trials[0].enrollment is None
    
    # ============================================================
    # SPONSOR MATCHING TESTS
    # ============================================================
    
    def test_match_sponsor_exact(self):
        """Exact sponsor name match."""
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Moderna, Inc.",
            ["Pfizer", "Moderna", "BioNTech"]
        )
        assert result == "Moderna"
    
    def test_match_sponsor_case_insensitive(self):
        """Case-insensitive matching."""
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "MODERNATX, INC.",
            ["Moderna", "Pfizer"]
        )
        assert result == "Moderna"
    
    def test_match_sponsor_partial(self):
        """Partial name matching."""
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "ModernaTX, Inc.",
            ["Moderna Therapeutics", "Pfizer"]
        )
        assert result == "Moderna Therapeutics"
    
    def test_match_sponsor_no_match(self):
        """Returns None when no match found."""
        result = ClinicalTrialsClient.match_sponsor_to_company(
            "Unknown Corp",
            ["Moderna", "Pfizer"]
        )
        assert result is None
    
    # ============================================================
    # ERROR HANDLING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Handles API error responses."""
        with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
            respx_mock.get("/studies").mock(
                return_value=httpx.Response(500, json={"error": "Server error"})
            )
            
            async with ClinicalTrialsClient() as client:
                with pytest.raises(ClinicalTrialsAPIError):
                    await client.search_studies()
    
    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """Handles invalid JSON response."""
        with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as respx_mock:
            respx_mock.get("/studies").mock(
                return_value=httpx.Response(200, content=b"not json")
            )
            
            async with ClinicalTrialsClient() as client:
                with pytest.raises(ClinicalTrialsAPIError):
                    await client.search_studies()
```

#### Unit Tests: OpenFDA Client

```python
# tests/test_openfda.py

import pytest
import respx
import httpx
from datetime import date
from src.clients.openfda import (
    OpenFDAClient,
    DrugApproval,
    OpenFDAAPIError,
)

class TestOpenFDAClient:
    """Unit tests for OpenFDA client."""
    
    # ============================================================
    # INITIALIZATION TESTS
    # ============================================================
    
    def test_init_without_api_key(self):
        """Client initializes without API key."""
        client = OpenFDAClient()
        assert client.api_key is None
        assert client.rate_limit == 40  # per minute without key
    
    def test_init_with_api_key(self):
        """Client initializes with API key."""
        client = OpenFDAClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.rate_limit == 240  # per minute with key
    
    # ============================================================
    # DRUG APPROVALS TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_get_drug_approvals_success(self, mock_openfda):
        """Successfully fetches drug approvals."""
        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="Moderna"
            )
        
        assert len(approvals) == 1
        assert total == 1
        assert approvals[0].sponsor_name == "MODERNATX, INC."
        assert approvals[0].brand_name == "SPIKEVAX"
    
    @pytest.mark.asyncio
    async def test_get_drug_approvals_builds_correct_query(self, mock_openfda):
        """Search query is correctly built."""
        async with OpenFDAClient() as client:
            await client.get_drug_approvals(sponsor_name="Moderna")
        
        request = mock_openfda.calls[0].request
        assert 'search=sponsor_name:"Moderna"' in str(request.url)
    
    @pytest.mark.asyncio
    async def test_get_drug_approvals_with_limit_and_skip(self, mock_openfda):
        """Pagination parameters are correctly applied."""
        async with OpenFDAClient() as client:
            await client.get_drug_approvals(limit=50, skip=100)
        
        request = mock_openfda.calls[0].request
        assert "limit=50" in str(request.url)
        assert "skip=100" in str(request.url)
    
    # ============================================================
    # DATA PARSING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_parses_approval_fields_correctly(self, mock_openfda):
        """All approval fields are correctly extracted."""
        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()
        
        approval = approvals[0]
        assert approval.application_number == "BLA761222"
        assert approval.sponsor_name == "MODERNATX, INC."
        assert approval.brand_name == "SPIKEVAX"
        assert approval.generic_name == "COVID-19 VACCINE, MRNA"
        assert approval.submission_type == "BLA"
        assert approval.submission_status == "AP"
        assert approval.dosage_form == "INJECTION, SUSPENSION"
        assert approval.route == "INTRAMUSCULAR"
        assert "ELASOMERAN" in approval.active_ingredients
    
    @pytest.mark.asyncio
    async def test_parses_approval_date(self, mock_openfda):
        """Approval date is correctly parsed from submission_status_date."""
        async with OpenFDAClient() as client:
            approvals, _ = await client.get_drug_approvals()
        
        assert approvals[0].approval_date == date(2022, 1, 31)
    
    @pytest.mark.asyncio
    async def test_handles_missing_openfda_fields(self):
        """Handles missing openfda fields gracefully."""
        response = {
            "meta": {"results": {"total": 1}},
            "results": [{
                "application_number": "NDA123",
                "sponsor_name": "Test",
                "products": [{"brand_name": "TestDrug"}],
                "submissions": [{"submission_type": "NDA"}],
                # Missing openfda section
            }],
        }
        
        with respx.mock(base_url="https://api.fda.gov") as respx_mock:
            respx_mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(200, json=response)
            )
            
            async with OpenFDAClient() as client:
                approvals, _ = await client.get_drug_approvals()
            
            assert approvals[0].generic_name is None
    
    # ============================================================
    # ERROR HANDLING TESTS
    # ============================================================
    
    @pytest.mark.asyncio
    async def test_handles_no_results(self):
        """Handles empty results correctly."""
        response = {
            "meta": {"results": {"total": 0}},
            "results": [],
        }
        
        with respx.mock(base_url="https://api.fda.gov") as respx_mock:
            respx_mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(200, json=response)
            )
            
            async with OpenFDAClient() as client:
                approvals, total = await client.get_drug_approvals(
                    sponsor_name="NonexistentCorp"
                )
            
            assert approvals == []
            assert total == 0
    
    @pytest.mark.asyncio
    async def test_handles_api_key_in_request(self):
        """API key is included in request when provided."""
        with respx.mock(base_url="https://api.fda.gov") as respx_mock:
            respx_mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(200, json={"meta": {"results": {"total": 0}}, "results": []})
            )
            
            async with OpenFDAClient(api_key="my_api_key") as client:
                await client.get_drug_approvals()
            
            request = respx_mock.calls[0].request
            assert "api_key=my_api_key" in str(request.url)
```

#### Integration Tests (Live API)

```python
# tests/test_integration.py

import pytest
import os

# Skip all integration tests by default
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable."
    ),
]

class TestEDGARIntegration:
    """Integration tests against live SEC EDGAR API."""
    
    @pytest.mark.asyncio
    async def test_fetch_moderna_filings(self):
        """Fetch real filings for Moderna (MRNA)."""
        from src.clients.edgar import EDGARClient
        
        email = os.getenv("SEC_CONTACT_EMAIL", "test@example.com")
        async with EDGARClient(contact_email=email) as client:
            result = await client.get_company_submissions("1682852")
        
        assert result["name"] == "Moderna, Inc."
        assert "MRNA" in result["tickers"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting_respected(self):
        """Verify rate limiting works with rapid requests."""
        from src.clients.edgar import EDGARClient
        import time
        
        email = os.getenv("SEC_CONTACT_EMAIL", "test@example.com")
        async with EDGARClient(contact_email=email) as client:
            start = time.time()
            
            # Make 15 rapid requests (should take >1 second with 10/sec limit)
            for _ in range(15):
                await client.get_company_submissions("1682852")
            
            elapsed = time.time() - start
            assert elapsed >= 1.0, "Rate limiting not working"


class TestClinicalTrialsIntegration:
    """Integration tests against live ClinicalTrials.gov API."""
    
    @pytest.mark.asyncio
    async def test_search_moderna_trials(self):
        """Search for Moderna clinical trials."""
        from src.clients.clinicaltrials import ClinicalTrialsClient
        
        async with ClinicalTrialsClient() as client:
            trials, _, total = await client.search_studies(
                sponsor="Moderna",
                page_size=5,
            )
        
        assert total > 0
        assert len(trials) <= 5
        # At least one trial should mention Moderna
        assert any("moderna" in t.sponsor.lower() for t in trials)


class TestOpenFDAIntegration:
    """Integration tests against live OpenFDA API."""
    
    @pytest.mark.asyncio
    async def test_search_moderna_approvals(self):
        """Search for Moderna FDA approvals."""
        from src.clients.openfda import OpenFDAClient
        
        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="Moderna",
                limit=5,
            )
        
        # Moderna has at least one approval (Spikevax)
        assert total >= 1
```

#### Rate Limiter Tests

```python
# tests/test_rate_limiter.py

import pytest
import asyncio
import time
from src.utils.rate_limiter import RateLimiter

class TestRateLimiter:
    """Unit tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_allows_burst(self):
        """Allows burst of requests up to limit."""
        limiter = RateLimiter(rate=10, burst=5)
        
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        
        # Burst should be nearly instant
        assert elapsed < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limits_after_burst(self):
        """Enforces rate limit after burst exhausted."""
        limiter = RateLimiter(rate=10, burst=2)
        
        # Exhaust burst
        await limiter.acquire()
        await limiter.acquire()
        
        start = time.monotonic()
        await limiter.acquire()  # Should wait
        elapsed = time.monotonic() - start
        
        # Should wait ~0.1 seconds (1/10 rate)
        assert 0.08 <= elapsed <= 0.2
    
    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        limiter = RateLimiter(rate=10, burst=10)
        
        # Exhaust all tokens
        for _ in range(10):
            await limiter.acquire()
        
        assert limiter.available_tokens < 1
        
        # Wait for refill
        await asyncio.sleep(0.5)
        
        # Should have ~5 tokens back
        assert 4 <= limiter.available_tokens <= 6
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Thread-safe under concurrent access."""
        limiter = RateLimiter(rate=100, burst=10)
        
        async def acquire_many():
            for _ in range(5):
                await limiter.acquire()
        
        # Run 10 concurrent tasks each acquiring 5 tokens
        tasks = [acquire_many() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Should complete without errors
```

---

### Deliverables for Phase 2

**Source Files**:
- [ ] `src/clients/base.py` - Base HTTP client with rate limiting, retries
- [ ] `src/clients/edgar.py` - SEC EDGAR client
- [ ] `src/clients/clinicaltrials.py` - ClinicalTrials.gov client
- [ ] `src/clients/openfda.py` - OpenFDA client
- [ ] `src/utils/rate_limiter.py` - Rate limiting utility

**Test Files**:
- [ ] `tests/conftest.py` - Shared fixtures and mock responses
- [ ] `tests/test_edgar.py` - EDGAR client tests (unit + mock)
- [ ] `tests/test_clinicaltrials.py` - ClinicalTrials client tests
- [ ] `tests/test_openfda.py` - OpenFDA client tests
- [ ] `tests/test_rate_limiter.py` - Rate limiter tests
- [ ] `tests/test_integration.py` - Live API integration tests

**Configuration**:
- [ ] `config/api_config.yaml` - API rate limits, endpoints, timeouts
- [ ] `pyproject.toml` - pytest configuration with coverage settings

**Scripts**:
- [ ] `scripts/test_connections.py` - Manual API connection verification

### Success Criteria

| Metric | Target |
|--------|--------|
| Test coverage | ≥95% |
| Unit tests passing | 100% |
| Mock tests passing | 100% |
| Integration tests passing | 100% (when enabled) |
| All edge cases covered | Yes |
| Rate limiting verified | Yes |
| Error handling complete | Yes |

### Running Tests

```bash
# Run all unit and mock tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run integration tests (against live APIs)
RUN_INTEGRATION_TESTS=1 pytest -m integration

# Run only fast tests (exclude slow/integration)
pytest -m "not slow and not integration"

# Run specific test file
pytest tests/test_edgar.py -v

# Run specific test
pytest tests/test_edgar.py::TestEDGARClient::test_pad_cik_short -v
```

---

## Phase 2.5: Corrections, Debugging & Visualization

### Objective
Address corrections identified during Phase 1-2 implementation, debug OpenFDA connection issues, and add visualization tools for announcement pattern analysis.

---

### 2.5.1 Correction: Remove EODHD Fundamentals Dependency

**Problem**: Currently using EODHD `/fundamentals/{ticker}` as primary source for company data (market cap, industry, CIK), falling back to SEC EDGAR. This creates unnecessary API calls and potential data inconsistencies.

**Solution**: Use SEC EDGAR as the **sole primary source** for company enrichment data.

#### Changes Required

**File: `src/clients/eodhd.py`**
- KEEP: `get_exchange_symbols()` - still needed to get initial ticker list
- KEEP: `get_historical_prices()` - needed for price charts
- REMOVE: `get_fundamentals()` method
- REMOVE: `get_bulk_fundamentals()` method
- REMOVE: Any caching logic specific to fundamentals

**File: `scripts/fetch_stock_list.py`**
- REMOVE: Call to `eodhd_client.get_fundamentals()`
- CHANGE: Use SEC EDGAR as primary source for:
  - CIK number
  - SIC code (industry classification)
  - Company name (official)
  - Market cap (from SEC filings or skip - see note below)

**File: `config/filters.yaml`**
- UPDATE: Industry filtering to use SIC codes instead of EODHD industry strings

#### SIC Codes for Biopharma Filtering

Since SEC EDGAR uses SIC codes, update filtering logic:

| SIC Code | Description | Include |
|----------|-------------|---------|
| 2833 | Medicinal Chemicals and Botanical Products | ✅ |
| 2834 | Pharmaceutical Preparations | ✅ |
| 2835 | In Vitro & In Vivo Diagnostic Substances | ✅ |
| 2836 | Biological Products, Except Diagnostic Substances | ✅ |
| 3826 | Laboratory Analytical Instruments | ❌ |
| 3841 | Surgical & Medical Instruments | ❌ |
| 5122 | Drugs, Drug Proprietaries, and Druggists' Sundries | ❌ |
| 8731 | Commercial Physical & Biological Research | ⚠️ Maybe |

```python
# src/utils/filters.py

BIOPHARMA_SIC_CODES = {
    "2833",  # Medicinal Chemicals
    "2834",  # Pharmaceutical Preparations  
    "2835",  # Diagnostic Substances
    "2836",  # Biological Products
}

def is_biopharma_sic(sic_code: str) -> bool:
    """Check if SIC code is biopharma-related."""
    return sic_code in BIOPHARMA_SIC_CODES
```

#### Market Cap Handling

**Option A (Recommended)**: Skip market cap filtering initially
- SEC EDGAR doesn't directly provide market cap
- Would need to calculate from shares outstanding × price
- Can filter later in Phase 3 pipeline

**Option B**: Calculate from SEC data
- Get shares outstanding from 10-K/10-Q filings
- Get current price from EODHD EOD endpoint
- Calculate: `market_cap = shares_outstanding * current_price`

For now, **use Option A** - collect all biopharma stocks, filter by market cap later using EODHD price data.

#### Updated Stock Universe Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPDATED PHASE 1 FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EODHD: Get all US tickers                                   │
│     GET /exchange-symbol-list/US                                │
│     └── Returns: ticker, name, exchange, type                   │
│                                                                 │
│  2. Filter: Common Stock only                                   │
│     └── Exclude: ETFs, ADRs, Preferred, Warrants               │
│                                                                 │
│  3. SEC EDGAR: Lookup each ticker                               │
│     GET https://www.sec.gov/cgi-bin/browse-edgar                │
│         ?action=getcompany&CIK={ticker}&type=&output=atom       │
│     └── Returns: CIK, company name                              │
│                                                                 │
│     GET /submissions/CIK{cik_padded}.json                       │
│     └── Returns: SIC code, SIC description, tickers, exchanges  │
│                                                                 │
│  4. Filter: Biopharma SIC codes only                            │
│     └── Keep: 2833, 2834, 2835, 2836                           │
│                                                                 │
│  5. EODHD: Get current price (for market cap calc)              │
│     GET /eod/{ticker}.US?api_token={key}&fmt=json&filter=last   │
│     └── Returns: last close price                               │
│                                                                 │
│  6. Optional: Calculate market cap, filter <$2B                 │
│                                                                 │
│  7. Save to stocks table                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Test Cases for Correction 2.5.1

```python
# tests/test_stock_universe_corrections.py

import pytest

class TestSECEdgarPrimarySource:
    """Tests for SEC EDGAR as primary data source."""
    
    @pytest.mark.asyncio
    async def test_no_eodhd_fundamentals_calls(self, mock_eodhd, mock_edgar):
        """Verify EODHD fundamentals endpoint is never called."""
        from scripts.fetch_stock_list import build_stock_universe
        
        await build_stock_universe()
        
        # EODHD should only be called for exchange symbols, not fundamentals
        eodhd_calls = [c for c in mock_eodhd.calls]
        for call in eodhd_calls:
            assert "/fundamentals/" not in str(call.request.url)
    
    @pytest.mark.asyncio
    async def test_sec_edgar_provides_sic_code(self, mock_edgar):
        """SEC EDGAR response includes SIC code."""
        from src.clients.edgar import EDGARClient
        
        async with EDGARClient(contact_email="test@example.com") as client:
            data = await client.get_company_submissions("1682852")
        
        assert "sic" in data
        assert data["sic"] == "2836"
    
    def test_biopharma_sic_filter(self):
        """SIC code filter correctly identifies biopharma."""
        from src.utils.filters import is_biopharma_sic
        
        assert is_biopharma_sic("2836") == True   # Biological Products
        assert is_biopharma_sic("2834") == True   # Pharma Preparations
        assert is_biopharma_sic("3841") == False  # Medical Instruments
        assert is_biopharma_sic("7372") == False  # Software
    
    @pytest.mark.asyncio
    async def test_stock_has_cik_from_edgar(self):
        """All stocks have CIK populated from SEC EDGAR."""
        from src.db.queries import get_all_stocks
        
        stocks = await get_all_stocks()
        for stock in stocks:
            assert stock.cik is not None
            assert len(stock.cik) <= 10
```

---

### 2.5.2 Debug: OpenFDA Connection Issues

**Problem**: Getting 0 announcements from OpenFDA while EDGAR and ClinicalTrials return data.

**Possible Causes**:
1. Search query not matching sponsor names correctly
2. Rate limiting without proper error handling
3. Empty results being silently swallowed
4. Date filtering too restrictive
5. API response structure changed

#### Diagnostic Script

Create `scripts/debug_openfda.py`:

```python
#!/usr/bin/env python3
"""
Debug script for OpenFDA connection issues.
Run: python scripts/debug_openfda.py
"""

import asyncio
import httpx
import json
from datetime import datetime, timedelta

# Test companies known to have FDA approvals
TEST_SPONSORS = [
    "Moderna",
    "MODERNATX",
    "Pfizer",
    "PFIZER",
    "Eli Lilly",
    "ELI LILLY",
    "AbbVie",
    "ABBVIE",
]

BASE_URL = "https://api.fda.gov"

async def test_basic_connection():
    """Test 1: Basic API connectivity."""
    print("\n" + "="*60)
    print("TEST 1: Basic API Connection")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        try:
            # Simple query - get any 1 result
            response = await client.get(
                f"{BASE_URL}/drug/drugsfda.json",
                params={"limit": 1},
                timeout=30.0,
            )
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connection OK")
                print(f"   Total results in database: {data['meta']['results']['total']}")
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False

async def test_sponsor_search():
    """Test 2: Search by sponsor name."""
    print("\n" + "="*60)
    print("TEST 2: Sponsor Name Search")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        for sponsor in TEST_SPONSORS:
            try:
                # Try different search formats
                queries = [
                    f'sponsor_name:"{sponsor}"',
                    f'sponsor_name:{sponsor}',
                    f'openfda.manufacturer_name:"{sponsor}"',
                ]
                
                for query in queries:
                    response = await client.get(
                        f"{BASE_URL}/drug/drugsfda.json",
                        params={"search": query, "limit": 5},
                        timeout=30.0,
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        count = data["meta"]["results"]["total"]
                        if count > 0:
                            print(f"✅ '{sponsor}' with query '{query}': {count} results")
                            # Show first result
                            if data.get("results"):
                                first = data["results"][0]
                                print(f"   First result: {first.get('sponsor_name', 'N/A')}")
                        else:
                            print(f"⚠️  '{sponsor}' with query '{query}': 0 results")
                    elif response.status_code == 404:
                        print(f"⚠️  '{sponsor}' with query '{query}': No matches (404)")
                    else:
                        print(f"❌ '{sponsor}': HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"❌ '{sponsor}': Exception - {e}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

async def test_recent_approvals():
    """Test 3: Get recent approvals (any sponsor)."""
    print("\n" + "="*60)
    print("TEST 3: Recent Approvals (Last 365 Days)")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # OpenFDA uses YYYYMMDD format
        date_range = f"[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]"
        
        try:
            response = await client.get(
                f"{BASE_URL}/drug/drugsfda.json",
                params={
                    "search": f"submissions.submission_status_date:{date_range}",
                    "limit": 10,
                },
                timeout=30.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                total = data["meta"]["results"]["total"]
                print(f"✅ Found {total} approvals in last 365 days")
                
                if data.get("results"):
                    print("\nSample recent approvals:")
                    for i, result in enumerate(data["results"][:5]):
                        sponsor = result.get("sponsor_name", "Unknown")
                        products = result.get("products", [])
                        brand = products[0].get("brand_name", "N/A") if products else "N/A"
                        print(f"   {i+1}. {brand} by {sponsor}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

async def test_response_structure():
    """Test 4: Verify response structure matches our parser."""
    print("\n" + "="*60)
    print("TEST 4: Response Structure Validation")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/drug/drugsfda.json",
            params={"search": 'sponsor_name:"Moderna"', "limit": 1},
            timeout=30.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check expected structure
            checks = [
                ("meta" in data, "meta section exists"),
                ("results" in data, "results section exists"),
                ("meta.results.total" if "meta" in data and "results" in data.get("meta", {}) else False, 
                 "meta.results.total exists"),
            ]
            
            for check, desc in checks:
                status = "✅" if check else "❌"
                print(f"{status} {desc}")
            
            if data.get("results"):
                result = data["results"][0]
                fields = [
                    "application_number",
                    "sponsor_name", 
                    "products",
                    "submissions",
                    "openfda",
                ]
                print("\nField presence in first result:")
                for field in fields:
                    status = "✅" if field in result else "❌"
                    print(f"  {status} {field}")
                    
                # Print actual structure for debugging
                print("\nActual response structure:")
                print(json.dumps(result, indent=2, default=str)[:2000])
        else:
            print(f"❌ Could not fetch data: {response.status_code}")

async def test_our_client():
    """Test 5: Test our actual OpenFDA client."""
    print("\n" + "="*60)
    print("TEST 5: Our OpenFDAClient Implementation")
    print("="*60)
    
    try:
        from src.clients.openfda import OpenFDAClient
        
        async with OpenFDAClient() as client:
            # Test 1: Basic search
            approvals, total = await client.get_drug_approvals(
                sponsor_name="Moderna",
                limit=5,
            )
            print(f"✅ get_drug_approvals('Moderna'): {total} total, {len(approvals)} returned")
            
            for approval in approvals:
                print(f"   - {approval.brand_name} ({approval.application_number})")
            
            # Test 2: Recent approvals
            recent = await client.get_recent_approvals(days_back=365)
            print(f"✅ get_recent_approvals(365 days): {len(recent)} results")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Client error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all diagnostic tests."""
    print("OpenFDA Connection Diagnostics")
    print("==============================")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    await test_basic_connection()
    await test_sponsor_search()
    await test_recent_approvals()
    await test_response_structure()
    await test_our_client()
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Common OpenFDA Issues and Fixes

**Issue 1: Sponsor name mismatch**
```python
# BAD: Exact match fails because FDA uses uppercase
search = 'sponsor_name:"Moderna"'

# GOOD: Use uppercase or wildcard
search = 'sponsor_name:"MODERNA*"'
# OR search the openfda.manufacturer_name field
search = 'openfda.manufacturer_name:"moderna"'
```

**Issue 2: 404 on no results**
```python
# OpenFDA returns 404 when no results found (not empty array!)
async def get_drug_approvals(self, sponsor_name: str) -> tuple[list, int]:
    try:
        response = await self._get("/drug/drugsfda.json", params)
        return self._parse_response(response), response["meta"]["results"]["total"]
    except NotFoundError:
        # 404 means no results, not an error
        return [], 0
```

**Issue 3: Rate limiting**
```python
# Check rate limit headers
if "X-RateLimit-Remaining" in response.headers:
    remaining = int(response.headers["X-RateLimit-Remaining"])
    if remaining < 5:
        await asyncio.sleep(60)  # Wait for reset
```

**Issue 4: Submission date format**
```python
# OpenFDA uses YYYYMMDD format, not ISO
date_str = "20240115"  # NOT "2024-01-15"

# Date range query
search = f"submissions.submission_status_date:[20230101+TO+20240115]"
```

#### Test Cases for OpenFDA Debugging

```python
# tests/test_openfda_debug.py

import pytest
import respx
import httpx

class TestOpenFDADebugging:
    """Debug tests for OpenFDA issues."""
    
    @pytest.mark.asyncio
    async def test_handles_404_as_empty_results(self):
        """404 response should return empty list, not raise error."""
        with respx.mock(base_url="https://api.fda.gov") as mock:
            mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
            )
            
            from src.clients.openfda import OpenFDAClient
            async with OpenFDAClient() as client:
                approvals, total = await client.get_drug_approvals(
                    sponsor_name="NonexistentCorp"
                )
            
            assert approvals == []
            assert total == 0
    
    @pytest.mark.asyncio
    async def test_sponsor_name_uppercase_matching(self):
        """Sponsor names should match regardless of case."""
        response_data = {
            "meta": {"results": {"total": 1}},
            "results": [{
                "sponsor_name": "MODERNATX, INC.",
                "application_number": "BLA761222",
                "products": [{"brand_name": "SPIKEVAX"}],
                "submissions": [],
            }]
        }
        
        with respx.mock(base_url="https://api.fda.gov") as mock:
            mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(200, json=response_data)
            )
            
            from src.clients.openfda import OpenFDAClient
            async with OpenFDAClient() as client:
                # Should find results with lowercase search
                approvals, total = await client.get_drug_approvals(
                    sponsor_name="moderna"  # lowercase
                )
            
            assert total == 1
            assert approvals[0].sponsor_name == "MODERNATX, INC."
    
    @pytest.mark.asyncio
    async def test_date_format_in_query(self):
        """Date queries should use YYYYMMDD format."""
        with respx.mock(base_url="https://api.fda.gov") as mock:
            route = mock.get("/drug/drugsfda.json").mock(
                return_value=httpx.Response(200, json={"meta": {"results": {"total": 0}}, "results": []})
            )
            
            from src.clients.openfda import OpenFDAClient
            async with OpenFDAClient() as client:
                await client.get_recent_approvals(days_back=30)
            
            # Check the query used correct date format
            request_url = str(route.calls[0].request.url)
            # Should contain YYYYMMDD format, not YYYY-MM-DD
            assert "20" in request_url  # Year prefix
            assert "-" not in request_url or "TO" in request_url
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_openfda_moderna(self):
        """Live test: Moderna should have FDA approvals."""
        import os
        if os.getenv("RUN_INTEGRATION_TESTS") != "1":
            pytest.skip("Integration tests disabled")
        
        from src.clients.openfda import OpenFDAClient
        async with OpenFDAClient() as client:
            approvals, total = await client.get_drug_approvals(
                sponsor_name="MODERNA"
            )
        
        # Moderna has at least Spikevax approved
        assert total >= 1, "Moderna should have at least 1 FDA approval"
        assert any("SPIKEVAX" in a.brand_name.upper() for a in approvals)
```

---

### 2.5.3 New Feature: Announcement Visualization Charts

**Purpose**: Generate price charts with announcement markers for pattern identification.

#### Visualization Requirements

1. **One chart per stock** - saved as PNG/HTML
2. **Full price history** from EODHD
3. **Announcement markers** on the chart:
   - Vertical lines or arrows at announcement dates
   - Color-coded by category (earnings=blue, trial=green, FDA=red, etc.)
   - Annotation with announcement ID for cross-reference
4. **Output**: `charts/` directory with `{ticker}_announcements.html` files
5. **Interactive**: Use Plotly for zoom/hover capabilities

#### Implementation

```python
# src/visualization/announcement_charts.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Optional

from src.clients.eodhd import EODHDClient
from src.db.queries import get_stock_announcements, get_all_stocks
from src.models.announcement import Announcement, AnnouncementCategory

# Color mapping for announcement categories
CATEGORY_COLORS = {
    AnnouncementCategory.EARNINGS: "#1f77b4",       # Blue
    AnnouncementCategory.TRIAL_START: "#2ca02c",    # Green
    AnnouncementCategory.TRIAL_UPDATE: "#98df8a",   # Light green
    AnnouncementCategory.TRIAL_RESULTS: "#006400",  # Dark green
    AnnouncementCategory.TRIAL_TERMINATED: "#d62728", # Red
    AnnouncementCategory.FDA_APPROVAL: "#ff7f0e",   # Orange
    AnnouncementCategory.FDA_REJECTION: "#8b0000",  # Dark red
    AnnouncementCategory.FDA_SUBMISSION: "#ffbb78", # Light orange
    AnnouncementCategory.PARTNERSHIP: "#9467bd",    # Purple
    AnnouncementCategory.FINANCING: "#8c564b",      # Brown
    AnnouncementCategory.EXECUTIVE: "#e377c2",      # Pink
    AnnouncementCategory.SAFETY: "#d62728",         # Red
    AnnouncementCategory.OTHER: "#7f7f7f",          # Gray
}

class AnnouncementChartGenerator:
    """
    Generates price charts with announcement markers.
    
    Reads from data/index/announcements.csv (Phase 3 output) and
    auto-detects CSV format. Only processes announcements with
    parse_status == "OK".
    
    Example:
        generator = AnnouncementChartGenerator(eodhd_api_key="...")
        await generator.generate_all_charts(output_dir="charts/")
    """
    
    def __init__(
        self,
        eodhd_api_key: str,
        announcements_csv: Path = Path("data/index/announcements.csv"),
    ):
        self.eodhd_client = EODHDClient(api_key=eodhd_api_key)
        self.announcements_csv = announcements_csv
    
    def load_announcements_from_csv(self, ticker: str) -> list[Announcement]:
        """
        Load announcements for a ticker from CSV.
        
        Auto-detects CSV format:
        - Phase 3 format: has 'parse_status', 'event_type' columns
        - Legacy format: has 'category' column directly
        
        Only returns announcements with parse_status == "OK" (if present).
        """
        announcements = []
        
        with open(self.announcements_csv, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            # Detect format
            is_phase3_format = "parse_status" in columns
            
            for row in reader:
                if row.get("ticker") != ticker:
                    continue
                
                # Skip non-OK records in Phase 3 format
                if is_phase3_format and row.get("parse_status") != "OK":
                    continue
                
                # Derive category
                if is_phase3_format:
                    category = self._derive_category_from_event_type(
                        row.get("event_type", "OTHER")
                    )
                else:
                    category = row.get("category", "OTHER")
                
                announcements.append(Announcement(
                    id=row.get("id", ""),
                    ticker=ticker,
                    announcement_date=row.get("published_at", row.get("announcement_date", "")),
                    category=category,
                    title=row.get("title", ""),
                    source=row.get("source", ""),
                ))
        
        return announcements
    
    def _derive_category_from_event_type(self, event_type: str) -> str:
        """Map event_type to display category."""
        event_type = event_type.upper()
        
        if "8K" in event_type or "EDGAR" in event_type:
            return "EARNINGS"
        elif "CT_" in event_type or "TRIAL" in event_type:
            if "TERMINATED" in event_type:
                return "TRIAL_TERMINATED"
            elif "RESULT" in event_type:
                return "TRIAL_RESULTS"
            else:
                return "TRIAL_UPDATE"
        elif "FDA" in event_type:
            if "APPROVAL" in event_type or "AP" in event_type:
                return "FDA_APPROVAL"
            else:
                return "FDA_SUBMISSION"
        else:
            return "OTHER"
        
    async def generate_chart(
        self,
        ticker: str,
        announcements: list[Announcement],
        output_path: Path,
        days_back: Optional[int] = None,  # None = all available data
    ) -> Path:
        """
        Generate a single stock chart with announcement markers.
        
        Args:
            ticker: Stock ticker symbol
            announcements: List of announcements for this stock
            output_path: Where to save the chart
            days_back: Limit price history (None for all)
            
        Returns:
            Path to generated chart file
        """
        # Fetch price data
        async with self.eodhd_client:
            prices = await self.eodhd_client.get_historical_prices(
                ticker=ticker,
                exchange="US",
                days_back=days_back,
            )
        
        if not prices:
            raise ValueError(f"No price data for {ticker}")
        
        # Convert to DataFrame
        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{ticker} Price with Announcements", "Volume"),
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            ),
            row=1, col=1,
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                marker_color="rgba(100,100,100,0.5)",
            ),
            row=2, col=1,
        )
        
        # Add announcement markers
        for ann in announcements:
            ann_date = pd.to_datetime(ann.announcement_date)
            
            # Only show if within price data range
            if ann_date < df["date"].min() or ann_date > df["date"].max():
                continue
            
            # Get price at announcement date for y-position
            price_at_date = df[df["date"] == ann_date]["high"].values
            if len(price_at_date) == 0:
                # Find nearest date
                nearest_idx = (df["date"] - ann_date).abs().argmin()
                y_pos = df.iloc[nearest_idx]["high"] * 1.05
            else:
                y_pos = price_at_date[0] * 1.05
            
            color = CATEGORY_COLORS.get(ann.category, "#7f7f7f")
            
            # Add vertical line
            fig.add_vline(
                x=ann_date,
                line_width=1,
                line_dash="dot",
                line_color=color,
                opacity=0.7,
                row=1, col=1,
            )
            
            # Add annotation with ID
            fig.add_annotation(
                x=ann_date,
                y=y_pos,
                text=f"#{ann.id}<br>{ann.category.value[:3]}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                font=dict(size=8, color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
                row=1, col=1,
            )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - Price History with {len(announcements)} Announcements",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
        )
        
        # Add category legend
        for category, color in CATEGORY_COLORS.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=category.value,
                    showlegend=True,
                )
            )
        
        # Save chart
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as interactive HTML
        html_path = output_path.with_suffix(".html")
        fig.write_html(str(html_path))
        
        # Also save as static PNG for quick review
        png_path = output_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1600, height=900)
        
        return html_path
    
    async def generate_all_charts(
        self,
        output_dir: str = "charts/",
        tickers: Optional[list[str]] = None,
    ) -> dict[str, Path]:
        """
        Generate charts for all stocks (or specified tickers).
        
        Args:
            output_dir: Directory to save charts
            tickers: Optional list of tickers (None = all stocks)
            
        Returns:
            Dict mapping ticker to chart path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get stocks to process
        if tickers is None:
            stocks = await get_all_stocks()
            tickers = [s.ticker for s in stocks]
        
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{total}] Generating chart for {ticker}...")
            
            try:
                # Get announcements for this stock
                announcements = await get_stock_announcements(ticker)
                
                if not announcements:
                    print(f"  ⚠️  No announcements for {ticker}, skipping")
                    continue
                
                # Generate chart
                chart_path = await self.generate_chart(
                    ticker=ticker,
                    announcements=announcements,
                    output_path=output_dir / f"{ticker}_announcements",
                )
                
                results[ticker] = chart_path
                print(f"  ✅ Saved: {chart_path}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Generate index page
        self._generate_index_page(results, output_dir)
        
        return results
    
    def _generate_index_page(self, charts: dict[str, Path], output_dir: Path):
        """Generate HTML index page linking all charts."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Biopharma Announcement Charts</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .chart-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
        .chart-card img { width: 100%; height: auto; }
        .chart-card a { text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>Biopharma Announcement Charts</h1>
    <p>Generated: {timestamp}</p>
    <p>Total charts: {count}</p>
    <div class="chart-grid">
        {cards}
    </div>
</body>
</html>
"""
        cards = []
        for ticker, path in sorted(charts.items()):
            png_path = path.with_suffix(".png").name
            html_path = path.with_suffix(".html").name
            cards.append(f"""
        <div class="chart-card">
            <h3><a href="{html_path}">{ticker}</a></h3>
            <a href="{html_path}"><img src="{png_path}" alt="{ticker} chart"></a>
        </div>
""")
        
        index_html = html.format(
            timestamp=datetime.now().isoformat(),
            count=len(charts),
            cards="\n".join(cards),
        )
        
        (output_dir / "index.html").write_text(index_html)
```

#### Script to Generate All Charts

```python
# scripts/generate_announcement_charts.py

#!/usr/bin/env python3
"""
Generate price charts with announcement markers for all stocks.

Reads announcements from: data/index/announcements.csv (Phase 3 output)
Outputs to: charts/

Usage:
    python scripts/generate_announcement_charts.py
    python scripts/generate_announcement_charts.py --ticker MRNA
"""

import asyncio
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CSV = "data/index/announcements.csv"

async def main():
    parser = argparse.ArgumentParser(description="Generate announcement charts")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to announcements CSV")
    parser.add_argument("--ticker", help="Generate chart for specific ticker only")
    parser.add_argument("--output", default="charts/", help="Output directory")
    args = parser.parse_args()
    
    from src.visualization.announcement_charts import AnnouncementChartGenerator
    
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set in environment")
        return
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        print(f"   Run 'python scripts/run_extraction.py --all' first")
        return
    
    generator = AnnouncementChartGenerator(
        eodhd_api_key=api_key,
        announcements_csv=csv_path,  # Auto-detects format
    )
    
    print("🎨 Generating announcement charts...")
    print(f"   Reading from: {csv_path}")
    print("="*60)
    
    tickers = [args.ticker] if args.ticker else None
    
    charts = await generator.generate_all_charts(
        output_dir=args.output,
        tickers=tickers,
    )
    
    print("="*60)
    print(f"✅ Generated {len(charts)} charts")
    print(f"📂 Open charts/index.html to view all charts")

if __name__ == "__main__":
    asyncio.run(main())
```

**Note**: The chart generator auto-detects the CSV format:
- **Phase 3 format** (`data/index/announcements.csv`): Columns include `parse_status`, `event_type`, etc. Only processes records with `parse_status == "OK"`.
- **Legacy format** (`data/all_announcements.csv`): Deprecated, but still supported for backwards compatibility.
```

#### Dependencies to Add

```
# Add to requirements.txt
plotly>=5.18.0
kaleido>=0.2.1  # For static image export (PNG)
```

#### Test Cases for Visualization

```python
# tests/test_visualization.py

import pytest
from pathlib import Path
from datetime import date
from unittest.mock import AsyncMock, MagicMock

class TestAnnouncementChartGenerator:
    """Tests for chart generation."""
    
    @pytest.fixture
    def mock_price_data(self):
        """Sample price data."""
        return [
            {"date": "2024-01-01", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
            {"date": "2024-01-02", "open": 10.5, "high": 12, "low": 10, "close": 11, "volume": 1500},
            {"date": "2024-01-03", "open": 11, "high": 13, "low": 10.5, "close": 12, "volume": 2000},
        ]
    
    @pytest.fixture
    def mock_announcements(self):
        """Sample announcements."""
        from src.models.announcement import Announcement, AnnouncementCategory
        return [
            Announcement(
                id=1,
                ticker="TEST",
                source="edgar",
                source_id="123",
                announcement_date=date(2024, 1, 2),
                title="Q4 Earnings",
                category=AnnouncementCategory.EARNINGS,
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_generates_html_and_png(self, tmp_path, mock_price_data, mock_announcements):
        """Chart generation creates both HTML and PNG files."""
        from src.visualization.announcement_charts import AnnouncementChartGenerator
        
        generator = AnnouncementChartGenerator(eodhd_api_key="test")
        generator.eodhd_client.get_historical_prices = AsyncMock(return_value=mock_price_data)
        
        output_path = tmp_path / "TEST_chart"
        result = await generator.generate_chart(
            ticker="TEST",
            announcements=mock_announcements,
            output_path=output_path,
        )
        
        assert result.exists()
        assert (tmp_path / "TEST_chart.html").exists()
        assert (tmp_path / "TEST_chart.png").exists()
    
    def test_category_colors_defined(self):
        """All announcement categories have colors defined."""
        from src.visualization.announcement_charts import CATEGORY_COLORS
        from src.models.announcement import AnnouncementCategory
        
        for category in AnnouncementCategory:
            assert category in CATEGORY_COLORS, f"Missing color for {category}"
    
    @pytest.mark.asyncio
    async def test_skips_stocks_without_announcements(self, tmp_path):
        """Stocks with no announcements are skipped."""
        from src.visualization.announcement_charts import AnnouncementChartGenerator
        from src.db import queries
        
        generator = AnnouncementChartGenerator(eodhd_api_key="test")
        
        # Mock to return no announcements
        queries.get_stock_announcements = AsyncMock(return_value=[])
        queries.get_all_stocks = AsyncMock(return_value=[MagicMock(ticker="TEST")])
        
        results = await generator.generate_all_charts(output_dir=str(tmp_path))
        
        assert len(results) == 0  # No charts generated
```

---

### Deliverables for Phase 2.5

**Correction 2.5.1 - Remove EODHD Fundamentals**:
- [ ] Update `src/clients/eodhd.py` - remove fundamentals methods
- [ ] Update `scripts/fetch_stock_list.py` - use SEC EDGAR only
- [ ] Create `src/utils/filters.py` - SIC code filtering
- [ ] Update `config/filters.yaml` - SIC codes
- [ ] Add tests in `tests/test_stock_universe_corrections.py`

**Debugging 2.5.2 - OpenFDA**:
- [ ] Create `scripts/debug_openfda.py` - diagnostic script
- [ ] Fix `src/clients/openfda.py` - handle 404, uppercase, date format
- [ ] Add tests in `tests/test_openfda_debug.py`

**Visualization 2.5.3 - Charts**:
- [ ] Create `src/visualization/announcement_charts.py`
- [ ] Create `scripts/generate_announcement_charts.py`
- [ ] Add `plotly` and `kaleido` to requirements.txt
- [ ] Add tests in `tests/test_visualization.py`
- [ ] Output: `charts/` directory with HTML/PNG files

### Success Criteria for Phase 2.5

| Task | Metric |
|------|--------|
| EODHD fundamentals removed | No calls to `/fundamentals/` endpoint |
| SEC EDGAR primary source | All stocks have CIK from EDGAR |
| SIC filtering works | Only 2833/2834/2835/2836 stocks included |
| OpenFDA returns data | > 0 announcements from OpenFDA |
| Charts generated | HTML + PNG for each stock with announcements |
| Tests passing | 100% pass rate, 95%+ coverage |

### Running Phase 2.5

```bash
# 1. Run OpenFDA diagnostics
python scripts/debug_openfda.py

# 2. Run tests for corrections
pytest tests/test_stock_universe_corrections.py -v
pytest tests/test_openfda_debug.py -v

# 3. Re-run stock universe with SEC EDGAR only
python scripts/fetch_stock_list.py

# 4. Re-run announcement collection
python scripts/run_pipeline.py

# 5. Generate visualization charts
python scripts/generate_announcement_charts.py

# 6. Open charts in browser
open charts/index.html
```

---

## Phase 3: Text Extraction & File Storage

### Objective
Extract full text content from all announcements, store raw files and extracted text in a deterministic folder structure, and maintain a CSV index for tracking and deduplication.

### Design Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| Hash-based ID | `sha256(url + published_date)` |
| SEC EDGAR 8-K | Extract main document + ALL exhibits |
| PDF extraction | `pymupdf` (fitz) - fast, good quality |
| ClinicalTrials.gov | All text fields concatenated |
| OpenFDA | Structured JSON preserved as-is |
| CSV management | Single file forever (`announcements.csv`) |
| Failure handling | Retry 3x → mark FAILED → keep raw → write empty text file |

---

### 3.1 Folder Structure

```
data/
├── raw/                          # Original fetched content
│   ├── edgar/
│   │   └── 2025-02-07/
│   │       └── MRNA/
│   │           ├── a1b2c3d4.html      # Main 8-K document
│   │           ├── a1b2c3d4_ex99-1.htm # Exhibit 99.1
│   │           └── a1b2c3d4_ex99-2.pdf # Exhibit 99.2
│   ├── clinicaltrials/
│   │   └── 2025-02-07/
│   │       └── MRNA/
│   │           └── e5f6g7h8.json
│   ├── openfda/
│   │   └── 2025-02-07/
│   │       └── MRNA/
│   │           └── i9j0k1l2.json
│   └── ir/                        # Future: IR page scraping
│       └── ...
│
├── text/                          # Extracted plaintext
│   ├── edgar/
│   │   └── 2025-02-07/
│   │       └── MRNA/
│   │           └── a1b2c3d4.txt      # Combined: 8-K + all exhibits
│   ├── clinicaltrials/
│   │   └── 2025-02-07/
│   │       └── MRNA/
│   │           └── e5f6g7h8.txt      # All fields concatenated
│   └── openfda/
│       └── 2025-02-07/
│           └── MRNA/
│               └── i9j0k1l2.json     # Preserved as JSON (not .txt)
│
└── index/
    └── announcements.csv           # Master index file
```

### Path Generation Rules

```python
# src/storage/paths.py

import hashlib
from datetime import date
from pathlib import Path

def generate_id(url: str, published_date: date) -> str:
    """
    Generate deterministic ID from URL + date.
    
    Args:
        url: Full URL of the announcement
        published_date: Publication/filing date
        
    Returns:
        First 16 chars of SHA256 hash (sufficient for uniqueness)
    """
    content = f"{url}|{published_date.isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def get_raw_path(
    source: str,
    published_date: date,
    ticker: str,
    announcement_id: str,
    extension: str,
    exhibit_name: str | None = None,
) -> Path:
    """
    Generate path for raw file storage.
    
    Examples:
        get_raw_path("edgar", date(2025,2,7), "MRNA", "a1b2c3d4", "html")
        → data/raw/edgar/2025-02-07/MRNA/a1b2c3d4.html
        
        get_raw_path("edgar", date(2025,2,7), "MRNA", "a1b2c3d4", "htm", "ex99-1")
        → data/raw/edgar/2025-02-07/MRNA/a1b2c3d4_ex99-1.htm
    """
    base = Path("data/raw") / source / published_date.isoformat() / ticker
    if exhibit_name:
        return base / f"{announcement_id}_{exhibit_name}.{extension}"
    return base / f"{announcement_id}.{extension}"

def get_text_path(
    source: str,
    published_date: date,
    ticker: str,
    announcement_id: str,
    extension: str = "txt",  # or "json" for OpenFDA
) -> Path:
    """
    Generate path for extracted text storage.
    """
    base = Path("data/text") / source / published_date.isoformat() / ticker
    return base / f"{announcement_id}.{extension}"
```

---

### 3.2 CSV Index Schema

**File**: `data/index/announcements.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | str | SHA256 hash (16 chars), **PRIMARY KEY** | `a1b2c3d4e5f6g7h8` |
| `ticker` | str | Stock ticker | `MRNA` |
| `source` | str | Data source enum | `edgar`, `clinicaltrials`, `openfda` |
| `event_type` | str | Specific event type | `EDGAR_8K`, `CT_STATUS_CHANGE`, `FDA_APPROVAL` |
| `published_at` | str | Publication date (ISO) | `2025-02-07` |
| `fetched_at` | str | When we fetched it (ISO datetime) | `2025-02-07T14:30:00` |
| `title` | str | Short title/description | `Form 8-K: Results of Operations` |
| `url` | str | Original URL | `https://sec.gov/...` |
| `external_id` | str | Source-specific ID (nullable) | `0001682852-25-000008` (accession) |
| `raw_path` | str | Relative path to raw file | `raw/edgar/2025-02-07/MRNA/a1b2c3d4.html` |
| `raw_paths_extra` | str | JSON array of exhibit paths (nullable) | `["raw/edgar/.../a1b2c3d4_ex99-1.htm"]` |
| `text_path` | str | Relative path to extracted text | `text/edgar/2025-02-07/MRNA/a1b2c3d4.txt` |
| `raw_mime` | str | MIME type of raw content | `text/html`, `application/pdf` |
| `raw_size_bytes` | int | Size of raw file(s) | `125000` |
| `text_size_bytes` | int | Size of extracted text | `45000` |
| `parse_status` | str | Extraction status | `OK`, `FAILED`, `PENDING` |
| `parse_attempts` | int | Number of extraction attempts | `1`, `2`, `3` |
| `error` | str | Error message if failed (nullable) | `PDF extraction failed: corrupt` |
| `extra_json` | str | Source-specific metadata (nullable) | `{"items": ["2.02", "7.01"]}` |

```python
# src/storage/csv_index.py

import csv
import json
import fcntl
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from typing import Optional
from enum import Enum

class ParseStatus(Enum):
    PENDING = "PENDING"
    OK = "OK"
    FAILED = "FAILED"

class Source(Enum):
    EDGAR = "edgar"
    CLINICALTRIALS = "clinicaltrials"
    OPENFDA = "openfda"
    IR = "ir"

@dataclass
class AnnouncementRecord:
    """Single row in the announcements CSV."""
    id: str
    ticker: str
    source: str
    event_type: str
    published_at: str
    fetched_at: str
    title: str
    url: str
    external_id: Optional[str]
    raw_path: str
    raw_paths_extra: Optional[str]  # JSON array
    text_path: str
    raw_mime: str
    raw_size_bytes: int
    text_size_bytes: int
    parse_status: str
    parse_attempts: int
    error: Optional[str]
    extra_json: Optional[str]

class AnnouncementIndex:
    """
    CSV-based index for announcement tracking.
    
    Thread-safe with file locking for concurrent access.
    
    Example:
        index = AnnouncementIndex("data/index/announcements.csv")
        
        # Check if exists
        if not index.exists("a1b2c3d4"):
            index.append(record)
        
        # Update status
        index.update_status("a1b2c3d4", ParseStatus.OK)
    """
    
    COLUMNS = [
        "id", "ticker", "source", "event_type", "published_at", "fetched_at",
        "title", "url", "external_id", "raw_path", "raw_paths_extra", "text_path",
        "raw_mime", "raw_size_bytes", "text_size_bytes", "parse_status",
        "parse_attempts", "error", "extra_json"
    ]
    
    def __init__(self, path: str | Path = "data/index/announcements.csv"):
        self.path = Path(path)
        self._ensure_file_exists()
        self._id_cache: set[str] | None = None
    
    def _ensure_file_exists(self):
        """Create CSV with headers if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
    
    def _load_id_cache(self) -> set[str]:
        """Load all IDs into memory for fast dedup checking."""
        if self._id_cache is None:
            self._id_cache = set()
            with open(self.path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._id_cache.add(row["id"])
        return self._id_cache
    
    def exists(self, announcement_id: str) -> bool:
        """Check if announcement ID already exists (O(1) with cache)."""
        return announcement_id in self._load_id_cache()
    
    def append(self, record: AnnouncementRecord) -> bool:
        """
        Append a new record to the CSV.
        
        Returns:
            True if appended, False if ID already exists (dedup)
        """
        if self.exists(record.id):
            return False
        
        with open(self.path, "a", newline="") as f:
            # File locking for thread safety
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerow(asdict(record))
                self._id_cache.add(record.id)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        return True
    
    def update_status(
        self,
        announcement_id: str,
        status: ParseStatus,
        text_size_bytes: int = 0,
        error: Optional[str] = None,
        parse_attempts: Optional[int] = None,
    ):
        """Update parse status for an existing record."""
        # Read all rows, modify matching one, write back
        rows = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["id"] == announcement_id:
                    row["parse_status"] = status.value
                    row["text_size_bytes"] = str(text_size_bytes)
                    if error:
                        row["error"] = error
                    if parse_attempts:
                        row["parse_attempts"] = str(parse_attempts)
                rows.append(row)
        
        with open(self.path, "w", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def get_pending(self) -> list[AnnouncementRecord]:
        """Get all records with PENDING status."""
        pending = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["parse_status"] == ParseStatus.PENDING.value:
                    pending.append(self._row_to_record(row))
        return pending
    
    def get_failed(self) -> list[AnnouncementRecord]:
        """Get all records with FAILED status."""
        failed = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["parse_status"] == ParseStatus.FAILED.value:
                    failed.append(self._row_to_record(row))
        return failed
    
    def get_by_ticker(self, ticker: str) -> list[AnnouncementRecord]:
        """Get all records for a specific ticker."""
        results = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["ticker"] == ticker:
                    results.append(self._row_to_record(row))
        return results
    
    def get_stats(self) -> dict:
        """Get summary statistics."""
        stats = {
            "total": 0,
            "by_source": {},
            "by_status": {"OK": 0, "FAILED": 0, "PENDING": 0},
            "by_ticker": {},
        }
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["total"] += 1
                stats["by_source"][row["source"]] = stats["by_source"].get(row["source"], 0) + 1
                stats["by_status"][row["parse_status"]] = stats["by_status"].get(row["parse_status"], 0) + 1
                stats["by_ticker"][row["ticker"]] = stats["by_ticker"].get(row["ticker"], 0) + 1
        return stats
    
    @staticmethod
    def _row_to_record(row: dict) -> AnnouncementRecord:
        """Convert CSV row dict to AnnouncementRecord."""
        return AnnouncementRecord(
            id=row["id"],
            ticker=row["ticker"],
            source=row["source"],
            event_type=row["event_type"],
            published_at=row["published_at"],
            fetched_at=row["fetched_at"],
            title=row["title"],
            url=row["url"],
            external_id=row["external_id"] or None,
            raw_path=row["raw_path"],
            raw_paths_extra=row["raw_paths_extra"] or None,
            text_path=row["text_path"],
            raw_mime=row["raw_mime"],
            raw_size_bytes=int(row["raw_size_bytes"]),
            text_size_bytes=int(row["text_size_bytes"]),
            parse_status=row["parse_status"],
            parse_attempts=int(row["parse_attempts"]),
            error=row["error"] or None,
            extra_json=row["extra_json"] or None,
        )
```

---

### 3.3 Text Extraction by Source

#### 3.3.1 SEC EDGAR Extraction

SEC EDGAR filings are HTML/XML with exhibits that can be HTML, PDF, or plain text.

```python
# src/extraction/edgar_extractor.py

import re
import fitz  # pymupdf
from bs4 import BeautifulSoup
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import httpx

from src.clients.edgar import EDGARClient
from src.storage.paths import get_raw_path, get_text_path, generate_id

@dataclass
class EDGARDocument:
    """Represents a single document from an SEC filing."""
    filename: str
    url: str
    doc_type: str  # "8-K", "EX-99.1", etc.
    content: bytes
    mime_type: str

@dataclass
class ExtractionResult:
    """Result of text extraction."""
    success: bool
    text: str
    error: Optional[str] = None
    
class EDGARExtractor:
    """
    Extract text from SEC EDGAR filings.
    
    Handles:
    - HTML documents (main filing)
    - HTML exhibits (press releases)
    - PDF exhibits
    - Plain text exhibits
    
    Example:
        extractor = EDGARExtractor()
        result = await extractor.extract_filing(
            cik="1682852",
            accession_number="0001682852-25-000008",
        )
    """
    
    def __init__(self, contact_email: str):
        self.client = EDGARClient(contact_email=contact_email)
    
    async def fetch_filing_documents(
        self,
        cik: str,
        accession_number: str,
    ) -> list[EDGARDocument]:
        """
        Fetch all documents for a filing (main doc + exhibits).
        
        Args:
            cik: Company CIK
            accession_number: Filing accession number
            
        Returns:
            List of EDGARDocument objects
        """
        # Get filing index to find all documents
        index_url = self._build_index_url(cik, accession_number)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                index_url,
                headers={"User-Agent": f"BiopharmaMonitor {self.client.contact_email}"}
            )
            response.raise_for_status()
        
        # Parse index to find documents
        documents = self._parse_filing_index(response.text, cik, accession_number)
        
        # Fetch each document
        fetched = []
        async with httpx.AsyncClient() as client:
            for doc in documents:
                try:
                    resp = await client.get(
                        doc.url,
                        headers={"User-Agent": f"BiopharmaMonitor {self.client.contact_email}"}
                    )
                    resp.raise_for_status()
                    doc.content = resp.content
                    doc.mime_type = resp.headers.get("content-type", "text/html")
                    fetched.append(doc)
                except Exception as e:
                    # Log but continue with other documents
                    print(f"Warning: Failed to fetch {doc.url}: {e}")
        
        return fetched
    
    def _build_index_url(self, cik: str, accession_number: str) -> str:
        """Build URL for filing index page."""
        acc_no_dashes = accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{accession_number}-index.html"
    
    def _parse_filing_index(
        self,
        index_html: str,
        cik: str,
        accession_number: str,
    ) -> list[EDGARDocument]:
        """Parse filing index to get list of documents."""
        soup = BeautifulSoup(index_html, "lxml")
        documents = []
        
        # Find document table
        table = soup.find("table", class_="tableFile")
        if not table:
            return documents
        
        acc_no_dashes = accession_number.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/"
        
        for row in table.find_all("tr")[1:]:  # Skip header
            cells = row.find_all("td")
            if len(cells) >= 4:
                doc_type = cells[3].get_text(strip=True)
                filename = cells[2].get_text(strip=True)
                
                # Include main document and exhibits
                if doc_type in ["8-K", "8-K/A"] or doc_type.startswith("EX-"):
                    documents.append(EDGARDocument(
                        filename=filename,
                        url=base_url + filename,
                        doc_type=doc_type,
                        content=b"",
                        mime_type="",
                    ))
        
        return documents
    
    def extract_text_from_document(self, doc: EDGARDocument) -> ExtractionResult:
        """
        Extract text from a single document.
        
        Handles HTML, PDF, and plain text.
        """
        mime = doc.mime_type.lower()
        
        try:
            if "pdf" in mime or doc.filename.lower().endswith(".pdf"):
                return self._extract_from_pdf(doc.content)
            elif "html" in mime or doc.filename.lower().endswith((".htm", ".html")):
                return self._extract_from_html(doc.content)
            else:
                # Assume plain text
                return ExtractionResult(
                    success=True,
                    text=doc.content.decode("utf-8", errors="replace"),
                )
        except Exception as e:
            return ExtractionResult(
                success=False,
                text="",
                error=str(e),
            )
    
    def _extract_from_html(self, content: bytes) -> ExtractionResult:
        """Extract text from HTML document."""
        try:
            soup = BeautifulSoup(content, "lxml")
            
            # Remove script and style elements
            for element in soup(["script", "style", "head", "meta", "link"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            
            return ExtractionResult(success=True, text=text.strip())
        except Exception as e:
            return ExtractionResult(success=False, text="", error=str(e))
    
    def _extract_from_pdf(self, content: bytes) -> ExtractionResult:
        """Extract text from PDF document using pymupdf."""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())
            
            doc.close()
            
            text = "\n\n".join(text_parts)
            return ExtractionResult(success=True, text=text.strip())
        except Exception as e:
            return ExtractionResult(success=False, text="", error=f"PDF extraction failed: {e}")
    
    async def extract_filing(
        self,
        cik: str,
        accession_number: str,
        ticker: str,
        published_date,
        max_retries: int = 3,
    ) -> tuple[list[Path], Path, ExtractionResult]:
        """
        Full extraction pipeline for a filing.
        
        Args:
            cik: Company CIK
            accession_number: Filing accession number
            ticker: Stock ticker
            published_date: Filing date
            max_retries: Max extraction attempts
            
        Returns:
            Tuple of (raw_paths, text_path, extraction_result)
        """
        from src.storage.paths import generate_id, get_raw_path, get_text_path
        
        announcement_id = generate_id(
            url=f"https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=40&accession={accession_number}",
            published_date=published_date,
        )
        
        # Fetch all documents
        documents = await self.fetch_filing_documents(cik, accession_number)
        
        if not documents:
            return [], Path(), ExtractionResult(success=False, text="", error="No documents found")
        
        # Save raw files
        raw_paths = []
        for doc in documents:
            ext = doc.filename.split(".")[-1] if "." in doc.filename else "html"
            
            if doc.doc_type in ["8-K", "8-K/A"]:
                raw_path = get_raw_path("edgar", published_date, ticker, announcement_id, ext)
            else:
                # Exhibit
                exhibit_name = doc.doc_type.lower().replace("-", "").replace(".", "")
                raw_path = get_raw_path("edgar", published_date, ticker, announcement_id, ext, exhibit_name)
            
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(doc.content)
            raw_paths.append(raw_path)
        
        # Extract text with retry
        combined_text = []
        all_success = True
        errors = []
        
        for attempt in range(max_retries):
            combined_text = []
            all_success = True
            errors = []
            
            for doc in documents:
                result = self.extract_text_from_document(doc)
                if result.success:
                    # Add document header
                    combined_text.append(f"=== {doc.doc_type}: {doc.filename} ===\n")
                    combined_text.append(result.text)
                    combined_text.append("\n\n")
                else:
                    all_success = False
                    errors.append(f"{doc.filename}: {result.error}")
            
            if all_success or attempt == max_retries - 1:
                break
        
        # Save text file
        text_path = get_text_path("edgar", published_date, ticker, announcement_id, "txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_text = "".join(combined_text)
        text_path.write_text(final_text)
        
        return raw_paths, text_path, ExtractionResult(
            success=all_success,
            text=final_text,
            error="; ".join(errors) if errors else None,
        )
```

#### 3.3.2 ClinicalTrials.gov Extraction

```python
# src/extraction/clinicaltrials_extractor.py

import json
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.storage.paths import generate_id, get_raw_path, get_text_path

@dataclass
class CTExtractionResult:
    """Result of ClinicalTrials.gov text extraction."""
    success: bool
    text: str
    error: Optional[str] = None

class ClinicalTrialsExtractor:
    """
    Extract text from ClinicalTrials.gov study data.
    
    Concatenates all relevant text fields into a single document.
    
    Fields extracted:
    - Official title
    - Brief summary
    - Detailed description
    - Conditions
    - Interventions (name + description)
    - Eligibility criteria
    - Primary/secondary outcomes
    - Study design
    - Arms/groups
    - Sponsor/collaborators
    - Locations (if needed)
    """
    
    # Fields to extract (in order)
    TEXT_FIELDS = [
        ("identificationModule", "officialTitle", "Official Title"),
        ("identificationModule", "briefTitle", "Brief Title"),
        ("descriptionModule", "briefSummary", "Brief Summary"),
        ("descriptionModule", "detailedDescription", "Detailed Description"),
        ("conditionsModule", "conditions", "Conditions"),
        ("conditionsModule", "keywords", "Keywords"),
        ("designModule", "studyType", "Study Type"),
        ("designModule", "phases", "Phases"),
        ("eligibilityModule", "eligibilityCriteria", "Eligibility Criteria"),
        ("eligibilityModule", "healthyVolunteers", "Healthy Volunteers"),
        ("eligibilityModule", "sex", "Sex"),
        ("eligibilityModule", "minimumAge", "Minimum Age"),
        ("eligibilityModule", "maximumAge", "Maximum Age"),
    ]
    
    def extract_from_study(self, study_json: dict) -> CTExtractionResult:
        """
        Extract all text from a study JSON response.
        
        Args:
            study_json: Full study object from ClinicalTrials.gov API
            
        Returns:
            CTExtractionResult with concatenated text
        """
        try:
            protocol = study_json.get("protocolSection", {})
            text_parts = []
            
            # Extract NCT ID first
            nct_id = protocol.get("identificationModule", {}).get("nctId", "Unknown")
            text_parts.append(f"=== Clinical Trial: {nct_id} ===\n")
            
            # Extract standard fields
            for module_name, field_name, label in self.TEXT_FIELDS:
                module = protocol.get(module_name, {})
                value = module.get(field_name)
                if value:
                    text_parts.append(f"\n## {label}\n")
                    if isinstance(value, list):
                        text_parts.append(", ".join(str(v) for v in value))
                    else:
                        text_parts.append(str(value))
            
            # Extract interventions (nested)
            interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
            if interventions:
                text_parts.append("\n\n## Interventions\n")
                for interv in interventions:
                    name = interv.get("name", "Unknown")
                    itype = interv.get("type", "")
                    desc = interv.get("description", "")
                    text_parts.append(f"- {name} ({itype}): {desc}\n")
            
            # Extract outcomes
            outcomes_module = protocol.get("outcomesModule", {})
            primary = outcomes_module.get("primaryOutcomes", [])
            if primary:
                text_parts.append("\n\n## Primary Outcomes\n")
                for outcome in primary:
                    measure = outcome.get("measure", "")
                    timeframe = outcome.get("timeFrame", "")
                    desc = outcome.get("description", "")
                    text_parts.append(f"- {measure} (Timeframe: {timeframe})\n  {desc}\n")
            
            secondary = outcomes_module.get("secondaryOutcomes", [])
            if secondary:
                text_parts.append("\n\n## Secondary Outcomes\n")
                for outcome in secondary:
                    measure = outcome.get("measure", "")
                    text_parts.append(f"- {measure}\n")
            
            # Extract sponsor info
            sponsors = protocol.get("sponsorCollaboratorsModule", {})
            lead = sponsors.get("leadSponsor", {})
            if lead:
                text_parts.append(f"\n\n## Sponsor\n{lead.get('name', 'Unknown')}")
            
            collaborators = sponsors.get("collaborators", [])
            if collaborators:
                text_parts.append("\n\n## Collaborators\n")
                for collab in collaborators:
                    text_parts.append(f"- {collab.get('name', 'Unknown')}\n")
            
            # Extract status info
            status = protocol.get("statusModule", {})
            if status:
                text_parts.append(f"\n\n## Status\n")
                text_parts.append(f"Overall Status: {status.get('overallStatus', 'Unknown')}\n")
                text_parts.append(f"Start Date: {status.get('startDateStruct', {}).get('date', 'Unknown')}\n")
                text_parts.append(f"Completion Date: {status.get('completionDateStruct', {}).get('date', 'Unknown')}\n")
            
            final_text = "".join(text_parts)
            return CTExtractionResult(success=True, text=final_text.strip())
            
        except Exception as e:
            return CTExtractionResult(success=False, text="", error=str(e))
    
    def save_extraction(
        self,
        study_json: dict,
        ticker: str,
        published_date: date,
        nct_id: str,
        max_retries: int = 3,
    ) -> tuple[Path, Path, CTExtractionResult]:
        """
        Save raw JSON and extracted text for a study.
        
        Returns:
            Tuple of (raw_path, text_path, extraction_result)
        """
        url = f"https://clinicaltrials.gov/study/{nct_id}"
        announcement_id = generate_id(url=url, published_date=published_date)
        
        # Save raw JSON
        raw_path = get_raw_path("clinicaltrials", published_date, ticker, announcement_id, "json")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(study_json, indent=2))
        
        # Extract text with retry
        result = None
        for attempt in range(max_retries):
            result = self.extract_from_study(study_json)
            if result.success:
                break
        
        # Save text file (empty if all retries failed)
        text_path = get_text_path("clinicaltrials", published_date, ticker, announcement_id, "txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(result.text if result.success else "")
        
        return raw_path, text_path, result
```

#### 3.3.3 OpenFDA Extraction

```python
# src/extraction/openfda_extractor.py

import json
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.storage.paths import generate_id, get_raw_path, get_text_path

@dataclass
class FDAExtractionResult:
    """Result of OpenFDA extraction."""
    success: bool
    json_data: dict  # Preserved as-is
    error: Optional[str] = None

class OpenFDAExtractor:
    """
    Extract/preserve OpenFDA data.
    
    Unlike other sources, OpenFDA data is preserved as structured JSON
    rather than converted to plain text. This is because:
    1. Drug labels are already structured (indications, warnings, dosage, etc.)
    2. The structure is valuable for downstream analysis
    3. Converting to plain text loses important categorization
    
    The "text" file is actually a JSON file with extension .json
    """
    
    def save_extraction(
        self,
        fda_result: dict,
        ticker: str,
        published_date: date,
        application_number: str,
    ) -> tuple[Path, Path, FDAExtractionResult]:
        """
        Save OpenFDA result as JSON.
        
        Both raw and "text" paths point to JSON files.
        
        Args:
            fda_result: Single result from OpenFDA API
            ticker: Stock ticker
            published_date: Approval/submission date
            application_number: NDA/BLA number
            
        Returns:
            Tuple of (raw_path, text_path, extraction_result)
        """
        url = f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{application_number}"
        announcement_id = generate_id(url=url, published_date=published_date)
        
        try:
            # Save raw JSON
            raw_path = get_raw_path("openfda", published_date, ticker, announcement_id, "json")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(fda_result, indent=2))
            
            # For OpenFDA, "text" is also JSON (preserved structure)
            # We use .json extension instead of .txt
            text_path = get_text_path("openfda", published_date, ticker, announcement_id, "json")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract just the key fields we care about
            extracted = self._extract_key_fields(fda_result)
            text_path.write_text(json.dumps(extracted, indent=2))
            
            return raw_path, text_path, FDAExtractionResult(
                success=True,
                json_data=extracted,
            )
            
        except Exception as e:
            # Still try to save raw if possible
            return raw_path, text_path, FDAExtractionResult(
                success=False,
                json_data={},
                error=str(e),
            )
    
    def _extract_key_fields(self, result: dict) -> dict:
        """
        Extract key fields from OpenFDA result.
        
        Preserves structure but filters to relevant fields.
        """
        extracted = {
            "application_number": result.get("application_number"),
            "sponsor_name": result.get("sponsor_name"),
            "products": [],
            "submissions": [],
            "openfda": result.get("openfda", {}),
        }
        
        # Extract product info
        for product in result.get("products", []):
            extracted["products"].append({
                "brand_name": product.get("brand_name"),
                "active_ingredients": product.get("active_ingredients", []),
                "dosage_form": product.get("dosage_form"),
                "route": product.get("route"),
                "marketing_status": product.get("marketing_status"),
            })
        
        # Extract submission info
        for submission in result.get("submissions", []):
            extracted["submissions"].append({
                "submission_type": submission.get("submission_type"),
                "submission_number": submission.get("submission_number"),
                "submission_status": submission.get("submission_status"),
                "submission_status_date": submission.get("submission_status_date"),
                "submission_class_code": submission.get("submission_class_code"),
            })
        
        return extracted
```

---

### 3.4 Ingestion Pipeline

```python
# src/extraction/pipeline.py

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from src.storage.csv_index import AnnouncementIndex, AnnouncementRecord, ParseStatus
from src.storage.paths import generate_id
from src.extraction.edgar_extractor import EDGARExtractor
from src.extraction.clinicaltrials_extractor import ClinicalTrialsExtractor
from src.extraction.openfda_extractor import OpenFDAExtractor
from src.clients.edgar import EDGARClient, SECFiling
from src.clients.clinicaltrials import ClinicalTrialsClient, ClinicalTrial
from src.clients.openfda import OpenFDAClient, DrugApproval

class ExtractionPipeline:
    """
    Main pipeline for fetching and extracting announcement content.
    
    Workflow for each announcement:
    1. Fetch raw content + metadata
    2. Write raw file(s) → get raw_path
    3. Extract plaintext → write text file → get text_path
    4. Compute ID (hash) and dedupe
    5. Append row to announcements.csv
    6. Set parse_status=OK or FAILED with error
    
    Example:
        pipeline = ExtractionPipeline(
            sec_email="contact@example.com",
            openfda_key=None,
        )
        await pipeline.process_all_announcements(tickers=["MRNA", "PFE"])
    """
    
    def __init__(
        self,
        sec_email: str,
        openfda_key: Optional[str] = None,
        index_path: str = "data/index/announcements.csv",
    ):
        self.edgar_extractor = EDGARExtractor(contact_email=sec_email)
        self.ct_extractor = ClinicalTrialsExtractor()
        self.fda_extractor = OpenFDAExtractor()
        self.index = AnnouncementIndex(index_path)
        self.openfda_key = openfda_key
    
    async def process_edgar_filing(
        self,
        filing: SECFiling,
        ticker: str,
    ) -> bool:
        """
        Process a single SEC EDGAR filing.
        
        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        # Generate ID for dedup check
        announcement_id = generate_id(filing.url, filing.filing_date)
        
        if self.index.exists(announcement_id):
            return False  # Already processed
        
        # Create initial record with PENDING status
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="edgar",
            event_type=f"EDGAR_{filing.form_type.replace('-', '_').replace('/', '_')}",
            published_at=filing.filing_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=f"{filing.form_type}: {', '.join(filing.items) or 'Filing'}",
            url=filing.url,
            external_id=filing.accession_number,
            raw_path="",  # Will be updated
            raw_paths_extra=None,
            text_path="",  # Will be updated
            raw_mime="text/html",
            raw_size_bytes=0,
            text_size_bytes=0,
            parse_status=ParseStatus.PENDING.value,
            parse_attempts=0,
            error=None,
            extra_json=json.dumps({"items": filing.items}),
        )
        
        # Extract content
        try:
            raw_paths, text_path, result = await self.edgar_extractor.extract_filing(
                cik=filing.cik,
                accession_number=filing.accession_number,
                ticker=ticker,
                published_date=filing.filing_date,
                max_retries=3,
            )
            
            # Update record
            record.raw_path = str(raw_paths[0]) if raw_paths else ""
            record.raw_paths_extra = json.dumps([str(p) for p in raw_paths[1:]]) if len(raw_paths) > 1 else None
            record.text_path = str(text_path)
            record.raw_size_bytes = sum(p.stat().st_size for p in raw_paths if p.exists())
            record.text_size_bytes = text_path.stat().st_size if text_path.exists() else 0
            record.parse_status = ParseStatus.OK.value if result.success else ParseStatus.FAILED.value
            record.parse_attempts = 3 if not result.success else 1
            record.error = result.error
            
        except Exception as e:
            record.parse_status = ParseStatus.FAILED.value
            record.parse_attempts = 3
            record.error = str(e)
        
        # Save to index
        self.index.append(record)
        return True
    
    async def process_clinical_trial(
        self,
        trial: ClinicalTrial,
        study_json: dict,
        ticker: str,
    ) -> bool:
        """
        Process a single ClinicalTrials.gov study.
        
        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        url = f"https://clinicaltrials.gov/study/{trial.nct_id}"
        announcement_id = generate_id(url, trial.last_update_date)
        
        if self.index.exists(announcement_id):
            return False
        
        # Extract and save
        raw_path, text_path, result = self.ct_extractor.save_extraction(
            study_json=study_json,
            ticker=ticker,
            published_date=trial.last_update_date,
            nct_id=trial.nct_id,
            max_retries=3,
        )
        
        # Create record
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="clinicaltrials",
            event_type=f"CT_{trial.status.value}",
            published_at=trial.last_update_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=trial.title[:200] if trial.title else trial.nct_id,
            url=url,
            external_id=trial.nct_id,
            raw_path=str(raw_path),
            raw_paths_extra=None,
            text_path=str(text_path),
            raw_mime="application/json",
            raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
            text_size_bytes=text_path.stat().st_size if text_path.exists() else 0,
            parse_status=ParseStatus.OK.value if result.success else ParseStatus.FAILED.value,
            parse_attempts=1 if result.success else 3,
            error=result.error,
            extra_json=json.dumps({
                "status": trial.status.value,
                "phase": trial.phase,
                "conditions": trial.conditions,
            }),
        )
        
        self.index.append(record)
        return True
    
    async def process_fda_approval(
        self,
        approval: DrugApproval,
        fda_result: dict,
        ticker: str,
    ) -> bool:
        """
        Process a single OpenFDA approval.
        
        Returns:
            True if processed (new), False if skipped (duplicate)
        """
        url = f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{approval.application_number}"
        published_date = approval.approval_date or datetime.now().date()
        announcement_id = generate_id(url, published_date)
        
        if self.index.exists(announcement_id):
            return False
        
        # Save extraction
        raw_path, text_path, result = self.fda_extractor.save_extraction(
            fda_result=fda_result,
            ticker=ticker,
            published_date=published_date,
            application_number=approval.application_number,
        )
        
        # Create record
        record = AnnouncementRecord(
            id=announcement_id,
            ticker=ticker,
            source="openfda",
            event_type=f"FDA_{approval.submission_type}",
            published_at=published_date.isoformat(),
            fetched_at=datetime.now().isoformat(),
            title=f"{approval.brand_name} ({approval.application_number})",
            url=url,
            external_id=approval.application_number,
            raw_path=str(raw_path),
            raw_paths_extra=None,
            text_path=str(text_path),
            raw_mime="application/json",
            raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
            text_size_bytes=text_path.stat().st_size if text_path.exists() else 0,
            parse_status=ParseStatus.OK.value if result.success else ParseStatus.FAILED.value,
            parse_attempts=1 if result.success else 3,
            error=result.error,
            extra_json=json.dumps({
                "submission_type": approval.submission_type,
                "brand_name": approval.brand_name,
                "generic_name": approval.generic_name,
            }),
        )
        
        self.index.append(record)
        return True
    
    def get_stats(self) -> dict:
        """Get extraction statistics."""
        return self.index.get_stats()
```

---

### 3.5 Comprehensive Test Suite

#### Test: Path Generation

```python
# tests/test_paths.py

import pytest
from datetime import date
from pathlib import Path

from src.storage.paths import generate_id, get_raw_path, get_text_path

class TestPathGeneration:
    """Tests for deterministic path generation."""
    
    def test_generate_id_deterministic(self):
        """Same URL + date always produces same ID."""
        url = "https://example.com/filing"
        d = date(2025, 2, 7)
        
        id1 = generate_id(url, d)
        id2 = generate_id(url, d)
        
        assert id1 == id2
        assert len(id1) == 16  # Truncated SHA256
    
    def test_generate_id_different_inputs(self):
        """Different inputs produce different IDs."""
        url = "https://example.com/filing"
        
        id1 = generate_id(url, date(2025, 2, 7))
        id2 = generate_id(url, date(2025, 2, 8))  # Different date
        id3 = generate_id("https://other.com", date(2025, 2, 7))  # Different URL
        
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_get_raw_path_basic(self):
        """Basic raw path generation."""
        path = get_raw_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="html",
        )
        
        assert path == Path("data/raw/edgar/2025-02-07/MRNA/a1b2c3d4.html")
    
    def test_get_raw_path_with_exhibit(self):
        """Raw path with exhibit name."""
        path = get_raw_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="htm",
            exhibit_name="ex99-1",
        )
        
        assert path == Path("data/raw/edgar/2025-02-07/MRNA/a1b2c3d4_ex99-1.htm")
    
    def test_get_text_path_default_extension(self):
        """Text path defaults to .txt extension."""
        path = get_text_path(
            source="edgar",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
        )
        
        assert path == Path("data/text/edgar/2025-02-07/MRNA/a1b2c3d4.txt")
    
    def test_get_text_path_custom_extension(self):
        """Text path with custom extension (for JSON)."""
        path = get_text_path(
            source="openfda",
            published_date=date(2025, 2, 7),
            ticker="MRNA",
            announcement_id="a1b2c3d4",
            extension="json",
        )
        
        assert path == Path("data/text/openfda/2025-02-07/MRNA/a1b2c3d4.json")
```

#### Test: CSV Index

```python
# tests/test_csv_index.py

import pytest
from pathlib import Path
import tempfile
import csv

from src.storage.csv_index import AnnouncementIndex, AnnouncementRecord, ParseStatus

class TestAnnouncementIndex:
    """Tests for CSV index management."""
    
    @pytest.fixture
    def temp_csv(self):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            yield Path(f.name)
    
    @pytest.fixture
    def sample_record(self):
        """Sample announcement record."""
        return AnnouncementRecord(
            id="a1b2c3d4e5f6g7h8",
            ticker="MRNA",
            source="edgar",
            event_type="EDGAR_8K",
            published_at="2025-02-07",
            fetched_at="2025-02-07T14:30:00",
            title="Form 8-K: Results",
            url="https://sec.gov/...",
            external_id="0001682852-25-000008",
            raw_path="raw/edgar/2025-02-07/MRNA/a1b2c3d4.html",
            raw_paths_extra=None,
            text_path="text/edgar/2025-02-07/MRNA/a1b2c3d4.txt",
            raw_mime="text/html",
            raw_size_bytes=125000,
            text_size_bytes=45000,
            parse_status="OK",
            parse_attempts=1,
            error=None,
            extra_json='{"items": ["2.02"]}',
        )
    
    def test_creates_file_with_headers(self, temp_csv):
        """Index creates CSV with correct headers."""
        # Delete file first so index creates it
        temp_csv.unlink()
        
        index = AnnouncementIndex(temp_csv)
        
        assert temp_csv.exists()
        with open(temp_csv) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == AnnouncementIndex.COLUMNS
    
    def test_append_record(self, temp_csv, sample_record):
        """Appends record to CSV."""
        index = AnnouncementIndex(temp_csv)
        result = index.append(sample_record)
        
        assert result == True
        assert index.exists(sample_record.id)
    
    def test_dedup_prevents_duplicate(self, temp_csv, sample_record):
        """Duplicate ID is rejected."""
        index = AnnouncementIndex(temp_csv)
        
        result1 = index.append(sample_record)
        result2 = index.append(sample_record)  # Same ID
        
        assert result1 == True
        assert result2 == False
    
    def test_exists_uses_cache(self, temp_csv, sample_record):
        """Exists check uses in-memory cache."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)
        
        # Should use cache, not read file
        assert index.exists(sample_record.id) == True
        assert index.exists("nonexistent") == False
    
    def test_update_status(self, temp_csv, sample_record):
        """Updates parse status."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)
        
        index.update_status(
            sample_record.id,
            ParseStatus.FAILED,
            error="Test error",
            parse_attempts=3,
        )
        
        # Re-read to verify
        records = index.get_failed()
        assert len(records) == 1
        assert records[0].error == "Test error"
        assert records[0].parse_attempts == 3
    
    def test_get_pending(self, temp_csv):
        """Gets records with PENDING status."""
        index = AnnouncementIndex(temp_csv)
        
        pending_record = AnnouncementRecord(
            id="pending123",
            ticker="TEST",
            source="edgar",
            event_type="EDGAR_8K",
            published_at="2025-02-07",
            fetched_at="2025-02-07T14:30:00",
            title="Pending",
            url="https://...",
            external_id=None,
            raw_path="",
            raw_paths_extra=None,
            text_path="",
            raw_mime="text/html",
            raw_size_bytes=0,
            text_size_bytes=0,
            parse_status="PENDING",
            parse_attempts=0,
            error=None,
            extra_json=None,
        )
        
        index.append(pending_record)
        
        pending = index.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "pending123"
    
    def test_get_stats(self, temp_csv, sample_record):
        """Gets summary statistics."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)
        
        stats = index.get_stats()
        
        assert stats["total"] == 1
        assert stats["by_source"]["edgar"] == 1
        assert stats["by_status"]["OK"] == 1
        assert stats["by_ticker"]["MRNA"] == 1
    
    def test_get_by_ticker(self, temp_csv, sample_record):
        """Gets records for specific ticker."""
        index = AnnouncementIndex(temp_csv)
        index.append(sample_record)
        
        records = index.get_by_ticker("MRNA")
        assert len(records) == 1
        
        records = index.get_by_ticker("OTHER")
        assert len(records) == 0
```

#### Test: EDGAR Extraction

```python
# tests/test_edgar_extraction.py

import pytest
from datetime import date
from pathlib import Path
import tempfile

from src.extraction.edgar_extractor import (
    EDGARExtractor,
    EDGARDocument,
    ExtractionResult,
)

class TestEDGARExtraction:
    """Tests for SEC EDGAR text extraction."""
    
    @pytest.fixture
    def sample_html(self):
        """Sample 8-K HTML content."""
        return b"""
        <!DOCTYPE html>
        <html>
        <head><title>Form 8-K</title></head>
        <body>
            <h1>FORM 8-K</h1>
            <p>CURRENT REPORT</p>
            <p>Item 2.02 Results of Operations</p>
            <p>The company announced quarterly results...</p>
            <script>alert('ignored')</script>
            <style>.ignored { display: none; }</style>
        </body>
        </html>
        """
    
    @pytest.fixture
    def sample_pdf_bytes(self):
        """Minimal valid PDF bytes (for testing)."""
        # This is a minimal PDF that can be parsed
        # In real tests, you might load a test fixture file
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
    
    def test_extract_from_html(self, sample_html):
        """Extracts text from HTML document."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        doc = EDGARDocument(
            filename="filing.html",
            url="https://sec.gov/...",
            doc_type="8-K",
            content=sample_html,
            mime_type="text/html",
        )
        
        result = extractor.extract_text_from_document(doc)
        
        assert result.success == True
        assert "FORM 8-K" in result.text
        assert "Item 2.02" in result.text
        assert "quarterly results" in result.text
        # Script and style should be removed
        assert "alert" not in result.text
        assert "ignored" not in result.text
    
    def test_extract_removes_excessive_whitespace(self, sample_html):
        """Excessive whitespace is cleaned up."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        # Add excessive whitespace
        html_with_spaces = sample_html.replace(b"<p>", b"<p>\n\n\n\n")
        
        doc = EDGARDocument(
            filename="filing.html",
            url="https://sec.gov/...",
            doc_type="8-K",
            content=html_with_spaces,
            mime_type="text/html",
        )
        
        result = extractor.extract_text_from_document(doc)
        
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result.text
    
    def test_extract_from_pdf_empty(self):
        """Handles empty/minimal PDF."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        # Minimal PDF with no text
        minimal_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
        
        doc = EDGARDocument(
            filename="exhibit.pdf",
            url="https://sec.gov/...",
            doc_type="EX-99.1",
            content=minimal_pdf,
            mime_type="application/pdf",
        )
        
        result = extractor.extract_text_from_document(doc)
        
        # Should succeed even with empty PDF
        assert result.success == True
        assert result.text == ""
    
    def test_extract_from_invalid_pdf(self):
        """Handles corrupted PDF gracefully."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        doc = EDGARDocument(
            filename="corrupt.pdf",
            url="https://sec.gov/...",
            doc_type="EX-99.1",
            content=b"not a pdf",
            mime_type="application/pdf",
        )
        
        result = extractor.extract_text_from_document(doc)
        
        assert result.success == False
        assert "PDF extraction failed" in result.error
    
    def test_extract_from_plain_text(self):
        """Handles plain text files."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        doc = EDGARDocument(
            filename="exhibit.txt",
            url="https://sec.gov/...",
            doc_type="EX-99.1",
            content=b"Plain text content here",
            mime_type="text/plain",
        )
        
        result = extractor.extract_text_from_document(doc)
        
        assert result.success == True
        assert result.text == "Plain text content here"
    
    def test_determines_type_from_extension(self):
        """Uses file extension when MIME type is unclear."""
        extractor = EDGARExtractor(contact_email="test@example.com")
        
        doc = EDGARDocument(
            filename="filing.htm",
            url="https://sec.gov/...",
            doc_type="8-K",
            content=b"<html><body>Test</body></html>",
            mime_type="application/octet-stream",  # Unclear MIME
        )
        
        result = extractor.extract_text_from_document(doc)
        
        # Should detect as HTML from extension
        assert result.success == True
        assert "Test" in result.text


class TestEDGARExtractionRetry:
    """Tests for extraction retry logic."""
    
    def test_retries_on_failure(self):
        """Retries extraction up to max_retries times."""
        # This would need mocking to test properly
        pass
    
    def test_writes_empty_text_on_final_failure(self, tmp_path):
        """Writes empty text file after all retries fail."""
        # This would need mocking to test properly
        pass
```

#### Test: ClinicalTrials Extraction

```python
# tests/test_clinicaltrials_extraction.py

import pytest
from datetime import date
import json

from src.extraction.clinicaltrials_extractor import (
    ClinicalTrialsExtractor,
    CTExtractionResult,
)

class TestClinicalTrialsExtraction:
    """Tests for ClinicalTrials.gov text extraction."""
    
    @pytest.fixture
    def sample_study_json(self):
        """Sample study JSON from API."""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT04470427",
                    "officialTitle": "A Phase 3 Study to Evaluate Efficacy of mRNA-1273",
                    "briefTitle": "COVE Study",
                },
                "descriptionModule": {
                    "briefSummary": "This study evaluates the vaccine efficacy.",
                    "detailedDescription": "Detailed protocol information here.",
                },
                "conditionsModule": {
                    "conditions": ["COVID-19", "SARS-CoV-2"],
                    "keywords": ["vaccine", "mRNA"],
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Adults 18+ years",
                    "healthyVolunteers": "Yes",
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "None",
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "BIOLOGICAL",
                            "name": "mRNA-1273",
                            "description": "mRNA vaccine candidate",
                        }
                    ],
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Vaccine Efficacy",
                            "timeFrame": "14 days post dose 2",
                            "description": "Prevention of COVID-19",
                        }
                    ],
                    "secondaryOutcomes": [
                        {"measure": "Safety profile"},
                    ],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "ModernaTX, Inc."},
                    "collaborators": [{"name": "BARDA"}],
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "startDateStruct": {"date": "2020-07-27"},
                    "completionDateStruct": {"date": "2023-06-30"},
                },
            },
            "hasResults": True,
        }
    
    def test_extract_all_fields(self, sample_study_json):
        """Extracts all relevant text fields."""
        extractor = ClinicalTrialsExtractor()
        result = extractor.extract_from_study(sample_study_json)
        
        assert result.success == True
        
        # Check key content is present
        assert "NCT04470427" in result.text
        assert "Phase 3 Study" in result.text
        assert "vaccine efficacy" in result.text.lower()
        assert "COVID-19" in result.text
        assert "mRNA-1273" in result.text
        assert "ModernaTX" in result.text
        assert "BARDA" in result.text
        assert "COMPLETED" in result.text
    
    def test_includes_section_headers(self, sample_study_json):
        """Text includes section headers for readability."""
        extractor = ClinicalTrialsExtractor()
        result = extractor.extract_from_study(sample_study_json)
        
        assert "## Official Title" in result.text
        assert "## Brief Summary" in result.text
        assert "## Interventions" in result.text
        assert "## Primary Outcomes" in result.text
    
    def test_handles_missing_fields(self):
        """Handles studies with missing optional fields."""
        minimal_study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000001",
                },
                "statusModule": {
                    "overallStatus": "UNKNOWN",
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Unknown"},
                },
            }
        }
        
        extractor = ClinicalTrialsExtractor()
        result = extractor.extract_from_study(minimal_study)
        
        assert result.success == True
        assert "NCT00000001" in result.text
    
    def test_handles_empty_arrays(self):
        """Handles empty arrays gracefully."""
        study_with_empty = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {
                    "primaryOutcomes": [],
                    "secondaryOutcomes": [],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Test"},
                    "collaborators": [],
                },
                "statusModule": {"overallStatus": "RECRUITING"},
            }
        }
        
        extractor = ClinicalTrialsExtractor()
        result = extractor.extract_from_study(study_with_empty)
        
        assert result.success == True
    
    def test_handles_malformed_json(self):
        """Returns failure for completely malformed input."""
        extractor = ClinicalTrialsExtractor()
        result = extractor.extract_from_study({})
        
        # Should handle gracefully (might succeed with empty text or fail)
        assert isinstance(result.text, str)
```

#### Test: OpenFDA Extraction

```python
# tests/test_openfda_extraction.py

import pytest
from datetime import date
import json
import tempfile
from pathlib import Path

from src.extraction.openfda_extractor import (
    OpenFDAExtractor,
    FDAExtractionResult,
)

class TestOpenFDAExtraction:
    """Tests for OpenFDA data preservation."""
    
    @pytest.fixture
    def sample_fda_result(self):
        """Sample OpenFDA result."""
        return {
            "application_number": "BLA761222",
            "sponsor_name": "MODERNATX, INC.",
            "products": [
                {
                    "brand_name": "SPIKEVAX",
                    "active_ingredients": [
                        {"name": "ELASOMERAN", "strength": "0.1 MG/ML"}
                    ],
                    "dosage_form": "INJECTION",
                    "route": "INTRAMUSCULAR",
                    "marketing_status": "Prescription",
                }
            ],
            "submissions": [
                {
                    "submission_type": "BLA",
                    "submission_number": "1",
                    "submission_status": "AP",
                    "submission_status_date": "20220131",
                    "submission_class_code": "Priority",
                }
            ],
            "openfda": {
                "brand_name": ["SPIKEVAX"],
                "generic_name": ["COVID-19 VACCINE, MRNA"],
            },
        }
    
    def test_preserves_as_json(self, sample_fda_result, tmp_path):
        """Preserves FDA data as JSON, not plain text."""
        extractor = OpenFDAExtractor()
        
        # Monkey-patch the path functions for testing
        import src.storage.paths as paths
        original_raw = paths.get_raw_path
        original_text = paths.get_text_path
        
        paths.get_raw_path = lambda *args, **kwargs: tmp_path / "raw.json"
        paths.get_text_path = lambda *args, **kwargs: tmp_path / "text.json"
        
        try:
            raw_path, text_path, result = extractor.save_extraction(
                fda_result=sample_fda_result,
                ticker="MRNA",
                published_date=date(2022, 1, 31),
                application_number="BLA761222",
            )
            
            assert result.success == True
            
            # Verify JSON structure preserved
            with open(text_path) as f:
                saved = json.load(f)
            
            assert saved["application_number"] == "BLA761222"
            assert saved["sponsor_name"] == "MODERNATX, INC."
            assert saved["products"][0]["brand_name"] == "SPIKEVAX"
        finally:
            paths.get_raw_path = original_raw
            paths.get_text_path = original_text
    
    def test_extracts_key_fields(self, sample_fda_result):
        """Extracts and preserves key fields."""
        extractor = OpenFDAExtractor()
        extracted = extractor._extract_key_fields(sample_fda_result)
        
        assert extracted["application_number"] == "BLA761222"
        assert extracted["sponsor_name"] == "MODERNATX, INC."
        assert len(extracted["products"]) == 1
        assert extracted["products"][0]["brand_name"] == "SPIKEVAX"
        assert len(extracted["submissions"]) == 1
    
    def test_handles_missing_openfda_section(self):
        """Handles results without openfda section."""
        result_no_openfda = {
            "application_number": "NDA123",
            "sponsor_name": "TEST",
            "products": [],
            "submissions": [],
        }
        
        extractor = OpenFDAExtractor()
        extracted = extractor._extract_key_fields(result_no_openfda)
        
        assert extracted["openfda"] == {}
```

#### Test: Full Pipeline Integration

```python
# tests/test_extraction_pipeline.py

import pytest
from datetime import date, datetime
from pathlib import Path
import tempfile
import json

from src.extraction.pipeline import ExtractionPipeline
from src.storage.csv_index import AnnouncementIndex, ParseStatus

class TestExtractionPipeline:
    """Integration tests for full extraction pipeline."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure."""
        (tmp_path / "raw").mkdir()
        (tmp_path / "text").mkdir()
        (tmp_path / "index").mkdir()
        return tmp_path
    
    @pytest.mark.asyncio
    async def test_dedup_prevents_reprocessing(self, temp_data_dir):
        """Same announcement is not processed twice."""
        # This would need mocking of the actual API calls
        pass
    
    @pytest.mark.asyncio
    async def test_stats_accurate(self, temp_data_dir):
        """Statistics reflect actual state."""
        # This would need mocking of the actual API calls
        pass
```

---

### 3.6 Scripts

#### Run Extraction Pipeline

```python
# scripts/run_extraction.py

#!/usr/bin/env python3
"""
Run the full text extraction pipeline.

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --ticker MRNA
    python scripts/run_extraction.py --source edgar
    python scripts/run_extraction.py --retry-failed
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Run text extraction pipeline")
    parser.add_argument("--ticker", help="Process specific ticker only")
    parser.add_argument("--source", choices=["edgar", "clinicaltrials", "openfda"], 
                       help="Process specific source only")
    parser.add_argument("--retry-failed", action="store_true",
                       help="Retry previously failed extractions")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics only, don't process")
    args = parser.parse_args()
    
    from src.extraction.pipeline import ExtractionPipeline
    
    pipeline = ExtractionPipeline(
        sec_email=os.getenv("SEC_CONTACT_EMAIL"),
        openfda_key=os.getenv("OPENFDA_API_KEY"),
    )
    
    if args.stats:
        stats = pipeline.get_stats()
        print("\n📊 Extraction Statistics")
        print("=" * 40)
        print(f"Total announcements: {stats['total']}")
        print(f"\nBy source:")
        for source, count in stats["by_source"].items():
            print(f"  {source}: {count}")
        print(f"\nBy status:")
        for status, count in stats["by_status"].items():
            print(f"  {status}: {count}")
        return
    
    # TODO: Implement actual processing logic
    print("Running extraction pipeline...")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Deliverables for Phase 3

**Storage Module**:
- [ ] `src/storage/paths.py` - Deterministic path generation
- [ ] `src/storage/csv_index.py` - CSV index management

**Extraction Module**:
- [ ] `src/extraction/edgar_extractor.py` - SEC EDGAR extraction
- [ ] `src/extraction/clinicaltrials_extractor.py` - ClinicalTrials extraction
- [ ] `src/extraction/openfda_extractor.py` - OpenFDA preservation
- [ ] `src/extraction/pipeline.py` - Main extraction pipeline

**Tests**:
- [ ] `tests/test_paths.py` - Path generation tests
- [ ] `tests/test_csv_index.py` - CSV index tests
- [ ] `tests/test_edgar_extraction.py` - EDGAR extraction tests
- [ ] `tests/test_clinicaltrials_extraction.py` - ClinicalTrials tests
- [ ] `tests/test_openfda_extraction.py` - OpenFDA tests
- [ ] `tests/test_extraction_pipeline.py` - Integration tests

**Scripts**:
- [ ] `scripts/run_extraction.py` - Run extraction pipeline

**Dependencies**:
```
# Add to requirements.txt
pymupdf>=1.23.0  # PDF extraction (fitz)
```

### Success Criteria for Phase 3

| Metric | Target |
|--------|--------|
| Test coverage | ≥95% |
| Path generation deterministic | 100% (same input = same output) |
| Deduplication works | No duplicate IDs in CSV |
| HTML extraction | Text extracted, scripts/styles removed |
| PDF extraction | Text extracted or empty on failure |
| Retry logic | 3 attempts before FAILED status |
| Failed records | Have raw file + empty text file |
| CSV valid | Parseable, no corruption |

### Running Phase 3

```bash
# 1. Run all tests
pytest tests/test_paths.py tests/test_csv_index.py tests/test_*_extraction.py -v

# 2. Run extraction pipeline (fetch + extract for all stocks)
python scripts/run_extraction.py --all

# 3. Check statistics
python scripts/run_extraction.py --stats

# 4. Retry failed extractions
python scripts/run_extraction.py --retry-failed

# 5. Process specific ticker
python scripts/run_extraction.py --ticker MRNA
```

---

## Phase 3.5: Post-Announcement Return Calculation

### Objective
Calculate and store the stock price return for each announcement at 30, 60, and 90 calendar days after the announcement date. This enables analysis of how announcements impact stock performance.

**Important**: Returns start from the **next trading day** (T+1) after the announcement, not the announcement day itself. This accounts for after-hours announcements where the first actionable moment is the next trading day's open.

### Design Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| Price point | Adjusted close (accounts for splits/dividends) |
| **Start price** | **Next trading day's close (T+1)** - accounts for after-hours announcements |
| End price (weekend/holiday) | Use closest available trading day |
| Days measurement | Calendar days (not trading days) |
| Return format | Decimal with 2 places (e.g., `15.43` for 15.43%) |
| Recent announcements | Leave as NULL (calculate later when data available) |
| File to update | `data/index/announcements.csv` + `data/index/announcements.parquet` |
| Processing strategy | Batch: fetch all price history per ticker first |
| Missing price data | Fail entire ticker batch, log error |

---

### 3.5.1 New CSV Columns

Add three new columns to `data/index/announcements.csv`:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `return_30d` | float (2 decimals) | % return from T+1 to day 30 | `15.43`, `-8.21`, `NULL` |
| `return_60d` | float (2 decimals) | % return from T+1 to day 60 | `22.50`, `-12.00`, `NULL` |
| `return_90d` | float (2 decimals) | % return from T+1 to day 90 | `45.67`, `-25.33`, `NULL` |

**NULL values indicate**:
- Announcement is too recent (30/60/90 days haven't passed)
- Price data unavailable for the ticker
- Calculation error (logged separately)

---

### 3.5.2 Return Calculation Logic

#### Formula

```
return = ((end_price - start_price) / start_price) * 100
```

Where:
- `start_price` = Adjusted close on **next trading day after announcement (T+1)**
- `end_price` = Adjusted close on announcement date + N days (or closest trading day)

**Why T+1?** Announcements often happen after market hours (earnings calls, FDA decisions, etc.). The first moment an investor can act on the news is the next trading day. Using T+1 gives a realistic measure of achievable returns.

#### Price Lookup Rules

```python
# src/returns/price_lookup.py

from datetime import date, timedelta
from typing import Optional
import bisect

class PriceLookup:
    """
    Efficient price lookup with trading day handling.
    
    Prices are stored as a sorted list of (date, adjusted_close) tuples
    for binary search efficiency.
    
    IMPORTANT: Start price uses NEXT trading day (T+1), not announcement day.
    This accounts for after-hours announcements.
    """
    
    def __init__(self, prices: list[tuple[date, float]]):
        """
        Args:
            prices: List of (date, adjusted_close) sorted by date ascending
        """
        self.dates = [p[0] for p in prices]
        self.prices = {p[0]: p[1] for p in prices}
        
        if not self.dates:
            raise ValueError("Price list cannot be empty")
    
    def get_start_price(self, announcement_date: date) -> Optional[tuple[date, float]]:
        """
        Get price for start of return calculation.
        
        Rules:
        - Use the NEXT trading day after announcement_date (T+1)
        - This accounts for after-hours announcements
        
        Returns:
            Tuple of (actual_date_used, price) or None if not found
        """
        # Find first trading day AFTER announcement_date
        idx = bisect.bisect_right(self.dates, announcement_date)
        
        if idx >= len(self.dates):
            # No trading days after announcement date
            return None
        
        next_trading_day = self.dates[idx]
        return (next_trading_day, self.prices[next_trading_day])
    
    def get_end_price(self, target_date: date) -> Optional[tuple[date, float]]:
        """
        Get price for end of return calculation.
        
        Rules:
        - If target_date is a trading day, use that day's close
        - If weekend/holiday, use CLOSEST trading day (prefer after, then before)
        
        Returns:
            Tuple of (actual_date_used, price) or None if not found
        """
        # Check exact match first
        if target_date in self.prices:
            return (target_date, self.prices[target_date])
        
        # Find insertion point
        idx = bisect.bisect_left(self.dates, target_date)
        
        # Find closest date (prefer after, then before)
        candidates = []
        
        # Date after
        if idx < len(self.dates):
            after_date = self.dates[idx]
            candidates.append((abs((after_date - target_date).days), after_date, "after"))
        
        # Date before
        if idx > 0:
            before_date = self.dates[idx - 1]
            candidates.append((abs((target_date - before_date).days), before_date, "before"))
        
        if not candidates:
            return None
        
        # Sort by distance, prefer "after" for ties
        candidates.sort(key=lambda x: (x[0], x[2] == "before"))
        _, closest_date, _ = candidates[0]
        
        return (closest_date, self.prices[closest_date])
    
    def has_sufficient_data(self, start_date: date, days_forward: int) -> bool:
        """
        Check if we have price data covering the required period.
        
        Args:
            start_date: Announcement date
            days_forward: Number of calendar days forward (30, 60, or 90)
            
        Returns:
            True if data exists for both start and end dates
        """
        end_date = start_date + timedelta(days=days_forward)
        
        # Check start price exists
        if self.get_start_price(start_date) is None:
            return False
        
        # Check end price exists (with some tolerance for weekends)
        # We need data up to at least end_date - 5 days (covers long weekends)
        if not self.dates:
            return False
        
        latest_date = self.dates[-1]
        return latest_date >= end_date - timedelta(days=5)
```

#### Return Calculator

```python
# src/returns/calculator.py

from datetime import date, timedelta
from typing import Optional
from dataclasses import dataclass

from src.returns.price_lookup import PriceLookup

@dataclass
class ReturnResult:
    """Result of a return calculation."""
    return_pct: Optional[float]  # None if cannot calculate
    start_date: Optional[date]
    start_price: Optional[float]
    end_date: Optional[date]
    end_price: Optional[float]
    error: Optional[str] = None

class ReturnCalculator:
    """
    Calculate post-announcement returns.
    
    Example:
        calculator = ReturnCalculator(price_lookup)
        result = calculator.calculate_return(
            announcement_date=date(2025, 1, 15),
            days_forward=30
        )
        print(f"30-day return: {result.return_pct:.2f}%")
    """
    
    def __init__(self, price_lookup: PriceLookup):
        self.prices = price_lookup
    
    def calculate_return(
        self,
        announcement_date: date,
        days_forward: int,
    ) -> ReturnResult:
        """
        Calculate return from announcement date to N days forward.
        
        Args:
            announcement_date: Date of the announcement
            days_forward: Number of calendar days (30, 60, or 90)
            
        Returns:
            ReturnResult with calculated return or error
        """
        # Get start price
        start_result = self.prices.get_start_price(announcement_date)
        if start_result is None:
            return ReturnResult(
                return_pct=None,
                start_date=None,
                start_price=None,
                end_date=None,
                end_price=None,
                error=f"No price data on or before {announcement_date}",
            )
        
        start_date, start_price = start_result
        
        # Calculate target end date
        target_end_date = announcement_date + timedelta(days=days_forward)
        
        # Check if enough time has passed
        today = date.today()
        if target_end_date > today:
            return ReturnResult(
                return_pct=None,
                start_date=start_date,
                start_price=start_price,
                end_date=None,
                end_price=None,
                error=f"Only {(today - announcement_date).days} days have passed, need {days_forward}",
            )
        
        # Get end price
        end_result = self.prices.get_end_price(target_end_date)
        if end_result is None:
            return ReturnResult(
                return_pct=None,
                start_date=start_date,
                start_price=start_price,
                end_date=None,
                end_price=None,
                error=f"No price data around {target_end_date}",
            )
        
        end_date, end_price = end_result
        
        # Calculate return
        return_pct = ((end_price - start_price) / start_price) * 100
        
        return ReturnResult(
            return_pct=round(return_pct, 2),
            start_date=start_date,
            start_price=start_price,
            end_date=end_date,
            end_price=end_price,
        )
    
    def calculate_all_returns(
        self,
        announcement_date: date,
    ) -> dict[int, ReturnResult]:
        """
        Calculate 30, 60, and 90 day returns.
        
        Returns:
            Dict mapping days (30, 60, 90) to ReturnResult
        """
        return {
            30: self.calculate_return(announcement_date, 30),
            60: self.calculate_return(announcement_date, 60),
            90: self.calculate_return(announcement_date, 90),
        }
```

---

### 3.5.3 Batch Processing Pipeline

```python
# src/returns/pipeline.py

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict

from src.clients.eodhd import EODHDClient
from src.returns.price_lookup import PriceLookup
from src.returns.calculator import ReturnCalculator
from src.utils.logging import get_logger

logger = get_logger(__name__)

CSV_PATH = Path("data/index/announcements.csv")

# New columns to add
RETURN_COLUMNS = ["return_30d", "return_60d", "return_90d"]

class ReturnPipeline:
    """
    Batch pipeline for calculating post-announcement returns.
    
    Strategy:
    1. Read all announcements from CSV
    2. Group by ticker
    3. For each ticker:
       a. Fetch full price history once (efficient!)
       b. Calculate returns for all announcements of that ticker
    4. Write updated CSV with new columns
    
    Example:
        pipeline = ReturnPipeline(eodhd_api_key="...")
        stats = await pipeline.run()
        print(f"Calculated {stats['calculated']} returns")
    """
    
    def __init__(self, eodhd_api_key: str):
        self.eodhd = EODHDClient(api_key=eodhd_api_key)
    
    async def run(
        self,
        csv_path: Path = CSV_PATH,
        force_recalculate: bool = False,
    ) -> dict:
        """
        Run the return calculation pipeline.
        
        Args:
            csv_path: Path to announcements CSV
            force_recalculate: If True, recalculate even if returns exist
            
        Returns:
            Stats dict with counts
        """
        stats = {
            "total_announcements": 0,
            "tickers_processed": 0,
            "tickers_failed": 0,
            "returns_calculated": 0,
            "returns_skipped_recent": 0,
            "returns_skipped_existing": 0,
            "returns_failed": 0,
        }
        
        # Read CSV
        rows = self._read_csv(csv_path)
        stats["total_announcements"] = len(rows)
        
        # Group by ticker
        by_ticker = defaultdict(list)
        for i, row in enumerate(rows):
            by_ticker[row["ticker"]].append((i, row))
        
        logger.info(f"Processing {len(by_ticker)} tickers with {len(rows)} total announcements")
        
        # Process each ticker
        for ticker, announcements in by_ticker.items():
            try:
                processed = await self._process_ticker(
                    ticker=ticker,
                    announcements=announcements,
                    rows=rows,
                    force_recalculate=force_recalculate,
                    stats=stats,
                )
                stats["tickers_processed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process ticker {ticker}: {e}")
                stats["tickers_failed"] += 1
                
                # Mark all announcements for this ticker as failed
                for idx, row in announcements:
                    if not row.get("return_30d"):
                        rows[idx]["return_30d"] = ""
                        rows[idx]["return_60d"] = ""
                        rows[idx]["return_90d"] = ""
                        stats["returns_failed"] += 3
        
        # Write updated CSV
        self._write_csv(csv_path, rows)
        
        logger.info(f"Pipeline complete: {stats}")
        return stats
    
    async def _process_ticker(
        self,
        ticker: str,
        announcements: list[tuple[int, dict]],
        rows: list[dict],
        force_recalculate: bool,
        stats: dict,
    ):
        """Process all announcements for a single ticker."""
        
        # Find date range needed
        dates = []
        for idx, row in announcements:
            pub_date = self._parse_date(row["published_at"])
            if pub_date:
                dates.append(pub_date)
        
        if not dates:
            logger.warning(f"No valid dates for ticker {ticker}")
            return
        
        min_date = min(dates)
        max_date = max(dates)
        
        # Fetch price history (with buffer for return calculation)
        # Need data from min_date to max_date + 90 days
        from datetime import timedelta
        start_fetch = min_date - timedelta(days=7)  # Buffer for weekends
        end_fetch = min(max_date + timedelta(days=95), date.today())
        
        logger.info(f"Fetching prices for {ticker}: {start_fetch} to {end_fetch}")
        
        prices = await self.eodhd.get_historical_prices(
            ticker=ticker,
            start_date=start_fetch,
            end_date=end_fetch,
        )
        
        if not prices:
            raise ValueError(f"No price data returned for {ticker}")
        
        # Build price lookup
        price_list = [(p["date"], p["adjusted_close"]) for p in prices]
        price_lookup = PriceLookup(price_list)
        calculator = ReturnCalculator(price_lookup)
        
        # Calculate returns for each announcement
        for idx, row in announcements:
            pub_date = self._parse_date(row["published_at"])
            if not pub_date:
                continue
            
            # Skip if already calculated (unless force)
            if not force_recalculate and row.get("return_30d"):
                stats["returns_skipped_existing"] += 3
                continue
            
            # Calculate all returns
            results = calculator.calculate_all_returns(pub_date)
            
            for days in [30, 60, 90]:
                result = results[days]
                col = f"return_{days}d"
                
                if result.return_pct is not None:
                    rows[idx][col] = f"{result.return_pct:.2f}"
                    stats["returns_calculated"] += 1
                elif "days have passed" in (result.error or ""):
                    rows[idx][col] = ""  # NULL - too recent
                    stats["returns_skipped_recent"] += 1
                else:
                    rows[idx][col] = ""  # NULL - other error
                    stats["returns_failed"] += 1
                    logger.warning(f"Return calc failed for {ticker} {pub_date} {days}d: {result.error}")
    
    def _read_csv(self, path: Path) -> list[dict]:
        """Read CSV and ensure return columns exist."""
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            
            # Add return columns if missing
            for col in RETURN_COLUMNS:
                if col not in fieldnames:
                    fieldnames.append(col)
            
            for row in reader:
                # Initialize return columns if missing
                for col in RETURN_COLUMNS:
                    if col not in row:
                        row[col] = ""
                rows.append(row)
        
        return rows
    
    def _write_csv(self, path: Path, rows: list[dict]):
        """Write CSV with all columns including returns."""
        if not rows:
            return
        
        # Preserve column order, ensure return columns are at end
        fieldnames = [k for k in rows[0].keys() if k not in RETURN_COLUMNS]
        fieldnames.extend(RETURN_COLUMNS)
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date from various formats."""
        if not date_str:
            return None
        
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(date_str.split("T")[0], "%Y-%m-%d").date()
            except ValueError:
                continue
        
        return None
```

---

### 3.5.4 EODHD Price History Integration

Add method to existing EODHD client:

```python
# Add to src/clients/eodhd.py

async def get_historical_prices(
    self,
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Fetch historical adjusted close prices.
    
    Args:
        ticker: Stock ticker (e.g., "MRNA")
        start_date: Start of date range
        end_date: End of date range
        
    Returns:
        List of dicts with keys: date, adjusted_close
        Sorted by date ascending
        
    Raises:
        EODHDAPIError: If API request fails
    """
    url = f"{self.base_url}/eod/{ticker}.US"
    params = {
        "api_token": self.api_key,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "fmt": "json",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    
    # Extract and normalize
    prices = []
    for row in data:
        prices.append({
            "date": datetime.strptime(row["date"], "%Y-%m-%d").date(),
            "adjusted_close": float(row["adjusted_close"]),
        })
    
    # Sort by date
    prices.sort(key=lambda x: x["date"])
    
    return prices
```

---

### 3.5.5 Comprehensive Test Suite

#### Test: Price Lookup

```python
# tests/test_price_lookup.py

import pytest
from datetime import date

from src.returns.price_lookup import PriceLookup

class TestPriceLookup:
    """Tests for price lookup with trading day handling."""
    
    @pytest.fixture
    def sample_prices(self):
        """Sample price data with gaps (weekends)."""
        return [
            (date(2025, 1, 6), 100.0),   # Monday
            (date(2025, 1, 7), 101.0),   # Tuesday
            (date(2025, 1, 8), 102.0),   # Wednesday
            (date(2025, 1, 9), 103.0),   # Thursday
            (date(2025, 1, 10), 104.0),  # Friday
            # Weekend gap: Jan 11-12
            (date(2025, 1, 13), 105.0),  # Monday
            (date(2025, 1, 14), 106.0),  # Tuesday
        ]
    
    def test_get_start_price_returns_next_trading_day(self, sample_prices):
        """Returns NEXT trading day price (T+1) for announcement."""
        lookup = PriceLookup(sample_prices)
        
        # Announcement on Tuesday Jan 7, start price should be Wednesday Jan 8
        result = lookup.get_start_price(date(2025, 1, 7))
        
        assert result == (date(2025, 1, 8), 102.0)  # T+1
    
    def test_get_start_price_weekend_uses_monday(self, sample_prices):
        """Saturday announcement uses Monday (next trading day)."""
        lookup = PriceLookup(sample_prices)
        
        # Saturday Jan 11 announcement, start price should be Monday Jan 13
        result = lookup.get_start_price(date(2025, 1, 11))
        
        assert result == (date(2025, 1, 13), 105.0)  # Next trading day
    
    def test_get_start_price_friday_uses_monday(self, sample_prices):
        """Friday announcement uses Monday (next trading day)."""
        lookup = PriceLookup(sample_prices)
        
        # Friday Jan 10 announcement, start price should be Monday Jan 13
        result = lookup.get_start_price(date(2025, 1, 10))
        
        assert result == (date(2025, 1, 13), 105.0)  # T+1 (skips weekend)
    
    def test_get_start_price_after_last_data_returns_none(self, sample_prices):
        """Returns None if no trading days after announcement."""
        lookup = PriceLookup(sample_prices)
        
        # Last data is Jan 14, announcement after that has no T+1
        result = lookup.get_start_price(date(2025, 1, 14))
        
        assert result is None
    
    def test_get_end_price_exact_date(self, sample_prices):
        """Returns exact date price when available."""
        lookup = PriceLookup(sample_prices)
        
        result = lookup.get_end_price(date(2025, 1, 8))
        
        assert result == (date(2025, 1, 8), 102.0)
    
    def test_get_end_price_weekend_uses_closest(self, sample_prices):
        """Uses closest trading day for weekend."""
        lookup = PriceLookup(sample_prices)
        
        # Saturday Jan 11 - Monday Jan 13 is closer than Friday Jan 10
        result = lookup.get_end_price(date(2025, 1, 11))
        
        # Jan 11 to Jan 13 = 2 days, Jan 11 to Jan 10 = 1 day
        # So Friday is closer
        assert result == (date(2025, 1, 10), 104.0)
    
    def test_get_end_price_sunday_prefers_monday(self, sample_prices):
        """Sunday prefers Monday (1 day) over Friday (2 days)."""
        lookup = PriceLookup(sample_prices)
        
        # Sunday Jan 12 - Monday Jan 13 is 1 day, Friday Jan 10 is 2 days
        result = lookup.get_end_price(date(2025, 1, 12))
        
        assert result == (date(2025, 1, 13), 105.0)
    
    def test_empty_prices_raises(self):
        """Raises error for empty price list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PriceLookup([])
    
    def test_has_sufficient_data_true(self, sample_prices):
        """Returns True when data covers required period."""
        lookup = PriceLookup(sample_prices)
        
        # Jan 6 + 7 days = Jan 13, we have data until Jan 14
        assert lookup.has_sufficient_data(date(2025, 1, 6), 7) == True
    
    def test_has_sufficient_data_false(self, sample_prices):
        """Returns False when data doesn't cover period."""
        lookup = PriceLookup(sample_prices)
        
        # Jan 6 + 30 days = Feb 5, we only have data until Jan 14
        assert lookup.has_sufficient_data(date(2025, 1, 6), 30) == False


class TestPriceLookupEdgeCases:
    """Edge case tests for price lookup."""
    
    def test_single_day_data(self):
        """Works with single day of data."""
        lookup = PriceLookup([(date(2025, 1, 6), 100.0)])
        
        assert lookup.get_start_price(date(2025, 1, 6)) == (date(2025, 1, 6), 100.0)
        assert lookup.get_start_price(date(2025, 1, 7)) == (date(2025, 1, 6), 100.0)
    
    def test_holiday_gap(self):
        """Handles multi-day holiday gaps."""
        # Christmas week gap
        prices = [
            (date(2024, 12, 24), 100.0),  # Christmas Eve (Tuesday)
            # Dec 25 closed (Christmas)
            (date(2024, 12, 26), 101.0),  # Thursday
            (date(2024, 12, 27), 102.0),  # Friday
            # Weekend
            (date(2024, 12, 30), 103.0),  # Monday
        ]
        lookup = PriceLookup(prices)
        
        # Christmas day should use Dec 24
        result = lookup.get_start_price(date(2024, 12, 25))
        assert result == (date(2024, 12, 24), 100.0)
```

#### Test: Return Calculator

```python
# tests/test_return_calculator.py

import pytest
from datetime import date
from freezegun import freeze_time

from src.returns.price_lookup import PriceLookup
from src.returns.calculator import ReturnCalculator, ReturnResult

class TestReturnCalculator:
    """Tests for return calculation."""
    
    @pytest.fixture
    def price_lookup(self):
        """Price data spanning 100 days."""
        prices = []
        start = date(2025, 1, 1)
        price = 100.0
        
        for i in range(150):
            d = date(2025, 1, 1 + i) if i < 31 else date(2025, 1, 1) + timedelta(days=i)
            # Skip weekends
            if d.weekday() < 5:  # Mon-Fri
                prices.append((d, price))
                price += 0.5  # Gradual increase
        
        return PriceLookup(prices)
    
    @pytest.fixture
    def calculator(self, price_lookup):
        return ReturnCalculator(price_lookup)
    
    @freeze_time("2025-06-01")  # Ensure enough time has passed
    def test_calculate_positive_return(self):
        """Calculates positive return correctly."""
        prices = [
            (date(2025, 1, 2), 100.0),
            (date(2025, 1, 3), 101.0),
            (date(2025, 2, 1), 115.0),  # 30 days later
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        assert result.return_pct == 15.0
        assert result.start_price == 100.0
        assert result.end_price == 115.0
    
    @freeze_time("2025-06-01")
    def test_calculate_negative_return(self):
        """Calculates negative return correctly."""
        prices = [
            (date(2025, 1, 2), 100.0),
            (date(2025, 2, 1), 85.0),  # 30 days later, dropped
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        assert result.return_pct == -15.0
    
    @freeze_time("2025-06-01")
    def test_rounds_to_two_decimals(self):
        """Returns are rounded to 2 decimal places."""
        prices = [
            (date(2025, 1, 2), 100.0),
            (date(2025, 2, 1), 115.567),
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        # (115.567 - 100) / 100 * 100 = 15.567, rounded to 15.57
        assert result.return_pct == 15.57
    
    @freeze_time("2025-01-20")  # Only 18 days after announcement
    def test_recent_announcement_returns_null(self):
        """Returns None for announcements too recent."""
        prices = [
            (date(2025, 1, 2), 100.0),
            (date(2025, 1, 3), 101.0),
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        assert result.return_pct is None
        assert "days have passed" in result.error
    
    @freeze_time("2025-06-01")
    def test_weekend_announcement_uses_monday(self):
        """Weekend announcement uses Monday's close as start (T+1)."""
        prices = [
            (date(2025, 1, 3), 100.0),   # Friday
            # Weekend
            (date(2025, 1, 6), 101.0),   # Monday
            (date(2025, 2, 3), 120.0),   # ~30 days later
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        # Saturday announcement - start price should be Monday (T+1)
        result = calculator.calculate_return(date(2025, 1, 4), 30)
        
        assert result.start_date == date(2025, 1, 6)  # Monday (T+1)
        assert result.start_price == 101.0
    
    @freeze_time("2025-06-01")
    def test_weekday_announcement_uses_next_day(self):
        """Weekday announcement uses next trading day's close (T+1)."""
        prices = [
            (date(2025, 1, 6), 100.0),   # Monday
            (date(2025, 1, 7), 105.0),   # Tuesday
            (date(2025, 2, 5), 120.0),   # ~30 days later
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        # Monday announcement - start price should be Tuesday (T+1)
        result = calculator.calculate_return(date(2025, 1, 6), 30)
        
        assert result.start_date == date(2025, 1, 7)  # Tuesday (T+1)
        assert result.start_price == 105.0
    
    @freeze_time("2025-06-01")
    def test_calculate_all_returns(self):
        """Calculates 30, 60, 90 day returns together."""
        prices = []
        for i in range(120):
            d = date(2025, 1, 1) + timedelta(days=i)
            if d.weekday() < 5:
                prices.append((d, 100.0 + i))
        
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        results = calculator.calculate_all_returns(date(2025, 1, 2))
        
        assert 30 in results
        assert 60 in results
        assert 90 in results
        assert all(r.return_pct is not None for r in results.values())


class TestReturnCalculatorErrors:
    """Error handling tests for return calculator."""
    
    @freeze_time("2025-06-01")
    def test_no_start_price_data(self):
        """Returns error when no start price available."""
        prices = [
            (date(2025, 2, 1), 100.0),  # Data starts after announcement
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        assert result.return_pct is None
        assert "No price data" in result.error
    
    @freeze_time("2025-06-01")
    def test_no_end_price_data(self):
        """Returns error when no end price available."""
        prices = [
            (date(2025, 1, 2), 100.0),
            (date(2025, 1, 3), 101.0),
            # No data for 30 days later
        ]
        lookup = PriceLookup(prices)
        calculator = ReturnCalculator(lookup)
        
        result = calculator.calculate_return(date(2025, 1, 2), 30)
        
        assert result.return_pct is None
        assert result.error is not None
```

#### Test: Return Pipeline Integration

```python
# tests/test_return_pipeline.py

import pytest
from datetime import date
from pathlib import Path
import csv
from unittest.mock import AsyncMock, patch

from src.returns.pipeline import ReturnPipeline

class TestReturnPipeline:
    """Integration tests for return pipeline."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample announcements CSV."""
        csv_path = tmp_path / "announcements.csv"
        
        rows = [
            {
                "id": "abc123",
                "ticker": "MRNA",
                "source": "edgar",
                "published_at": "2025-01-15",
                "title": "8-K Filing",
            },
            {
                "id": "def456",
                "ticker": "MRNA",
                "source": "edgar",
                "published_at": "2025-02-01",
                "title": "Another Filing",
            },
            {
                "id": "ghi789",
                "ticker": "PFE",
                "source": "edgar",
                "published_at": "2025-01-20",
                "title": "PFE Filing",
            },
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        return csv_path
    
    @pytest.fixture
    def mock_prices(self):
        """Mock price data."""
        from datetime import timedelta
        prices = []
        for i in range(150):
            d = date(2025, 1, 1) + timedelta(days=i)
            if d.weekday() < 5:
                prices.append({
                    "date": d,
                    "adjusted_close": 100.0 + i * 0.5,
                })
        return prices
    
    @pytest.mark.asyncio
    async def test_adds_return_columns(self, sample_csv, mock_prices):
        """Adds return columns to CSV."""
        pipeline = ReturnPipeline(eodhd_api_key="test")
        
        with patch.object(pipeline.eodhd, 'get_historical_prices', new_callable=AsyncMock) as mock:
            mock.return_value = mock_prices
            
            with patch('src.returns.pipeline.date') as mock_date:
                mock_date.today.return_value = date(2025, 6, 1)
                mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
                
                stats = await pipeline.run(csv_path=sample_csv)
        
        # Read updated CSV
        with open(sample_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Check columns exist
        assert "return_30d" in rows[0]
        assert "return_60d" in rows[0]
        assert "return_90d" in rows[0]
    
    @pytest.mark.asyncio
    async def test_batch_processing_per_ticker(self, sample_csv, mock_prices):
        """Fetches prices once per ticker, not per announcement."""
        pipeline = ReturnPipeline(eodhd_api_key="test")
        
        with patch.object(pipeline.eodhd, 'get_historical_prices', new_callable=AsyncMock) as mock:
            mock.return_value = mock_prices
            
            with patch('src.returns.pipeline.date') as mock_date:
                mock_date.today.return_value = date(2025, 6, 1)
                mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
                
                await pipeline.run(csv_path=sample_csv)
        
        # Should be called twice: once for MRNA, once for PFE
        assert mock.call_count == 2
    
    @pytest.mark.asyncio
    async def test_preserves_existing_data(self, sample_csv, mock_prices):
        """Doesn't overwrite existing return values unless forced."""
        # Pre-populate one return value
        with open(sample_csv) as f:
            rows = list(csv.DictReader(f))
        
        rows[0]["return_30d"] = "10.00"
        
        fieldnames = list(rows[0].keys()) + ["return_30d", "return_60d", "return_90d"]
        with open(sample_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        pipeline = ReturnPipeline(eodhd_api_key="test")
        
        with patch.object(pipeline.eodhd, 'get_historical_prices', new_callable=AsyncMock) as mock:
            mock.return_value = mock_prices
            
            stats = await pipeline.run(csv_path=sample_csv, force_recalculate=False)
        
        # Read result
        with open(sample_csv) as f:
            result_rows = list(csv.DictReader(f))
        
        # First row should keep original value
        assert result_rows[0]["return_30d"] == "10.00"
    
    @pytest.mark.asyncio
    async def test_handles_missing_price_data(self, sample_csv):
        """Fails entire ticker batch when price data missing."""
        pipeline = ReturnPipeline(eodhd_api_key="test")
        
        with patch.object(pipeline.eodhd, 'get_historical_prices', new_callable=AsyncMock) as mock:
            mock.return_value = []  # No prices
            
            stats = await pipeline.run(csv_path=sample_csv)
        
        assert stats["tickers_failed"] == 2  # MRNA and PFE both fail
    
    @pytest.mark.asyncio
    async def test_stats_accuracy(self, sample_csv, mock_prices):
        """Stats accurately reflect processing."""
        pipeline = ReturnPipeline(eodhd_api_key="test")
        
        with patch.object(pipeline.eodhd, 'get_historical_prices', new_callable=AsyncMock) as mock:
            mock.return_value = mock_prices
            
            with patch('src.returns.pipeline.date') as mock_date:
                mock_date.today.return_value = date(2025, 6, 1)
                mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
                
                stats = await pipeline.run(csv_path=sample_csv)
        
        assert stats["total_announcements"] == 3
        assert stats["tickers_processed"] == 2
        assert stats["returns_calculated"] > 0
```

---

### 3.5.6 Script: Calculate Returns

```python
# scripts/calculate_returns.py

#!/usr/bin/env python3
"""
Calculate post-announcement returns for all announcements.

Usage:
    python scripts/calculate_returns.py
    python scripts/calculate_returns.py --force       # Recalculate all
    python scripts/calculate_returns.py --stats       # Show stats only
    python scripts/calculate_returns.py --ticker MRNA # Process one ticker
"""

import asyncio
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Calculate post-announcement returns")
    parser.add_argument("--force", action="store_true", help="Recalculate even if returns exist")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--ticker", help="Process specific ticker only")
    parser.add_argument("--csv", default="data/index/announcements.csv", help="Path to CSV")
    args = parser.parse_args()
    
    from src.returns.pipeline import ReturnPipeline
    from src.storage.csv_index import AnnouncementIndex
    
    csv_path = Path(args.csv)
    
    if args.stats:
        # Just show current state
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        
        total = len(rows)
        with_30d = sum(1 for r in rows if r.get("return_30d"))
        with_60d = sum(1 for r in rows if r.get("return_60d"))
        with_90d = sum(1 for r in rows if r.get("return_90d"))
        
        print("\n📊 Return Calculation Statistics")
        print("=" * 40)
        print(f"Total announcements: {total}")
        print(f"With 30d return: {with_30d} ({with_30d/total*100:.1f}%)")
        print(f"With 60d return: {with_60d} ({with_60d/total*100:.1f}%)")
        print(f"With 90d return: {with_90d} ({with_90d/total*100:.1f}%)")
        return
    
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("❌ EODHD_API_KEY not set in .env")
        return
    
    print(f"📈 Calculating returns for announcements in {csv_path}")
    
    pipeline = ReturnPipeline(eodhd_api_key=api_key)
    
    stats = await pipeline.run(
        csv_path=csv_path,
        force_recalculate=args.force,
    )
    
    print("\n✅ Complete!")
    print(f"   Tickers processed: {stats['tickers_processed']}")
    print(f"   Tickers failed: {stats['tickers_failed']}")
    print(f"   Returns calculated: {stats['returns_calculated']}")
    print(f"   Skipped (recent): {stats['returns_skipped_recent']}")
    print(f"   Skipped (existing): {stats['returns_skipped_existing']}")
    print(f"   Failed: {stats['returns_failed']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3.5.5 Parquet Dataset (ML-Ready)

Generate and maintain a Parquet file that mirrors the CSV but includes the full extracted text content. This enables ML/NLP analysis without needing to read individual text files.

#### Design Decisions

| Decision | Choice |
|----------|--------|
| Location | `data/index/announcements.parquet` (same directory as CSV) |
| Sync strategy | Always keep in sync (auto-update when CSV changes) |
| Records included | All records (including parse_status=FAILED) |
| Text column name | `raw_text` |
| Text size limit | No limit (store full text) |

#### Parquet Schema

The Parquet file contains all CSV columns PLUS:

| Column | Type | Description |
|--------|------|-------------|
| `raw_text` | string | Full extracted text from `text_path` file (empty string if FAILED or missing) |

```python
# src/storage/parquet_sync.py

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional

CSV_PATH = Path("data/index/announcements.csv")
PARQUET_PATH = Path("data/index/announcements.parquet")
TEXT_BASE_PATH = Path("data/text")

class ParquetSync:
    """
    Maintains a Parquet mirror of the announcements CSV with full text content.
    
    The Parquet file is always kept in sync with the CSV and includes
    the full extracted text for each announcement, making it suitable
    for ML/NLP analysis.
    
    Example:
        sync = ParquetSync()
        sync.update()  # Regenerate Parquet from CSV + text files
        
        # Load for analysis
        df = pd.read_parquet("data/index/announcements.parquet")
        print(df[["ticker", "title", "raw_text", "return_30d"]].head())
    """
    
    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        parquet_path: Path = PARQUET_PATH,
        text_base_path: Path = TEXT_BASE_PATH,
    ):
        self.csv_path = csv_path
        self.parquet_path = parquet_path
        self.text_base_path = text_base_path
    
    def update(self) -> dict:
        """
        Regenerate Parquet file from CSV and text files.
        
        Returns:
            Stats dict with counts
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        stats = {
            "total_records": len(df),
            "text_loaded": 0,
            "text_missing": 0,
            "text_empty": 0,
        }
        
        # Load text content for each record
        raw_texts = []
        
        for _, row in df.iterrows():
            text_path = row.get("text_path", "")
            
            if not text_path:
                raw_texts.append("")
                stats["text_missing"] += 1
                continue
            
            # Handle relative paths
            full_path = Path(text_path)
            if not full_path.is_absolute():
                full_path = Path("data") / text_path.replace("data/", "", 1)
            
            try:
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    raw_texts.append(content)
                    
                    if content.strip():
                        stats["text_loaded"] += 1
                    else:
                        stats["text_empty"] += 1
                else:
                    raw_texts.append("")
                    stats["text_missing"] += 1
            except Exception as e:
                raw_texts.append("")
                stats["text_missing"] += 1
        
        # Add raw_text column
        df["raw_text"] = raw_texts
        
        # Convert return columns to float (handle empty strings)
        for col in ["return_30d", "return_60d", "return_90d"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Write Parquet
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, index=False, engine="pyarrow")
        
        stats["parquet_size_mb"] = round(self.parquet_path.stat().st_size / 1024 / 1024, 2)
        
        return stats
    
    def get_dataframe(self) -> pd.DataFrame:
        """Load Parquet as DataFrame."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}. Run update() first.")
        return pd.read_parquet(self.parquet_path)
    
    def get_ml_dataset(
        self,
        min_text_length: int = 100,
        require_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Get filtered dataset suitable for ML training.
        
        Args:
            min_text_length: Minimum characters in raw_text
            require_returns: Only include records with return_30d calculated
            
        Returns:
            Filtered DataFrame
        """
        df = self.get_dataframe()
        
        # Filter by text length
        df = df[df["raw_text"].str.len() >= min_text_length]
        
        # Filter by returns
        if require_returns:
            df = df[df["return_30d"].notna()]
        
        return df
```

#### Script to Update Parquet

```python
# scripts/update_parquet.py

#!/usr/bin/env python3
"""
Update Parquet dataset from CSV and text files.

Usage:
    python scripts/update_parquet.py
    python scripts/update_parquet.py --stats  # Show stats only
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Update Parquet dataset")
    parser.add_argument("--stats", action="store_true", help="Show current stats only")
    parser.add_argument("--ml-ready", action="store_true", help="Show ML-ready dataset stats")
    args = parser.parse_args()
    
    from src.storage.parquet_sync import ParquetSync
    import pandas as pd
    
    sync = ParquetSync()
    
    if args.stats:
        if not Path("data/index/announcements.parquet").exists():
            print("❌ Parquet file not found. Run without --stats first.")
            return
        
        df = sync.get_dataframe()
        print("\n📊 Parquet Dataset Statistics")
        print("=" * 40)
        print(f"Total records: {len(df)}")
        print(f"With text: {(df['raw_text'].str.len() > 0).sum()}")
        print(f"With 30d return: {df['return_30d'].notna().sum()}")
        print(f"Unique tickers: {df['ticker'].nunique()}")
        print(f"File size: {Path('data/index/announcements.parquet').stat().st_size / 1024 / 1024:.2f} MB")
        
        if args.ml_ready:
            ml_df = sync.get_ml_dataset()
            print(f"\n🤖 ML-Ready Dataset: {len(ml_df)} records")
        return
    
    print("📦 Updating Parquet dataset...")
    stats = sync.update()
    
    print("\n✅ Parquet updated!")
    print(f"   Total records: {stats['total_records']}")
    print(f"   Text loaded: {stats['text_loaded']}")
    print(f"   Text empty: {stats['text_empty']}")
    print(f"   Text missing: {stats['text_missing']}")
    print(f"   File size: {stats['parquet_size_mb']} MB")
    print(f"\n📁 Output: data/index/announcements.parquet")

if __name__ == "__main__":
    main()
```

#### Auto-Sync Integration

The Parquet file should be automatically updated after:
1. Running `run_extraction.py` (Phase 3)
2. Running `calculate_returns.py` (Phase 3.5)

Add to the end of both scripts:

```python
# At end of run_extraction.py and calculate_returns.py
from src.storage.parquet_sync import ParquetSync

print("\n📦 Syncing Parquet dataset...")
sync = ParquetSync()
stats = sync.update()
print(f"   Parquet updated: {stats['text_loaded']} records with text")
```

#### Test Cases for Parquet Sync

```python
# tests/test_parquet_sync.py

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.storage.parquet_sync import ParquetSync

class TestParquetSync:
    """Tests for Parquet sync functionality."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temp data structure."""
        # Create CSV
        csv_path = tmp_path / "index" / "announcements.csv"
        csv_path.parent.mkdir(parents=True)
        
        csv_content = """id,ticker,source,published_at,title,text_path,parse_status,return_30d
abc123,MRNA,edgar,2025-01-15,8-K Filing,text/edgar/2025-01-15/MRNA/abc123.txt,OK,15.43
def456,MRNA,edgar,2025-02-01,Another Filing,text/edgar/2025-02-01/MRNA/def456.txt,OK,
ghi789,PFE,edgar,2025-01-20,PFE Filing,,FAILED,"""
        
        csv_path.write_text(csv_content)
        
        # Create text files
        text_dir = tmp_path / "text" / "edgar" / "2025-01-15" / "MRNA"
        text_dir.mkdir(parents=True)
        (text_dir / "abc123.txt").write_text("Full text content of the filing...")
        
        text_dir2 = tmp_path / "text" / "edgar" / "2025-02-01" / "MRNA"
        text_dir2.mkdir(parents=True)
        (text_dir2 / "def456.txt").write_text("Another filing text content")
        
        return tmp_path
    
    def test_update_creates_parquet(self, temp_data_dir):
        """Creates Parquet file from CSV."""
        sync = ParquetSync(
            csv_path=temp_data_dir / "index" / "announcements.csv",
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
            text_base_path=temp_data_dir / "text",
        )
        
        stats = sync.update()
        
        assert (temp_data_dir / "index" / "announcements.parquet").exists()
        assert stats["total_records"] == 3
    
    def test_parquet_has_raw_text_column(self, temp_data_dir):
        """Parquet includes raw_text column."""
        sync = ParquetSync(
            csv_path=temp_data_dir / "index" / "announcements.csv",
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
            text_base_path=temp_data_dir / "text",
        )
        sync.update()
        
        df = sync.get_dataframe()
        
        assert "raw_text" in df.columns
        assert df.loc[df["id"] == "abc123", "raw_text"].iloc[0] == "Full text content of the filing..."
    
    def test_handles_missing_text_files(self, temp_data_dir):
        """Gracefully handles missing text files."""
        sync = ParquetSync(
            csv_path=temp_data_dir / "index" / "announcements.csv",
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
            text_base_path=temp_data_dir / "text",
        )
        
        stats = sync.update()
        
        # ghi789 has no text_path, should be empty
        df = sync.get_dataframe()
        assert df.loc[df["id"] == "ghi789", "raw_text"].iloc[0] == ""
        assert stats["text_missing"] >= 1
    
    def test_get_ml_dataset_filters(self, temp_data_dir):
        """ML dataset filters by text length and returns."""
        sync = ParquetSync(
            csv_path=temp_data_dir / "index" / "announcements.csv",
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
            text_base_path=temp_data_dir / "text",
        )
        sync.update()
        
        ml_df = sync.get_ml_dataset(min_text_length=10, require_returns=True)
        
        # Only abc123 has both text and return
        assert len(ml_df) == 1
        assert ml_df.iloc[0]["id"] == "abc123"
    
    def test_return_columns_are_numeric(self, temp_data_dir):
        """Return columns are converted to numeric."""
        sync = ParquetSync(
            csv_path=temp_data_dir / "index" / "announcements.csv",
            parquet_path=temp_data_dir / "index" / "announcements.parquet",
            text_base_path=temp_data_dir / "text",
        )
        sync.update()
        
        df = sync.get_dataframe()
        
        assert df["return_30d"].dtype == float
        assert df.loc[df["id"] == "abc123", "return_30d"].iloc[0] == 15.43
        assert pd.isna(df.loc[df["id"] == "def456", "return_30d"].iloc[0])
```

#### Dependencies

```
# Add to requirements.txt
pyarrow>=14.0.0
```

---

### Deliverables for Phase 3.5

**Source Files**:
- [ ] `src/returns/price_lookup.py` - Price lookup with T+1 trading day handling
- [ ] `src/returns/calculator.py` - Return calculation logic
- [ ] `src/returns/pipeline.py` - Batch processing pipeline
- [ ] `src/storage/parquet_sync.py` - Parquet mirror with raw_text
- [ ] Update `src/clients/eodhd.py` - Add `get_historical_prices()` method

**Test Files**:
- [ ] `tests/test_price_lookup.py` - Price lookup tests (~15 tests)
- [ ] `tests/test_return_calculator.py` - Calculator tests (~12 tests)
- [ ] `tests/test_return_pipeline.py` - Pipeline integration tests (~8 tests)
- [ ] `tests/test_parquet_sync.py` - Parquet sync tests (~6 tests)

**Scripts**:
- [ ] `scripts/calculate_returns.py` - Run return calculation
- [ ] `scripts/update_parquet.py` - Update Parquet dataset

**Output Files**:
- [ ] `data/index/announcements.csv` - Updated with return columns
- [ ] `data/index/announcements.parquet` - ML-ready dataset with raw_text

### Success Criteria for Phase 3.5

| Metric | Target |
|--------|--------|
| Test coverage | ≥95% |
| **Start price** | **Uses T+1 (next trading day)** |
| Weekend/holiday handling | Correct price selection |
| Recent announcements | NULL (not error) |
| Return precision | 2 decimal places |
| Batch efficiency | 1 API call per ticker |
| Idempotent | Re-running doesn't duplicate |
| Parquet in sync | Auto-updates after CSV changes |
| Parquet has raw_text | Full text content loaded |

### Running Phase 3.5

```bash
# 1. Run all tests
pytest tests/test_price_lookup.py tests/test_return_calculator.py tests/test_return_pipeline.py tests/test_parquet_sync.py -v

# 2. Check current state
python scripts/calculate_returns.py --stats

# 3. Calculate returns (auto-syncs Parquet)
python scripts/calculate_returns.py

# 4. Force recalculate all
python scripts/calculate_returns.py --force

# 5. Manually update Parquet (if needed)
python scripts/update_parquet.py

# 6. Check Parquet dataset stats
python scripts/update_parquet.py --stats --ml-ready

# 7. Verify results
head -5 data/index/announcements.csv
python -c "import pandas as pd; print(pd.read_parquet('data/index/announcements.parquet').head())"
```

---

## Master Pipeline (Unified Workflow)

After implementing Phases 1-3.5, use the master pipeline script for daily operations:

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    scripts/run_pipeline.py                          │
│                    (Master Pipeline Script)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Phase 1       │      │ Phase 3       │      │ Phase 3.5     │
│ Stock Fetch   │  →   │ Extraction    │  →   │ Returns +     │
│               │      │               │      │ Parquet Sync  │
│ fetch_stock   │      │ run_          │      │ calculate_    │
│ _list.py      │      │ extraction.py │      │ returns.py    │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ data/         │      │ data/index/   │      │ data/index/   │
│ stocks.csv    │      │ announcements │      │ announcements │
│               │      │ .csv          │      │ .csv + returns│
│               │      │               │      │               │
│               │      │ data/raw/     │      │ announcements │
│               │      │ data/text/    │      │ .parquet      │
│               │      │               │      │ (+ raw_text)  │
└───────────────┘      └───────────────┘      └───────────────┘
                                │
                                ▼
                       ┌───────────────┐
                       │ Phase 2.5     │
                       │ Charts        │
                       │               │
                       │ generate_     │
                       │ announcement_ │
                       │ charts.py     │
                       └───────────────┘
                                │
                                ▼
                       ┌───────────────┐
                       │ charts/       │
                       │ index.html    │
                       │ {ticker}.html │
                       │ {ticker}.png  │
                       └───────────────┘
```

### Master Pipeline Script Usage

```bash
# Run complete pipeline (stocks → extraction → returns → charts)
python scripts/run_pipeline.py

# Skip stock fetch (use existing data/stocks.csv)
python scripts/run_pipeline.py --no-stocks

# Just fetch data, skip charts
python scripts/run_pipeline.py --no-charts

# Skip return calculation
python scripts/run_pipeline.py --no-returns

# Limit announcements per source (for testing)
python scripts/run_pipeline.py --limit 10

# Verbose output
python scripts/run_pipeline.py --verbose
```

### Master Pipeline Implementation

```python
# scripts/run_pipeline.py

#!/usr/bin/env python3
"""
Master pipeline script - runs all phases in sequence.

Usage:
    python scripts/run_pipeline.py              # Run everything
    python scripts/run_pipeline.py --no-stocks  # Skip stock fetch
    python scripts/run_pipeline.py --no-charts  # Skip chart generation
    python scripts/run_pipeline.py --no-returns # Skip return calculation
    python scripts/run_pipeline.py --limit 10   # Limit per source
"""

import asyncio
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with code {result.returncode}")
        return False
    
    print(f"✅ {description} complete")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run complete biopharma pipeline")
    parser.add_argument("--no-stocks", action="store_true", help="Skip stock fetch")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-returns", action="store_true", help="Skip return calculation")
    parser.add_argument("--limit", type=int, help="Limit announcements per source")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    python = sys.executable
    
    # Phase 1: Stock fetch
    if not args.no_stocks:
        if not run_command(
            [python, "scripts/fetch_stock_list.py"],
            "Phase 1: Fetching stock universe"
        ):
            return 1
    
    # Phase 3: Extraction
    extraction_cmd = [python, "scripts/run_extraction.py", "--all"]
    if args.limit:
        extraction_cmd.extend(["--limit", str(args.limit)])
    
    if not run_command(extraction_cmd, "Phase 3: Extracting announcements"):
        return 1
    
    # Phase 3.5: Returns
    if not args.no_returns:
        if not run_command(
            [python, "scripts/calculate_returns.py"],
            "Phase 3.5: Calculating returns"
        ):
            print("⚠️ Return calculation failed, continuing...")
    
    # Phase 2.5: Charts
    if not args.no_charts:
        if not run_command(
            [python, "scripts/generate_announcement_charts.py"],
            "Phase 2.5: Generating charts"
        ):
            print("⚠️ Chart generation failed, continuing...")
    
    print(f"\n{'='*60}")
    print("🎉 Pipeline complete!")
    print(f"{'='*60}")
    print("\nOutputs:")
    print("  📊 Stocks:        data/stocks.csv")
    print("  📋 Announcements: data/index/announcements.csv")
    print("  📁 Raw files:     data/raw/")
    print("  📄 Text files:    data/text/")
    print("  📈 Charts:        charts/index.html")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Deprecated Scripts

The following scripts are **deprecated** and should not be used:

| Deprecated Script | Replacement |
|-------------------|-------------|
| `scripts/fetch_all_announcements.py` | `scripts/run_extraction.py --all` |
| `scripts/test_fetch_announcements.py` | `scripts/run_extraction.py --limit 5` |

These scripts display deprecation warnings and point to the new workflow.

### Daily Operations (Cron)

For daily automated runs:

```bash
# Add to crontab: crontab -e
# Run daily at 6 AM ET
0 6 * * * cd /path/to/biopharma-monitor && /path/to/venv/bin/python scripts/run_pipeline.py >> logs/pipeline.log 2>&1
```

---

## Phase 4: Database Storage (MySQL)

### Objective
For each stock in our universe, systematically fetch announcements from all APIs and store them in a unified format.

### Unified Announcement Schema

```sql
CREATE TABLE announcements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    source ENUM('edgar', 'clinicaltrials', 'openfda', 'ir_scrape', 'fda_scrape') NOT NULL,
    source_id VARCHAR(100),        -- Original ID from source (e.g., accession number, NCT ID)
    announcement_date DATE NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT,                  -- Full text or summary
    url VARCHAR(1000),             -- Link to original
    category ENUM('earnings', 'trial_start', 'trial_update', 'trial_results', 
                  'trial_terminated', 'fda_approval', 'fda_rejection', 'fda_submission',
                  'partnership', 'financing', 'executive', 'safety', 'other') DEFAULT 'other',
    sentiment ENUM('positive', 'negative', 'neutral'),  -- AI-generated (Phase 5)
    ai_summary TEXT,               -- AI-generated summary (Phase 5)
    raw_data JSON,                 -- Original API response
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE,
    UNIQUE KEY unique_source_id (source, source_id),
    INDEX idx_ticker_date (ticker, announcement_date DESC),
    INDEX idx_category (category),
    INDEX idx_source (source),
    INDEX idx_announcement_date (announcement_date DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```
```

### Category Taxonomy

| Category | Description | Sources |
|----------|-------------|---------|
| `earnings` | Financial results, revenue reports | EDGAR 8-K Item 2.02 |
| `trial_start` | New clinical trial initiated | ClinicalTrials.gov |
| `trial_update` | Status change, enrollment update | ClinicalTrials.gov |
| `trial_results` | Results posted, top-line data | ClinicalTrials.gov, EDGAR |
| `trial_terminated` | Trial stopped early | ClinicalTrials.gov |
| `fda_approval` | Drug approved | OpenFDA, FDA CDER |
| `fda_rejection` | CRL, refusal to file | EDGAR 8-K, IR |
| `fda_submission` | NDA/BLA submitted | EDGAR 8-K, IR |
| `partnership` | Licensing deal, collaboration | EDGAR 8-K Item 1.01 |
| `financing` | Offering, equity raise | EDGAR 8-K |
| `executive` | C-suite changes | EDGAR 8-K Item 5.02 |
| `safety` | Adverse events, clinical hold | OpenFDA, EDGAR |
| `other` | Uncategorized | Any |

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SCHEDULER (hourly)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FETCH ORCHESTRATOR                        │
│  - Iterates through stock universe                          │
│  - Calls each data source client                            │
│  - Handles failures per-stock (doesn't stop pipeline)       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  EDGAR Fetch  │    │  CT.gov Fetch │    │ OpenFDA Fetch │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    NORMALIZER                                │
│  - Converts each source format to unified schema            │
│  - Deduplicates against existing announcements              │
│  - Assigns preliminary category                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE                                   │
│  - Inserts new announcements                                │
│  - Updates existing if content changed                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI ENRICHMENT (async)                     │
│  - Summarizes content                                       │
│  - Assigns sentiment                                        │
│  - Refines category                                         │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Steps

#### Step 3.1: Build Normalizers
```python
class AnnouncementNormalizer:
    """
    Convert source-specific data to unified Announcement model.
    
    Each source has a normalizer method:
    - normalize_edgar(filing) -> Announcement
    - normalize_clinicaltrials(study) -> Announcement  
    - normalize_openfda(record) -> Announcement
    """
```

#### Step 3.2: Build Orchestrator
```python
class FetchOrchestrator:
    """
    Coordinates fetching across all sources for all stocks.
    
    Features:
    - Configurable parallelism (respect rate limits)
    - Per-stock error isolation
    - Progress tracking
    - Resume from failure
    - Incremental mode (only fetch since last run)
    """
```

#### Step 3.3: Implement Deduplication
```python
def is_duplicate(announcement: Announcement) -> bool:
    """
    Check if announcement already exists.
    Uses (source, source_id) as unique key.
    """
```

#### Step 3.4: Initial Backfill
```python
def backfill(days: int = 365):
    """
    Populate database with 1 year of historical announcements.
    
    Strategy:
    - Fetch last 365 days of data from each source
    - Process in weekly batches to avoid memory issues
    - Implement checkpointing to resume on failure
    - Log progress and errors
    - Estimated time: 2-4 hours for full backfill
    
    Note: Some sources may have rate limits that extend this.
    SEC EDGAR: ~10 req/sec = manageable
    ClinicalTrials: May need overnight run for full history
    """
```

### Deliverables for Phase 3
- [ ] `src/pipeline/normalizers.py` - Source-specific normalizers
- [ ] `src/pipeline/orchestrator.py` - Fetch orchestrator
- [ ] `src/pipeline/storage.py` - Database operations
- [ ] `src/models/announcement.py` - Announcement data model
- [ ] `scripts/backfill.py` - Historical data backfill
- [ ] `scripts/run_pipeline.py` - Run single pipeline iteration
- [ ] Database migrations

### Success Criteria
- [ ] Pipeline completes for all stocks without crashing
- [ ] Duplicates properly detected and skipped
- [ ] Announcements correctly categorized (>80% accuracy)
- [ ] Backfill completes for 90 days of history
- [ ] Incremental runs complete in <15 minutes

---

## Phase 5: Web Scraping for Additional Sources

### Objective
Supplement API data with scraped data from FDA CDER and company IR pages.

### 4.1 FDA CDER Scraping

**Target URLs**:
- Drug Approvals: `https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/novel-drug-approvals-2024`
- Drug Approval Letters: `https://www.accessdata.fda.gov/scripts/cder/daf/`

**Data to Extract**:
- Drug name (brand and generic)
- Approval date
- Indication
- Sponsor company (map to our stocks)
- Approval type (NME, BLA, etc.)

**Implementation**:
```python
class FDACDERScraper:
    """
    Scrapes FDA CDER for drug approvals.
    
    Approach:
    1. Check RSS feed first (if available)
    2. Scrape approval pages for new entries
    3. Match sponsor to our stock universe
    4. Extract approval letters if linked
    
    Frequency: Daily
    """
```

### 4.2 Company IR Page Scraping

**Challenge**: Each company has a different IR page structure.

**Approach**:
```python
class IRScraper:
    """
    Generic IR page scraper with company-specific adapters.
    
    Common patterns to detect:
    - RSS feeds (preferred)
    - Press release listing pages
    - News/media sections
    
    For each stock:
    1. Try to find RSS feed at common URLs (/feed, /rss, etc.)
    2. If no RSS, identify press release page structure
    3. Build company-specific parser or use generic heuristics
    """
```

**Priority Companies**:
Start with top 50 stocks by trading volume, manually verify scrapers work.

**Anti-Detection**:
```python
SCRAPING_CONFIG = {
    "user_agent_rotation": True,
    "request_delay_range": (2, 5),  # seconds
    "respect_robots_txt": True,
    "max_requests_per_domain": 10,  # per run
}
```

### Deliverables for Phase 4
- [ ] `src/scrapers/fda_cder.py` - FDA CDER scraper
- [ ] `src/scrapers/ir_scraper.py` - Generic IR scraper
- [ ] `src/scrapers/adapters/` - Company-specific adapters
- [ ] `config/ir_feeds.yaml` - Known RSS feed URLs
- [ ] Manual verification for top 50 companies

### Success Criteria
- [ ] FDA approvals captured within 24 hours
- [ ] IR press releases captured for 50+ companies
- [ ] No IP bans or blocking
- [ ] Scrapers resilient to minor HTML changes

---

## Phase 6: AI Summarization & Enrichment (DEFERRED)

> **Note**: This phase is deferred until the core pipeline (Phases 1-4) is stable and working.

### Objective
Use AI to summarize announcements, assign sentiment, and improve categorization.

### Future AI Tasks

| Task | Input | Output |
|------|-------|--------|
| Summarize | Full announcement text | 2-3 sentence summary |
| Sentiment | Full text + category | positive/negative/neutral |
| Categorize | Full text | Refined category from taxonomy |
| Extract Entities | Full text | Drug names, trial phases, etc. |

### Implementation Options (to evaluate later)

1. **Claude API** - High quality, cost per token
2. **Ollama/Local** - Free, needs GPU, lower quality
3. **Hybrid** - Local for bulk, Claude for complex

### Schema Ready for AI
The `announcements` table already includes:
- `sentiment` column (nullable)
- `ai_summary` column (nullable)
- `is_processed` flag for tracking

When ready to implement, create `src/enrichment/` module.

---

## Phase 7: Dashboard & Notifications

### Objective
Create a simple dashboard to review announcements and send Slack/Discord alerts for important news.

### 6.1 Dashboard (Streamlit)

A lightweight web UI to browse and filter announcements.

**Features**:
- [ ] Filter by ticker, date range, category, source
- [ ] Sort by date, sentiment, category
- [ ] Search announcement content
- [ ] Mark announcements as "reviewed" or "watchlist"
- [ ] Export filtered results to CSV
- [ ] Basic stats (announcements per day, by category)

**Tech**: Streamlit (simple, Python-native, no frontend skills needed)

```python
# scripts/dashboard.py
import streamlit as st
import pandas as pd
from src.db.queries import get_announcements

st.title("Biopharma Announcement Monitor")

# Filters
col1, col2, col3 = st.columns(3)
ticker = col1.selectbox("Ticker", ["All"] + get_tickers())
category = col2.selectbox("Category", ["All"] + CATEGORIES)
days = col3.slider("Days back", 1, 365, 30)

# Display
df = get_announcements(ticker=ticker, category=category, days=days)
st.dataframe(df, use_container_width=True)
```

**Run**: `streamlit run scripts/dashboard.py`

### 6.2 Slack/Discord Webhook Notifications

Send alerts when new announcements match certain criteria.

**Alert Triggers** (configurable):
```yaml
# config/alerts.yaml
alerts:
  - name: "FDA Approvals"
    categories: [fda_approval]
    priority: high
    
  - name: "Trial Results"
    categories: [trial_results]
    priority: high
    
  - name: "Watchlist Activity"
    tickers: [MRNA, NVAX, BNTX]  # Your watchlist
    priority: medium
    
  - name: "All Announcements"
    enabled: false  # Too noisy, disabled by default
```

**Webhook Integration**:
```python
# src/notifications/webhook.py
import httpx

class WebhookNotifier:
    def __init__(self, webhook_url: str, platform: str = "slack"):
        self.webhook_url = webhook_url
        self.platform = platform  # "slack" or "discord"
    
    def send(self, announcement: Announcement):
        if self.platform == "slack":
            payload = self._format_slack(announcement)
        else:
            payload = self._format_discord(announcement)
        
        httpx.post(self.webhook_url, json=payload)
    
    def _format_slack(self, a: Announcement) -> dict:
        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"📢 {a.ticker}: {a.category}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{a.title}*\n{a.content[:500]}..."}},
                {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Source: {a.source} | <{a.url}|View Original>"}]}
            ]
        }
```

**Setup**:
1. Create Slack App → Incoming Webhooks → Copy URL
2. Or Discord → Server Settings → Integrations → Webhooks → Copy URL
3. Add to `config/alerts.yaml`

### Deliverables for Phase 6
- [ ] `scripts/dashboard.py` - Streamlit dashboard
- [ ] `src/notifications/webhook.py` - Slack/Discord sender
- [ ] `config/alerts.yaml` - Alert configuration
- [ ] Cron integration to send alerts after daily pipeline run

### Success Criteria
- [ ] Dashboard loads in <3 seconds
- [ ] Filters work correctly
- [ ] Slack/Discord messages formatted nicely
- [ ] Alerts sent within 5 minutes of pipeline completion

---

## Project Structure

```
biopharma-monitor/
├── config/
│   ├── filters.yaml          # Stock filter criteria
│   ├── api_config.yaml       # API settings, rate limits
│   ├── eodhd.yaml            # EODHD API key and settings
│   ├── database.yaml         # MySQL connection settings
│   ├── alerts.yaml           # Slack/Discord webhook config
│   ├── prompts.yaml          # AI prompt templates (Phase 5)
│   └── ir_feeds.yaml         # Known IR RSS feeds
├── data/
│   ├── stocks.csv            # Stock universe (Phase 1 MVP output)
│   └── stocks_quality_report.csv  # Data quality report
├── logs/
│   └── pipeline.log          # Daily pipeline logs
├── sql/
│   ├── schema.sql            # Full database schema
│   └── migrations/           # Schema migrations
├── src/
│   ├── clients/
│   │   ├── base.py           # Base HTTP client with rate limiting
│   │   ├── eodhd.py          # EODHD API client (stock data)
│   │   ├── edgar.py          # SEC EDGAR client
│   │   ├── clinicaltrials.py # ClinicalTrials.gov client
│   │   └── openfda.py        # OpenFDA client
│   ├── scrapers/
│   │   ├── fda_cder.py       # FDA CDER scraper
│   │   ├── ir_scraper.py     # IR page scraper
│   │   └── adapters/         # Company-specific adapters
│   ├── pipeline/
│   │   ├── orchestrator.py   # Fetch orchestrator
│   │   ├── normalizers.py    # Data normalizers
│   │   └── storage.py        # Database operations
│   ├── enrichment/           # Phase 5 (deferred)
│   │   ├── summarizer.py     # AI summarization
│   │   ├── sentiment.py      # Sentiment analysis
│   │   └── categorizer.py    # Category refinement
│   ├── notifications/
│   │   └── webhook.py        # Slack/Discord webhook sender
│   ├── models/
│   │   ├── stock.py          # Stock data model
│   │   └── announcement.py   # Announcement data model
│   ├── db/
│   │   ├── connection.py     # MySQL connection pool
│   │   └── queries.py        # SQL query helpers
│   └── utils/
│       ├── rate_limiter.py   # Rate limiting
│       ├── cache.py          # Response caching
│       └── logging.py        # Logging setup
├── scripts/
│   ├── fetch_stock_list.py   # Build stock universe (EODHD + SEC)
│   ├── enrich_stocks.py      # Add missing CIK, verify data
│   ├── find_ir_urls.py       # Discover IR page URLs
│   ├── backfill.py           # Historical data fetch (1 year)
│   ├── run_pipeline.py       # Daily pipeline run (cron target)
│   ├── dashboard.py          # Streamlit dashboard
│   └── test_connections.py   # API connection tests
├── tests/
│   ├── test_eodhd.py
│   ├── test_edgar.py
│   ├── test_clinicaltrials.py
│   ├── test_openfda.py
│   └── test_normalizers.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── prompt.md                 # This file
```

---

## Open Questions & Decisions Needed

### Phase 1 — ✅ DECIDED
- [x] **Market cap thresholds**: <$2B (micro + small cap)
- [x] **Exchange coverage**: NASDAQ, NYSE, NYSE American, OTCQX, OTCQB
- [x] **Industry scope**: Biotechnology + Pharmaceuticals only
- [x] **Data source API**: EODHD (primary) + SEC EDGAR + free sources

### Phase 2 — ✅ DECIDED
- [x] **8-K items**: All items (comprehensive monitoring)
- [x] **ClinicalTrials events**: All status changes
- [x] **Testing approach**: Maximum coverage (unit + integration + mocks, 95% target)

### Phase 3 — ✅ DECIDED (Text Extraction & File Storage)
- [x] **Hash-based ID**: `sha256(url + published_date)` - first 16 chars
- [x] **SEC EDGAR 8-K**: Extract main document + ALL exhibits
- [x] **PDF extraction**: pymupdf (fitz) - fast, good quality
- [x] **ClinicalTrials.gov**: All text fields concatenated (comprehensive)
- [x] **OpenFDA**: Structured JSON preserved as-is (not converted to text)
- [x] **CSV management**: Single file forever (`announcements.csv`)
- [x] **Failure handling**: Retry 3x → mark FAILED → keep raw → write empty text file

### Phase 3.5 — ✅ DECIDED (Post-Announcement Returns + Parquet Dataset)
- [x] **Price point**: Adjusted close (accounts for splits/dividends)
- [x] **Start price**: **Next trading day's close (T+1)** - accounts for after-hours announcements
- [x] **End price (weekend/holiday)**: Use closest available trading day
- [x] **Days measurement**: Calendar days (30, 60, 90)
- [x] **Return format**: Decimal with 2 places (e.g., `15.43`)
- [x] **Recent announcements**: Leave as NULL (calculate later when data available)
- [x] **Files to update**: `data/index/announcements.csv` + `data/index/announcements.parquet`
- [x] **Processing strategy**: Batch - fetch all price history per ticker first
- [x] **Missing price data**: Fail entire ticker batch, log error
- [x] **Parquet location**: Same directory as CSV (`data/index/`)
- [x] **Parquet sync**: Always keep in sync (auto-update)
- [x] **Parquet records**: Include all (even FAILED)
- [x] **Parquet text column**: `raw_text` (full text, no size limit)

### Phase 4 — ✅ DECIDED (Database Storage)
- [x] **Monitoring frequency**: Once daily (6 AM ET)
- [x] **Historical backfill**: 1 year (365 days)

### Phase 5 — NEEDS INPUT (Web Scraping)
- [ ] **Scraping priority**: Which companies to prioritize for IR scraping?

### Phase 6 — ✅ DECIDED (AI Summarization)
- [x] **AI provider**: Deferred - add later when core pipeline stable

### Phase 7 — ✅ DECIDED (Dashboard & Notifications)
- [x] **Dashboard**: Streamlit (simple table/filter UI)
- [x] **Notifications**: Slack/Discord webhook

### General — ✅ DECIDED
- [x] **Deployment**: Local Mac machine
- [x] **Database**: MySQL 8.0
- [x] **API keys**: EODHD ✅

---

## Getting Started

### Prerequisites (Mac)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+
brew install python@3.11

# Install MySQL
brew install mysql
brew services start mysql

# Secure MySQL installation
mysql_secure_installation
```

### Project Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd biopharma-monitor

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up MySQL database
mysql -u root -p -e "CREATE DATABASE biopharma_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
mysql -u root -p -e "CREATE USER 'biopharma'@'localhost' IDENTIFIED BY 'your_password_here';"
mysql -u root -p -e "GRANT ALL PRIVILEGES ON biopharma_monitor.* TO 'biopharma'@'localhost';"
mysql -u root -p biopharma_monitor < sql/schema.sql

# 5. Copy and edit config files
cp config/database.example.yaml config/database.yaml
cp config/filters.example.yaml config/filters.yaml
cp config/api_config.example.yaml config/api_config.yaml
# Edit config/database.yaml with your MySQL credentials

# 6. Verify setup
python scripts/test_connections.py

# 7. Run Phase 1
python scripts/fetch_stock_list.py
```

### Daily Cron Setup (Mac)

```bash
# Edit crontab
crontab -e

# Add daily run at 6 AM ET (adjust for your timezone)
# Mac uses local time, so if you're in PT, use 3 AM
0 6 * * * cd /path/to/biopharma-monitor && /path/to/venv/bin/python scripts/run_pipeline.py >> logs/pipeline.log 2>&1
```

### Running the Dashboard

```bash
# Start Streamlit dashboard (runs on http://localhost:8501)
streamlit run scripts/dashboard.py
```

---

## Appendix A: EODHD API Reference

This section provides detailed API documentation for the EODHD endpoints we'll use. Claude Code should reference this when implementing the clients.

### Base URL
```
https://eodhd.com/api/
```

### Authentication
All requests require `api_token` parameter:
```
?api_token={YOUR_API_TOKEN}
```

### Rate Limits
- Depends on your subscription plan
- Free: 20 requests/day
- Paid plans: up to 100,000 requests/day
- Check your limits: `GET /user?api_token={TOKEN}`

---

### Endpoint 1: Exchange Symbol List

**Purpose**: Get all tickers for an exchange (used to build our stock universe)

**URL**: `GET /exchange-symbol-list/{EXCHANGE_CODE}`

**Parameters**:
| Param | Required | Description |
|-------|----------|-------------|
| `api_token` | Yes | Your API key |
| `fmt` | No | `json` or `csv` (default: csv) |

**Exchange Codes for US**:
- `US` - Combined NASDAQ, NYSE, NYSE ARCA, and OTC markets
- `NASDAQ` - NASDAQ only
- `NYSE` - NYSE only
- `AMEX` - NYSE American only

**Example Request**:
```bash
curl "https://eodhd.com/api/exchange-symbol-list/US?api_token={TOKEN}&fmt=json"
```

**Example Response**:
```json
[
  {
    "Code": "AAPL",
    "Name": "Apple Inc",
    "Country": "USA",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Type": "Common Stock",
    "Isin": "US0378331005"
  },
  {
    "Code": "MRNA",
    "Name": "Moderna Inc",
    "Country": "USA",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Type": "Common Stock",
    "Isin": "US60770K1079"
  }
]
```

**Key Fields**:
| Field | Description | Use |
|-------|-------------|-----|
| `Code` | Ticker symbol | Primary identifier |
| `Name` | Company name | Display name |
| `Exchange` | Specific exchange | Filter/categorize |
| `Type` | Security type | Filter for "Common Stock" |
| `Isin` | ISIN identifier | Cross-reference |

**Filtering Tip**: Filter `Type == "Common Stock"` to exclude ETFs, ADRs, Preferred, etc.

---

### Endpoint 2: Fundamentals Data

**Purpose**: Get detailed company information including market cap, industry, sector

**URL**: `GET /fundamentals/{TICKER}.{EXCHANGE}`

**Parameters**:
| Param | Required | Description |
|-------|----------|-------------|
| `api_token` | Yes | Your API key |
| `fmt` | No | `json` only for fundamentals |
| `filter` | No | Return specific section only (e.g., `General`, `Highlights`) |

**Example Request**:
```bash
# Full fundamentals
curl "https://eodhd.com/api/fundamentals/MRNA.US?api_token={TOKEN}&fmt=json"

# Only General section (faster, smaller response)
curl "https://eodhd.com/api/fundamentals/MRNA.US?api_token={TOKEN}&filter=General"

# Multiple sections
curl "https://eodhd.com/api/fundamentals/MRNA.US?api_token={TOKEN}&filter=General,Highlights"
```

**Response Structure** (key sections for our use case):

```json
{
  "General": {
    "Code": "MRNA",
    "Type": "Common Stock",
    "Name": "Moderna Inc",
    "Exchange": "NASDAQ",
    "CurrencyCode": "USD",
    "CurrencyName": "US Dollar",
    "CurrencySymbol": "$",
    "CountryName": "USA",
    "CountryISO": "US",
    "ISIN": "US60770K1079",
    "CUSIP": "60770K107",
    "CIK": "1682852",
    "Sector": "Healthcare",
    "Industry": "Biotechnology",
    "Description": "Moderna, Inc. is a biotechnology company...",
    "FullTimeEmployees": 3900,
    "UpdatedAt": "2024-01-15",
    "WebURL": "https://www.modernatx.com",
    "LogoURL": "https://eodhd.com/img/logos/US/MRNA.png",
    "Phone": "617-714-6500",
    "Address": "200 Technology Square",
    "City": "Cambridge",
    "State": "MA",
    "Country": "USA",
    "Zip": "02139"
  },
  "Highlights": {
    "MarketCapitalization": 45000000000,
    "MarketCapitalizationMln": 45000,
    "EBITDA": 1200000000,
    "PERatio": 25.5,
    "PEGRatio": 1.2,
    "WallStreetTargetPrice": 150.00,
    "BookValue": 35.20,
    "DividendShare": 0,
    "DividendYield": 0,
    "EarningsShare": 5.50,
    "EPSEstimateCurrentYear": 6.00,
    "EPSEstimateNextYear": 7.50,
    "EPSEstimateNextQuarter": 1.50,
    "EPSEstimateCurrentQuarter": 1.40,
    "MostRecentQuarter": "2023-12-31",
    "ProfitMargin": 0.15,
    "OperatingMarginTTM": 0.12,
    "ReturnOnAssetsTTM": 0.08,
    "ReturnOnEquityTTM": 0.18,
    "RevenueTTM": 8000000000,
    "RevenuePerShareTTM": 20.50,
    "QuarterlyRevenueGrowthYOY": -0.05,
    "GrossProfitTTM": 6000000000,
    "DilutedEpsTTM": 5.50,
    "QuarterlyEarningsGrowthYOY": 0.10
  },
  "Valuation": {
    "TrailingPE": 25.5,
    "ForwardPE": 20.0,
    "PriceSalesTTM": 5.5,
    "PriceBookMRQ": 3.2,
    "EnterpriseValue": 42000000000,
    "EnterpriseValueRevenue": 5.25,
    "EnterpriseValueEbitda": 35.0
  },
  "SharesStats": {
    "SharesOutstanding": 390000000,
    "SharesFloat": 380000000,
    "PercentInsiders": 2.5,
    "PercentInstitutions": 75.0,
    "SharesShort": 15000000,
    "SharesShortPriorMonth": 14000000,
    "ShortRatio": 2.5,
    "ShortPercentOutstanding": 0.04,
    "ShortPercentFloat": 0.04
  }
}
```

**Key Fields for Our Project**:
| Path | Description | Our Field |
|------|-------------|-----------|
| `General.Code` | Ticker | `ticker` |
| `General.Name` | Company name | `company_name` |
| `General.Exchange` | Exchange | `exchange` |
| `General.Sector` | Sector | `sector` |
| `General.Industry` | Industry | `industry` ⭐ Filter on this |
| `General.CIK` | SEC CIK number | `cik` |
| `General.CUSIP` | CUSIP identifier | `cusip` |
| `General.WebURL` | Company website | `website` |
| `Highlights.MarketCapitalization` | Market cap | `market_cap` ⭐ Filter on this |

---

### Endpoint 3: Bulk Fundamentals (if available on your plan)

**Purpose**: Get fundamentals for entire exchange in one request (more efficient)

**URL**: `GET /bulk-fundamentals/{EXCHANGE}`

**Note**: Returns CSV only, limited fields. Use individual fundamentals for full data.

**Example**:
```bash
curl "https://eodhd.com/api/bulk-fundamentals/US?api_token={TOKEN}&fmt=csv"
```

---

### Endpoint 4: Exchanges List

**Purpose**: Get list of all supported exchanges

**URL**: `GET /exchanges-list/`

**Example Response**:
```json
[
  {
    "Name": "USA Stocks",
    "Code": "US",
    "OperatingMIC": "XNAS,XNYS",
    "Country": "USA",
    "Currency": "USD",
    "CountryISO2": "US",
    "CountryISO3": "USA"
  }
]
```

---

### Industry Values in EODHD

Common industry values for biopharma filtering:
- `"Biotechnology"`
- `"Drug Manufacturers—General"`
- `"Drug Manufacturers—Specialty & Generic"`
- `"Pharmaceutical Retailers"`
- `"Diagnostics & Research"`

**Recommended filter logic**:
```python
BIOPHARMA_INDUSTRIES = [
    "biotechnology",
    "drug manufacturers",
    "pharmaceutical",
]

def is_biopharma(industry: str) -> bool:
    if not industry:
        return False
    industry_lower = industry.lower()
    return any(term in industry_lower for term in BIOPHARMA_INDUSTRIES)
```

---

### Error Handling

**Common HTTP Status Codes**:
| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Parse response |
| 401 | Invalid API key | Check credentials |
| 403 | Plan limit exceeded | Upgrade or wait |
| 404 | Ticker not found | Skip, log warning |
| 429 | Rate limited | Backoff and retry |
| 500 | Server error | Retry with backoff |

**Error Response Format**:
```json
{
  "error": "API key is invalid or not provided"
}
```

---

### API Call Costs

| Endpoint | Cost |
|----------|------|
| Exchange symbol list | 1 call |
| Single fundamentals | 1 call |
| Bulk fundamentals | 100 calls |
| EOD prices | 1 call |

---

| Date | Version | Changes |
|------|---------|---------|
| TBD | 0.1.0 | Initial project setup, Phase 1 (stock universe) |
| TBD | 0.2.0 | Phase 2: API connections (EODHD, EDGAR, CT.gov, OpenFDA) |
| TBD | 0.3.0 | Phase 3: Data pipeline & announcement aggregation |
| TBD | 0.4.0 | Phase 4: Web scrapers (FDA CDER, IR pages) |
| TBD | 0.5.0 | Phase 5: AI enrichment (deferred) |
| TBD | 0.6.0 | Phase 6: Dashboard & Slack/Discord notifications |

---

## Dependencies (requirements.txt)

```
# HTTP & API clients
httpx>=0.25.0
tenacity>=8.2.0          # Retry logic

# Database
mysql-connector-python>=8.2.0
sqlalchemy>=2.0.0

# Web scraping
beautifulsoup4>=4.12.0
playwright>=1.40.0       # For JS-heavy pages
lxml>=4.9.0

# Data processing
pandas>=2.0.0
pyyaml>=6.0.0

# Dashboard
streamlit>=1.29.0

# Utilities
python-dotenv>=1.0.0     # Environment variables
click>=8.1.0             # CLI interface
rich>=13.0.0             # Pretty console output
schedule>=1.2.0          # Simple scheduling (alternative to cron)

# Testing (Phase 2 - IMPORTANT)
pytest>=7.4.0
pytest-asyncio>=0.23.0   # Async test support
pytest-cov>=4.1.0        # Coverage reporting
pytest-timeout>=2.2.0    # Test timeouts
respx>=0.20.0            # Mock httpx requests
freezegun>=1.2.0         # Time mocking
factory-boy>=3.3.0       # Test data factories

# Type checking
mypy>=1.7.0
types-PyYAML
types-requests
```

---

## Environment Variables

Create a `.env` file in the project root (never commit this!):

```bash
# .env
EODHD_API_KEY=your_eodhd_api_key_here

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=biopharma
MYSQL_PASSWORD=your_password_here
MYSQL_DATABASE=biopharma_monitor

# Notifications (Phase 6)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
# OR
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy

# Optional: SEC EDGAR requires contact email in User-Agent
SEC_CONTACT_EMAIL=your_email@example.com
```

Add to `.gitignore`:
```
.env
config/database.yaml
config/eodhd.yaml
config/alerts.yaml
*.log
__pycache__/
.pytest_cache/
```
