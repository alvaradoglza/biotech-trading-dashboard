-- Biopharma Monitor Database Schema
-- MySQL 8.0+

-- ============================================================================
-- STOCKS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    market_cap DECIMAL(15,2),
    market_cap_category ENUM('nano', 'micro', 'small', 'mid', 'large') DEFAULT 'small',
    sector VARCHAR(100),
    industry VARCHAR(200),
    sic VARCHAR(4),
    sic_description VARCHAR(200),
    cik VARCHAR(10),
    website VARCHAR(500),
    ir_url VARCHAR(500),
    data_quality_score INT DEFAULT 0,
    data_source ENUM('eodhd', 'sec_fallback', 'manual') DEFAULT 'eodhd',
    shares_outstanding BIGINT,
    last_price DECIMAL(12,4),
    isin VARCHAR(12),
    cusip VARCHAR(9),
    country VARCHAR(50) DEFAULT 'US',
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_stocks_cik (cik),
    INDEX idx_stocks_exchange (exchange),
    INDEX idx_stocks_market_cap (market_cap),
    INDEX idx_stocks_sic (sic),
    INDEX idx_stocks_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- ANNOUNCEMENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS announcements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    source ENUM('edgar', 'clinicaltrials', 'openfda', 'ir_scrape', 'fda_scrape') NOT NULL,
    source_id VARCHAR(100),        -- Original ID from source (e.g., accession number, NCT ID)
    announcement_date DATE NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT,                  -- Full text or summary
    url VARCHAR(1000),             -- Link to original
    category ENUM(
        'earnings',
        'trial_start',
        'trial_update',
        'trial_results',
        'trial_terminated',
        'trial_suspended',
        'fda_approval',
        'fda_rejection',
        'fda_submission',
        'partnership',
        'financing',
        'executive_change',
        'other'
    ) DEFAULT 'other',

    -- AI enrichment fields (Phase 5 - nullable for now)
    ai_summary TEXT,
    sentiment ENUM('positive', 'negative', 'neutral'),
    is_processed BOOLEAN DEFAULT FALSE,

    -- Metadata
    raw_data JSON,                 -- Store original API response for debugging
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Constraints
    FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE CASCADE,
    UNIQUE KEY unique_source_id (source, source_id),

    -- Indexes
    INDEX idx_ticker_date (ticker, announcement_date DESC),
    INDEX idx_category (category),
    INDEX idx_source (source),
    INDEX idx_announcement_date (announcement_date DESC),
    INDEX idx_is_processed (is_processed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- CLINICAL TRIALS TRACKING TABLE
-- ============================================================================
-- Tracks all clinical trials for our stocks (separate from announcements)

CREATE TABLE IF NOT EXISTS clinical_trials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    nct_id VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(500) NOT NULL,
    sponsor VARCHAR(255),
    status VARCHAR(50),
    phase VARCHAR(50),
    conditions TEXT,               -- JSON array of conditions
    interventions TEXT,            -- JSON array of interventions
    enrollment INT,
    start_date DATE,
    completion_date DATE,
    last_update_date DATE,
    has_results BOOLEAN DEFAULT FALSE,
    url VARCHAR(500),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE SET NULL,
    INDEX idx_ct_ticker (ticker),
    INDEX idx_ct_status (status),
    INDEX idx_ct_last_update (last_update_date DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- FDA APPROVALS TRACKING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fda_approvals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    application_number VARCHAR(20) NOT NULL UNIQUE,
    sponsor_name VARCHAR(255),
    brand_name VARCHAR(255),
    generic_name VARCHAR(255),
    approval_date DATE,
    submission_type VARCHAR(20),   -- NDA, BLA, ANDA
    submission_status VARCHAR(20),
    dosage_form VARCHAR(100),
    route VARCHAR(100),
    active_ingredients TEXT,       -- JSON array
    url VARCHAR(500),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (ticker) REFERENCES stocks(ticker) ON DELETE SET NULL,
    INDEX idx_fda_ticker (ticker),
    INDEX idx_fda_approval_date (approval_date DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================================
-- PIPELINE RUN LOG
-- ============================================================================
-- Track pipeline execution for debugging and monitoring

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_type ENUM('full', 'incremental', 'backfill', 'single_stock') NOT NULL,
    ticker VARCHAR(10),            -- NULL for full runs
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status ENUM('running', 'completed', 'failed') DEFAULT 'running',
    stocks_processed INT DEFAULT 0,
    announcements_found INT DEFAULT 0,
    announcements_new INT DEFAULT 0,
    errors TEXT,                   -- JSON array of error messages

    INDEX idx_pr_status (status),
    INDEX idx_pr_started (started_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
