"""Shared pytest fixtures for tests."""

import pytest
import respx
from httpx import Response


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def sample_eodhd_symbol():
    """Sample symbol from EODHD exchange-symbol-list."""
    return {
        "Code": "MRNA",
        "Name": "Moderna Inc",
        "Country": "USA",
        "Exchange": "NASDAQ",
        "Currency": "USD",
        "Type": "Common Stock",
        "Isin": "US60770K1079",
    }


@pytest.fixture
def sample_eodhd_fundamentals():
    """Sample fundamentals response from EODHD."""
    return {
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
            "Description": "Moderna, Inc. operates as a biotechnology company...",
            "WebURL": "https://www.modernatx.com",
            "Phone": "617-714-6500",
            "Address": "200 Technology Square",
            "City": "Cambridge",
            "State": "MA",
            "ZIP": "02139",
            "FullTimeEmployees": 3900,
        },
        "Highlights": {
            "MarketCapitalization": 15000000000,
            "EBITDA": -1500000000,
            "PERatio": None,
            "PEGRatio": None,
            "WallStreetTargetPrice": 150.00,
            "BookValue": 25.50,
            "DividendShare": 0.0,
            "DividendYield": 0.0,
            "EarningsShare": -5.50,
            "EPSEstimateCurrentYear": -3.20,
            "EPSEstimateNextYear": 2.50,
            "EPSEstimateNextQuarter": -1.20,
            "EPSEstimateCurrentQuarter": -1.50,
            "MostRecentQuarter": "2024-06-30",
            "ProfitMargin": -0.25,
            "OperatingMarginTTM": -0.30,
            "ReturnOnAssetsTTM": -0.08,
            "ReturnOnEquityTTM": -0.15,
            "RevenueTTM": 6500000000,
            "RevenuePerShareTTM": 17.50,
            "QuarterlyRevenueGrowthYOY": -0.45,
            "GrossProfitTTM": 4500000000,
            "DilutedEpsTTM": -5.50,
        },
    }


@pytest.fixture
def sample_small_cap_fundamentals():
    """Sample fundamentals for a small-cap biopharma company."""
    return {
        "General": {
            "Code": "SNDX",
            "Name": "Syndax Pharmaceuticals Inc",
            "Exchange": "NASDAQ",
            "CurrencyCode": "USD",
            "CountryISO": "US",
            "CIK": "1395937",
            "Sector": "Healthcare",
            "Industry": "Biotechnology",
            "WebURL": "https://www.syndax.com",
        },
        "Highlights": {
            "MarketCapitalization": 800000000,  # $800M
        },
    }


@pytest.fixture
def sample_non_biopharma_fundamentals():
    """Sample fundamentals for a non-biopharma company."""
    return {
        "General": {
            "Code": "AAPL",
            "Name": "Apple Inc",
            "Exchange": "NASDAQ",
            "CurrencyCode": "USD",
            "CountryISO": "US",
            "CIK": "320193",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "WebURL": "https://www.apple.com",
        },
        "Highlights": {
            "MarketCapitalization": 3000000000000,  # $3T
        },
    }


@pytest.fixture
def sample_sec_company_tickers():
    """Sample SEC company_tickers.json response."""
    return {
        "0": {"cik_str": 1682852, "ticker": "MRNA", "title": "Moderna, Inc."},
        "1": {"cik_str": 1395937, "ticker": "SNDX", "title": "Syndax Pharmaceuticals, Inc."},
        "2": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }


@pytest.fixture
def sample_sec_submissions():
    """Sample SEC submissions response for a biopharma company."""
    return {
        "cik": "1682852",
        "entityType": "operating",
        "sic": "2836",
        "sicDescription": "Biological Products, Except Diagnostic Substances",
        "name": "Moderna, Inc.",
        "tickers": ["MRNA"],
        "exchanges": ["NASDAQ"],
        "ein": "81-3674991",
        "category": "Large accelerated filer",
        "fiscalYearEnd": "1231",
        "stateOfIncorporation": "DE",
        "website": "https://www.modernatx.com",
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "8-K"],
                "filingDate": ["2024-05-01", "2024-04-15", "2024-03-20"],
                "accessionNumber": ["0001682852-24-000001", "0001682852-24-000002", "0001682852-24-000003"],
                "primaryDocument": ["mrna-20240331.htm", "mrna-8k.htm", "mrna-8k.htm"],
            }
        },
    }


@pytest.fixture
def sample_sec_companyfacts():
    """Sample SEC companyfacts response with shares outstanding."""
    return {
        "cik": 1682852,
        "entityName": "Moderna, Inc.",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "Entity Common Stock, Shares Outstanding",
                    "description": "Indicate number of shares outstanding",
                    "units": {
                        "shares": [
                            {
                                "end": "2024-03-31",
                                "val": 385000000,
                                "accn": "0001682852-24-000001",
                                "form": "10-Q",
                                "filed": "2024-05-01",
                            },
                            {
                                "end": "2023-12-31",
                                "val": 383000000,
                                "accn": "0001682852-24-000002",
                                "form": "10-K",
                                "filed": "2024-02-15",
                            },
                        ]
                    },
                }
            },
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "label": "Common Stock, Shares, Outstanding",
                    "units": {
                        "shares": [
                            {
                                "end": "2024-03-31",
                                "val": 385000000,
                                "accn": "0001682852-24-000001",
                                "form": "10-Q",
                                "filed": "2024-05-01",
                            },
                        ]
                    },
                }
            },
        },
    }


@pytest.fixture
def sample_biopharma_submissions():
    """Sample SEC submissions for a small-cap biopharma (SIC 2834)."""
    return {
        "cik": "1395937",
        "entityType": "operating",
        "sic": "2834",
        "sicDescription": "Pharmaceutical Preparations",
        "name": "Syndax Pharmaceuticals, Inc.",
        "tickers": ["SNDX"],
        "exchanges": ["NASDAQ"],
        "fiscalYearEnd": "1231",
        "stateOfIncorporation": "DE",
        "website": "https://www.syndax.com",
        "filings": {"recent": {}},
    }


# ============================================================================
# SEC EDGAR Fixtures (Phase 2)
# ============================================================================


@pytest.fixture
def sample_edgar_submissions_with_8k():
    """Sample SEC EDGAR submissions with 8-K filings.

    Uses dates relative to today to ensure tests work regardless of current date.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    dates = [
        (today - timedelta(days=30)).strftime("%Y-%m-%d"),  # 8-K
        (today - timedelta(days=45)).strftime("%Y-%m-%d"),  # 10-Q
        (today - timedelta(days=60)).strftime("%Y-%m-%d"),  # 8-K
        (today - timedelta(days=62)).strftime("%Y-%m-%d"),  # 8-K/A
        (today - timedelta(days=300)).strftime("%Y-%m-%d"), # 10-K
    ]

    return {
        "cik": "1682852",
        "name": "Moderna, Inc.",
        "tickers": ["MRNA"],
        "exchanges": ["NASDAQ"],
        "sic": "2836",
        "sicDescription": "Biological Products, Except Diagnostic Substances",
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "8-K", "8-K/A", "10-K"],
                "filingDate": dates,
                "accessionNumber": [
                    "0001682852-24-000100",
                    "0001682852-24-000090",
                    "0001682852-24-000080",
                    "0001682852-24-000075",
                    "0001682852-24-000010",
                ],
                "primaryDocument": [
                    "mrna-8k_20241215.htm",
                    "mrna-10q_20241114.htm",
                    "mrna-8k_20241020.htm",
                    "mrna-8ka_20241018.htm",
                    "mrna-10k_20240228.htm",
                ],
                "items": ["2.02, 9.01", "", "7.01, 8.01, 9.01", "7.01", ""],
                "size": [150000, 5000000, 200000, 180000, 10000000],
            }
        },
    }


@pytest.fixture
def sample_edgar_empty_submissions():
    """Sample SEC EDGAR submissions with no filings."""
    return {
        "cik": "9999999999",
        "name": "Empty Company, Inc.",
        "tickers": [],
        "exchanges": [],
        "filings": {"recent": {}},
    }


# ============================================================================
# ClinicalTrials.gov Fixtures (Phase 2)
# ============================================================================


@pytest.fixture
def sample_clinicaltrials_study():
    """Sample ClinicalTrials.gov API v2 study response.

    Uses dates relative to today to ensure tests work regardless of current date.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    last_update = (today - timedelta(days=15)).strftime("%Y-%m-%d")

    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT04470427",
                "briefTitle": "A Study of mRNA-1273 Vaccine in Adults",
                "officialTitle": "A Phase 3 Study to Evaluate Efficacy, Safety, and Immunogenicity of mRNA-1273 SARS-CoV-2 Vaccine",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-07-27"},
                "completionDateStruct": {"date": "2022-12-31"},
                "lastUpdatePostDateStruct": {"date": last_update},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Moderna, Inc."},
            },
            "conditionsModule": {
                "conditions": ["COVID-19", "SARS-CoV-2 Infection"],
            },
            "designModule": {
                "phases": ["PHASE3"],
                "enrollmentInfo": {"count": 30000},
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "mRNA-1273"},
                    {"name": "Placebo"},
                ],
            },
            "descriptionModule": {
                "briefSummary": "This study will evaluate the efficacy of mRNA-1273...",
            },
        },
        "resultsSection": {"participantFlowModule": {}},
    }


@pytest.fixture
def sample_clinicaltrials_search_response(sample_clinicaltrials_study):
    """Sample ClinicalTrials.gov search response with multiple studies.

    Uses dates relative to today for consistent test behavior.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    last_update = (today - timedelta(days=30)).strftime("%Y-%m-%d")

    return {
        "studies": [
            sample_clinicaltrials_study,
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT05889169",
                        "briefTitle": "mRNA-1283 Study in Adults",
                    },
                    "statusModule": {
                        "overallStatus": "RECRUITING",
                        "startDateStruct": {"date": "2023-06-01"},
                        "lastUpdatePostDateStruct": {"date": last_update},
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "Moderna, Inc."},
                    },
                    "conditionsModule": {"conditions": ["Respiratory Syncytial Virus"]},
                    "designModule": {"phases": ["PHASE2", "PHASE3"], "enrollmentInfo": {"count": 5000}},
                    "armsInterventionsModule": {"interventions": [{"name": "mRNA-1283"}]},
                    "descriptionModule": {},
                },
            },
        ],
        "nextPageToken": "abc123",
        "totalCount": 25,
    }


@pytest.fixture
def sample_clinicaltrials_empty_response():
    """Sample ClinicalTrials.gov empty search response."""
    return {
        "studies": [],
        "totalCount": 0,
    }


# ============================================================================
# OpenFDA Fixtures (Phase 2)
# ============================================================================


@pytest.fixture
def sample_openfda_drug_approval():
    """Sample OpenFDA Drugs@FDA response for a drug approval.

    Uses dates relative to today for consistent test behavior.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    approval_date = (today - timedelta(days=30)).strftime("%Y%m%d")

    return {
        "results": [
            {
                "application_number": "BLA761222",
                "sponsor_name": "Moderna TX, Inc.",
                "products": [
                    {
                        "brand_name": "SPIKEVAX",
                        "dosage_form": "INJECTION, SUSPENSION",
                        "route": "INTRAMUSCULAR",
                        "active_ingredients": [
                            {"name": "ELASOMERAN", "strength": "50 MCG/0.5ML"}
                        ],
                    }
                ],
                "submissions": [
                    {
                        "submission_type": "BLA",
                        "submission_status": "AP",
                        "submission_status_date": approval_date,
                    }
                ],
                "openfda": {
                    "brand_name": ["SPIKEVAX"],
                    "generic_name": ["COVID-19 VACCINE, MRNA"],
                },
            }
        ],
        "meta": {
            "results": {"total": 1, "skip": 0, "limit": 100},
        },
    }


@pytest.fixture
def sample_openfda_multiple_approvals():
    """Sample OpenFDA response with multiple approvals."""
    return {
        "results": [
            {
                "application_number": "BLA761222",
                "sponsor_name": "Moderna TX, Inc.",
                "products": [{"brand_name": "SPIKEVAX", "dosage_form": "INJECTION"}],
                "submissions": [{"submission_type": "BLA", "submission_status": "AP", "submission_status_date": "20220131"}],
                "openfda": {},
            },
            {
                "application_number": "NDA214900",
                "sponsor_name": "Other Pharma Inc.",
                "products": [{"brand_name": "TESTDRUG", "dosage_form": "TABLET"}],
                "submissions": [{"submission_type": "NDA", "submission_status": "AP", "submission_status_date": "20231015"}],
                "openfda": {},
            },
        ],
        "meta": {"results": {"total": 2, "skip": 0, "limit": 100}},
    }


@pytest.fixture
def sample_openfda_empty_response():
    """Sample OpenFDA empty response."""
    return {
        "results": [],
        "meta": {"results": {"total": 0}},
    }


@pytest.fixture
def sample_openfda_drug_label():
    """Sample OpenFDA drug label response."""
    return {
        "results": [
            {
                "effective_time": "20240101",
                "indications_and_usage": ["For active immunization to prevent COVID-19..."],
                "warnings": ["Anaphylaxis has been reported following administration..."],
                "dosage_and_administration": ["Administer intramuscularly..."],
                "openfda": {
                    "application_number": ["BLA761222"],
                    "brand_name": ["SPIKEVAX"],
                    "generic_name": ["COVID-19 VACCINE, MRNA"],
                    "manufacturer_name": ["Moderna TX, Inc."],
                },
            }
        ],
    }


# ============================================================================
# Mock respx fixtures
# ============================================================================


@pytest.fixture
def mock_edgar_api():
    """Create a respx mock router for SEC EDGAR API."""
    with respx.mock(base_url="https://data.sec.gov") as mock:
        yield mock


@pytest.fixture
def mock_clinicaltrials_api():
    """Create a respx mock router for ClinicalTrials.gov API."""
    with respx.mock(base_url="https://clinicaltrials.gov/api/v2") as mock:
        yield mock


@pytest.fixture
def mock_openfda_api():
    """Create a respx mock router for OpenFDA API."""
    with respx.mock(base_url="https://api.fda.gov") as mock:
        yield mock
