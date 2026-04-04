#!/usr/bin/env python3
"""Fetch and filter biopharma stocks from EODHD with SEC fallback.

This script:
1. Checks if EODHD Fundamentals is available (preflight)
2. If available: Uses EODHD Fundamentals for industry/market cap
3. If not (403): Falls back to SEC SIC codes + EODHD EOD prices
4. Filters for biopharma industry and small-cap (<$2B)
5. Outputs filtered stocks to data/stocks.csv
6. Generates a quality report
"""

import asyncio
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.eodhd import EODHDClient, StockFilter
from src.models.stock import Stock
from src.pipeline.stock_fetcher import StockDataFetcher
from src.utils.config import load_config, load_env, get_env, get_project_root
from src.utils.logging import get_logger, setup_file_logging

logger = get_logger(__name__)
console = Console()


async def fetch_and_filter_stocks(
    fetcher: StockDataFetcher,
    eodhd_client: EODHDClient,
    limit: int | None = None,
    progress: Progress | None = None,
) -> tuple[list[Stock], list[dict], dict[str, int]]:
    """Fetch and filter biopharma stocks.

    Args:
        fetcher: StockDataFetcher instance
        eodhd_client: EODHD client for symbol list
        limit: Maximum number of stocks to process (for testing)
        progress: Rich progress bar

    Returns:
        Tuple of (filtered stocks, quality issues, stats)
    """
    # Fetch all US symbols (this works on all EODHD plans)
    symbols = await eodhd_client.get_exchange_symbols("US")
    logger.info(f"Fetched {len(symbols)} total US symbols")

    # Create progress callback
    task_id = None
    if progress:
        task_id = progress.add_task(
            "[cyan]Processing stocks...",
            total=limit or len([s for s in symbols if fetcher.stock_filter.filter_symbol(s)]),
        )

    def progress_callback(ticker: str, index: int, total: int):
        if progress and task_id is not None:
            progress.update(
                task_id,
                advance=1,
                description=f"[cyan]Processing {ticker}...",
            )

    # Fetch and filter
    stocks, issues, stats = await fetcher.fetch_filtered_stocks(
        symbols,
        limit=limit,
        progress_callback=progress_callback,
    )

    return stocks, issues, stats


def save_stocks_csv(stocks: list[Stock], output_path: Path) -> None:
    """Save stocks to CSV file.

    Args:
        stocks: List of Stock objects
        output_path: Path to output CSV file
    """
    data = [stock.to_dict() for stock in stocks]
    df = pd.DataFrame(data)

    column_order = [
        "ticker",
        "company_name",
        "exchange",
        "market_cap",
        "market_cap_category",
        "sector",
        "industry",
        "sic",
        "sic_description",
        "cik",
        "website",
        "ir_url",
        "data_quality_score",
        "data_source",
        "shares_outstanding",
        "last_price",
        "isin",
        "cusip",
        "country",
        "currency",
    ]
    df = df.reindex(columns=[c for c in column_order if c in df.columns])

    df = df.sort_values(
        ["market_cap"],
        ascending=False,
        na_position="last",
    )

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(stocks)} stocks to {output_path}")


def save_quality_report(issues: list[dict], output_path: Path) -> None:
    """Save quality report to CSV file.

    Args:
        issues: List of quality issue dictionaries
        output_path: Path to output CSV file
    """
    if not issues:
        logger.info("No quality issues to report")
        return

    df = pd.DataFrame(issues)
    df = df.sort_values("quality_score", ascending=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved quality report with {len(issues)} issues to {output_path}")


def print_summary(stocks: list[Stock], issues: list[dict], stats: dict) -> None:
    """Print summary table to console.

    Args:
        stocks: List of filtered stocks
        issues: List of quality issues
        stats: Fetch statistics
    """
    console.print()

    summary_table = Table(title="Stock Universe Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Processed", str(stats.get("processed", 0)))
    summary_table.add_row("Matched Filters", str(len(stocks)))
    summary_table.add_row("Skipped", str(stats.get("skipped", 0)))
    summary_table.add_row("Errors", str(stats.get("errors", 0)))

    if stocks:
        # Market cap breakdown
        market_cap_counts = {}
        for stock in stocks:
            cat = stock.market_cap_category.value if stock.market_cap_category else "unknown"
            market_cap_counts[cat] = market_cap_counts.get(cat, 0) + 1

        console.print()
        for cat, count in sorted(market_cap_counts.items()):
            summary_table.add_row(f"  {cat.title()}", str(count))

        with_cik = sum(1 for s in stocks if s.cik)
        with_website = sum(1 for s in stocks if s.website)
        with_ir = sum(1 for s in stocks if s.ir_url)
        with_market_cap = sum(1 for s in stocks if s.market_cap)

        summary_table.add_row("With CIK", f"{with_cik} ({100*with_cik//len(stocks)}%)")
        summary_table.add_row("With Website", f"{with_website} ({100*with_website//len(stocks)}%)")
        summary_table.add_row("With IR URL", f"{with_ir} ({100*with_ir//len(stocks) if stocks else 0}%)")
        summary_table.add_row("With Market Cap", f"{with_market_cap} ({100*with_market_cap//len(stocks)}%)")
        summary_table.add_row("Quality Issues", str(len(issues)))

    console.print(summary_table)

    if stocks:
        console.print()
        top_table = Table(title="Top 10 Stocks by Market Cap")
        top_table.add_column("Ticker", style="cyan")
        top_table.add_column("Company", style="white")
        top_table.add_column("Market Cap", style="green")
        top_table.add_column("Industry/SIC", style="yellow")
        top_table.add_column("Quality", style="magenta")

        sorted_stocks = sorted(
            [s for s in stocks if s.market_cap],
            key=lambda x: x.market_cap or 0,
            reverse=True,
        )[:10]

        for stock in sorted_stocks:
            market_cap_str = (
                f"${stock.market_cap/1e9:.2f}B"
                if stock.market_cap and stock.market_cap >= 1e9
                else f"${stock.market_cap/1e6:.1f}M"
                if stock.market_cap
                else "N/A"
            )
            industry_str = stock.industry or stock.sic_description or stock.sic or "N/A"
            top_table.add_row(
                stock.ticker,
                stock.company_name[:40],
                market_cap_str,
                industry_str[:30],
                str(stock.data_quality_score),
            )

        console.print(top_table)


@click.command()
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of stocks to process (for testing)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output CSV file path",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to filters.yaml config file",
)
def main(limit: int | None, output: str | None, config: str | None) -> None:
    """Fetch and filter biopharma stocks from EODHD."""
    load_env()
    setup_file_logging()

    project_root = get_project_root()

    eodhd_api_key = get_env("EODHD_API_KEY", required=True)

    config_path = config or str(project_root / "config" / "filters.yaml")
    filter_config = load_config(config_path)

    eodhd_config_path = project_root / "config" / "eodhd.yaml"
    eodhd_config = (
        load_config(str(eodhd_config_path)) if eodhd_config_path.exists() else {}
    )

    output_path = Path(output) if output else project_root / "data" / "stocks.csv"
    quality_path = output_path.parent / "stocks_quality_report.csv"

    stock_filter = StockFilter.from_config(filter_config)

    console.print("[bold cyan]Biopharma Stock Universe Builder[/bold cyan]")
    console.print(f"Config: {config_path}")
    console.print(f"Output: {output_path}")
    if limit:
        console.print(f"[yellow]Limit: {limit} stocks (testing mode)[/yellow]")
    console.print()

    async def run():
        rate_limit = eodhd_config.get("rate_limit", {}).get("requests_per_second", 5)

        async with EODHDClient(eodhd_api_key, requests_per_second=rate_limit) as eodhd_client:
            fetcher = StockDataFetcher(
                eodhd_client=eodhd_client,
                stock_filter=stock_filter,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                stocks, issues, stats = await fetch_and_filter_stocks(
                    fetcher,
                    eodhd_client,
                    limit=limit,
                    progress=progress,
                )

            return stocks, issues, stats

    stocks, issues, stats = asyncio.run(run())

    if stocks:
        save_stocks_csv(stocks, output_path)
        save_quality_report(issues, quality_path)
        print_summary(stocks, issues, stats)
    else:
        console.print("[red]No stocks matched the filter criteria[/red]")
        console.print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
