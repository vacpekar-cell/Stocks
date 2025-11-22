"""
Text-based helper for downloading FMP fundamentals without memorizing CLI flags.

The UI asks for tickers, metrics, and output options, then reuses the existing
`fetch_fmp_fundamentals` helpers to download and save CSV files.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, List

import requests

from fetch_fmp_fundamentals import (
    compute_ttm_series,
    extract_metric_series,
    fetch_income_statements,
    write_csv,
)


@dataclass
class UiConfig:
    api_key: str
    tickers: List[str]
    metrics: List[str]
    period: str
    limit: int
    ttm: bool
    out_dir: str


DEFAULT_TICKERS = ["TSLA"]
DEFAULT_METRICS = ["epsdiluted"]
DEFAULT_PERIOD = "quarter"
DEFAULT_LIMIT = 160
DEFAULT_OUT_DIR = "data/fmp"
DEFAULT_TTM = True


def prompt_list(prompt: str, default: List[str]) -> List[str]:
    raw = input(f"{prompt} [{','.join(default)}]: ").strip()
    if not raw:
        return default
    items = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return items or default


def prompt_period(default: str) -> str:
    raw = input(f"Period (quarter/annual) [{default}]: ").strip().lower()
    if raw in {"quarter", "annual"}:
        return raw
    return default


def prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} ({suffix}) ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default


def resolve_api_key() -> str:
    env_key = os.environ.get("FMP_API_KEY")
    if env_key:
        print("Using FMP_API_KEY from environment.")
        return env_key
    manual = input("Enter FMP API key: ").strip()
    if manual:
        return manual
    raise SystemExit("Missing API key. Set FMP_API_KEY or enter it when prompted.")


def gather_config() -> UiConfig:
    print("\nFMP fundamentals downloader (interactive)")
    print("Press Enter to accept defaults in brackets. Separate multiple items with commas.\n")

    tickers = prompt_list("Tickers", DEFAULT_TICKERS)
    metrics = prompt_list("Metrics", DEFAULT_METRICS)
    period = prompt_period(DEFAULT_PERIOD)
    limit = prompt_int("Number of statements to fetch", DEFAULT_LIMIT)
    ttm = prompt_bool("Compute trailing-twelve-month (TTM) sums?", DEFAULT_TTM)
    out_dir = input(f"Output directory [{DEFAULT_OUT_DIR}]: ").strip() or DEFAULT_OUT_DIR
    api_key = resolve_api_key()

    return UiConfig(
        api_key=api_key,
        tickers=[t.upper() for t in tickers],
        metrics=metrics,
        period=period,
        limit=limit,
        ttm=ttm,
        out_dir=out_dir,
    )


def download_metric(
    symbol: str,
    metric: str,
    statements: List[dict],
    *,
    ttm: bool,
    out_dir: str,
) -> str:
    series: Iterable[dict]
    metric_name = metric
    if ttm:
        series = compute_ttm_series(statements, metric)
        metric_name = f"{metric}_ttm"
    else:
        series = extract_metric_series(statements, metric)
    return write_csv(symbol, metric_name, series, out_dir)


def run_downloads(config: UiConfig) -> None:
    print(
        f"\nFetching {config.period} income statements (limit {config.limit}) "
        f"for {len(config.tickers)} ticker(s)..."
    )
    for symbol in config.tickers:
        try:
            statements = fetch_income_statements(
                symbol, config.api_key, limit=config.limit, period=config.period
            )
        except requests.HTTPError as exc:
            print(f"[HTTP {symbol}] {exc}")
            continue
        except requests.RequestException as exc:
            print(f"[Network {symbol}] {exc}")
            continue

        for metric in config.metrics:
            try:
                path = download_metric(symbol, metric, statements, ttm=config.ttm, out_dir=config.out_dir)
                print(f"Saved {symbol} {metric} -> {path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[Error {symbol}:{metric}] {exc}")


if __name__ == "__main__":
    cfg = gather_config()
    run_downloads(cfg)
    print("\nDone.")
    sys.exit(0)
