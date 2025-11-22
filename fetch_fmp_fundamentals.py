"""
Download long-run fundamental time series from Financial Modeling Prep (FMP).

The script focuses on income-statement metrics (e.g., epsdiluted) and can compute
trailing-twelve-month (TTM) sums from quarterly reports. It writes tidy CSV files
with date/value pairs for easy downstream use.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download fundamental metrics (e.g., epsdiluted) from FMP and optionally compute TTM values."
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list of tickers (e.g., TSLA,AAPL).",
    )
    parser.add_argument(
        "--metrics",
        default="epsdiluted",
        help=(
            "Comma-separated metric fields from the income-statement endpoint (e.g., epsdiluted,ebit). "
            "See https://site.financialmodelingprep.com/developer/docs/#Income-Statement"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/fmp",
        help="Directory where per-ticker CSV files will be written.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="FMP API key. If omitted, the script uses the FMP_API_KEY environment variable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=160,
        help="Number of quarterly statements to request (160 ~ 40 years).",
    )
    parser.add_argument(
        "--period",
        default="quarter",
        choices=["quarter", "annual"],
        help="Period granularity for the income statements.",
    )
    parser.add_argument(
        "--ttm",
        action="store_true",
        help="If set, compute trailing-twelve-month sums for each metric.",
    )
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set FMP_API_KEY.")
    return api_key


def fetch_income_statements(symbol: str, api_key: str, *, limit: int, period: str) -> List[Dict]:
    url = (
        f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
        f"?period={period}&limit={limit}&apikey={api_key}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected response for {symbol}: {data}")
    return data


def compute_ttm_series(rows: List[Dict], metric: str) -> Iterable[Dict[str, Optional[float]]]:
    # FMP returns the newest first; reverse to chronological order for rolling windows.
    chron_rows = list(reversed(rows))
    window: List[float] = []
    for entry in chron_rows:
        raw_value = entry.get(metric)
        if raw_value is None:
            window.clear()
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            window.clear()
            continue
        window.append(numeric)
        if len(window) > 4:
            window.pop(0)
        ttm_value = sum(window) if len(window) == 4 else None
        yield {"date": entry.get("date"), f"{metric}_ttm": ttm_value}


def extract_metric_series(rows: List[Dict], metric: str) -> Iterable[Dict[str, Optional[float]]]:
    chron_rows = list(reversed(rows))
    for entry in chron_rows:
        raw_value = entry.get(metric)
        try:
            numeric = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            numeric = None
        yield {"date": entry.get("date"), metric: numeric}


def write_csv(symbol: str, metric: str, rows: Iterable[Dict[str, Optional[float]]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{symbol}_{metric}.csv"
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["date", metric]
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get("date"), row.get(metric)])
    return path


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        raise SystemExit("No tickers provided.")
    if not metrics:
        raise SystemExit("No metrics provided.")

    print(f"Downloading {args.period} income statements for {len(tickers)} tickers...")

    for symbol in tickers:
        statements = fetch_income_statements(symbol, api_key, limit=args.limit, period=args.period)
        for metric in metrics:
            if args.ttm:
                series = compute_ttm_series(statements, metric)
                metric_name = f"{metric}_ttm"
            else:
                series = extract_metric_series(statements, metric)
                metric_name = metric
            path = write_csv(symbol, metric_name, series, args.out_dir)
            print(f"Saved {symbol} {metric_name} -> {path}")


if __name__ == "__main__":
    main()
