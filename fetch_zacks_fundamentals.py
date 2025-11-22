"""
Utility to pull fundamental chart data from Zacks (e.g., EPS diluted TTM) and store it as CSV.

The script targets URLs such as:
    https://www.zacks.com/stock/chart/TSLA/fundamental/eps-diluted-ttm
and attempts two extraction strategies:
1) Direct JSON feed at `https://www.zacks.com/includes/fundamental_charts.php`.
2) Fallback scraping of inline JSON embedded in the HTML chart page.

Usage examples:
    python fetch_zacks_fundamentals.py --ticker TSLA --metric eps-diluted-ttm --out data/tsla_eps.csv
    python fetch_zacks_fundamentals.py --tickers TSLA,AAPL,MSFT --metric eps-diluted-ttm --out-dir data/zacks

Notes:
- Zacks sometimes blocks requests that look like bots. Supplying a User-Agent and, if needed, a proxy
  (HTTP(S)_PROXY environment variables) improves the odds of success.
- If the parser cannot locate usable data, the script will save the HTML to disk to help adjust regexes
  without re-downloading.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; StocksDataBot/1.0)"


@dataclass
class Point:
    date: datetime
    value: float


def fetch_bytes(url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=15) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} while fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc.reason}") from exc


def try_json_payload(raw: str) -> Optional[Sequence]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, (list, dict)) else None
    except json.JSONDecodeError:
        return None


def _coerce_points(sequence) -> List[Point]:
    # direct list of [date, value]
    if isinstance(sequence, list) and sequence and all(isinstance(item, list) and len(item) >= 2 for item in sequence):
        return _convert_pairs(sequence)

    # list of dicts with common keys
    if isinstance(sequence, list) and sequence and all(isinstance(item, dict) for item in sequence):
        keys = sequence[0].keys()
        date_key = next((k for k in keys if k.lower() in {"date", "x", "label"}), None)
        value_key = next((k for k in keys if k.lower() in {"value", "y", "eps", "val"}), None)
        if date_key and value_key:
            pairs = [[item[date_key], item[value_key]] for item in sequence if date_key in item and value_key in item]
            return _convert_pairs(pairs)

    # dict container with nested series/data keys
    if isinstance(sequence, dict):
        for key in ("series", "data", "items", "values"):
            if key in sequence:
                return _coerce_points(sequence[key])
    return []


def _convert_pairs(pairs: Iterable[Sequence]) -> List[Point]:
    points: List[Point] = []
    for raw_date, raw_value, *_ in pairs:
        try:
            date = _parse_date(raw_date)
            value = float(raw_value)
            points.append(Point(date=date, value=value))
        except Exception:
            continue
    return points


def _parse_date(value) -> datetime:
    # Accept ISO, US, or quarter labels.
    if isinstance(value, (int, float)):
        # Assume timestamp in seconds
        return datetime.utcfromtimestamp(value)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m/%d/%y", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    # Fallback: quarter format like Q1 2023
    match = re.match(r"Q([1-4])\s+(\d{4})", text)
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)
    raise ValueError(f"Unrecognized date format: {value}")


def extract_from_html(html: str) -> List[Point]:
    # Look for obvious JSON blobs inside script tags
    patterns = [
        r"fundamentalChartData\s*=\s*(\{.*?\});",
        r"chartData\s*=\s*(\[.*?\]);",
        r"series\s*:\s*(\[.*?\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        candidate_raw = match.group(1)
        payload = try_json_payload(candidate_raw)
        if payload:
            points = _coerce_points(payload)
            if points:
                return points
    return []


def extract_points(ticker: str, metric: str, *, user_agent: str, save_failed_html: pathlib.Path) -> List[Point]:
    api_url = f"https://www.zacks.com/includes/fundamental_charts.php?t={ticker}&chart={metric}"
    html_url = f"https://www.zacks.com/stock/chart/{ticker}/fundamental/{metric}"

    # Try direct feed first
    try:
        api_bytes = fetch_bytes(api_url, user_agent=user_agent)
        payload = try_json_payload(api_bytes.decode("utf-8", errors="ignore"))
        if payload:
            points = _coerce_points(payload)
            if points:
                return points
    except RuntimeError as exc:
        print(f"[warn] {exc}")

    # Fallback to HTML scrape
    try:
        html_bytes = fetch_bytes(html_url, user_agent=user_agent)
        html_text = html_bytes.decode("utf-8", errors="ignore")
        points = extract_from_html(html_text)
        if points:
            return points
        save_failed_html.write_text(html_text, encoding="utf-8")
        raise RuntimeError(
            "Failed to locate chart data in HTML; saved the page to " f"{save_failed_html} for manual regex tuning."
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Could not extract data for {ticker}: {exc}") from exc


def write_csv(path: pathlib.Path, points: List[Point]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = sorted(points, key=lambda p: p.date)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "value"])
        for p in points:
            writer.writerow([p.date.date().isoformat(), f"{p.value:.6f}"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="Single ticker to download (mutually exclusive with --tickers)")
    parser.add_argument("--tickers", help="Comma-separated tickers to download")
    parser.add_argument("--metric", default="eps-diluted-ttm", help="Fundamental chart metric slug (default: eps-diluted-ttm)")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Override User-Agent header")
    parser.add_argument("--out", dest="out_file", help="Output CSV file (for single ticker)")
    parser.add_argument("--out-dir", default="data/zacks", help="Directory for multi-ticker output")
    parser.add_argument(
        "--html-dump-dir",
        default="data/zacks_failed_html",
        help="Where to save HTML when parsing fails (debug aid)",
    )
    return parser.parse_args(argv)


def _ensure_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.ticker:
        tickers = [args.ticker]
    if args.tickers:
        tickers.extend([t.strip() for t in args.tickers.split(",") if t.strip()])
    if not tickers:
        raise SystemExit("Provide --ticker or --tickers")
    return tickers


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tickers = _ensure_tickers(args)
    html_dump_dir = pathlib.Path(args.html_dump_dir)
    html_dump_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        out_path = (
            pathlib.Path(args.out_file)
            if args.out_file and len(tickers) == 1
            else pathlib.Path(args.out_dir) / f"{ticker.lower()}_{args.metric.replace('/', '-')}.csv"
        )
        failed_html = html_dump_dir / f"{ticker.lower()}_{args.metric.replace('/', '-')}.html"
        try:
            points = extract_points(ticker, args.metric, user_agent=args.user_agent, save_failed_html=failed_html)
            if not points:
                raise RuntimeError("No points extracted")
            write_csv(out_path, points)
            print(f"[ok] {ticker} -> {out_path} ({len(points)} points)")
        except Exception as exc:
            print(f"[fail] {ticker}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
