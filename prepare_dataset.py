#!/usr/bin/env python3
"""Prepare neural-network datasets from dated Finviz Excel exports.

The snapshots stored in this repository contain one row per ticker and dozens of
columns copied from Finviz. Each file is named ``DD.MM.YYYY.xlsx`` and
represents the situation at the close of a Friday session. The training setup
requires 66 inputs (33 derived values from the "current" snapshot + the same 33
values from the snapshot that is four weeks older) and three targets describing
how the ticker performed 4, 13 and 26 weeks into the future.

The mapping between Excel columns (1-based) and neural network nodes is defined
below. When any required value is missing the entire row is discarded, with the
exception of nodes 6–9 where the specification demands an explicit ``0``
fallback.

For every snapshot that has the necessary neighbours the script:

* locates the file that is four weeks older (look-back),
* locates the files that are 4, 13 and 26 weeks newer (targets),
* matches rows by ticker symbol (column 2),
* computes the 33 nodes for both current and look-back rows, and
* reads target values from columns 48, 49 and 50 of the future snapshots.

Outputs (saved under ``--output-dir``):

``training_features.csv``
    The 66-node feature matrix.

``training_targets.csv``
    Three regression targets: 4w, 13w and 26w performance.

``training_metadata.csv``
    Ticker and timestamp provenance for each row, useful for debugging or
    cross-referencing with future snapshots.

Use ``--focus-date`` when you want to verify a single dataset before processing
all snapshots.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - guard for environments without pandas
    pd = None  # type: ignore[assignment]

DATE_PATTERN = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
MAGNITUDE_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
BOOL_MAP = {"yes": 1.0, "no": 0.0}
TARGET_COLUMN_INDEX = {4: 48, 13: 49, 26: 50}
NODE_COUNT = 33


@dataclass
class Snapshot:
    date: dt.date
    path: Path
    df: "pd.DataFrame"
    columns: List[str]


# -------------------------------------------------------------------------------------- helpers

def _coerce_numeric(value) -> float:
    """Convert various Finviz-formatted cells into floats."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text == "-":
        return math.nan

    lowered = text.lower()
    if lowered in BOOL_MAP:
        return BOOL_MAP[lowered]

    text = text.replace(",", "")

    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return math.nan

    suffix_match = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([KMBTkmbt])", text)
    if suffix_match:
        base = float(suffix_match.group(1))
        suffix = suffix_match.group(2).upper()
        return base * MAGNITUDE_SUFFIXES.get(suffix, 1.0)

    try:
        return float(text)
    except ValueError:
        return math.nan


def _parse_snapshot_date(path: Path) -> Optional[dt.date]:
    match = DATE_PATTERN.search(path.name)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _load_snapshot(path: Path) -> Tuple["pd.DataFrame", List[str]]:
    if pd is None:
        raise SystemExit("pandas is required. Install it via 'pip install pandas openpyxl'.")

    df = pd.read_excel(path, engine="openpyxl")
    df = df.dropna(how="all")

    if df.shape[1] < 80:
        raise ValueError(f"Soubor '{path}' musí obsahovat alespoň 80 sloupců.")

    ticker_col = df.columns[1]
    df = df.rename(columns={ticker_col: "Ticker"})
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""]
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    df = df.set_index("Ticker", drop=False)

    columns = list(df.columns)
    return df, columns


def _find_neighbor_index(
    current_idx: int,
    snapshots: Sequence[Snapshot],
    target_days: int,
    tolerance: int,
    direction: int,
) -> Optional[int]:
    current_date = snapshots[current_idx].date

    if direction > 0:
        iterable: Iterable[int] = range(current_idx + 1, len(snapshots))
        delta_fn = lambda idx: (snapshots[idx].date - current_date).days
    else:
        iterable = range(current_idx - 1, -1, -1)
        delta_fn = lambda idx: (current_date - snapshots[idx].date).days

    best_idx = None
    best_delta = None

    for idx in iterable:
        delta = delta_fn(idx)
        if delta < 0:
            continue
        if delta > target_days + tolerance:
            break
        distance = abs(delta - target_days)
        if distance <= tolerance and (best_idx is None or distance < best_delta):
            best_idx = idx
            best_delta = distance

    return best_idx


def _row_value(row: "pd.Series", columns: List[str], column_index: int) -> float:
    try:
        col_name = columns[column_index - 1]
    except IndexError:
        return math.nan
    return _coerce_numeric(row[col_name])


def _compute_nodes(row: "pd.Series", columns: List[str]) -> Optional[List[float]]:
    cache: Dict[int, float] = {}

    def col(idx: int) -> float:
        if idx not in cache:
            cache[idx] = _row_value(row, columns, idx)
        return cache[idx]

    def ratio(numerator_idx: int, denominator_idx: int) -> float:
        numerator = col(numerator_idx)
        denominator = col(denominator_idx)
        if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
            return math.nan
        return numerator / denominator

    def reciprocal(idx: int) -> float:
        value = col(idx)
        if math.isnan(value) or value == 0:
            return math.nan
        return 1.0 / value

    def reciprocal_or_zero(idx: int) -> float:
        value = col(idx)
        if math.isnan(value):
            return 0.0
        if value == 0:
            return math.nan
        return 1.0 / value

    def raw(idx: int) -> float:
        return col(idx)

    def raw_or_zero(idx: int) -> float:
        value = col(idx)
        if math.isnan(value):
            return 0.0
        return value

    nodes = [
        ratio(16, 80),
        reciprocal(5),
        ratio(17, 80),
        reciprocal(7),
        ratio(1, 80),
        reciprocal_or_zero(9),
        reciprocal_or_zero(10),
        raw_or_zero(14),
        raw_or_zero(14),
        raw(18),
        raw(19),
        raw(20),
        raw(21),
        raw(22),
        raw(23),
        raw(24),
        raw(30),
        raw(31),
        raw(32),
        raw(33),
        raw(37),
        raw(38),
        raw(39),
        raw(40),
        raw(41),
        raw(42),
        raw(43),
        raw(44),
        raw(45),
        raw(46),
        ratio(68, 3),
        raw(71),
        ratio(75, 80),
    ]

    if any(math.isnan(value) or math.isinf(value) for value in nodes):
        return None

    return nodes


def _extract_target(
    row: "pd.Series", columns: List[str], column_index: int
) -> Optional[float]:
    value = _row_value(row, columns, column_index)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def build_dataset(
    data_dir: Path,
    lookback_weeks: int,
    gap_tolerance: int,
    focus_date: Optional[dt.date] = None,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    snapshot_entries: List[Tuple[dt.date, Path]] = []
    for path in sorted(data_dir.glob("*.xls*")):
        snap_date = _parse_snapshot_date(path)
        if snap_date:
            snapshot_entries.append((snap_date, path))

    if len(snapshot_entries) < 2:
        raise RuntimeError("Jsou potřeba alespoň dva snapshoty.")

    snapshot_entries.sort(key=lambda item: item[0])

    snapshots: List[Snapshot] = []
    for snap_date, snap_path in snapshot_entries:
        df, columns = _load_snapshot(snap_path)
        snapshots.append(Snapshot(date=snap_date, path=snap_path, df=df, columns=columns))

    feature_rows: List[List[float]] = []
    target_rows: List[List[float]] = []
    metadata_rows: List[Dict[str, str]] = []

    lookback_days = lookback_weeks * 7

    for idx, snapshot in enumerate(snapshots):
        if focus_date and snapshot.date != focus_date:
            continue

        lookback_idx = _find_neighbor_index(idx, snapshots, lookback_days, gap_tolerance, direction=-1)
        if lookback_idx is None:
            logging.debug("Přeskakuji %s – chybí starší snapshot.", snapshot.path.name)
            continue

        future_indices: Dict[int, int] = {}
        for weeks, column_idx in TARGET_COLUMN_INDEX.items():
            target_idx = _find_neighbor_index(
                idx, snapshots, weeks * 7, gap_tolerance, direction=1
            )
            if target_idx is None:
                break
            future_indices[weeks] = target_idx

        if len(future_indices) != len(TARGET_COLUMN_INDEX):
            logging.debug("Přeskakuji %s – chybí některý z budoucích snapshotů.", snapshot.path.name)
            continue

        lookback_snapshot = snapshots[lookback_idx]
        future_snapshots = {weeks: snapshots[i] for weeks, i in future_indices.items()}

        tickers = set(snapshot.df.index) & set(lookback_snapshot.df.index)
        for future in future_snapshots.values():
            tickers &= set(future.df.index)

        if not tickers:
            logging.debug("%s – žádné společné tickery.", snapshot.path.name)
            continue

        for ticker in sorted(tickers):
            current_row = snapshot.df.loc[ticker]
            lookback_row = lookback_snapshot.df.loc[ticker]

            current_nodes = _compute_nodes(current_row, snapshot.columns)
            if current_nodes is None:
                continue

            lookback_nodes = _compute_nodes(lookback_row, lookback_snapshot.columns)
            if lookback_nodes is None:
                continue

            targets: Dict[int, float] = {}
            target_valid = True
            for weeks, future_snapshot in future_snapshots.items():
                column_index = TARGET_COLUMN_INDEX[weeks]
                future_row = future_snapshot.df.loc[ticker]
                value = _extract_target(future_row, future_snapshot.columns, column_index)
                if value is None:
                    target_valid = False
                    break
                targets[weeks] = value

            if not target_valid:
                continue

            feature_rows.append(current_nodes + lookback_nodes)
            target_rows.append([targets[4], targets[13], targets[26]])
            metadata_rows.append(
                {
                    "Ticker": ticker,
                    "base_snapshot": snapshot.date.isoformat(),
                    "lookback_snapshot": lookback_snapshot.date.isoformat(),
                    "target_4w_snapshot": future_snapshots[4].date.isoformat(),
                    "target_13w_snapshot": future_snapshots[13].date.isoformat(),
                    "target_26w_snapshot": future_snapshots[26].date.isoformat(),
                }
            )

    if not feature_rows:
        raise RuntimeError("Nepodařilo se vytvořit žádný tréninkový vzorek – zkontrolujte vstupní data.")

    node_names = [f"node_{i:02d}" for i in range(1, NODE_COUNT + 1)]
    feature_columns = [f"{name}_curr" for name in node_names] + [
        f"{name}_prev" for name in node_names
    ]

    features_df = pd.DataFrame(feature_rows, columns=feature_columns)
    targets_df = pd.DataFrame(target_rows, columns=["target_4w", "target_13w", "target_26w"])
    metadata_df = pd.DataFrame(metadata_rows)

    logging.info(
        "Hotovo: %s řádků, %s vstupních atributů, %s cílových hodnot.",
        len(features_df),
        len(feature_columns),
        targets_df.shape[1],
    )

    return features_df, targets_df, metadata_df


# -------------------------------------------------------------------------------------- CLI

def _parse_focus_date(value: Optional[str]) -> Optional[dt.date]:
    if value is None:
        return None
    match = DATE_PATTERN.fullmatch(value.strip())
    if not match:
        raise argparse.ArgumentTypeError("Očekáván formát DD.MM.RRRR.")
    day, month, year = map(int, match.groups())
    try:
        return dt.date(year, month, day)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Složka se snapshoty (.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Kam uložit CSV soubory.",
    )
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=4,
        help="Počet týdnů mezi aktuálním a starším snapshotem.",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=int,
        default=7,
        help="Povolená odchylka (ve dnech) při hledání sousedních snapshotů.",
    )
    parser.add_argument(
        "--focus-date",
        type=_parse_focus_date,
        help="Volitelně zpracuj pouze snapshot s konkrétním datem (DD.MM.RRRR).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Úroveň logování.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not args.data_dir.exists():
        raise SystemExit(f"Složka '{args.data_dir}' neexistuje.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_df, targets_df, metadata_df = build_dataset(
        data_dir=args.data_dir,
        lookback_weeks=args.lookback_weeks,
        gap_tolerance=args.gap_tolerance,
        focus_date=args.focus_date,
    )

    features_path = args.output_dir / "training_features.csv"
    targets_path = args.output_dir / "training_targets.csv"
    metadata_path = args.output_dir / "training_metadata.csv"

    features_df.to_csv(features_path, index=False)
    targets_df.to_csv(targets_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    logging.info("Výstupy uloženy do %s, %s a %s", features_path, targets_path, metadata_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
