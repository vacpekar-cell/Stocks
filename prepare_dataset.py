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

RELAXED_DATE_PATTERN = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:[ _-].*)?$")
MAGNITUDE_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
BOOL_MAP = {"yes": 1.0, "no": 0.0}
TARGET_COLUMN_INDEX = {4: 48, 13: 49, 26: 50}
WEEKEND_TOLERANCE_DAYS = 2  # allow Saturday/Sunday captures to pair with Friday baselines
NODE_COUNT = 33


@dataclass
class Snapshot:
    date: dt.date
    path: Path
    df: "pd.DataFrame"
    columns: List[str]


class SnapshotFormatError(ValueError):
    """Raised when a snapshot file is missing required columns or formatting."""


@dataclass
class NodeComputation:
    values: Optional[List[float]]
    failed_node: Optional[int] = None


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
    stem = path.stem
    match = RELAXED_DATE_PATTERN.match(stem)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        snap_date = dt.date(year, month, day)
    except ValueError:
        return None

    canonical_name = snap_date.strftime("%d.%m.%Y")
    if stem != canonical_name:
        logging.info(
            "Soubor %s interpretován jako %s (upraven název)",
            path.name,
            canonical_name,
        )

    return snap_date


def _normalize_header(value: object) -> str:
    return str(value).strip().lower()


def _detect_ticker_column(columns: Sequence[str]) -> Optional[str]:
    """Return the column whose header indicates it stores ticker symbols."""

    normalized_targets = {"ticker", "ticker symbol", "symbol"}
    for name in columns:
        normalized = _normalize_header(name)
        if normalized in normalized_targets or "ticker" in normalized:
            return name
    return None


def _load_snapshot(path: Path) -> Tuple["pd.DataFrame", List[str]]:
    if pd is None:
        raise SystemExit("pandas is required. Install it via 'pip install pandas openpyxl'.")

    df = pd.read_excel(path, engine="openpyxl")
    df = df.dropna(how="all")

    # Some newer exports inserted the "Sector" and "Industry" columns (typically
    # near the ticker column). Remove them wherever they appear so that the
    # remaining columns keep the original numbering expected by the node
    # specification.
    columns = list(df.columns)
    drop_candidates = [
        name for name in columns if _normalize_header(name) in {"sector", "industry"}
    ]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)
        columns = list(df.columns)

    if df.shape[1] < 80:
        raise SnapshotFormatError(f"Soubor '{path.name}' musí obsahovat alespoň 80 sloupců.")

    ticker_col = _detect_ticker_column(df.columns)
    if ticker_col is None:
        if len(df.columns) < 2:
            raise SnapshotFormatError(
                f"Soubor '{path.name}' musí obsahovat sloupec s ticker symboly."
            )
        ticker_col = df.columns[1]
        logging.debug(
            "%s – nenašel jsem sloupec s názvem 'Ticker', používám druhý sloupec (%s)",
            path.name,
            ticker_col,
        )
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
    direction: int,
    tolerance_days: int = WEEKEND_TOLERANCE_DAYS,
) -> Optional[int]:
    current_date = snapshots[current_idx].date

    if direction > 0:
        iterable: Iterable[int] = range(current_idx + 1, len(snapshots))
        delta_fn = lambda idx: (snapshots[idx].date - current_date).days
    else:
        iterable = range(current_idx - 1, -1, -1)
        delta_fn = lambda idx: (current_date - snapshots[idx].date).days

    for idx in iterable:
        delta = delta_fn(idx)
        if delta < 0:
            continue
        if delta > target_days + tolerance_days:
            break
        if abs(delta - target_days) <= tolerance_days:
            return idx

    return None


def _neighbor_window(
    base_date: dt.date,
    target_days: int,
    direction: int,
    tolerance_days: int = WEEKEND_TOLERANCE_DAYS,
) -> Tuple[dt.date, dt.date]:
    """Return the acceptable date range for a neighbor snapshot."""

    delta = dt.timedelta(days=target_days)
    tolerance = dt.timedelta(days=tolerance_days)
    if direction < 0:
        center = base_date - delta
    else:
        center = base_date + delta
    return center - tolerance, center + tolerance


def _log_missing_neighbor(
    snapshot: Snapshot,
    label: str,
    target_days: int,
    direction: int,
    tolerance_days: int = WEEKEND_TOLERANCE_DAYS,
) -> None:
    window_start, window_end = _neighbor_window(
        snapshot.date, target_days, direction, tolerance_days
    )
    logging.warning(
        "%s (%s) – nenalezen %s snapshot v intervalu %s až %s.",
        snapshot.path.name,
        snapshot.date.isoformat(),
        label,
        window_start.isoformat(),
        window_end.isoformat(),
    )


def _format_node_failures(counts: Dict[int, int]) -> str:
    items = sorted(((idx, cnt) for idx, cnt in counts.items() if cnt), key=lambda item: item[1], reverse=True)
    if not items:
        return "žádné chyby"
    return ", ".join(f"node_{idx:02d}: {cnt}" for idx, cnt in items)


def _row_value(row: "pd.Series", columns: List[str], column_index: int) -> float:
    try:
        col_name = columns[column_index - 1]
    except IndexError:
        return math.nan
    return _coerce_numeric(row[col_name])


def _compute_nodes(row: "pd.Series", columns: List[str]) -> NodeComputation:
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

    for idx, value in enumerate(nodes, start=1):
        if math.isnan(value) or math.isinf(value):
            return NodeComputation(values=None, failed_node=idx)

    return NodeComputation(values=nodes)


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
    skipped_snapshots: List[str] = []
    for snap_date, snap_path in snapshot_entries:
        try:
            df, columns = _load_snapshot(snap_path)
        except SnapshotFormatError as exc:
            skipped_snapshots.append(f"{snap_path.name}: {exc}")
            logging.warning("Přeskakuji %s – %s", snap_path.name, exc)
            continue
        except Exception as exc:  # pragma: no cover - unexpected read errors
            skipped_snapshots.append(f"{snap_path.name}: {exc}")
            logging.warning("Přeskakuji %s – nepodařilo se načíst (%s)", snap_path.name, exc)
            continue
        snapshots.append(Snapshot(date=snap_date, path=snap_path, df=df, columns=columns))

    if len(snapshots) < 2:
        reason = "Nepodařilo se načíst dostatek snapshotů."
        if skipped_snapshots:
            reason += " Problémové soubory: " + "; ".join(skipped_snapshots)
        raise RuntimeError(reason)

    snapshots_by_date: Dict[dt.date, List[Snapshot]] = {}
    for snapshot in snapshots:
        snapshots_by_date.setdefault(snapshot.date, []).append(snapshot)

    canonical_snapshots: List[Snapshot] = []
    for snap_date in sorted(snapshots_by_date):
        candidates = snapshots_by_date[snap_date]
        if len(candidates) > 1:
            candidates.sort(
                key=lambda snap: (snap.df.shape[0], snap.df.shape[1], snap.path.name),
                reverse=True,
            )
            chosen = candidates[0]
            logging.info(
                "%s – nalezeno %s souborů, používám %s",
                snap_date.isoformat(),
                len(candidates),
                chosen.path.name,
            )
        else:
            chosen = candidates[0]
        canonical_snapshots.append(chosen)

    snapshots = canonical_snapshots

    feature_rows: List[List[float]] = []
    target_rows: List[List[float]] = []
    metadata_rows: List[Dict[str, str]] = []

    lookback_days = lookback_weeks * 7

    skip_reason_counts = {
        "missing_lookback": 0,
        "missing_future": {weeks: 0 for weeks in TARGET_COLUMN_INDEX},
        "missing_tickers": 0,
    }
    ticker_skip_counts = {
        "current_nodes": 0,
        "lookback_nodes": 0,
        "targets": 0,
    }
    node_failure_counts = {
        "current": {i: 0 for i in range(1, NODE_COUNT + 1)},
        "lookback": {i: 0 for i in range(1, NODE_COUNT + 1)},
    }
    target_failure_counts = {weeks: 0 for weeks in TARGET_COLUMN_INDEX}

    for idx, snapshot in enumerate(snapshots):
        if focus_date and snapshot.date != focus_date:
            continue

        lookback_idx = _find_neighbor_index(idx, snapshots, lookback_days, direction=-1)
        if lookback_idx is None:
            skip_reason_counts["missing_lookback"] += 1
            _log_missing_neighbor(snapshot, "starší", lookback_days, direction=-1)
            continue

        future_indices: Dict[int, int] = {}
        for weeks, column_idx in TARGET_COLUMN_INDEX.items():
            target_idx = _find_neighbor_index(
                idx, snapshots, weeks * 7, direction=1
            )
            if target_idx is None:
                break
            future_indices[weeks] = target_idx

        if len(future_indices) != len(TARGET_COLUMN_INDEX):
            missing_weeks = sorted(set(TARGET_COLUMN_INDEX) - set(future_indices))
            for weeks in missing_weeks:
                skip_reason_counts["missing_future"][weeks] += 1
                _log_missing_neighbor(snapshot, f"budoucí ({weeks}t)", weeks * 7, direction=1)
            continue

        lookback_snapshot = snapshots[lookback_idx]
        future_snapshots = {weeks: snapshots[i] for weeks, i in future_indices.items()}

        tickers = set(snapshot.df.index) & set(lookback_snapshot.df.index)
        for future in future_snapshots.values():
            tickers &= set(future.df.index)

        if not tickers:
            skip_reason_counts["missing_tickers"] += 1
            logging.warning("%s (%s) – žádné společné tickery se všemi snapshoty.", snapshot.path.name, snapshot.date.isoformat())
            continue

        shared_tickers = len(tickers)
        snapshot_success = 0
        snapshot_current_fail = 0
        snapshot_lookback_fail = 0
        snapshot_target_fail = 0

        for ticker in sorted(tickers):
            current_row = snapshot.df.loc[ticker]
            lookback_row = lookback_snapshot.df.loc[ticker]

            current_nodes = _compute_nodes(current_row, snapshot.columns)
            if current_nodes.values is None:
                ticker_skip_counts["current_nodes"] += 1
                snapshot_current_fail += 1
                if current_nodes.failed_node:
                    node_failure_counts["current"][current_nodes.failed_node] += 1
                continue

            lookback_nodes = _compute_nodes(lookback_row, lookback_snapshot.columns)
            if lookback_nodes.values is None:
                ticker_skip_counts["lookback_nodes"] += 1
                snapshot_lookback_fail += 1
                if lookback_nodes.failed_node:
                    node_failure_counts["lookback"][lookback_nodes.failed_node] += 1
                continue

            targets: Dict[int, float] = {}
            target_valid = True
            for weeks, future_snapshot in future_snapshots.items():
                column_index = TARGET_COLUMN_INDEX[weeks]
                future_row = future_snapshot.df.loc[ticker]
                value = _extract_target(future_row, future_snapshot.columns, column_index)
                if value is None:
                    target_valid = False
                    target_failure_counts[weeks] += 1
                    break
                targets[weeks] = value

            if not target_valid:
                ticker_skip_counts["targets"] += 1
                snapshot_target_fail += 1
                continue

            feature_rows.append(current_nodes.values + lookback_nodes.values)
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
            snapshot_success += 1

        if snapshot_success == 0:
            logging.warning(
                "%s (%s) – žádné platné tickery (společné: %s, aktuální selhaly: %s, starší selhaly: %s, cíle selhaly: %s).",
                snapshot.path.name,
                snapshot.date.isoformat(),
                shared_tickers,
                snapshot_current_fail,
                snapshot_lookback_fail,
                snapshot_target_fail,
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

    logging.info(
        "Diagnostika snapshotů – bez staršího: %s, budoucí 4t: %s, 13t: %s, 26t: %s, bez tickerů: %s.",
        skip_reason_counts["missing_lookback"],
        skip_reason_counts["missing_future"][4],
        skip_reason_counts["missing_future"][13],
        skip_reason_counts["missing_future"][26],
        skip_reason_counts["missing_tickers"],
    )
    logging.info(
        "Diagnostika tickerů – aktuální uzly selhaly: %s, starší uzly selhaly: %s, chybějící cíle: %s.",
        ticker_skip_counts["current_nodes"],
        ticker_skip_counts["lookback_nodes"],
        ticker_skip_counts["targets"],
    )
    logging.info(
        "Diagnostika uzlů (aktuální) – %s",
        _format_node_failures(node_failure_counts["current"]),
    )
    logging.info(
        "Diagnostika uzlů (starší) – %s",
        _format_node_failures(node_failure_counts["lookback"]),
    )
    logging.info(
        "Diagnostika cílů – 4t: %s, 13t: %s, 26t: %s",
        target_failure_counts[4],
        target_failure_counts[13],
        target_failure_counts[26],
    )

    return features_df, targets_df, metadata_df


# -------------------------------------------------------------------------------------- CLI

def _parse_focus_date(value: Optional[str]) -> Optional[dt.date]:
    if value is None:
        return None
    match = RELAXED_DATE_PATTERN.fullmatch(value.strip())
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
