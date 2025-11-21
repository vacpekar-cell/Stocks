#!/usr/bin/env python3
"""Assemble a prediction-ready feature matrix from the newest snapshot pair.

This utility directly reuses ``prepare_dataset._compute_nodes`` so that any
adjustment to the node definitions automatically carries over to the prediction
path. It processes only the most recent snapshot and its look-back neighbour
that is exactly four weeks older (with the same weekend tolerance). The
resulting CSV matches the 66-input layout expected by
``neural_network_training_with_ui.py`` for inference, and an auxiliary Excel
file is emitted that retains only the rows used for the prediction run.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency guard
    pd = None  # type: ignore[assignment]

import prepare_dataset as prep


def _collect_dated_paths(data_dir: Path) -> List[Tuple[pd.Timestamp, Path]]:
    """Return dated snapshot candidates while suppressing noisy info logs."""

    if pd is None:
        raise SystemExit("pandas is required. Install it via 'pip install pandas openpyxl'.")

    entries: List[Tuple[pd.Timestamp, Path]] = []
    previous_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        for path in sorted(data_dir.glob("*.xls*")):
            snap_date = prep._parse_snapshot_date(path)  # type: ignore[attr-defined]
            if snap_date:
                entries.append((pd.Timestamp(snap_date), path))
    finally:
        logging.disable(previous_disable)

    entries.sort(key=lambda item: item[0])
    return entries


def _load_best_snapshot(date: pd.Timestamp, paths: List[Path]) -> prep.Snapshot:
    """Load the richest snapshot for a given date, warning on format errors."""

    candidates: List[prep.Snapshot] = []
    for snap_path in paths:
        try:
            df, columns, column_offset = prep._load_snapshot(snap_path, date.date())  # type: ignore[attr-defined]
        except prep.SnapshotFormatError as exc:  # type: ignore[attr-defined]
            logging.warning("Přeskakuji %s – %s", snap_path.name, exc)
            continue
        candidates.append(prep.Snapshot(date=date.date(), path=snap_path, df=df, columns=columns, column_offset=column_offset))

    if not candidates:
        raise RuntimeError(f"Žádný snapshot {date.date().isoformat()} nešel načíst kvůli formátu.")

    candidates.sort(key=lambda snap: (snap.df.shape[0], snap.df.shape[1], snap.path.name), reverse=True)
    return candidates[0]


def _discover_snapshots(data_dir: Path) -> List[prep.Snapshot]:
    entries = _collect_dated_paths(data_dir)
    if not entries:
        raise RuntimeError("Nebyl nalezen žádný snapshot soubor.")

    by_date: dict[pd.Timestamp, List[Path]] = {}
    for snap_ts, snap_path in entries:
        by_date.setdefault(snap_ts, []).append(snap_path)

    newest_date = max(by_date)

    lookback_date: Optional[pd.Timestamp] = None
    best_delta = None
    for candidate_date in by_date:
        if candidate_date >= newest_date:
            continue
        delta = (newest_date - candidate_date).days
        if abs(delta - 28) <= prep.WEEKEND_TOLERANCE_DAYS:  # type: ignore[attr-defined]
            score = abs(delta - 28)
            if best_delta is None or score < best_delta:
                best_delta = score
                lookback_date = candidate_date

    if lookback_date is None:
        raise RuntimeError(
            f"Pro nejnovější snapshot {newest_date.date().isoformat()} nebyl nalezen soubor 4 týdny zpět."
        )

    newest_snap = _load_best_snapshot(newest_date, by_date[newest_date])
    lookback_snap = _load_best_snapshot(lookback_date, by_date[lookback_date])

    return [lookback_snap, newest_snap]


def _current_and_lookback(snapshots: Sequence[prep.Snapshot]) -> Tuple[prep.Snapshot, prep.Snapshot]:
    if len(snapshots) < 2:
        raise RuntimeError("Je potřeba alespoň dvojice snapshotů.")

    current_idx = len(snapshots) - 1
    lookback_idx = prep._find_neighbor_index(current_idx, snapshots, target_days=28, direction=-1)  # type: ignore[attr-defined]
    if lookback_idx is None:
        raise RuntimeError(
            f"Pro nejnovější snapshot {snapshots[current_idx].date.isoformat()} nebyl nalezen soubor 4 týdny zpět."
        )
    return snapshots[current_idx], snapshots[lookback_idx]


def _compute_feature_rows(current: prep.Snapshot, lookback: prep.Snapshot):
    common = current.df.index.intersection(lookback.df.index)
    rows: List[List[float]] = []
    used_tickers: List[str] = []

    for ticker in common:
        cur_row = current.df.loc[ticker]
        lb_row = lookback.df.loc[ticker]

        cur_nodes = prep._compute_nodes(cur_row, current.columns, current.column_offset)  # type: ignore[attr-defined]
        if cur_nodes.values is None:
            continue
        lb_nodes = prep._compute_nodes(lb_row, lookback.columns, lookback.column_offset)  # type: ignore[attr-defined]
        if lb_nodes.values is None:
            continue

        rows.append(cur_nodes.values + lb_nodes.values)
        used_tickers.append(ticker)

    return rows, used_tickers


def _write_simulated_excel(current: prep.Snapshot, used_tickers: List[str], output_dir: Path):
    filtered = current.df.loc[used_tickers]
    suffix_path = current.path.with_name(current.path.stem + " sim" + current.path.suffix)
    dest = output_dir / suffix_path.name
    filtered.to_excel(dest, index=False)
    return dest


def generate_prediction_inputs(data_dir: Path, output_dir: Path):
    snapshots = _discover_snapshots(data_dir)
    current, lookback = _current_and_lookback(snapshots)

    logging.info("Aktuální snapshot: %s", current.path.name)
    logging.info("Look-back snapshot: %s", lookback.path.name)

    feature_rows, used_tickers = _compute_feature_rows(current, lookback)
    if not feature_rows:
        raise RuntimeError("Žádné použitelné řádky pro predikci.")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "prediction_features.csv"
    pd.DataFrame(feature_rows).to_csv(features_path, index=False, header=False)

    sim_path = _write_simulated_excel(current, used_tickers, output_dir)

    logging.info("Vytvořeno %s řádků pro predikci.", len(feature_rows))
    logging.info("CSV s atributy: %s", features_path)
    logging.info("Filtrovaný Excel: %s", sim_path)

    return features_path, sim_path


def main():
    parser = argparse.ArgumentParser(description="Vytvoř CSV pro predikci z posledního snapshotu + look-back.")
    parser.add_argument("data_dir", nargs="?", default=Path("."), type=Path, help="Složka se snapshoty (default .)")
    parser.add_argument("--output-dir", default=Path("processed"), type=Path, help="Kam uložit výstupy (default processed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        generate_prediction_inputs(args.data_dir, args.output_dir)
    except Exception as exc:  # pragma: no cover - CLI surface
        logging.error("Selhalo: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
