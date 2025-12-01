"""Create a Kelly-inspired portfolio allocation from model predictions.

This utility combines three data sources:
1. **Predictions**: a CSV/Excel file that contains per-ticker forecasts and
   probabilities for a specific time horizon.
2. **Metadata**: an optional file that enriches tickers with human-friendly
   information (e.g. company name, sector).
3. **Weekly price snapshots**: files with historical prices in column 80 (or 82),
   one file per week. Correlations are computed directly from these prices using
   the same horizon as the predictions.

The script filters predictions for a selected horizon, estimates expected returns
using the provided probabilities and an assumed downside scenario, computes a
correlation matrix from historical prices, and applies a Kelly allocation that
accounts for the covariance of returns. The resulting portfolio weights are
saved alongside the input predictions and (if supplied) metadata.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _infer_date_from_name(path: Path) -> Optional[pd.Timestamp]:
    match = re.search(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", path.stem)
    if not match:
        return None
    try:
        return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))
    except ValueError:
        return None


def load_predictions(
    predictions_path: Path,
    ticker_col: str,
    horizon_col: Optional[str],
    return_col: str,
    prob_col: str,
    horizon_filter: Optional[str],
    min_probability: float,
    top_n: Optional[int],
) -> pd.DataFrame:
    df = _read_table(predictions_path)
    required_cols = {ticker_col, return_col, prob_col}
    missing = required_cols - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Predikční soubor postrádá povinné sloupce: {missing_cols}")

    if horizon_col and horizon_col in df.columns and horizon_filter:
        df = df[df[horizon_col] == horizon_filter]
    elif horizon_filter:
        raise ValueError(
            "Nebyl zadán sloupec s horizontem, ale je požadován filtr na horizont."
        )

    df = df.rename(columns={
        ticker_col: "ticker",
        return_col: "predicted_return",
        prob_col: "probability",
    })

    df["probability"] = df["probability"].astype(float)
    df = df[df["probability"] >= min_probability]

    if top_n:
        df = df.nlargest(top_n, "predicted_return")

    return df.reset_index(drop=True)


def attach_metadata(predictions: pd.DataFrame, metadata_path: Path, ticker_col: str) -> pd.DataFrame:
    metadata = _read_table(metadata_path)
    if ticker_col not in metadata.columns:
        raise ValueError(f"V metadata souboru chybí sloupec '{ticker_col}'.")

    return predictions.merge(metadata, how="left", left_on="ticker", right_on=ticker_col)


def _select_price_series(df: pd.DataFrame, ticker_col: str, price_col_index: int, price_col_name: Optional[str]) -> pd.DataFrame:
    if price_col_name:
        if price_col_name not in df.columns:
            raise ValueError(f"Sloupec s cenou '{price_col_name}' nebyl ve vstupu nalezen.")
        price_series = df[price_col_name]
    else:
        if price_col_index >= df.shape[1]:
            raise ValueError(
                f"Požadovaný cenový sloupec s indexem {price_col_index} přesahuje šířku dataframů ({df.shape[1]} sloupců)."
            )
        price_series = df.iloc[:, price_col_index]

    if ticker_col not in df.columns:
        raise ValueError(f"V cenovém souboru chybí sloupec '{ticker_col}'.")

    return pd.DataFrame({"ticker": df[ticker_col], "price": price_series})


def load_price_history(
    price_pattern: str,
    ticker_col: str,
    price_col_index: int,
    price_col_name: Optional[str],
) -> pd.DataFrame:
    paths = sorted(Path().glob(price_pattern))
    if not paths:
        raise FileNotFoundError(f"Nenalezeny žádné cenové soubory pro pattern '{price_pattern}'.")

    snapshots: list[pd.DataFrame] = []
    for idx, path in enumerate(paths):
        df = _read_table(path)
        price_df = _select_price_series(df, ticker_col, price_col_index, price_col_name)
        date = _infer_date_from_name(path) or pd.Timestamp(idx)
        price_df["date"] = date
        snapshots.append(price_df)

    prices = pd.concat(snapshots, ignore_index=True)
    pivot = prices.pivot_table(index="date", columns="ticker", values="price")
    return pivot.sort_index()


def compute_horizon_returns(price_table: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    if horizon_steps <= 0:
        raise ValueError("Horizon steps must be positive.")
    return price_table.pct_change(periods=horizon_steps).dropna(how="all")


def compute_expected_returns(predictions: pd.DataFrame, assumed_downside: float) -> pd.Series:
    upside = predictions["predicted_return"].astype(float)
    probability = predictions["probability"].astype(float)
    expected = probability * upside + (1 - probability) * assumed_downside
    return expected


def kelly_allocation(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    fraction: float,
    allow_shorts: bool,
    max_weight: float,
) -> pd.Series:
    tickers = expected_returns.index
    covariance = covariance.reindex(index=tickers, columns=tickers).fillna(0.0)

    # Regularize covariance to avoid singular matrices.
    diagonal = np.diag(covariance.to_numpy())
    if np.any(diagonal == 0):
        noise = np.eye(len(tickers)) * 1e-6
        covariance = covariance + pd.DataFrame(noise, index=tickers, columns=tickers)

    cov_inv = np.linalg.pinv(covariance.to_numpy())
    raw_weights = cov_inv @ expected_returns.to_numpy()

    if not allow_shorts:
        raw_weights = np.clip(raw_weights, 0.0, None)

    scaled = raw_weights * fraction
    if scaled.sum() > 0:
        scaled = scaled / scaled.sum()

    weights = pd.Series(scaled, index=tickers)
    if max_weight is not None:
        weights = weights.clip(upper=max_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()

    return weights


def build_portfolio(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = load_predictions(
        predictions_path=Path(args.predictions),
        ticker_col=args.ticker_column,
        horizon_col=args.horizon_column,
        return_col=args.return_column,
        prob_col=args.probability_column,
        horizon_filter=args.horizon,
        min_probability=args.min_probability,
        top_n=args.top_n,
    )

    if args.metadata:
        predictions = attach_metadata(predictions, Path(args.metadata), args.ticker_column)

    price_history = load_price_history(
        price_pattern=args.price_pattern,
        ticker_col=args.ticker_column,
        price_col_index=args.price_column_index,
        price_col_name=args.price_column_name,
    )

    horizon_returns = compute_horizon_returns(price_history, args.horizon_steps)

    tickers = predictions["ticker"].unique()
    returns_subset = horizon_returns[tickers].dropna(axis=1, how="all")

    missing_tickers = set(tickers) - set(returns_subset.columns)
    if missing_tickers:
        raise ValueError(
            "V cenové historii chybí tickery: " + ", ".join(sorted(missing_tickers))
        )

    covariance = returns_subset.cov()
    expected = compute_expected_returns(predictions, args.assumed_downside)
    expected.index = predictions["ticker"]

    weights = kelly_allocation(
        expected_returns=expected,
        covariance=covariance,
        fraction=args.kelly_fraction,
        allow_shorts=args.allow_shorts,
        max_weight=args.max_weight,
    )

    output = predictions.copy()
    output["expected_return"] = expected.values
    output["kelly_weight"] = output["ticker"].map(weights).fillna(0.0)
    output = output.sort_values("kelly_weight", ascending=False)

    correlation = returns_subset.corr()
    return output, correlation


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vytvořte portfolio pomocí Kellyho kritéria z predikcí.")
    parser.add_argument("--predictions", required=True, help="CSV/XLSX s predikcemi a pravděpodobnostmi.")
    parser.add_argument("--metadata", help="CSV/XLSX s metadaty akcií (sloupec Ticker).")
    parser.add_argument("--price-pattern", default="data/weekly_prices/*.xlsx", help="Glob pattern pro cenové snapshoty.")
    parser.add_argument(
        "--price-column-index",
        type=int,
        default=79,
        help="Index cenového sloupce (0-based). 79 odpovídá 80. sloupci v Excelu.",
    )
    parser.add_argument(
        "--price-column-name",
        help="Volitelný název cenového sloupce; pokud je zadán, index se ignoruje.",
    )
    parser.add_argument("--ticker-column", default="Ticker", help="Název sloupce s tickerem ve všech souborech.")
    parser.add_argument("--horizon-column", help="Sloupec s horizontem v predikčním souboru.")
    parser.add_argument("--horizon", help="Hodnota horizontu, kterou chceme použít (např. '4w').")
    parser.add_argument("--return-column", default="prediction", help="Sloupec s predikovaným výnosem.")
    parser.add_argument("--probability-column", default="probability", help="Sloupec s pravděpodobností úspěchu.")
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.5,
        help="Filtr na minimální pravděpodobnost, která vstoupí do portfolia.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Volitelný limit počtu nejlepších predikcí (podle predikovaného výnosu).",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=4,
        help="Počet kroků pro výpočet výnosů (např. 4 týdny odpovídají horizontu predikce).",
    )
    parser.add_argument(
        "--assumed-downside",
        type=float,
        default=-0.02,
        help="Odhadovaný pokles v případě neúspěchu predikce (v decimální podobě, např. -0.02 = -2 %).",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=1.0,
        help="Frakce Kellyho váhy (např. 0.5 pro poloviční Kelly).",
    )
    parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Povolit záporné váhy (short pozice). Výchozí je long-only.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.25,
        help="Maximální váha jedné pozice po normalizaci (None = bez limitu).",
    )
    parser.add_argument("--output", default="portfolio_allocation.csv", help="Výstupní CSV s váhami portfolia.")
    parser.add_argument(
        "--correlation-output",
        default="portfolio_correlations.csv",
        help="Výstupní CSV s korelační maticí pro daný horizont.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    portfolio, correlation = build_portfolio(args)

    portfolio.to_csv(args.output, index=False)
    correlation.to_csv(args.correlation_output)

    print(f"Uloženo portfolio do: {args.output}")
    print(f"Uložena korelační matice do: {args.correlation_output}")


if __name__ == "__main__":
    main()
