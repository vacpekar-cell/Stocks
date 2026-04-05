import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Chybí balíček yfinance. Nainstalujte ho: pip install yfinance"
    ) from exc


@dataclass
class PortfolioDefinition:
    name: str
    start_date: pd.Timestamp
    tickers: List[str]
    weights: np.ndarray


def parse_reference_date(path: Path) -> pd.Timestamp:
    """Extract date from filename like '03.04.2025 108nodes new meta.xlsx'."""
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", path.name)
    if not match:
        raise ValueError(
            "Ve jméně souboru nebylo nalezeno datum ve formátu DD.MM.YYYY."
        )
    return pd.Timestamp(datetime.strptime(match.group(1), "%d.%m.%Y").date())


def load_portfolios_from_sheet(
    excel_path: Path, sheet_name: str, rows: int = 10, col_step: int = 6
) -> List[PortfolioDefinition]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    start_col = 1  # B
    portfolios_raw: List[Dict] = []
    block_idx = 0

    while start_col + 1 < df.shape[1]:
        ticker_col = df.iloc[:rows, start_col]
        weight_col = df.iloc[:rows, start_col + 1]

        tickers = [str(t).strip().upper() for t in ticker_col if pd.notna(t) and str(t).strip()]
        raw_weights = [float(w) for w in weight_col[: len(tickers)] if pd.notna(w)]

        if tickers and raw_weights and len(tickers) == len(raw_weights):
            weights = np.array(raw_weights, dtype=float) / 100.0
            if weights.sum() <= 0:
                raise ValueError(f"Portfolio v bloku {block_idx + 1} má nulovou váhu.")
            weights = weights / weights.sum()
            portfolios_raw.append(
                {
                    "block_idx": block_idx,
                    "tickers": tickers,
                    "weights": weights,
                }
            )

        start_col += col_step
        block_idx += 1

    if not portfolios_raw:
        raise ValueError(
            "V listu nebyla nalezena žádná platná portfolia (sloupce B/C, H/I, N/O, ...)."
        )

    reference_date = parse_reference_date(excel_path)
    newest_idx = portfolios_raw[-1]["block_idx"]

    portfolios: List[PortfolioDefinition] = []
    for p in portfolios_raw:
        weeks_back = newest_idx - p["block_idx"]
        start_date = reference_date - timedelta(weeks=weeks_back)
        portfolios.append(
            PortfolioDefinition(
                name=f"Portfolio {p['block_idx'] + 1}",
                start_date=pd.Timestamp(start_date),
                tickers=p["tickers"],
                weights=p["weights"],
            )
        )

    return portfolios


def download_prices(tickers: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=(start_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("Nepodařilo se stáhnout žádná tržní data.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            closes = data["Close"]
        else:
            closes = data.xs("Close", axis=1, level=0)
    else:
        closes = data.to_frame(name=tickers[0]) if len(tickers) == 1 else data

    closes = closes.sort_index().ffill().dropna(how="all")
    if closes.empty:
        raise ValueError("Data obsahují pouze prázdné hodnoty po čištění.")

    return closes


def compute_portfolio_curve(prices: pd.DataFrame, portfolio: PortfolioDefinition) -> pd.Series:
    p = prices.loc[prices.index >= portfolio.start_date, portfolio.tickers].copy()
    p = p.dropna(how="all").ffill().dropna(how="any")
    if p.empty:
        raise ValueError(f"Chybí data pro {portfolio.name} od {portfolio.start_date.date()}.")

    weekly = p.resample("W-FRI").last().dropna(how="any")
    if weekly.empty:
        raise ValueError(f"Po převodu na týdenní data nejsou data pro {portfolio.name}.")

    base = weekly.iloc[0]
    rel = weekly.divide(base)
    curve = rel.mul(portfolio.weights, axis=1).sum(axis=1)
    curve.name = portfolio.name
    return curve


def aggregate_live_portfolios(curves: Dict[str, pd.Series]) -> pd.Series:
    union_index = sorted(set().union(*[c.index for c in curves.values()]))
    matrix = pd.DataFrame(index=union_index)
    for name, curve in curves.items():
        matrix[name] = curve.reindex(union_index)
    return matrix.sum(axis=1, skipna=True).rename("Celkové portfolio (součet)")


def aggregate_aligned_portfolios(curves: Dict[str, pd.Series]) -> pd.Series:
    aligned = []
    for curve in curves.values():
        weeks = np.arange(len(curve))
        aligned.append(pd.Series(curve.values, index=weeks))
    aligned_df = pd.concat(aligned, axis=1)
    return aligned_df.mean(axis=1).rename("Virtuální portfolio (průměr)")


def plot_results(
    curves: Dict[str, pd.Series],
    combined_live: pd.Series,
    sp500_curve: pd.Series,
    combined_aligned: pd.Series,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Graph 1: absolute timeline
    ax = axes[0]
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, alpha=0.6, linewidth=1.8, label=name)
    ax.plot(
        combined_live.index,
        combined_live.values,
        color="black",
        linewidth=2.8,
        label=combined_live.name,
    )
    ax.plot(
        sp500_curve.index,
        sp500_curve.values,
        color="tab:orange",
        linestyle="--",
        linewidth=2.2,
        label="S&P 500",
    )
    ax.set_title("Vývoj jednotlivých portfolií v čase (týdenní body)")
    ax.set_ylabel("Hodnota (start = 1)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)

    # Graph 2: aligned timeline
    ax2 = axes[1]
    for name, curve in curves.items():
        weeks = np.arange(len(curve))
        ax2.plot(weeks, curve.values, alpha=0.6, linewidth=1.8, label=name)
    ax2.plot(
        combined_aligned.index,
        combined_aligned.values,
        color="black",
        linewidth=2.8,
        label=combined_aligned.name,
    )
    ax2.set_title("Portfolia přenesená do společného začátku")
    ax2.set_xlabel("Počet týdnů od startu")
    ax2.set_ylabel("Hodnota (start = 1)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", ncol=2)

    fig.tight_layout()

    out_png = output_dir / "portfolio_strategy_comparison.png"
    fig.savefig(out_png, dpi=160)

    # Save data for further analysis
    out_abs = output_dir / "portfolio_curves_absolute.csv"
    out_rel = output_dir / "portfolio_curves_aligned.csv"

    abs_df = pd.concat([*curves.values(), combined_live, sp500_curve], axis=1)
    abs_df.to_csv(out_abs, index_label="date")

    aligned_df = pd.concat(
        [
            *[
                pd.Series(curve.values, index=np.arange(len(curve)), name=name)
                for name, curve in curves.items()
            ],
            combined_aligned,
        ],
        axis=1,
    )
    aligned_df.to_csv(out_rel, index_label="week_since_start")

    print(f"Graf uložen do: {out_png}")
    print(f"Absolutní křivky: {out_abs}")
    print(f"Zarovnané křivky: {out_rel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vyhodnocení týdenních portfolií a porovnání se S&P500"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("03.04.2025 108nodes new meta.xlsx"),
        help="Cesta k .xlsx souboru s portfolii",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="sheet1",
        help="Název listu s portfolii",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Složka pro grafy a exporty",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="^GSPC",
        help="Ticker benchmarku (výchozí S&P500: ^GSPC)",
    )

    args = parser.parse_args()

    portfolios = load_portfolios_from_sheet(args.input, args.sheet)
    all_tickers = sorted(set(sum([p.tickers for p in portfolios], [])))

    start = min(p.start_date for p in portfolios)
    end = pd.Timestamp.today().normalize()

    prices = download_prices(all_tickers, start, end)
    benchmark_prices = download_prices([args.benchmark], start, end)

    curves: Dict[str, pd.Series] = {}
    for p in portfolios:
        curves[p.name] = compute_portfolio_curve(prices, p)

    combined_live = aggregate_live_portfolios(curves)
    combined_aligned = aggregate_aligned_portfolios(curves)

    sp = benchmark_prices.loc[benchmark_prices.index >= start]
    sp_weekly = sp.resample("W-FRI").last().dropna()
    sp500_curve = (sp_weekly / sp_weekly.iloc[0]).iloc[:, 0].rename("S&P 500")

    plot_results(curves, combined_live, sp500_curve, combined_aligned, args.output_dir)


if __name__ == "__main__":
    main()
