import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import yfinance as yf


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


def _sheet_exists_case_insensitive(excel_path: Path, requested_name: str) -> bool:
    xls = pd.ExcelFile(excel_path)
    names = xls.sheet_names
    return requested_name in names or requested_name.lower() in {n.lower() for n in names}


def _looks_like_portfolio_layout(excel_path: Path, sheet_name: str) -> bool:
    """
    Lightweight validation that the selected sheet likely follows expected layout:
    at least one non-empty ticker in column B and one numeric weight in column C.
    """
    resolved = resolve_sheet_name_case_insensitive(excel_path, sheet_name)
    df = pd.read_excel(excel_path, sheet_name=resolved, header=None, nrows=10)
    if df.shape[1] < 3:
        return False

    tickers = df.iloc[:10, 1].dropna().astype(str).str.strip()
    weights = pd.to_numeric(df.iloc[:10, 2], errors="coerce").dropna()
    return (tickers != "").any() and len(weights) > 0


def resolve_latest_input_file(
    explicit_path: Optional[Path] = None,
    preferred_sheet: str = "Sheet1",
) -> Path:
    """
    Resolve input workbook.
    - If explicit path is provided, use it.
    - Otherwise, find the newest file by date in filename DD.MM.YYYY among *.xlsx in CWD.
    """
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Soubor neexistuje: {explicit_path}")
        return explicit_path

    candidates = list(Path(".").glob("*.xlsx"))
    valid_candidates: List[Tuple[pd.Timestamp, Path]] = []
    for path in candidates:
        try:
            file_date = parse_reference_date(path)  # keep only files from expected weekly series
            if _sheet_exists_case_insensitive(path, preferred_sheet) and _looks_like_portfolio_layout(
                path, preferred_sheet
            ):
                valid_candidates.append((file_date, path))
        except Exception:
            # Ignore files from other workflows or with incompatible format
            continue

    if not valid_candidates:
        raise FileNotFoundError(
            "Nebyl nalezen žádný validní .xlsx soubor se Sheet1 (nebo sheet1) a očekávaným rozložením portfolia. "
            "Zadejte soubor ručně přes --input."
        )

    valid_candidates.sort(key=lambda x: x[0])
    return valid_candidates[-1][1]


def resolve_sheet_name_case_insensitive(excel_path: Path, requested_name: str) -> str:
    """Resolve sheet name in case-insensitive mode."""
    xls = pd.ExcelFile(excel_path)
    names = xls.sheet_names
    if requested_name in names:
        return requested_name

    lowered_map = {name.lower(): name for name in names}
    matched = lowered_map.get(requested_name.lower())
    if matched:
        return matched

    raise ValueError(
        f"Worksheet '{requested_name}' nebyl nalezen. Dostupné listy: {', '.join(names)}"
    )


def load_portfolios_from_sheet(
    excel_path: Path, sheet_name: str, rows: int = 10, col_step: int = 6
) -> List[PortfolioDefinition]:
    resolved_sheet = resolve_sheet_name_case_insensitive(excel_path, sheet_name)
    df = pd.read_excel(excel_path, sheet_name=resolved_sheet, header=None)

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
        # Typical for newest portfolio when less than one trading week elapsed.
        return pd.Series([1.0], index=[portfolio.start_date], name=portfolio.name)

    weekly = p.resample("W-FRI").last().dropna(how="any")
    if weekly.empty:
        return pd.Series([1.0], index=[portfolio.start_date], name=portfolio.name)

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
    return matrix.mean(axis=1, skipna=True).rename("Celkové portfolio (průměr)")


def aggregate_aligned_portfolios(curves: Dict[str, pd.Series]) -> pd.Series:
    aligned = []
    for curve in curves.values():
        weeks = np.arange(len(curve))
        aligned.append(pd.Series(curve.values, index=weeks))
    aligned_df = pd.concat(aligned, axis=1)
    return aligned_df.mean(axis=1).rename("Virtuální portfolio (průměr)")


def fit_exponential_growth(curves: Dict[str, pd.Series]) -> Tuple[pd.Series, float]:
    """
    Fit y = a * exp(b * t) across all aligned points from all portfolios.
    Returns fitted curve on integer weeks and annualized growth rate in decimals.
    """
    points = []
    max_week = 0

    for curve in curves.values():
        t = np.arange(len(curve))
        y = curve.values
        valid = np.isfinite(y) & (y > 0)
        t = t[valid]
        y = y[valid]
        if len(y) > 0:
            points.append(pd.DataFrame({"t": t, "y": y}))
            max_week = max(max_week, int(t.max()))

    if not points:
        raise ValueError("Nelze spočítat exponenciální fit: chybí validní body > 0.")

    all_points = pd.concat(points, ignore_index=True)
    # log(y) = log(a) + b*t
    b, log_a = np.polyfit(all_points["t"].values, np.log(all_points["y"].values), 1)
    a = float(np.exp(log_a))

    fit_x = np.linspace(0, max_week, max(200, max_week * 10 + 1))
    fit_y = a * np.exp(b * fit_x)
    annual_growth = float(np.exp(b * 52) - 1.0)

    return pd.Series(fit_y, index=fit_x, name="Exponenciální fit"), annual_growth


def plot_results(
    curves: Dict[str, pd.Series],
    combined_live: pd.Series,
    sp500_curve: pd.Series,
    combined_aligned: pd.Series,
    exp_fit: pd.Series,
    annual_growth_rate: float,
    output_dir: Path,
) -> plt.Figure:
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
    ax2.plot(
        exp_fit.index,
        exp_fit.values,
        color="tab:red",
        linestyle=":",
        linewidth=2.5,
        label=f"Exponenciální fit (CAGR ~ {annual_growth_rate * 100:.2f} % p.a.)",
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
    return fig


def run_analysis(
    input_file: Path,
    sheet_name: str,
    output_dir: Path,
    benchmark: str,
) -> Tuple[plt.Figure, float]:
    portfolios = load_portfolios_from_sheet(input_file, sheet_name)
    all_tickers = sorted(set(sum([p.tickers for p in portfolios], [])))

    start = min(p.start_date for p in portfolios)
    end = pd.Timestamp.today().normalize()

    prices = download_prices(all_tickers, start, end)
    benchmark_prices = download_prices([benchmark], start, end)

    curves: Dict[str, pd.Series] = {}
    for p in portfolios:
        curves[p.name] = compute_portfolio_curve(prices, p)

    combined_live = aggregate_live_portfolios(curves)
    combined_aligned = aggregate_aligned_portfolios(curves)
    exp_fit, annual_growth_rate = fit_exponential_growth(curves)

    sp = benchmark_prices.loc[benchmark_prices.index >= start]
    sp_weekly = sp.resample("W-FRI").last().dropna()
    sp500_curve = (sp_weekly / sp_weekly.iloc[0]).iloc[:, 0].rename("S&P 500")

    fig = plot_results(
        curves,
        combined_live,
        sp500_curve,
        combined_aligned,
        exp_fit,
        annual_growth_rate,
        output_dir,
    )
    return fig, annual_growth_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vyhodnocení týdenních portfolií a porovnání se S&P500"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cesta k .xlsx souboru s portfolii",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Sheet1",
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
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Spustí skript pouze v CLI režimu (bez GUI okna).",
    )

    args = parser.parse_args()

    if args.no_gui:
        input_file = resolve_latest_input_file(args.input, preferred_sheet=args.sheet)
        print(f"Použitý vstupní soubor: {input_file}")
        _, annual_growth_rate = run_analysis(input_file, args.sheet, args.output_dir, args.benchmark)
        print(f"Odhadnutá roční rychlost růstu z exponenciálního fitu: {annual_growth_rate * 100:.2f} % p.a.")
        return

    class PortfolioGuiApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Portfolio Strategy Visualizer")
            self.geometry("1300x900")

            self.input_var = tk.StringVar(value=str(resolve_latest_input_file(args.input, args.sheet)))
            self.sheet_var = tk.StringVar(value=args.sheet)
            self.benchmark_var = tk.StringVar(value=args.benchmark)
            self.output_var = tk.StringVar(value=str(args.output_dir))
            self.status_var = tk.StringVar(value="Připraveno.")

            self._build_ui()
            self.canvas = None
            self.toolbar = None

        def _build_ui(self) -> None:
            top = ttk.Frame(self, padding=10)
            top.pack(fill=tk.X)

            ttk.Label(top, text="Soubor:").grid(row=0, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky="ew", padx=6)
            ttk.Button(top, text="Vybrat...", command=self._pick_file).grid(row=0, column=2, padx=4)

            ttk.Label(top, text="List:").grid(row=1, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.sheet_var, width=20).grid(row=1, column=1, sticky="w", padx=6)
            ttk.Label(top, text="Benchmark:").grid(row=1, column=1, sticky="e")
            ttk.Entry(top, textvariable=self.benchmark_var, width=12).grid(row=1, column=2, sticky="w")

            ttk.Label(top, text="Output složka:").grid(row=2, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.output_var, width=70).grid(row=2, column=1, sticky="ew", padx=6)
            ttk.Button(top, text="Spustit analýzu", command=self._run).grid(row=2, column=2, padx=4)

            top.columnconfigure(1, weight=1)

            ttk.Label(self, textvariable=self.status_var, padding=(10, 0)).pack(anchor="w")
            self.plot_frame = ttk.Frame(self, padding=10)
            self.plot_frame.pack(fill=tk.BOTH, expand=True)

        def _pick_file(self) -> None:
            file_path = filedialog.askopenfilename(
                title="Vyber Excel soubor",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
            )
            if file_path:
                self.input_var.set(file_path)

        def _clear_plot(self) -> None:
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
            if self.toolbar is not None:
                self.toolbar.destroy()
                self.toolbar = None

        def _run(self) -> None:
            try:
                self.status_var.set("Počítám...")
                self.update_idletasks()
                input_file = Path(self.input_var.get())
                output_dir = Path(self.output_var.get())
                fig, annual_growth = run_analysis(
                    input_file=input_file,
                    sheet_name=self.sheet_var.get(),
                    output_dir=output_dir,
                    benchmark=self.benchmark_var.get(),
                )
                self._clear_plot()
                self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
                self.toolbar.update()
                self.status_var.set(
                    f"Hotovo. CAGR z exponenciálního fitu: {annual_growth * 100:.2f} % p.a. | Výstup: {output_dir}"
                )
            except Exception as exc:
                messagebox.showerror("Chyba", str(exc))
                self.status_var.set("Chyba při výpočtu.")

    app = PortfolioGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
