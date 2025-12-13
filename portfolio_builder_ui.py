import datetime as dt
import os
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import importlib.util

import numpy as np
import pandas as pd


_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec is not None and _yf_spec.loader is not None:
    yf = importlib.util.module_from_spec(_yf_spec)
    _yf_spec.loader.exec_module(yf)
else:  # pragma: no cover - handled at runtime
    yf = None


# ---- Data structures -----------------------------------------------------


@dataclass
class StockRecord:
    ticker: str
    sharpe: float
    forecast_pct: float
    std_pct: float


# ---- Portfolio math helpers ---------------------------------------------


def _safe_covariance_matrix(stds: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Build a covariance matrix from std deviations and correlations."""

    # Ensure diagonals are 1 for correlation and guard against NaNs
    np.fill_diagonal(corr, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    stds = np.asarray(stds).reshape(-1, 1)
    cov = corr * (stds @ stds.T)

    # Add a small ridge for numerical stability
    cov += np.eye(len(stds)) * 1e-8
    return cov


def _rebalance_with_caps(raw_weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Normalize weights, apply caps, and redistribute remainders."""

    weights = np.clip(raw_weights, 0.0, None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # Iteratively cap overweight positions and redistribute to others
    iteration_guard = 0
    while True:
        iteration_guard += 1
        if iteration_guard > 25:
            break

        overweight = weights > max_weight
        if not overweight.any():
            break

        excess = weights[overweight] - max_weight
        weights[overweight] = max_weight

        remaining = ~overweight
        remaining_sum = weights[remaining].sum()
        # Distribute excess proportionally to remaining weights; if none, spread evenly
        if remaining.any():
            if remaining_sum == 0:
                weights[remaining] = excess.sum() / remaining.sum()
            else:
                weights[remaining] += weights[remaining] / remaining_sum * excess.sum()
        else:
            # All are capped; break to avoid division by zero
            break

        # Re-normalize to 1 after redistribution
        weights = weights / weights.sum()

    return weights


def optimize_weights(expected_returns: np.ndarray, corr_matrix: np.ndarray, stds: np.ndarray, max_weight: float) -> np.ndarray:
    """
    Compute a long-only, capped-weight portfolio that maximizes the Sharpe ratio.

    This uses the closed-form tangency portfolio as a starting point and then applies
    non-negativity and max-weight caps with iterative redistribution.
    """

    cov = _safe_covariance_matrix(stds, corr_matrix.copy())
    try:
        inv_cov = np.linalg.pinv(cov)
        raw = inv_cov @ expected_returns
    except np.linalg.LinAlgError:
        raw = expected_returns.copy()

    weights = _rebalance_with_caps(raw, max_weight=max_weight)
    return weights


def portfolio_sharpe(weights: np.ndarray, expected_returns: np.ndarray, corr_matrix: np.ndarray, stds: np.ndarray) -> float:
    cov = _safe_covariance_matrix(stds, corr_matrix.copy())
    port_ret = float(weights @ expected_returns)
    volatility = float(np.sqrt(weights @ cov @ weights.T))
    if volatility <= 0:
        return 0.0
    return port_ret / volatility


# ---- Data fetching -------------------------------------------------------


class ReturnCache:
    def __init__(self):
        self.cache: dict[str, pd.Series] = {}

    def get_returns(self, ticker: str, log_fn) -> pd.Series:
        if ticker in self.cache:
            return self.cache[ticker]

        if yf is None:
            log_fn(f"yfinance není dostupné, korelace pro {ticker} nastavena na 0.")
            self.cache[ticker] = pd.Series(dtype=float)
            return self.cache[ticker]

        try:
            data = yf.download(ticker, period="3mo", interval="1wk", progress=False, auto_adjust=True)
            returns = data["Close"].pct_change().dropna()
        except Exception as exc:  # pragma: no cover - relies on network
            log_fn(f"Nepodařilo se stáhnout {ticker}: {exc}. Korelace nastavena na 0.")
            returns = pd.Series(dtype=float)

        self.cache[ticker] = returns
        return returns


def build_correlation_matrix(tickers: list[str], cache: ReturnCache, log_fn) -> np.ndarray:
    if not tickers:
        return np.zeros((0, 0))

    returns = {}
    for ticker in tickers:
        returns[ticker] = cache.get_returns(ticker, log_fn)

    aligned = pd.DataFrame(returns).dropna(how="all")
    if aligned.empty:
        log_fn("Není dostatek dat pro korelace, použiji nulovou korelaci.")
        return np.zeros((len(tickers), len(tickers)))

    corr = aligned.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    return corr.values


# ---- Portfolio construction ---------------------------------------------


class PortfolioBuilder:
    def __init__(self, records: list[StockRecord], max_positions: int = 10, max_weight: float = 0.25, log_fn=print):
        self.records = sorted(records, key=lambda r: r.sharpe, reverse=True)
        self.max_positions = max_positions
        self.max_weight = max_weight
        self.log = log_fn
        self.cache = ReturnCache()

    def build(self):
        if not self.records:
            self.log("Seznam akcií je prázdný.")
            return []

        portfolio: list[StockRecord] = [self.records[0]]
        self.log(f"1) Přidávám {portfolio[0].ticker} s nejvyšším Sharpe {portfolio[0].sharpe:.3f}.")

        while len(portfolio) < min(self.max_positions, len(self.records)):
            remaining = [r for r in self.records if r.ticker not in {p.ticker for p in portfolio}]
            candidates = remaining[:10]
            if not candidates:
                break

            best_choice = None
            best_sharpe = -np.inf

            # Build correlation matrix for current portfolio plus candidate to evaluate
            for candidate in candidates:
                trial_list = portfolio + [candidate]
                tickers = [r.ticker for r in trial_list]
                corr_matrix = build_correlation_matrix(tickers, self.cache, self.log)
                expected = np.array([r.forecast_pct for r in trial_list])
                stds = np.array([r.std_pct for r in trial_list])

                weights = optimize_weights(expected, corr_matrix, stds, self.max_weight)
                sharpe = portfolio_sharpe(weights, expected, corr_matrix, stds)

                self.log(
                    f"   Testuji {candidate.ticker}: Sharpe portfolia {sharpe:.3f} při vahách {[f'{w:.2%}' for w in weights]}"
                )

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_choice = (candidate, weights, corr_matrix, expected, stds)

            if best_choice is None:
                self.log("Žádný vhodný kandidát, končím.")
                break

            chosen, weights, corr_matrix, expected, stds = best_choice
            portfolio.append(chosen)
            self.log(
                f"{len(portfolio)}) Přidávám {chosen.ticker}, Sharpe portfolia nyní {best_sharpe:.3f}."
            )

            # Recompute weights for the current portfolio to report them
            weights = optimize_weights(expected, corr_matrix, stds, self.max_weight)
            for t, w in zip([r.ticker for r in portfolio], weights):
                self.log(f"   Váha {t}: {w:.2%}")

        return portfolio


# ---- UI ------------------------------------------------------------------


class PortfolioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Portfolio Builder")

        self.records: list[StockRecord] = []

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="Načíst CSV", command=self.load_csv).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Počet pozic:").pack(side=tk.LEFT, padx=5)
        self.count_var = tk.IntVar(value=10)
        ttk.Entry(control_frame, textvariable=self.count_var, width=6).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Max váha jedné pozice (např. 0.25):").pack(side=tk.LEFT, padx=5)
        self.cap_var = tk.DoubleVar(value=0.25)
        ttk.Entry(control_frame, textvariable=self.cap_var, width=8).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="Postavit portfolio", command=self.start_build).pack(side=tk.RIGHT)

        self.tree = ttk.Treeview(main, columns=("ticker", "sharpe", "forecast", "std"), show="headings", height=8)
        for col, text in zip(self.tree["columns"], ["Ticker", "Sharpe", "Forecast %", "Std %"]):
            self.tree.heading(col, text=text)
        self.tree.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        self.log_box = tk.Text(main, height=16, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def log(self, message: str):
        self.log_box.configure(state=tk.NORMAL)
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _read_stock_table(self, path: str) -> pd.DataFrame:
        """Robustně načte tabulku bez ohledu na hlavičky či oddělovač."""

        # Nejprve se pokusíme autodetekovat oddělovač a nečekat hlavičku
        try:
            df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
        except Exception as exc:
            raise RuntimeError(f"Nelze načíst soubor: {exc}")

        # Pokud je podezření na hlavičku a zároveň málo sloupců, zkusíme ještě variantu s hlavičkou
        if df.shape[1] < 4:
            try:
                df_with_header = pd.read_csv(path, sep=None, engine="python", comment="#")
            except Exception:
                df_with_header = pd.DataFrame()

            if df_with_header.shape[1] >= 4:
                df = df_with_header

        if df.shape[1] < 4:
            raise ValueError("Soubor musí mít alespoň 4 sloupce: ticker, sharpe, forecast %, std %.")

        return df

    @staticmethod
    def _to_float(value) -> float:
        # Podpora čárky jako desetinné tečky i běžného zápisu
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
        return float(value)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV / CLS", "*.csv *.cls"), ("All files", "*.*")])
        if not path:
            return

        try:
            df = self._read_stock_table(path)
        except Exception as exc:
            messagebox.showerror("Chyba", str(exc))
            return

        self.records = []
        for _, row in df.iterrows():
            try:
                rec = StockRecord(
                    ticker=str(row.iloc[0]).strip(),
                    sharpe=self._to_float(row.iloc[1]),
                    forecast_pct=self._to_float(row.iloc[2]),
                    std_pct=self._to_float(row.iloc[3]),
                )
                self.records.append(rec)
            except Exception:
                self.log(f"Přeskakuji řádek: {row}")

        self.tree.delete(*self.tree.get_children())
        for rec in self.records:
            self.tree.insert("", tk.END, values=(rec.ticker, f"{rec.sharpe:.3f}", f"{rec.forecast_pct:.3f}", f"{rec.std_pct:.3f}"))

        self.log(
            f"Načteno {len(self.records)} záznamů z {os.path.basename(path)}. Hlavička není nutná; očekávám pořadí: ticker, sharpe, forecast %, std %."
        )

    def start_build(self):
        if not self.records:
            messagebox.showwarning("Upozornění", "Nejdříve načtěte CSV se seznamem akcií.")
            return

        try:
            count = int(self.count_var.get())
            cap = float(self.cap_var.get())
        except Exception:
            messagebox.showerror("Chyba", "Neplatné číslo pro počet pozic nebo max váhu.")
            return

        if count <= 0:
            messagebox.showerror("Chyba", "Počet pozic musí být kladný.")
            return

        def worker():
            builder = PortfolioBuilder(self.records, max_positions=count, max_weight=cap, log_fn=self.log)
            portfolio = builder.build()
            tickers = ", ".join(r.ticker for r in portfolio)
            self.log(f"Hotovo. Portfolio: {tickers}")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
