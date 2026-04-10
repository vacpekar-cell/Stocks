import datetime as dt
import os
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
import time

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

    # Simple shrinkage towards identity to stabilize covariance estimates
    shrink = 0.1
    corr = (1 - shrink) * corr + shrink * np.eye(len(corr))

    stds = np.asarray(stds).reshape(-1, 1)
    cov = corr * (stds @ stds.T)

    # Add a small ridge for numerical stability
    cov += np.eye(len(stds)) * 1e-8
    return cov


def _rebalance_with_caps(raw_weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Project weights onto the capped simplex (sum=1, 0<=w<=cap)."""

    weights = np.clip(np.asarray(raw_weights, dtype=float), 0.0, None)
    n = len(weights)
    if n == 0:
        return weights

    if weights.sum() == 0:
        weights = np.ones_like(weights)

    # Allow users to input either 0.25 or 25 for a 25 % cap
    cap = float(max_weight)
    if cap > 1.0:
        cap = cap / 100.0
    cap = max(0.0, min(cap, 1.0))

    # If the requested cap makes the constraint infeasible, lift it just enough
    if cap * n < 1.0:
        cap = 1.0 / n

    # Binary search for theta such that sum(min(max(w - theta, 0), cap)) == 1
    v = weights
    low = v.min() - cap
    high = v.max()

    def project(theta: float) -> np.ndarray:
        return np.clip(v - theta, 0.0, cap)

    for _ in range(100):
        mid = (low + high) / 2.0
        proj = project(mid)
        s = proj.sum()
        if abs(s - 1.0) < 1e-10:
            weights = proj
            break
        if s > 1.0:
            low = mid
        else:
            high = mid
        weights = proj

    if weights.sum() == 0:
        weights = np.ones_like(weights) / n
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


def _projected_gradient_sharpe(
    expected_returns: np.ndarray,
    corr_matrix: np.ndarray,
    stds: np.ndarray,
    max_weight: float,
    seeds: list[np.ndarray] | None = None,
    steps: int = 400,
    step_size: float = 0.12,
    random_restarts: int = 5,
    stop_event: threading.Event | None = None,
    max_seconds: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Projected gradient ascent for Sharpe with capped, long-only weights.

    - Uses multiple restarts (seeded by greedy weights, equal weights, and risk/return heuristics).
    - Projects onto the capped simplex after every gradient step to enforce constraints.
    - Returns the best weights and Sharpe it encountered across all restarts.
    """

    n = len(expected_returns)
    if n == 0:
        return np.array([]), 0.0

    cov = _safe_covariance_matrix(stds, corr_matrix.copy())
    deadline = time.monotonic() + max_seconds

    def _project(w: np.ndarray) -> np.ndarray:
        return _rebalance_with_caps(w, max_weight)

    # Build default seeds if none are supplied
    default_seeds: list[np.ndarray] = []
    default_seeds.append(np.ones(n) / n)  # equal weight
    default_seeds.append(np.clip(expected_returns, 0, None))  # proportional to expected returns

    seeds = seeds or []
    seeds = [s for s in seeds if isinstance(s, np.ndarray) and len(s) == n]
    seeds.extend(default_seeds)

    rng = np.random.default_rng()
    for _ in range(random_restarts):
        seeds.append(rng.random(n))

    best_w = _project(seeds[0]) if seeds else np.ones(n) / n
    best_sh = portfolio_sharpe(best_w, expected_returns, corr_matrix, stds)

    for seed in seeds:
        if stop_event is not None and stop_event.is_set():
            break
        if time.monotonic() >= deadline:
            break
        w = _project(seed)
        local_best_w = w.copy()
        local_best_sh = portfolio_sharpe(w, expected_returns, corr_matrix, stds)
        current_step = step_size

        for _ in range(steps):
            if stop_event is not None and stop_event.is_set():
                break
            if time.monotonic() >= deadline:
                break
            cov_w = cov @ w
            numer = float(expected_returns @ w)
            denom = float(np.sqrt(w @ cov_w))
            if denom <= 1e-12:
                break

            grad = expected_returns / denom - (numer / (denom**3)) * cov_w
            w = w + current_step * grad
            w = _project(w)

            sh = portfolio_sharpe(w, expected_returns, corr_matrix, stds)
            if sh > local_best_sh + 1e-10:
                local_best_sh = sh
                local_best_w = w.copy()

            current_step *= 0.995

        if local_best_sh > best_sh + 1e-10:
            best_sh = local_best_sh
            best_w = local_best_w

    return best_w, best_sh


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


def _coerce_to_series(series, ticker: str, log_fn) -> pd.Series:
    """Convert arbitrary return containers to a 1D Series, falling back to empty."""

    if isinstance(series, pd.Series):
        return series

    if isinstance(series, pd.DataFrame):
        if series.empty:
            return pd.Series(dtype=float)
        # Prefer first column if multiple are present
        return series.iloc[:, 0]

    try:
        arr = np.asarray(series)
    except Exception:
        log_fn(f"Nelze převést návratnosti pro {ticker}, používám prázdné.")
        return pd.Series(dtype=float)

    if arr.size == 0:
        return pd.Series(dtype=float)

    if arr.ndim > 1:
        log_fn(f"Očekával jsem 1D návratnosti pro {ticker}, redukuji tvar {arr.shape} na vektor.")
        arr = arr.reshape(-1)

    try:
        return pd.Series(arr)
    except Exception:
        log_fn(f"Nelze vytvořit sérii návratností pro {ticker}, používám prázdné.")
        return pd.Series(dtype=float)


def build_correlation_matrix(tickers: list[str], cache: ReturnCache, log_fn) -> np.ndarray:
    if not tickers:
        return np.zeros((0, 0))

    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        series = cache.get_returns(ticker, log_fn)
        series_map[ticker] = _coerce_to_series(series, ticker, log_fn)

    # pd.concat is more forgiving when some entries are scalars/empty; if still empty, fall back to zeros
    aligned = pd.concat(series_map, axis=1).dropna(how="all") if series_map else pd.DataFrame()
    if not aligned.empty:
        aligned.columns = tickers
    if aligned.empty:
        log_fn("Není dostatek dat pro korelace, použiji nulovou korelaci.")
        return np.zeros((len(tickers), len(tickers)))

    corr = aligned.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    return corr.values


# ---- Portfolio construction ---------------------------------------------


class PortfolioBuilder:
    def __init__(
        self,
        records: list[StockRecord],
        max_positions: int = 10,
        max_weight: float = 0.25,
        log_fn=print,
        stop_event: threading.Event | None = None,
    ):
        self.records = sorted(records, key=lambda r: r.sharpe, reverse=True)
        self.max_positions = max_positions
        self.max_weight = max_weight
        self.log = log_fn
        self.cache = ReturnCache()
        self.stop_event = stop_event

    def _should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            self.log("Výpočet byl zastaven uživatelem.")
            return True
        return False

    def build(self) -> tuple[list[StockRecord], np.ndarray, float]:
        positive_records = [r for r in self.records if r.sharpe > 0]
        if not positive_records:
            self.log("Žádné akcie s pozitivním Sharpe ratio, nemohu sestavit portfolio.")
            return [], np.array([]), 0.0

        self.records = positive_records
        if not self.records:
            self.log("Seznam akcií je prázdný.")
            return [], np.array([]), 0.0

        portfolio = self._optimize_sparse_portfolio()
        if not portfolio:
            portfolio = [self.records[0]]
            self.log(f"1) Přidávám {portfolio[0].ticker} s nejvyšším Sharpe {portfolio[0].sharpe:.3f}.")

        while len(portfolio) < min(self.max_positions, len(self.records)):
            if self._should_stop():
                return portfolio, np.array([]), 0.0
            remaining = [r for r in self.records if r.ticker not in {p.ticker for p in portfolio}]
            candidates = remaining[:50]
            if not candidates:
                break

            best_choice = None
            best_sharpe = -np.inf

            # Build correlation matrix for current portfolio plus candidate to evaluate
            for candidate in candidates:
                if self._should_stop():
                    return portfolio, np.array([]), 0.0
                trial_list = portfolio + [candidate]
                tickers = [r.ticker for r in trial_list]
                corr_matrix = build_correlation_matrix(tickers, self.cache, self.log)
                expected = np.array([r.forecast_pct for r in trial_list])
                stds = np.array([r.std_pct for r in trial_list])

                weights = optimize_weights(expected, corr_matrix, stds, self.max_weight)
                # Skip candidates that end up with ~0 váhu po optimalizaci
                if weights[-1] < 1e-4:
                    self.log(
                        f"   Přeskakuji {candidate.ticker}, optimální váha {weights[-1]:.4f} je příliš nízká."
                    )
                    continue
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

            # Recompute weights for the current portfolio, drop nulové váhy
            weights = optimize_weights(expected, corr_matrix, stds, self.max_weight)
            tickers_with_weights = list(zip([r.ticker for r in portfolio], weights))
            filtered = [(t, w) for t, w in tickers_with_weights if w >= 1e-4]
            if len(filtered) < len(tickers_with_weights):
                removed = {t for t, _ in tickers_with_weights} - {t for t, _ in filtered}
                for t in removed:
                    self.log(f"   Odstraňuji {t}, váha po optimalizaci je zanedbatelná.")
                portfolio = [r for r in portfolio if r.ticker in {t for t, _ in filtered}]
                weights = optimize_weights(
                    np.array([r.forecast_pct for r in portfolio]),
                    build_correlation_matrix([r.ticker for r in portfolio], self.cache, self.log),
                    np.array([r.std_pct for r in portfolio]),
                    self.max_weight,
                )

            for t, w in zip([r.ticker for r in portfolio], weights):
                self.log(f"   Váha {t}: {w:.2%}")

        # Final weights for the finished portfolio (avoid nulové váhy)
        if portfolio:
            greedy_corr = build_correlation_matrix([r.ticker for r in portfolio], self.cache, self.log)
            expected = np.array([r.forecast_pct for r in portfolio])
            stds = np.array([r.std_pct for r in portfolio])
            greedy_weights = optimize_weights(expected, greedy_corr, stds, self.max_weight)
            tickers_with_weights = list(zip([r.ticker for r in portfolio], greedy_weights))
            filtered = [(t, w) for t, w in tickers_with_weights if w >= 1e-4]
            if len(filtered) < len(tickers_with_weights):
                removed = {t for t, _ in tickers_with_weights} - {t for t, _ in filtered}
                for t in removed:
                    self.log(f"Konečná váha {t}: 0.00% (vyřazeno pro zanedbatelnou váhu)")
                portfolio = [r for r in portfolio if r.ticker in {t for t, _ in filtered}]
                greedy_corr = build_correlation_matrix([r.ticker for r in portfolio], self.cache, self.log)
                expected = np.array([r.forecast_pct for r in portfolio])
                stds = np.array([r.std_pct for r in portfolio])
                greedy_weights = optimize_weights(expected, greedy_corr, stds, self.max_weight)

            greedy_sharpe = portfolio_sharpe(greedy_weights, expected, greedy_corr, stds)
            self.log(f"Greedy Sharpe finálního portfolia: {greedy_sharpe:.3f}")

            refined_portfolio, refined_weights, refined_sharpe = self._refine_final_portfolio()
            best_portfolio = portfolio
            best_weights = greedy_weights
            best_sharpe = greedy_sharpe

            if refined_sharpe > best_sharpe + 1e-6:
                self.log(
                    f"Refinement zlepšil Sharpe na {refined_sharpe:.3f}; aktualizuji portfolia a váhy."
                )
                best_portfolio = refined_portfolio
                best_weights = refined_weights
                best_sharpe = refined_sharpe

            # Projected-gradient polish on the best-so-far set (dlouhé váhy, capped)
            corr_pg = build_correlation_matrix([r.ticker for r in best_portfolio], self.cache, self.log)
            expected_pg = np.array([r.forecast_pct for r in best_portfolio])
            stds_pg = np.array([r.std_pct for r in best_portfolio])
            pg_weights, pg_sharpe = _projected_gradient_sharpe(
                expected_pg,
                corr_pg,
                stds_pg,
                self.max_weight,
                seeds=[best_weights],
                stop_event=self.stop_event,
            )
            if pg_sharpe > best_sharpe + 1e-6:
                self.log(
                    f"Projektovaný gradient zvýšil Sharpe na {pg_sharpe:.3f}; používám vylepšené váhy."
                )
                best_weights = pg_weights
                best_sharpe = pg_sharpe

            # Drop nulové/zanedbatelné váhy po finální optimalizaci a případně ještě jednou zkus optimalizovat
            portfolio, final_weights, final_sharpe = self._prune_and_polish_final(
                best_portfolio, best_weights, best_sharpe
            )

            # Pokud máme méně pozic než požadavek, zkusíme portfolio znovu rozšířit
            if len(portfolio) < self.max_positions:
                portfolio, final_weights, final_sharpe = self._expand_portfolio(
                    portfolio, final_weights, final_sharpe
                )

            self.log(f"Konečný Sharpe po vyčištění: {final_sharpe:.3f}")
            for t, w in zip([r.ticker for r in portfolio], final_weights):
                self.log(f"Konečná váha {t}: {w:.2%}")

            # seřazení podle váhy pro finální reporting
            order = np.argsort(final_weights)[::-1]
            portfolio = [portfolio[i] for i in order]
            final_weights = final_weights[order]

        return portfolio, final_weights if 'final_weights' in locals() else np.array([]), final_sharpe if 'final_sharpe' in locals() else 0.0

    def _optimize_sparse_portfolio(self) -> list[StockRecord]:
        """Greedy + swap search for a sparse Markowitz-style portfolio."""
        if self._should_stop():
            return []

        pool_size = min(150, len(self.records))
        pool = self.records[:pool_size]
        tickers = [r.ticker for r in pool]
        self.log(f"Používám kandidátní pool {pool_size} akcií s pozitivním Sharpe.")

        corr_pool = build_correlation_matrix(tickers, self.cache, self.log)
        expected_pool = np.array([r.forecast_pct for r in pool])
        stds_pool = np.array([r.std_pct for r in pool])

        def submatrix(indices: list[int]) -> np.ndarray:
            idx = np.array(indices, dtype=int)
            return corr_pool[np.ix_(idx, idx)]

        def score(indices: list[int]) -> tuple[float, np.ndarray]:
            corr = submatrix(indices)
            expected = expected_pool[indices]
            stds = stds_pool[indices]
            weights = optimize_weights(expected, corr, stds, self.max_weight)
            sharpe = portfolio_sharpe(weights, expected, corr, stds)
            return sharpe, weights

        # Greedy forward selection
        target_size = min(self.max_positions, len(pool))
        selected = [0]
        best_sharpe, _ = score(selected)
        self.log(f"Greedy start: {pool[0].ticker}, Sharpe {best_sharpe:.3f}")

        while len(selected) < target_size:
            if self._should_stop():
                return [pool[i] for i in selected]
            best_candidate = None
            best_candidate_sharpe = best_sharpe
            for idx in range(len(pool)):
                if idx in selected:
                    continue
                trial = selected + [idx]
                sharpe, _ = score(trial)
                if sharpe > best_candidate_sharpe:
                    best_candidate_sharpe = sharpe
                    best_candidate = idx
            if best_candidate is None:
                break
            selected.append(best_candidate)
            best_sharpe = best_candidate_sharpe
            self.log(f"Greedy přidávám {pool[best_candidate].ticker}, Sharpe {best_sharpe:.3f}")

        # Local swap search
        max_passes = 4
        for _ in range(max_passes):
            if self._should_stop():
                break
            improved = False
            for i, idx_out in enumerate(list(selected)):
                if self._should_stop():
                    break
                for idx_in in range(len(pool)):
                    if idx_in in selected:
                        continue
                    trial = selected.copy()
                    trial[i] = idx_in
                    sharpe, _ = score(trial)
                    if sharpe > best_sharpe + 1e-6:
                        selected = trial
                        best_sharpe = sharpe
                        improved = True
                        self.log(
                            f"Swap vylepšil Sharpe na {best_sharpe:.3f} "
                            f"({pool[idx_out].ticker} -> {pool[idx_in].ticker})."
                        )
                        break
                if improved:
                    break
            if not improved:
                break

        return [pool[i] for i in selected]

    def _prune_and_polish_final(
        self, portfolio: list[StockRecord], weights: np.ndarray, sharpe: float
    ) -> tuple[list[StockRecord], np.ndarray, float]:
        """Remove near-zero allocations and re-optimize the remaining set.

        This prevents závěrečné portfolio from obsahující pozice s váhou ~0 a dává
        optimalizátoru ještě jednu šanci tyto prostředky přerozdělit.
        """

        tol = 1e-4
        current_portfolio = portfolio
        current_weights = np.array(weights, dtype=float)
        current_sharpe = float(sharpe)

        changed = True
        while changed and len(current_portfolio) > 0:
            if self._should_stop():
                break
            changed = False
            mask = current_weights > tol
            if mask.all():
                break

            removed = [r.ticker for r, keep in zip(current_portfolio, mask) if not keep]
            for ticker in removed:
                self.log(f"   Odstraňuji {ticker}, váha po optimalizaci je zanedbatelná.")

            current_portfolio = [r for r, keep in zip(current_portfolio, mask) if keep]
            current_weights = current_weights[mask]
            if len(current_portfolio) == 0:
                return [], np.array([]), 0.0

            # Rebalance removed weight back onto the capped simplex
            current_weights = _rebalance_with_caps(current_weights, self.max_weight)
            changed = True

        if not current_portfolio:
            return [], np.array([]), 0.0

        tickers = [r.ticker for r in current_portfolio]
        corr = build_correlation_matrix(tickers, self.cache, self.log)
        expected = np.array([r.forecast_pct for r in current_portfolio])
        stds = np.array([r.std_pct for r in current_portfolio])

        # Seed the gradient polish with the cleaned weights and a fresh closed-form start
        seed_weights = [current_weights]
        closed_form = optimize_weights(expected, corr, stds, self.max_weight)
        seed_weights.append(closed_form)

        pg_weights, pg_sharpe = _projected_gradient_sharpe(
            expected, corr, stds, self.max_weight, seeds=seed_weights, stop_event=self.stop_event
        )

        best_weights = current_weights
        best_sharpe = current_sharpe

        if pg_sharpe > best_sharpe + 1e-6:
            self.log(
                f"   Dodatečný gradient zvýšil Sharpe na {pg_sharpe:.3f} po odstranění nulových vah."
            )
            best_weights = pg_weights
            best_sharpe = pg_sharpe

        # Final pass to drop any residual ~0 váhy a renormalizovat
        mask = best_weights > tol
        if not mask.all():
            removed = [t for t, keep in zip(tickers, mask) if not keep]
            for ticker in removed:
                self.log(f"   Odstraňuji {ticker}, váha po polish je zanedbatelná.")
            current_portfolio = [r for r, keep in zip(current_portfolio, mask) if keep]
            best_weights = best_weights[mask]
            if len(current_portfolio) == 0:
                return [], np.array([]), 0.0
            tickers = [r.ticker for r in current_portfolio]
            corr = build_correlation_matrix(tickers, self.cache, self.log)
            expected = np.array([r.forecast_pct for r in current_portfolio])
            stds = np.array([r.std_pct for r in current_portfolio])
            best_weights = _rebalance_with_caps(best_weights, self.max_weight)
            best_sharpe = portfolio_sharpe(best_weights, expected, corr, stds)

        return current_portfolio, best_weights, best_sharpe

    def _expand_portfolio(
        self, portfolio: list[StockRecord], weights: np.ndarray, sharpe: float
    ) -> tuple[list[StockRecord], np.ndarray, float]:
        """Zkus přidat další tickery, pokud jich je méně než cílový počet.

        - Vezme nejlepší dosud nepoužité tickery a spustí gradientní optimalizaci
          na rozšířeném koši.
        - Poté ponechá maximálně `max_positions` pozic s nejvyšší vahou a znovu
          je renormalizuje na omezený simplex.
        - Nové řešení přijme pouze tehdy, pokud Sharpe neklesne (s tolerancí
          1e-6) oproti vstupu. Tím se udrží kvalita portfolia, ale pokud existuje
          řešení se stejným či vyšším Sharpe a vyšším počtem pozic, použije se.
        """

        if len(portfolio) >= self.max_positions:
            return portfolio, weights, sharpe

        unused = [r for r in self.records if r.ticker not in {p.ticker for p in portfolio}]
        if not unused:
            return portfolio, weights, sharpe

        slots_to_fill = self.max_positions - len(portfolio)
        # Zvažme výrazně více kandidátů než je potřeba, aby optimalizace měla prostor
        candidate_count = max(slots_to_fill * 6, self.max_positions * 4, 30)
        extra = unused[: min(len(unused), candidate_count)]
        expanded = portfolio + extra

        if self._should_stop():
            return portfolio, weights, sharpe

        tickers = [r.ticker for r in expanded]
        corr = build_correlation_matrix(tickers, self.cache, self.log)
        expected = np.array([r.forecast_pct for r in expanded])
        stds = np.array([r.std_pct for r in expanded])

        # Seed: původní váhy rozšířené o malou hodnotu pro nové tituly
        padded = np.zeros(len(expanded))
        padded[: len(weights)] = weights
        if padded.sum() <= 0:
            padded[: len(portfolio)] = 1.0 / len(portfolio)
        remaining = max(0.0, 1.0 - padded.sum())
        if remaining > 0 and len(expanded) > len(weights):
            padded[len(weights) :] = remaining / (len(expanded) - len(weights))

        seeds = [padded, np.ones(len(expanded)) / len(expanded), np.clip(expected, 0, None)]

        pg_weights, pg_sharpe = _projected_gradient_sharpe(
            expected,
            corr,
            stds,
            self.max_weight,
            seeds=seeds,
            random_restarts=8,
            steps=500,
            stop_event=self.stop_event,
            max_seconds=40.0,
        )

        # Ponecháme nejvyšší váhy do cílového počtu, zbytek zahodíme
        top_indices = np.argsort(pg_weights)[::-1][: self.max_positions]
        mask = np.zeros_like(pg_weights, dtype=bool)
        mask[top_indices] = True

        filtered_portfolio = [r for r, keep in zip(expanded, mask) if keep]
        filtered_weights = pg_weights[mask]

        # Keep weights strictly positive before projection to avoid nulové alokace
        filtered_weights = np.maximum(filtered_weights, 1e-6)
        filtered_weights = _rebalance_with_caps(filtered_weights, self.max_weight)
        filtered_sharpe = portfolio_sharpe(
            filtered_weights,
            np.array([r.forecast_pct for r in filtered_portfolio]),
            build_correlation_matrix([r.ticker for r in filtered_portfolio], self.cache, self.log),
            np.array([r.std_pct for r in filtered_portfolio]),
        )

        if filtered_sharpe + 1e-6 >= sharpe and len(filtered_portfolio) >= len(portfolio):
            self.log(
                f"   Rozšíření na {len(filtered_portfolio)} pozic s Sharpe {filtered_sharpe:.3f} (původně {sharpe:.3f})."
            )
            return filtered_portfolio, filtered_weights, filtered_sharpe

        return portfolio, weights, sharpe

    def _refine_final_portfolio(self) -> tuple[list[StockRecord], np.ndarray, float]:
        """Global re-optimization from širší množiny tickerů.

        Vezmeme širší koš (až 5× více tickerů než cílových pozic, max. 100),
        spočítáme optimální váhy a iterativně odřezáváme tituly s nejnižšími
        vahami, dokud nezůstaneme u požadovaného počtu pozic. Tím se eliminuje
        vliv dřívějších greedy kroků a uvolněný cap (např. 25 %) má šanci
        najít přinejmenším stejně dobré řešení jako přísnější cap.
        """

        if not self.records:
            return [], np.array([]), 0.0

        pool_size = min(len(self.records), max(1, min(self.max_positions * 8, 200)))
        candidate_pool = self.records[:pool_size]

        # Start with full pool
        current = candidate_pool.copy()
        while len(current) > self.max_positions:
            if self._should_stop():
                break
            tickers = [r.ticker for r in current]
            corr = build_correlation_matrix(tickers, self.cache, self.log)
            expected = np.array([r.forecast_pct for r in current])
            stds = np.array([r.std_pct for r in current])
            weights = optimize_weights(expected, corr, stds, self.max_weight)

            # Drop the smallest weight to satisfy the position limit
            drop_idx = int(np.argmin(weights))
            if weights[drop_idx] <= 1e-8:
                # If weights collapse to zeros, bail out
                break
            self.log(
                f"Refinement: odebírám {current[drop_idx].ticker} (váha {weights[drop_idx]:.4%}) z širšího koše."
            )
            current.pop(drop_idx)

        # Final optimization on the remaining set
        tickers = [r.ticker for r in current]
        corr = build_correlation_matrix(tickers, self.cache, self.log)
        expected = np.array([r.forecast_pct for r in current])
        stds = np.array([r.std_pct for r in current])
        final_weights = optimize_weights(expected, corr, stds, self.max_weight)
        final_sharpe = portfolio_sharpe(final_weights, expected, corr, stds)

        return current, final_weights, final_sharpe


# ---- UI ------------------------------------------------------------------


class PortfolioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Portfolio Builder")

        self.records: list[StockRecord] = []
        self.stop_event: threading.Event | None = None
        self.worker_thread: threading.Thread | None = None

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

        self.stop_button = ttk.Button(control_frame, text="Zastavit", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=(5, 0))

        self.start_button = ttk.Button(control_frame, text="Sestavit portfolio", command=self.start_build)
        self.start_button.pack(side=tk.RIGHT)

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
        """Převede hodnotu na float bez dalších úprav."""

        if isinstance(value, str):
            value = value.replace("%", "").replace(",", ".").strip()
        return float(value)

    @staticmethod
    def _to_percent_decimal(value) -> float:
        """Převede hodnotu na desetinné vyjádření procent.

        - Pokud je vstup ve tvaru řetězce s `%`, odstraní znak a vydělí 100.
        - Pokud je vstup číslo, považuje ho za již správně škálované (např. 0.031
          znamená 3.1 %). Nevkládá se žádná heuristika typu ">1 je procento",
          aby byla zachována konzistence napříč sloupci.
        """

        percent = False
        if isinstance(value, str):
            percent = "%" in value
            value = value.replace("%", "").replace(",", ".").strip()

        number = float(value)
        return number / 100.0 if percent else number

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
                    forecast_pct=self._to_percent_decimal(row.iloc[2]),
                    std_pct=self._to_percent_decimal(row.iloc[3]),
                )
                if rec.ticker:
                    self.records.append(rec)
                else:
                    raise ValueError("ticker je prázdný")
            except Exception:
                self.log(f"Přeskakuji řádek: {row}")

        self.tree.delete(*self.tree.get_children())
        for rec in self.records:
            self.tree.insert("", tk.END, values=(rec.ticker, f"{rec.sharpe:.3f}", f"{rec.forecast_pct:.3f}", f"{rec.std_pct:.3f}"))

        self.log(
            f"Načteno {len(self.records)} záznamů z {os.path.basename(path)}. Hlavička není nutná; očekávám pořadí: ticker, sharpe, forecast %, std %."
        )

    def start_build(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Probíhá výpočet", "Vyčkejte na dokončení nebo použijte 'Zastavit'.")
            return

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

        self.stop_event = threading.Event()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        def worker():
            try:
                builder = PortfolioBuilder(
                    self.records,
                    max_positions=count,
                    max_weight=cap,
                    log_fn=self.log,
                    stop_event=self.stop_event,
                )
                portfolio, weights, sharpe = builder.build()
                if self.stop_event.is_set():
                    self.log("Výpočet ukončen na žádost uživatele.")
                    return
                self._report_results(portfolio, weights, sharpe)
            except Exception as exc:  # pragma: no cover - UI error path
                self.log(f"Chyba při výpočtu: {exc}")
                messagebox.showerror("Chyba", str(exc))
            finally:
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def request_stop(self):
        if self.stop_event is not None:
            self.stop_event.set()

    def _report_results(self, portfolio: list[StockRecord], weights: np.ndarray, sharpe: float):
        if not portfolio or weights.size == 0:
            self.log("Výsledek je prázdný.")
            return

        tickers = [r.ticker for r in portfolio]
        corr = build_correlation_matrix(tickers, ReturnCache(), self.log)
        expected = np.array([r.forecast_pct for r in portfolio])
        stds = np.array([r.std_pct for r in portfolio])
        cov = _safe_covariance_matrix(stds, corr.copy())
        port_ret = float(weights @ expected)
        port_vol = float(np.sqrt(weights @ cov @ weights))

        self.log(f"Hotovo. Portfolio: {', '.join(tickers)}")
        self.log(
            f"Sharpe: {sharpe:.3f}, očekávaný výnos portfolia: {port_ret*100:.2f} %, směrodatná odchylka: {port_vol*100:.2f} %"
        )

        table = pd.DataFrame(
            {
                "Ticker": tickers,
                "Váha %": weights * 100,
                "Očekávaný výnos %": expected * 100,
                "Směrodatná odchylka %": stds * 100,
            }
        )
        self.log("Váhy a parametry (seřazeno dle váhy):")
        for line in table.to_string(index=False, float_format=lambda x: f"{x:6.2f}").splitlines():
            self.log(f"   {line}")

        corr_df = pd.DataFrame(corr, index=tickers, columns=tickers)
        self.log("Korelační matice finálního portfolia:")
        for line in corr_df.round(2).to_string().splitlines():
            self.log(f"   {line}")


def main():
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
