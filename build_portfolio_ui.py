"""Jednoduché grafické rozhraní pro skript `build_portfolio.py`.

Umožňuje zvolit soubory s predikcemi, metadaty a cenami, nastavit klíčové
parametry Kellyho alokace a uložit výslednou alokaci i korelační matici.
"""
from __future__ import annotations

import argparse
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from build_portfolio import build_portfolio


DEFAULTS = {
    "price_pattern": "data/weekly_prices/*.xlsx",
    "price_column_index": 79,
    "ticker_column": "Ticker",
    "return_column": "prediction",
    "probability_column": "probability",
    "min_probability": 0.5,
    "horizon_steps": 4,
    "assumed_downside": -0.02,
    "kelly_fraction": 1.0,
    "max_weight": 0.25,
    "output": "portfolio_allocation.csv",
    "correlation_output": "portfolio_correlations.csv",
}


class PortfolioUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kelly Portfolio Builder")
        self.root.geometry("720x650")

        self.predictions_var = tk.StringVar()
        self.metadata_var = tk.StringVar()
        self.price_pattern_var = tk.StringVar(value=DEFAULTS["price_pattern"])
        self.price_column_index_var = tk.IntVar(value=DEFAULTS["price_column_index"])
        self.price_column_name_var = tk.StringVar()
        self.ticker_column_var = tk.StringVar(value=DEFAULTS["ticker_column"])
        self.horizon_column_var = tk.StringVar()
        self.horizon_var = tk.StringVar()
        self.return_column_var = tk.StringVar(value=DEFAULTS["return_column"])
        self.prob_column_var = tk.StringVar(value=DEFAULTS["probability_column"])
        self.min_prob_var = tk.DoubleVar(value=DEFAULTS["min_probability"])
        self.top_n_var = tk.StringVar()
        self.horizon_steps_var = tk.IntVar(value=DEFAULTS["horizon_steps"])
        self.assumed_downside_var = tk.DoubleVar(value=DEFAULTS["assumed_downside"])
        self.kelly_fraction_var = tk.DoubleVar(value=DEFAULTS["kelly_fraction"])
        self.allow_shorts_var = tk.BooleanVar(value=False)
        self.max_weight_var = tk.StringVar(value=str(DEFAULTS["max_weight"]))
        self.output_var = tk.StringVar(value=DEFAULTS["output"])
        self.corr_output_var = tk.StringVar(value=DEFAULTS["correlation_output"])

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(container, text="Soubory")
        file_frame.pack(fill=tk.X, pady=5)
        self._add_file_picker(file_frame, "Predikce", self.predictions_var, required=True)
        self._add_file_picker(file_frame, "Metadata (volitelné)", self.metadata_var)

        price_frame = ttk.LabelFrame(container, text="Cenová historie")
        price_frame.pack(fill=tk.X, pady=5)
        self._add_entry(price_frame, "Glob pattern cen", self.price_pattern_var)
        self._add_entry(price_frame, "Index cenového sloupce (0=1. sloupec)", self.price_column_index_var)
        self._add_entry(price_frame, "Název cenového sloupce (přepíše index)", self.price_column_name_var)
        self._add_entry(price_frame, "Sloupec s tickerem", self.ticker_column_var)

        prediction_frame = ttk.LabelFrame(container, text="Predikce a horizont")
        prediction_frame.pack(fill=tk.X, pady=5)
        self._add_entry(prediction_frame, "Sloupec s horizontem (volitelné)", self.horizon_column_var)
        self._add_entry(prediction_frame, "Hodnota horizontu (např. 4w)", self.horizon_var)
        self._add_entry(prediction_frame, "Sloupec s predikovaným výnosem", self.return_column_var)
        self._add_entry(prediction_frame, "Sloupec s pravděpodobností", self.prob_column_var)
        self._add_entry(prediction_frame, "Min. pravděpodobnost", self.min_prob_var)
        self._add_entry(prediction_frame, "Top N dle predikce (volitelné)", self.top_n_var)
        self._add_entry(prediction_frame, "Počet kroků horizontu (např. týdny)", self.horizon_steps_var)

        risk_frame = ttk.LabelFrame(container, text="Parametry Kellyho alokace")
        risk_frame.pack(fill=tk.X, pady=5)
        self._add_entry(risk_frame, "Předpokládaný pokles při neúspěchu", self.assumed_downside_var)
        self._add_entry(risk_frame, "Kelly frakce", self.kelly_fraction_var)
        ttk.Checkbutton(risk_frame, text="Povolit short pozice", variable=self.allow_shorts_var).pack(fill=tk.X, padx=5, pady=2)
        self._add_entry(risk_frame, "Max. váha jedné pozice (None = bez limitu)", self.max_weight_var)

        output_frame = ttk.LabelFrame(container, text="Výstupy")
        output_frame.pack(fill=tk.X, pady=5)
        self._add_file_picker(output_frame, "CSV s alokací", self.output_var, save=True)
        self._add_file_picker(output_frame, "CSV s korelacemi", self.corr_output_var, save=True)

        action_frame = ttk.Frame(container)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="Vytvořit portfolio", command=self._on_run).pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(container, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_file_picker(self, parent: ttk.Frame, label: str, variable: tk.StringVar, required: bool = False, save: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        def choose_file() -> None:
            if save:
                path = filedialog.asksaveasfilename()
            else:
                path = filedialog.askopenfilename()
            if path:
                variable.set(path)

        ttk.Button(row, text="Vybrat", command=choose_file).pack(side=tk.LEFT)
        if required:
            ttk.Label(row, text="*").pack(side=tk.LEFT, padx=4)

    def _add_entry(self, parent: ttk.Frame, label: str, variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=33, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _on_run(self) -> None:
        thread = threading.Thread(target=self._run_portfolio, daemon=True)
        thread.start()

    def _run_portfolio(self) -> None:
        try:
            args = self._gather_arguments()
        except ValueError as exc:
            messagebox.showerror("Neplatné parametry", str(exc))
            return

        self._append_log("Spouštím výpočet portfolia...")
        try:
            portfolio, correlation = build_portfolio(args)
        except Exception as exc:  # noqa: BLE001 - zobrazit chybu uživateli
            messagebox.showerror("Chyba při výpočtu", str(exc))
            return

        portfolio.to_csv(args.output, index=False)
        correlation.to_csv(args.correlation_output)
        self._append_log(f"Uloženo portfolio: {args.output}")
        self._append_log(f"Uloženy korelace: {args.correlation_output}")
        messagebox.showinfo("Hotovo", "Portfolio bylo úspěšně vytvořeno.")

    def _gather_arguments(self):
        predictions = self.predictions_var.get().strip()
        if not predictions:
            raise ValueError("Musíte vybrat soubor s predikcemi.")

        metadata = self.metadata_var.get().strip() or None
        price_pattern = self.price_pattern_var.get().strip() or DEFAULTS["price_pattern"]
        price_column_index = self.price_column_index_var.get()
        price_column_name = self.price_column_name_var.get().strip() or None
        ticker_column = self.ticker_column_var.get().strip() or DEFAULTS["ticker_column"]
        horizon_column = self.horizon_column_var.get().strip() or None
        horizon = self.horizon_var.get().strip() or None
        return_column = self.return_column_var.get().strip() or DEFAULTS["return_column"]
        prob_column = self.prob_column_var.get().strip() or DEFAULTS["probability_column"]
        min_prob = float(self.min_prob_var.get())
        top_n_text = self.top_n_var.get().strip()
        top_n = int(top_n_text) if top_n_text else None
        horizon_steps = int(self.horizon_steps_var.get())
        assumed_downside = float(self.assumed_downside_var.get())
        kelly_fraction = float(self.kelly_fraction_var.get())
        allow_shorts = bool(self.allow_shorts_var.get())
        max_weight_text = self.max_weight_var.get().strip()
        max_weight = float(max_weight_text) if max_weight_text.lower() != "none" and max_weight_text else None
        output = self.output_var.get().strip() or DEFAULTS["output"]
        corr_output = self.corr_output_var.get().strip() or DEFAULTS["correlation_output"]

        return argparse.Namespace(
            predictions=predictions,
            metadata=metadata,
            price_pattern=price_pattern,
            price_column_index=price_column_index,
            price_column_name=price_column_name,
            ticker_column=ticker_column,
            horizon_column=horizon_column,
            horizon=horizon,
            return_column=return_column,
            probability_column=prob_column,
            min_probability=min_prob,
            top_n=top_n,
            horizon_steps=horizon_steps,
            assumed_downside=assumed_downside,
            kelly_fraction=kelly_fraction,
            allow_shorts=allow_shorts,
            max_weight=max_weight,
            output=output,
            correlation_output=corr_output,
        )


def main() -> None:
    root = tk.Tk()
    PortfolioUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
