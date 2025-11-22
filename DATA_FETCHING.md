# Stahování delších řad fundamentů ze Zacks

Tento repozitář nyní obsahuje skript `fetch_zacks_fundamentals.py`, který se pokusí stáhnout časovou řadu
z fundamentálních grafů na Zacks (např. EPS diluted TTM) a uložit ji do CSV.

## Jak skript funguje
1. Nejprve zavolá nezdokumentovaný JSON feed `https://www.zacks.com/includes/fundamental_charts.php`,
   který některé stránky používají pro vykreslení grafu.
2. Pokud feed selže nebo vrátí neočekávaný formát, skript stáhne HTML verzi grafu a hledá JSON bloky
   se sériemi přímo v `<script>` tagu. Když se nepodaří nic najít, HTML uloží do `data/zacks_failed_html/`
   pro ruční ladění regexů.

## Příklad použití
- Jedna akcie a jeden metriku:
  ```bash
  python fetch_zacks_fundamentals.py --ticker TSLA --metric eps-diluted-ttm --out data/tsla_eps.csv
  ```
- Více tickerů:
  ```bash
  python fetch_zacks_fundamentals.py --tickers TSLA,AAPL,MSFT --metric eps-diluted-ttm --out-dir data/zacks
  ```

## Tipy proti blokování
- Přidejte vlastní User-Agent (parametr `--user-agent`).
- Pokud síť vyžaduje proxy, nastavte proměnné `HTTP_PROXY` / `HTTPS_PROXY`.
- Pokud parsování selže, otevřete uložený HTML soubor z `data/zacks_failed_html/` a upravte regexy
  v `extract_from_html` tak, aby odpovídaly nové struktuře stránky.

## Omezení
- Zacks může kdykoliv změnit strukturu stránky nebo blokovat automatizované přístupy. Skript proto obsahuje
  ladicí výstup a úložiště selhaných HTML pro rychlou úpravu.
- Hodnoty se ukládají v pořadí datumu s přesností na šest desetin. Pokud stránka používá čtvrtletní popisky
  (např. `Q1 2023`), datum se uloží jako první den čtvrtletí.
