# Stahování delších řad fundamentů

Primárním zdrojem je nyní **Financial Modeling Prep (FMP)**, který poskytuje několik desetiletí
čtvrtletních i ročních výkazů přes REST API. Skript `fetch_fmp_fundamentals.py` stáhne
požadované metriky z výkazu zisku a ztráty a uloží je do CSV.

## Požadavky
- API klíč FMP: nastavte proměnnou `FMP_API_KEY` nebo předejte `--api-key`.
- Python balík `requests` (už je používán i jinde v repozitáři).

## Příklad použití
Stáhnout 40 let kvartálních výkazů pro TSLA a AAPL a spočítat TTM hodnoty z pole `epsdiluted`:
```bash
python fetch_fmp_fundamentals.py \
  --tickers TSLA,AAPL \
  --metrics epsdiluted \
  --ttm \
  --limit 160 \
  --out-dir data/fmp
```
Výstupní soubory budou pojmenovány jako `TSLA_epsdiluted_ttm.csv` a obsahují sloupce `date,epsdiluted_ttm`.

## Tipy a poznámky
- `--metrics` může obsahovat více polí oddělených čárkou (např. `epsdiluted,ebit`). Dostupná pole
  najdete v dokumentaci FMP k endpointu Income Statement.
- Přepínač `--period annual` stáhne roční výkazy; bez `--ttm` se uloží surové hodnoty.
- `--limit` určujte podle hloubky dat, kterou chcete (160 kvartálů ~ 40 let, 40 ročních výkazů ~ 40 let).
- Pokud potřebujete jiné typy fundamentů, lze analogicky rozšířit skript na další FMP endpointy
  (balance sheet, cash flow, key metrics).

## Poznámka k Zacks
Původní skript `fetch_zacks_fundamentals.py` je v repozitáři zachován pro případné experimenty,
ale Zacks často blokuje automatické dotazy. Primární doporučený zdroj je nyní FMP.
