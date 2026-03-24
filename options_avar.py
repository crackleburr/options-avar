#!/usr/bin/env python3
"""
Options AVAR Calculator

Reads a comma-delimited list of stock symbols from a file, fetches the
nearest-expiration option chain for each, identifies the ATM strike
(call delta closest to 0.50), retrieves last prices for ATM ± 5 strikes,
and computes AVAR (population variance of the option prices).

Install dependencies:
    pip install yfinance numpy scipy
"""

import sys
import math
from datetime import datetime

import yfinance as yf
import numpy as np
from scipy.stats import norm


SYMBOLS_FILE = "/mnt/e/temp/symbols.txt"


# ─── Pricing helpers ──────────────────────────────────────────────────────────

def fetch_spot(ticker: yf.Ticker) -> float:
    """Return the most recent close price using intraday history."""
    hist = ticker.history(period="1d", interval="1m")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    hist = ticker.history(period="2d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise ValueError("Could not retrieve price data.")


def fetch_risk_free_rate() -> float:
    """Return the current 10-year Treasury yield; fallback to 4.5%."""
    try:
        rate = fetch_spot(yf.Ticker("^TNX")) / 100
        return rate
    except Exception:
        return 0.045


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta N(d1)."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S >= K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def clean_price(p) -> float | None:
    """Return None for missing, NaN, or zero prices."""
    if p is None:
        return None
    try:
        f = float(p)
    except (TypeError, ValueError):
        return None
    return f if (not math.isnan(f) and f > 0) else None


def calc_avar(prices: list) -> float | None:
    """
    AVAR = population variance of all valid option prices.
    Returns None if fewer than 2 valid prices exist.
    """
    valid = [p for p in prices if p is not None]
    return float(np.var(valid)) if len(valid) >= 2 else None


# ─── Per-symbol calculation ────────────────────────────────────────────────────

def get_avar_for_symbol(symbol: str, r: float) -> float | None:
    """
    Fetch option data for *symbol* and return its AVAR, or None on failure.
    Prints status lines prefixed with the symbol name.
    """
    ticker = yf.Ticker(symbol)

    try:
        spot = fetch_spot(ticker)
    except Exception:
        print(f"  [{symbol}] ERROR: could not retrieve spot price.")
        return None

    exps = ticker.options
    if not exps:
        print(f"  [{symbol}] ERROR: no listed options found.")
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    exp = exps[0]
    if exp == today and len(exps) > 1:
        exp = exps[1]

    try:
        chain = ticker.option_chain(exp)
    except Exception as exc:
        print(f"  [{symbol}] ERROR: failed to load option chain: {exc}")
        return None

    calls = chain.calls.copy()
    puts  = chain.puts.copy()

    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
    T = max((exp_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600), 1 / 365)

    def row_delta(row) -> float:
        iv = row["impliedVolatility"]
        if not iv or math.isnan(iv) or iv <= 0:
            iv = 0.30
        return bs_call_delta(spot, row["strike"], T, r, iv)

    calls["delta"] = calls.apply(row_delta, axis=1)
    calls["d_dist"] = (calls["delta"] - 0.50).abs()

    min_d = calls["d_dist"].min()
    tied  = calls[np.isclose(calls["d_dist"], min_d, atol=1e-6)]
    atm_k = float(tied["strike"].max())

    all_strikes = sorted(calls["strike"].unique().tolist())
    try:
        atm_idx = next(i for i, k in enumerate(all_strikes) if math.isclose(k, atm_k))
    except StopIteration:
        print(f"  [{symbol}] ERROR: could not locate ATM strike.")
        return None

    lo          = max(0, atm_idx - 5)
    hi          = min(len(all_strikes) - 1, atm_idx + 5)
    sel_strikes = all_strikes[lo : hi + 1]

    c_map    = dict(zip(calls["strike"], calls["lastPrice"]))
    p_map    = dict(zip(puts["strike"],  puts["lastPrice"]))
    c_prices = [clean_price(c_map.get(k)) for k in sel_strikes]
    p_prices = [clean_price(p_map.get(k)) for k in sel_strikes]

    return calc_avar(c_prices + p_prices)


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_symbols(path: str) -> list[str]:
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError as exc:
        sys.exit(f"Cannot open symbols file '{path}': {exc}")
    symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
    if not symbols:
        sys.exit(f"No symbols found in '{path}'.")
    return symbols


def main() -> None:
    symbols = load_symbols(SYMBOLS_FILE)
    print(f"Symbols loaded: {', '.join(symbols)}\n")

    r = fetch_risk_free_rate()
    print(f"Risk-free rate : {r:.2%}\n")

    results: list[tuple[str, float | None]] = []

    for symbol in symbols:
        print(f"Processing {symbol} …")
        avar = get_avar_for_symbol(symbol, r)
        results.append((symbol, avar))
        if avar is not None:
            print(f"  [{symbol}] AVAR = {avar:.6f}")
        else:
            print(f"  [{symbol}] AVAR = N/A")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    bar = "─" * 30
    print(bar)
    print(f"  {'Symbol':<10}  {'AVAR':>12}")
    print(bar)
    for symbol, avar in results:
        avar_str = f"{avar:.6f}" if avar is not None else "N/A"
        print(f"  {symbol:<10}  {avar_str:>12}")
    print(bar)


if __name__ == "__main__":
    main()
