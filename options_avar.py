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


def calc_avar(
    spot: float,
    r: float,
    T: float,
    sel_strikes: list,
    c_prices: list,
    p_prices: list,
) -> float | None:
    """
    AVAR using the spreadsheet's model-free implied variance methodology.

    Forward price F = spot * e^(r*T).

    Integrand per strike:
        K > F  (OTM calls): (K - F)² * C(K) / K²
        K < F  (OTM puts):  (F - K)² * P(K) / K²
        K ≈ F  (ATM):       0  (weight is zero)

    AVAR = (Sum_Call - Sum_Put) / (Sum_Call + Sum_Put)

    The prefactor 2*e^(rT) / (T * F²) is identical for both sums and
    cancels in the ratio, so only the raw integrand sums are needed.
    """
    F = spot * math.exp(r * T)

    sum_call = 0.0
    sum_put  = 0.0

    for K, C, P in zip(sel_strikes, c_prices, p_prices):
        if K > F and C is not None:
            sum_call += (K - F) ** 2 * C / K ** 2
        elif K < F and P is not None:
            sum_put  += (F - K) ** 2 * P / K ** 2
        # K ≈ F → zero weight, skip

    total = sum_call + sum_put
    if total == 0:
        return None, False

    one_sided = sum_call == 0 or sum_put == 0
    return (sum_call - sum_put) / total, one_sided


# ─── Per-symbol calculation ────────────────────────────────────────────────────

def get_avar_for_symbol(symbol: str, r: float) -> tuple[float | None, bool]:
    """
    Fetch option data for *symbol* and return (AVAR, one_sided).
    one_sided=True means all strikes fell on one side of the forward price,
    making the AVAR unreliable. Returns (None, False) on data errors.
    """
    ticker = yf.Ticker(symbol)

    try:
        spot = fetch_spot(ticker)
    except Exception:
        print(f"  [{symbol}] ERROR: could not retrieve spot price.")
        return None, False

    exps = ticker.options
    if not exps:
        print(f"  [{symbol}] ERROR: no listed options found.")
        return None, False

    today = datetime.now().strftime("%Y-%m-%d")
    exp = exps[0]
    if exp == today and len(exps) > 1:
        exp = exps[1]

    try:
        chain = ticker.option_chain(exp)
    except Exception as exc:
        print(f"  [{symbol}] ERROR: failed to load option chain: {exc}")
        return None, False

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
        return None, False

    lo          = max(0, atm_idx - 5)
    hi          = min(len(all_strikes) - 1, atm_idx + 5)
    sel_strikes = all_strikes[lo : hi + 1]

    c_map    = dict(zip(calls["strike"], calls["lastPrice"]))
    p_map    = dict(zip(puts["strike"],  puts["lastPrice"]))
    c_prices = [clean_price(c_map.get(k)) for k in sel_strikes]
    p_prices = [clean_price(p_map.get(k)) for k in sel_strikes]

    return calc_avar(spot, r, T, sel_strikes, c_prices, p_prices)


# ─── Strategy recommendation ──────────────────────────────────────────────────

def recommend_strategy(avar: float) -> str:
    if avar < -0.15:
        return "Sell puts / put credit spread (rich downside premium)"
    elif avar < -0.05:
        return "Bull put spread (modest downside skew)"
    elif avar <= 0.05:
        return "Iron condor / straddle (balanced variance)"
    elif avar <= 0.15:
        return "Bear call spread (modest upside skew)"
    else:
        return "Sell calls / call credit spread (rich upside premium)"


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

    results: list[tuple[str, float | None, bool]] = []

    for symbol in symbols:
        print(f"Processing {symbol} …")
        avar, one_sided = get_avar_for_symbol(symbol, r)
        results.append((symbol, avar, one_sided))
        if avar is not None:
            warn = "  ⚠ one-sided strike coverage" if one_sided else ""
            print(f"  [{symbol}] AVAR = {avar:.6f}{warn}")
        else:
            print(f"  [{symbol}] AVAR = N/A")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    bar = "─" * 72
    print(bar)
    print(f"  {'Symbol':<10}  {'AVAR':>10}  {'Recommended Strategy'}")
    print(bar)
    for symbol, avar, one_sided in results:
        avar_str = f"{avar:.6f}" if avar is not None else "N/A"
        if avar is None:
            strategy = "N/A"
        elif one_sided:
            strategy = f"{recommend_strategy(avar)}  ⚠ unreliable (one-sided)"
        else:
            strategy = recommend_strategy(avar)
        print(f"  {symbol:<10}  {avar_str:>10}  {strategy}")
    print(bar)


if __name__ == "__main__":
    main()
