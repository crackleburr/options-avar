#!/usr/bin/env python3
"""
Options AVAR Calculator

Fetches the nearest-expiration option chain for a given stock symbol,
identifies the ATM strike (call delta closest to 0.50), retrieves last
prices for ATM ± 5 strikes, and computes AVAR (population variance of
the 22 option prices).

Install dependencies:
    pip install yfinance numpy scipy
"""

import sys
import math
from datetime import datetime

import yfinance as yf
import numpy as np
from scipy.stats import norm


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


# ─── Display helpers ──────────────────────────────────────────────────────────

def fmt(p: float | None, width: int = 9) -> str:
    if p is None:
        return "N/A".center(width)
    return f"${p:.2f}".rjust(width)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    symbol = input("Stock symbol: ").strip().upper()
    if not symbol:
        sys.exit("No symbol provided.")

    print(f"\nFetching data for {symbol} …")
    ticker = yf.Ticker(symbol)

    # Current spot price
    try:
        spot = fetch_spot(ticker)
    except Exception:
        sys.exit(f"Could not retrieve price data for '{symbol}'.")
    print(f"Spot price  : ${spot:.2f}")

    # Nearest expiration date — skip today (0DTE) if a later date exists
    exps = ticker.options
    if not exps:
        sys.exit(f"No listed options found for '{symbol}'.")
    today = datetime.now().strftime("%Y-%m-%d")
    exp = exps[0]
    if exp == today and len(exps) > 1:
        exp = exps[1]
        print(f"Expiration  : {exp}  (skipped 0DTE {today})")
    else:
        print(f"Expiration  : {exp}")

    # Option chain
    try:
        chain = ticker.option_chain(exp)
    except Exception as exc:
        sys.exit(f"Failed to load option chain: {exc}")

    calls = chain.calls.copy()
    puts  = chain.puts.copy()

    # Time to expiry in years (floor at 1 day)
    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
    T = max((exp_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600), 1 / 365)

    # Risk-free rate
    r = fetch_risk_free_rate()
    print(f"Risk-free   : {r:.2%}")

    # Compute call delta for every strike
    def row_delta(row) -> float:
        iv = row["impliedVolatility"]
        if not iv or math.isnan(iv) or iv <= 0:
            iv = 0.30          # fallback IV when missing
        return bs_call_delta(spot, row["strike"], T, r, iv)

    calls["delta"] = calls.apply(row_delta, axis=1)
    calls["d_dist"] = (calls["delta"] - 0.50).abs()

    # ATM strike: call delta closest to 0.50; higher strike wins a tie
    min_d   = calls["d_dist"].min()
    tied    = calls[np.isclose(calls["d_dist"], min_d, atol=1e-6)]
    atm_k   = float(tied["strike"].max())
    atm_d   = float(tied.loc[np.isclose(tied["strike"], atm_k), "delta"].iat[0])
    print(f"ATM strike  : ${atm_k:.2f}  (call Δ = {atm_d:.4f})")

    # Select ATM ± 5 strikes
    all_strikes = sorted(calls["strike"].unique().tolist())
    try:
        atm_idx = next(i for i, k in enumerate(all_strikes) if math.isclose(k, atm_k))
    except StopIteration:
        sys.exit("Could not locate ATM strike in strike list.")

    lo          = max(0, atm_idx - 5)
    hi          = min(len(all_strikes) - 1, atm_idx + 5)
    sel_strikes = all_strikes[lo : hi + 1]

    n = len(sel_strikes)
    if n < 11:
        print(f"Warning: only {n} strikes available (expected 11).")

    # Last-price lookups
    c_map = dict(zip(calls["strike"], calls["lastPrice"]))
    p_map = dict(zip(puts["strike"],  puts["lastPrice"]))

    c_prices = [clean_price(c_map.get(k)) for k in sel_strikes]
    p_prices = [clean_price(p_map.get(k)) for k in sel_strikes]

    # AVAR
    avar = calc_avar(c_prices + p_prices)

    # ── Output ────────────────────────────────────────────────────────────────
    bar = "─" * 50

    print(f"\n{bar}")
    if avar is not None:
        print(f"  AVAR  =  {avar:.6f}")
    else:
        print("  AVAR  =  N/A  (insufficient price data)")
    print(f"{bar}")

    hdr_c = "Call"
    hdr_s = "Strike"
    hdr_p = "Put"
    print(f"\n  {hdr_c:>9}   {hdr_s:>8}   {hdr_p:<9}")
    print(f"  {'─'*9}   {'─'*8}   {'─'*9}")

    for i, k in enumerate(sel_strikes):
        atm_tag = "  ◄ ATM" if math.isclose(k, atm_k) else ""
        print(f"  {fmt(c_prices[i]):>9}   {k:>8.2f}   {fmt(p_prices[i]):<9}{atm_tag}")

    print()


if __name__ == "__main__":
    main()
