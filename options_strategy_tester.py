# options_strategy_tester.py
# Run:  pip install yfinance pandas numpy matplotlib
#       python options_strategy_tester.py

import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------- USER SETTINGS -----------------
TICKER = "TSLA"
# choose a strategy by uncommenting one block in build_strategy() below
USE_NEAREST_EXPIRY = True   # if False, set EXPIRY manually
EXPIRY = None               # e.g. "2025-12-19"
RISK_FREE = 0.05            # for discounting (used only in parity helper)
SLIPPAGE_PER_LEG = 0.02     # $/contract slippage cushion for mark-to-market
ST_RANGE_PCT = (0.6, 1.4)   # payoff chart S_T range as % of spot (min, max)
STEPS = 200                 # points on payoff curve
# -------------------------------------------------

def get_spot_and_chain(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError("No price history returned. Check ticker or internet.")
    S0 = float(hist["Close"].iloc[-1])
    exps = tk.options
    if not exps:
        raise RuntimeError("No options found for this ticker.")
    if USE_NEAREST_EXPIRY:
        expiry = exps[0]
    else:
        expiry = EXPIRY or exps[0]
    oc = tk.option_chain(expiry)
    calls = oc.calls.rename(columns={"bid":"call_bid","ask":"call_ask","lastPrice":"call_last"})
    puts  = oc.puts .rename(columns={"bid":"put_bid","ask":"put_ask","lastPrice":"put_last"})
    return S0, expiry, calls, puts

def mid(b, a):
    if b is None or a is None:
        return np.nan
    return (float(b) + float(a)) / 2.0

# ---- basic legs (expiry payoff per share) ----
def call_payoff_long(ST, K, premium):   return np.maximum(ST - K, 0.0) - premium
def call_payoff_short(ST, K, premium):  return -call_payoff_long(ST, K, premium)
def put_payoff_long(ST, K, premium):    return np.maximum(K - ST, 0.0) - premium
def put_payoff_short(ST, K, premium):   return -put_payoff_long(ST, K, premium)
def stock_long(ST, S0):                 return ST - S0
def stock_short(ST, S0):                return -stock_long(ST, S0)

# ---- strategy builder: return list of legs with quotes ----
def build_strategy(S0, calls, puts):
    """
    Choose ONE example by uncommenting its block.
    Replace strikes with what you want. This function returns a list of legs:
      each leg is dict(type, side, strike, premium, qty, asset)
    where asset in {'call','put','stock'} and qty is number of shares/option contracts *per share* basis.
    We work per-share; multiply by 100 later for 1 US option contract.
    """

    # ===== Example A: ATM Long Straddle (buy call + buy put) =====
    # Pick strike nearest spot:
    k = float((calls.assign(diff=(calls["strike"] - S0).abs())
               .sort_values("diff")
               .iloc[0]["strike"]))
    c_mid = mid(calls.loc[calls["strike"]==k, "call_bid"].values[0],
                calls.loc[calls["strike"]==k, "call_ask"].values[0])
    p_mid = mid(puts.loc[puts["strike"]==k, "put_bid"].values[0],
                puts.loc[puts["strike"]==k, "put_ask"].values[0])
    legs = [
        {"asset":"call","side":"long","strike":k,"premium":c_mid, "qty":1.0},
        {"asset":"put", "side":"long","strike":k,"premium":p_mid, "qty":1.0},
    ]
    return legs

    # ===== Example B: Bull Call Spread (buy K1, sell K2) =====
    # K1 = round_to_step(S0*0.95, calls["strike"].diff().dropna().mode().iloc[0])
    # K2 = round_to_step(S0*1.05, calls["strike"].diff().dropna().mode().iloc[0])
    # c1 = calls.loc[calls["strike"]==K1].iloc[0]
    # c2 = calls.loc[calls["strike"]==K2].iloc[0]
    # legs = [
    #     {"asset":"call","side":"long","strike":float(K1),"premium":mid(c1["call_bid"],c1["call_ask"]),"qty":1.0},
    #     {"asset":"call","side":"short","strike":float(K2),"premium":mid(c2["call_bid"],c2["call_ask"]),"qty":1.0},
    # ]
    # return legs

    # ===== Example C: Covered Call (long stock + short OTM call) =====
    # Kc = float((calls[calls["strike"]>=S0].sort_values("strike").iloc[0]["strike"]))
    # c  = calls.loc[calls["strike"]==Kc].iloc[0]
    # legs = [
    #     {"asset":"stock","side":"long","strike":None,"premium":S0,"qty":1.0},  # stock premium is entry price
    #     {"asset":"call","side":"short","strike":float(Kc),"premium":mid(c["call_bid"],c["call_ask"]),"qty":1.0},
    # ]
    # return legs

    # ===== Example D: Protective Put (long stock + long put) =====
    # Kp = float((puts[puts["strike"]<=S0].sort_values("strike", ascending=False).iloc[0]["strike"]))
    # p  = puts.loc[puts["strike"]==Kp].iloc[0]
    # legs = [
    #     {"asset":"stock","side":"long","strike":None,"premium":S0,"qty":1.0},
    #     {"asset":"put","side":"long","strike":float(Kp),"premium":mid(p["put_bid"],p["put_ask"]),"qty":1.0},
    # ]
    # return legs

def leg_payoff(ST, leg, S0):
    K = leg["strike"]; prem = float(leg["premium"]); q = float(leg["qty"])
    if leg["asset"]=="stock":
        val = stock_long(ST, S0) if leg["side"]=="long" else stock_short(ST, S0)
    elif leg["asset"]=="call":
        if leg["side"]=="long":  val = call_payoff_long(ST, K, prem)
        else:                    val = call_payoff_short(ST, K, prem)
    elif leg["asset"]=="put":
        if leg["side"]=="long":  val = put_payoff_long(ST, K, prem)
        else:                    val = put_payoff_short(ST, K, prem)
    else:
        val = 0.0
    return q * val

def strategy_payoff_curve(S0, legs, st_lo_pct=0.6, st_hi_pct=1.4, steps=200):
    ST = np.linspace(S0*st_lo_pct, S0*st_hi_pct, steps)
    payoff = np.zeros_like(ST)
    for leg in legs:
        payoff += leg_payoff(ST, leg, S0)
    return ST, payoff

def mark_to_market_cost(legs):
    """
    Very rough current P&L estimate at entry using mids and a per-leg slippage cushion.
    For a long option you pay (mid + slippage); for a short you receive (mid - slippage).
    For stock we assume entry at spot with no extra cost here.
    Returned per-share cost; multiply by 100 for 1 contract.
    """
    total = 0.0
    for leg in legs:
        if leg["asset"]=="stock":  # assume enter at spot (no extra)
            continue
        prem = float(leg["premium"])
        if np.isnan(prem): continue
        if leg["side"]=="long":
            total += (prem + SLIPPAGE_PER_LEG) * leg["qty"]
        else:
            total -= (prem - SLIPPAGE_PER_LEG) * leg["qty"]
    return total

def pretty_legs(legs):
    out = []
    for L in legs:
        if L["asset"]=="stock":
            out.append(f"{L['side'].upper():5s} 1x STOCK")
        else:
            out.append(f"{L['side'].upper():5s} 1x {L['asset'].upper()}  K={L['strike']:.2f}  prem≈{float(L['premium']):.2f}")
    return "\n".join(out)

def main():
    S0, expiry, calls, puts = get_spot_and_chain(TICKER)
    legs = build_strategy(S0, calls, puts)

    # Chart payoff at expiry (per share)
    ST, payoff = strategy_payoff_curve(S0, legs, ST_RANGE_PCT[0], ST_RANGE_PCT[1], STEPS)

    # Quick current “cost” estimate at entry using mids + slippage
    entry_cost_ps = mark_to_market_cost(legs)  # per share
    entry_cost_per_contract = entry_cost_ps * 100.0

    print(f"\nTicker: {TICKER} | Spot S0 = {S0:.2f} | Expiry = {expiry}")
    print("Strategy legs (per 1 share / 1 option each):")
    print(pretty_legs(legs))
    print(f"\nApprox entry cash (per share) using mids ± slippage: {entry_cost_ps:.2f}")
    print(f"Approx entry cash (per 1 US option contract = 100 shares): {entry_cost_per_contract:.2f}")

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(ST, payoff, linewidth=2)
    plt.axvline(S0, linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.title(f"{TICKER} Strategy Payoff at Expiry ({expiry})")
    plt.xlabel("Underlying price at expiry  $S_T$")
    plt.ylabel("Payoff per share at expiry")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
