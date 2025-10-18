# options_strategy_lab.py
# Run:  pip install yfinance pandas numpy matplotlib
#       python options_strategy_lab.py

import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ======== USER CHOICES ========
TICKER = "TSLA"
USE_NEAREST_EXPIRY = True     # if False, set EXPIRY manually
EXPIRY = None                 # e.g. "2025-12-19"
STRATEGY = "bull_call_spread" # choose from STRATEGIES below
QTY_PER_LEG = 1.0             # per-share basis (×100 = per contract)
WIDTH_PCT = 0.05              # for spreads/condors/butterflies: wing spacing as % of spot
OTM_PCT  = 0.05               # for strangles/covered calls: OTM distance as % of spot
SLIPPAGE_PER_LEG = 0.02       # $/share cost cushion when estimating entry cash
ST_RANGE_PCT = (0.6, 1.4)     # payoff x-axis range as % of spot
STEPS = 300                   # points on payoff curve
# ==============================

STRATEGIES = {
    # Bullish
    "long_call": "Buy 1 call",
    "short_put": "Sell 1 put",
    "bull_call_spread": "Buy call, Sell higher-strike call",
    "bull_put_spread": "Sell put, Buy lower-strike put",

    # Bearish
    "long_put": "Buy 1 put",
    "short_call": "Sell 1 call",
    "bear_put_spread": "Buy put, Sell lower-strike put",
    "bear_call_spread": "Sell call, Buy higher-strike call",

    # Neutral
    "covered_call": "Long stock, Sell call",
    "short_straddle": "Sell call & put (ATM)",
    "short_strangle": "Sell OTM call & put",
    "iron_condor": "Bull put + Bear call (4 legs)",
    "iron_butterfly": "ATM short straddle + wings (4 legs)",

    # Volatile
    "long_straddle": "Buy call & put (ATM)",
    "long_strangle": "Buy OTM call & put",
}

def mid(b, a):
    if b is None or a is None: return np.nan
    return (float(b) + float(a)) / 2.0

def get_spot_chain(ticker, expiry_pref=None, use_nearest=True):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty: raise RuntimeError("No price history returned.")
    S0 = float(hist["Close"].iloc[-1])
    exps = tk.options
    if not exps: raise RuntimeError("No options found for this ticker.")
    expiry = exps[0] if use_nearest else (expiry_pref or exps[0])
    oc = tk.option_chain(expiry)
    calls = oc.calls.rename(columns={"bid":"call_bid","ask":"call_ask","lastPrice":"call_last"})
    puts  = oc.puts .rename(columns={"bid":"put_bid","ask":"put_ask","lastPrice":"put_last"})
    return S0, expiry, calls, puts

def nearest_strike(df, target):
    # pick strike nearest to target
    i = (df["strike"] - target).abs().argsort().iloc[0]
    return float(df.iloc[i]["strike"])

def pick_otm_strike(df, S0, pct, side="call"):
    # for calls: strike >= S0*(1+pct) ; for puts: strike <= S0*(1-pct)
    if side == "call":
        c = df[df["strike"] >= S0*(1+pct)].sort_values("strike")
        return float(c.iloc[0]["strike"]) if not c.empty else nearest_strike(df, S0*(1+pct))
    else:
        p = df[df["strike"] <= S0*(1-pct)].sort_values("strike", ascending=True)
        return float(p.iloc[-1]["strike"]) if not p.empty else nearest_strike(df, S0*(1-pct))

# ---- Payoff primitives (per share) ----
def call_long(ST, K, prem):   return np.maximum(ST-K,0.0) - prem
def call_short(ST, K, prem):  return -call_long(ST, K, prem)
def put_long(ST, K, prem):    return np.maximum(K-ST,0.0) - prem
def put_short(ST, K, prem):   return -put_long(ST, K, prem)
def stock_long(ST, S0):       return ST - S0
def stock_short(ST, S0):      return -stock_long(ST, S0)

def leg_value(ST, leg, S0):
    K = leg.get("strike")
    prem = float(leg["premium"]) if leg["premium"] is not None else 0.0
    q = float(leg["qty"])
    t = leg["type"]; side = leg["side"]
    if t=="stock":
        v = stock_long(ST,S0) if side=="long" else stock_short(ST,S0)
    elif t=="call":
        v = call_long(ST,K,prem) if side=="long" else call_short(ST,K,prem)
    elif t=="put":
        v = put_long(ST,K,prem) if side=="long" else put_short(ST,K,prem)
    else:
        v = 0.0
    return q*v

def payoff_curve(S0, legs, lo=0.6, hi=1.4, steps=300):
    ST = np.linspace(S0*lo, S0*hi, steps)
    pay = np.zeros_like(ST)
    for L in legs:
        pay += leg_value(ST, L, S0)
    return ST, pay

def entry_cash_estimate(legs, slippage=0.02):
    # crude “cash at entry” using mids ± slippage; per share; + for outflow, - for inflow
    cash = 0.0
    for L in legs:
        if L["type"]=="stock":  # assume enter at S0, ignore here
            continue
        prem = float(L["premium"]) if L["premium"] is not None else 0.0
        if np.isnan(prem): continue
        if L["side"]=="long":  cash += (prem + slippage) * L["qty"]
        else:                  cash -= (prem - slippage) * L["qty"]
    return cash

def build_legs(strategy, S0, calls, puts, width_pct=0.05, otm_pct=0.05, qty=1.0):
    # pre-compute helpful strikes
    K_atm = nearest_strike(calls, S0)
    K_call_otm = pick_otm_strike(calls, S0, otm_pct, side="call")
    K_put_otm  = pick_otm_strike(puts,  S0, otm_pct, side="put")
    width = max(S0*width_pct, 0.01)

    def mid_call(K):
        row = calls[calls["strike"]==K]
        return mid(row["call_bid"].iloc[0], row["call_ask"].iloc[0]) if not row.empty else np.nan
    def mid_put(K):
        row = puts[puts["strike"]==K]
        return mid(row["put_bid"].iloc[0], row["put_ask"].iloc[0]) if not row.empty else np.nan

    legs = []

    if strategy == "long_call":
        pc = mid_call(K_atm)
        legs = [{"type":"call","side":"long","strike":K_atm,"premium":pc,"qty":qty}]

    elif strategy == "short_put":
        pp = mid_put(K_atm)
        legs = [{"type":"put","side":"short","strike":K_atm,"premium":pp,"qty":qty}]

    elif strategy == "bull_call_spread":
        K1 = nearest_strike(calls, S0)              # buy near ATM
        K2 = nearest_strike(calls, S0 + width)      # sell higher
        legs = [
            {"type":"call","side":"long","strike":K1,"premium":mid_call(K1),"qty":qty},
            {"type":"call","side":"short","strike":K2,"premium":mid_call(K2),"qty":qty},
        ]

    elif strategy == "bull_put_spread":
        K1 = nearest_strike(puts, S0)               # sell near ATM
        K2 = nearest_strike(puts, S0 - width)       # buy lower
        legs = [
            {"type":"put","side":"short","strike":K1,"premium":mid_put(K1),"qty":qty},
            {"type":"put","side":"long","strike":K2,"premium":mid_put(K2),"qty":qty},
        ]

    elif strategy == "long_put":
        pp = mid_put(K_atm)
        legs = [{"type":"put","side":"long","strike":K_atm,"premium":pp,"qty":qty}]

    elif strategy == "short_call":
        pc = mid_call(K_atm)
        legs = [{"type":"call","side":"short","strike":K_atm,"premium":pc,"qty":qty}]

    elif strategy == "bear_put_spread":
        K1 = nearest_strike(puts, S0)               # buy near ATM
        K2 = nearest_strike(puts, S0 - width)       # sell lower
        legs = [
            {"type":"put","side":"long","strike":K1,"premium":mid_put(K1),"qty":qty},
            {"type":"put","side":"short","strike":K2,"premium":mid_put(K2),"qty":qty},
        ]

    elif strategy == "bear_call_spread":
        K1 = nearest_strike(calls, S0)              # sell near ATM
        K2 = nearest_strike(calls, S0 + width)      # buy higher
        legs = [
            {"type":"call","side":"short","strike":K1,"premium":mid_call(K1),"qty":qty},
            {"type":"call","side":"long","strike":K2,"premium":mid_call(K2),"qty":qty},
        ]

    elif strategy == "covered_call":
        Kc = pick_otm_strike(calls, S0, otm_pct, side="call")
        legs = [
            {"type":"stock","side":"long","strike":None,"premium":S0,"qty":qty},
            {"type":"call","side":"short","strike":Kc,"premium":mid_call(Kc),"qty":qty},
        ]

    elif strategy == "short_straddle":
        legs = [
            {"type":"call","side":"short","strike":K_atm,"premium":mid_call(K_atm),"qty":qty},
            {"type":"put","side":"short","strike":K_atm,"premium":mid_put(K_atm),"qty":qty},
        ]

    elif strategy == "short_strangle":
        legs = [
            {"type":"call","side":"short","strike":K_call_otm,"premium":mid_call(K_call_otm),"qty":qty},
            {"type":"put","side":"short","strike":K_put_otm, "premium":mid_put(K_put_otm), "qty":qty},
        ]

    elif strategy == "iron_condor":
        # Sell put spread (Kp_short, Kp_long) + Sell call spread (Kc_short, Kc_long)
        Kp_short = pick_otm_strike(puts,  S0, otm_pct, side="put")
        Kp_long  = nearest_strike(puts,  Kp_short - width)
        Kc_short = pick_otm_strike(calls, S0, otm_pct, side="call")
        Kc_long  = nearest_strike(calls, Kc_short + width)
        legs = [
            {"type":"put","side":"short","strike":Kp_short,"premium":mid_put(Kp_short),"qty":qty},
            {"type":"put","side":"long","strike":Kp_long, "premium":mid_put(Kp_long), "qty":qty},
            {"type":"call","side":"short","strike":Kc_short,"premium":mid_call(Kc_short),"qty":qty},
            {"type":"call","side":"long","strike":Kc_long, "premium":mid_call(Kc_long), "qty":qty},
        ]

    elif strategy == "iron_butterfly":
        # Short ATM straddle + long wings
        Ku = nearest_strike(calls, S0 + width)
        Kd = nearest_strike(puts,  S0 - width)
        legs = [
            {"type":"call","side":"short","strike":K_atm,"premium":mid_call(K_atm),"qty":qty},
            {"type":"put", "side":"short","strike":K_atm,"premium":mid_put(K_atm),"qty":qty},
            {"type":"call","side":"long","strike":Ku,   "premium":mid_call(Ku),   "qty":qty},
            {"type":"put", "side":"long","strike":Kd,   "premium":mid_put(Kd),    "qty":qty},
        ]

    elif strategy == "long_straddle":
        legs = [
            {"type":"call","side":"long","strike":K_atm,"premium":mid_call(K_atm),"qty":qty},
            {"type":"put", "side":"long","strike":K_atm,"premium":mid_put(K_atm), "qty":qty},
        ]

    elif strategy == "long_strangle":
        legs = [
            {"type":"call","side":"long","strike":K_call_otm,"premium":mid_call(K_call_otm),"qty":qty},
            {"type":"put", "side":"long","strike":K_put_otm, "premium":mid_put(K_put_otm), "qty":qty},
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # fill NaNs with 0 premiums to avoid math errors (but warn)
    for L in legs:
        if L["type"] != "stock" and (L["premium"] is None or np.isnan(L["premium"])):
            print(f"Warning: missing quote for {L}; setting premium=0.0")
            L["premium"] = 0.0
    return legs

def pretty(legs):
    out=[]
    for L in legs:
        if L["type"]=="stock":
            out.append(f"{L['side'].upper():5s} STOCK x{L['qty']:.0f}")
        else:
            out.append(f"{L['side'].upper():5s} {L['type'].upper():4s} K={L['strike']:.2f} prem≈{float(L['premium']):.2f} x{L['qty']:.0f}")
    return "\n".join(out)

def main():
    print(f"Strategy: {STRATEGY} — {STRATEGIES.get(STRATEGY,'')}")
    S0, expiry, calls, puts = get_spot_chain(TICKER, EXPIRY, USE_NEAREST_EXPIRY)
    legs = build_legs(STRATEGY, S0, calls, puts, width_pct=WIDTH_PCT, otm_pct=OTM_PCT, qty=QTY_PER_LEG)

    entry_ps = entry_cash_estimate(legs, SLIPPAGE_PER_LEG)  # per share
    print(f"\n{TICKER} spot S0 = {S0:.2f} | Expiry = {expiry}")
    print("Legs (per share basis):")
    print(pretty(legs))
    print(f"\nApprox entry cash (per share): {entry_ps:.2f}")
    print(f"Approx entry cash (per 1 US option contract = 100 shares): {entry_ps*100:.2f}")

    ST, payoff = payoff_curve(S0, legs, ST_RANGE_PCT[0], ST_RANGE_PCT[1], STEPS)
    plt.figure(figsize=(10,6))
    plt.plot(ST, payoff, linewidth=2)
    plt.axvline(S0, linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.title(f"{TICKER} — {STRATEGY.replace('_',' ').title()} — Expiry {expiry}")
    plt.xlabel("Underlying price at expiry  $S_T$")
    plt.ylabel("Payoff per share at expiry")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
