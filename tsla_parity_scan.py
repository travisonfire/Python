import math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

############################################
# User settings
############################################
TICKER = "TSLA"
RISK_FREE_ANNUAL = 0.05     # flat risk-free (replicate with your curve if you like)
DIVIDEND_YIELD = 0.00       # TSLA ~0, keep here for completeness
MAX_BIDASK_FRAC = 0.25      # discard options where (ask-bid)/mid > 25%
MIN_VOLUME = 10             # liquidity filter
MIN_OI = 50                 # open interest filter
DELTA_THRESHOLD = 0.05      # minimum abs(deviation) to care about (dollars)
EXTRA_BUFFER = 0.02         # extra buffer above txn cost before flagging (dollars)
STOCK_SPREAD_PCT = 0.0005   # est stock half-spread as % of S0 (e.g., 5 bps)
CSV_OUT = "tsla_parity_scan.csv"

############################################
# Helpers
############################################
def years_to_expiry(expiry_yyyymmdd: str) -> float:
    now = datetime.now(timezone.utc)
    expiry = datetime.strptime(expiry_yyyymmdd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = (expiry - now).total_seconds()/86400.0
    return max(days/365.0, 0.0)

def parity_deviation(S0, K, r, T, C_mid, P_mid, q=0.0):
    # Δ = (C - P) - (S e^{-qT} - K e^{-rT})
    return (C_mid - P_mid) - (S0*math.exp(-q*T) - K*math.exp(-r*T))

def implied_txn_cost(call_bid, call_ask, put_bid, put_ask, S0):
    # Approx half-spread cost if you lift/ hit both sides once
    call_half = (call_ask - call_bid) / 2.0
    put_half  = (put_ask  - put_bid)  / 2.0
    stock_half = STOCK_SPREAD_PCT * S0
    return max(call_half, 0) + max(put_half, 0) + stock_half

def classify(delta, cost, extra=EXTRA_BUFFER):
    # Only flag if |Δ| exceeds the *economic* hurdle: txn cost + buffer
    hurdle = cost + extra
    if delta > hurdle:
        return "EXPENSIVE (sell C, buy P, buy stock, borrow)"
    elif delta < -hurdle:
        return "CHEAP (buy C, sell P, short stock, invest)"
    else:
        return "FAIR"

############################################
# Data
############################################
tk = yf.Ticker(TICKER)
S0 = tk.history(period="1d")["Close"].iloc[-1]
expiries = tk.options

rows = []
for expiry in expiries:
    T = years_to_expiry(expiry)
    if T <= 0: 
        continue

    oc = tk.option_chain(expiry)
    calls = oc.calls.rename(columns={
        "bid":"call_bid", "ask":"call_ask", "lastPrice":"call_last", "volume":"call_vol", "openInterest":"call_oi"
    })
    puts  = oc.puts.rename(columns={
        "bid":"put_bid", "ask":"put_ask", "lastPrice":"put_last", "volume":"put_vol", "openInterest":"put_oi"
    })

    df = pd.merge(
        calls[["contractSymbol","strike","call_bid","call_ask","call_last","call_vol","call_oi"]],
        puts[["strike","put_bid","put_ask","put_last","put_vol","put_oi"]],
        on="strike",
        how="inner"
    )
    if df.empty:
        continue

    # mids
    df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
    df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0

    # filters
    # drop NaNs, negative, zero mids
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["call_bid","call_ask","put_bid","put_ask","call_mid","put_mid"])
    df = df[(df["call_mid"] > 0) & (df["put_mid"] > 0)]
    # spread / mid filter
    df["call_spread_frac"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"].replace(0, np.nan)
    df["put_spread_frac"]  = (df["put_ask"]  - df["put_bid"])  / df["put_mid"].replace(0, np.nan)
    df = df[(df["call_spread_frac"] <= MAX_BIDASK_FRAC) & (df["put_spread_frac"] <= MAX_BIDASK_FRAC)]
    # liquidity
    df = df[(df["call_vol"] >= MIN_VOLUME) & (df["put_vol"] >= MIN_VOLUME) & (df["call_oi"] >= MIN_OI) & (df["put_oi"] >= MIN_OI)]

    if df.empty:
        continue

    # compute Δ and txn costs
    df["T"] = T
    df["r"] = RISK_FREE_ANNUAL
    df["q"] = DIVIDEND_YIELD
    df["S0"] = S0
    df["delta"] = df.apply(lambda x: parity_deviation(x.S0, x.strike, x.r, x.T, x.call_mid, x.put_mid, x.q), axis=1)
    df["txn_cost_est"] = df.apply(lambda x: implied_txn_cost(x.call_bid, x.call_ask, x.put_bid, x.put_ask, S0), axis=1)
    df["signal"] = df.apply(lambda x: classify(x.delta, x.txn_cost_est, EXTRA_BUFFER), axis=1)
    df["abs_econ_gap"] = (df["delta"].abs() - (df["txn_cost_est"] + EXTRA_BUFFER))

    df["expiry"] = expiry
    rows.append(df)

full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if full.empty:
    print("No qualifying options after filters (liquidity/spread). Try loosening thresholds.")
else:
    full = full.sort_values(["abs_econ_gap"], ascending=False)
    cols = ["expiry","strike","S0","r","T","call_bid","call_ask","put_bid","put_ask","call_mid","put_mid",
            "delta","txn_cost_est","signal","abs_econ_gap","call_vol","put_vol","call_oi","put_oi"]
    full[cols].to_csv(CSV_OUT, index=False)
    print(f"TSLA spot S0 = {S0:.4f}")
    print(f"Scanned {full['expiry'].nunique()} expiries, {len(full)} strike pairs after filters.")
    print(f"Saved to {CSV_OUT}\n")
    print("Top 15 by *economic* gap (|Δ| minus txn cost + buffer):\n")
    print(full[cols].head(15).to_string(index=False))
