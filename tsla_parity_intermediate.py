import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------- Settings (edit these) -----------------
TICKER = "TSLA"
RISK_FREE = 0.05      # flat annual risk-free rate for discounting
DIV_YIELD = 0.00      # TSLA ~0
MAX_SPREAD_FRAC = 0.35  # drop quotes with (ask-bid)/mid > 35%
MIN_OI = 10             # require some open interest
MIN_VOL = 1             # require some volume
TXN_BUFFER = 0.05       # extra $ buffer beyond estimated half-spreads
CSV_OUT = "tsla_parity_intermediate.csv"
N_EXPIRIES = 5          # scan the first N expiries
TOP_N = 20              # show top N signals
# ---------------------------------------------------------

def years_to_expiry(exp_str: str) -> float:
    now = datetime.now(timezone.utc)
    t = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = (t - now).total_seconds() / 86400.0
    return max(days / 365.0, 1/365)  # avoid 0

def main():
    tk = yf.Ticker(TICKER)
    S0 = float(tk.history(period="1d")["Close"].iloc[-1])
    expiries = tk.options[:N_EXPIRIES]  # first few expiries

    all_rows = []
    for exp in expiries:
        T = years_to_expiry(exp)
        oc = tk.option_chain(exp)

        calls = oc.calls.rename(columns={
            "bid":"call_bid","ask":"call_ask","lastPrice":"call_last",
            "openInterest":"call_oi","volume":"call_vol"
        })
        puts = oc.puts.rename(columns={
            "bid":"put_bid","ask":"put_ask","lastPrice":"put_last",
            "openInterest":"put_oi","volume":"put_vol"
        })

        df = pd.merge(
            calls[["strike","call_bid","call_ask","call_last","call_oi","call_vol"]],
            puts[ ["strike","put_bid","put_ask","put_last","put_oi","put_vol"]],
            on="strike", how="inner"
        )

        if df.empty:
            continue

        # Coerce numerics and drop bad rows
        num_cols = ["strike","call_bid","call_ask","put_bid","put_ask","call_last","put_last",
                    "call_oi","put_oi","call_vol","put_vol"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["call_bid","call_ask","put_bid","put_ask"])

        # Mid prices
        df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
        df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0
        df = df[(df["call_mid"] > 0) & (df["put_mid"] > 0)].copy()

        # Basic quality filters
        df["call_spread_frac"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"]
        df["put_spread_frac"]  = (df["put_ask"]  - df["put_bid"])  / df["put_mid"]
        df = df[(df["call_spread_frac"] <= MAX_SPREAD_FRAC) & (df["put_spread_frac"] <= MAX_SPREAD_FRAC)]
        df = df[(df["call_oi"] >= MIN_OI) & (df["put_oi"] >= MIN_OI)
                & (df["call_vol"] >= MIN_VOL) & (df["put_vol"] >= MIN_VOL)]
        if df.empty:
            continue

        # Vectorized parity deviation Δ = (C-P) - (S e^{-qT} - K e^{-rT})
        S_term = S0 * np.exp(-DIV_YIELD * T)
        K_term = df["strike"].values * np.exp(-RISK_FREE * T)
        delta = (df["call_mid"].values - df["put_mid"].values) - (S_term - K_term)

        # Rough transaction-cost hurdle: half-spreads on options
        call_half = (df["call_ask"].values - df["call_bid"].values) / 2.0
        put_half  = (df["put_ask"].values  - df["put_bid"].values)  / 2.0
        hurdle = np.maximum(call_half, 0) + np.maximum(put_half, 0) + TXN_BUFFER

        df["expiry"] = exp
        df["S0"] = S0
        df["T"] = T
        df["delta"] = delta
        df["hurdle"] = hurdle
        df["econ_gap"] = np.abs(df["delta"]) - df["hurdle"]
        df["signal"] = np.where(df["delta"] > df["hurdle"], "EXPENSIVE",
                          np.where(df["delta"] < -df["hurdle"], "CHEAP", "FAIR"))
        all_rows.append(df)

    if not all_rows:
        print("No rows after filters. Try increasing MAX_SPREAD_FRAC or lowering MIN_OI/MIN_VOL.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    full = full.sort_values("econ_gap", ascending=False)

    cols = ["expiry","strike","S0","T","call_bid","call_ask","put_bid","put_ask",
            "call_mid","put_mid","delta","hurdle","econ_gap","signal",
            "call_vol","put_vol","call_oi","put_oi"]
    full[cols].to_csv(CSV_OUT, index=False)

    print(f"TSLA spot S0 = {S0:.4f}")
    print(f"Scanned {full['expiry'].nunique()} expiries, {len(full)} strike pairs after filters.")
    print(f"Saved results to {CSV_OUT}\n")
    print("Top signals (after transaction-cost hurdle):\n")
    print(full[cols].head(TOP_N).to_string(index=False))

if __name__ == "__main__":
    main()
