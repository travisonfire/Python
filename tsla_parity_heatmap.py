import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------- SETTINGS ----------------
TICKER = "TSLA"
RISK_FREE = 0.05          # annual rate
DIV_YIELD = 0.00          # TSLA ≈ 0 dividend
FIRST_N_EXPIRIES = 8
MAX_SPREAD_FRAC = 0.35
MIN_OI = 25
MIN_VOL = 5
TXN_BUFFER = 0.03
CSV_OUT = "tsla_parity_heatmap.csv"
# ------------------------------------------

def years_to_expiry(exp_str: str) -> float:
    now = datetime.now(timezone.utc)
    t = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = max((t - now).total_seconds() / 86400.0, 0.0)
    return max(days / 365.0, 1 / 365)

def main():
    tk = yf.Ticker(TICKER)
    S0 = float(tk.history(period="1d")["Close"].iloc[-1])
    expiries = tk.options[:FIRST_N_EXPIRIES]

    all_rows = []
    for exp in expiries:
        T = years_to_expiry(exp)
        oc = tk.option_chain(exp)
        calls = oc.calls.rename(columns={"bid":"call_bid","ask":"call_ask","openInterest":"call_oi","volume":"call_vol"})
        puts  = oc.puts.rename(columns={"bid":"put_bid","ask":"put_ask","openInterest":"put_oi","volume":"put_vol"})
        df = pd.merge(calls[["strike","call_bid","call_ask","call_oi","call_vol"]],
                      puts [["strike","put_bid","put_ask","put_oi","put_vol"]],
                      on="strike", how="inner")
        if df.empty:
            continue

        for c in ["strike","call_bid","call_ask","put_bid","put_ask","call_oi","put_oi","call_vol","put_vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["call_bid","call_ask","put_bid","put_ask"], inplace=True)
        df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
        df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0
        df = df[(df["call_mid"] > 0) & (df["put_mid"] > 0)]

        df["call_spread_frac"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"]
        df["put_spread_frac"]  = (df["put_ask"]  - df["put_bid"])  / df["put_mid"]
        df = df[(df["call_spread_frac"] <= MAX_SPREAD_FRAC) & (df["put_spread_frac"] <= MAX_SPREAD_FRAC)]
        df = df[(df["call_oi"] >= MIN_OI) & (df["put_oi"] >= MIN_OI) &
                (df["call_vol"] >= MIN_VOL) & (df["put_vol"] >= MIN_VOL)]
        if df.empty:
            continue

        S_term = S0 * np.exp(-DIV_YIELD * T)
        K_term = df["strike"].values * np.exp(-RISK_FREE * T)
        delta = (df["call_mid"].values - df["put_mid"].values) - (S_term - K_term)

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
        print("No data after filters.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    full.sort_values("econ_gap", ascending=False, inplace=True)
    full.to_csv(CSV_OUT, index=False)

    print(f"\n{TICKER} spot = {S0:.2f}")
    print(f"Saved to {CSV_OUT}")
    print("Top 10 signals:\n", full[["expiry","strike","delta","hurdle","econ_gap","signal"]].head(10))

    # ---------- Heatmap ----------
    df = full.copy()
    df["exp_short"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")
    pivot = df.pivot_table(index="exp_short", columns="strike", values="delta", aggfunc="mean")

    if not pivot.empty:
        plt.figure(figsize=(12, 6))
        plt.imshow(pivot.values, aspect="auto")
        plt.title(f"{TICKER} Put–Call Parity Δ Heatmap")
        plt.xlabel("Strike")
        plt.ylabel("Expiry")
        plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=90)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Δ (C−P−(S−PV(K)))")
        plt.tight_layout()
        plt.show()
    else:
        print("Heatmap skipped: pivot empty.")

if __name__ == "__main__":
    main()
