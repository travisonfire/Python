import math, time, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ==================== SETTINGS ====================
TICKERS = ["AMD", "NVDA", "TSM"]  # <- add/remove tickers
FIRST_N_EXPIRIES = 6                # scan first N expiries per ticker
RISK_FREE = 0.05                    # flat annual r for discounting
DIV_YIELD = 0.00                    # use 0 for most non-div payers; set per ticker if you want
MAX_SPREAD_FRAC = 0.35              # drop rows where (ask-bid)/mid > this
MIN_OI = 25                         # min open interest
MIN_VOL = 5                         # min volume
STOCK_HALF_SPREAD_BPS = 1.0         # est stock half-spread (bps of spot)
TXN_BUFFER = 0.03                   # extra $ cushion over half-spreads
REFRESH_SECONDS = 120               # how often to rescan (e.g., 120s)
OUT_CSV = "parity_multi_scan.csv"
TOP_N_PER_TICKER = 8                # print top N per ticker each cycle
# ===================================================

def years_to_expiry(exp_str: str) -> float:
    now = datetime.now(timezone.utc)
    t = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = max((t - now).total_seconds() / 86400.0, 0.0)
    return max(days / 365.0, 1/365)  # avoid zero

def scan_ticker(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        return pd.DataFrame()
    S0 = float(hist["Close"].iloc[-1])

    expiries = tk.options[:FIRST_N_EXPIRIES] if tk.options else []
    if not expiries:
        return pd.DataFrame()

    all_rows = []
    for exp in expiries:
        T = years_to_expiry(exp)
        oc = tk.option_chain(exp)
        calls = oc.calls.rename(columns={
            "bid":"call_bid","ask":"call_ask","lastPrice":"call_last",
            "openInterest":"call_oi","volume":"call_vol"
        })
        puts  = oc.puts.rename(columns={
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
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["call_bid","call_ask","put_bid","put_ask"])

        # Mid prices
        df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
        df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0
        df = df[(df["call_mid"] > 0) & (df["put_mid"] > 0)].copy()

        # Quality filters
        df["call_spread_frac"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"]
        df["put_spread_frac"]  = (df["put_ask"]  - df["put_bid"])  / df["put_mid"]
        df = df[(df["call_spread_frac"] <= MAX_SPREAD_FRAC) & (df["put_spread_frac"] <= MAX_SPREAD_FRAC)]
        df = df[(df["call_oi"] >= MIN_OI) & (df["put_oi"] >= MIN_OI) &
                (df["call_vol"] >= MIN_VOL) & (df["put_vol"] >= MIN_VOL)]
        if df.empty:
            continue

        # Vectorized parity delta: Δ = (C-P) - (S e^{-qT} - K e^{-rT})
        S_term = S0 * np.exp(-DIV_YIELD * T)
        K_term = df["strike"].values * np.exp(-RISK_FREE * T)
        delta  = (df["call_mid"].values - df["put_mid"].values) - (S_term - K_term)

        # Transaction-cost hurdle: option half-spreads + stock half-spread + buffer
        call_half = (df["call_ask"].values - df["call_bid"].values) / 2.0
        put_half  = (df["put_ask"].values  - df["put_bid"].values)  / 2.0
        stock_half= (STOCK_HALF_SPREAD_BPS/1e4) * S0
        hurdle = np.maximum(call_half, 0) + np.maximum(put_half, 0) + stock_half + TXN_BUFFER

        out = df.copy()
        out["ticker"] = ticker
        out["expiry"] = exp
        out["S0"]     = S0
        out["T"]      = T
        out["r"]      = RISK_FREE
        out["q"]      = DIV_YIELD
        out["delta"]  = delta
        out["hurdle"] = hurdle
        out["econ_gap"] = np.abs(delta) - hurdle
        out["signal"] = np.where(out["delta"] > out["hurdle"], "EXPENSIVE",
                          np.where(out["delta"] < -out["hurdle"], "CHEAP", "FAIR"))
        all_rows.append(out)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

def print_top(full: pd.DataFrame):
    if full.empty:
        print("No rows after filters.")
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== Parity scan @ {ts} =====")
    for ticker in full["ticker"].unique():
        sub = full[full["ticker"] == ticker].sort_values("econ_gap", ascending=False)
        cols = ["ticker","expiry","strike","S0","delta","hurdle","econ_gap","signal",
                "call_bid","call_ask","put_bid","put_ask","call_oi","put_oi","call_vol","put_vol"]
        print(f"\n--- {ticker}: Top {min(TOP_N_PER_TICKER, len(sub))} ---")
        print(sub[cols].head(TOP_N_PER_TICKER).to_string(index=False))

def main_loop():
    print("Starting multi-ticker parity scanner… (Ctrl+C to stop)")
    print(f"Tickers: {', '.join(TICKERS)} | Refresh: {REFRESH_SECONDS}s")
    while True:
        try:
            frames = []
            for t in TICKERS:
                try:
                    df = scan_ticker(t)
                    frames.append(df)
                except Exception as e:
                    print(f"[{t}] scan error: {e}")
            full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not full.empty:
                full.sort_values(["ticker","econ_gap"], ascending=[True, False], inplace=True)
                full.to_csv(OUT_CSV, index=False)
            os.system('cls' if os.name == 'nt' else 'clear')
            print_top(full)
            print(f"\nSaved latest snapshot to {OUT_CSV}")
            time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("\nStopping scanner. Bye!")
            break

if __name__ == "__main__":
    main_loop()
