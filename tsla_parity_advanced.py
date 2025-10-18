import math
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------- Utils -----------------------------

def years_to_expiry(exp_str: str) -> float:
    now = datetime.now(timezone.utc)
    t = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = max((t - now).total_seconds() / 86400.0, 0.0)
    return max(days / 365.0, 1/365)  # avoid zero

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def quick_yield_curve_flat(default_r=0.05):
    """Fallback: single flat rate."""
    return {"flat": default_r}

def quick_yield_curve_treasury():
    """
    Very rough curve from Yahoo Treasury indices (may differ from true curve):
      ^IRX ~ 13-week, ^FVX ~ 5y, ^TNX ~ 10y (yields in %)
    We'll logarithmically interpolate on maturity.
    """
    try:
        tickers = yf.download(["^IRX","^FVX","^TNX"], period="5d", interval="1d", progress=False)["Close"].ffill().iloc[-1]
        y_3m = float(tickers["^IRX"]) / 100.0
        y_5y = float(tickers["^FVX"]) / 100.0
        y_10y= float(tickers["^TNX"]) / 100.0
        # store as simple (T, r) points in years
        return {"points": [(0.25, y_3m), (5.0, y_5y), (10.0, y_10y)]}
    except Exception:
        return quick_yield_curve_flat()

def interp_rate(T, curve):
    """Return a crude annual cont. rate for horizon T (years), from curve dict."""
    if "flat" in curve:
        return float(curve["flat"])
    pts = sorted(curve["points"], key=lambda x: x[0])
    Ts, Rs = zip(*pts)
    T = max(min(T, Ts[-1]), Ts[0])
    # log-linear in maturity (simple)
    return float(np.interp(T, Ts, Rs))

def parity_delta(S0, K, r, T, C_mid, P_mid, q_eff=0.0):
    # Δ = (C - P) - (S0 * e^{-q_eff*T} - K * e^{-r*T})
    return (C_mid - P_mid) - (S0 * math.exp(-q_eff * T) - K * math.exp(-r * T))

def est_txn_cost(call_bid, call_ask, put_bid, put_ask, S0, stock_half_spread_bps):
    call_half = max((call_ask - call_bid) / 2.0, 0.0)
    put_half  = max((put_ask  - put_bid ) / 2.0, 0.0)
    stock_half= (stock_half_spread_bps / 1e4) * S0
    return call_half + put_half + stock_half

# ----------------------------- Main -----------------------------

def run(args):
    tk = yf.Ticker(args.ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError("No spot history returned. Check ticker or internet.")
    S0 = float(hist["Close"].iloc[-1])

    # expiries
    expiries = tk.options
    if not expiries:
        raise RuntimeError("No option expiries returned for this ticker.")
    if args.first_n_expiries > 0:
        expiries = expiries[:args.first_n_expiries]

    # carry: q_eff = q - borrow
    q_eff_input = args.div_yield - args.borrow_yield

    # yield curve
    if args.curve == "treasury":
        curve = quick_yield_curve_treasury()
    else:
        curve = quick_yield_curve_flat(args.risk_free)

    rows = []
    for exp in expiries:
        T = years_to_expiry(exp)
        r_T = interp_rate(T, curve)
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
            calls[["contractSymbol","strike","call_bid","call_ask","call_last","call_oi","call_vol"]],
            puts[ ["strike","put_bid","put_ask","put_last","put_oi","put_vol"]],
            on="strike", how="inner"
        )
        if df.empty:
            continue

        # Coerce types
        num_cols = ["strike","call_bid","call_ask","put_bid","put_ask","call_last","put_last","call_oi","put_oi","call_vol","put_vol"]
        df = coerce_numeric(df, num_cols)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["call_bid","call_ask","put_bid","put_ask"])

        # Mids & basic filters
        df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
        df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0
        df = df[(df["call_mid"] > 0) & (df["put_mid"] > 0)].copy()

        # Spread quality & liquidity
        df["call_spread_frac"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"]
        df["put_spread_frac"]  = (df["put_ask"]  - df["put_bid"])  / df["put_mid"]
        df = df[(df["call_spread_frac"] <= args.max_spread_frac) & (df["put_spread_frac"] <= args.max_spread_frac)]
        df = df[(df["call_oi"] >= args.min_oi) & (df["put_oi"] >= args.min_oi) &
                (df["call_vol"] >= args.min_vol) & (df["put_vol"] >= args.min_vol)]
        if df.empty:
            continue

        # Vectorized Δ & costs
        S_term = S0 * np.exp(-q_eff_input * T)
        K_term = df["strike"].values * np.exp(-r_T * T)
        delta = (df["call_mid"].values - df["put_mid"].values) - (S_term - K_term)

        call_half = (df["call_ask"].values - df["call_bid"].values) / 2.0
        put_half  = (df["put_ask"].values  - df["put_bid"].values)  / 2.0
        stock_half= (args.stock_half_spread_bps / 1e4) * S0
        hurdle = np.maximum(call_half, 0) + np.maximum(put_half, 0) + stock_half + args.extra_buffer

        # Signals
        signal = np.where(delta > hurdle, "EXPENSIVE (sell C, buy P, buy stock, borrow)",
                   np.where(delta < -hurdle, "CHEAP (buy C, sell P, short stock, invest)", "FAIR"))

        out = df.copy()
        out["expiry"] = exp
        out["S0"] = S0
        out["T"] = T
        out["r_T"] = r_T
        out["q_eff"] = q_eff_input
        out["delta"] = delta
        out["hurdle"] = hurdle
        out["econ_gap"] = np.abs(delta) - hurdle
        out["signal"] = signal
        rows.append(out)

    if not rows:
        print("No rows after filters. Try loosening max spread / liquidity thresholds.")
        return

    full = pd.concat(rows, ignore_index=True)
    full.sort_values("econ_gap", ascending=False, inplace=True)

    cols = ["expiry","strike","S0","T","r_T","q_eff",
            "call_bid","call_ask","put_bid","put_ask",
            "call_mid","put_mid","call_vol","put_vol","call_oi","put_oi",
            "call_spread_frac","put_spread_frac",
            "delta","hurdle","econ_gap","signal"]
    full[cols].to_csv(args.out_csv, index=False)

    print(f"{args.ticker} spot S0 = {S0:.4f}")
    print(f"Scanned {full['expiry'].nunique()} expiries, {len(full)} strike pairs after filters.")
    print(f"Saved results to {args.out_csv}\n")
    print("Top signals (economic gap = |Δ| - hurdle):\n")
    print(full[cols].head(args.top_n).to_string(index=False))

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Advanced Put–Call Parity Scanner")
    p.add_argument("--ticker", default="TSLA", help="Equity ticker (default TSLA)")
    p.add_argument("--first-n-expiries", type=int, default=8, help="Scan first N expiries (0 = all)")
    p.add_argument("--risk-free", type=float, default=0.05, help="Flat annual risk-free rate if not using treasury curve")
    p.add_argument("--curve", choices=["flat","treasury"], default="flat", help="Discount curve mode")
    p.add_argument("--div-yield", type=float, default=0.00, help="Continuous dividend yield q (dec)")
    p.add_argument("--borrow-yield", type=float, default=0.00, help="Stock borrow cost b (dec); q_eff = q - b")
    p.add_argument("--max-spread-frac", type=float, default=0.35, help="Drop if (ask-bid)/mid > this")
    p.add_argument("--min-oi", type=int, default=25, help="Min open interest")
    p.add_argument("--min-vol", type=int, default=5,  help="Min volume")
    p.add_argument("--stock-half-spread-bps", type=float, default=1.0, help="Estimated half-spread of stock in bps")
    p.add_argument("--extra-buffer", type=float, default=0.03, help="Extra $ cushion beyond half-spreads")
    p.add_argument("--top-n", type=int, default=25, help="Rows to print")
    p.add_argument("--out-csv", default="parity_advanced_scan.csv", help="Output CSV")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
