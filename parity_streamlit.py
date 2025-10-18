import math, numpy as np, pandas as pd, yfinance as yf
import streamlit as st
from datetime import datetime, timezone

st.set_page_config(page_title="Parity Scanner", layout="wide")
st.title("Advanced Put–Call Parity Scanner")

# Sidebar controls
ticker = st.sidebar.text_input("Ticker", value="TSLA")
use_treasury = st.sidebar.checkbox("Use Yahoo Treasury Curve (^IRX,^FVX,^TNX)", value=False)
flat_r = st.sidebar.number_input("Flat risk-free rate (dec)", value=0.05, step=0.005, format="%.3f")
div_yield = st.sidebar.number_input("Dividend yield q (dec)", value=0.00, step=0.001, format="%.3f")
borrow_yield = st.sidebar.number_input("Borrow cost b (dec)", value=0.00, step=0.001, format="%.3f")
q_eff = div_yield - borrow_yield

max_spread_frac = st.sidebar.slider("Max (ask-bid)/mid", 0.05, 0.8, 0.35, 0.05)
min_oi = st.sidebar.number_input("Min open interest", value=25, step=5)
min_vol = st.sidebar.number_input("Min volume", value=5, step=1)
stock_half_spread_bps = st.sidebar.number_input("Stock half-spread (bps)", value=1.0, step=0.5, format="%.1f")
extra_buffer = st.sidebar.number_input("Extra $ buffer", value=0.03, step=0.01, format="%.2f")
first_n = st.sidebar.number_input("First N expiries (0=all)", value=8, step=1)

@st.cache_data(ttl=180)
def load_curve(use_treasury: bool, flat_r: float):
    if not use_treasury:
        return {"flat": float(flat_r)}
    try:
        tickers = yf.download(["^IRX","^FVX","^TNX"], period="5d", interval="1d", progress=False)["Close"].ffill().iloc[-1]
        y_3m = float(tickers["^IRX"])/100.0
        y_5y = float(tickers["^FVX"])/100.0
        y_10y= float(tickers["^TNX"])/100.0
        return {"points": [(0.25, y_3m), (5.0, y_5y), (10.0, y_10y)]}
    except Exception:
        return {"flat": float(flat_r)}

def interp_rate(T, curve):
    if "flat" in curve:
        return float(curve["flat"])
    pts = sorted(curve["points"], key=lambda x: x[0])
    Ts, Rs = zip(*pts)
    T = max(min(T, Ts[-1]), Ts[0])
    return float(np.interp(T, Ts, Rs))

def years_to_expiry(exp_str: str) -> float:
    now = datetime.now(timezone.utc)
    t = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = max((t - now).total_seconds() / 86400.0, 0.0)
    return max(days/365.0, 1/365)

@st.cache_data(ttl=60)
def get_chain_df(ticker):
    tk = yf.Ticker(ticker)
    spot_hist = tk.history(period="1d")
    if spot_hist.empty:
        return None, []
    S0 = float(spot_hist["Close"].iloc[-1])
    return S0, tk.options

def scan_once(ticker, S0, expiries, curve, first_n, q_eff,
              max_spread_frac, min_oi, min_vol, stock_half_spread_bps, extra_buffer):
    rows = []
    exps = expiries if (first_n == 0) else expiries[:first_n]
    for exp in exps:
        T = years_to_expiry(exp)
        r_T = interp_rate(T, curve)
        oc = yf.Ticker(ticker).option_chain(exp)
        calls = oc.calls.rename(columns={"bid":"call_bid","ask":"call_ask",
                                         "lastPrice":"call_last","openInterest":"call_oi","volume":"call_vol"})
        puts  = oc.puts.rename(columns={"bid":"put_bid","ask":"put_ask",
                                         "lastPrice":"put_last","openInterest":"put_oi","volume":"put_vol"})
        df = pd.merge(
            calls[["strike","call_bid","call_ask","call_last","call_oi","call_vol"]],
            puts[ ["strike","put_bid","put_ask","put_last","put_oi","put_vol"]],
            on="strike", how="inner"
        )
        if df.empty: 
            continue
        # numeric & mids
        num_cols = ["strike","call_bid","call_ask","put_bid","put_ask","call_last","put_last","call_oi","put_oi","call_vol","put_vol"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["call_bid","call_ask","put_bid","put_ask"])
        df["call_mid"] = (df["call_bid"] + df["call_ask"])/2.0
        df["put_mid"]  = (df["put_bid"]  + df["put_ask"] )/2.0
        df = df[(df["call_mid"]>0)&(df["put_mid"]>0)].copy()
        # filters
        df["call_spread_frac"] = (df["call_ask"]-df["call_bid"])/df["call_mid"]
        df["put_spread_frac"]  = (df["put_ask"] -df["put_bid"] )/df["put_mid"]
        df = df[(df["call_spread_frac"]<=max_spread_frac)&(df["put_spread_frac"]<=max_spread_frac)]
        df = df[(df["call_oi"]>=min_oi)&(df["put_oi"]>=min_oi)&(df["call_vol"]>=min_vol)&(df["put_vol"]>=min_vol)]
        if df.empty: 
            continue
        # vectorized Δ
        S_term = S0 * np.exp(-q_eff * T)
        K_term = df["strike"].values * np.exp(-r_T * T)
        delta = (df["call_mid"].values - df["put_mid"].values) - (S_term - K_term)
        # hurdle
        call_half = (df["call_ask"].values - df["call_bid"].values)/2.0
        put_half  = (df["put_ask"].values  - df["put_bid"].values )/2.0
        stock_half= (stock_half_spread_bps/1e4) * S0
        hurdle = np.maximum(call_half,0)+np.maximum(put_half,0)+stock_half+extra_buffer
        signal = np.where(delta>hurdle,"EXPENSIVE",
                   np.where(delta<-hurdle,"CHEAP","FAIR"))
        out = df.copy()
        out["expiry"]=exp; out["S0"]=S0; out["T"]=T; out["r_T"]=r_T; out["q_eff"]=q_eff
        out["delta"]=delta; out["hurdle"]=hurdle
        out["econ_gap"]=np.abs(delta)-hurdle
        out["signal"]=signal
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# Main
curve = load_curve(use_treasury, flat_r)
S0, expiries = get_chain_df(ticker)
if S0 is None or not expiries:
    st.error("No data returned. Check ticker or internet.")
else:
    st.subheader(f"{ticker} Spot: ${S0:.2f}")
    run_btn = st.button("Scan")
    if run_btn:
        full = scan_once(ticker, S0, expiries, curve, int(first_n), q_eff,
                         max_spread_frac, int(min_oi), int(min_vol),
                         float(stock_half_spread_bps), float(extra_buffer))
        if full.empty:
            st.info("No rows after filters. Loosen thresholds.")
        else:
            full = full.sort_values("econ_gap", ascending=False)
            st.write("Top results (after transaction-cost hurdle):")
            show_cols = ["expiry","strike","S0","T","r_T","q_eff",
                         "call_bid","call_ask","put_bid","put_ask",
                         "call_mid","put_mid","delta","hurdle","econ_gap","signal",
                         "call_vol","put_vol","call_oi","put_oi"]
            st.dataframe(full[show_cols].head(200), use_container_width=True)

            # Optional heatmap inside Streamlit
            st.write("Heatmap: mean Δ by expiry × strike")
            df = full.copy()
            df["exp_short"] = pd.to_datetime(df["expiry"]).dt.strftime("%m-%d")
            pv = df.pivot_table(index="exp_short", columns="strike", values="delta", aggfunc="mean")
            if pv.empty:
                st.info("Heatmap pivot empty after filters.")
            else:
                st.dataframe(pv, use_container_width=True)
