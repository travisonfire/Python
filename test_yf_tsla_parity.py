# test_yf_tsla_parity.py
import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

ticker = "TSLA"
tk = yf.Ticker(ticker)

# 1) Spot price
hist = tk.history(period="1d")
if hist.empty:
    raise RuntimeError("No TSLA history returned. Check internet or try again.")
S0 = float(hist["Close"].iloc[-1])

# 2) Choose nearest expiry
opts = tk.options
if not opts:
    raise RuntimeError("No option expiries returned for TSLA.")
expiry = opts[0]

# 3) Pull calls/puts for that expiry
oc = tk.option_chain(expiry)
calls = oc.calls[["strike","bid","ask"]].rename(columns={"bid":"call_bid","ask":"call_ask"})
puts  = oc.puts[ ["strike","bid","ask"]].rename(columns={"bid":"put_bid", "ask":"put_ask"})

# 4) Merge & build mid prices (coerce to numeric just in case)
df = pd.merge(calls, puts, on="strike", how="inner").copy()
for c in ["strike","call_bid","call_ask","put_bid","put_ask"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.dropna(subset=["strike","call_bid","call_ask","put_bid","put_ask"], inplace=True)

df["call_mid"] = (df["call_bid"] + df["call_ask"]) / 2.0
df["put_mid"]  = (df["put_bid"]  + df["put_ask"])  / 2.0
df = df[(df["call_mid"]>0) & (df["put_mid"]>0)].copy()

# 5) Time to expiry (years) and simple rates
today = datetime.utcnow()
T = (datetime.strptime(expiry, "%Y-%m-%d") - today).days / 365.0
T = max(T, 1/365)   # avoid zero
r = 0.05            # flat risk-free for test
q = 0.00            # TSLA dividend ~0

# 6) Compute parity delta for a few strikes near-the-money
df["abs_moneyness"] = (df["strike"] - S0).abs()
test = df.nsmallest(5, "abs_moneyness").copy()

test["delta"] = (
    (test["call_mid"] - test["put_mid"])
    - (S0 * np.exp(-q*T) - test["strike"] * np.exp(-r*T))
)

print(f"\nTSLA spot S0 = {S0:.2f}")
print(f"Expiry = {expiry}, T ≈ {T*365:.0f} days\n")
print(test[["strike","call_mid","put_mid","delta"]].to_string(index=False))

print("\nInterpretation:")
print("  delta > 0  → options rich vs parity")
print("  delta < 0  → options cheap vs parity")
print("  (Small values are usually within bid/ask noise on a quick test.)")
