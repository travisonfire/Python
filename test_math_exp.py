import math
import pandas as pd
import numpy as np

# Simple check to verify math.exp works with floats
S0 = 100.0
K = 95.0
r = 0.05
T = 1.0
q = 0.02

# Basic parity formula
delta = (7.0 - 3.0) - (S0 * math.exp(-q * T) - K * math.exp(-r * T))
print("Delta =", delta)

# Small DataFrame test to ensure numeric calculations work
data = {
    "S0": [100.0, 200.0],
    "K": [95.0, 210.0],
    "r": [0.05, 0.04],
    "T": [1.0, 0.5],
    "q": [0.02, 0.01],
    "C": [7.0, 12.5],
    "P": [3.0, 9.0]
}
df = pd.DataFrame(data)
df["delta"] = (df["C"] - df["P"]) - (df["S0"] * np.exp(-df["q"] * df["T"]) - df["K"] * np.exp(-df["r"] * df["T"]))
print(df)
