import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.signal import argrelextrema

# -------------------------------------------------
# Arbeitsverzeichnis auf Ordner des Skripts setzen
# -------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# gemeinsamer Reader im Repo-Root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbital_data  # noqa: E402

# -------------------------------------------------
# INPUT
# -------------------------------------------------
data_file = orbital_data.TAB_FILE  # Orbital_param.tab im Repo-Root
AGE_MIN_YR_B2K = 115_000
AGE_MAX_YR_B2K = 250_000

# BP -> b2k shift
SHIFT_YEARS = 50

# Marker-Farben (kein Rot)
COL_ECC = "#1f77b4"  # blau
COL_OBL = "#9467bd"  # violett
COL_PREC = "#2ca02c"  # grün

# Extrema-Empfindlichkeit (größer = weniger Punkte)
EXTREMA_ORDER = 3


def get_extrema_idx(series: pd.Series, order: int = 3):
    arr = np.asarray(series, dtype=float)
    max_idx = argrelextrema(arr, np.greater, order=order)[0]
    min_idx = argrelextrema(arr, np.less, order=order)[0]
    return max_idx, min_idx


# -------------------------------------------------
# Load
# -------------------------------------------------
df = orbital_data.read_pangaea_tab(data_file)
print("Spalten im DataFrame nach dem Einlesen:", df.columns.tolist())

# Spaltennamen (Kurzform, siehe orbital_data)
col_age = orbital_data.COL_AGE  # ka BP
col_ecc = orbital_data.COL_ECC
col_obl = orbital_data.COL_OBL  # deg
col_omega = orbital_data.COL_OMEGA
col_insol = orbital_data.COL_INSOL  # Insolation 65°N July [W/m²]

# -------------------------------------------------
# Alter umrechnen: ka BP -> yr b2k
#  - ka BP -> yr BP: * 1000
#  - BP (1950) -> b2k (2000): +50 Jahre
# -------------------------------------------------
df["Age_yr_b2k"] = df[col_age] * 1000.0 + SHIFT_YEARS

# Filterzeitraum
df = df[
    (df["Age_yr_b2k"] >= AGE_MIN_YR_B2K) & (df["Age_yr_b2k"] <= AGE_MAX_YR_B2K)
].copy()
df = df.dropna(subset=["Age_yr_b2k", col_insol, col_ecc, col_obl, col_omega])
df = df.sort_values("Age_yr_b2k")

age = df["Age_yr_b2k"]
insol = df[col_insol]
ecc = df[col_ecc]

# OBL steht in der .tab bereits in Grad
obl = df[col_obl]

# Precession index: e*sin(omega)
omega_rad = np.deg2rad(df[col_omega])
prec_index = ecc * np.sin(omega_rad)

# -------------------------------------------------
# Extrema bestimmen (in den PARAMETERN, nicht in Insolation!)
# Indizes beziehen sich auf die sortierte Zeitreihe.
# -------------------------------------------------
ecc_max, ecc_min = get_extrema_idx(ecc, order=EXTREMA_ORDER)
obl_max, obl_min = get_extrema_idx(obl, order=EXTREMA_ORDER)
prec_max, prec_min = get_extrema_idx(prec_index, order=EXTREMA_ORDER)

# -------------------------------------------------
# Plot im Corchia-Stil (nur mit Insolation als x)
# -------------------------------------------------
fig = plt.figure(figsize=(10, 30), dpi=100)
ax = fig.add_subplot(111)

# Linie: Insolation vs Age
ax.plot(insol, age, linewidth=1, color="black", label="Insolation")

# Marker: an den Alterspositionen der Extrema, aber auf der Insolation-Linie
# (d.h. x = Insolation zu diesem Zeitpunkt, y = Age zu diesem Zeitpunkt)
ax.scatter(
    insol.iloc[ecc_max],
    age.iloc[ecc_max],
    marker="x",
    s=160,
    linewidths=2.0,
    color=COL_ECC,
    zorder=5,
    label="Ecc maxima",
)
ax.scatter(
    insol.iloc[ecc_min],
    age.iloc[ecc_min],
    marker="x",
    s=160,
    linewidths=2.0,
    color=COL_ECC,
    alpha=0.45,
    zorder=5,
    label="Ecc minima",
)

ax.scatter(
    insol.iloc[obl_max],
    age.iloc[obl_max],
    marker="D",
    s=160,
    linewidths=1.5,
    edgecolors="white",
    color=COL_OBL,
    zorder=5,
    label="Obl maxima",
)
ax.scatter(
    insol.iloc[obl_min],
    age.iloc[obl_min],
    marker="D",
    s=160,
    linewidths=1.5,
    edgecolors="white",
    color=COL_OBL,
    alpha=0.45,
    zorder=5,
    label="Obl minima",
)

ax.scatter(
    insol.iloc[prec_max],
    age.iloc[prec_max],
    marker="o",
    s=120,
    linewidths=1.2,
    edgecolors="white",
    color=COL_PREC,
    zorder=5,
    label="Prec index maxima",
)
ax.scatter(
    insol.iloc[prec_min],
    age.iloc[prec_min],
    marker="o",
    s=120,
    linewidths=1.2,
    edgecolors="white",
    color=COL_PREC,
    alpha=0.45,
    zorder=5,
    label="Prec index minima",
)

# Y-Achse wie bei dir: oben jung, unten alt
ax.set_ylim(AGE_MAX_YR_B2K, AGE_MIN_YR_B2K)
ax.margins(y=0)

# X-Achse oben
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

# Raster wie Corchia
ax.yaxis.set_major_locator(MultipleLocator(10_000))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1000)}"))
ax.grid(axis="y", which="major", color="#cccccc", linewidth=1)

ax.yaxis.set_minor_locator(MultipleLocator(2_000))
ax.tick_params(axis="y", which="minor", length=4, width=0.8)

# Labels (fett, größer Stil)
ax.set_xlabel(
    "Insolation 65°N July [W/m²]", fontsize=24, labelpad=12, fontweight="bold"
)
ax.set_ylabel("Age [ka]", fontsize=24, labelpad=12, fontweight="bold")

# Tickgrößen (wie im EPICA-Skript)
ax.tick_params(axis="x", labelsize=26)
ax.tick_params(axis="y", labelsize=26)

# Legende (optional, du kannst sie auch auskommentieren)
ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=13, frameon=True)

# Export
out_base = "insolation_vs_age_orbital_extrema"
plt.savefig(f"{out_base}.jpg", bbox_inches="tight")
plt.savefig(f"{out_base}.svg", bbox_inches="tight")
plt.close(fig)

print(f"Saved ✓ {out_base}.jpg / {out_base}.svg")
