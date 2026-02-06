import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import Normalize

# ------------------------------------------------------------
# SETTINGS  (NOW IN ka b2k)
# ------------------------------------------------------------
DATA_FILE = "orbital_param.csv"

AGE_MIN_B2K, AGE_MAX_B2K = 115, 250  # <-- ka b2k (NOT BP)
SHIFT_YEARS = 50  # BP -> b2k = +50 years
SHIFT_KA = SHIFT_YEARS / 1000.0  # 0.05 ka

DPI_JPG = 600
CMAP = "viridis_r"  # low=yellow, high=purple (we invert axis for 128 top)


def robust_read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    df.columns = df.columns.str.strip()
    return df


def export_figure(fig, out_base):
    fig.savefig(f"{out_base}.jpg", dpi=DPI_JPG, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"✔ saved {out_base}.jpg")
    print(f"✔ saved {out_base}.svg")


def scatter_plot(
    x,
    y,
    age_b2k,
    xlabel,
    ylabel,
    out_base,
    x_major=None,
    x_minor=None,
    x_fmt=None,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Fix colour scale to requested b2k range
    norm = Normalize(vmin=AGE_MIN_B2K, vmax=AGE_MAX_B2K)

    sc = ax.scatter(
        x,
        y,
        c=age_b2k,
        cmap=CMAP,
        norm=norm,
        s=50,
        alpha=0.85,
        edgecolors="none",
    )

    # Colorbar: 128 at TOP, 220 at BOTTOM, and colors: 128 yellow -> 220 purple
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Age [kyr b2k]", fontsize=20, fontweight="bold")
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=16)

    # Explicit ticks (ensure 128 is shown at the top)
    # Choose intermediate ticks you like; keep them within [128,220]
    ticks = [AGE_MIN_B2K, 140, 160, 180, 200, AGE_MAX_B2K]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    # Axis labels (bold)
    ax.set_xlabel(xlabel, fontsize=24, labelpad=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=24, labelpad=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)

    # Optional x-axis cleanup
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    if x_minor is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    if x_fmt is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(x_fmt))

    export_figure(fig, out_base)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = robust_read_table(DATA_FILE)

    # Column names (as in your file)
    col_age_bp = "Age"  # Age column is ka BP in your CSV
    col_ecc = "ECC"
    col_obl = "OBL"
    col_omega = "OMEGA"
    col_insol = "EXI"

    for c in [col_age_bp, col_ecc, col_obl, col_omega, col_insol]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert filter window from b2k -> BP (because the file is BP)
    age_min_bp = AGE_MIN_B2K - SHIFT_KA
    age_max_bp = AGE_MAX_B2K - SHIFT_KA

    sub = df[(df[col_age_bp] >= age_min_bp) & (df[col_age_bp] <= age_max_bp)].copy()
    sub = sub.sort_values(col_age_bp)

    # Convert ages to b2k for plotting
    age_b2k = sub[col_age_bp] + SHIFT_KA

    ecc = sub[col_ecc]
    insol = sub[col_insol]

    # Obliquity scaling fix (22340..24582 -> 22.340..24.582 deg)
    obl_raw = sub[col_obl].copy()
    obl = obl_raw / 1000.0 if obl_raw.max() > 100 else obl_raw

    # Precession index: e * sin(omega)
    omega_rad = np.deg2rad(sub[col_omega])
    prec_index = ecc * np.sin(omega_rad)

    # --------------------------------------------------------
    # 1) Insolation vs Eccentricity
    # --------------------------------------------------------
    scatter_plot(
        x=ecc,
        y=insol,
        age_b2k=age_b2k,
        xlabel="Eccentricity [-]",
        ylabel="Insolation 65°N July [W/m²]",
        out_base="corr_insolation_vs_ecc",
        x_major=0.01,
        x_minor=0.002,
        x_fmt="%.2f",
    )

    # --------------------------------------------------------
    # 2) Insolation vs Obliquity
    # --------------------------------------------------------
    scatter_plot(
        x=obl,
        y=insol,
        age_b2k=age_b2k,
        xlabel="Obliquity [deg]",
        ylabel="Insolation 65°N July [W/m²]",
        out_base="corr_insolation_vs_obliquity",
        x_major=0.5,
        x_minor=0.1,
        x_fmt="%.1f",
    )

    # --------------------------------------------------------
    # 3) Insolation vs Precession index
    # --------------------------------------------------------
    scatter_plot(
        x=prec_index,
        y=insol,
        age_b2k=age_b2k,
        xlabel=r"Precession index  $e\cdot\sin(\omega)$ [-]",
        ylabel="Insolation 65°N July [W/m²]",
        out_base="corr_insolation_vs_precession_index",
        x_major=0.02,
        x_minor=0.005,
        x_fmt="%.2f",
    )


if __name__ == "__main__":
    main()
