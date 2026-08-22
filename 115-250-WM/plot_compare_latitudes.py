"""Compare the published 65 deg N reference insolation with the site insolation.

Panel (a) shows both mid-month July insolation curves over the age window,
panel (b) their difference together with obliquity, which is what the difference
between the two latitudes essentially tracks.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import insolation_site  # noqa: E402
import orbital_data  # noqa: E402

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
DATA_FILE = "orbital_param_wm.tab"  # written by prepare_data_wm.py
COL_INSOL_SITE = "EXI_WM"

AGE_MIN_B2K, AGE_MAX_B2K = 115, 250  # ka b2k
SHIFT_KA = 0.05  # BP -> b2k

OUT_BASE = "compare_insolation_65N_vs_wm"
DPI_JPG = 600

COL_REF = "#4d4d4d"  # 65 deg N reference curve
COL_SITE = "#1f77b4"  # site curve
COL_OBL = "#9467bd"  # obliquity

FS_LABEL = 20
FS_TICK = 16
FS_LEGEND = 15
FS_PANEL = 24


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    site = insolation_site.read_site()
    df = orbital_data.read_pangaea_tab(DATA_FILE)
    df["age"] = df[orbital_data.COL_AGE] + SHIFT_KA
    sub = df[(df.age >= AGE_MIN_B2K) & (df.age <= AGE_MAX_B2K)].sort_values("age")

    age = sub["age"]
    ref = sub[orbital_data.COL_INSOL]
    local = sub[COL_INSOL_SITE]
    difference = local - ref

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, height_ratios=[2, 1]
    )

    # --- (a) both curves -------------------------------------------------
    ax_top.plot(age, ref, color=COL_REF, linewidth=2, label="65\u00b0N (EXI, PANGAEA)")
    ax_top.plot(
        age,
        local,
        color=COL_SITE,
        linewidth=2,
        label=f"{site.label} ({site.name})",
    )
    ax_top.set_ylabel("Insolation July [W/m\u00b2]", fontsize=FS_LABEL, fontweight="bold")
    ax_top.legend(loc="lower right", fontsize=FS_LEGEND, frameon=True)
    ax_top.text(
        0.01, 0.94, "(a)", transform=ax_top.transAxes, fontsize=FS_PANEL,
        fontweight="bold", va="top",
    )

    # --- (b) difference and obliquity ------------------------------------
    ax_bottom.plot(
        age, difference, color=COL_SITE, linewidth=2,
        label=f"{site.label} \u2212 65\u00b0N",
    )
    ax_bottom.set_ylabel("Difference [W/m\u00b2]", fontsize=FS_LABEL, fontweight="bold")
    ax_bottom.set_xlabel("Age [ka b2k]", fontsize=FS_LABEL, fontweight="bold")

    ax_obl = ax_bottom.twinx()
    ax_obl.plot(
        age, sub[orbital_data.COL_OBL], color=COL_OBL, linewidth=2, linestyle="--",
        label="Obliquity",
    )
    ax_obl.set_ylabel("Obliquity [deg]", fontsize=FS_LABEL, fontweight="bold",
                      color=COL_OBL)
    ax_obl.tick_params(axis="y", labelsize=FS_TICK, colors=COL_OBL)
    ax_obl.invert_yaxis()  # high obliquity = small difference

    correlation = np.corrcoef(difference, sub[orbital_data.COL_OBL])[0, 1]
    handles = ax_bottom.get_legend_handles_labels()[0] + ax_obl.get_legend_handles_labels()[0]
    labels = (
        ax_bottom.get_legend_handles_labels()[1]
        + [f"Obliquity (inverted axis), r = {correlation:+.2f}"]
    )
    ax_bottom.legend(handles, labels, loc="lower right", fontsize=FS_LEGEND, frameon=True)
    ax_bottom.text(
        0.01, 0.92, "(b)", transform=ax_bottom.transAxes, fontsize=FS_PANEL,
        fontweight="bold", va="top",
    )

    for axis in (ax_top, ax_bottom):
        axis.set_xlim(AGE_MIN_B2K, AGE_MAX_B2K)
        axis.xaxis.set_major_locator(MultipleLocator(10))
        axis.xaxis.set_minor_locator(MultipleLocator(2))
        axis.tick_params(axis="both", labelsize=FS_TICK)
        axis.grid(axis="x", color="#dddddd", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(f"{OUT_BASE}.jpg", dpi=DPI_JPG, bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\u2714 saved {OUT_BASE}.jpg")
    print(f"\u2714 saved {OUT_BASE}.svg")

    print(
        f"  mean difference {difference.mean():+.1f} W/m\u00b2 "
        f"(min {difference.min():+.1f}, max {difference.max():+.1f}), "
        f"r(difference, obliquity) = {correlation:+.3f}"
    )


if __name__ == "__main__":
    main()
