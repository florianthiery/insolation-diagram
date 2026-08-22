"""Recompute the orbital table for the latitude of the Walsdorfer Maar.

Reads the canonical ``Orbital_param.tab`` from the repository root, adds a
column ``EXI_WM`` holding mid-month July insolation at the site latitude, and
writes both a .tab file (same format as the source, with a provenance header)
and a plain CSV derivative into this directory.

The orbital elements themselves are properties of the Earth's orbit and are
copied over unchanged; only the insolation column is site dependent. Before
anything is written, the same formula is used to recompute the published 65 deg N
column, which serves as a self-test of the implementation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import insolation_site  # noqa: E402
import orbital_data  # noqa: E402

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
OUT_TAB = "orbital_param_wm.tab"
OUT_CSV = "orbital_param_wm.csv"

COL_INSOL_SITE = "EXI_WM"  # mid-month July insolation at the site
FLOAT_FORMAT = "%.2f"  # insolation, as published for EXI
CSV_FLOAT_FORMAT = "%g"
LINE_TERMINATOR = "\n"

# Column header written into the .tab file, mirroring the PANGAEA style
TAB_HEADER = {
    orbital_data.COL_AGE: "Age [ka BP]",
    "ECC_413k_notched_out": "ECC (413k component notched out)",
    orbital_data.COL_OMEGA: "OMEGA [deg]",
    orbital_data.COL_OBL: "OBL [deg]",
    orbital_data.COL_INSOL: "EXI [W/m**2] (mid-month insolation 65N)",
    COL_INSOL_SITE: "EXI_WM [W/m**2] (mid-month insolation, site latitude)",
}

# -------------------------------------------------
# BUILD
# -------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

site = insolation_site.read_site()
df = orbital_data.read_pangaea_tab()

check = insolation_site.validate(df)
print(
    f"self-test against published 65\u00b0N column: "
    f"RMS {check['rms']:.4f} W/m\u00b2, max {check['max_abs']:.4f} W/m\u00b2 "
    f"over {check['n']} rows"
)

df[COL_INSOL_SITE] = insolation_site.insolation_for_table(df, site.latitude).round(2)

description = [
    "/* DATA DESCRIPTION:",
    "Derived dataset - NOT a PANGAEA download.",
    f"Source:\tOrbital_param.tab, Berger, A; Loutre, Marie-France (1999): "
    f"Parameters of the Earths orbit for the last 5 Million years in 1 kyr "
    f"resolution. PANGAEA, https://doi.org/10.1594/PANGAEA.56040",
    f"Derivation:\tcolumn {COL_INSOL_SITE} added by {Path(__file__).name}; "
    f"mid-month July insolation (true solar longitude "
    f"{insolation_site.LAMBDA_MID_JULY:.0f} deg, solar constant "
    f"{insolation_site.SOLAR_CONSTANT:.0f} W/m**2) recomputed after Berger (1978) "
    f"for the site latitude. All other columns copied unchanged.",
    f"Site:\t{site.name}, {site.latitude:.6f} N, {site.longitude:.6f} E",
    f"Validation:\tthe same formula reproduces the published 65 deg N column "
    f"EXI to RMS {check['rms']:.4f} W/m**2, max {check['max_abs']:.4f} W/m**2.",
    "*/",
]

columns = list(df.columns)
header = "\t".join(TAB_HEADER.get(c, c) for c in columns)
with open(OUT_TAB, "w", encoding="utf-8", newline=LINE_TERMINATOR) as handle:
    handle.write("\n".join(description) + "\n")
    handle.write(header + "\n")
    df.to_csv(handle, sep="\t", index=False, header=False, float_format="%g")
print(f"\u2714 saved {OUT_TAB} ({len(df)} rows) for {site.name} ({site.label})")

df.to_csv(
    OUT_CSV, index=False, float_format=CSV_FLOAT_FORMAT, lineterminator=LINE_TERMINATOR
)
print(f"\u2714 saved {OUT_CSV}")
