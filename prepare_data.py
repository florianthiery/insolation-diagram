"""Derive ``orbital_param.csv`` from the canonical ``Orbital_param.tab``.

The CSV is a convenience derivative for anyone who wants the table without the
PANGAEA metadata header: same rows, same values, short column names, point as
decimal separator. Nothing in this repository reads it - the plotting scripts
read the .tab file directly - so it can be regenerated or deleted at any time.
"""

import os

import orbital_data

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
FLOAT_FORMAT = "%g"  # keep the published precision, no trailing zeros
LINE_TERMINATOR = "\n"  # same file on Windows and Linux

# -------------------------------------------------
# BUILD
# -------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = orbital_data.read_pangaea_tab()
df.to_csv(
    orbital_data.CSV_FILE,
    index=False,
    float_format=FLOAT_FORMAT,
    lineterminator=LINE_TERMINATOR,
)

print(
    f"\u2714 saved {orbital_data.CSV_FILE.name} "
    f"({len(df)} rows, {len(df.columns)} columns) "
    f"from {orbital_data.TAB_FILE.name}"
)
