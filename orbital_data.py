"""Reader for the PANGAEA orbital parameter export.

``Orbital_param.tab`` is the unmodified download from

    Berger, A. & Loutre, M.-F. (1999): Parameters of the Earths orbit for the
    last 5 Million years in 1 kyr resolution. PANGAEA,
    https://doi.org/10.1594/PANGAEA.56040

It is the canonical input of this repository: tab separated, preceded by a
PANGAEA metadata header that is delimited by a line containing ``*/``.

Column names are shortened to the identifiers used by the plotting scripts:

    Age [ka BP]                          -> Age    (ka BP)
    ECC                                  -> ECC
    ECC (413k component notched out)     -> ECC_413k_notched_out
    OMEGA [deg]                          -> OMEGA  (deg)
    OBL [deg]                            -> OBL    (deg)
    Prec                                 -> Prec
    EXI [W/m**2] (mid-month insolation)  -> EXI    (W/m2, 65 deg N July)
    ETP                                  -> ETP

Values are taken over unchanged - in particular ``OBL`` is in degrees and
``Prec`` is dimensionless, as published. (The superseded per-directory
``orbital_param.csv`` files had lost the decimal point in both columns, which
is why the plotting scripts used to divide ``OBL`` by 1000.)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
TAB_FILE = Path(__file__).resolve().parent / "Orbital_param.tab"

# Derived CSV written by prepare_data.py
CSV_FILE = Path(__file__).resolve().parent / "orbital_param.csv"

HEADER_END = "*/"  # last line of the PANGAEA metadata block

COL_AGE = "Age"  # ka BP
COL_ECC = "ECC"
COL_OBL = "OBL"  # deg
COL_OMEGA = "OMEGA"  # deg
COL_INSOL = "EXI"  # W/m2, mid-month insolation 65 deg N July

REQUIRED = (COL_AGE, COL_ECC, COL_OBL, COL_OMEGA, COL_INSOL)


def _short_name(raw: str) -> str:
    """``'OBL [deg]'`` -> ``'OBL'``, keeping the two eccentricity columns apart."""
    if "413k" in raw:
        return "ECC_413k_notched_out"
    return re.split(r"\s*[\[(]", raw.strip(), maxsplit=1)[0].strip()


def read_pangaea_tab(path: str | Path = TAB_FILE) -> pd.DataFrame:
    """Read the PANGAEA export, skipping its metadata header."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()

    skiprows = 0
    for index, line in enumerate(lines):
        if line.strip() == HEADER_END:
            skiprows = index + 1
            break
    else:
        raise ValueError(
            f"{path.name}: no '{HEADER_END}' line found - is this a PANGAEA export?"
        )

    df = pd.read_csv(path, sep="\t", skiprows=skiprows, encoding="utf-8")
    df.columns = [_short_name(c) for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {df.columns.tolist()}")

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
