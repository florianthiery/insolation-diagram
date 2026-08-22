"""Insolation at an arbitrary latitude from the Berger & Loutre orbital parameters.

The PANGAEA table supplies the orbital elements (eccentricity, obliquity,
longitude of perihelion) - which are properties of the Earth's orbit and
therefore independent of any site - together with one derived insolation column,
``EXI``, that is valid for 65 deg N only. Mid-month insolation for any other
latitude can be recomputed from the same elements with the standard expression
of Berger (1978):

    W = S0/pi * (a/r)^2 * (H0 sin(phi) sin(delta) + cos(phi) cos(delta) sin(H0))

    sin(delta) = sin(eps) sin(lambda)
    a/r        = (1 + e cos(lambda - varpi)) / (1 - e^2)
    cos(H0)    = -tan(phi) tan(delta)          (H0 = 0 or pi at the poles)

with phi the latitude, eps the obliquity, lambda the true solar longitude
measured from the moving vernal equinox, varpi the longitude of perihelion in
the same reference, and H0 the half-day length.

Convention
----------
The three constants below were not assumed but determined by fitting the
expression to the published ``EXI`` column over all 5001 rows of the table:

    lambda = 120 deg    mid-month July, months being 30 deg increments of true
                        solar longitude starting at the vernal equinox
    varpi  = OMEGA + 180 deg
    S0     = 1360 W/m2

With these values the formula reproduces ``EXI`` to a root-mean-square
difference of 0.005 W/m2 and a maximum difference of 0.016 W/m2, i.e. to the
rounding precision of the published values. :func:`validate` re-runs that check.

Reference
---------
Berger, A. (1978): Long-term variations of daily insolation and Quaternary
climatic changes. Journal of the Atmospheric Sciences, 35(12), 2362-2367.
https://doi.org/10.1175/1520-0469(1978)035<2362:LTVODI>2.0.CO;2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import orbital_data

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
SOLAR_CONSTANT = 1360.0  # W/m2, as used by Berger & Loutre
LAMBDA_MID_JULY = 120.0  # deg true solar longitude
PERIHELION_OFFSET = 180.0  # deg, OMEGA -> longitude of perihelion

# Tolerance of the self-test against the published 65 deg N column [W/m2]
VALIDATION_TOLERANCE = 0.05
REFERENCE_LATITUDE = 65.0

SITE_GEOJSON = Path(__file__).resolve().parent / "maare.geojson"


@dataclass(frozen=True)
class Site:
    name: str
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        """Axis label fragment, e.g. ``50.3 deg N``."""
        hemisphere = "N" if self.latitude >= 0 else "S"
        return f"{abs(self.latitude):.1f}\u00b0{hemisphere}"


def read_site(path: str | Path = SITE_GEOJSON, index: int = 0) -> Site:
    """Read a point site from a GeoJSON FeatureCollection."""
    with Path(path).open(encoding="utf-8") as handle:
        collection = json.load(handle)
    feature = collection["features"][index]
    longitude, latitude = feature["geometry"]["coordinates"][:2]
    name = feature.get("properties", {}).get("name", "site")
    return Site(name=name, latitude=float(latitude), longitude=float(longitude))


# -------------------------------------------------
# INSOLATION
# -------------------------------------------------
def mid_month_insolation(
    latitude: float,
    eccentricity,
    obliquity_deg,
    omega_deg,
    solar_longitude_deg: float = LAMBDA_MID_JULY,
    solar_constant: float = SOLAR_CONSTANT,
):
    """Mid-month insolation [W/m2] at ``latitude`` for the given orbital elements."""
    phi = np.deg2rad(latitude)
    lam = np.deg2rad(solar_longitude_deg)
    eps = np.deg2rad(np.asarray(obliquity_deg, dtype=float))
    varpi = np.deg2rad(np.asarray(omega_deg, dtype=float) + PERIHELION_OFFSET)
    ecc = np.asarray(eccentricity, dtype=float)

    sin_delta = np.sin(eps) * np.sin(lam)
    delta = np.arcsin(sin_delta)

    # inverse normalised Sun-Earth distance a/r
    rho = (1.0 + ecc * np.cos(lam - varpi)) / (1.0 - ecc**2)

    # half-day length; clipped for polar night / midnight sun
    half_day = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))

    return (
        solar_constant
        / np.pi
        * rho**2
        * (
            half_day * np.sin(phi) * sin_delta
            + np.cos(phi) * np.cos(delta) * np.sin(half_day)
        )
    )


def insolation_for_table(df: pd.DataFrame, latitude: float):
    """Apply :func:`mid_month_insolation` to a table read by ``orbital_data``."""
    return mid_month_insolation(
        latitude=latitude,
        eccentricity=df[orbital_data.COL_ECC],
        obliquity_deg=df[orbital_data.COL_OBL],
        omega_deg=df[orbital_data.COL_OMEGA],
    )


def validate(df: pd.DataFrame) -> dict:
    """Recompute the published 65 deg N column and report the deviation."""
    recomputed = insolation_for_table(df, REFERENCE_LATITUDE)
    published = df[orbital_data.COL_INSOL].to_numpy(dtype=float)
    difference = np.asarray(recomputed, dtype=float) - published

    report = {
        "n": int(difference.size),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(np.abs(difference))),
    }
    if report["max_abs"] > VALIDATION_TOLERANCE:
        raise ValueError(
            "Insolation formula does not reproduce the published 65\u00b0N column "
            f"(max deviation {report['max_abs']:.3f} W/m\u00b2 > "
            f"{VALIDATION_TOLERANCE} W/m\u00b2)"
        )
    return report
