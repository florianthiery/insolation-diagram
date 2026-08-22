"""Build the multi-panel overall figure 128-220 from the four single figures.

Panel (A) is the insolation curve produced by ``plot_insolation.py``, panels
(B)-(D) are the correlation plots produced by ``plot_correlation.py``.  The
panels are placed as vector graphics on a millimetre canvas, reproducing the
layout that used to be assembled by hand in Inkscape.

Run ``plot_insolation.py`` and ``plot_correlation.py`` first - this script only
composes their SVG output.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svg_compose import Panel, compose, rasterise  # noqa: E402

# -------------------------------------------------
# INPUT / OUTPUT
# -------------------------------------------------
PANEL_A_SVG = "insolation_vs_age_orbital_extrema.svg"
PANEL_BCD_SVG = [
    "corr_insolation_vs_ecc.svg",
    "corr_insolation_vs_obliquity.svg",
    "corr_insolation_vs_precession_index.svg",
]
OUT_BASE = "overall_128-220"

# -------------------------------------------------
# LAYOUT (all lengths in mm, origin = top left)
# -------------------------------------------------
CANVAS_HEIGHT_MM = 640.0  # = 3 rows of 200 mm + 2 gaps of 20 mm
CANVAS_MARGIN_MM = 0.0  # extra space right of the last column

RIGHT_COLUMN_X_MM = 260.0  # left edge of panels (B)-(D)
ROW_HEIGHT_MM = 200.0
ROW_GAP_MM = 20.0

LABEL_FONT_MM = 17.6389  # = 50 pt
LABEL_A_MM = (9.01, 20.47)  # text baseline of "(A)"
LABEL_BCD_X_MM = 259.29
LABEL_BCD_Y_MM = 192.65  # baseline within the first row, repeated per row

JPG_DPI = 198.4375  # -> 5000 px canvas height, as in the Inkscape export

# -------------------------------------------------
# BUILD
# -------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

panels = [Panel(PANEL_A_SVG, x_mm=0.0, y_mm=0.0, height_mm=CANVAS_HEIGHT_MM)]
labels = [("(A)", *LABEL_A_MM)]

for row, (svg, name) in enumerate(zip(PANEL_BCD_SVG, "BCD")):
    y_mm = row * (ROW_HEIGHT_MM + ROW_GAP_MM)
    panels.append(
        Panel(svg, x_mm=RIGHT_COLUMN_X_MM, y_mm=y_mm, height_mm=ROW_HEIGHT_MM)
    )
    labels.append((f"({name})", LABEL_BCD_X_MM, LABEL_BCD_Y_MM + y_mm))

missing = [p.path for p in panels if not Path(p.path).exists()]
if missing:
    raise SystemExit(
        "Missing panel(s): "
        + ", ".join(missing)
        + "\nRun plot_insolation.py and plot_correlation.py first."
    )

compose(
    panels=panels,
    labels=labels,
    canvas_height_mm=CANVAS_HEIGHT_MM,
    canvas_margin_mm=CANVAS_MARGIN_MM,
    label_font_mm=LABEL_FONT_MM,
    out_svg=f"{OUT_BASE}.svg",
)
rasterise(f"{OUT_BASE}.svg", f"{OUT_BASE}.jpg", dpi=JPG_DPI)
