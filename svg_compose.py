"""Compose several stand-alone matplotlib SVGs into one multi-panel figure.

This replaces the manual Inkscape step: the panels are placed - as vector
graphics, not as pixels - on a millimetre canvas, panel labels are added, and
the result is written as SVG and rasterised to JPG.

The only non-standard dependency is PyMuPDF, and only for the JPG; the SVG is
built with the standard library alone.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

PT_PER_MM = 72.0 / 25.4

# Attributes whose value may contain a "url(#id)" reference
_URL_ATTRS = ("clip-path", "style", "fill", "stroke", "mask", "filter")
_URL_RE = re.compile(r"url\(#([^)]+)\)")


@dataclass(frozen=True)
class Panel:
    """A stand-alone SVG placed on the canvas.

    ``x_mm`` / ``y_mm`` are the top-left corner of the panel box, ``height_mm``
    its target height; the width follows from the aspect ratio of the source.
    """

    path: str
    x_mm: float
    y_mm: float
    height_mm: float


def _length_pt(value: str) -> float:
    """Parse an SVG length. matplotlib writes pt, which is what we assume."""
    return float(re.sub(r"[a-z%]+$", "", value.strip()))


def _panel_size_pt(root: ET.Element) -> tuple[float, float]:
    view_box = root.get("viewBox")
    if view_box:
        _, _, width, height = (float(v) for v in view_box.replace(",", " ").split())
        return width, height
    return _length_pt(root.get("width")), _length_pt(root.get("height"))


def _prefix_ids(root: ET.Element, prefix: str) -> None:
    """Make all ids of a panel unique, so that panels cannot shadow each other.

    matplotlib reuses ids such as ``DejaVuSans-30`` across figures; without
    prefixing, a glyph or clip path of the first panel would be referenced by
    all following ones.
    """
    known = {el.get("id") for el in root.iter() if el.get("id")}

    def rename(name: str) -> str:
        return f"{prefix}{name}" if name in known else name

    for element in root.iter():
        if element.get("id"):
            element.set("id", rename(element.get("id")))

        for attr in (f"{{{XLINK_NS}}}href", "href"):
            value = element.get(attr)
            if value and value.startswith("#"):
                element.set(attr, "#" + rename(value[1:]))

        for attr in _URL_ATTRS:
            value = element.get(attr)
            if value and "url(#" in value:
                element.set(
                    attr, _URL_RE.sub(lambda m: f"url(#{rename(m.group(1))})", value)
                )


def _panel_group(panel: Panel, index: int) -> tuple[ET.Element, float]:
    """Return the placed panel as an SVG group plus its width in mm."""
    tree = ET.parse(panel.path)
    root = tree.getroot()
    width_pt, height_pt = _panel_size_pt(root)

    scale = panel.height_mm / height_pt  # mm per pt of the source figure
    width_mm = width_pt * scale

    _prefix_ids(root, f"p{index}-")

    group = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": f"panel{index}",
            "transform": (
                f"matrix({scale},0,0,{scale},{panel.x_mm},{panel.y_mm})"
            ),
        },
    )
    group.extend(list(root))
    return group, width_mm


def _label(text: str, x_mm: float, y_mm: float, font_mm: float) -> ET.Element:
    element = ET.Element(
        f"{{{SVG_NS}}}text",
        {
            "x": f"{x_mm}",
            "y": f"{y_mm}",
            "style": (
                f"font-family:DejaVu Sans,sans-serif;font-weight:bold;"
                f"font-size:{font_mm}px;fill:#000000"
            ),
        },
    )
    element.text = text
    return element


def compose(
    panels: list[Panel],
    labels: list[tuple[str, float, float]],
    canvas_height_mm: float,
    canvas_margin_mm: float,
    label_font_mm: float,
    out_svg: str,
) -> None:
    """Write the composed SVG. The canvas width follows from the placed panels."""
    groups = []
    canvas_width_mm = 0.0
    for index, panel in enumerate(panels, start=1):
        group, width_mm = _panel_group(panel, index)
        groups.append(group)
        canvas_width_mm = max(canvas_width_mm, panel.x_mm + width_mm)
    canvas_width_mm += canvas_margin_mm

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{canvas_width_mm}mm",
            "height": f"{canvas_height_mm}mm",
            "viewBox": f"0 0 {canvas_width_mm} {canvas_height_mm}",
            "version": "1.1",
        },
    )
    background = ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{canvas_width_mm}",
            "height": f"{canvas_height_mm}",
            "style": "fill:#ffffff",
            "id": "background",
        },
    )
    background.tail = "\n"
    for group in groups:
        group.tail = "\n"
        root.append(group)
    for text, x_mm, y_mm in labels:
        element = _label(text, x_mm, y_mm, label_font_mm)
        element.tail = "\n"
        root.append(element)

    ET.ElementTree(root).write(out_svg, encoding="utf-8", xml_declaration=True)
    print(f"\u2714 saved {out_svg}")


def rasterise(svg_path: str, jpg_path: str, dpi: float, quality: int = 95) -> None:
    """Render the composed SVG to JPG (vector source, so any DPI is sharp)."""
    try:
        import pymupdf
    except ImportError:  # older wheels only expose the deprecated name
        import fitz as pymupdf
    from PIL import Image

    zoom = dpi / 72.0  # the page rectangle is in pt, whatever the SVG units are
    with pymupdf.open(svg_path) as document:
        pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    image.save(jpg_path, quality=quality)
    print(f"\u2714 saved {jpg_path} ({image.width}\u00d7{image.height} px)")
