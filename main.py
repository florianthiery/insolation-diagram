"""Build all figures of this repository.

Runs ``prepare_data.py`` once (derives the CSV from the .tab file), then per age
range and in this order:

1. ``plot_insolation.py``   -> panel (A)
2. ``plot_correlation.py``  -> panels (B)-(D)
3. ``plot_overall.py``      -> the composed multi-panel figure

Usage
-----
    python main.py                 # both age ranges
    python main.py --115-250       # only 115-250 ka b2k, 65 deg N
    python main.py --128-220       # only 128-220 ka b2k, 65 deg N
    python main.py --115-250-WM    # only the Walsdorfer Maar latitude
    python main.py --list          # show what would be run
    python main.py --no-data       # skip the CSV derivation step
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Run once in the repository root, before the age ranges
ROOT_SCRIPTS = ("prepare_data.py",)

# Executed in this order - plot_overall.py composes the output of the other two
SCRIPTS = (
    "plot_insolation.py",
    "plot_correlation.py",
    "plot_overall.py",
)

# One directory per age range / latitude, with the scripts it needs.
# 115-250-WM recomputes the insolation for the Walsdorfer Maar latitude first
# and adds a comparison against the 65 deg N reference curve at the end.
RANGES = {
    "115-250": SCRIPTS,
    "128-220": SCRIPTS,
    "115-250-WM": ("prepare_data_wm.py",) + SCRIPTS + ("plot_compare_latitudes.py",),
}


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_script(script: str, range_name: str = "") -> None:
    """Run one script in its own directory, streaming its output."""
    path = ROOT / range_name / script if range_name else ROOT / script
    title = f"{range_name} / {script}" if range_name else script
    if not path.exists():
        raise SystemExit(f"\u2717 not found: {path.relative_to(ROOT)}")

    print(f"\n=== {title} " + "=" * max(3, 46 - len(title)))
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(path)], cwd=path.parent)
    if result.returncode != 0:
        raise SystemExit(
            f"\u2717 {title} failed with exit code {result.returncode}"
        )
    print(f"  ({time.perf_counter() - started:.1f} s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for range_name in RANGES:
        parser.add_argument(
            f"--{range_name}",
            dest=range_name.replace("-", "_"),
            action="store_true",
            help=f"build only the {range_name} ka b2k figures",
        )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="skip prepare_data.py and use the existing derived CSV",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the scripts that would be run and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = [r for r in RANGES if getattr(args, r.replace("-", "_"))]
    if not selected:  # no flag given -> everything
        selected = list(RANGES)

    root_scripts = () if args.no_data else ROOT_SCRIPTS

    if args.list:
        for script in root_scripts:
            print(script)
        for range_name in selected:
            for script in RANGES[range_name]:
                print(f"{range_name}/{script}")
        return

    started = time.perf_counter()
    for script in root_scripts:
        run_script(script)
    for range_name in selected:
        for script in RANGES[range_name]:
            run_script(script, range_name)

    print(
        f"\n\u2714 done: {', '.join(selected)} "
        f"({time.perf_counter() - started:.1f} s total)"
    )


if __name__ == "__main__":
    main()
