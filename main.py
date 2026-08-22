"""Build all figures of this repository.

Runs, per age range and in this order:

1. ``plot_insolation.py``   -> panel (A)
2. ``plot_correlation.py``  -> panels (B)-(D)
3. ``plot_overall.py``      -> the composed multi-panel figure

Usage
-----
    python main.py                 # both age ranges
    python main.py --115-250       # only 115-250 ka b2k
    python main.py --128-220       # only 128-220 ka b2k
    python main.py --list          # show what would be run
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

# Age ranges in ka b2k; each is a directory in the repository root
RANGES = ("115-250", "128-220")

# Executed in this order - plot_overall.py composes the output of the other two
SCRIPTS = (
    "plot_insolation.py",
    "plot_correlation.py",
    "plot_overall.py",
)


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_script(range_name: str, script: str) -> None:
    """Run one script in its own directory, streaming its output."""
    path = ROOT / range_name / script
    if not path.exists():
        raise SystemExit(f"\u2717 not found: {path.relative_to(ROOT)}")

    print(f"\n=== {range_name} / {script} " + "=" * (46 - len(range_name) - len(script)))
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(path)], cwd=path.parent)
    if result.returncode != 0:
        raise SystemExit(
            f"\u2717 {range_name}/{script} failed with exit code {result.returncode}"
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

    if args.list:
        for range_name in selected:
            for script in SCRIPTS:
                print(f"{range_name}/{script}")
        return

    started = time.perf_counter()
    for range_name in selected:
        for script in SCRIPTS:
            run_script(range_name, script)

    print(
        f"\n\u2714 done: {', '.join(selected)} "
        f"({time.perf_counter() - started:.1f} s total)"
    )


if __name__ == "__main__":
    main()
