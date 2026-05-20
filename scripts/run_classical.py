"""Run the classical computer-vision demos for one or all course units.

Every demo writes its result figures to ``reports/figures/classical/``. These
figures are what the written report embeds as evidence of coverage.

Examples
--------
    python -m scripts.run_classical --unit all      # everything (default)
    python -m scripts.run_classical --unit 2        # just Unit 2
    python -m scripts.run_classical --unit 3 5 7    # a custom selection
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classical import (unit1_image_basics, unit2_features,  # noqa: E402
                           unit2_hough_ransac, unit2_sampling, unit3_stereo,
                           unit3_transforms, unit4_model_fitting,
                           unit5_learning, unit6_motion, unit7_representation)

# Each unit maps to one or more demo modules, run in order.
UNITS = {
    "1": [unit1_image_basics],
    "2": [unit2_sampling, unit2_features, unit2_hough_ransac],
    "3": [unit3_transforms, unit3_stereo],
    "4": [unit4_model_fitting],
    "5": [unit5_learning],
    "6": [unit6_motion],
    "7": [unit7_representation],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", nargs="+", default=["all"],
                        choices=["all", *UNITS.keys()],
                        help="which course unit(s) to run (default: all)")
    args = parser.parse_args()

    selected = list(UNITS.keys()) if "all" in args.unit else sorted(set(args.unit))
    print(f"Running classical CV demos for unit(s): {', '.join(selected)}\n")

    start = time.time()
    n_modules = 0
    for unit in selected:
        for module in UNITS[unit]:
            module.demo()
            n_modules += 1
            print()
    elapsed = time.time() - start
    print(f"Done — {n_modules} demo module(s) in {elapsed:.1f}s.")
    print("Figures written to reports/figures/classical/")


if __name__ == "__main__":
    main()
