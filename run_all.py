"""Regenerate every result table and figure of the combined-cycle study, in dependency order.

Usage:
    python run_all.py           full run, including the direct-model reference GA (~3 min extra)
    python run_all.py --quick   skip the direct-model reference GA
"""

import argparse
import runpy
import sys
import time
import warnings

from src.props.gas import A6ExtrapolationWarning


def _module(name):
    return lambda: runpy.run_module(name, run_name="__main__", alter_sys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the direct-model reference GA")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore", A6ExtrapolationWarning)

    from src import optimize, pareto, sensitivity

    steps = [
        ("validation (Check-1/2/3, Cengel, combined table)", _module("src.validation")),
        ("gas turbine eta vs rp study", _module("src.gas_turbine")),
        ("baseline combined cycle", _module("src.combined_cycle")),
        ("schematic", _module("src.schematic")),
        ("DOE (500-point LHS)", _module("src.doe")),
        ("surrogates (RSM + ANN)", _module("src.surrogates")),
        ("optimisation (surrogate GA + verification)", lambda: optimize.main(direct_reference=not args.quick)),
        ("sensitivity (HRSG limit, levers)", sensitivity.main),
        ("Pareto fronts", pareto.main),
    ]
    timings = []
    start = time.perf_counter()
    for name, step in steps:
        print(f"\n{'=' * 100}\n{name}\n{'=' * 100}", flush=True)
        t0 = time.perf_counter()
        step()
        timings.append((name, time.perf_counter() - t0))
    total = time.perf_counter() - start

    print(f"\n{'=' * 100}\nTiming\n{'=' * 100}")
    for name, seconds in timings:
        print(f"{name:<55s} {seconds:7.1f} s")
    print(f"{'total':<55s} {total:7.1f} s")


if __name__ == "__main__":
    main()
