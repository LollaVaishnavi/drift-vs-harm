"""Fail the build if line coverage is below a floor.

A tiny script rather than a plugin flag, so the floor is visible in the Makefile and
the CI log and cannot be silently relaxed by a buried config key. Reads the coverage
json pytest-cov already wrote; does not re-run anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_coverage.py <coverage.json> <floor_percent>", file=sys.stderr)
        return 2
    report_path = Path(argv[0])
    floor = float(argv[1])
    data = json.loads(report_path.read_text())
    percent = data["totals"]["percent_covered"]
    print(f"line coverage: {percent:.2f}% (floor {floor:.0f}%)")
    if percent + 1e-9 < floor:
        print(f"FAIL: coverage {percent:.2f}% is below the floor {floor:.0f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
