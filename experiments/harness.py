"""Shared plumbing for the five experiments.

Each experiment answers one question the README asks and writes its findings to
docs/experiments/<name>.json. They all run the same seeded suite, so the baseline
is built once here and reused. Keeping the write and print helpers in one place
means every experiment produces the same JSON shape, which is what the receipts
tooling reads back.
"""

from __future__ import annotations

import json
from pathlib import Path

from driftharm.config import load_policy
from driftharm.harness import build_baseline, run_suite
from driftharm.scenarios import default_suite

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "experiments"

_CACHE: dict[str, object] = {}


def suite_outcomes():
    """Run (once) and return the default suite under the bundled policy."""
    if "outcomes" not in _CACHE:
        policy = load_policy()
        baseline = build_baseline(policy)
        _CACHE["policy"] = policy
        _CACHE["baseline"] = baseline
        _CACHE["outcomes"] = run_suite(policy, default_suite(), baseline=baseline)
    return _CACHE["outcomes"]


def policy():
    suite_outcomes()
    return _CACHE["policy"]


def baseline():
    suite_outcomes()
    return _CACHE["baseline"]


def write_json(name: str, payload: dict) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def print_table(title: str, rows: list[tuple], headers: tuple) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(title)
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 2)
