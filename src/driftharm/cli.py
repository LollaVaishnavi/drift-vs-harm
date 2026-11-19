"""Command line entry point.

The exit codes are the interface a CI job or a cron monitor reads, so they are
deliberate and documented. The distinction that matters is between "the tool did its
job and the answer is that your monitor is blind" (a silent failure exists) and "the
tool could not answer" (no scenario had measurable harm). Collapsing those two into
one non-zero code is exactly the mistake this project is about, so they are kept
apart.

Exit codes:
  0   TRACKED        every harmed scenario also raised a drift alarm in this run
  10  SILENT_FAILURE at least one harmed scenario raised no drift alarm
  20  NOT_MEASURABLE no scenario had measurable harm, so nothing could be judged
  2   usage error    (argparse default for bad arguments)
"""

from __future__ import annotations

import argparse
import sys

from driftharm import __version__
from driftharm.config import load_policy
from driftharm.report import render_json, render_summary, render_table
from driftharm.scenarios import default_suite, suite_by_family

EXIT_TRACKED = 0
EXIT_SILENT_FAILURE = 10
EXIT_NOT_MEASURABLE = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftharm",
        description=(
            "Replay labelled shift scenarios and report whether a population-drift "
            "alarm actually predicts model harm."
        ),
    )
    parser.add_argument("--version", action="version", version=f"driftharm {__version__}")
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit", help="run the scenario suite and report the drift-vs-harm gap")
    audit.add_argument("--policy", default=None, help="path to a policy.toml (default: bundled)")
    audit.add_argument(
        "--family",
        default="all",
        choices=["all", "null", "covariate", "concept", "mixed"],
        help="restrict to one ground-truth family",
    )
    audit.add_argument(
        "--format",
        default="table",
        choices=["table", "json"],
        help="table for humans, json for the receipts tooling",
    )
    return parser


def audit_exit_code(outcomes) -> int:
    """Map a suite run to an exit code, keeping 'answer is no' apart from 'no answer'."""
    measurable = [o for o in outcomes if o.measurable]
    if not measurable:
        return EXIT_NOT_MEASURABLE
    if any(o.silent_failure for o in measurable):
        return EXIT_SILENT_FAILURE
    return EXIT_TRACKED


def run_audit(policy_path: str | None, family: str, output_format: str, stream) -> int:
    policy = load_policy(policy_path)
    specs = default_suite()
    if family != "all":
        specs = suite_by_family(specs).get(family, [])

    from driftharm.harness import run_suite  # local import keeps --version fast

    outcomes = run_suite(policy, specs)

    if output_format == "json":
        print(render_json(outcomes), file=stream)
    else:
        print(render_table(outcomes), file=stream)
        print(render_summary(outcomes), file=stream)

    return audit_exit_code(outcomes)


def main(argv: list[str] | None = None, stream=None) -> int:
    stream = stream if stream is not None else sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(stream)
        return EXIT_TRACKED
    if args.command == "audit":
        return run_audit(args.policy, args.family, args.format, stream)
    parser.print_help(stream)  # pragma: no cover - argparse rejects unknown commands first
    return EXIT_TRACKED  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
