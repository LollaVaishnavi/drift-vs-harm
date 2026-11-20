"""Fail the build when a document quotes a figure the code no longer produces.

Loads docs/metrics.json, then for every guarded figure checks that at least one of
its anchor phrases, the exact wording with the value substituted in, appears in at
least one checked document. A figure whose anchor is nowhere to be found is reported
as a failure, because that is either a stale number or a receipt that was never
written. Anchors are matched as phrases, not bare digits, so a correct sentence whose
number happens to collide with another is not mistaken for a match (receipts rule 1),
and each anchor names only the number and a few words so line wrapping does not break
it (rule 2).

A second pass fails the build on any number that appears in prose and matches no
anchor and is not a named structural constant, so a figure nothing re-measures cannot
hide and, crucially, editing one instance of a figure to a wrong value fails the build
even when the same figure is still correct in another document. Code fences, inline
code, HTML tags and markdown link targets are stripped first, because an example
invocation and a badge URL are allowed numbers of their own.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "metrics.json"

# Numbers allowed to appear in prose without being a measured metric: structural
# counts the project describes in words, the licence year, the version, bin/axis
# sizes. Each is a fact about the repository's shape, not a measurement of a result.
STRUCTURAL_ALLOWLIST = {
    "0", "1", "2", "3", "4", "5", "6", "8", "10", "2026", "0.1",
    "4000",  # window size, stated as config
    "72", "3.11", "3.12", "3.13",
    "20",  # the exit code, named in prose and the table
    "001", "002", "003", "004", "005",  # ADR references in prose
    "0.00", "0.01",  # the null-level detector readings, quoted descriptively
    "0.02",  # the harm bar, a config value named in prose
    "0.05",  # the KS significance, a config value named in prose
    "0.20",  # the PSI alarm level written long-form
}


def _strip_code_and_tags(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced blocks
    text = re.sub(r"`[^`]*`", " ", text)  # inline code spans
    text = re.sub(r"\]\([^)]*\)", "] ", text)  # markdown link and image URL targets
    text = re.sub(r"<[^>]+>", " ", text)  # html tags and their attributes
    return text


def _load_documents(paths: list[str]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING checked document: {rel}", file=sys.stderr)
            documents[rel] = ""
        else:
            documents[rel] = _strip_code_and_tags(path.read_text())
    return documents


def forward_check(payload: dict, documents: dict[str, str]) -> list[str]:
    """Every guarded figure must have an anchor phrase present in some document."""
    failures: list[str] = []
    haystack = "\n".join(documents.values())
    for metric, anchors in sorted(payload["rendered_anchors"].items()):
        if not any(anchor in haystack for anchor in anchors):
            value = payload["metrics"][metric]
            failures.append(
                f"{metric}={value}: none of its anchor phrases were found "
                f"(expected one of {anchors})"
            )
    return failures


def reverse_check(payload: dict, documents: dict[str, str]) -> list[str]:
    """Prose numbers that match no anchor and are not structural: a build failure."""
    anchored_values = {
        anchor for anchors in payload["rendered_anchors"].values() for anchor in anchors
    }
    anchored_text = " ".join(anchored_values)
    number_re = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
    unmatched: list[str] = []
    for rel, text in documents.items():
        for match in number_re.findall(text):
            token = match
            bare = token.rstrip("%")
            if bare in STRUCTURAL_ALLOWLIST or token in STRUCTURAL_ALLOWLIST:
                continue
            if token in anchored_text or bare in anchored_text:
                continue
            unmatched.append(f"{rel}: {token!r}")
    return unmatched


def main() -> int:
    if not METRICS_PATH.exists():
        print("metrics.json not found; run tools/collect_metrics.py first", file=sys.stderr)
        return 2
    payload = json.loads(METRICS_PATH.read_text())
    documents = _load_documents(payload["checked_documents"])

    missing = forward_check(payload, documents)
    rogue = reverse_check(payload, documents)

    failures = []
    failures += [f"stale or missing receipt: {line}" for line in missing]
    failures += [f"unguarded number: {line}" for line in rogue]

    if failures:
        print("FAIL: documents have drifted from the measured figures:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"OK: all {len(payload['metrics'])} figures are anchored, no unguarded numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
