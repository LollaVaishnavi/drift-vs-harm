"""Render the README's diagram source to a committed SVG.

GitHub will not reliably render a mermaid fence (its lazily loaded renderer throws on
some diagrams), so the diagram is rendered here to an SVG, committed, and embedded in
the README with <img src>. Three fixes from past builds are applied, each of which
otherwise produces a broken image with nothing in the console to explain it:

  - htmlLabels off, so labels are <text>/<tspan> and the file is valid XML that loads
    as <img src> (html labels wrap unclosed <br> in a foreignObject and fail to load);
  - intrinsic width and height written onto the root from the viewBox, because mermaid
    emits width="100%" with a max-width style, which has no aspect ratio as an image;
  - a painted light background rectangle, because transparent comes out dark-on-dark
    for edge and subgraph labels in a dark theme.

This is a dev-time tool. It is not part of `make verify`, and CI never runs it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT_SVG = ROOT / "docs" / "diagrams" / "architecture.svg"
MANIFEST = ROOT / "docs" / "diagrams" / "manifest.json"
BACKGROUND = "#f6f8fa"


def extract_source(markdown: str) -> str:
    match = re.search(r"```mermaid-source\n(.*?)```", markdown, flags=re.DOTALL)
    if not match:
        raise SystemExit("no ```mermaid-source fence found in README.md")
    return match.group(1).strip()


def _run_mmdc(source: str, out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mmd = tmp_path / "diagram.mmd"
        mmd.write_text(source + "\n")
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "htmlLabels": False,
            "flowchart": {"htmlLabels": False, "useMaxWidth": False},
        }))
        puppeteer = tmp_path / "puppeteer.json"
        puppeteer.write_text(json.dumps({"args": ["--no-sandbox"]}))
        subprocess.run(
            [
                "mmdc", "-i", str(mmd), "-o", str(out_path),
                "-c", str(config), "-p", str(puppeteer),
                "-b", "transparent",
            ],
            check=True,
        )


def _give_intrinsic_size_and_background(svg: str) -> str:
    view = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not view:
        raise SystemExit("rendered SVG has no viewBox to size from")
    width, height = view.group(1), view.group(2)

    # Replace the percentage width and max-width style with explicit dimensions.
    svg = re.sub(r'\swidth="100%"', f' width="{width}" height="{height}"', svg, count=1)
    svg = re.sub(r'style="[^"]*max-width:[^"]*"', "", svg, count=1)

    # Insert a painted background rectangle as the first child of the root <svg>.
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BACKGROUND}"/>'
    svg = re.sub(r"(<svg[^>]*>)", r"\1" + rect, svg, count=1)
    return svg


def main() -> int:
    source = extract_source(README.read_text())
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    _run_mmdc(source, OUT_SVG)

    svg = OUT_SVG.read_text()
    svg = _give_intrinsic_size_and_background(svg)
    OUT_SVG.write_text(svg)

    if "foreignObject" in svg:
        raise SystemExit("SVG still contains foreignObject: htmlLabels were not disabled")

    MANIFEST.write_text(json.dumps({
        "source": "README.md ```mermaid-source fence",
        "output": str(OUT_SVG.relative_to(ROOT)),
        "background": BACKGROUND,
        "html_labels": False,
    }, indent=2) + "\n")
    print(f"wrote {OUT_SVG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
