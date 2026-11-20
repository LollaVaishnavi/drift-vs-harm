"""Record a replay of the real CLI, paced by each command's measured wall time.

Every line of terminal text is captured from a command that actually ran: the script
runs each command, records its combined output, exit code and duration, then drives an
HTML terminal that types the command and reveals its captured output paced by that
duration, and records the page to webm before transcoding to an MP4 and an autoplaying
GIF. Nothing is hand typed, so the demo cannot drift from the tool. The import is warmed
by a setup command before timing begins, so the first segment does not pay for it.

Order of segments is the argument, not the feature list: the baseline where nothing
changed, the headline silent failure, the false alarms no one wants, the whole suite,
then the receipts. Dev-time only; needs playwright, chromium and ffmpeg.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "docs" / "video"
MANIFEST = VIDEO_DIR / "manifest.json"
GIF = VIDEO_DIR / "demo.gif"
MP4 = VIDEO_DIR / "demo.mp4"

PY = [sys.executable, "-m", "driftharm.cli"]
SEGMENTS = [
    (["python", "-m", "driftharm.cli", "audit", "--family", "null", "--quiet"],
     "baseline: nothing changed, so nothing should fire"),
    (["python", "-m", "driftharm.cli", "audit", "--family", "concept", "--quiet"],
     "concept drift: real harm, zero drift alarms"),
    (["python", "-m", "driftharm.cli", "audit", "--family", "covariate", "--quiet"],
     "covariate shift: loud alarms, no harm"),
    (["python", "-m", "driftharm.cli", "audit", "--quiet"],
     "the whole suite: a silent-failure exit code"),
    (["python", "tools/check_numbers.py"],
     "receipts: every figure matches the code"),
]


def _display(command: list[str]) -> str:
    """How the command reads in the terminal: the console script, not python -m."""
    if "driftharm.cli" in command:
        tail = command[command.index("driftharm.cli") + 1:]
        return "driftharm " + " ".join(tail)
    return " ".join(command)


def _capture(command: list[str]) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    duration = time.perf_counter() - start
    output = (proc.stdout + proc.stderr).rstrip("\n")
    return {
        "command": _display(command),
        "output": output,
        "exit_code": proc.returncode,
        "duration_s": round(duration, 3),
    }


def _player_html(segments: list[dict]) -> str:
    data = json.dumps(segments)
    return """<!DOCTYPE html><html><head><meta charset=utf-8><style>
  body{margin:0;background:#0d1117}
  #term{font-family:'DejaVu Sans Mono',monospace;font-size:14px;line-height:1.45;
    color:#c9d1d9;padding:18px;white-space:pre-wrap}
  .cmd{color:#7ee787}.note{color:#8b949e}.exit{color:#ff7b72}
</style></head><body><div id=term></div><script>
const SEG = __DATA__;
const term = document.getElementById('term');
function line(cls, text){const d=document.createElement('div'); if(cls)d.className=cls;
  d.textContent=text; term.appendChild(d); window.scrollTo(0,document.body.scrollHeight);}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
async function run(){
  for(const s of SEG){
    line('note', '# '+s.note);
    line('cmd', '$ '+s.command);
    const pace = Math.min(2500, Math.max(400, s.duration_s*1000));
    await sleep(pace);
    for(const l of s.output.split('\\n')) line('', l);
    line('exit', 'exit '+s.exit_code);
    line('', '');
    await sleep(1200);
  }
  window.__DONE__ = true;
}
run();
</script></body></html>""".replace("__DATA__", data)


def main() -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    # Warm the import so the first timed command does not pay for it.
    subprocess.run([*PY, "--version"], cwd=ROOT, capture_output=True, text=True)

    segments = [dict(_capture(cmd), note=note) for cmd, note in SEGMENTS]
    player = VIDEO_DIR / "_player.html"
    player.write_text(_player_html(segments))

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 860, "height": 560},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 860, "height": 560},
        )
        page = context.new_page()
        page.goto(player.resolve().as_uri())
        page.wait_for_function("() => window.__DONE__ === true", timeout=60000)
        page.wait_for_timeout(600)
        video_path = page.video.path()
        context.close()
        browser.close()

    webm = Path(video_path)
    # MP4 for quality.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-movflags", "+faststart",
         "-pix_fmt", "yuv420p", "-vf", "scale=860:-2", str(MP4)],
        check=True, capture_output=True,
    )
    # Two-pass palette GIF, kept small.
    palette = VIDEO_DIR / "_palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-vf", "fps=11,scale=720:-1:flags=lanczos,palettegen",
         str(palette)], check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-i", str(palette),
         "-lavfi", "fps=11,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse", str(GIF)],
        check=True, capture_output=True,
    )
    palette.unlink(missing_ok=True)
    player.unlink(missing_ok=True)
    webm.unlink(missing_ok=True)

    MANIFEST.write_text(json.dumps({
        "segments": [
            {"command": s["command"], "exit_code": s["exit_code"], "duration_s": s["duration_s"]}
            for s in segments
        ],
        "gif": str(GIF.relative_to(ROOT)),
        "mp4": str(MP4.relative_to(ROOT)),
        "note": "every line is captured stdout from a command that ran; pacing is measured",
    }, indent=2) + "\n")
    size_mb = GIF.stat().st_size / 1e6
    print(f"wrote {GIF.relative_to(ROOT)} ({size_mb:.1f} MB) and {MP4.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
