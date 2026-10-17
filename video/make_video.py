"""Render the 1-minute demo video (silent, 1920x1080) from recorded runs and results.json.

  python video/make_video.py            -> video/holdout_demo.mp4 + video/timeline.json

Frames are real `rich` renders of demo.py's State (the same code the live terminal demo uses), exported to SVG,
rasterized with rsvg-convert, and stitched with ffmpeg. Every number on screen is read from results/ and runs/.
Requires: rsvg-convert, magick (ImageMagick), ffmpeg.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.align import Align  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.text import Text  # noqa: E402

import demo  # noqa: E402

W, H = 150, 42
OUT = ROOT / "video"
FR = OUT / "frames"
REPLAY_RUN = "openshell_r1"
BG = "#0c0c0c"


def console() -> Console:
    return Console(record=True, width=W, height=H, file=io.StringIO(), force_terminal=True, color_system="truecolor")


def svg_frame(renderable, name: str, title: str = "HOLDOUT: Shake-Up") -> Path:
    c = console()
    c.print(renderable)
    svg = FR / f"{name}.svg"
    svg.write_text(c.export_svg(title=title))
    png = FR / f"{name}.raw.png"
    subprocess.run(["rsvg-convert", "-w", "1880", str(svg), "-o", str(png)], check=True)
    out = FR / f"{name}.png"
    subprocess.run(["magick", str(png), "-resize", "1880x1040", "-background", BG, "-gravity", "center",
                    "-extent", "1920x1080", str(out)], check=True)
    return out


def image_frame(src: Path, name: str, caption: str) -> Path:
    out = FR / f"{name}.png"
    subprocess.run(["magick", str(src), "-resize", "1800x930", "-background", "white", "-gravity", "center",
                    "-extent", "1920x1000", "-gravity", "south", "-background", "white", "-splice", "0x80",
                    "-font", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "-pointsize", "38", "-fill", "#222222", "-gravity", "north",
                    "-annotate", "+0+28", caption, str(out)], check=True)
    return out


def main():
    if FR.exists():
        shutil.rmtree(FR)
    FR.mkdir(parents=True)
    res = json.loads((ROOT / "results" / "results.json").read_text())
    b = res["boosting_bot"]
    kb, lb = b["boards"]["kaggle"], b["boards"]["ladder"]
    frames: list[tuple[Path, float, str]] = []  # (png, seconds, scene)

    # ---- scene 1: title card (0:00-0:10)
    title = Group(
        Text(" "), Text("HOLDOUT: Shake-Up", style="bold white", justify="center"), Text(" "),
        Text("Self-improving agents keep a change when their score goes up.", style="italic", justify="center"),
        Text("That is how Kaggle teams overfit the public leaderboard,", style="italic", justify="center"),
        Text("and how agent teams learn the scoreboard instead of the task.", style="italic", justify="center"),
        Text(" "),
        Text("An AI Kaggle team on Crusoe Managed Inference · all NVIDIA Nemotron · code sandboxed in NVIDIA OpenShell",
             style="cyan", justify="center"),
        Text("Built today at The Recursive Agents Hackathon (Crusoe x Cerebral Valley)", style="dim", justify="center"),
        Text(" "))
    frames.append((svg_frame(Panel(Align.center(title, vertical="middle"), border_style="blue", height=H - 2), "s1_title"), 10.0, "title"))

    # ---- scene 2: boosting calibration (0:10-0:17)
    frames.append((image_frame(ROOT / "results" / "boosting_bot.png", "s2_boost",
                               f"Zero-skill random guesses: Kaggle board reports {kb['reported_public']['mean'][-1]:.2f}, "
                               f"the Ladder {lb['reported_public']['mean'][-1]:.2f} (private stays ~0.5)"), 7.0, "boosting"))

    # ---- scene 3: replay of the OpenShell round (0:17-0:40)
    events = [json.loads(l) for l in open(ROOT / "runs" / REPLAY_RUN / "events.jsonl")]
    comp = json.loads((ROOT / "runs" / REPLAY_RUN / "arena.json").read_text())["comps"][0]
    st = demo.State(comp)
    header = "recorded round · every agent solution executed inside NVIDIA OpenShell"
    durations = {"round_start": 0.0, "submission": 0.7, "playbook": 2.2, "final": 0.7}
    i = 0
    for ev in events:
        if st.process(ev) and durations.get(ev["type"], 0) > 0:
            frames.append((svg_frame(st.render(header), f"s3_{i:03d}"), durations[ev["type"]], "replay"))
            i += 1
    # keep the replay at 17 s so the reveal holds for ~6 s
    used = sum(d for _, d, s in frames if s == "replay")
    if used < 17.0:
        p, d, s = frames[-1]
        frames[-1] = (p, d + (17.0 - used), s)
    # reveal
    st.revealed = True
    st.log(Text("REVEAL PRIVATE: re-sorting both boards by the sealed private score", style="bold white on blue"))
    for arm in ("naive", "holdout"):
        f = st.final.get(arm)
        if f and f.get("id"):
            st.log(Text(f"{arm:7s} pick {f['id']}: reported {f['reported']:.3f} -> private {f['private']:.3f} "
                        f"(inflation {f['reported'] - f['private']:+.3f})", style=f"bold {demo.ARM_STYLE[arm]}"))
    replay_total = sum(d for _, d, s in frames if s == "replay")
    frames.append((svg_frame(st.render(header + "  ·  PRIVATE REVEALED"), "s3_reveal"), 40.0 - 17.0 - replay_total, "reveal"))

    # ---- scene 4: transfer (0:40-0:52)
    t = res["transfer"]
    ran = t["submissions_ran"]
    frames.append((image_frame(ROOT / "results" / "transfer.png", "s4_transfer",
                               f"Unseen competitions: code that ran  gen-0 {ran['gen-0'][0]}/{ran['gen-0'][1]} · "
                               f"naive-final {ran['naive-final'][0]}/{ran['naive-final'][1]} · "
                               f"HOLDOUT-final {ran['holdout-final'][0]}/{ran['holdout-final'][1]}"), 12.0, "transfer"))

    # ---- scene 5: close (0:52-1:00)
    gp = res.get("gate_pooled", {})
    e = res["efficiency"]
    close = Group(
        Text(" "), Text("Everyone builds the optimizer. We built the thing that catches it lying.", style="bold white", justify="center"),
        Text(" "),
        Text(f"Gate: naive curator admitted {gp['naive']['admitted']}/{gp['naive']['proposed']} rules · "
             f"HOLDOUT admitted {gp['holdout']['admitted']}/{gp['holdout']['proposed']}", justify="center"),
        Text(f"Boosting attack: Kaggle board {kb['reported_public']['mean'][-1]:.3f} vs Ladder {lb['reported_public']['mean'][-1]:.3f} "
             f"(private {kb['private']['mean'][-1]:.3f} / {lb['private']['mean'][-1]:.3f})", justify="center"),
        Text(f"Full two-arm run on Crusoe: ${e['arena_run']['usd']:.2f} · {e['arena_run']['calls']} Nemotron calls", justify="center"),
        Text(" "),
        Text("github.com/himavanthkar/crusoehackv1ourvers", style="bold cyan", justify="center"),
        Text("HOLDOUT · Crusoe Managed Inference · NVIDIA Nemotron · NVIDIA OpenShell", style="dim", justify="center"),
        Text(" "))
    frames.append((svg_frame(Panel(Align.center(close, vertical="middle"), border_style="green", height=H - 2), "s5_close"), 8.0, "close"))

    # ---- stitch
    lst = FR / "list.txt"
    with open(lst, "w") as f:
        for p, d, _ in frames:
            f.write(f"file '{p.name}'\nduration {d:.3f}\n")
        f.write(f"file '{frames[-1][0].name}'\n")
    out = OUT / "holdout_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "fps=30,format=yuv420p", "-t", "60", "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", str(out)],
                   check=True, cwd=FR)
    # timeline for the voiceover script
    tl, t0 = [], 0.0
    for p, d, s in frames:
        if tl and tl[-1]["scene"] == s:
            tl[-1]["end"] = round(t0 + d, 2)
        else:
            tl.append({"scene": s, "start": round(t0, 2), "end": round(t0 + d, 2)})
        t0 += d
    (OUT / "timeline.json").write_text(json.dumps({"total_s": round(t0, 2), "scenes": tl}, indent=1))
    print(json.dumps({"total_s": round(t0, 2), "scenes": tl}))
    print(out)


if __name__ == "__main__":
    main()
