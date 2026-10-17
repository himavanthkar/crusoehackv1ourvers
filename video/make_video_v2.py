"""Demo video v2 ("a shared brain that only learns proven lessons"), with narration.

  python video/make_video_v2.py      -> video/holdout_demo_v2.mp4 (+ _silent.mp4, voiceover_v2.txt)

Narration is macOS `say` (Samantha). Each scene lasts as long as its line plus a short pause, so audio and video stay
in sync. Frames are real `rich` renders of demo.py from runs/openshell_r1 plus charts from results/. Every number is
read from results/results.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "video"))

from rich import box  # noqa: E402
from rich.align import Align  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

import demo  # noqa: E402
import make_video as mv  # noqa: E402

OUT = ROOT / "video"
FR = OUT / "frames_v2"
VOICE, RATE = "Samantha", 180


def dur(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True).stdout)


def tts(i: int, text: str) -> tuple[Path, float]:
    p = FR / f"vo_{i}.aiff"
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(p), text], check=True)
    return p, dur(p)


def card(lines, color: str):
    return Panel(Align.center(Group(*lines), vertical="middle"), border_style=color, height=mv.H - 2)


def main():
    if FR.exists():
        shutil.rmtree(FR)
    FR.mkdir(parents=True)
    mv.FR = FR  # render helpers write into this folder
    r = json.loads((ROOT / "results" / "results.json").read_text())
    gp, t, b = r["gate_pooled"], r["transfer"], r["boosting_bot"]["boards"]
    ran = t["submissions_ran"]
    kb, lb = b["kaggle"]["reported_public"]["mean"][-1], b["ladder"]["reported_public"]["mean"][-1]
    sys.path.insert(0, str(ROOT))
    from holdout.report import gate_summary
    per_seed = {rid: gate_summary(json.loads((ROOT / "runs" / rid / "arena.json").read_text())) for rid in ("m2", "m2s1")}

    lines = [
        ("title", "Self-improving agents learn the scoreboard, not the task. HOLDOUT is a shared brain that only learns proven lessons."),
        ("replay", "An AI Kaggle team on Crusoe, all Nemotron. Researchers write code that runs inside NVIDIA OpenShell, "
                   "and a curator rewrites the team's playbook every round. HOLDOUT's Ladder rejects gains within noise, "
                   "and its gate refuses rules built on them."),
        ("gate", f"Across two seeds, the naive curator admitted all {gp['naive']['admitted']} rules it wrote. "
                 f"HOLDOUT's gate admitted {gp['holdout']['admitted']} of {gp['holdout']['proposed']}."),
        ("transfer", f"On datasets the team never saw, code from the naive playbook ran {ran['naive-final'][0]} times out of "
                     f"{ran['naive-final'][1]}. A rule backed mostly by noise broke it. HOLDOUT's ran {ran['holdout-final'][0]} "
                     f"out of {ran['holdout-final'][1]}."),
        ("boost", f"Against random guessing, a normal leaderboard reports {kb:.2f}. The Ladder holds it to {lb:.2f}."),
        ("close", "Final picks didn't differ at our scale. What changed is what the team learned. HOLDOUT, on Crusoe."),
    ]
    audio = {}
    for i, (scene, text) in enumerate(lines):
        audio[scene] = (*tts(i, text), text)
    seconds = {s: round(audio[s][1] + 0.8, 2) for s in audio}

    frames: list[tuple[Path, float, str]] = []
    # title
    frames.append((mv.svg_frame(card([
        Text("HOLDOUT", style="bold white", justify="center"),
        Text("a shared brain that only learns proven lessons", style="bold cyan", justify="center"), Text(" "),
        Text("Self-improving agents grade themselves on the scoreboard they optimize,", style="italic", justify="center"),
        Text("so they learn the scoreboard, not the task.", style="italic", justify="center"), Text(" "),
        Text("Crusoe Managed Inference · NVIDIA Nemotron · NVIDIA OpenShell", style="dim", justify="center")], "blue"),
        "a_title"), seconds["title"], "title"))
    # replay of the OpenShell round, paced to fill its narration
    events = [json.loads(l) for l in open(ROOT / "runs" / "openshell_r1" / "events.jsonl")]
    st = demo.State(json.loads((ROOT / "runs" / "openshell_r1" / "arena.json").read_text())["comps"][0])
    header = "every agent solution executed inside NVIDIA OpenShell"
    weights = {"submission": 1.0, "playbook": 3.5, "final": 1.0}
    shown = []
    for ev in events:
        if st.process(ev) and ev["type"] in weights:
            shown.append((mv.svg_frame(st.render(header), f"b_{len(shown):03d}"), weights[ev["type"]]))
    total_w = sum(w for _, w in shown)
    for p, w in shown:
        frames.append((p, seconds["replay"] * w / total_w, "replay"))
    # gate card
    tb = Table(box=box.SIMPLE_HEAD, title="What the shared brain was allowed to learn", title_style="bold white")
    tb.add_column("seed", justify="center")
    tb.add_column("NAIVE: proposed / admitted", justify="center", style="red")
    tb.add_column("HOLDOUT: proposed / admitted", justify="center", style="green")
    for seed, rid in ((0, "m2"), (1, "m2s1")):
        g = per_seed[rid]
        tb.add_row(str(seed), f"{g['naive']['proposed']} / {g['naive']['admitted']}", f"{g['holdout']['proposed']} / {g['holdout']['admitted']}")
    tb.add_row("total", f"{gp['naive']['proposed']} / {gp['naive']['admitted']}", f"{gp['holdout']['proposed']} / {gp['holdout']['admitted']}",
               style="bold")
    frames.append((mv.svg_frame(card([Align.center(tb), Text(" "),
                                      Text(f"The gate rejected {gp['holdout']['rejected']} rules and stripped {gp['holdout']['stripped']} "
                                           "citations to gains the Ladder did not accept.", justify="center")], "magenta"),
                                 "c_gate"), seconds["gate"], "gate"))
    # transfer chart
    frames.append((mv.image_frame(ROOT / "results" / "transfer.png", "d_transfer",
                                  f"Unseen datasets, code that ran: naive playbook {ran['naive-final'][0]}/{ran['naive-final'][1]} · "
                                  f"HOLDOUT playbook {ran['holdout-final'][0]}/{ran['holdout-final'][1]} · start {ran['gen-0'][0]}/{ran['gen-0'][1]}"),
                   seconds["transfer"], "transfer"))
    # boosting chart
    frames.append((mv.image_frame(ROOT / "results" / "boosting_bot.png", "e_boost",
                                  f"Zero-skill random guesses: normal board reports {kb:.2f}, the Ladder {lb:.2f}"), seconds["boost"], "boost"))
    # close
    frames.append((mv.svg_frame(card([
        Text("Final picks did not differ measurably at our scale.", justify="center"),
        Text("What changed is what the team learned, and that is what transferred.", style="bold white", justify="center"), Text(" "),
        Text("github.com/himavanthkar/crusoehackv1ourvers", style="bold cyan", justify="center"),
        Text("HOLDOUT · Crusoe Managed Inference · NVIDIA Nemotron · NVIDIA OpenShell", style="dim", justify="center")], "green"),
        "f_close"), seconds["close"], "close"))

    # stitch video
    lst = FR / "list.txt"
    with open(lst, "w") as f:
        for p, d, _ in frames:
            f.write(f"file '{p.name}'\nduration {d:.3f}\n")
        f.write(f"file '{frames[-1][0].name}'\n")
    total = round(sum(d for _, d, _ in frames), 2)
    silent = OUT / "holdout_demo_v2_silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=30,format=yuv420p",
                    "-t", str(total), "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", str(silent)], check=True, cwd=FR)
    # narration at each scene start
    starts, t0 = {}, 0.0
    for _, d, s in frames:
        starts.setdefault(s, t0)
        t0 += d
    inputs, filt = [], []
    for i, (scene, _) in enumerate(lines):
        ms = int((starts[scene] + 0.3) * 1000)
        inputs += ["-i", str(audio[scene][0])]
        filt.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    mix = ";".join(filt) + ";" + "".join(f"[a{i}]" for i in range(len(lines))) + f"amix=inputs={len(lines)}:normalize=0[aout]"
    out = OUT / "holdout_demo_v2.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs, "-filter_complex", mix, "-map", "0:v",
                    "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(total), str(out)], check=True)
    script = "\n".join(f"{starts[s]:5.1f}s  [{s}]  {audio[s][2]}" for s, _ in lines)
    (OUT / "voiceover_v2.txt").write_text(f"Total {total} s. Read each line at its start time.\n\n{script}\n")
    print(json.dumps({"total_s": total, "scenes": {s: {"start": round(starts[s], 2), "seconds": seconds[s], "audio_s": round(audio[s][1], 2)}
                                                    for s, _ in lines}}))
    print(out)


if __name__ == "__main__":
    main()
