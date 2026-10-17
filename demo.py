"""HOLDOUT: Shake-Up, terminal demo.

  python demo.py --replay runs/m2            replay a recorded run (no network needed)
  python demo.py --replay runs/m2 --comp diabetes --speed 2 --no-pause
  python demo.py --intro --replay runs/m2    pitch flow: title card + boosting chart, then the replay
  python demo.py --live --rounds 1           one small live round on Crusoe, then the reveal
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent
console = Console()
ARM_TITLE = {"naive": "NAIVE team  ·  Kaggle leaderboard", "holdout": "HOLDOUT team  ·  Ladder leaderboard + gate"}
ARM_STYLE = {"naive": "red", "holdout": "green"}


class State:
    def __init__(self, comp: str):
        self.comp = comp
        self.round = 0
        self.subs = {"naive": {}, "holdout": {}}
        self.final = {}
        self.ticker: list[Text] = []
        self.playbook = {"naive": [], "holdout": []}
        self.gate_counts = {"naive": [0, 0], "holdout": [0, 0]}
        self.revealed = False

    def log(self, t: Text):
        self.ticker.append(t)
        self.ticker = self.ticker[-9:]

    def process(self, ev: dict):
        arm, typ = ev.get("arm"), ev["type"]
        if ev.get("comp") not in (None, self.comp) or arm not in self.subs and typ not in ("run_start", "run_end"):
            return False
        if typ == "round_start":
            self.round = max(self.round, ev["round"])
        elif typ == "submission":
            self.subs[arm][ev["id"]] = ev
            tag = Text(f"{'KAGGLE' if arm == 'naive' else 'LADDER'} ", style=f"bold {ARM_STYLE[arm]}")
            body = Text(f"{arm:7s} {ev['id']:10s} {ev['kind']:8s} ")
            if not ev["ok"]:
                err = (ev.get("error") or "").strip().splitlines()
                body.append(f"sandbox: failed ({(err[-1] if err else 'error')[:60]})", style="dim")
            elif arm == "naive":
                body.append(f"reported {ev['reported']:.3f}  ")
                body.append("accepted, new best" if ev["accepted"] else "shown on the board", style="bold red" if ev["accepted"] else "")
            else:
                if ev["accepted"]:
                    body.append(f"accepted, significant gain -> board shows {ev['reported']:.3f}", style="bold green")
                else:
                    body.append(f"rejected, gain within noise -> board still shows {ev['reported']:.3f}", style="yellow")
            self.log(tag + body)
        elif typ == "playbook":
            adm = [g for g in ev["gate"] if g.get("admitted")]
            rej = [g for g in ev["gate"] if "evidence_proposed" in g and not g.get("admitted")]
            self.gate_counts[arm][0] += len(adm)
            self.gate_counts[arm][1] += len(rej)
            self.playbook[arm] = ev["rules"]
            t = Text("CURATOR ", style="bold magenta") + Text(f"{arm:7s} R{ev['round']} playbook: {len(adm)} rules admitted, {len(rej)} rejected")
            self.log(t)
            for g in rej[:2]:
                self.log(Text(f"   GATE rejected: \"{g['rule'][:70]}\" ({g['reason']})", style="yellow"))
            for r in ev.get("added", [])[:1]:
                self.log(Text(f"   + \"{r[:90]}\"", style="magenta"))
        elif typ == "structure_change":
            self.log(Text(f"STRUCTURE {arm}: {json.dumps(ev['change'])[:110]}", style="cyan"))
        elif typ == "structure_verdict":
            self.log(Text(f"STRUCTURE {arm}: change {ev['verdict']}", style="cyan"))
        elif typ == "final":
            self.final[arm] = ev
        return True

    # ------------------------------------------------------------ rendering
    def board(self, arm: str) -> Table:
        rows = [s for s in self.subs[arm].values() if s["ok"]]
        t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        final_id = self.final.get(arm, {}).get("id")
        if not self.revealed:
            rows.sort(key=lambda s: (-(s["true_public"] if arm == "naive" else s["reported"]), s["id"]))
            t.add_column("#", width=3)
            t.add_column("submission")
            t.add_column("kind")
            t.add_column("reported public", justify="right")
            t.add_column("", width=9)
            for i, s in enumerate(rows[:11], 1):
                pick = s["id"] == final_id
                t.add_row(str(i), s["id"], s["kind"], f"{s['reported']:.3f}",
                          "NEW BEST" if s["accepted"] else "", style="bold" if pick else None)
        else:
            pub_rank = {s["id"]: i for i, s in enumerate(sorted(rows, key=lambda s: (-s["true_public"], s["id"])), 1)}
            rows.sort(key=lambda s: (-s["private"], s["id"]))
            t.add_column("priv #", width=6)
            t.add_column("submission")
            t.add_column("public", justify="right")
            t.add_column("private", justify="right")
            t.add_column("move", width=8)
            for i, s in enumerate(rows[:11], 1):
                d = pub_rank[s["id"]] - i
                mv = Text(f"↑{d}", style="green") if d > 0 else Text(f"↓{-d}", style="red") if d < 0 else Text("=")
                pick = s["id"] == final_id
                t.add_row(str(i), s["id"] + (" ◀ pick" if pick else ""), f"{s['true_public']:.3f}", f"{s['private']:.3f}", mv,
                          style="bold reverse" if pick else None)
            fr = next((s for s in rows if s["id"] == final_id), None)
            if fr:
                rank = rows.index(fr) + 1
                if rank > 11:
                    t.add_row(str(rank), fr["id"] + " ◀ pick", f"{fr['true_public']:.3f}", f"{fr['private']:.3f}",
                              Text(f"↓{rank - pub_rank[fr['id']]}", style="red"), style="bold reverse")
        return t

    def arm_panel(self, arm: str) -> Panel:
        n = len(self.subs[arm])
        ok = sum(s["ok"] for s in self.subs[arm].values())
        adm, rej = self.gate_counts[arm]
        sub = f"{ok}/{n} scored · playbook rules admitted {adm} / rejected {rej}"
        fin = self.final.get(arm)
        if fin and fin.get("id"):
            if self.revealed:
                delta = fin["private"] - fin["reported"]
                sub = (f"pick {fin['id']}: reported {fin['reported']:.3f} -> private {fin['private']:.3f} "
                       f"({delta:+.3f})")
            else:
                sub = f"final pick {fin['id']} at reported public {fin['reported']:.3f}"
        return Panel(self.board(arm), title=f"[bold {ARM_STYLE[arm]}]{ARM_TITLE[arm]}", subtitle=sub,
                     border_style=ARM_STYLE[arm])

    def render(self, header: str) -> Layout:
        lay = Layout()
        lay.split_column(Layout(name="h", size=3), Layout(name="b"), Layout(name="t", size=11))
        hdr = Text(f"HOLDOUT: Shake-Up   ·   {self.comp}   ·   round {self.round}   ·   {header}", style="bold")
        lay["h"].update(Panel(Align.center(hdr), border_style="white"))
        lay["b"].split_row(Layout(self.arm_panel("naive")), Layout(self.arm_panel("holdout")))
        lay["t"].update(Panel(Group(*self.ticker) if self.ticker else Text(""), title="agents and gates",
                              border_style="magenta"))
        return lay


def pick_comp(run_dir: Path) -> str:
    res = ROOT / "results" / "results.json"
    arena = json.loads((run_dir / "arena.json").read_text())
    if res.exists():
        r = json.loads(res.read_text())
        if r.get("arena_run") == run_dir.name:
            sh = r["shakeup"]  # default: the competition where the naive pick inflated most
            infl = {c: sh["naive"]["per_comp"][c].get("inflation") or 0 for c in arena["comps"]}
            return max(infl, key=infl.get)
    return arena["comps"][0]


def reveal(state: State, live: Live, pause: bool, header: str):
    if pause:
        live.stop()
        console.input("\n[bold]Press Enter to REVEAL PRIVATE leaderboard[/bold] ")
        live.start()
    else:
        time.sleep(2.0)
    state.revealed = True
    state.log(Text("REVEAL PRIVATE: re-sorting both boards by the sealed private score", style="bold white on blue"))
    for arm in ("naive", "holdout"):
        f = state.final.get(arm)
        if f and f.get("id"):
            state.log(Text(f"{arm:7s} pick {f['id']}: reported {f['reported']:.3f} -> private {f['private']:.3f} "
                           f"(inflation {f['reported'] - f['private']:+.3f})", style=f"bold {ARM_STYLE[arm]}"))
    live.update(state.render(header + "  ·  PRIVATE REVEALED"))


def epilogue(open_charts: bool):
    res = ROOT / "results" / "results.json"
    if not res.exists():
        return
    r = json.loads(res.read_text())
    lines = []
    bb = r.get("boosting_bot")
    if bb:
        k, l = bb["boards"]["kaggle"], bb["boards"]["ladder"]
        lines.append(f"Boosting attack (random labels, k={bb['k_max']}): Kaggle reports {k['reported_public']['mean'][-1]:.3f} "
                     f"(private {k['private']['mean'][-1]:.3f}); Ladder reports {l['reported_public']['mean'][-1]:.3f} "
                     f"(private {l['private']['mean'][-1]:.3f})")
    for arm in ("naive", "holdout"):
        m = r["shakeup"][arm]["mean"]
        if m:
            lines.append(f"{arm:7s} mean over {len(r['comps'])} dev competitions: reported {m['reported_public']:.3f}, "
                         f"private {m['private']:.3f}, inflation {m['inflation']:+.3f}")
    t = r.get("transfer")
    if t:
        m = t["mean_final_private"]
        ran = t.get("submissions_ran", {})
        lines.append("Transfer to unseen competitions, private of final pick (valid cells; submissions that ran): " + "; ".join(
            f"{k} {m[k]:.3f} ({t['valid_cells'][k]}/{t['cells']}; {ran[k][0]}/{ran[k][1]})" if k in m else f"{k} no valid pick"
            for k in ("gen-0", "naive-final", "holdout-final")))
    e = r.get("efficiency")
    if e:
        a = e["arena_run"]
        lines.append(f"Crusoe: arena run {a['calls']} live Nemotron calls, {a['prompt_tokens'] + a['completion_tokens']:,} tokens, "
                     f"~${a['usd']:.2f}; everything today ~${e['project_total']['usd']:.2f}")
    console.print(Panel("\n".join(lines), title="Results (results/results.json)", border_style="blue"))
    if open_charts and sys.platform == "darwin":
        for name in ("boosting_bot.png", "shakeup.png", "transfer.png"):
            p = ROOT / "results" / name
            if p.exists():
                subprocess.run(["open", str(p)])


def intro(pause: bool, open_charts: bool):
    """Title card: the pitch line and the boosting-attack calibration (no LLM, guaranteed to show the effect)."""
    lines = [Text("HOLDOUT: Shake-Up", style="bold"),
             Text("Self-improving agents keep a change when their score goes up.", style="italic"),
             Text("That is how Kaggle teams overfit the public leaderboard, and how agent teams learn the scoreboard instead of the task.", style="italic"),
             Text("")]
    bb = ROOT / "results" / "boosting_bot.json"
    if bb.exists():
        b = json.loads(bb.read_text())
        k, l = b["boards"]["kaggle"], b["boards"]["ladder"]
        lines += [Text(f"Calibration: {b['k_max']} RANDOM label vectors (zero skill) attack a public board of n={b['public_n']} on {b['competition']}"),
                  Text(f"  Kaggle board reports {k['reported_public']['mean'][-1]:.3f}   private {k['private']['mean'][-1]:.3f}", style="bold red"),
                  Text(f"  Ladder board reports {l['reported_public']['mean'][-1]:.3f}   private {l['private']['mean'][-1]:.3f}", style="bold green")]
    console.print(Panel(Group(*lines), border_style="blue", title="Recursive Agents Hackathon · Crusoe x Nemotron"))
    if open_charts and sys.platform == "darwin" and (ROOT / "results" / "boosting_bot.png").exists():
        subprocess.run(["open", str(ROOT / "results" / "boosting_bot.png")])
    if pause:
        console.input("[bold]Press Enter to replay the AI Kaggle team[/bold] ")
    else:
        time.sleep(3)


def replay(run_dir: Path, comp: str | None, speed: float, pause: bool, open_charts: bool):
    comp = comp or pick_comp(run_dir)
    events = [json.loads(l) for l in open(run_dir / "events.jsonl")]
    state = State(comp)
    header = f"replay of {run_dir.name}"
    delay = {"submission": 0.45, "playbook": 1.6, "structure_change": 0.8, "structure_verdict": 0.8, "final": 0.6, "round_start": 0.5}
    with Live(state.render(header), console=console, refresh_per_second=10, screen=False) as live:
        for ev in events:
            if state.process(ev):
                live.update(state.render(header))
                time.sleep(delay.get(ev["type"], 0.0) / speed)
        reveal(state, live, pause, header)
    epilogue(open_charts)


def run_live(comp: str, rounds: int, pause: bool, open_charts: bool):
    sys.path.insert(0, str(ROOT))
    from holdout.arena import run_arms
    q: queue.Queue = queue.Queue()
    run_id = time.strftime("live-%H%M%S")
    th = threading.Thread(target=lambda: run_arms(run_id, [comp], rounds, on_event=q.put), daemon=True)
    th.start()
    state = State(comp)
    header = f"LIVE on Crusoe · {run_id}"
    with Live(state.render(header), console=console, refresh_per_second=10) as live:
        while th.is_alive() or not q.empty():
            try:
                ev = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if state.process(ev):
                live.update(state.render(header))
        reveal(state, live, pause, header)
    console.print(f"Recorded to runs/{run_id}/ (replay with: python demo.py --replay runs/{run_id})")
    epilogue(open_charts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--comp")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--intro", action="store_true", help="title card + boosting-attack chart before the replay")
    a = ap.parse_args()
    if a.intro:
        intro(not a.no_pause, not a.no_open)
    if a.live:
        run_live(a.comp or "diabetes", a.rounds, not a.no_pause, not a.no_open)
    elif a.replay:
        replay(a.replay, a.comp, a.speed, not a.no_pause, not a.no_open)
    elif not a.intro:
        ap.print_help()
