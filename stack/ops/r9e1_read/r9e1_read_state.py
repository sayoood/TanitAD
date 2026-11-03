#!/usr/bin/env python3
"""r9e1_read_state.py -- the FILE-ONLY state machine behind r9e1_read.sh's WAIT mode (refcv9 E1 C0 / C1 read).

stdlib only: no torch, no numpy, no GPU. It reads the chain's append-only ``CHAIN_STATUS.txt`` and the two run dirs' marker
files and answers WAIT / READY / ABORT. The chain (``stack/ops/r9e1_gate3/r9gate3_chain.sh``) writes one event per line,
``<utc> <EVENT> <text>``; the events used here are ``START``, ``C0-LAUNCH`` / ``C1-LAUNCH`` (the text carries ``run dir <path>``),
``C0-DONE`` / ``C1-DONE``, ``C0-FAILED`` / ``C0-ZEROGRAD-FAIL`` / ``C0-REFUSED`` / ``C0-NOT-LAUNCHED`` (and the C1 twins),
``STOP-*`` (a pre-launch stop) and ``ALLDONE``.

THE FILE IS APPEND-ONLY ACROSS CHAIN RESTARTS. An earlier chain attempt's ``STOP-EXPLORE-NOT-PASS`` precedes the later ``START``
that actually trained; reading the whole file would abort on a stop that has long been superseded. Only the lines AFTER THE LAST
``START`` are read. A ``STOP-*`` there is a pre-launch stop (no arm trains after it), so it does not abort the wait: the chain may
be restarted, and the restart is a new ``START`` that resets the segment. It is reported in the state so the log shows it.

THE DONE RULE is the chain's own (``r9gate3_chain.sh``, section G): ``<run>/summary.json`` matches ``"done": true`` AND carries
``"zerograd_gate"`` AND ``<run>/ZEROGRAD_FAIL.json`` does not exist. A ``C1-DONE`` line is necessary; the markers are re-verified
here so that a state file cannot say READY over a run that is not finished. READY additionally needs each arm's final EMA
checkpoint ``<run>/model_ema.pt`` (the trainer's ``step == args.steps`` save) and ``<run>/config.json`` (the dump builds the model
from it), and the two ``summary.json`` ``step`` fields equal (to each other and, when the launch tree's ``r9e1_<arm>.argv.json`` is
there, to its ``steps``).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

ARMS = ("C0", "C1")
BAD_KINDS = ("FAILED", "ZEROGRAD-FAIL", "REFUSED", "NOT-LAUNCHED")
_LINE = re.compile(r"^(\S+Z)\s+(\S+)\s*(.*)$")
_RUN_DIR = re.compile(r"run dir\s+(\S+)")
_DONE_TRUE = re.compile(r'"done"\s*:\s*true')


def parse_status(text: str) -> list:
    """[(utc, event, rest, raw_line)] for every well-formed line (anything else is ignored, never an error)."""
    out = []
    for raw in text.splitlines():
        m = _LINE.match(raw.strip())
        if m:
            out.append((m.group(1), m.group(2), m.group(3), raw.strip()))
    return out


def last_segment(lines: list) -> list:
    idx = [i for i, l in enumerate(lines) if l[1] == "START"]
    return lines[idx[-1] + 1:] if idx else lines


def run_dir_of(arm: str, launch_line, root: Path) -> Path:
    if launch_line is not None:
        m = _RUN_DIR.search(launch_line[2])
        if m:
            return Path(m.group(1))
    return root / "runs" / ("r9e1-" + arm)


def marker_problems(run: Path) -> list:
    """The chain's own done rule, as a list of what is missing (empty = DONE)."""
    summ = run / "summary.json"
    if not summ.is_file() or summ.stat().st_size == 0:
        return ["%s is absent or empty" % summ]
    txt = summ.read_text(encoding="utf-8", errors="replace")
    out = []
    if not _DONE_TRUE.search(txt):
        out.append('%s has no "done": true' % summ)
    if '"zerograd_gate"' not in txt:
        out.append('%s has no "zerograd_gate" record' % summ)
    if (run / "ZEROGRAD_FAIL.json").exists():
        out.append("%s exists" % (run / "ZEROGRAD_FAIL.json"))
    return out


def _expected_steps(root: Path, arm: str):
    f = root / "launch_tree" / "stack" / "ops" / "runs.d" / ("r9e1_%s.argv.json" % arm)
    try:
        return int(json.loads(f.read_text(encoding="utf-8"))["steps"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def checkpoint_problems(runs: dict, root: Path) -> list:
    out, steps = [], {}
    for arm, run in runs.items():
        ck = run / "model_ema.pt"
        if not ck.is_file() or ck.stat().st_size == 0:
            out.append("%s: final EMA checkpoint %s is absent or empty" % (arm, ck))
        if not (run / "config.json").is_file():
            out.append("%s: %s is absent (the dump builds the model from it)" % (arm, run / "config.json"))
        try:
            steps[arm] = int(json.loads((run / "summary.json").read_text(encoding="utf-8"))["step"])
        except (OSError, ValueError, KeyError, TypeError):
            out.append("%s: summary.json has no integer `step`" % arm)
    if len(steps) == 2 and steps["C0"] != steps["C1"]:
        out.append("the arms stopped at different steps: C0 %d vs C1 %d" % (steps["C0"], steps["C1"]))
    for arm, st in steps.items():
        want = _expected_steps(root, arm)
        if want is not None and st != want:
            out.append("%s: summary step %d != the launch argv's steps %d" % (arm, st, want))
    return out


def classify(status_text: str, root) -> dict:
    """-> {state: WAIT|READY|ABORT, reason, line, c0_run, c1_run, sig, stops}. ``sig`` changes when the picture changes (log dedup)."""
    root = Path(root)
    seg = last_segment(parse_status(status_text))
    got = {a: {"launch": None, "done": None, "bad": None} for a in ARMS}
    alldone, stops = None, []
    for l in seg:
        ev = l[1]
        m = re.match(r"^(C[01])-(.+)$", ev)
        if m and m.group(1) in got:
            arm, kind = m.group(1), m.group(2)
            if kind == "LAUNCH":
                got[arm]["launch"] = l
            elif kind == "DONE":
                got[arm]["done"] = l
            elif kind in BAD_KINDS and got[arm]["bad"] is None:
                got[arm]["bad"] = l
        elif ev == "ALLDONE":
            alldone = l
        elif ev.startswith("STOP-"):
            stops.append(l[3])
    runs = {a: run_dir_of(a, got[a]["launch"], root) for a in ARMS}
    base = {"c0_run": str(runs["C0"]), "c1_run": str(runs["C1"]), "stops": stops}

    def state(s, reason, line=""):
        d = dict(base, state=s, reason=reason, line=line)
        d["sig"] = "|".join([s] + ["%s:%s" % (a, "bad" if got[a]["bad"] else "done" if got[a]["done"] else
                                              "launched" if got[a]["launch"] else "-") for a in ARMS] + [str(len(stops))])
        return d

    for a in ARMS:
        if got[a]["bad"]:
            return state("ABORT", "%s did not finish: %s" % (a, got[a]["bad"][1]), got[a]["bad"][3])
    if got["C0"]["done"] and got["C1"]["done"]:
        probs = []
        for a in ARMS:
            probs += ["%s: %s" % (a, p) for p in marker_problems(runs[a])]
        if not probs:
            probs = checkpoint_problems(runs, root)
        if probs:
            return state("ABORT", "C0-DONE and C1-DONE are logged but the run dirs disagree: " + "; ".join(probs),
                         got["C1"]["done"][3])
        return state("READY", "both arms done: markers and final EMA checkpoints verified", got["C1"]["done"][3])
    if alldone:
        return state("ABORT", "ALLDONE without %s" % " and ".join(a + "-DONE" for a in ARMS if not got[a]["done"]), alldone[3])
    why = "waiting for the chain: " + ", ".join("%s %s" % (a, "done" if got[a]["done"] else "training" if got[a]["launch"]
                                                         else "not launched") for a in ARMS)
    if stops:
        why += " (last chain stop: %s)" % stops[-1][:160]
    return state("WAIT", why, stops[-1] if stops else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", required=True, help="CHAIN_STATUS.txt (may not exist yet)")
    ap.add_argument("--root", required=True, help="the r9e1 root (runs/, launch_tree/)")
    ap.add_argument("--out", required=True, help="a shell-sourceable file: S_STATE, S_REASON, S_LINE, S_C0_RUN, S_C1_RUN, S_SIG")
    a = ap.parse_args(argv)
    p = Path(a.status)
    text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    d = classify(text, a.root)
    if not p.is_file():
        d["reason"] = "no status file yet at %s: " % p + d["reason"]
    body = "".join("S_%s=%s\n" % (k.upper(), shlex.quote(str(d[k]))) for k in ("state", "reason", "line", "c0_run", "c1_run", "sig"))
    tmp = Path(a.out + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(a.out)
    print(d["state"], d["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
