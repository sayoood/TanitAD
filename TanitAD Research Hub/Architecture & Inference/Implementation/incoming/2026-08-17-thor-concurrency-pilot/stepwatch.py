#!/usr/bin/env python3
"""stepwatch — derive INSTANTANEOUS per-step time from v6F-SW-30k's train_log.jsonl.

⛔ WHY THIS EXISTS (the pilot's headline instrument finding, MEASURED 2026-08-17).

``train_v6_staged.py`` logs ``step_s`` with its own note::

    "elapsed/step over the 6300 steps THIS process ran
     (NOT accumulated over --log-every, and NOT divided by the resumed step number)"

That is a **CUMULATIVE MEAN over every step since process start**, not a per-step
time. It is therefore almost perfectly INSENSITIVE to a transient slowdown, and the
concurrency pilot's stated abort criterion — ``step_s > 28.0`` — is an instrument
STRUCTURALLY UNABLE TO FIRE for the effect it is meant to catch:

    at N=6300 steps and elapsed=166,791.9 s, driving the cumulative mean to 28.0
    requires k further steps at instantaneous rate r, where k*(r-28) = 9608.1:
      r = 27.7 s/step (the intended +5 % trip point) -> NEVER (asymptote 27.7)
      r = 30.0 s/step (+13 %)                        -> k = 4804 steps ~ 40 h
      r = 40.0 s/step (+51 %)                        -> k =  801 steps ~ 8.9 h

The pilot lasts ~1-2 h. Nothing short of a total stall could move it. The observed
"26.47-26.66 all day" band is likewise not day variation: it is a converging mean
decaying monotonically (26.5505 -> 26.4749 over the last 51 points, never once up).

⇒ The admissible metric is the FIRST DIFFERENCE. Each line carries step_s and N
(steps this process ran), so elapsed_i = step_s_i * N_i is recoverable exactly, and

    r_inst(i) = (step_s_i*N_i - step_s_{i-1}*N_{i-1}) / (N_i - N_{i-1})

is the true mean per-step time over just that 50-step window. Same family as the
`df` / Thor `free` / cgroup `usage_in_bytes` traps in CLAUDE.md: a counter that
aggregates the wrong scope, read as if it answered the question.

Usage:
    stepwatch.py parse  <raw.txt>  <out.jsonl>     # R|step|step_s|N|gnorm|loss rows
    stepwatch.py report <a.jsonl> [<b.jsonl> ...]  # phase medians/ranges
"""
from __future__ import annotations
import json
import statistics as st
import sys


def parse(raw_path: str, out_path: str) -> int:
    rows = []
    for ln in open(raw_path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln.startswith("R|"):
            continue
        p = ln.split("|")
        if len(p) < 6:
            continue
        try:
            rows.append({"step": int(p[1]), "step_s": float(p[2]), "n": int(p[3]),
                         "gnorm": float(p[4]), "loss": float(p[5])})
        except ValueError:
            continue
    rows.sort(key=lambda r: r["step"])
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(rows)


def inst(rows: list[dict]) -> list[dict]:
    """First-difference the cumulative mean into per-window mean step time."""
    out = []
    for a, b in zip(rows, rows[1:]):
        dn = b["n"] - a["n"]
        if dn <= 0:
            continue
        r = (b["step_s"] * b["n"] - a["step_s"] * a["n"]) / dn
        out.append({"step": b["step"], "n": b["n"], "dn": dn, "r_inst": r,
                    "gnorm": b["gnorm"], "loss": b["loss"]})
    return out


def _fmt(name: str, vals: list[float], unit: str = "") -> str:
    if not vals:
        return f"  {name:<26} n=0  (no data)"
    return (f"  {name:<26} n={len(vals):<3} median={st.median(vals):.4f}{unit}  "
            f"min={min(vals):.4f}  max={max(vals):.4f}  "
            f"iqr=[{_q(vals,.25):.4f}, {_q(vals,.75):.4f}]")


def _q(v: list[float], f: float) -> float:
    s = sorted(v)
    i = f * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def report(paths: list[str]) -> None:
    for p in paths:
        rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
        ii = inst(rows)
        label = p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        print(f"\n=== {label} ===  steps {rows[0]['step']}..{rows[-1]['step']}"
              f"  ({len(rows)} logged points -> {len(ii)} instantaneous)")
        print(_fmt("r_inst (s/step) TRUE", [x["r_inst"] for x in ii], " s"))
        print(_fmt("step_s (cumulative MEAN)", [r["step_s"] for r in rows], " s")
              + "   <- NOT usable as an abort trip")
        print(_fmt("gnorm", [r["gnorm"] for r in rows]))
        print(_fmt("loss", [r["loss"] for r in rows]))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "parse":
        print(f"parsed {parse(sys.argv[2], sys.argv[3])} rows -> {sys.argv[3]}")
    elif sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
