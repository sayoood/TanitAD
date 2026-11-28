#!/usr/bin/env python3
"""E-THOR-CLK: did pinning the clocks buy back any of the 33 % deficit?

⛔ THE DISCARD RULE, WHICH IS THE WHOLE POINT. `jetson_clocks` landed between two
logged rows. The logging interval that CONTAINS the change spans TWO hardware
configurations and is attributable to NEITHER — averaging through it would
manufacture a number that is a blend. So the interval whose left endpoint is the
change step is DROPPED, and "after" starts only where BOTH endpoints follow it.
(Same discipline as the machine boundary in RETRACTION_LOG C68.)

Both sides use the identical marginal identity, from the trainer's own
cumulative `step_s`:   T(S) = step_s(S) * S ,  marginal = dT/dS
so this is a like-for-like comparison, not marginal-vs-cumulative.
"""
import json
import os
import re
import sys

P = os.path.expanduser("~/experiments/v6F-SW-30k/train_log.jsonl")
MARK = os.path.expanduser("~/experiments/v6F-SW-30k/E_THOR_CLK_step.txt")
CHANGE = int(open(MARK).read().strip())
NOTE_N = re.compile(r"over the (\d+) steps")

rows = []
for ln in open(P, errors="ignore"):
    if not ln.startswith("{") or '"step_s"' not in ln:
        continue
    try:
        d = json.loads(ln)
    except ValueError:
        continue
    m = NOTE_N.search(d.get("step_s_note", ""))
    if m and "step_s" in d:
        rows.append({"step": d["step"], "n": int(m.group(1)),
                     "T": d["step_s"] * int(m.group(1))})

# keep only the CURRENT process (n increases monotonically within one)
seg, cur = [], []
for r in rows:
    if cur and r["n"] <= cur[-1]["n"]:
        seg.append(cur); cur = []
    cur.append(r)
seg.append(cur)
live = seg[-1]

before = [r for r in live if r["step"] <= CHANGE]
after = [r for r in live if r["step"] > CHANGE]


def marg(rs):
    if len(rs) < 2:
        return None
    return ((rs[-1]["T"] - rs[0]["T"]) / (rs[-1]["n"] - rs[0]["n"]),
            rs[-1]["n"] - rs[0]["n"], rs[0]["step"], rs[-1]["step"])


print(f"clock change at step {CHANGE} · rows before {len(before)} · after {len(after)}")
print(f"⛔ DISCARDED straddling interval {CHANGE} -> {CHANGE + 50} "
      f"(spans both configurations)")

b = marg(before)
if b:
    print(f"\nBEFORE (EMC 3200) marginal {b[0]:6.2f} s/step  "
          f"over {b[1]} steps, {b[2]}->{b[3]}")

if len(after) < 2:
    print(f"\nAFTER: {len(after)} row(s) past the change — need >=2 for a "
          f"marginal. Refusing to quote.")
    sys.exit(1)

a = marg(after)
print(f"AFTER  (EMC 4266) marginal {a[0]:6.2f} s/step  "
      f"over {a[1]} steps, {a[2]}->{a[3]}")
if b:
    d = a[0] - b[0]
    print(f"\n  DELTA {d:+.2f} s/step  ({d / b[0] * 100:+.1f} %)  "
          f"=> {'FASTER' if d < 0 else 'SLOWER' if d > 0 else 'FLAT'}")
    print(f"  A40 reference (matched-width marginal) 20.46 s/step")
    print(f"  Thor/A40 was {b[0] / 20.46:.3f}x, now {a[0] / 20.46:.3f}x")
    left = 30000 - a[3]
    print(f"\n  remaining {left} steps: {left * a[0] / 86400:.2f} days "
          f"(was {left * b[0] / 86400:.2f} at the old clock)")
print("\n  EVIDENCE: MEASURED (ours) · same emitter, same identity, both sides · "
      "straddling interval discarded, not averaged")
