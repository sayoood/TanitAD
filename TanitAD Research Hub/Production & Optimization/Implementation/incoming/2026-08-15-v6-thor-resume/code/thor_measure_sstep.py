#!/usr/bin/env python3
"""Thor vs A40 s/step for the v6F S-W run — MARGINAL, not cumulative.

⛔ THREE TRAPS THIS AVOIDS, all of which would have produced a quotable-looking
   number that is wrong:

1. `step_s` is NOT a per-step time and NOT the `--log-every` accumulation the old
   trainers emitted. `train_v6_staged.py:1406` computes
       step_s = (now - t0) / (step - start_step)
   i.e. the CUMULATIVE MEAN over the steps *this process* ran. On a resume the
   first rows are inflated by startup (dataset scan + the O4 saliency pass over
   319 002 windows) amortised over very few steps. Quoting row 1 as "Thor's
   speed" would report the startup, not the training.

2. `train_log.jsonl` is APPENDED across processes, and the banked A40 log already
   ends at step 6300 while the Thor resume STARTS at 6250 — so the step numbers
   overlap and a `step > 6250` filter silently mixes two machines. The process
   boundary is found from the note's own divisor (`step - start_step`) resetting,
   never from the step number.

3. Comparing Thor's marginal against the A40's *cumulative* would be a category
   error. This computes the marginal for BOTH from the same identity:
       T(S) = step_s(S) * S            (S = steps this process ran)
       marginal[S1,S2] = (T(S2) - T(S1)) / (S2 - S1)

Exits 1 (quoting nothing) until the resumed segment has >=2 rows.
"""
import json
import os
import re
import sys

P = os.path.expanduser("~/experiments/v6F-SW-30k/train_log.jsonl")
NOTE_N = re.compile(r"over the (\d+) steps")

rows = []
with open(P, errors="ignore") as fh:
    for ln in fh:
        if not ln.startswith("{") or '"step_s"' not in ln:
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        m = NOTE_N.search(d.get("step_s_note", ""))
        if not m or "step_s" not in d:
            continue
        n = int(m.group(1))                      # steps THIS process ran
        if n <= 0:
            continue
        rows.append({"step": d["step"], "n": n, "T": d["step_s"] * n,
                     "cum": d["step_s"], "loss": d.get("loss")})

# Process boundaries: `n` is monotonically increasing within one process.
segs, cur = [], []
for r in rows:
    if cur and r["n"] <= cur[-1]["n"]:
        segs.append(cur)
        cur = []
    cur.append(r)
if cur:
    segs.append(cur)

print(f"{len(rows)} step_s rows in {len(segs)} process segment(s)")
for i, s in enumerate(segs):
    print(f"  seg {i}: steps {s[0]['step']}..{s[-1]['step']}  "
          f"n {s[0]['n']}..{s[-1]['n']}  rows {len(s)}")


def marginal(seg, lo=None):
    """Marginal s/step over the segment (or its tail from index `lo`)."""
    a, b = seg[lo if lo is not None else 0], seg[-1]
    ds = b["n"] - a["n"]
    return None if ds <= 0 else ((b["T"] - a["T"]) / ds, ds, b["step"])


# ⛔ The Thor segment is identifiable ONLY as a NEW process segment. With one
# segment the file still holds the banked A40 run alone, and naming it "THOR"
# would attribute the A40's own rate to Thor — the precise misattribution this
# script exists to prevent. Report the A40 side and refuse the rest.
if len(segs) < 2:
    a40 = segs[-1]
    w = min(6, len(a40) - 1)
    m = marginal(a40, len(a40) - 1 - w) if w >= 1 else None
    print(f"\nA40 (banked) cumulative over {a40[-1]['n']} steps : "
          f"{a40[-1]['cum']:6.2f} s/step")
    if m:
        print(f"A40 (banked) MARGINAL over its last {m[1]:>4} steps: "
              f"{m[0]:6.2f} s/step")
    print("\nNo second process segment yet — the Thor resume has not written a "
          "step_s row (first one lands at step 6300, i.e. 50 steps in).\n"
          "Refusing to quote anything as Thor's rate.")
    sys.exit(1)

thor, a40 = segs[-1], segs[-2]
print()
if a40:
    # A40 reference: the tail of the banked segment, matched in width to what
    # Thor has so far — same-width windows, per the exponent-window rule.
    w = max(2, len(thor))
    m = marginal(a40, max(0, len(a40) - w))
    print(f"A40  cumulative over its whole process : {a40[-1]['cum']:6.2f} s/step "
          f"({a40[-1]['n']} steps)")
    if m:
        print(f"A40  MARGINAL over its last {m[1]:>4} steps    : {m[0]:6.2f} s/step")

if len(thor) < 2:
    print(f"\nTHOR resumed segment has {len(thor)} row(s) — need >=2 for a "
          f"marginal. Refusing to quote startup as a training rate.")
    if thor:
        print(f"  (row 1 cumulative {thor[0]['cum']:.2f} s/step over "
              f"{thor[0]['n']} steps INCLUDES startup — not comparable)")
    sys.exit(1)

mt = marginal(thor)
print(f"\nTHOR MARGINAL over {mt[1]} steps (to step {mt[2]}) : "
      f"{mt[0]:6.2f} s/step   [{len(thor)} rows]")
print(f"THOR cumulative incl. startup                : {thor[-1]['cum']:6.2f} s/step")

if a40:
    ma = marginal(a40, max(0, len(a40) - max(2, len(thor))))
    if ma:
        r = mt[0] / ma[0]
        print(f"\n  RATIO Thor/A40 (marginal vs marginal) : {r:6.3f}x  "
              f"=> Thor is {'FASTER' if r < 1 else 'SLOWER'} by "
              f"{abs(1 - r) * 100:.1f}%")
    left = 30000 - thor[-1]["step"]
    print(f"\n  remaining {left} steps @ Thor marginal = "
          f"{left * mt[0] / 86400:.2f} days ({left * mt[0] / 3600:.1f} h)")
    if ma:
        print(f"  the A40 would need                     "
              f"{left * ma[0] / 86400:.2f} days for the same remainder")
print("\n  EVIDENCE: MEASURED (ours) · "
      f"{P} · both sides from the same emitter and the same identity")
