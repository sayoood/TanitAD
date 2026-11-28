"""The >=GT bar across seeds -- a RANK test and a spread comparison, not a mean difference.

⛔ WHY NOT A MEAN DIFFERENCE. At n=2 I compared cell means against the larger
within-cell range and reported "WITHIN NOISE". That test is correct but extremely
conservative, and it throws away the thing that actually distinguishes these cells:
bar-OFF is WILD and bar-ON is TIGHT. A test built to compare centres cannot see a
difference in spread.

⭐ THE ADMISSIBLE STATISTIC HERE IS SEPARATION, and it is exact. With n_on and n_off
samples and no ties, the probability that every ON value falls below every OFF value
by chance is 1 / C(n_on+n_off, n_on) -- the one-sided Mann-Whitney extreme. That is a
LITERAL derived from the design, not fitted to the data.

⛔ TIER T0, training-side, non-parity corpus. This is a diagnostic about the TRAINING
dynamics of the RL stage. Any capability claim requires T1 and a four-family panel.
⛔ It is also NOT evidence that the bar improves driving -- only that, on this rig, the
arms carrying it land in a much narrower band on this training-side statistic.
"""
import json, os, statistics as st
from math import comb

BASE = r"C:\Users\Admin\tanitad-data\rl-pilot"
CELLS = {("off", 0): "p1-grpo-2k-repro", ("on", 0): "p1-grpo-2k-gtbar"}
for s in (1, 2, 3, 4):
    CELLS[("on", s)] = f"gtbar-s{s}"
    CELLS[("off", s)] = f"off-s{s}"

METRICS = (("dR3", "sel-ADE drift %", 100.0, "R3_rel"),
           ("dR1", "fan reward", 1.0, "R1"),
           ("dR2", "fan collision pp", 100.0, "R2"))

def load(tag):
    p = os.path.join(BASE, tag, "pilot_summary.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

data, fabs = {}, []
for (setting, seed), tag in CELLS.items():
    d = load(tag)
    if d is None:
        continue
    pd = d.get("pilot_delta", {}) or {}
    row = {k: (pd.get(src) or 0.0) * scale for k, _, scale, src in METRICS}
    data.setdefault(setting, {})[seed] = row
    if setting == "on" and isinstance(d.get("frac_above_bar_mean"), float):
        fabs.append(d["frac_above_bar_mean"])
    # ⛔ the control must genuinely have the bar off
    if setting == "off" and d.get("use_gt_bar"):
        print(f"⛔ ABORT: {tag} is labelled OFF but reports use_gt_bar=True")
        raise SystemExit(2)

for m, label, _, _ in METRICS:
    on = [data["on"][k][m] for k in sorted(data.get("on", {}))]
    off = [data["off"][k][m] for k in sorted(data.get("off", {}))]
    print(f"\n=== {label} ({m}) ===")
    print(f"  bar ON  n={len(on)}  {[round(v,2) for v in on]}")
    print(f"  bar OFF n={len(off)} {[round(v,2) for v in off]}")
    if len(on) < 2 or len(off) < 2:
        print("  (need >=2 per cell)"); continue
    print(f"  mean  ON {st.mean(on):+10.4f}   OFF {st.mean(off):+10.4f}")
    # the ratio is printed only when it exists; the stdevs ALWAYS print. Writing this
    # as one conditional expression silently swallowed the whole line when ON's stdev
    # was 0 -- an output that disappears is worse than one that says "n/a".
    s_on, s_off = st.pstdev(on), st.pstdev(off)
    ratio = f"{s_off / s_on:.1f}x" if s_on > 0 else "n/a (ON stdev is 0)"
    print(f"  stdev ON {s_on:10.4f}   OFF {s_off:10.4f}   ratio {ratio}")
    # exact separation probability, one-sided, no ties
    if max(on) < min(off):
        p = 1.0 / comb(len(on) + len(off), len(on))
        print(f"  ⭐ COMPLETE SEPARATION: every ON below every OFF. exact one-sided p = 1/{comb(len(on)+len(off),len(on))} = {p:.4f}")
    elif min(on) > max(off):
        p = 1.0 / comb(len(on) + len(off), len(on))
        print(f"  ⭐ COMPLETE SEPARATION (other direction). exact one-sided p = {p:.4f}")
    else:
        lo, hi = max(min(on), min(off)), min(max(on), max(off))
        print(f"  ⛔ OVERLAP on [{lo:.2f}, {hi:.2f}] -- no separation claim")

if fabs:
    print(f"\nfrac_above_bar across {len(fabs)} bar-ON arms: {[round(f,4) for f in fabs]}")
    print(f"  mean {st.mean(fabs):.4f}   range {max(fabs)-min(fabs):.4f}"
          f"   -- the mask admits about {100*st.mean(fabs):.0f} % of sampled anchors")

print("\n⛔ TIER T0 training-side, non-parity corpus. Not a capability claim; that needs T1.")
print("⛔ A separation on a training-side statistic is not evidence the car drives better.")
