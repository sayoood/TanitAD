"""The 2x2 over (>=GT bar, seed) -- read the lever against THIS RIG'S OWN noise.

⛔ WHY THIS SHAPE. The paired episode-cluster bootstrap resamples EPISODES with the
models held fixed, so it answers "would another draw of episodes say this?" and is
structurally blind to "would another TRAINING RUN say this?". On this programme's rig
an arm with ZERO levers moved once read "separably worse" on 5 of 9 family metrics,
and H-ESTIM-SEED-1 puts the false-positive rate for "separated" at 14.3 %.

⇒ The admissible comparison is the LEVER effect against the SEED effect:
     lever  = mean(bar ON)  - mean(bar OFF)      across seeds
     noise  = |seed1 - seed0| within the SAME setting
   A lever effect smaller than the noise it is read against is NOT an effect.

⛔ This is T0, training-side. Any capability claim requires T1. No paired CI is
computed here and none is implied: n = 2 per cell.
"""
import json, os, sys

BASE = r"C:\Users\Admin\tanitad-data\rl-pilot"
ARMS = {
    # ⭐ the bar-OFF seed-0 cell is the RE-RUN through the SAME library, not the
    # banked 2026-09-10 arm -- so the only difference from ("on", 0) is the flag.
    # The re-run reproduces the banked arm on all 9 quantities to full float
    # precision, so this is a framing improvement, not a different number.
    ("off", 0): "p1-grpo-2k-repro",
    ("on", 0):  "p1-grpo-2k-gtbar",
    ("on", 1):  "gtbar-s1",
    ("off", 1): "off-s1",
}

def load(tag):
    p = os.path.join(BASE, tag, "pilot_summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def metrics(d):
    pd = d.get("pilot_delta", {}) or {}
    rae = d.get("reward_audit_exercising") or d.get("reward_audit") or {}
    return {
        "dR1": pd.get("R1"),
        "dR2_pp": (pd.get("R2") or 0) * 100.0,
        "dR3_pct": (pd.get("R3_rel") or 0) * 100.0,
        "frac_above_bar": d.get("frac_above_bar_mean"),
        "use_gt_bar": d.get("use_gt_bar"),
        "steps": d.get("steps"),
        "audit": rae.get("verdict"),
        "audit_flagged": rae.get("flagged"),
    }

cells, missing = {}, []
for key, tag in ARMS.items():
    d = load(tag)
    if d is None:
        missing.append(f"{key} -> {tag}")
        continue
    cells[key] = metrics(d)

print("=" * 96)
print(f"{'setting':10s} {'seed':>4s} {'gt_bar':>7s} {'steps':>6s} {'frac_above':>11s} "
      f"{'dR1':>9s} {'dR2 pp':>9s} {'dR3 %':>9s} {'audit':>10s}")
print("-" * 96)
for setting in ("off", "on"):
    for seed in (0, 1):
        c = cells.get((setting, seed))
        if not c:
            print(f"{setting:10s} {seed:>4d}   (missing)")
            continue
        fab = c["frac_above_bar"]
        print(f"{setting:10s} {seed:>4d} {str(c['use_gt_bar']):>7s} {str(c['steps']):>6s} "
              f"{(f'{fab:.4f}' if isinstance(fab, float) else 'null'):>11s} "
              f"{c['dR1']:>+9.4f} {c['dR2_pp']:>+9.3f} {c['dR3_pct']:>+9.2f} {str(c['audit']):>10s}")
print("=" * 96)

if missing:
    print("\n⛔ INCOMPLETE -- refusing to compute a lever effect. Missing:")
    for m in missing:
        print("   ", m)
    sys.exit(0)

print("\nLEVER vs NOISE  (⛔ a lever smaller than its noise is NOT an effect)\n")
print(f"{'metric':10s} {'lever (on-off)':>16s} {'noise |s1-s0| off':>20s} {'noise |s1-s0| on':>19s} {'verdict':>28s}")
for m, lab in (("dR1", "dR1"), ("dR2_pp", "dR2 pp"), ("dR3_pct", "dR3 %")):
    on = (cells[("on", 0)][m] + cells[("on", 1)][m]) / 2.0
    off = (cells[("off", 0)][m] + cells[("off", 1)][m]) / 2.0
    lever = on - off
    n_off = abs(cells[("off", 1)][m] - cells[("off", 0)][m])
    n_on = abs(cells[("on", 1)][m] - cells[("on", 0)][m])
    noise = max(n_off, n_on)
    if noise == 0:
        verdict = "noise floor is 0 -- suspect"
    elif abs(lever) > 2 * noise:
        verdict = f"survives ({abs(lever)/noise:.1f}x its noise)"
    elif abs(lever) > noise:
        verdict = f"MARGINAL ({abs(lever)/noise:.1f}x)"
    else:
        verdict = f"⛔ WITHIN NOISE ({abs(lever)/noise:.2f}x)"
    print(f"{lab:10s} {lever:>+16.4f} {n_off:>20.4f} {n_on:>19.4f} {verdict:>28s}")

print("\n⛔ TIER: T0 training-side. n = 2 per cell, no paired CI, non-parity corpus.")
print("⛔ Nothing here is a capability claim; that requires T1.")
