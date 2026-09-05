"""Evaluate the PRE-REGISTERED outcomes of H-EGO-LIT-4 mechanically, from panel_report.json.

⛔ The outcomes were committed in `GOALS_AND_CLAIMS.md` and `SPEC.md` BEFORE the panel ran.
This file only READS them off the measured report; it must never introduce a criterion the
pre-registration does not contain. Where the pre-registration is silent, this prints
UNDECIDED rather than inventing a rule.

Two things it enforces that a human read would drift on:

 1. ⛔ **OUTCOME IV first.** If the anti-echo gate does not FAIL A5_regress (the
    image-ablated deliberate regression), the panel is VOID and no other row means
    anything. A gate that passes an arm which is an echo BY CONSTRUCTION is decoration.
 2. ⭐ **The noise floor.** Every headline delta is printed beside A0b_replicate's delta
    on the same metric -- same flags, same seed, run twice. The paired bootstrap resamples
    windows with the models held FIXED, so it cannot see run-to-run training variance, and
    MEASURED on this rig that variance is not small. A delta no larger than the
    replicate's is NOT attributable to the lever, whatever its CI says.

usage: python verdict.py [panel_report.json]
"""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\run_wbank\panel_report.json")
J = json.load(open(P, encoding="utf-8"))
A = J["arms"]

LON = ["LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2"]
LAT = ["LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps"]
TAC = ["TAC_traj_lat_correct", "TAC_traj_lon_correct"]
OUT = []


def w(s=""):
    OUT.append(s)


def got(name):
    return name in A and A[name].get("status") == "MEASURED"


def d(name, tag, metric):
    """paired delta dict for arm[tag] minus A0, or None."""
    try:
        return A[name]["paired_vs_A0"][tag]["plan_2s"][metric]
    except (KeyError, TypeError):
        return None


def fmt(r):
    if not r or r.get("status") == "UNAVAILABLE":
        return "n/a"
    return (f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
            f"{'SEP' if r.get('separated') else 'ns'}")


def floor(metric, tag):
    """A0b_replicate's delta on the same metric = the eval-level noise floor."""
    r = d("A0b_replicate", tag, metric)
    return r


def attributable(r, f):
    """Is a delta larger than the same-config replicate's, in the same direction?"""
    if not r or r.get("status") == "UNAVAILABLE":
        return None
    if not r.get("separated"):
        return False
    if f is None or f.get("status") == "UNAVAILABLE":
        return None                      # no floor measured -> cannot say
    return abs(r["delta"]) > abs(f["delta"])


w(f"# H-EGO-LIT-4 — pre-registered outcome evaluation")
w(f"_source: {P.name}; {J.get('_scope','')}_")
w(f"_n_windows {J.get('n_windows')} from {J.get('n_episodes_pool')} val episodes; "
  f"estimator {J.get('estimator')}_\n")

# ---------------------------------------------------------------- OUTCOME IV
w("## OUTCOME IV (checked FIRST) — does the gate FAIL the deliberate regression?\n")
if not got("A5_regress"):
    w("⛔ A5_regress ABSENT — the panel cannot be validated. VOID.")
    void = True
else:
    g = A["A5_regress"]["GATE"]
    v2b = A["A5_regress"]["gate2b"]["verdict"]
    scene = A["A5_regress"]["gate2b"]["sources"]["scene"]["degradation_rel"]
    raised = bool(g.get("raised"))
    w(f"A5_regress (`--ablate-frames`, an echo BY CONSTRUCTION): gate2b **{v2b}**, "
      f"scene degradation {scene:+.4f}, gate raised **{raised}**")
    void = not raised
    if raised:
        w(f"\n✅ the gate FAILS the regression ⇒ the panel is VALID.")
        if v2b == "ECHOING":
            w("   verdict is ECHOING — the informative failure: wrong ego hurts, wrong scene does not.")
        else:
            w(f"   ⚠️ verdict is {v2b}, not ECHOING. The gate still FAILED the arm (correct), but "
              f"{v2b} is the UNPOWERED reading — it says the probe could not establish USE of "
              "either input, not that the scene specifically was ignored. Report it as such.")
    else:
        w("\n⛔⛔ the gate PASSED an arm that is an echo by construction ⇒ **PANEL VOID** "
          "(SPEC OUTCOME IV). No other row below is admissible.")
w()

# ---------------------------------------------------------------- noise floor
w("## The run-to-run noise floor (added mid-panel; not in the original pre-registration)\n")
if got("A0b_replicate"):
    w("A0b_replicate = A0's flags and A0's seed, run a second time. Its paired delta vs A0 "
      "is pure nondeterminism at the level the panel scores.\n")
    w("| metric | withheld regime | kept regime |")
    w("|---|---|---|")
    for m in LON + LAT + ["ade_m"]:
        w(f"| {m} | {fmt(floor(m,'w'))} | {fmt(floor(m,'k'))} |")
    w("\n⚠️ Any arm's delta that is not clearly LARGER than the corresponding cell is not "
      "attributable to its lever, regardless of its own CI.")
else:
    w("⚠️ A0b_replicate NOT present — only the train-log floor (noise_floor.py) is available, "
      "and no delta below can be separated from training nondeterminism at eval level.")
w()

# ---------------------------------------------------------------- the echo instrument
w("## The echo instrument, kept regime (the deployed regime)\n")
w("| arm | bank | gate2b verdict | scene rel | ego rel | gate raised |")
w("|---|---|---|---|---|---|")
for n in sorted(A):
    if not got(n):
        w(f"| {n} | — | {A[n].get('status')} | | | |")
        continue
    r = A[n]["gate2b"]
    w(f"| {n} | {A[n]['withheld_bank'].get('mode')} | **{r['verdict']}** | "
      f"{r['sources']['scene']['degradation_rel']:+.4f} | "
      f"{r['sources']['ego']['degradation_rel']:+.4f} | "
      f"{bool(A[n]['GATE'].get('raised'))} |")
c = A.get("A0_fixed", {}).get("gate2b_constant_predictor_control")
if c:
    w(f"\nconstant-predictor control (MUST read exactly 0.0000 / 0.0000): "
      f"scene {c['sources']['scene']['degradation_rel']:+.4f}, "
      f"ego {c['sources']['ego']['degradation_rel']:+.4f}")
w()

# ---------------------------------------------------------------- the outcomes
w("## The pre-registered outcomes\n")
fired = []


def row(label, cond, detail):
    mark = "🔥 FIRES" if cond is True else ("— no" if cond is False else "? undecided")
    w(f"- **{label}** — {mark}\n    {detail}")
    if cond is True:
        fired.append(label)


# (1) A1 adopt
if got("A1_pred"):
    v = A["A1_pred"]["gate2b"]["verdict"]
    sp = d("A1_pred", "w", "LON_speed_mae_mps")
    al = d("A1_pred", "w", "LON_along_mae_m")
    better = bool(sp and sp.get("separated") and sp["delta"] < 0
                  and al and al.get("separated") and al["delta"] < 0)
    a12 = None
    try:
        a12 = J["A1_minus_A2"]["w"]["LON_speed_mae_mps"]
    except (KeyError, TypeError):
        pass
    ctrl_ok = bool(a12 and a12.get("separated") and a12["delta"] < 0)
    att = attributable(sp, floor("LON_speed_mae_mps", "w"))
    row("ADOPT `pred` for refcv5",
        (v == "READS_BOTH" and better and ctrl_ok and att is not False) if att is not None
        else None,
        f"A1 gate2b={v}; withheld speed {fmt(sp)}; withheld along {fmt(al)}; "
        f"A1−A2 speed {fmt(a12)}; larger than replicate floor: {att}")
    row("REFUSE `pred` (A1 reads ECHOING ⇒ the loop echoes its own prior)",
        v == "ECHOING",
        f"A1 gate2b={v}, scene rel "
        f"{A['A1_pred']['gate2b']['sources']['scene']['degradation_rel']:+.4f}")
    row("BANK GEOMETRY IS NOT THE BINDING CONSTRAINT (A1 ≈ A2)",
        bool(a12 and not a12.get("separated")),
        f"A1−A2 withheld speed {fmt(a12)} — a control that matches the arm means the gain "
        f"is not information")

# (2) A4 retire the per-window roll
if got("A4_none"):
    sp4 = d("A4_none", "w", "LON_speed_mae_mps")
    v4 = A["A4_none"]["gate2b"]["verdict"]
    row("RETIRE the per-window roll (A4 ≥ A0 on echo AND families)",
        bool(sp4 and sp4.get("separated") and sp4["delta"] < 0 and v4 == "READS_BOTH"),
        f"A4 gate2b={v4}; withheld speed {fmt(sp4)}")

# (3) A3 PlanTF
if got("A3_drop25"):
    sp3 = d("A3_drop25", "k", "LON_speed_mae_mps")
    v3v = A["A3_drop25"]["gate2b"]["verdict"]
    row("PlanTF signature — keep ego_dropout 0.5 (A3 better families, lost separation)",
        bool(sp3 and sp3.get("separated") and sp3["delta"] < 0 and v3v != "READS_BOTH"),
        f"A3 gate2b={v3v}; kept speed {fmt(sp3)}")

# (4) A2 vs A0
if got("A2_random"):
    sp2 = d("A2_random", "w", "LON_speed_mae_mps")
    row("10 m/s IS A BIASED CHOICE (A2 ≫ A0 ⇒ re-run A0 at the marginal's mean)",
        bool(sp2 and sp2.get("separated") and sp2["delta"] < 0),
        f"A2 withheld speed {fmt(sp2)}; training marginal mean "
        f"{(A['A2_random'].get('random_pool_rebuilt') or {}).get('mean_ms')} m/s vs the fixed 10.0")
    row("THE HARM IS ANY SPEED-BLIND BANK, NOT 10 m/s (A2 ≈ A0)",
        bool(sp2 and not sp2.get("separated")),
        f"A2 withheld speed {fmt(sp2)}")

w()
w("## Summary\n")
if void:
    w("⛔ **PANEL VOID** — the gate did not fail the deliberate regression.")
else:
    w(f"outcomes fired: {fired if fired else 'NONE — the panel is inconclusive on its own terms'}")
w()
w("_RIG scope: ~19 M params, 48 non-parity training episodes, 2,000 steps, one seed per "
  "arm. Validates a DESIGN and a GATE, never a model claim (H-SCALE-2); nothing here "
  "enters MODEL_REGISTRY.md._")

text = "\n".join(OUT)
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write(text)
try:
    print(text)
except Exception:
    print(text.encode("ascii", "replace").decode("ascii"))
