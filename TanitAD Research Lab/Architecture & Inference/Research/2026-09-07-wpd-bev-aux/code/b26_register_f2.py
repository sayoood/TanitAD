# -*- coding: utf-8 -*-
"""WP-D step 26 -- carry the PLANNER replicate floor into `E-BEV-AUX-1`.

Reads the register FRESH (degraded mount, interleaved same-breath control),
makes TARGETED replacements only, writes to LOCAL disk. Same discipline as b22.
"""
import io
import json
import os
import sys
import time

RAW = r"C:\Users\Admin\wpd-probe\raw"
REG = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Project Steering\GOALS_AND_CLAIMS.md"
CTRL = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\CLAUDE.md"
OUT = r"C:\Users\Admin\wpd-probe\GOALS_AND_CLAIMS.new.md"
L = lambda n: json.load(io.open(os.path.join(RAW, n), encoding="utf-8"))
D1, FB, FC = L("paired_5b.json"), L("paired_floor_wpdD0b_vs_D0.json"), L("paired_floor_wpdD0c_vs_D0.json")
fl = lambda k: max(abs(FB["metrics"][k]["delta"]), abs(FC["metrics"][k]["delta"]))
rt = lambda k: abs(D1["metrics"][k]["delta"]) / fl(k)
nsep = sum(1 for k in D1["metrics"] if FB["metrics"][k]["separated"])

src = None
for i in range(120):
    ok_c = False
    try:
        ok_c = len(io.open(CTRL, "rb").read()) > 1000
    except Exception:
        pass
    try:
        src = io.open(REG, encoding="utf-8").read()
        print(f"[read] OK attempt {i}, {len(src)} chars, control_ok={ok_c}")
        break
    except Exception as e:
        if i % 10 == 0:
            print(f"[read] attempt {i}: control_ok={ok_c} FAIL {e!r}"[:100], flush=True)
        time.sleep(3)
if src is None:
    print("EXHAUSTED -- INCONCLUSIVE (the mount). Nothing written.")
    sys.exit(2)


def sub(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)


sub("(every metric is an ERROR, so positive = worse).",
 f"(every metric is an ERROR, so positive = worse). "
 f"⛔⛔ **CORRECTED 2026-09-10 — `8 of 9` IS WRONG AND THE ADE HEADLINE IS RIG NOISE.** The two "
 f"replicate checkpoints went through the IDENTICAL T1 arm (`raw/paired_floor_wpdD0b_vs_D0.json`, "
 f"`raw/paired_floor_wpdD0c_vs_D0.json`; 3,422 windows / 40 episodes each, `eid`/`window_start`/"
 f"`gt`/`v0` element-wise identical, mean |pred(D0b)−pred(D0)| = {FB['mean_abs_pred_diff_m']:.4f} m). "
 f"**`D0b` — D0's flags, D0's SEED, ZERO levers moved — is itself *separably worse* than D0 on "
 f"{nsep} OF THE SAME 9 METRICS ({100.0*nsep/9:.1f} %),** reproducing the headline ADE **+0.02610** "
 f"at **+0.02460** and EXCEEDING the lever on all three LONGITUDINAL metrics. Against a per-metric "
 f"floor `max(|D0b−D0|, |D0c−D0|)`: **ADE {rt('ADE_m'):.2f}×, FDE {rt('FDE_m'):.2f}×, LON 0.40–0.60× "
 f"— ALL INSIDE THE FLOOR**; only **LATERAL heading {rt('LAT_heading_mae_deg'):.1f}×, curvature "
 f"{rt('LAT_curvature_mae_1pm'):.1f}×, yaw-rate {rt('LAT_yawrate_mae_radps'):.1f}×** clear 3× (cross-track "
 f"{rt('LAT_cross_mae_m'):.2f}× does not), and on all four lateral metrics the replicates are "
 f"unseparated and mostly the OPPOSITE sign. ⇒ **`F2` survives ONLY as a LATERAL claim on 3 of 9 "
 f"metrics** — *\"the aux term costs LATERAL accuracy (heading, curvature, yaw-rate)\"* — and NOT as "
 f"*\"the planner is worse\"*. ⭐⭐ **The four-family rule earned itself in reverse here: ADE was the "
 f"SPURIOUS part, and an ADE-only report would have made `F2` entirely false.** ⚠️ This rig's "
 f"false-positive rate for `separated` on a one-seed pair is **{100.0*nsep/9:.1f} %**, against the "
 f"**14.3 %** `CLAUDE.md` records for the v7-tiny rig — a new measured fact about the REF-C planner "
 f"rig, registered under `H-ESTIM-SEED-1`.")

sub("⇒ **the claim fails on BOTH halves of its own success condition, for consistent reasons: the "
    "target's INFORMATION did nothing (a shuffled target does the same) and the term's GRADIENT "
    "cost the planner accuracy",
    "⇒ ⛔ **CORRECTED 2026-09-10: the claim still FAILS its success condition, but ONLY because "
    "`A2` was never demonstrated (a straddling CI, which no noise floor rescues) — NOT for the two "
    "reasons given below, both of which are now inside their rigs' replicate floors.** The "
    "superseded reasoning read: *the claim fails on BOTH halves of its own success condition, for "
    "consistent reasons: the target's INFORMATION did nothing (a shuffled target does the same) "
    "and the term's GRADIENT cost the planner accuracy")

io.open(OUT, "w", encoding="utf-8", newline="\n").write(src)
print("WROTE", OUT, len(src), "chars")
for t in ("CORRECTED 2026-09-10", "survives ONLY as a LATERAL claim"):
    print(f"  contains {t!r}: {src.count(t)}")
