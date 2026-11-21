"""H2 L2 — DEV split. Design diagnostics + the POWER-RULE selection of tau*.

CONFIRM IS NOT TOUCHED BY THIS SCRIPT. It asserts that on load.

Everything here is design work on the development split, which is exactly what a development split
is for. The one number that decides the workstream is produced by `l2_confirm.py`, once, at the
tau* this script writes into `l2_dev.json`.

usage:  python l2_dev.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E01 = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-25-h2-e0-e1", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, E01)
from h2e_stats import paired_cluster_lift                                  # noqa: E402
from l2_label import (CONFIRM_CHUNKS, DEV_CHUNKS, TAU_GRID, response_l1,   # noqa: E402
                      response_r2, trigger_l1, trigger_l2, trigger_l2_percam)

TAB = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\l2tab")
OUT = os.path.abspath(os.path.join(HERE, ".."))

# power floor, pre-registered Sec 3
MIN_POS_FRAMES = 200
MIN_POS_CLUSTERS = 40


def load(chunks):
    parts = []
    for ch in chunks:
        p = os.path.join(TAB, f"l2_{ch}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True)


D = load(DEV_CHUNKS)
assert not (set(D.chunk.unique()) & set(CONFIRM_CHUNKS)), "CONFIRM LEAKED INTO DEV"
have = sorted(D.chunk.unique())
eid = D.clip_id.to_numpy()
print(f"DEV: {D.chunk.nunique()} chunks {have}\n     {D.clip_id.nunique():,} clips / {len(D):,} frames")

res = {"split": "DEV", "chunks": have, "n_clips": int(D.clip_id.nunique()),
       "n_frames": int(len(D)), "confirm_chunks_present": 0,
       "power_floor": {"min_pos_frames": MIN_POS_FRAMES, "min_pos_clusters": MIN_POS_CLUSTERS}}

# ------------------------------------------------------------------ 1. RESPONSE base rates
R_pre = response_r2(D, freeflow=-0.5)          # as pre-registered
R_amd = response_r2(D, freeflow=None)          # amendment A1 candidate
R_l1 = response_l1(D)
res["response_base_rates"] = {
    "R2_prereg_with_freeflow": round(float(R_pre.mean()), 5),
    "R2_amended_no_freeflow": round(float(R_amd.mean()), 5),
    "L1_response_dv4_le_-1 (the refuted one)": round(float(R_l1.mean()), 5),
    "prereg_band": [0.01, 0.15]}
print("\n== RESPONSE base rates (DEV) ==")
for k, v in res["response_base_rates"].items():
    if isinstance(v, float):
        print(f"   {k:42s} {v:8.3%}")

# ------------------------------------------- 2. AMENDMENT A1 diagnostic: does free-flow anti-select?
a1 = []
for tau in TAU_GRID:
    g = trigger_l2(D, tau, resolvable=True)
    if g.sum() < 5:
        continue
    a1.append(dict(tau=tau, n_pos=int(g.sum()),
                   freeflow_rate_trigger=round(float((D.alon_pre.to_numpy() >= -0.5)[g].mean()), 4),
                   freeflow_rate_all=round(float((D.alon_pre.to_numpy() >= -0.5).mean()), 4),
                   vmin_rate_trigger=round(float((D.ego_v.to_numpy() >= 3.0)[g].mean()), 4),
                   vmin_rate_all=round(float((D.ego_v.to_numpy() >= 3.0).mean()), 4)))
res["A1_freeflow_diagnostic"] = a1
print("\n== A1: is the pre-registered free-flow clause anti-selecting the positives? ==")
for r in a1:
    print(f"   tau={r['tau']:4.1f} n+={r['n_pos']:5d}  P(free-flow | trigger) {r['freeflow_rate_trigger']:6.1%}"
          f"   vs all frames {r['freeflow_rate_all']:6.1%}")

# ------------------- 3. AMENDMENT A2 diagnostic — DECISION RULE STATED IN THE RESULTS DOC BEFOREHAND
#   Pop U = only UNRESOLVABLE off-front conflicts (braking never clears them)
#   Pop R = at least one RESOLVABLE off-front conflict
#   Pop N = no off-front conflict at all
#   Rule: if P(R2 | U) < P(R2 | R), the unresolvable population is dilution and is excluded.
off_all = np.maximum(D.areq_off_L.to_numpy(), D.areq_off_R.to_numpy())
off_res = np.maximum(D.areq_off_L_res.to_numpy(), D.areq_off_R_res.to_numpy())
clean = D.areq_seen.to_numpy() == 0.0
popU = clean & (off_res == 0.0) & (off_all >= 8.0)
popR = clean & (off_res >= 0.5)
popN = clean & (off_all == 0.0)
res["A2_resolvable_diagnostic"] = {
    "P_R2_given_unresolvable_only": round(float(R_amd[popU].mean()), 5), "n_U": int(popU.sum()),
    "P_R2_given_resolvable": round(float(R_amd[popR].mean()), 5), "n_R": int(popR.sum()),
    "P_R2_given_no_offfront": round(float(R_amd[popN].mean()), 5), "n_N": int(popN.sum()),
    "share_of_areq_ge_2_population_that_is_unresolvable": round(
        float((off_all >= 8.0).sum() / max((off_all >= 2.0).sum(), 1)), 4)}
DECIDE_RESOLVABLE = bool(R_amd[popU].mean() < R_amd[popR].mean())
res["A2_decision_exclude_unresolvable"] = DECIDE_RESOLVABLE
print("\n== A2: is the not-avoidable-by-braking population signal or dilution? ==")
d2 = res["A2_resolvable_diagnostic"]
print(f"   P(R2 | unresolvable only) {d2['P_R2_given_unresolvable_only']:7.3%}  n={d2['n_U']:6d}")
print(f"   P(R2 | resolvable)        {d2['P_R2_given_resolvable']:7.3%}  n={d2['n_R']:6d}")
print(f"   P(R2 | no off-front)      {d2['P_R2_given_no_offfront']:7.3%}  n={d2['n_N']:6d}")
print(f"   -> exclude unresolvable from the trigger: {DECIDE_RESOLVABLE}")

# ------------------------------------------------------------------ 4. tau* by the POWER RULE
rows = []
for tau in TAU_GRID:
    g = trigger_l2(D, tau, resolvable=DECIDE_RESOLVABLE)
    rows.append(dict(tau=tau, n_pos=int(g.sum()), rate=round(float(g.mean()), 6),
                     n_pos_clusters=int(pd.Series(eid[g]).nunique()) if g.any() else 0))
res["tau_power_table"] = rows
elig = [r for r in rows
        if r["n_pos"] >= MIN_POS_FRAMES and r["n_pos_clusters"] >= MIN_POS_CLUSTERS]
TAU_STAR = min(r["tau"] for r in elig) if elig else None
res["tau_star"] = TAU_STAR
res["tau_star_rule"] = ("smallest tau in the grid with n+ >= 200 frames AND >= 40 trigger-positive "
                        "episode-clusters on DEV; the rule NEVER reads the lift")
print("\n== tau* by the POWER rule (never reads the lift) ==")
for r in rows:
    ok = "OK " if (r["n_pos"] >= MIN_POS_FRAMES and r["n_pos_clusters"] >= MIN_POS_CLUSTERS) else "   "
    print(f"  {ok} tau={r['tau']:4.1f}  rate {r['rate']:8.4%}  n+ {r['n_pos']:6d}  clusters {r['n_pos_clusters']:5d}")
print(f"  ==> tau* = {TAU_STAR}")

# ------------------------------------------------------- 5. the FULL DEV response curve (descriptive)
curve = []
for tau in TAU_GRID:
    g = trigger_l2(D, tau, resolvable=DECIDE_RESOLVABLE)
    if g.sum() < 5:
        curve.append(dict(tau=tau, n_pos=int(g.sum()), lift=None))
        continue
    L = paired_cluster_lift(R_amd, g, eid, n_boot=1000)
    curve.append(dict(tau=tau, rate=round(float(g.mean()), 6), n_pos=int(g.sum()),
                      lift=L["lift"], lo=L["lo"], hi=L["hi"],
                      p_pos=L["p_resp_given_pos"], p_neg=L["p_resp_given_neg"],
                      n_pos_clusters=int(pd.Series(eid[g]).nunique())))
res["DEV_response_curve"] = curve
print("\n== DEV response curve (DESCRIPTIVE — cannot move tau*) ==")
print(f"{'tau':>5} {'rate':>9} {'n+':>7} {'P(R2|+)':>9} {'P(R2|-)':>9} {'lift':>7}  95% CI")
for c in curve:
    if c.get("lift") is None:
        print(f"{c['tau']:5.1f} {'--':>9} {c['n_pos']:7d}   (too few)")
        continue
    print(f"{c['tau']:5.1f} {c['rate']:9.4%} {c['n_pos']:7d} {c['p_pos']:9.2%} {c['p_neg']:9.2%} "
          f"{c['lift']:7.2f}x  [{c['lo']:.2f}, {c['hi']:.2f}]")

# ---------------------------------------------------- 6. the refuted L1 label on the SAME DEV frames
gl1 = trigger_l1(D)
res["DEV_L1_headtohead"] = {
    "L1_gate_rate": round(float(gl1.mean()), 5),
    "L1_gate_with_L1_response": paired_cluster_lift(R_l1, gl1, eid, n_boot=1000),
    "L1_gate_with_R2_response": paired_cluster_lift(R_amd, gl1, eid, n_boot=1000)}
print("\n== the refuted L1_gate, on these same DEV frames ==")
h = res["DEV_L1_headtohead"]
print(f"   L1_gate rate {h['L1_gate_rate']:.3%}")
print(f"   L1_gate x L1 response : {h['L1_gate_with_L1_response']['lift']:.2f}x "
      f"[{h['L1_gate_with_L1_response']['lo']:.2f}, {h['L1_gate_with_L1_response']['hi']:.2f}]")
print(f"   L1_gate x R2 response : {h['L1_gate_with_R2_response']['lift']:.2f}x "
      f"[{h['L1_gate_with_R2_response']['lo']:.2f}, {h['L1_gate_with_R2_response']['hi']:.2f}]")

# ------------------------------------------------------------------ 7. coverage at tau*
if TAU_STAR is not None:
    g = trigger_l2(D, TAU_STAR, resolvable=DECIDE_RESOLVABLE)
    gl, gr = trigger_l2_percam(D, TAU_STAR, resolvable=DECIDE_RESOLVABLE)
    lab = g & R_amd
    res["DEV_coverage_at_tau_star"] = {
        "tau": TAU_STAR, "trigger_rate": round(float(g.mean()), 6),
        "label_rate": round(float(lab.mean()), 6),
        "n_trigger_frames": int(g.sum()), "n_label_frames": int(lab.sum()),
        "n_trigger_clips": int(pd.Series(eid[g]).nunique()),
        "n_label_clips": int(pd.Series(eid[lab]).nunique()),
        "left_rate": round(float(gl.mean()), 6), "right_rate": round(float(gr.mean()), 6),
        "both_rate": round(float((gl & gr).mean()), 6)}
    print(f"\n== DEV coverage at tau*={TAU_STAR} ==")
    for k, v in res["DEV_coverage_at_tau_star"].items():
        print(f"   {k:20s} {v}")

json.dump(res, open(os.path.join(OUT, "l2_dev.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'l2_dev.json')}")
