"""H2 L2 — THE HELD-OUT CONFIRMATION. One threshold, one look, one verdict.

tau* and the resolvable decision are READ FROM `l2_dev.json`. This script may not choose either.
Section 1 is the pre-registered verdict; everything after it is descriptive and cannot move it.

usage:  python l2_confirm.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E01 = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-25-h2-e0-e1", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, E01)
from h2e_stats import paired_cluster_diff, paired_cluster_lift               # noqa: E402
from taniteval.ci import _draws, episode_index                               # noqa: E402
from l2_label import (CONFIRM_CHUNKS, DEV_CHUNKS, TAU_GRID, response_l1,     # noqa: E402
                      response_r2, trigger_l1, trigger_l2, trigger_l2_percam)

TAB = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\l2tab")
OUT = os.path.abspath(os.path.join(HERE, ".."))

PASS_LIFT = 1.5          # pre-registered, same bar E1 used
MIN_POS_FRAMES = 200
MIN_POS_CLUSTERS = 40
SPEED_BINS = [0, 3, 6, 9, 12, 15, 1e9]

dev = json.load(open(os.path.join(OUT, "l2_dev.json")))
TAU = dev["tau_star"]
RESOLVABLE = dev["A2_decision_exclude_unresolvable"]
assert TAU is not None, "DEV produced no powered tau -> UNDERPOWERED, nothing to confirm"

parts = [pd.read_parquet(os.path.join(TAB, f"l2_{c}.parquet")) for c in CONFIRM_CHUNKS
         if os.path.exists(os.path.join(TAB, f"l2_{c}.parquet"))]
D = pd.concat(parts, ignore_index=True)
assert not (set(D.chunk.unique()) & set(DEV_CHUNKS)), "DEV LEAKED INTO CONFIRM"
eid = D.clip_id.to_numpy()
print(f"CONFIRM: {D.chunk.nunique()} chunks / {D.clip_id.nunique():,} clips / {len(D):,} frames")
print(f"  tau* = {TAU} m/s^2   (read from l2_dev.json, not chosen here)")
print(f"  exclude-unresolvable = {RESOLVABLE}")

R = response_r2(D, freeflow=None)
res = {"split": "CONFIRM", "chunks": sorted(D.chunk.unique().tolist()),
       "n_clips": int(D.clip_id.nunique()), "n_frames": int(len(D)),
       "dev_chunks_present": 0,
       "tau_star": TAU, "tau_star_source": "l2_dev.json (power rule, never reads the lift)",
       "exclude_unresolvable": RESOLVABLE,
       "pass_bar_lift": PASS_LIFT,
       "response_base_rate": round(float(R.mean()), 5)}

# ============================================================ 1. THE VERDICT (pre-registered)
g = trigger_l2(D, TAU, resolvable=RESOLVABLE)
lift = paired_cluster_lift(R, g, eid)                       # B = 2000, seed 0
diff = paired_cluster_diff(R, g, eid)
n_cl = int(pd.Series(eid[g]).nunique()) if g.any() else 0
powered = (int(g.sum()) >= MIN_POS_FRAMES) and (n_cl >= MIN_POS_CLUSTERS)
verdict = ("GO" if (lift["excludes_1_above"] and lift["lift"] >= PASS_LIFT and powered)
           else ("UNDERPOWERED" if not powered else "BOUND"))
res["VERDICT"] = {"verdict": verdict, "lift": lift, "difference_form": diff,
                  "trigger_rate": round(float(g.mean()), 6),
                  "n_trigger_frames": int(g.sum()), "n_trigger_clusters": n_cl,
                  "powered": bool(powered)}
print("\n" + "=" * 80)
print(f"L2 HELD-OUT VERDICT (pre-registered, tau = {TAU} m/s^2 only):  {verdict}")
print("=" * 80)
print(f"  held-out lift        : {lift['lift']:.2f}x   95% CI [{lift['lo']:.4f}, {lift['hi']:.4f}]")
print(f"  P(R2 | trigger+)     : {lift['p_resp_given_pos']:.2%}  (n = {lift['n_pos']:,} frames)")
print(f"  P(R2 | trigger-)     : {lift['p_resp_given_neg']:.2%}  (n = {lift['n_neg']:,} frames)")
print(f"  risk difference      : {diff['delta']:+.4f}  CI [{diff['lo']:+.4f}, {diff['hi']:+.4f}]"
      f"  separated={diff['separated']}")
print(f"  trigger rate         : {g.mean():.4%} of frames")
print(f"  episode-clusters     : {lift['n_episodes']:,}  ({n_cl} trigger-positive)")
print(f"  estimator            : {lift['estimator']}")
print(f"  draws used/skipped   : {lift['n_draws_used']}/{lift['n_draws_skipped']}")

# ============================================================ 2. the FULL response curve (descriptive)
curve = []
for tau in TAU_GRID:
    gg = trigger_l2(D, tau, resolvable=RESOLVABLE)
    if gg.sum() < 5:
        curve.append(dict(tau=tau, n_pos=int(gg.sum()), lift=None))
        continue
    L = paired_cluster_lift(R, gg, eid, n_boot=1000)
    curve.append(dict(tau=tau, rate=round(float(gg.mean()), 6), n_pos=int(gg.sum()),
                      n_pos_clusters=int(pd.Series(eid[gg]).nunique()),
                      lift=L["lift"], lo=L["lo"], hi=L["hi"],
                      p_pos=L["p_resp_given_pos"], p_neg=L["p_resp_given_neg"]))
res["CONFIRM_response_curve"] = curve
print("\n-- FULL response curve on CONFIRM (descriptive; only tau* decides) --")
print(f"{'tau':>5} {'rate':>9} {'n+':>7} {'eps+':>6} {'P(R2|+)':>9} {'P(R2|-)':>9} {'lift':>7}  95% CI")
for c in curve:
    if c.get("lift") is None:
        print(f"{c['tau']:5.1f} {'--':>9} {c['n_pos']:7d}  (too few)")
        continue
    print(f"{c['tau']:5.1f} {c['rate']:9.4%} {c['n_pos']:7d} {c['n_pos_clusters']:6d} "
          f"{c['p_pos']:9.2%} {c['p_neg']:9.2%} {c['lift']:7.2f}x  [{c['lo']:.2f}, {c['hi']:.2f}]")

# ============================================================ 3. coverage + class balance
gl, gr = trigger_l2_percam(D, TAU, resolvable=RESOLVABLE)
lab = g & R
clsL = D.cls_L.to_numpy()
clsR = D.cls_R.to_numpy()
cls_counts = pd.concat([pd.Series(clsL[gl]), pd.Series(clsR[gr])]).value_counts().to_dict()
res["coverage"] = {
    "trigger_rate": round(float(g.mean()), 6), "label_rate": round(float(lab.mean()), 6),
    "n_trigger_frames": int(g.sum()), "n_label_frames": int(lab.sum()),
    "n_clips_total": int(D.clip_id.nunique()),
    "n_clips_trigger_positive": n_cl,
    "n_clips_label_positive": int(pd.Series(eid[lab]).nunique()) if lab.any() else 0,
    "left_rate": round(float(gl.mean()), 6), "right_rate": round(float(gr.mean()), 6),
    "both_cameras_rate": round(float((gl & gr).mean()), 6),
    "mean_cameras_per_frame_instantaneous": round(1.0 + float(gl.mean()) + float(gr.mean()), 5),
    "triggering_agent_class": {str(k): int(v) for k, v in cls_counts.items()},
    "response_base_rate": round(float(R.mean()), 5),
    "L1_response_base_rate_for_contrast": round(float(response_l1(D).mean()), 5)}
print("\n-- coverage / class balance at tau* --")
for k, v in res["coverage"].items():
    print(f"   {k:38s} {v}")

# ============================================================ 4. strata
strata = {}
for nm, m in [("junction", D.junction.to_numpy()), ("lane_change", D.lane_change.to_numpy())]:
    for inout, mm in [("in", m), ("out", ~m)]:
        sub = mm
        if g[sub].sum() < 5 or sub.sum() < 100:
            strata[f"{nm}_{inout}"] = {"n_frames": int(sub.sum()), "n_pos": int(g[sub].sum()),
                                       "lift": None}
            continue
        L = paired_cluster_lift(R[sub], g[sub], eid[sub], n_boot=1000)
        strata[f"{nm}_{inout}"] = {"n_frames": int(sub.sum()), "n_pos": int(g[sub].sum()),
                                   "trigger_rate": round(float(g[sub].mean()), 6),
                                   "lift": L["lift"], "lo": L["lo"], "hi": L["hi"]}
res["strata"] = strata
print("\n-- strata (kinematic detectors, situ_full.py thresholds verbatim) --")
for k, v in strata.items():
    if v["lift"] is None:
        print(f"   {k:16s} n={v['n_frames']:7d} n+={v['n_pos']:5d}  (too few)")
    else:
        print(f"   {k:16s} n={v['n_frames']:7d} n+={v['n_pos']:5d} rate {v['trigger_rate']:7.4%}"
              f"  lift {v['lift']:5.2f}x [{v['lo']:.2f}, {v['hi']:.2f}]")


# ============================================================ 5. speed-matched (Mantel-Haenszel)
def mh_lift(resp, gate, v, bins):
    b = np.digitize(v, bins[1:-1])
    num = den = 0.0
    for s in np.unique(b):
        m = b == s
        n1 = gate[m].sum()
        n0 = (~gate[m]).sum()
        N = m.sum()
        if n1 == 0 or n0 == 0:
            continue
        num += resp[m & gate].sum() * n0 / N
        den += resp[m & ~gate].sum() * n1 / N
    return num / den if den > 0 else float("nan")


v = D.ego_v.to_numpy()
uniq, idx_by_ep = episode_index(eid)
pt = mh_lift(R, g, v, SPEED_BINS)
bs = []
for sel in _draws(uniq, idx_by_ep, 1000, 0):
    x = mh_lift(R[sel], g[sel], v[sel], SPEED_BINS)
    if np.isfinite(x):
        bs.append(x)
lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) > 50 else (float("nan"), float("nan"))
res["speed_matched_MH"] = {"lift": round(float(pt), 4), "lo": round(float(lo), 4),
                           "hi": round(float(hi), 4), "bins_m_s": SPEED_BINS[:-1] + ["inf"],
                           "mean_v_trigger_pos": round(float(v[g].mean()), 3),
                           "mean_v_trigger_neg": round(float(v[~g].mean()), 3),
                           "estimator": "Mantel-Haenszel pooled RR, episode-cluster bootstrap B=1000"}
print(f"\n-- speed-matched MH lift: {pt:.2f}x [{lo:.2f}, {hi:.2f}]   "
      f"(mean v: trigger+ {v[g].mean():.2f} vs trigger- {v[~g].mean():.2f} m/s)")

# ============ 6. already-braking stratifier (the clause amendment A1 removed from the definition)
ff = D.alon_pre.to_numpy() >= -0.5
sens_ff = {}
for nm, m in [("free_flow_at_t", ff), ("already_braking_at_t", ~ff)]:
    if g[m].sum() < 5:
        sens_ff[nm] = {"n_pos": int(g[m].sum()), "lift": None}
        continue
    L = paired_cluster_lift(R[m], g[m], eid[m], n_boot=1000)
    sens_ff[nm] = {"n_frames": int(m.sum()), "n_pos": int(g[m].sum()), "lift": L["lift"],
                   "lo": L["lo"], "hi": L["hi"], "p_pos": L["p_resp_given_pos"],
                   "p_neg": L["p_resp_given_neg"]}
res["stratified_by_already_braking"] = sens_ff
print("\n-- A1 stratifier: the free-flow clause reported instead of imposed --")
for k, s in sens_ff.items():
    if s["lift"] is None:
        print(f"   {k:22s} n+={s['n_pos']} (too few)")
    else:
        print(f"   {k:22s} n+={s['n_pos']:5d}  lift {s['lift']:5.2f}x [{s['lo']:.2f}, {s['hi']:.2f}]")

# ============================================================ 7. sensitivities (all descriptive)
sens = {}


def add(nm, gate, resp=None, nb=1000):
    rr = R if resp is None else resp
    if gate.sum() < 5:
        sens[nm] = {"n_pos": int(gate.sum()), "lift": None}
        return
    L = paired_cluster_lift(rr, gate, eid, n_boot=nb)
    sens[nm] = {"rate": round(float(gate.mean()), 6), "n_pos": int(gate.sum()),
                "n_pos_clusters": int(pd.Series(eid[gate]).nunique()),
                "lift": L["lift"], "lo": L["lo"], "hi": L["hi"],
                "p_pos": L["p_resp_given_pos"], "p_neg": L["p_resp_given_neg"]}


add("PRIMARY (ego path-speed, agent CV, resolvable, crop scope)", g)
add("S1 ego pure-CV straight (isolates M2 = the route fix)",
    trigger_l2(D, TAU, ego="cv"))
add("S2 agent REALISED future (isolates M1 = the agent-reaction fix)",
    trigger_l2(D, TAU, agent="real"))
add("S3 unresolvable INCLUDED (undoes amendment A2)",
    trigger_l2(D, TAU, resolvable=False))
add("S4 residual scope: outside the FULL 120.5 deg front field",
    trigger_l2(D, TAU, resolvable=RESOLVABLE, scope="residual"))
for br in (-1.5, -2.5, -3.0):
    add(f"S5 response brake threshold {br} m/s^2", g, resp=response_r2(D, brake=br, freeflow=None))
add("S6 response WITH the pre-registered free-flow clause (undoes A1)", g,
    resp=response_r2(D, freeflow=-0.5))
add("S7 response = the REFUTED L1 one (dv4 <= -1 m/s)", g, resp=response_l1(D))
res["sensitivities"] = sens
print("\n-- sensitivities (descriptive; none can move the verdict) --")
for k, s in sens.items():
    if s["lift"] is None:
        print(f"   {k:60s} n+={s['n_pos']:5d} (too few)")
    else:
        print(f"   {k:60s} n+={s['n_pos']:5d} lift {s['lift']:6.2f}x [{s['lo']:.2f}, {s['hi']:.2f}]")

# ============================================ 8. the refuted L1_gate on the IDENTICAL CONFIRM frames
gl1 = trigger_l1(D)
res["L1_head_to_head_same_frames"] = {
    "L1_gate_rate": round(float(gl1.mean()), 6),
    "L1_gate x L1_response (the refuted pair)": paired_cluster_lift(response_l1(D), gl1, eid, n_boot=1000),
    "L1_gate x R2_response (new response, old trigger)": paired_cluster_lift(R, gl1, eid, n_boot=1000),
    "L2_trigger x L1_response (new trigger, old response)": paired_cluster_lift(response_l1(D), g, eid, n_boot=1000),
    "L2_trigger x R2_response (both new) = THE VERDICT": {"lift": lift["lift"], "lo": lift["lo"], "hi": lift["hi"]}}
print("\n-- 2x2: which half of the fix carries the signal? (same CONFIRM frames) --")
for k, vv in res["L1_head_to_head_same_frames"].items():
    if isinstance(vv, dict) and "lift" in vv:
        print(f"   {k:56s} {vv['lift']:6.2f}x [{vv['lo']:.2f}, {vv['hi']:.2f}]")

# ============================================================ 9. per-chunk (no single chunk drives it)
per = []
for ch, sub in D.groupby("chunk"):
    m = (D.chunk == ch).to_numpy()
    if g[m].sum() < 5:
        per.append(dict(chunk=str(ch), n_clips=int(sub.clip_id.nunique()),
                        n_pos=int(g[m].sum()), lift=None))
        continue
    L = paired_cluster_lift(R[m], g[m], eid[m], n_boot=500)
    per.append(dict(chunk=str(ch), n_clips=int(sub.clip_id.nunique()), n_pos=int(g[m].sum()),
                    lift=L["lift"], lo=L["lo"], hi=L["hi"]))
res["per_chunk"] = per
ok = [p["lift"] for p in per if p["lift"] is not None]
res["per_chunk_summary"] = {"n_scored": len(ok), "n_above_1": int(sum(x > 1 for x in ok)),
                            "median": round(float(np.median(ok)), 3) if ok else None}
print("\n-- per-chunk --")
for p in per:
    if p["lift"] is None:
        print(f"   {p['chunk']}: n+={p['n_pos']:4d} (too few)")
    else:
        print(f"   {p['chunk']}: clips {p['n_clips']:3d} n+={p['n_pos']:5d}  "
              f"lift {p['lift']:6.2f}x [{p['lo']:.2f}, {p['hi']:.2f}]")
print(f"   summary: {res['per_chunk_summary']}")

# ============================================================ 10. C-EFF on the L2 trigger
for name, hyst in (("instantaneous", 0), ("hyst_1s", 10), ("hyst_2s", 20)):
    if hyst == 0:
        L_, R_ = gl, gr
    else:
        L_ = np.zeros(len(D), bool)
        R_ = np.zeros(len(D), bool)
        for cid, sub in D.groupby("clip_id", sort=False):
            i = sub.index.to_numpy()
            for src, dst in ((gl, L_), (gr, R_)):
                s = src[i]
                if not s.any():
                    continue
                k = np.convolve(s.astype(float), np.ones(2 * hyst + 1), mode="same") > 0
                dst[i] = k
    res.setdefault("C_EFF_on_L2", {})[name] = {
        "left": round(float(L_.mean()), 6), "right": round(float(R_.mean()), 6),
        "either": round(float((L_ | R_).mean()), 6),
        "cams_per_frame": round(1.0 + float(L_.mean()) + float(R_.mean()), 5),
        "saved_vs_7": round(1 - (1.0 + float(L_.mean()) + float(R_.mean())) / 7.0, 4)}
print("\n-- C-EFF re-derived on the L2 trigger --")
for k, vv in res["C_EFF_on_L2"].items():
    print(f"   {k:14s} either {vv['either']:8.4%}  cams/frame {vv['cams_per_frame']:.4f}  "
          f"saved vs 7 {vv['saved_vs_7']:.2%}")

json.dump(res, open(os.path.join(OUT, "l2_confirm.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'l2_confirm.json')}")
print(f"\n>>> VERDICT: {verdict} <<<")
