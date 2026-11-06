#!/usr/bin/env python3
"""Analysis of refe_ego_probe results.npz per PREREG.md (thresholds fixed there; evaluated mechanically here).  CPU only.

    python analyze_ego_probe.py --run <run dir containing results.npz + results_meta.json> --out <dir> [--boot 2000]

Every interval is a PAIRED EPISODE(LOG)-CLUSTER BOOTSTRAP: logs resampled with replacement, ONE resample matrix shared by every statistic and arm (seed 20261106), 95 % percentile CI.
It answers 'would another draw of logs say this?' only (not training seed, not inference seed beyond control C1).
"""
import argparse
import json
import os

import numpy as np

DT = 0.2
TS = (1.0, 2.0, 4.0)
SEED = 20261106
SLOW_FRAC = 0.15
BIAS_BAR = 0.5


def step_speed(P, ts=TS):
    xy = np.asarray(P, np.float64)[..., :2]
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    v = np.linalg.norm(xy - prev, axis=-1) / DT
    return v[..., [int(round(t / DT)) - 1 for t in ts]]


def gt_speed(pose20, t20, ts=TS):
    """Same one-step chord, with the logged states' TRUE timestamps."""
    out = []
    for T in ts:
        i = int(round(T / DT)) - 1
        dq = np.linalg.norm(pose20[:, i, :2] - (pose20[:, i - 1, :2] if i > 0 else 0.0), axis=-1)
        dtt = t20[:, i] - (t20[:, i - 1] if i > 0 else 0.0)
        out.append(dq / dtt)
    return np.stack(out, -1)


class Boot:
    def __init__(self, log_idx, n_logs, B, seed=SEED):
        self.li, self.L = log_idx, n_logs
        self.idx = np.random.default_rng(seed).integers(0, n_logs, (B, n_logs))

    def mean(self, vals, mask):
        m = mask & np.isfinite(vals)
        S = np.bincount(self.li[m], weights=vals[m], minlength=self.L)
        C = np.bincount(self.li[m], minlength=self.L).astype(float)
        n = int(m.sum())
        if n == 0:
            return {"mean": None, "lo": None, "hi": None, "n": 0, "n_logs": 0}
        pt = float(vals[m].mean())
        cs = C[self.idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = S[self.idx].sum(1) / cs
        bs = bs[np.isfinite(bs)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"mean": pt, "lo": float(lo), "hi": float(hi), "n": n, "n_logs": int((C > 0).sum())}


def fmt(d, nd=3):
    if d["mean"] is None:
        return "n=0"
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def fmt_u(d, nd=3):
    if d["mean"] is None:
        return "n=0"
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Z = np.load(os.path.join(a.run, "results.npz"), allow_pickle=False)
    meta = json.load(open(os.path.join(a.run, "results_meta.json")))
    arms = [str(x) for x in Z["arms"]]
    ai = {x: i for i, x in enumerate(arms)}
    tab_arms = [x for x in arms if x != "A0r"]          # A0r is a CONTROL (run on every k-th token only; its rows elsewhere are copies of A0) -- never tabled as an arm
    N = len(Z["token"])
    logs = np.array([str(x) for x in Z["log"]])
    ulogs, li = np.unique(logs, return_inverse=True)
    boot = Boot(li, len(ulogs), a.boot)
    P = Z["traj_pick"].astype(np.float64)                      # [N, A, 20, 3]
    ego = Z["ego_real"]
    v0, ax0 = ego[:, 6], ego[:, 2]
    gv = Z["gt_valid"].astype(bool)
    vgt = gt_speed(Z["gt_pose20"], Z["gt_t20"])                # [N, 3]
    vgt = np.where(gv[:, None], vgt, np.nan)
    allm = np.ones(N, bool)
    cruise = gv & (v0 > 5.0) & (np.abs(ax0) < 0.3) & (np.abs(vgt[:, 2] - v0) < 1.0)
    cruise_alt_state = gv & (v0 > 5.0) & (np.abs(ax0) < 0.3) & (np.abs(Z["gt_speed20"][:, 19] - v0) < 1.0)
    strata = np.array([str(x) for x in Z["stratum"]])
    vp = {x: step_speed(P[:, ai[x]]) for x in arms}            # [N, 3]
    chord = {x: np.linalg.norm(P[:, ai[x], -1, :2], axis=-1) for x in arms}
    latmax = {x: np.abs(P[:, ai[x], :, 1]).max(-1) for x in arms}
    sp_all = Z["speed_all"].astype(np.float64)                 # [N, A, 64, 3]
    out = {"n_tokens": N, "n_logs": int(len(ulogs)), "n_cruise": int(cruise.sum()), "n_cruise_logs": int(len(np.unique(li[cruise]))), "n_cruise_alt_state_gt": int(cruise_alt_state.sum()),
           "boot": {"B": a.boot, "seed": SEED, "estimator": "paired episode(log)-cluster bootstrap, percentile 95% CI, one shared resample matrix"},
           "controls": {k: meta[k] for k in ("C1_determinism", "C2_identity_vs_table", "C3_reachability", "C4_memo", "C5_ego_slots_real", "controls_pass")}}
    sets = {"ALL": allm, "CRUISE": cruise}
    lines = []
    L = lines.append
    L(f"# READOUT -- REFe ego probe (T0, open loop; u400 sha256 {meta['ckpt_sha256'][:16]})")
    L(f"n tokens {N} over {len(ulogs)} logs; CRUISE {int(cruise.sum())} tokens over {out['n_cruise_logs']} logs (CRUISE on state-speed GT instead: {int(cruise_alt_state.sum())}). Intervals: paired log-cluster bootstrap, B={a.boot}, 95% percentile.")
    L(f"controls_pass = {meta['controls_pass']} | C1 {json.dumps(meta['C1_determinism'])} | C2 {json.dumps(meta['C2_identity_vs_table'])}")
    c3s = json.dumps({k: "%d/%d" % (v["n_ok"], v["n"]) for k, v in meta["C3_reachability"].items()})
    L(f"C3 {c3s} | C4 {json.dumps(meta['C4_memo'])} | C5 {json.dumps(meta['C5_ego_slots_real'])}")
    L("")
    # ---- table 1: arm vs A0 -------------------------------------------------------------------------------------------------------------------------------
    t1 = {}
    for sname, m in sets.items():
        L(f"## Arm vs A0 -- {sname} (n={int(m.sum())})  [LONGITUDINAL: dv@1/2/4, dchord | LATERAL: dmaxlat | joint: ADE between picks]")
        L("| arm | ADE vs A0 (m) | dv@1s (m/s) | dv@2s | dv@4s | dchord4s (m) | dmaxlat (m) | pick-idx changed |")
        L("|---|---|---|---|---|---|---|---|")
        for x in tab_arms:
            if x == "A0":
                continue
            ade = np.linalg.norm(P[:, ai[x], :, :2] - P[:, ai["A0"], :, :2], axis=-1).mean(-1)
            dv = vp[x] - vp["A0"]
            row = {"ade_vs_A0": boot.mean(ade, m), "dv1": boot.mean(dv[:, 0], m), "dv2": boot.mean(dv[:, 1], m), "dv4": boot.mean(dv[:, 2], m),
                   "dchord4": boot.mean(chord[x] - chord["A0"], m), "dmaxlat": boot.mean(latmax[x] - latmax["A0"], m),
                   "pick_changed": boot.mean((Z["pick"][:, ai[x]] != Z["pick"][:, ai["A0"]]).astype(float), m)}
            t1.setdefault(sname, {})[x] = row
            L(f"| {x} | {fmt_u(row['ade_vs_A0'])} | {fmt(row['dv1'])} | {fmt(row['dv2'])} | {fmt(row['dv4'])} | {fmt(row['dchord4'])} | {fmt(row['dmaxlat'])} | {fmt_u(row['pick_changed'], 3)} |")
        L("")
    out["arm_vs_A0"] = t1
    # ---- table 2: vs GT -----------------------------------------------------------------------------------------------------------------------------------
    t2 = {}
    L("## Planned v@4s vs logged GT v@4s (same one-step chord estimator, GT with true timestamps)  [LONGITUDINAL]")
    L("| arm | b = plan v4 - GT v4, ALL (m/s) | b, CRUISE (m/s) | b - b_A0, CRUISE (paired) | share CRUISE with >15% slow-down | mean plan v4 / v0, CRUISE |")
    L("|---|---|---|---|---|---|")
    slow_gt = ((v0 - vgt[:, 2]) / np.maximum(v0, 1e-6) > SLOW_FRAC).astype(float)
    for x in tab_arms:
        b = vp[x][:, 2] - vgt[:, 2]
        bd = vp[x][:, 2] - vp["A0"][:, 2]
        slow = ((v0 - vp[x][:, 2]) / np.maximum(v0, 1e-6) > SLOW_FRAC).astype(float)
        row = {"b_all": boot.mean(b, allm), "b_cruise": boot.mean(b, cruise), "b_minus_A0_cruise": boot.mean(bd, cruise), "slow_share_cruise": boot.mean(slow, cruise),
               "ratio_v4_over_v0_cruise": boot.mean(vp[x][:, 2] / np.maximum(v0, 1e-6), cruise)}
        t2[x] = row
        L(f"| {x} | {fmt(row['b_all'])} | {fmt(row['b_cruise'])} | {fmt(row['b_minus_A0_cruise']) if x != 'A0' else '-'} | {fmt_u(row['slow_share_cruise'], 3)} | {fmt_u(row['ratio_v4_over_v0_cruise'], 3)} |")
    gt_row = {"slow_share_cruise_GT": boot.mean(slow_gt, cruise), "ratio_gt_v4_over_v0_cruise": boot.mean(vgt[:, 2] / np.maximum(v0, 1e-6), cruise)}
    t2["GT"] = gt_row
    L(f"| GT (reference) | - | - | - | {fmt_u(gt_row['slow_share_cruise_GT'], 3)} | {fmt_u(gt_row['ratio_gt_v4_over_v0_cruise'], 3)} |")
    L("")
    out["vs_gt"] = t2
    # ---- table 3: selection vs generation (descriptive) --------------------------------------------------------------------------------------------------------
    L("## Selection vs generation (DESCRIPTIVE): delta of the MEAN over all 64 proposals' v@T vs A0, CRUISE")
    L("| arm | dmean-proposal v@1s | v@2s | v@4s | (compare: pick dv@4s) |")
    L("|---|---|---|---|---|")
    t3 = {}
    for x in tab_arms:
        if x == "A0":
            continue
        d = sp_all[:, ai[x]].mean(1) - sp_all[:, ai["A0"]].mean(1)       # [N, 3]
        t3[x] = {f"dmeanprop_v{int(T)}": boot.mean(d[:, j], cruise) for j, T in enumerate(TS)}
        L(f"| {x} | {fmt(t3[x]['dmeanprop_v1'])} | {fmt(t3[x]['dmeanprop_v2'])} | {fmt(t3[x]['dmeanprop_v4'])} | {fmt(t1['CRUISE'][x]['dv4'])} |")
    L("")
    out["selection_vs_generation"] = t3
    # ---- ax gain + strata (descriptive) ------------------------------------------------------------------------------------------------------------------------
    gain = vp["AXP"][:, 2] - vp["AXM"][:, 2]
    out["ax_gain_v4_per_mps2"] = {s: boot.mean(gain, m) for s, m in sets.items()}
    L("## ax sensitivity: (plan v@4 under ax=+0.5) - (plan v@4 under ax=-0.5), per 1.0 m/s^2 of ax: " + "; ".join(f"{s} {fmt(out['ax_gain_v4_per_mps2'][s])}" for s in sets))
    L("")
    out["strata_cruise_b"] = {}
    L("## DESCRIPTIVE: b (plan v4 - GT v4) on CRUISE by stratum (zero = banked u400 PDMS exactly 0; random = non-zeroed)")
    for sname in sorted(set(strata)):
        mm = cruise & (strata == sname)
        out["strata_cruise_b"][sname] = {x: boot.mean(vp[x][:, 2] - vgt[:, 2], mm) for x in ("A0", "AX0", "GREY")}
        L(f"* {sname} (n={int(mm.sum())}): " + "; ".join(f"{x} {fmt(out['strata_cruise_b'][sname][x])}" for x in ("A0", "AX0", "GREY")))
    L("")
    # ---- families ------------------------------------------------------------------------------------------------------------------------------------------------
    L("## Families: LONGITUDINAL and LATERAL as tabled above. TACTICAL: n/a (REFe has no manoeuvre-decision/class output; only the pick-index change share is reported, descriptive). STRATEGIC: PAUSED (PI ruling 2026-09-19, ITEM 25) -- not a pass.")
    L("")
    # ---- committed decision text --------------------------------------------------------------------------------------------------------------------------------
    bA = t2["A0"]["b_cruise"]
    ok = lambda d: d["mean"] is not None
    R2 = ok(bA) and bA["mean"] <= -BIAS_BAR and bA["hi"] < 0
    R1_arms = [x for x in ("AX0", "GREY") if ok(t2[x]["b_minus_A0_cruise"]) and t2[x]["b_minus_A0_cruise"]["mean"] <= -BIAS_BAR and t2[x]["b_minus_A0_cruise"]["hi"] < 0]
    R1 = ok(bA) and abs(bA["mean"]) < BIAS_BAR and bool(R1_arms)
    R3 = ok(bA) and ok(t2["AX0"]["b_cruise"]) and abs(bA["mean"]) < BIAS_BAR and abs(t2["AX0"]["b_cruise"]["mean"]) < BIAS_BAR
    dd = t2["DEAD"]["b_minus_A0_cruise"]
    R4 = ok(dd) and dd["mean"] <= -BIAS_BAR and dd["hi"] < 0
    verdict = "R2" if R2 else ("R1" if R1 else ("R3" if R3 else "UNDECIDED"))
    texts = {"R2": "A0 itself plans a CRUISE slow-down: the policy has a slow bias even in distribution.",
             "R1": "A0 tracks GT on CRUISE while " + "/".join(R1_arms) + " make it slow down: the policy leans on ego/images and the sim's slow-down is partly policy.",
             "R3": "A0 and AX0 both track GT on CRUISE: the sim's slow-down comes from sim-side inputs.", "UNDECIDED": "No committed reading holds; see the numbers."}
    controls_ok = bool(meta["controls_pass"])
    out["decision"] = {"R2_policy_slow_bias": bool(R2), "R1_leans_on_ego_or_images": bool(R1), "R1_arms": R1_arms, "R3_sim_side": bool(R3), "R4_dead_slots_alone_ADDITIONAL": bool(R4),
                       "verdict_by_precedence": verdict, "verdict_text": texts[verdict], "controls_pass": controls_ok,
                       "inputs": {"b_A0_cruise": bA, "b_AX0_cruise": t2["AX0"]["b_cruise"], "AX0_minus_A0": t2["AX0"]["b_minus_A0_cruise"], "GREY_minus_A0": t2["GREY"]["b_minus_A0_cruise"],
                                  "DEAD_minus_A0": dd}}
    L("## COMMITTED DECISION (PREREG thresholds, mechanically evaluated)" + ("" if controls_ok else "  **CONTROLS DID NOT ALL PASS -- verdict below is provisional, read the controls first**"))
    L(f"* b_A0(CRUISE) = {fmt(bA)} m/s | b_AX0 = {fmt(t2['AX0']['b_cruise'])} | AX0-A0 = {fmt(t2['AX0']['b_minus_A0_cruise'])} | GREY-A0 = {fmt(t2['GREY']['b_minus_A0_cruise'])} | DEAD-A0 = {fmt(dd)}")
    L(f"* R2 (policy slow bias in distribution): {R2} | R1 (leans on ego/images; arms {R1_arms}): {R1} | R3 (sim-side inputs): {R3} | R4 ADDITIONAL (dead slots alone): {R4}")
    L(f"* **VERDICT: {verdict} -- {texts[verdict]}**")
    L("")
    L("Tier T0. Estimator: paired episode(log)-cluster bootstrap (answers 'another draw of logs' only; one checkpoint, deterministic planner => no training/inference-seed variance claimed beyond C1).")
    json.dump(out, open(os.path.join(a.out, "readout.json"), "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    open(os.path.join(a.out, "READOUT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
