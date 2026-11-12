"""ruleshield_read.py -- PREREG_V8F_PROOF_ADD3 (RULESHIELD).  CPU, zero training, no GPU.  Tier T1-kin (own-motion open loop, first tick of each eval-141 window): a diagnostic, never driving performance.

Inputs (read-only): the banked T1-kin tick dumps of stopbal_BAL-NULL_s0 / _s1 (tick 0 = the first-tick plan, identical to the dump's `ol`), their STOPDEC npz (label rows), eval-141.
Usage:  python ruleshield_read.py [--out RESULT_RULESHIELD.json] [--verdict VERDICT_RULESHIELD.md] [--smoke N] [--nboot 2000]
Reading choices the card does not fix (R1-R6, printed in the verdict; none moves a bar):
 R1 plan grid: the dumped plan is 30 knots at 0.2 s (not 60 at 0.1 s): the shield runs at dt = 0.2 s, q = a*dt = 0.6 m/s, a = 3.0 m/s2.
 R2 start state: v_0 = the plan's own first-segment speed (path origin -> knot 1 / 0.2 s); B(v_0) > d_0 = INFEASIBLE-AT-START. The measured v0 is read beside (infeasible_at_measured_v0).
 R3 d_0 (both variants): arc length ON THE PLAN'S OWN PATH (extended straight beyond the last knot) of the point nearest to the rest point (x, y) in the tick frame.
    ORACLE: the logged rest point = STOPDEC cons_gt stop_x_m / stop_y_m (finite exactly on the stop_hit windows); T-CUE: T's own t_cons_pred stop_x_m / stop_y_m (de-standardised) where p(STOP_POINT) >= 0.5.
 R4 STOP window set = stop_hit (goal_y STOP_POINT & goal_w, stopdec_read.py:100) = 131; OVERRUN = max(0, plan path length at 6 s - d_0); `overrun_tail` adds the braking distance of the step after the last segment, B(max(v_end - q, 0)). s2 is required for ORACLE; for T-CUE it is read where NULL overruns and shown as None when it has no such window.
 R5 verdict precedence: controls fail -> SHIELD-BROKEN; else SHIELD-COSTS if an ORACLE lateral/longitudinal cell (incl. ADE/FDE, which the card files under LONGITUDINAL) is worse with a CI excluding 0 at BOTH seeds; else HOLDS-AND-GAINS if the
    ORACLE tactical-band (2-6 s) ADE on stop windows has a CI excluding 0 on the low side at both seeds; else HOLDS-NO-GAIN. Cost cells at one seed only are printed. A separated CI is necessary, not sufficient (H-ESTIM-SEED-1: no replicate arm here).
 R6 T-CUE outcome is computed by the same rule and labelled diagnostic; its cue-miss residual (overrun against the LOGGED rest point) is reported apart.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = HERE + "/tree"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["STOPBAL_TREE"] = TREE
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, TREE + "/stack")
sys.path.insert(0, TREE + "/taniteval")
sys.path.insert(0, HERE)
THEORY = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
sys.path.insert(0, THEORY + "/stopbal/read")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import tanitad  # noqa: E402
assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(os.path.abspath(TREE).replace("\\", "/")), tanitad.__file__
import stopbal_geom as SG  # noqa: E402
import read_lib_stopbal as BL  # noqa: E402
from taniteval import ci as CI  # noqa: E402
import stop_shield as SH  # noqa: E402

T1 = "F:/TanitAD-scratch/stopbal_read/t1"
SDEC = THEORY + "/stopbal/read/out/stopdec"
SEEDS = (0, 1)
DT, A = 0.2, SH.A_BRAKE
Q = A * DT
STOP_GI = 6                  # GOAL22 index of STOP_POINT (v8f_mini.py:46 STOP_I)
CI_STOP_X, CI_STOP_Y = 16, 17   # CONS_NAMES indices of stop_x_m / stop_y_m (v8f_mini.py:49-52)
TOL = 1e-6
NB, SEED = 2000, 0
VARIANTS = ("ORACLE", "T-CUE")
BANDS3 = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30), "full_0_6s": (0, 30)}
LON_FAM = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "dk_hw_abs_err_m", "dk_tg_abs_err_s")
LAT_FAM = ("cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")


def load(seed):
    tag = f"BAL-NULL_s{seed}_m10_causal_heads"
    fs = sorted(glob.glob(f"{T1}/dump_{tag}/ep*.npz"))
    assert len(fs) == 141, (tag, len(fs))
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("g", "ol", "v0", "eid", "clip_index", "ws")}
    t = np.load(f"{T1}/ticks/ticks_{tag}.npz")
    m = t["tick"] == 0
    assert m.sum() == len(d["ws"]) == 2399
    assert np.array_equal(t["ci"][m], d["clip_index"]) and np.array_equal(t["t0"][m], d["ws"]), "tick-0 rows do not align with the dump windows"
    assert np.array_equal(t["plan"][m], d["ol"]), "control: tick-0 plan must equal the dump's open-loop plan bit for bit"
    assert (t["age_t"][m] == 0).all() and t["gt_ok"][m].all()
    z = np.load(f"{SDEC}/stopdec_BAL-NULL_s{seed}.npz")
    assert np.array_equal(z["clip_index"], d["clip_index"]) and np.array_equal(z["ws"], d["ws"]), "STOPDEC rows differ from the dump windows"
    sha = json.load(open(f"{T1}/done_{tag}.txt", encoding="utf-8"))["ckpt_sha256"]
    cg = z["cons_gt"]
    return {"plan": t["plan"][m].astype(np.float64), "g": d["g"], "v0": d["v0"], "eid": d["eid"], "ci": d["clip_index"], "ws": d["ws"],
            "stop": z["stop_hit"].astype(bool), "has": z["has"].astype(bool), "or_xy": np.column_stack([cg[:, CI_STOP_X], cg[:, CI_STOP_Y]]),
            "p_stop": t["t_goal_prob"][m][:, STOP_GI].astype(np.float64), "t_xy": t["t_cons_pred"][m][:, [CI_STOP_X, CI_STOP_Y]].astype(np.float64),
            "ckpt_sha256": sha, "tag": tag}


# ------------------------------------------------------------------------------------------------ the shield over windows
def cue_d0(plan, xy):
    """d_0 (R3) or None when the cue is unusable (non-finite xy)."""
    if not np.all(np.isfinite(xy)):
        return None
    return SH.rest_arc(plan, xy)[0]


def apply_variant(plan, d0s, fallback=True):
    """plan [N,30,2], d0s list (None = no cue).  -> shielded plans [N,30,2] (windows with no cue are the SAME array row, copied), per-window records."""
    out, rec = plan.copy(), []
    for i, d0 in enumerate(d0s):
        if d0 is None:
            rec.append(None)
            continue
        r = SH.shield_plan(plan[i], d0, a=A, dt=DT, fallback=fallback)
        out[i] = r["plan"]
        rec.append(r)
    return out, rec


def overrun_of(plan_i, d_ref, dt=DT):
    """(within-horizon overrun m, with braking tail) of a plan against the path distance d_ref."""
    v = SH.seg_speeds(plan_i, dt)
    s_end = float(v.sum() * dt)
    return max(0.0, s_end - d_ref), max(0.0, s_end + float(SH.brake_dist(max(v[-1] - Q, 0.0), Q, DT)) - d_ref)


def run_seed(seed, smoke=0):
    X = load(seed)
    N = len(X["ws"])
    sel = np.arange(N) if not smoke else np.linspace(0, N - 1, smoke).round().astype(int)
    for k in ("plan", "g", "v0", "eid", "ci", "ws", "stop", "has", "or_xy", "p_stop", "t_xy"):
        X[k] = X[k][sel]
    N = len(sel)
    plan = X["plan"]
    stop = X["stop"]
    d0_or = [cue_d0(plan[i], X["or_xy"][i]) if stop[i] else None for i in range(N)]
    flag = X["p_stop"] >= 0.5
    d0_t = [cue_d0(plan[i], X["t_xy"][i]) if (flag[i] and X["t_xy"][i, 0] > 0) else None for i in range(N)]
    cues = {"ORACLE": d0_or, "T-CUE": d0_t}
    # reference distance on the PLAN path to the LOGGED rest point, for every stop window (the overrun yardstick)
    d_log = np.array([d0_or[i] if d0_or[i] is not None else np.nan for i in range(N)])
    res = {"tag": X["tag"], "ckpt_sha256": X["ckpt_sha256"], "n_windows": int(N), "n_stop": int(stop.sum()), "n_nonstop": int((~stop).sum()), "n_has_label": int(X["has"].sum()),
           "n_tcue_flag": int(flag.sum()), "n_tcue_flag_and_stop": int((flag & stop).sum()), "n_tcue_active": int(sum(x is not None for x in d0_t))}
    se = np.hypot(X["t_xy"][stop, 0] - X["or_xy"][stop, 0], X["t_xy"][stop, 1] - X["or_xy"][stop, 1])
    res["tcue_stop_xy_err_m"] = {"median": float(np.median(se)), "mean": float(se.mean()), "n": int(stop.sum())}
    J = SG.join({"v0": X["v0"], "ws": X["ws"], "clip_index": X["ci"]}) if not smoke else None
    arrays = {}

    def metrics(P):
        if J is None:
            return None
        a = SG.arm_arrays(P, X["g"], J)
        a["full_0_6s"] = BL.band_rows(P, X["g"], 0, 30)
        return a
    arrays["NULL"] = metrics(plan)
    ctl, shielded, recs = {}, {}, {}
    for var in VARIANTS:
        P, rec = apply_variant(plan, cues[var])
        shielded[var], recs[var] = P, rec
        cued = [i for i in range(N) if rec[i] is not None]
        feas = [i for i in cued if rec[i]["feasible"]]
        infeas = [i for i in cued if not rec[i]["feasible"]]
        # s1: overrun against the cue's own d_0 (exactly 0 on every cued feasible window) -- within the horizon and with the braking tail
        o1 = np.array([overrun_of(P[i], cues[var][i]) for i in feas]) if feas else np.zeros((0, 2))
        # s3: the shielded points lie on the NULL plan's own (extended) path
        s3 = [float(np.max(SH.dist_to_path(*SH.path_of(plan[i])[:2], SH.path_of(plan[i])[3], P[i]))) for i in cued]
        # s4: windows with no cue are bit-identical to NULL
        nocue = [i for i in range(N) if rec[i] is None]
        s4 = bool(all(np.array_equal(P[i], plan[i]) for i in nocue))
        # s2: mutation M-C1a (no fallback) on the same cue windows
        Pm, recm = apply_variant(plan, cues[var], fallback=False)
        mut_over = np.array([overrun_of(Pm[i], cues[var][i])[0] if recm[i] is not None else 0.0 for i in range(N)])
        null_over = np.array([overrun_of(plan[i], cues[var][i])[0] if recm[i] is not None else 0.0 for i in range(N)])
        feas_mask = np.zeros(N, bool)
        feas_mask[feas] = True
        both = feas_mask & (null_over > TOL)
        ctl[var] = {"n_cued": len(cued), "n_feasible": len(feas), "n_infeasible_at_start": len(infeas),
                    "s1_max_overrun_m": float(o1[:, 0].max()) if len(o1) else 0.0, "s1_max_overrun_tail_m": float(o1[:, 1].max()) if len(o1) else 0.0,
                    "s1_pass": bool((o1 <= TOL).all()) if len(o1) else True,
                    "s2_null_overruns_n": int((null_over > TOL).sum()), "s2_mutation_overruns_n": int((mut_over[feas_mask] > TOL).sum()),
                    "s2_pass": bool((mut_over[both] > TOL).any()) if both.any() else None,
                    "s3_max_cross_to_null_path_m": float(max(s3)) if s3 else 0.0, "s3_pass": bool((max(s3) if s3 else 0.0) <= TOL),
                    "s4_noncue_bit_identical": s4, "s4_n_noncue": len(nocue),
                    "infeasible_at_measured_v0": int(sum(float(SH.brake_dist(X["v0"][i], Q, DT)) > cues[var][i] for i in cued)),
                    "n_path_extended": int(sum(rec[i]["s_shield"] > SH.path_of(plan[i])[1][-1] + 1e-9 for i in cued))}
        ctl[var]["controls_pass"] = bool(ctl[var]["s1_pass"] and (ctl[var]["s2_pass"] is True or (var != "ORACLE" and ctl[var]["s2_pass"] is None)) and ctl[var]["s3_pass"] and ctl[var]["s4_noncue_bit_identical"])
        arrays[var] = metrics(P)
    res["controls"] = ctl
    # overrun on the stop windows against the LOGGED rest point (the cue-miss residual for T-CUE)
    sidx = np.where(stop)[0]
    ov = {}
    for name, P in (("NULL", plan), ("ORACLE", shielded["ORACLE"]), ("T-CUE", shielded["T-CUE"])):
        o = np.array([overrun_of(P[i], d_log[i]) for i in sidx])
        ov[name] = {"share_overrun": float((o[:, 0] > TOL).mean()), "mean_m": float(o[:, 0].mean()), "mean_tail_m": float(o[:, 1].mean()), "share_tail": float((o[:, 1] > TOL).mean())}
    inf_o = [i for i in sidx if recs["ORACLE"][i] is not None and not recs["ORACLE"][i]["feasible"]]
    feas_o = [i for i in sidx if recs["ORACLE"][i] is not None and recs["ORACLE"][i]["feasible"]]
    ov["ORACLE_feasible_only"] = {"n": len(feas_o), "NULL_share": float(np.mean([overrun_of(plan[i], d_log[i])[0] > TOL for i in feas_o])) if feas_o else None,
                                  "shield_share": float(np.mean([overrun_of(shielded["ORACLE"][i], d_log[i])[0] > TOL for i in feas_o])) if feas_o else None}
    ov["ORACLE_infeasible_at_start"] = {"n": len(inf_o), "NULL_share_overrun": float(np.mean([overrun_of(plan[i], d_log[i])[0] > TOL for i in inf_o])) if inf_o else None,
                                        "braked_share_overrun": float(np.mean([overrun_of(shielded["ORACLE"][i], d_log[i])[0] > TOL for i in inf_o])) if inf_o else None}
    tc_flagged = [i for i in sidx if recs["T-CUE"][i] is not None]
    ov["T-CUE_stop_windows_cued"] = len(tc_flagged)
    ov["T-CUE_stop_windows_missed"] = int(len(sidx) - len(tc_flagged))
    res["overrun_vs_logged_rest_stop_windows"] = ov
    res["lon_label_changed_share"] = {v: (float(np.mean(arrays[v]["_lon"] != arrays["NULL"]["_lon"])) if J is not None else None) for v in VARIANTS}
    # ---- cells: variant minus NULL, paired episode-cluster bootstrap B 2,000, per band x metric x window set
    cells = {}
    if J is not None:
        sets = {"stop_131": stop, "all_2399": np.ones(N, bool)}
        for var in VARIANTS:
            for band in BANDS3:
                for met in sorted(arrays["NULL"][band]):
                    if met not in LON_FAM + LAT_FAM:
                        continue
                    for sn, mk in sets.items():
                        a, b = np.asarray(arrays[var][band][met], np.float64), np.asarray(arrays["NULL"][band][met], np.float64)
                        m = mk & np.isfinite(a) & np.isfinite(b)
                        if not m.any():
                            cells[f"{var}|{band}|{met}|{sn}"] = {"status": "NOT-APPLICABLE", "n_windows": 0}
                            continue
                        r = CI.paired_episode_cluster_bootstrap(a[m], b[m], X["eid"][m], n_boot=NB_G[0], seed=SEED)
                        cells[f"{var}|{band}|{met}|{sn}"] = {"status": "OK", "family": "LONGITUDINAL" if met in LON_FAM else "LATERAL", "null_mean": float(b[m].mean()), "arm_mean": float(a[m].mean()),
                                                              "delta": float(r["point"]) if "point" in r else float(a[m].mean() - b[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]),
                                                              "separated": bool(r["separated"]), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(X["eid"][m])))}
            # OVERRUN delta on the stop windows (metres), paired
            a = np.array([overrun_of(shielded[var][i], d_log[i])[0] for i in sidx])
            b = np.array([overrun_of(plan[i], d_log[i])[0] for i in sidx])
            r = CI.paired_episode_cluster_bootstrap(a, b, X["eid"][sidx], n_boot=NB_G[0], seed=SEED)
            cells[f"{var}|OVERRUN_vs_logged_rest_m|stop_131"] = {"status": "OK", "family": "LONGITUDINAL", "null_mean": float(b.mean()), "arm_mean": float(a.mean()), "delta": float(a.mean() - b.mean()),
                                                                 "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": len(sidx)}
    res["cells"] = cells
    return res


NB_G = [NB]


def outcome(per_seed, var="ORACLE"):
    ctl_ok = all(per_seed[s]["controls"][var]["controls_pass"] for s in per_seed)
    if not ctl_ok:
        return "SHIELD-BROKEN", {"reason": "a control failed", "controls": {s: per_seed[s]["controls"][var] for s in per_seed}}
    costs, gains = {}, {}
    for s, r in per_seed.items():
        for k, c in r["cells"].items():
            if not k.startswith(var + "|") or c.get("status") != "OK" or "OVERRUN" in k:
                continue
            if c["separated"] and c["delta"] > 0:
                costs.setdefault(k, []).append(s)
        g = r["cells"].get(f"{var}|tactical_2_6s|ade_m|stop_131")
        gains[s] = bool(g and g.get("separated") and g["delta"] < 0)
    both = {k: v for k, v in costs.items() if len(v) == len(per_seed)}
    info = {"cost_cells_both_seeds": sorted(both), "cost_cells_one_seed": {k: v for k, v in costs.items() if len(v) < len(per_seed)}, "ade_gain_by_seed": gains}
    if both:
        return "SHIELD-COSTS", info
    if all(gains.values()):
        return "SHIELD-HOLDS-AND-GAINS", info
    return "SHIELD-HOLDS-NO-GAIN", info


def verdict_md(R):
    ps = {int(s): R["seeds"][s] for s in R["seeds"]}
    L = ["# VERDICT_RULESHIELD (PREREG_V8F_PROOF_ADD3, frozen 2026-11-13)", "",
         f"Evidence class: MEASURED (ours; zero training; first-tick T1-kin plans of banked `stopbal_BAL-NULL_s0/_s1`, dumps under `{T1}`, STOPDEC rows under `{SDEC}`). Tier **T1-kin**, diagnostic, never driving performance.",
         "Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, shield minus NULL on the same windows (answers only 'another draw of episodes'; no replicate arm, so H-ESTIM-SEED-1 applies: a separated CI is necessary, not sufficient).",
         "STRATEGIC family: **PAUSED** (PI item 25), not a pass, not omitted. TACTICAL: T's goal/lon/lat outputs are unchanged by construction (the shield acts below T); plan-derived lon label changed share is printed.", "",
         f"## Outcome (ORACLE, the card's label): **{R['outcome']['ORACLE']['label']}**", "", f"```\n{json.dumps(R['outcome']['ORACLE']['info'], indent=1)[:1800]}\n```", "",
         f"T-CUE (diagnostic, R6): **{R['outcome']['T-CUE']['label']}**; T-CUE availability: **{R['tcue_availability']}**", "",
         "Reference beside (not a gate): STOPPERC S0-C forced the logged stop in the tactical code at coverage 0.7 and read -0.957 / -0.855 m tactical ADE.", "",
         "## Windows and controls (per seed)", "", "| seed | n | stop | non-stop | T-CUE flagged | flagged and stop | T-CUE active |", "|---|---|---|---|---|---|---|"]
    for s, r in ps.items():
        L.append(f"| {s} | {r['n_windows']} | {r['n_stop']} | {r['n_nonstop']} | {r['n_tcue_flag']} | {r['n_tcue_flag_and_stop']} | {r['n_tcue_active']} |")
    L += ["", "| seed | variant | cued | feasible | INFEASIBLE-AT-START | s1 max overrun m (tail) | s1 | s2 NULL overruns / mutation overruns | s2 | s3 max cross m | s3 | s4 | infeasible at measured v0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in ps.items():
        for v in VARIANTS:
            c = r["controls"][v]
            L.append(f"| {s} | {v} | {c['n_cued']} | {c['n_feasible']} | {c['n_infeasible_at_start']} | {c['s1_max_overrun_m']:.2e} ({c['s1_max_overrun_tail_m']:.2e}) | {c['s1_pass']} | {c['s2_null_overruns_n']} / {c['s2_mutation_overruns_n']} | {c['s2_pass']} | {c['s3_max_cross_to_null_path_m']:.2e} | {c['s3_pass']} | {c['s4_noncue_bit_identical']} | {c['infeasible_at_measured_v0']} |")
    L += ["", "## OVERRUN on the 131 stop windows against the LOGGED rest point (path distance along the plan's own path)", "", "| seed | arm | share overrun | mean m | share incl. braking tail | mean tail m |", "|---|---|---|---|---|---|"]
    for s, r in ps.items():
        for n in ("NULL", "ORACLE", "T-CUE"):
            o = r["overrun_vs_logged_rest_stop_windows"][n]
            L.append(f"| {s} | {n} | {o['share_overrun']:.3f} | {o['mean_m']:.3f} | {o['share_tail']:.3f} | {o['mean_tail_m']:.3f} |")
    for s, r in ps.items():
        o = r["overrun_vs_logged_rest_stop_windows"]
        L.append(f"\nseed {s}: ORACLE feasible-only {json.dumps(o['ORACLE_feasible_only'])}; INFEASIBLE-AT-START (apart) {json.dumps(o['ORACLE_infeasible_at_start'])}; T-CUE cued stop windows {o['T-CUE_stop_windows_cued']}, missed {o['T-CUE_stop_windows_missed']}; plan lon label changed share {json.dumps(r['lon_label_changed_share'])}")
    L += ["", "## Cells (variant minus NULL; delta [lo, hi]; * = CI excludes 0)", ""]
    for var in VARIANTS:
        L += [f"### {var}", "", "| cell | " + " | ".join(f"seed {s}" for s in ps) + " |", "|---|" + "---|" * len(ps)]
        keys = sorted({k for r in ps.values() for k in r["cells"] if k.startswith(var + "|")})
        for k in keys:
            row = []
            for s, r in ps.items():
                c = r["cells"].get(k)
                row.append("-" if not c or c.get("status") != "OK" else f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{'*' if c['separated'] else ''} (n {c['n_windows']})")
            L.append(f"| {k[len(var) + 1:]} | " + " | ".join(row) + " |")
        L.append("")
    L += ["## Reading choices", "", "```", __doc__.split("Reading choices the card does not fix")[1].strip(), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE + "/RESULT_RULESHIELD.json")
    ap.add_argument("--verdict", default=HERE + "/VERDICT_RULESHIELD.md")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=NB)
    a = ap.parse_args()
    NB_G[0] = a.nboot
    t0 = time.time()
    per = {}
    for s in SEEDS:
        per[s] = run_seed(s, a.smoke)
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
    R = {"card": "PREREG_V8F_PROOF_ADD3.md", "tier": "T1-kin", "evidence_class": "MEASURED (ours, zero training)", "a_brake": A, "dt": DT, "n_boot": a.nboot, "boot_seed": SEED,
         "tanitad_file": os.path.abspath(tanitad.__file__), "smoke": a.smoke, "seeds": {str(s): per[s] for s in SEEDS}, "strategic": "PAUSED (PI item 25)"}
    R["outcome"] = {}
    if a.smoke:
        R["outcome"] = {v: {"label": "SMOKE (no cells)", "info": {}} for v in VARIANTS}
    else:
        for v in VARIANTS:
            lab, info = outcome(per, v)
            R["outcome"][v] = {"label": lab, "info": info}
    n_flag = sum(per[s]["n_tcue_flag"] for s in SEEDS)
    R["tcue_availability"] = ("AVAILABLE (weak): T emits stop_x_m / stop_y_m (v8f_mini.py:49-52 CONS_NAMES, cons slots 16/17), finite on every window; median stop-xy error on the stop windows "
                              + " / ".join(f"{per[s]['tcue_stop_xy_err_m']['median']:.1f} m (seed {s})" for s in SEEDS) + "; flag p>=0.5 on " + " / ".join(f"{per[s]['n_tcue_flag']} (seed {s}, {per[s]['n_tcue_flag_and_stop']} stop)" for s in SEEDS)
                              + " windows. Work item: a better stop-distance output + its conformal envelope (theorem T3 residual risk = cue miss rate).")
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    open(a.verdict, "w", encoding="utf-8").write(verdict_md(R))
    print("WROTE", a.out, a.verdict, {v: R["outcome"][v]["label"] for v in VARIANTS}, f"{time.time() - t0:.0f}s")
    bad = [(s, v) for s in SEEDS for v in VARIANTS if not per[s]["controls"][v]["controls_pass"]]
    if bad:
        print("CONTROL PROBLEMS:", bad)
        sys.exit(2)


if __name__ == "__main__":
    main()
