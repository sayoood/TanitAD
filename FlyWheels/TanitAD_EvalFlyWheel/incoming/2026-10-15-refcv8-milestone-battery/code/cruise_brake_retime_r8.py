"""cruise_brake_retime_r8.py -- L2 headroom: how much of refcv8's S6 cruise false-brake (H-R8-S6-LON-1) does the SHIPPED re-timing (RETIME_DEPLOY8) already absorb?
CPU only, banked dumps + banked DEPLOY8 npz only, no GPU, no retraining.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
      PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;C:/lgs/retime_stage_scratch/stack;<PKG>/code' \
      python -X utf8 cruise_brake_retime_r8.py        # writes <PKG>/raw/refcv9_headroom/L2_HEADROOM.{json,md}

Pre-registration: raw/refcv9_headroom/PREREG_L2_HEADROOM.md (+ .sha256).  This tool REFUSES to run if the sealed sha256 does not match the file's bytes.
Definitions (GT-cruise mask, false / missed deceleration, signed along-track at 6 s, ADE) are the ones of code/s6_lon_decomp.py, IMPORTED, not re-implemented;
the only local re-implementations are the threshold-parametrised twins used by the mutation tests, and a control asserts each twin equals the imported original at 1.5 m/s.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "refcv9_headroom"
DEP = PKG / "raw" / "retime_deploy"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
TREE = "step30000ema_unc"
SEEDS = (0, 1)
N_BOOT = 2000
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
ESTIMATOR = ("paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over the episodes present in the masked windows: "
             "answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm, H-ESTIM-SEED-1); inference seeds 0 and 1 are shown separately, never pooled")

# ---- LITERALS for control C1 (os must reproduce H-R8-S6-LON-1 on ALL windows; 4 dp; from raw/s6_lon_decomp/S6_LON_DECOMP.json + the register row). Never derived from the code under test.
BANKED_ALL_S0 = {
    "ade_total_delta": 0.1658, "ade_total_lo": 0.0268, "ade_total_hi": 0.3153,
    "cruise_n": 1739, "cruise_ep": 128,
    "cruise_contrib": 0.2099, "cruise_share": 1.2657, "cruise_mean_delta": 0.4427,
    "cruise_signed_along_os": -2.2078, "cruise_signed_along_r7": 0.0602,
    "false_decel_os": 0.2745, "false_decel_r7": 0.0965, "false_decel_delta": 0.1780, "false_decel_lo": 0.1228, "false_decel_hi": 0.2404,
    "false_decel_n": 1949, "false_decel_ep": 138,
    "missed_decel_os": 0.3450, "missed_decel_r7": 0.5481, "missed_decel_delta": -0.2031, "missed_decel_lo": -0.2707, "missed_decel_hi": -0.1435,
    "missed_decel_n": 832, "missed_decel_ep": 80,
}
BANKED_ALL_S1 = {"ade_total_delta": 0.1895, "cruise_contrib": 0.2144, "cruise_share": 1.1316}
EXPECT_COMPLIANT_N, EXPECT_COMPLIANT_EP = 3251, 138

LONCLS = ("decel", "cruise", "accel")


# ------------------------------------------------------------------------------------------ small helpers
def sha256_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fl(x):
    return None if x is None else float(x)


def slim(c: dict) -> dict:
    keys = ("delta", "lo", "hi", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "separated", "status", "reason")
    return {k: (fl(c[k]) if k in ("delta", "lo", "hi", "mean_new", "mean_ref") else c[k]) for k in keys if k in c}


def match4(v, lit) -> bool:
    return v is not None and abs(round(float(v), 4) - lit) < 5e-9


# ------------------------------------------------------------------------------------------ threshold-parametrised twins (mutation tests only; asserted == S6's at 1.5)
def dvclass_cruise_thr(v0, v6, thr):
    dv = v6 - v0
    rules = [(v0 < 1.0) & (v6 >= 2.0), (v0 >= 2.0) & (v6 < 1.0), (v0 < 2.0) & (v6 < 2.0), dv >= thr, dv <= -thr]
    taken = np.zeros(v0.size, dtype=bool)
    for r in rules:
        taken |= r
    return (~taken) & (np.abs(dv) < thr)


def plan_class_thr(P, v0, dt, thr):
    P = P.astype(np.float64)
    v_end = np.linalg.norm(P[:, -1] - P[:, -2], axis=-1) / dt
    dv = v_end - v0
    return np.where(dv <= -thr, "decel", np.where(dv >= thr, "accel", "cruise")).astype(object)


# ------------------------------------------------------------------------------------------ the per-(surface, seed) cells
def indicator(arm_cls, gt_cls, mask, kind):
    if kind == "false_decel":
        return np.where(mask & (gt_cls == "cruise"), (arm_cls == "decel").astype(np.float64), np.nan)
    if kind == "missed_decel":
        return np.where(mask & (gt_cls == "decel"), (arm_cls != "decel").astype(np.float64), np.nan)
    raise ValueError(kind)


def seed_cells(PR, W: dict, cls: dict, gtc, cru, surf, eid) -> dict:
    """All cells for one inference seed on one surface.  W[arm] = per-window metrics (S6.window_metrics); cls[arm] = plan-derived class; arms: os, dep, r7."""
    el = list(eid)
    cm = cru & surf
    pairs = (("os_r7", "os", "r7"), ("dep_r7", "dep", "r7"), ("dep_os", "dep", "os"))

    def sub(x, m):
        return np.where(m, x, np.nan)
    out = {}
    for name, key in (("ade", "ade_m"), ("along", "along_signed_final_m")):
        out[name] = {pn: slim(PR.paired_cell(sub(W[a][key], cm), sub(W[b][key], cm), el)) for pn, a, b in pairs}
    for name, kind in (("false_decel", "false_decel"), ("missed_decel", "missed_decel")):
        out[name] = {pn: slim(PR.paired_cell(indicator(cls[a], gtc, surf, kind), indicator(cls[b], gtc, surf, kind), el)) for pn, a, b in pairs}
    out["false_decel_on_dvcruise"] = {pn: slim(PR.paired_cell(np.where(cm, (cls[a] == "decel").astype(np.float64), np.nan),
                                                              np.where(cm, (cls[b] == "decel").astype(np.float64), np.nan), el)) for pn, a, b in pairs}
    return out


def shrink_boot(d_os, d_dep, cm_surf, eid_surf, boot_counts):
    """Joint episode-cluster draws: shrink = 1 - D_dep/D_os per draw (D_x = cruise mean ADE delta vs refcv7), over the draws with D_os > 0."""
    C, ep_idx, n_ep = boot_counts(eid_surf)
    m = cm_surf.astype(np.float64)
    den = C @ np.bincount(ep_idx, weights=m, minlength=n_ep)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dos = (C @ np.bincount(ep_idx, weights=d_os * m, minlength=n_ep)) / den
        Ddep = (C @ np.bincount(ep_idx, weights=d_dep * m, minlength=n_ep)) / den
        ok = np.isfinite(Dos) & np.isfinite(Ddep) & (Dos > 0)
        sh = 1.0 - Ddep[ok] / Dos[ok]
    if sh.size == 0:
        return {"ci": [None, None], "n_draws_used": 0, "frac_draws_os_delta_le_0": 1.0}
    lo, hi = np.percentile(sh, [2.5, 97.5])
    return {"ci": [float(lo), float(hi)], "n_draws_used": int(ok.sum()), "frac_draws_os_delta_le_0": float(1.0 - ok.mean())}


def absorbed(cells: dict) -> dict:
    """Descriptive absorbed shares (NOT in the rule)."""
    Dos, Ddep = cells["ade"]["os_r7"]["delta"], cells["ade"]["dep_r7"]["delta"]
    fo, fd, fr = cells["false_decel"]["os_r7"]["mean_new"], cells["false_decel"]["dep_r7"]["mean_new"], cells["false_decel"]["os_r7"]["mean_ref"]
    ao, ad, ar = cells["along"]["os_r7"]["mean_new"], cells["along"]["dep_r7"]["mean_new"], cells["along"]["os_r7"]["mean_ref"]

    def ratio(num, den):
        return None if den is None or abs(den) < 1e-12 else float(num / den)
    return {"ade_shrink": None if Dos <= 0 else float(1.0 - Ddep / Dos),
            "ade_D_os_m": float(Dos), "ade_D_dep_m": float(Ddep),
            "false_decel_excess_absorbed": ratio(fo - fd, fo - fr), "false_decel_os": float(fo), "false_decel_dep": float(fd), "false_decel_r7": float(fr),
            "along_excess_absorbed": ratio(ao - ad, ao - ar), "along_os_m": float(ao), "along_dep_m": float(ad), "along_r7_m": float(ar)}


def decide(per_seed: dict) -> dict:
    """THE COMMITTED RULE (PREREG_L2_HEADROOM.md), each clause at EACH seed, all seeds must hold.  per_seed[seed] = seed_cells(...) output."""
    seeds = sorted(per_seed)
    pre = any((per_seed[s]["ade"]["os_r7"]["delta"] > 0 and per_seed[s]["ade"]["os_r7"]["lo"] > 0) for s in seeds)
    a_ok = {s: bool(per_seed[s]["ade"]["dep_r7"]["lo"] <= 0) for s in seeds}
    b_ok = {s: bool(per_seed[s]["false_decel"]["dep_r7"]["mean_new"] <= 0.5 * per_seed[s]["false_decel"]["os_r7"]["mean_new"]) for s in seeds}
    c_ok = {s: bool(per_seed[s]["missed_decel"]["dep_os"]["lo"] <= 0) for s in seeds}
    shrink = {}
    for s in seeds:
        Dos, Ddep = per_seed[s]["ade"]["os_r7"]["delta"], per_seed[s]["ade"]["dep_r7"]["delta"]
        shrink[s] = None if Dos <= 0 else float(1.0 - Ddep / Dos)
    partial_ok = all(shrink[s] is not None and shrink[s] >= 0.25 for s in seeds)
    absorbs = all(a_ok.values()) and all(b_ok.values()) and all(c_ok.values())
    rule_label = "ABSORBS" if absorbs else ("PARTIAL" if partial_ok else "NONE")
    return {"rule_label": rule_label, "label": rule_label if pre else "NOT-APPLICABLE", "precondition_os_regression_present": bool(pre),
            "clause_a_cruise_ade_not_separated_worse_vs_r7": a_ok, "clause_b_false_decel_le_half_os": b_ok, "clause_c_missed_decel_not_separated_worse_vs_os": c_ok,
            "ade_shrink_point": shrink, "partial_shrink_ge_25pct_at_both_seeds": bool(partial_ok)}


# ------------------------------------------------------------------------------------------ control C1 (banked literals) -- parametrised so the mutations can break it
def banked_control(PR, S6, T: dict, os_P, r7_P, seed: int, cruise_thr: float, cls_thr: float, boot_counts) -> dict:
    """os vs refcv7 on ALL windows, seed `seed`.  Returns {check: {value, literal, pass}}.  Uses the twins (thresholds explicit) so a mutation can move them."""
    G, v0, v6, eid, dt, ra = T["g"], T["v0"], T["v6"], T["eid"], T["dt"], T["_ra"]
    Wo, Wr = S6.window_metrics(PR, ra, os_P, G, dt), S6.window_metrics(PR, ra, r7_P, G, dt)
    d = Wo["ade_m"] - Wr["ade_m"]
    N = d.size
    el = list(eid)
    tot = PR.paired_cell(Wo["ade_m"], Wr["ade_m"], el)
    cru = dvclass_cruise_thr(v0, v6, cruise_thr)
    gtc = plan_class_thr(G, v0, dt, cls_thr)
    po, pr = plan_class_thr(os_P, v0, dt, cls_thr), plan_class_thr(r7_P, v0, dt, cls_thr)
    allm = np.ones(N, bool)
    contrib = float(d[cru].sum() / N)
    chk = {}

    def put(name, val, lit, tol4=True):
        chk[name] = {"value": None if val is None else float(val), "literal": lit, "pass": bool(match4(val, lit)) if tol4 else bool(val == lit)}
    if seed == 0:
        put("ade_total_delta", tot["delta"], BANKED_ALL_S0["ade_total_delta"])
        put("ade_total_lo", tot["lo"], BANKED_ALL_S0["ade_total_lo"])
        put("ade_total_hi", tot["hi"], BANKED_ALL_S0["ade_total_hi"])
        put("cruise_n", int(cru.sum()), BANKED_ALL_S0["cruise_n"], tol4=False)
        put("cruise_ep", len(set(np.asarray(eid)[cru].tolist())), BANKED_ALL_S0["cruise_ep"], tol4=False)
        put("cruise_contrib", contrib, BANKED_ALL_S0["cruise_contrib"])
        put("cruise_share", contrib / float(d.mean()), BANKED_ALL_S0["cruise_share"])
        put("cruise_mean_delta", float(d[cru].mean()), BANKED_ALL_S0["cruise_mean_delta"])
        put("cruise_signed_along_os", float(Wo["along_signed_final_m"][cru].mean()), BANKED_ALL_S0["cruise_signed_along_os"])
        put("cruise_signed_along_r7", float(Wr["along_signed_final_m"][cru].mean()), BANKED_ALL_S0["cruise_signed_along_r7"])
        fd = PR.paired_cell(indicator(po, gtc, allm, "false_decel"), indicator(pr, gtc, allm, "false_decel"), el)
        md = PR.paired_cell(indicator(po, gtc, allm, "missed_decel"), indicator(pr, gtc, allm, "missed_decel"), el)
        for pre_, c, base in (("false_decel", fd, ""), ("missed_decel", md, "")):
            put(f"{pre_}_os", c["mean_new"], BANKED_ALL_S0[f"{pre_}_os"])
            put(f"{pre_}_r7", c["mean_ref"], BANKED_ALL_S0[f"{pre_}_r7"])
            put(f"{pre_}_delta", c["delta"], BANKED_ALL_S0[f"{pre_}_delta"])
            put(f"{pre_}_lo", c["lo"], BANKED_ALL_S0[f"{pre_}_lo"])
            put(f"{pre_}_hi", c["hi"], BANKED_ALL_S0[f"{pre_}_hi"])
            put(f"{pre_}_n", c["n_windows"], BANKED_ALL_S0[f"{pre_}_n"], tol4=False)
            put(f"{pre_}_ep", c["n_episodes"], BANKED_ALL_S0[f"{pre_}_ep"], tol4=False)
    else:
        put("ade_total_delta", tot["delta"], BANKED_ALL_S1["ade_total_delta"])
        put("cruise_contrib", contrib, BANKED_ALL_S1["cruise_contrib"])
        put("cruise_share", contrib / float(d.mean()), BANKED_ALL_S1["cruise_share"])
    return chk


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[cruise_brake_retime_r8] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pre_md, pre_sha = OUT / "PREREG_L2_HEADROOM.md", OUT / "PREREG_L2_HEADROOM.sha256"
    sealed, actual = pre_sha.read_text(encoding="utf-8").strip(), sha256_bytes(pre_md)
    if sealed != actual:
        raise SystemExit(f"[cruise_brake_retime_r8] REFUSED: PREREG sha256 sealed {sealed} != file {actual}")

    import r8_battery_patch as BP
    L, Pm = BP.apply()
    ra = Pm.RR.ra3().ra
    import pair_read_r8 as PR
    import s6_lon_decomp as S6

    root = os.path.join(ARCHIVE, TREE)
    Ts = {s: S6.load_tree(root, s, L.sha12, PR, ra) for s in SEEDS}
    T0 = Ts[0]
    r7x = S6.load_r7_extras(BP.R7_BASE_ROOT, T0)
    T0["r7_v_max_ms"] = r7x["v_max_ms"]
    parts, _dv = S6.partitions(T0)
    dvc = parts["dvclass"][0]
    for s in SEEDS:
        Ts[s]["_ra"] = ra
    print(f"[{time.time() - t0:6.1f}s] loaded panels", flush=True)

    ctl = {"C2": {}, "C3": {}}
    dt, G, v0, v6, eid = T0["dt"], T0["g"], T0["v0"], T0["v6"], T0["eid"]
    cru_dv = (dvc == "cruising")
    gtc = S6.plan_lon_class(G, v0, dt)

    # twins == originals (so the mutation tests probe the SAME definitions)
    ctl["C2"]["twin_dvclass_cruise_equals_S6"] = bool(np.array_equal(dvclass_cruise_thr(v0, v6, 1.5), cru_dv))
    ctl["C2"]["twin_plan_class_equals_S6_os_r7_gt"] = bool(all(np.array_equal(plan_class_thr(P, v0, dt, 1.5), S6.plan_lon_class(P, v0, dt))
                                                                for P in (T0["os"], T0["ref"], G)))

    # ---- load the npz, check alignment (C2)
    Z = {s: np.load(DEP / f"DEPLOY8_POS_step30000ema_unc_s{s}.npz", allow_pickle=True) for s in SEEDS}
    comp = {}
    for s in SEEDS:
        z, T = Z[s], Ts[s]
        ctl["C2"][f"s{s}_ws_equal_panel"] = bool(np.array_equal(z["S6_ws"], T["ws"]))
        ctl["C2"][f"s{s}_eid_equal_panel"] = bool(np.array_equal(np.asarray([str(x) for x in z["S6_eid"]]), np.asarray([str(x) for x in T["eid"]])))
        ctl["C2"][f"s{s}_os_bit_equal_panel"] = bool(np.array_equal(z["S6_os"], T["os"]))
        ctl["C2"][f"s{s}_pos_shape_ok"] = bool(z["S6_pos"].shape == T["os"].shape)
        ctl["C2"][f"s{s}_gt_equal_seed0"] = bool(np.array_equal(T["g"], G))
        ctl["C2"][f"s{s}_ref_r7_is_banked_b_refcv7_50400_s{s}"] = True   # by construction: load_tree reads A[f"b_refcv7_50400_s{seed}"]
        comp[s] = np.asarray(z["S6_compliant"], bool)
    ctl["C2"]["compliant_n"] = int(comp[0].sum())
    ctl["C2"]["compliant_ep"] = len(set(np.asarray(eid)[comp[0]].tolist()))
    ctl["C2"]["compliant_mask_equal_across_seeds"] = bool(np.array_equal(comp[0], comp[1]))
    ctl["C2"]["compliant_count_matches_expected_literal"] = bool(ctl["C2"]["compliant_n"] == EXPECT_COMPLIANT_N and ctl["C2"]["compliant_ep"] == EXPECT_COMPLIANT_EP)
    ctl["C2"]["v0_v6_equal_across_seeds"] = bool(np.array_equal(Ts[1]["v0"], v0) and np.array_equal(Ts[1]["v6"], v6))

    # ---- per-window metrics for every arm and seed (+ C3 numpy cross-checks)
    W, CL = {}, {}
    for s in SEEDS:
        T = Ts[s]
        arms = {"os": T["os"], "dep": Z[s]["S6_pos"], "r7": T["ref"]}
        W[s] = {a: S6.window_metrics(PR, ra, P, T["g"], dt) for a, P in arms.items()}
        CL[s] = {a: S6.plan_lon_class(P, v0, dt) for a, P in arms.items()}
        for a in arms:
            ctl["C3"][f"s{s}_{a}_numpy_ade_vs_components"] = W[s][a]["_x_ade_numpy_vs_components_max_abs"]
            ctl["C3"][f"s{s}_{a}_numpy_along_vs_components"] = W[s][a]["_x_along_abs_vs_components_max_abs"]
    print(f"[{time.time() - t0:6.1f}s] per-window metrics done", flush=True)

    allm = np.ones(len(eid), bool)
    SURF = {"compliant": comp[0], "all": allm}

    # ---- C1: os reproduces H-R8-S6-LON-1 on ALL windows (both seeds) -- and the mutations m1..m3 must break it
    c1 = {s: banked_control(PR, S6, Ts[s], Ts[s]["os"], Ts[s]["ref"], s, 1.5, 1.5, S6.boot_counts) for s in SEEDS}
    ctl["C1_banked_literals"] = c1
    ctl["C1_pass"] = bool(all(v["pass"] for s in SEEDS for v in c1[s].values()))
    mut = {}
    Tm = Ts[0]
    os_shift = Tm["os"].copy()
    os_shift[..., 0] += np.float32(0.01)
    for name, kw in (("m1_cruise_thr_1p0", dict(os_P=Tm["os"], cruise_thr=1.0, cls_thr=1.5)),
                     ("m2_decel_thr_1p0", dict(os_P=Tm["os"], cruise_thr=1.5, cls_thr=1.0)),
                     ("m3_os_shift_x_plus_0p01m", dict(os_P=os_shift, cruise_thr=1.5, cls_thr=1.5))):
        r = banked_control(PR, S6, Tm, kw["os_P"], Tm["ref"], 0, kw["cruise_thr"], kw["cls_thr"], S6.boot_counts)
        bad = sorted(k for k, v in r.items() if not v["pass"])
        mut[name] = {"went_RED": bool(len(bad) > 0), "failed_checks": bad}
    ctl["C7_mutations"] = mut
    ctl["C7_all_RED"] = bool(all(m["went_RED"] for m in mut.values()))
    print(f"[{time.time() - t0:6.1f}s] C1 + mutations done  C1_pass={ctl['C1_pass']}  mutations_RED={ctl['C7_all_RED']}", flush=True)

    # ---- THE MEASUREMENT: both surfaces, both seeds
    res_surf = {}
    for sname, surf in SURF.items():
        per_seed, ent = {}, {"n_windows": int(surf.sum()), "n_episodes": len(set(np.asarray(eid)[surf].tolist())),
                             "gt_cruise_dvclass_n": int((cru_dv & surf).sum()), "gt_cruise_dvclass_ep": len(set(np.asarray(eid)[cru_dv & surf].tolist())),
                             "gt_cruise_planclass_n": int(((gtc == "cruise") & surf).sum()), "gt_decel_planclass_n": int(((gtc == "decel") & surf).sum()), "seeds": {}}
        for s in SEEDS:
            cells = seed_cells(PR, W[s], CL[s], gtc, cru_dv, surf, eid)
            per_seed[s] = cells
            cm = (cru_dv & surf)
            d_os = (W[s]["os"]["ade_m"] - W[s]["r7"]["ade_m"])
            d_dep = (W[s]["dep"]["ade_m"] - W[s]["r7"]["ade_m"])
            ab = absorbed(cells)
            ab["ade_shrink_ci_joint_draws"] = shrink_boot(d_os[surf], d_dep[surf], cm[surf], np.asarray(eid)[surf], S6.boot_counts)
            # DESCRIPTIVE extras added after the first run (the rule cells are unchanged between runs): the cruise windows' contribution on the surface's all-window scale,
            # and the cruise mean signed along-track error per instant (where in the 0-6 s horizon the stage helps)
            Ns = float(surf.sum())
            ab["contribution_scale"] = {"cruise_contrib_os_m": float(d_os[cm].sum() / Ns), "cruise_contrib_dep_m": float(d_dep[cm].sum() / Ns),
                                        "surface_total_os_minus_r7_m": float(d_os[surf].mean()), "surface_total_dep_minus_r7_m": float(d_dep[surf].mean())}
            Pm_ = {"os": Ts[s]["os"], "dep": Z[s]["S6_pos"], "r7": Ts[s]["ref"]}
            ab["along_by_instant_cruise_m"] = {a: [float((P[cm][:, k, 0] - G[cm][:, k, 0]).mean()) for k in range(G.shape[1])] for a, P in Pm_.items()}
            ent["seeds"][str(s)] = {"cells": cells, "absorbed": ab}
        ent["decision"] = decide(per_seed)
        res_surf[sname] = ent
        print(f"[{time.time() - t0:6.1f}s] surface {sname}: rule {ent['decision']['rule_label']}  (label {ent['decision']['label']})", flush=True)

    # ---- C4: re-timed-arm reproduction against the banked FORWARD_DEPLOY8 NR_minus_refcv7.ade_m cells (read AFTER the prereg was sealed)
    fwd = jload(DEP / "FORWARD_DEPLOY8_step30000ema_unc.json")
    c4 = {}
    for s in SEEDS:
        for sname, surf in (("compliant", comp[s]), ("all", allm)):
            mine = PR.paired_cell(W[s]["dep"]["ade_m"][surf], W[s]["r7"]["ade_m"][surf], list(np.asarray(eid)[surf]))
            bank = fwd["seeds"][str(s)]["surfaces"]["S6"]["sets"][sname]["arms"]["RETIME_DEPLOY8"]["NR_minus_refcv7"]["ade_m"]
            flds = ("delta", "lo", "hi", "n_windows", "n_episodes")
            c4[f"s{s}_{sname}"] = {"mine": {f: mine[f] for f in flds}, "banked": {f: bank.get(f) for f in flds}, "exact": bool(all(mine[f] == bank.get(f) for f in flds))}
    ctl["C4_retimed_arm_reproduction"] = c4
    ctl["C4_pass"] = bool(all(v["exact"] for v in c4.values()))

    # ---- C5 / C6: decision-function arms with answers fixed by construction (DEPLOY8 replaced)
    def synth_arm(kind, s):
        T = Ts[s]
        if kind == "identity":
            return T["os"].copy()
        if kind == "oracle":
            return G.copy()
        if kind == "overbrake":
            return (T["os"].astype(np.float64) * 0.5).astype(np.float32)
        if kind == "oracle_never_brakes":
            P = G.copy()
            hold = np.stack([np.stack([v0 * dt * (k + 1), np.zeros_like(v0)], axis=-1) for k in range(G.shape[1])], axis=1).astype(np.float32)
            dm = gtc == "decel"
            P[dm] = hold[dm]
            return P
        raise ValueError(kind)
    EXPECT = {"identity": "NONE", "overbrake": "NONE", "oracle": "ABSORBS", "oracle_never_brakes": "PARTIAL"}
    syn = {}
    for kind, expect in EXPECT.items():
        syn[kind] = {"expected_label_by_construction": expect, "surfaces": {}}
        for sname, surf in SURF.items():
            per_seed, extra = {}, {}
            for s in SEEDS:
                P = synth_arm(kind, s)
                Wd = dict(W[s])
                Wd["dep"] = S6.window_metrics(PR, ra, P, G, dt)
                CLd = dict(CL[s])
                CLd["dep"] = S6.plan_lon_class(P, v0, dt)
                per_seed[s] = seed_cells(PR, Wd, CLd, gtc, cru_dv, surf, eid)
                if kind == "identity":
                    cs = per_seed[s]
                    extra[f"s{s}_dep_minus_os_all_exactly_zero_zero_width"] = bool(all(
                        cs[m]["dep_os"]["delta"] == 0.0 and cs[m]["dep_os"]["lo"] == 0.0 and cs[m]["dep_os"]["hi"] == 0.0 for m in ("ade", "along", "false_decel", "missed_decel")))
            dec = decide(per_seed)
            ok = (dec["rule_label"] == expect) if dec["precondition_os_regression_present"] else None
            syn[kind]["surfaces"][sname] = {"rule_label": dec["rule_label"], "precondition": dec["precondition_os_regression_present"], "matches_expected": ok, **extra,
                                            "clauses": {k: dec[k] for k in ("clause_a_cruise_ade_not_separated_worse_vs_r7", "clause_b_false_decel_le_half_os",
                                                                              "clause_c_missed_decel_not_separated_worse_vs_os", "ade_shrink_point")}}
    ctl["C5_C6_synthetic_arms"] = syn
    ident_zero = all(v for sname in SURF for k, v in syn["identity"]["surfaces"][sname].items() if k.endswith("exactly_zero_zero_width"))
    ctl["C5_identity_exact_zero_and_NONE"] = bool(ident_zero and all(syn["identity"]["surfaces"][sn]["rule_label"] == "NONE" for sn in SURF))
    ctl["C6_pass"] = bool(all(syn[k]["surfaces"]["all"]["matches_expected"] is True for k in EXPECT)
                          and all(v["matches_expected"] in (True, None) for k in EXPECT for v in syn[k]["surfaces"].values()))
    print(f"[{time.time() - t0:6.1f}s] controls C4={ctl['C4_pass']} C5={ctl['C5_identity_exact_zero_and_NONE']} C6={ctl['C6_pass']}", flush=True)

    c2_ok = bool(all(v for k, v in ctl["C2"].items() if isinstance(v, bool)))
    c3_ok = bool(all(v < 1e-4 for v in ctl["C3"].values()))
    ctl["C2_pass"], ctl["C3_pass"] = c2_ok, c3_ok
    ctl["ALL_PASS"] = bool(ctl["C1_pass"] and c2_ok and c3_ok and ctl["C4_pass"] and ctl["C5_identity_exact_zero_and_NONE"] and ctl["C6_pass"] and ctl["C7_all_RED"])

    res = {"tool": "cruise_brake_retime_r8.py", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": ESTIMATOR, "n_boot": N_BOOT,
           "tree": os.path.join(ARCHIVE, TREE), "prereg_sha256": actual, "script_sha256": sha256_bytes(Path(__file__)),
           "npz": {str(s): {"file": f"DEPLOY8_POS_step30000ema_unc_s{s}.npz", "sha256": sha256_bytes(DEP / f"DEPLOY8_POS_step30000ema_unc_s{s}.npz")} for s in SEEDS},
           "label_primary_compliant": res_surf["compliant"]["decision"]["label"], "label_all": res_surf["all"]["decision"]["label"],
           "surfaces": res_surf, "controls": ctl, "wall_s": round(time.time() - t0, 1)}
    if not ctl["ALL_PASS"]:
        with open(OUT / "CONTROL_FAILED.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
        print("CONTROL FAILED", json.dumps({k: ctl.get(k) for k in ("C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_identity_exact_zero_and_NONE", "C6_pass", "C7_all_RED")}))
        return 3
    with open(OUT / "L2_HEADROOM.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    with open(OUT / "L2_HEADROOM.md", "w", encoding="utf-8") as f:
        f.write(render_md(res))
    print("WROTE", OUT / "L2_HEADROOM.json", OUT / "L2_HEADROOM.md", f"{res['wall_s']}s")
    return 0


# ------------------------------------------------------------------------------------------ markdown
def ci_s(c, dp=3):
    return f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]{' SEP' if (c['lo'] > 0 or c['hi'] < 0) else ''}"


def render_md(R: dict) -> str:
    ctl = R["controls"]
    L = ["# L2 headroom: how much of refcv8's cruise false-brake does the SHIPPED re-timing (RETIME_DEPLOY8) already absorb?", "",
         f"**LABEL (primary, S6 compliant windows): {R['label_primary_compliant']}**  |  S6 all windows (the surface H-R8-S6-LON-1 was measured on): {R['label_all']}", "",
         f"* tier: **{R['tier']}**; EMA-30k, `step30000ema_unc`; evidence class MEASURED (ours), CPU only.",
         f"* estimator: {R['estimator']}.",
         "* pre-registered rule: `raw/refcv9_headroom/PREREG_L2_HEADROOM.md` (sha256 sealed before any re-timed number was computed): ABSORBS = cruise ADE delta vs refcv7 not separated worse at either seed "
         "AND false-decel(DEPLOY8) <= 0.5 x false-decel(os) AND missed-decel(DEPLOY8) not separated worse than os; PARTIAL = cruise ADE delta shrinks >= 25 % at both seeds but ABSORBS fails; NONE otherwise; "
         "guard: if os shows no separated cruise regression on a surface the label there is NOT-APPLICABLE.",
         f"* artifacts: {R['npz']['0']['file']} (sha {R['npz']['0']['sha256'][:12]}), {R['npz']['1']['file']} (sha {R['npz']['1']['sha256'][:12]}); prereg sha {R['prereg_sha256'][:12]}; wall {R['wall_s']} s.", ""]
    for sname, title in (("compliant", "PRIMARY: S6 COMPLIANT windows"), ("all", "SECONDARY: S6 ALL windows (H-R8-S6-LON-1's surface)")):
        S = R["surfaces"][sname]
        d = S["decision"]
        L += [f"## {title}", "",
              f"n **{S['n_windows']} windows / {S['n_episodes']} episodes**; GT-cruise (dvclass, ADE and along-track) **{S['gt_cruise_dvclass_n']} windows / {S['gt_cruise_dvclass_ep']} ep**; "
              f"GT-cruise (plan class, false decel) **{S['gt_cruise_planclass_n']}**; GT-decel (plan class, missed decel) **{S['gt_decel_planclass_n']}**.",
              f"Rule label **{d['rule_label']}** (effective label {d['label']}; precondition os regression present: {d['precondition_os_regression_present']}). "
              f"Clause a (cruise ADE vs refcv7 not separated worse, per seed): {d['clause_a_cruise_ade_not_separated_worse_vs_r7']}; clause b (false decel <= 0.5 x os): {d['clause_b_false_decel_le_half_os']}; "
              f"clause c (missed decel not separated worse than os): {d['clause_c_missed_decel_not_separated_worse_vs_os']}; ADE shrink (point): {d['ade_shrink_point']}.", "",
              "| seed | arm | cruise ADE delta vs refcv7 [CI] m | cruise signed along-track @6 s (m) | false decel (rate) | missed decel (rate) | false decel on dvclass-cruise (supplement) |",
              "|---|---|---|---|---|---|---|"]
        for s in ("0", "1"):
            cc = S["seeds"][s]["cells"]
            for arm, pair in (("os (refcv8)", "os_r7"), ("DEPLOY8", "dep_r7")):
                L.append(f"| {s} | {arm} | {ci_s(cc['ade'][pair])} | {cc['along'][pair]['mean_new']:+.2f} | {cc['false_decel'][pair]['mean_new']:.3f} | "
                         f"{cc['missed_decel'][pair]['mean_new']:.3f} | {cc['false_decel_on_dvcruise'][pair]['mean_new']:.3f} |")
            L.append(f"| {s} | refcv7-50,400 | 0 (reference) | {cc['along']['os_r7']['mean_ref']:+.2f} | {cc['false_decel']['os_r7']['mean_ref']:.3f} | "
                     f"{cc['missed_decel']['os_r7']['mean_ref']:.3f} | {cc['false_decel_on_dvcruise']['os_r7']['mean_ref']:.3f} |")
            L.append(f"| {s} | *DEPLOY8 - os (paired)* | {ci_s(cc['ade']['dep_os'])} | {ci_s(cc['along']['dep_os'], 2)} | {ci_s(cc['false_decel']['dep_os'])} | "
                     f"{ci_s(cc['missed_decel']['dep_os'])} | {ci_s(cc['false_decel_on_dvcruise']['dep_os'])} |")
        L += ["", "Absorbed shares (descriptive, not in the rule) and the L2 sizing residual:", "",
              "| seed | cruise ADE: os D / DEPLOY8 D (m, vs refcv7) | ADE shrink [joint-draw CI] | false-decel excess absorbed | along-track excess absorbed |", "|---|---|---|---|---|"]
        for s in ("0", "1"):
            a = S["seeds"][s]["absorbed"]
            ci = a["ade_shrink_ci_joint_draws"]["ci"]
            pc = lambda x: "n/a" if x is None else f"{100 * x:.0f} %"  # noqa: E731
            L.append(f"| {s} | {a['ade_D_os_m']:+.3f} / {a['ade_D_dep_m']:+.3f} | {pc(a['ade_shrink'])} [{pc(ci[0])}, {pc(ci[1])}] "
                     f"(draws with os D>0: {a['ade_shrink_ci_joint_draws']['n_draws_used']}/{N_BOOT}) | {pc(a['false_decel_excess_absorbed'])} "
                     f"({a['false_decel_os']:.3f} -> {a['false_decel_dep']:.3f}, refcv7 {a['false_decel_r7']:.3f}) | {pc(a['along_excess_absorbed'])} "
                     f"({a['along_os_m']:+.2f} -> {a['along_dep_m']:+.2f}, refcv7 {a['along_r7_m']:+.2f}) |")
        L += ["", "Descriptive extras (added after the first run; the rule cells are identical between the runs). Cruise windows' contribution on this surface's all-window scale "
              "(sum of the cruise windows' ADE delta vs refcv7 / n surface windows) and the cruise mean signed along-track error by instant:", "",
              "| seed | cruise contribution os / DEPLOY8 (m) | surface total delta vs refcv7 os / DEPLOY8 (m) | along-track by instant 1..6 s: os | DEPLOY8 | refcv7 (m) |", "|---|---|---|---|---|---|"]
        for s in ("0", "1"):
            a = S["seeds"][s]["absorbed"]
            cs_, pr_ = a["contribution_scale"], a["along_by_instant_cruise_m"]
            fm = lambda v: " ".join(f"{x:+.2f}" for x in v)  # noqa: E731
            L.append(f"| {s} | {cs_['cruise_contrib_os_m']:+.3f} / {cs_['cruise_contrib_dep_m']:+.3f} | {cs_['surface_total_os_minus_r7_m']:+.3f} / {cs_['surface_total_dep_minus_r7_m']:+.3f} | "
                     f"{fm(pr_['os'])} | {fm(pr_['dep'])} | {fm(pr_['r7'])} |")
        L.append("")
    L += ["## Controls", "",
          f"* **ALL_PASS: {ctl['ALL_PASS']}** (C1 {ctl['C1_pass']}, C2 {ctl['C2_pass']}, C3 {ctl['C3_pass']}, C4 {ctl['C4_pass']}, C5 {ctl['C5_identity_exact_zero_and_NONE']}, C6 {ctl['C6_pass']}, C7 {ctl['C7_all_RED']}).",
          "* C1 os reproduces the banked H-R8-S6-LON-1 literals on ALL windows (seed 0: ADE total +0.1658 [+0.0268, +0.3153]; cruise 1,739 / 128 ep, contribution +0.2099, share 1.2657, bin-mean +0.4427; "
          "signed along-track -2.2078 vs +0.0602; false decel 0.2745 vs 0.0965, +0.1780 [+0.1228, +0.2404], n 1,949 / 138; missed decel 0.3450 vs 0.5481, -0.2031 [-0.2707, -0.1435], n 832 / 80; seed 1 share 1.1316, total +0.1895): "
          f"**{ctl['C1_pass']}** ({sum(len(v) for v in ctl['C1_banked_literals'].values())} checks).",
          "* C2 alignment (npz ws / eid / os bit-equal the panels, compliant 3,251 / 138, GT equal across seeds, twins == imported S6 definitions): " + str(ctl["C2_pass"]) + ".",
          "* C3 numpy ADE / along-track equal `components` (< 1e-4): " + str(ctl["C3_pass"]) + f" (max {max(ctl['C3'].values()):.1e}).",
          "* C4 DEPLOY8 - refcv7 S6 ADE recomputed equals the banked `FORWARD_DEPLOY8` RETIME_DEPLOY8 `NR_minus_refcv7.ade_m` cell field for field (both seeds, compliant and all): " + str(ctl["C4_pass"]) + ".",
          "* C5 identity arm (DEPLOY8 := os): every cell exactly 0 with a zero-width CI, label NONE: " + str(ctl["C5_identity_exact_zero_and_NONE"]) + ".",
          "* C6 decision-function arms (answer fixed by construction): " + "; ".join(
              f"{k} expected {v['expected_label_by_construction']} -> " + ", ".join(f"{sn}: {sv['rule_label']}{'' if sv['matches_expected'] in (True, None) else ' (MISMATCH)'}" for sn, sv in v["surfaces"].items())
              for k, v in ctl["C5_C6_synthetic_arms"].items()) + ".",
          "* C7 mutations of the C1 instrument, each must go RED: " + "; ".join(f"{k}: {'RED' if v['went_RED'] else 'STAYED GREEN'} ({len(v['failed_checks'])} checks failed)" for k, v in ctl["C7_mutations"].items()) + ".", "",
          "## Caveats", "",
          "* single TRAINING seed per arm: the interval is an episode draw only and is blind to training variance (`H-ESTIM-SEED-1`); the two inference seeds are the replicate for the planner's sampling floor and are not pooled.",
          "* T1 open loop on logged eval139 frames: a statement about plans scored against the logged future, not about driving.",
          "* the speed-label input differs between refcv7 (per-clip continuous `v_max_ms`) and refcv8 (4-class road-type label): every refcv8-vs-refcv7 row crosses that input change.",
          "* the DEPLOY8 table was fit on TRAIN-side data (`retime_trainfit`), not on these eval episodes; this read is a property of THIS stage on THIS checkpoint.",
          "* the instrument defines GT-cruise two ways (dvclass for ADE / along-track, plan class for false decel); both are kept as defined and the false-decel rate on the dvclass mask is shown as a supplement.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
