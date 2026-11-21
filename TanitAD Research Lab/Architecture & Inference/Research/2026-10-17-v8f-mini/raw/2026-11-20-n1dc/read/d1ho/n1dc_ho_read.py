"""n1dc_ho_read.py -- the D1-HO read of PREREG_V8F_N1DC.md (sha256 2eae7964, FROZEN), DEV BOX, CPU, zero training, one process.
Derived from the banked pod reader n1dw_ho_read.py: d1_row, paired, kin_acc, labels, cells_of, K1 / K-CONST are copied UNCHANGED.  Every root path is a CLI flag or env var.
Executors: "hb" = T1-kin/hb0.2 (EXECUTOR OF RECORD, decides) and "old" = T1-kin/old (beside).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0
(answers 'another draw of EPISODES' only; training variance = the two seeds).  Tier T1-kin (m=10 causal) for the cells; the T0 block reads the tick-0 plan `ol` of the same dumps.
The eval-141 guard and STRATEGIC are NOT here: every label that needs them prints PENDING-GUARD.
Run (real read):  python -I n1dc_ho_read.py --out OUT            (inputs under ./in, tree under ./tree; see HANDOFF_READ.md)
Replay:           python -I n1dc_ho_read.py --replay-w 2 --out OUT  (N1DC slots read the N1DW2 hb0.2 dumps; compare with replay_n1dw.py)
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
NB, SEED = 2000, 0
DT = 0.2
BAR_FLOOR = 0.0383
HO_WINDOWS, HO_EPISODES = 5736, 497
CL = ("lane_keep", "turn_left", "turn_right")
BANDS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30)}
METRICS = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "dk_hw_abs_err_m", "dk_tg_abs_err_s")
CELLS = tuple(f"{b}:{m}" for b in BANDS for m in METRICS) + ("TAC-LAT", "TAC-LON")
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL", "dk_hw_abs_err_m": "LONGITUDINAL",
          "dk_tg_abs_err_s": "LONGITUDINAL", "cross_mae_m": "LATERAL", "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"}
SAME = ("eid", "clip_index", "ws", "g", "fl", "v0")
GRID_SHA = "ca289d0f40a8f71043796fe626b26ecaa141853f843acdb1eecbf5648b9ab148"
BLOBS = {"taniteval/taniteval/four_families.py": "96839329", "taniteval/taniteval/ci.py": "528276c5", "stack/tanitad/refs/refc_tactical.py": "b7e44c3e", "taniteval/taniteval/lead_metrics.py": "25190ad6"}
F_HO_CARD = 0.042364016736401666
N1D_T0_BANKED = {"N1D_s0": 0.6265690376569037, "N1D_s1": 0.6349372384937239}      # PREREG_N1D_T0T1 / RESULT_T0T1.json level.T0.point
N1R = N1D = N1DC = FF = CI = BL = np = torch = None
G = {}                                                                          # run configuration, set by setup()


def setup(a):
    """put the exported tree on the path (env + sys.path), import the numeric stack and the pinned modules, verify the four blobs"""
    global np, torch, FF, CI, BL
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    tree = os.path.abspath(a.tree).replace("\\", "/")
    os.environ["N1DC_TREE"] = tree
    os.environ["TANITEVAL_STACK_OVERRIDE"] = tree + "/stack"
    sys.path.insert(0, tree + "/taniteval")
    sys.path.insert(0, tree + "/stack")
    sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True
    import numpy as _np
    import torch as _torch
    np, torch = _np, _torch
    torch.set_num_threads(2)
    import read_lib as _BL
    from taniteval import four_families as _FF
    BL, FF = _BL, _FF
    spec = importlib.util.spec_from_file_location("ci_dev", tree + "/taniteval/taniteval/ci.py")
    CI = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CI)
    bad = [f"{p}: {hashlib.sha256(open(tree + '/' + p, 'rb').read()).hexdigest()[:8]} != {w}" for p, w in BLOBS.items() if hashlib.sha256(open(tree + '/' + p, 'rb').read()).hexdigest()[:8] != w]
    G.update(tree=tree, inp=os.path.abspath(a.inp).replace("\\", "/"), blob_problems=bad)


# ------------------------------------------------ dump paths
def dpath(ex, arm, replay_w=None):
    """dump directory of a read arm.  arm = N1R_s0 | N1D_s0 | N1D_s0Z | N1DC_s0 | N1DC_s0Z (D-channel zero twin) | N1DC_s0U (skip zeroed at inference).  Returns a path or None."""
    I = G["inp"]
    zero, zuse = arm.endswith("Z"), arm.endswith("U")
    base = arm[:-1] if (zero or zuse) else arm
    if base.startswith("N1R"):
        return None if zero or zuse else (f"{I}/n1dw/hb/out/hb0.2/n1r/t1" if ex == "hb" else f"{I}/navrbd_step0/out/t1") + f"/dump_NAVRBD-HO-{base}_m10_causal_heads"
    if base.startswith("N1DC"):
        s = base[-1]
        if replay_w is not None:                                           # REPLAY: the N1DC slots read the N1DW{w} hb0.2 dumps (zero twin stands in for the zuse dump)
            if ex != "hb":
                return None
            sub = "t1_zero" if (zero or zuse) else "t1"
            return f"{I}/n1dw/hb/out/hb0.2/n1dw/{sub}/dump_N1DW-HO-N1DW{replay_w}_s{s}{'_zero' if (zero or zuse) else ''}_m10_causal_heads"
        ex_dir = "hb0.2" if ex == "hb" else "hbnone"
        name = f"dump_N1DC-HO-N1DC_s{s}{'_zero' if zero else '_zuse' if zuse else ''}_m10_causal_heads"
        subs = ("t1_zero",) if zero else ("t1", "t1_zuse", "t1_zero")       # the zuse dump's sub-directory is not fixed by the brief: first existing wins
        for sub in subs:
            p = f"{I}/n1dc/{ex_dir}/{sub}/{name}"
            if os.path.isdir(p):
                return p
        return f"{I}/n1dc/{ex_dir}/{subs[0]}/{name}"
    sub = "t1_zero" if zero else "t1"
    if ex == "hb":
        return f"{I}/n1dw/hb/out/hb0.2/n1d/{sub}/dump_NAVRBD-HO-{base}{'_zero' if zero else ''}_m10_causal_heads"
    return f"{I}/navrbd/out/{sub}/dump_NAVRBD-HO-{base}{'_zero' if zero else ''}_m10_causal_heads"


def load_dump_n(d, n):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == n, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


# ------------------------------------------------ pure logic (copied unchanged from n1dw_ho_read.py)
def labels(X):
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def cells_of(x, g):
    out = {}
    n = len(g)
    for band, (a, b) in BANDS.items():
        r = BL.band_rows(x["cl"], g, a, b)
        for m in METRICS:
            out[f"{band}:{m}"] = np.asarray(r[m], np.float64) if m in r else np.full(n, np.nan)
    return out


def check_three_class(lab):
    lab = np.asarray(lab)
    assert lab.min() >= 0 and lab.max() <= 2
    return lab


def kin_acc(g_lab, p_lab):
    return (np.asarray(p_lab) == np.asarray(g_lab)).astype(np.float64)


def paired(a, b, e, nb=NB):
    a, b, e = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(e)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-COMPUTABLE", "n_windows": 0, "n_episodes": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], e[m], n_boot=nb, seed=SEED)
    return {"status": "OK", "delta": float(a[m].mean() - b[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e[m])))}


def d1_row(acc_arm, acc_ref, eid, bar):
    r = paired(acc_arm, acc_ref, eid)
    r.update({"bar": float(bar), "gain": bool(r["delta"] >= bar and r["lo"] > 0), "cost": bool(r["delta"] <= -bar and r["hi"] < 0),
              "acc_arm": float(np.mean(acc_arm)), "acc_ref": float(np.mean(acc_ref))})
    return r


def level(a, e):
    r = CI.episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(e), reduce="mean", n_boot=NB, seed=SEED)
    return {"point": float(np.mean(a)), "lo": float(r["lo"]), "hi": float(r["hi"])}


def cellrows(C, arm, ref, eid):
    rows = {}
    for c in CELLS:
        r = paired(C[arm][c], C[ref][c], eid)
        if r["status"] != "OK":
            r["reason"] = "no lead block exists for held-out clips; n = 0"
        r["family"] = FAMILY[c.split(":")[-1]]
        r["higher_is_better"] = c in ("TAC-LAT", "TAC-LON")
        rows[c] = r
    return rows


# ------------------------------------------------ one executor
def read_executor(ex, a, banked, prob, notes):
    """returns the per-executor block; prob collects INVALID-making problems"""
    rw = a.replay_w
    arms = ["N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1D_s0Z", "N1D_s1Z"] + [f"N1DC_s{s}{z}" for s in (0, 1) for z in ("", "Z", "U")]
    X, miss = {}, []
    for arm in arms:
        d = dpath(ex, arm, rw)
        if d is None or not os.path.isdir(d):
            miss.append(arm)
            continue
        X[arm] = load_dump_n(d, HO_EPISODES)
    blk = {"dump_dirs": {x: dpath(ex, x, rw) for x in X}, "missing": miss}
    need = ["N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1D_s0Z", "N1D_s1Z"]
    if [m for m in miss if m in need]:
        prob.append(f"[{ex}] comparator dump missing: {[m for m in miss if m in need]}")
        return blk
    has_c = all(f"N1DC_s{s}" in X for s in (0, 1))
    if not has_c:
        (notes if (rw is not None and ex != "hb") else prob).append(f"[{ex}] N1DC dumps missing" + (" (replay: no old-executor N1DW dumps pulled; comparators and T0 only)" if rw is not None else ""))
    ref = X["N1R_s0"]
    eid = ref["eid"]
    if len(eid) != HO_WINDOWS or len(np.unique(eid)) != HO_EPISODES:
        prob.append(f"[{ex}] grid count {len(eid)} / {len(np.unique(eid))}")
    for k_, x in X.items():
        for k in SAME:
            if not np.array_equal(x[k], ref[k]):
                prob.append(f"[{ex}] window/model-free array differs from N1R_s0: {k} in {k_}")
    glat, glon = labels(ref["g"])
    check_three_class(glat)
    C, plat, T0 = {}, {}, {}
    for k_, x in X.items():
        cl_lat, cl_lon = labels(x["cl"])
        c = cells_of(x, ref["g"])
        c["TAC-LAT"] = kin_acc(glat, check_three_class(cl_lat))
        c["TAC-LON"] = (cl_lon == glon).astype(np.float64)
        C[k_], plat[k_] = c, cl_lat
        T0[k_] = kin_acc(glat, check_three_class(labels(x["ol"])[0]))
    if a.mutant and has_c and ex == "hb":                                    # MUTANT: shift one arm's TAC-LAT by one window
        C["N1DC_s0"]["TAC-LAT"] = np.roll(C["N1DC_s0"]["TAC-LAT"], 1)
        notes.append("MUTANT ACTIVE: N1DC_s0 TAC-LAT shifted by one window")
    acc = {k_: C[k_]["TAC-LAT"] for k_ in C}
    # ---- K-RE
    kre = {"acc": {}, "pass": True}
    bacc = banked["hb" if ex == "hb" else "old"]["D1_HO_each_acc"]
    for k_ in need:
        kre["acc"][k_] = {"live": float(acc[k_].mean()), "banked": bacc[k_], "equal": bool(float(acc[k_].mean()) == bacc[k_])}
        kre["pass"] &= kre["acc"][k_]["equal"]
    kre["banked_source"] = "RESULT_N1DW_D1HO_HB.json D1_HO_each_acc" if ex == "hb" else "RESULT_NAVRBD.json D1_HO_each_acc (+ N1D-N1R deltas)"
    if ex == "old":
        kre["delta"] = {}
        for s in (0, 1):
            r = d1_row(acc[f"N1D_s{s}"], acc[f"N1R_s{s}"], eid, F_HO_CARD)
            b = banked["old"]["D1_HO"][f"N1D_s{s}-N1R_s{s}"]
            ok = bool(r["delta"] == b["delta"] and round(r["lo"], 4) == round(b["lo"], 4) and round(r["hi"], 4) == round(b["hi"], 4))
            kre["delta"][f"N1D_s{s}-N1R_s{s}"] = {"live": [r["delta"], r["lo"], r["hi"]], "banked": [b["delta"], b["lo"], b["hi"]], "equal": ok}
            kre["pass"] &= ok
    blk["K-RE"] = kre
    if not kre["pass"]:
        prob.append(f"[{ex}] K-RE: recomputed N1R / N1D / zero-twin values != banked")
    # ---- K1, K-CONST
    k1 = paired(acc["N1R_s0"], acc["N1R_s0"], eid)
    blk["K1_self_pair_N1R_s0"] = {"delta": k1["delta"], "lo": k1["lo"], "hi": k1["hi"], "pass": bool(k1["delta"] == 0.0 and k1["lo"] == 0.0 and k1["hi"] == 0.0)}
    maj = int(np.bincount(glat, minlength=3).argmax())
    share = float((glat == maj).mean())
    cacc = float((np.full(len(glat), maj) == glat).mean())
    blk["K-CONST"] = {"majority_class": CL[maj], "majority_share": share, "constant_accuracy": cacc, "must_read": 0.338215, "pass": bool(cacc == share and round(share, 6) == 0.338215)}
    for k in ("K1_self_pair_N1R_s0", "K-CONST"):
        if not blk[k]["pass"]:
            prob.append(f"[{ex}] {k} failed")
    # ---- T0 (tick-0 decoded-plan TAC-LAT) and its control
    t0 = {k_: level(T0[k_], eid) for k_ in T0 if not k_.endswith(("Z", "U"))}
    blk["T0_level"] = t0
    kt0 = {k_: {"recomputed": t0[k_]["point"], "banked_PREREG_N1D_T0T1": v, "abs_diff": abs(t0[k_]["point"] - v), "pass": bool(abs(t0[k_]["point"] - v) <= 1e-12)} for k_, v in N1D_T0_BANKED.items()}
    blk["K-T0_N1D_rederived"] = {"cells": kt0, "pass": bool(all(v["pass"] for v in kt0.values())), "note": "N1D T0 = TAC-LAT of the tick-0 plan `ol` of the same dumps (as n1dwt0t1_read.py); tick 0 does not depend on the rollout executor"}
    if not blk["K-T0_N1D_rederived"]["pass"]:
        prob.append(f"[{ex}] K-T0: N1D T0 TAC-LAT != 0.6266 / 0.6349 (PREREG_N1D_T0T1)")
    # ---- floors, bar
    f_live = float(abs(acc["N1R_s1"].mean() - acc["N1R_s0"].mean()))
    f_ho = max(F_HO_CARD, f_live)
    blk["floor"] = {"F_HO_card": F_HO_CARD, "F_HO_live": f_live, "F_HO_used": f_ho, "seed_draw_N1D": float(abs(acc["N1D_s1"].mean() - acc["N1D_s0"].mean()))}
    blk["accuracy"] = {k_: float(v.mean()) for k_, v in acc.items()}
    blk["gt_class_share"] = {c: float((glat == i).mean()) for i, c in enumerate(CL)}
    if not has_c:
        return blk
    sd = float(abs(acc["N1DC_s1"].mean() - acc["N1DC_s0"].mean()))
    bar = float(max(BAR_FLOOR, f_ho, blk["floor"]["seed_draw_N1D"], sd))
    blk["floor"].update({"seed_draw_N1DC": sd, "bar": bar, "bar_rule": "max(0.0383, F_HO, |N1D_s1-N1D_s0|, |N1DC_s1-N1DC_s0|)"})
    D1, D1R = {}, {}
    for s in (0, 1):
        D1[f"N1DC_s{s}"] = d1_row(acc[f"N1DC_s{s}"], acc[f"N1D_s{s}"], eid, bar)          # DECIDING: N1DC - N1D
        D1R[f"N1DC_s{s}"] = d1_row(acc[f"N1DC_s{s}"], acc[f"N1R_s{s}"], eid, bar)         # beside: N1DC - N1R
    blk["D1_HO_deciding_N1DC-N1D"], blk["D1_HO_beside_N1DC-N1R"] = D1, D1R
    blk["R1_LON_LAT_HO_beside_no_bar"] = {f"N1DC_s{s}": {"N1DC-N1R": cellrows(C, f"N1DC_s{s}", f"N1R_s{s}", eid), "N1DC-N1D": cellrows(C, f"N1DC_s{s}", f"N1D_s{s}", eid)} for s in (0, 1)}
    # ---- T0 deltas
    blk["T0_N1DC_minus_N1D"] = {f"s{s}": paired(T0[f"N1DC_s{s}"], T0[f"N1D_s{s}"], eid) for s in (0, 1)}
    blk["T0_vs_T1_N1DC"] = {f"s{s}": {"T0": t0[f"N1DC_s{s}"]["point"], "T1": float(acc[f"N1DC_s{s}"].mean()), "T0_minus_T1": paired(T0[f"N1DC_s{s}"], acc[f"N1DC_s{s}"], eid)} for s in (0, 1)}
    blk["T0_target_note"] = "card section 0: target 0.73 (CEIL-AT-PLAN) against N1D decoded 0.6266 / 0.6349"
    # ---- Z-USE (skip zeroed at inference) and D-channel zero twin
    zu, zc = {}, {}
    for s in (0, 1):
        arm = f"N1DC_s{s}"
        if arm + "U" in X:
            zu[arm] = {"TAC-LAT": paired(acc[arm], acc[arm + "U"], eid), "cells": cellrows(C, arm, arm + "U", eid)}
        else:
            zu[arm] = {"status": "NOT-YET (no zuse dump)"}
        if arm + "Z" in X:
            zc[arm] = {"TAC-LAT": paired(acc[arm], acc[arm + "Z"], eid), "cells": cellrows(C, arm, arm + "Z", eid)}
    for s in (0, 1):
        zc[f"N1D_s{s}"] = {"TAC-LAT": paired(acc[f"N1D_s{s}"], acc[f"N1D_s{s}Z"], eid)}
    blk["Z_USE_skip_zeroed_real_minus_zuse"] = zu
    blk["Z_CHANNEL_D_zero_twin_real_minus_zero"] = zc
    blk["Z-EVAL_HO"] = "NOT-YET (needs the all-zero-channel-history mask of the held-out grid, --ho-zero-hist-mask; the eval-141 Z-EVAL belongs to the guard read)"
    return blk


# ------------------------------------------------ outcome label (card section 4), D1-HO part only
def outcome(blk):
    D = blk["D1_HO_deciding_N1DC-N1D"]
    gain = {s: D[f"N1DC_s{s}"]["gain"] for s in (0, 1)}
    cost = {s: D[f"N1DC_s{s}"]["cost"] for s in (0, 1)}
    PG = "PENDING-GUARD"
    lab = {}
    lab["N1DC-GAIN / N1DC-GAIN-COST (D1-HO gain above the bar at both seeds)"] = (PG + " (D1-HO part holds; GAIN if the guard holds at both seeds, GAIN-COST if a guard cost at one seed)" if gain[0] and gain[1] else "not fired")
    lab["N1DC-ONE-SEED / N1DC-GAIN-COST (D1-HO gain at exactly one seed)"] = (PG + f" (gain at seed {[s for s in (0, 1) if gain[s]]}; ONE-SEED if the guard holds at both, GAIN-COST if a guard cost at one seed)" if gain[0] != gain[1] else "not fired")
    lab["N1DC-COST (D1-HO a cost at both seeds, or guard cost at both)"] = ("FIRED on the D1-HO part (cost at both seeds)" if cost[0] and cost[1] else PG + " (D1-HO shows no cost at both seeds; the guard part needs eval-141)")
    lab["N1DC-NO-GAIN (otherwise)"] = (PG + " (D1-HO shows no gain at either seed; NO-GAIN unless the guard cost at both seeds fires COST)" if not (gain[0] or gain[1]) and not (cost[0] and cost[1]) else "not fired")
    return {"labels": lab, "gain_by_seed": {f"s{s}": gain[s] for s in (0, 1)}, "cost_by_seed": {f"s{s}": cost[s] for s in (0, 1)}, "guard": PG}


# ------------------------------------------------ banked values, report
def load_banked(a):
    b = a.banked_dir
    return {"hb": json.load(open(b + "/RESULT_N1DW_D1HO_HB.json", encoding="utf-8")), "old": json.load(open(b + "/RESULT_NAVRBD.json", encoding="utf-8"))}


def run(a):
    """one full read; returns (report, rc); never calls sys.exit"""
    t0 = time.time()
    setup(a)
    os.makedirs(a.out, exist_ok=True)
    banked = load_banked(a)
    prob, notes = [], []
    prob += [f"blob mismatch {m}" for m in G["blob_problems"]]
    grid = f"{G['inp']}/navrbd_step0/out/inputs/eval_grid_x_feasible.json"
    gsha = hashlib.sha256(open(grid, "rb").read()).hexdigest()
    if gsha != GRID_SHA:
        prob.append("grid sha256 differs")
    rep = {"tool": "n1dc_ho_read.py", "card": "PREREG_V8F_N1DC.md sha256 2eae7964 (FROZEN 2026-11-20)", "evidence_class": "MEASURED (ours; T1-kin dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={NB} seed 0 ({CI.__file__}); answers 'another draw of EPISODES' only; training variance = the two seeds; the executor is deterministic",
           "executor_of_record": "T1-kin/hb0.2 (v8f_t1_eval_hb.py --kappa-max 0.2, bcf267ce; PI ruling D16-D); T1-kin/old reported beside", "tier": "T1-kin (m=10 causal) cells; T0 block = tick-0 plan",
           "limit": "optimistic by construction: route supplied along the driven path", "grid_sha256": gsha, "replay_weight": a.replay_w, "mutant": bool(a.mutant), "problems": prob, "notes": notes}
    execs = {}
    for ex in a.execs:
        execs[ex] = read_executor(ex, a, banked, prob, notes)
        print(f"executor {ex} done {time.time() - t0:.0f}s", flush=True)
    rep["executors"] = execs
    hb = execs.get("hb", {})
    rep["outcome_D1HO_alone_hb0.2"] = outcome(hb) if "D1_HO_deciding_N1DC-N1D" in hb else "NOT-COMPUTABLE (N1DC dumps missing)"
    if "old" in execs and "D1_HO_deciding_N1DC-N1D" in execs["old"]:
        rep["outcome_D1HO_alone_old_beside"] = outcome(execs["old"])
    rep["status"] = "VALID" if not prob else "INVALID (" + "; ".join(prob) + ")"
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out + "/RESULT_N1DC_D1HO.json", "w"), indent=1, default=float)
    open(a.out + "/READ_N1DC_D1HO.txt", "w", encoding="utf-8").write(text(rep))
    return rep, (1 if prob else 0)


def text(rep):
    L = []
    P = L.append
    P("READ_N1DC_D1HO (MEASURED; T1-kin; D1-HO 5,736 windows / 497 held-out episodes; dev box CPU; B 2,000 seed 0 paired episode-cluster bootstrap = 'another draw of EPISODES' only)")
    P(f"card PREREG_V8F_N1DC.md sha256 2eae7964; executor of record T1-kin/hb0.2, T1-kin/old beside; eval-141 guard and STRATEGIC NOT here: every label that needs them is PENDING-GUARD")
    P(f"STATUS: {rep['status']}" + (f"   REPLAY w={rep['replay_weight']} (N1DC slots = N1DW dumps)" if rep["replay_weight"] else "") + ("   MUTANT" if rep["mutant"] else ""))
    P("")
    for ex, b in rep["executors"].items():
        P(f"=== executor {ex} ===")
        if "K-RE" not in b:
            P("  not read: " + str(b.get("missing")))
            continue
        P(f"  K-RE {'PASS' if b['K-RE']['pass'] else 'FAIL'} ({b['K-RE']['banked_source']}); K1 {'PASS' if b['K1_self_pair_N1R_s0']['pass'] else 'FAIL'}; K-CONST {b['K-CONST']['constant_accuracy']:.6f} {'PASS' if b['K-CONST']['pass'] else 'FAIL'}; K-T0 (N1D re-derived) {'PASS' if b['K-T0_N1D_rederived']['pass'] else 'FAIL'}")
        P("  accuracy " + ", ".join(f"{k} {v:.4f}" for k, v in b["accuracy"].items() if not k.endswith(("Z", "U"))))
        P("  T0 level " + ", ".join(f"{k} {v['point']:.4f} [{v['lo']:.4f},{v['hi']:.4f}]" for k, v in b["T0_level"].items()))
        if "D1_HO_deciding_N1DC-N1D" not in b:
            continue
        f = b["floor"]
        P(f"  bar {f['bar']:.6f} = max(0.0383, F_HO {f['F_HO_used']:.4f}, |N1D draw| {f['seed_draw_N1D']:.4f}, |N1DC draw| {f['seed_draw_N1DC']:.4f})")
        for k, nm in (("D1_HO_deciding_N1DC-N1D", "DECIDING N1DC - N1D"), ("D1_HO_beside_N1DC-N1R", "beside   N1DC - N1R")):
            for arm, r in b[k].items():
                P(f"  {nm} {arm}: {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] sep {r['separated']} gain {r['gain']} cost {r['cost']}")
        for s in (0, 1):
            t = b["T0_N1DC_minus_N1D"][f"s{s}"]
            P(f"  T0 TAC-LAT s{s}: N1DC {b['T0_level'][f'N1DC_s{s}']['point']:.4f} vs N1D {b['T0_level'][f'N1D_s{s}']['point']:.4f}; N1DC - N1D {t['delta']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}]; T1 {b['T0_vs_T1_N1DC'][f's{s}']['T1']:.4f}")
        for arm, z in b["Z_USE_skip_zeroed_real_minus_zuse"].items():
            P(f"  Z-USE (skip zeroed) {arm}: " + (f"{z['TAC-LAT']['delta']:+.4f} [{z['TAC-LAT']['lo']:+.4f},{z['TAC-LAT']['hi']:+.4f}]" if "TAC-LAT" in z else z["status"]))
        for arm, z in b["Z_CHANNEL_D_zero_twin_real_minus_zero"].items():
            P(f"  D-channel zero twin {arm}: {z['TAC-LAT']['delta']:+.4f} [{z['TAC-LAT']['lo']:+.4f},{z['TAC-LAT']['hi']:+.4f}]")
        P("  R1 cells on D1-HO (no bar; dk cells NOT-COMPUTABLE n=0) are in the JSON: R1_LON_LAT_HO_beside_no_bar")
        P("")
    if isinstance(rep["outcome_D1HO_alone_hb0.2"], dict):
        P("OUTCOME (card section 4, D1-HO under hb0.2 alone)")
        for k, v in rep["outcome_D1HO_alone_hb0.2"]["labels"].items():
            P(f"  {k}: {v}")
    P("problems: " + str(rep["problems"] or "none"))
    P("notes: " + str(rep["notes"] or "none"))
    return "\n".join(L)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", default=os.environ.get("N1DC_TREE", HERE + "/tree"), help="git archive export of 134f5da1 (stack + taniteval)")
    ap.add_argument("--inp", default=os.environ.get("N1DC_IN", HERE + "/in"), help="input root: n1dw/ navrbd/ navrbd_step0/ n1dc/ (pod /root/strat1 layout)")
    ap.add_argument("--banked-dir", default=os.environ.get("N1DC_BANKED", HERE + "/in/banked"), help="RESULT_N1DW_D1HO_HB.json, RESULT_NAVRBD.json")
    ap.add_argument("--out", default=os.environ.get("N1DC_OUT", HERE + "/out"))
    ap.add_argument("--execs", nargs="+", default=["hb", "old"], choices=["hb", "old"])
    ap.add_argument("--replay-w", type=int, default=None, choices=[2, 4], help="REPLAY: read the N1DW{w} hb0.2 dumps in the N1DC slots")
    ap.add_argument("--mutant", action="store_true", help="shift N1DC_s0 TAC-LAT by one window (the replay must go RED)")
    return ap


def main():
    a = parser().parse_args()
    rep, rc = run(a)
    print(text(rep)[:6000])
    print("WROTE", a.out, "status", rep["status"][:60], flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
