#!/usr/bin/env python3
"""gates_exec.py -- the four gates of PREREG_EXEC_STANDSTILL.md section 4, each with its deliberate-regression (RED) arm, on the gate subset's rollouts.

Pure checks (numpy only; the unit tests in tests/ call them on synthetic arrays), then a production mode that reads the rollouts written by gate_rollouts.sh.

  G-ID    eps 1e-3 reproduces the base rollouts bit for bit.              RED: eps 0.1 must differ on launch windows.
  G-MOV   on windows whose every executed step exceeds 0.1 m, hold == base exactly.   RED: an over-aggressive hold (eps 2.0) must differ there.
  G-PHYS  under hold no executed step shorter than eps changes the heading.           RED: hold disabled (base executor) must violate it.
  G-P10   P10's prior is unchanged, only its executed history differs.    RED: a mutated floor (prior output + 1e-3 m) must break the tick-0 identity.

usage: python -I gates_exec.py --gate-dir C:/Users/Admin/exec_standstill_scratch/gate --out <raw/GATES.json>
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import sys

import numpy as np

DT = 0.2
EPS_HOLD = 0.1
EPS_BASE = 1e-3
SUB = "C:/Users/Admin/exec_standstill_scratch/subset"
BANKED_C0_S0_DUMP = "C:/Users/Admin/rr_scratch/out/C0_s0/dump_C0_s0"
ORIG_TOOL = "C:/Users/Admin/exec_standstill_scratch/tree/stack/scripts/v8f_t1_eval_ri.py"
EXEC_TOOL = "C:/Users/Admin/exec_standstill_scratch/tree/stack/scripts/v8f_t1_eval_exec.py"


# ------------------------------------------------------------------------------------------------ pure layer
def eq(a, b):
    """Bit equality of two arrays (NaN == NaN), shapes included."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def per_window_equal(a, b):
    """[N] bool: window i of a and b are bit-equal over every trailing axis (NaN == NaN)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = (a == b) | (np.isnan(a) & np.isnan(b))
    return ok.reshape(len(a), -1).all(1)


def step_len(track):
    """[N, 60] the executed tangent norm per 0.2 s interval of every 10 Hz frame: the track's v column times DT (v = nrm / DT in hermite_frames)."""
    return np.asarray(track, np.float64)[..., 3] * DT


def phys_violations(track, psi0, eps):
    """G-PHYS: frames whose step < eps (strictly, with a 1e-9 relative tie band excluded) must carry the heading of the frame before them (the first frame: psi0).
    -> dict(n_below, n_windows_below, n_violations, n_windows_violating, n_tie)."""
    tr = np.asarray(track, np.float64)
    st = step_len(tr)
    psi = tr[..., 2]
    prev = np.concatenate([np.asarray(psi0, np.float64)[:, None], psi[:, :-1]], axis=1)
    below = st < eps * (1 - 1e-9)
    tie = np.abs(st - eps) <= eps * 1e-9
    viol = below & (psi != prev)
    return {"n_frames": int(st.size), "n_below": int(below.sum()), "n_windows_below": int(below.any(1).sum()), "n_violations": int(viol.sum()),
            "n_windows_violating": int(viol.any(1).sum()), "n_tie": int(tie.sum())}


def all_steps_above(track, eps):
    """[N] bool: every executed frame of the window has step >= eps (1e-9 relative margin above)."""
    return (step_len(track) >= eps * (1 + 1e-9)).all(1)


def g_id_identity(base, other, launch_mask):
    """Identity of two runs' executed paths (cl, fl) on all windows and on the launch windows. -> dict."""
    out = {}
    for k in ("cl", "fl"):
        pw = per_window_equal(base[k], other[k])
        out[k] = {"all_equal": bool(pw.all()), "n_unequal": int((~pw).sum()), "n_unequal_launch": int((~pw & launch_mask).sum()), "max_abs_dev_m": float(np.nanmax(np.abs(base[k] - other[k])))}
    out["identical"] = bool(out["cl"]["all_equal"] and out["fl"]["all_equal"])
    return out


def tick0_identity(fl_a, fl_b, k=10):
    """G-P10: the P10 floor's first k executed knots are its tick-0 plan; the history that plan reads is purely logged, so it is the same under both executors. -> per-window bool [N]."""
    return per_window_equal(np.asarray(fl_a)[:, :k], np.asarray(fl_b)[:, :k])


# ------------------------------------------------------------------------------------------------ production layer
def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("cl", "fl", "ol", "ws", "eid", "g", "v0")}


def load_run(root, tag):
    ex = os.path.join(root, tag, f"exec_ch_{tag}.npz")
    t = np.load(os.path.join(root, tag, "ticks", f"ticks_{tag}.npz"), allow_pickle=True)
    r = {"dump": load_dump(os.path.join(root, tag, f"dump_{tag}")), "ticks": {k: t[k] for k in t.files}}
    if os.path.exists(ex):
        z = np.load(ex, allow_pickle=True)
        r["exc"] = {k: z[k] for k in z.files}
    return r


def fn_source(path, name):
    src = open(path, encoding="utf-8").read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise KeyError(name)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", default="C:/Users/Admin/exec_standstill_scratch/gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p10-red-dir", default="", help="the G_P10_RED rollout directory root (gate_p10_red.py's output); empty = not run")
    a = ap.parse_args(argv)
    info = json.load(open(f"{SUB}/subset_idx.json", encoding="utf-8"))
    idx = np.array(info["idx"])
    cls = np.array(info["class"])
    R0 = load_run(a.gate_dir, "G_R0_orig")
    R1, R2, R3, R4 = (load_run(a.gate_dir, t) for t in ("G_R1_default", "G_R2_eps1e-3", "G_R3_hold01", "G_R4_over2"))
    v0 = R2["dump"]["v0"]
    launch = (cls == "L") & (v0 < 0.5)                    # launches from (near) standstill, the windows the artifact lives on
    stopped = v0 < 0.5
    res = {"subset": {"n_windows": int(len(idx)), "class_counts": info["class_counts_subset"], "n_launch_from_stop": int(launch.sum()), "n_stopped": int(stopped.sum())}}

    # ---- G-ID
    id_orig_default = g_id_identity(R0["dump"], R1["dump"], launch)
    id_orig_explicit = g_id_identity(R0["dump"], R2["dump"], launch)
    tick_keys = [k for k in R0["ticks"] if k not in ("tag", "ckpt", "clip_names")]
    ticks_equal = {k: eq(R0["ticks"][k], R2["ticks"][k]) for k in tick_keys}
    ol_equal = eq(R0["dump"]["ol"], R2["dump"]["ol"])
    exc_equal = {k: eq(R1["exc"][k], R2["exc"][k]) for k in R1["exc"] if k != "exec_eps"}
    banked = load_dump(BANKED_C0_S0_DUMP)
    bsub = {k: banked[k][idx] for k in ("cl", "fl", "ol", "ws", "eid", "g", "v0")}
    assert eq(bsub["ws"], R2["dump"]["ws"]) and eq(bsub["g"], R2["dump"]["g"]), "subset windows are not the banked windows"
    id_banked = g_id_identity(bsub, R2["dump"], launch)
    id_banked["ol_equal"] = eq(bsub["ol"], R2["dump"]["ol"])
    red_id = g_id_identity(R2["dump"], R3["dump"], launch)
    id_pass = bool(id_orig_default["identical"] and id_orig_explicit["identical"] and ol_equal and all(ticks_equal.values()) and all(exc_equal.values()))
    id_red = bool(red_id["cl"]["n_unequal_launch"] > 0 and red_id["fl"]["n_unequal_launch"] > 0)         # the RED arm must differ on launch windows (both rollouts)
    res["G-ID"] = {"PASS": bool(id_pass and id_red), "same_session_unpatched_vs_exec_default": id_orig_default, "same_session_unpatched_vs_exec_eps1e-3": id_orig_explicit,
                   "ticks_fields_equal": ticks_equal, "ol_equal": ol_equal, "exec_ch_default_vs_eps1e-3_equal": exc_equal,
                   "vs_banked_C0_s0_rollout_on_the_subset": id_banked, "banked_note": "different session, tool 0721fc01 on 16 threads vs 949c4b7a-derived on 12; reported, not the gate (thread-order float noise is a known property, see recover_read.EXEC_TOL_M)",
                   "RED_eps0.1_vs_base": {**red_id, "differs_on_launch_windows_cl_and_fl": id_red}}

    # ---- G-MOV
    ex2, ex3, ex4 = R2["exc"], R3["exc"], R4["exc"]
    mov_cl, mov_fl = all_steps_above(ex2["cl_track"], EPS_HOLD), all_steps_above(ex2["fl_track"], EPS_HOLD)
    mov_both = mov_cl & mov_fl
    mov = {}
    for name, ex in (("hold_eps0.1", ex3), ("RED_over_eps2.0", ex4)):
        ent = {}
        for role, m in (("cl", mov_cl), ("fl", mov_fl)):
            pw = per_window_equal(R2["dump"][role], (R3 if ex is ex3 else R4)["dump"][role]) & per_window_equal(ex2[role + "_track"], ex[role + "_track"]) & per_window_equal(ex2[role + "_ch1"], ex[role + "_ch1"])
            ent[role] = {"n_moving_windows": int(m.sum()), "n_equal_on_moving": int((pw & m).sum()), "n_unequal_on_moving": int((~pw & m).sum())}
        ent["all_equal_on_moving"] = bool(ent["cl"]["n_unequal_on_moving"] == 0 and ent["fl"]["n_unequal_on_moving"] == 0)
        mov[name] = ent
    nonvac = bool(mov_cl.sum() >= 20 and mov_fl.sum() >= 20)
    mov_pass = bool(mov["hold_eps0.1"]["all_equal_on_moving"] and nonvac)
    mov_red = bool(not mov["RED_over_eps2.0"]["all_equal_on_moving"])
    res["G-MOV"] = {"PASS": bool(mov_pass and mov_red), "n_moving_model": int(mov_cl.sum()), "n_moving_floor": int(mov_fl.sum()), "n_moving_both": int(mov_both.sum()), "non_vacuous_ge20": nonvac,
                    "hold_eps0.1_vs_base": mov["hold_eps0.1"], "RED_over_aggressive_hold_eps2.0_vs_base": {**mov["RED_over_eps2.0"], "differs_on_moving_windows": mov_red}}

    # ---- G-PHYS
    tk = R2["ticks"]
    psi0 = np.array([tk["pose"][(tk["tick"] == 0) & (tk["win"] == w)][0][2] for w in range(len(idx))])
    ph = {}
    for name, ex in (("hold_eps0.1", ex3), ("RED_hold_disabled_base", ex2)):
        ph[name] = {role: phys_violations(ex[role + "_track"], psi0, EPS_HOLD) for role in ("cl", "fl")}
    nonvac_ph = bool(ph["hold_eps0.1"]["cl"]["n_below"] > 0 and ph["hold_eps0.1"]["fl"]["n_below"] > 0)
    ph_pass = bool(all(ph["hold_eps0.1"][r]["n_violations"] == 0 for r in ("cl", "fl")) and nonvac_ph)
    ph_red = bool(all(ph["RED_hold_disabled_base"][r]["n_violations"] > 0 for r in ("cl", "fl")))
    res["G-PHYS"] = {"PASS": bool(ph_pass and ph_red), "hold_eps0.1": ph["hold_eps0.1"], "non_vacuous": nonvac_ph, "RED_hold_disabled": {**ph["RED_hold_disabled_base"], "violates": ph_red}}

    # ---- G-P10
    t0_eq = tick0_identity(R2["dump"]["fl"], R3["dump"]["fl"])
    later_changed = ~per_window_equal(R2["dump"]["fl"][:, 10:], R3["dump"]["fl"][:, 10:])
    src_o, src_e = fn_source(ORIG_TOOL, "floor_planner"), fn_source(EXEC_TOOL, "floor_planner")
    p10 = {"floor_planner_source_sha256_unpatched": sha(src_o), "floor_planner_source_sha256_exec": sha(src_e), "floor_planner_source_identical": src_o == src_e,
           "tick0_plan_identical_all_windows": bool(t0_eq.all()), "n_windows_tick0_unequal": int((~t0_eq).sum()),
           "n_windows_later_knots_differ_under_hold": int(later_changed.sum()), "n_stopped_windows_later_knots_differ": int((later_changed & stopped).sum()), "non_vacuous": bool(later_changed.sum() > 0)}
    p10_pass = bool(p10["floor_planner_source_identical"] and p10["tick0_plan_identical_all_windows"] and p10["non_vacuous"])
    red = None
    if a.p10_red_dir:
        Rm = load_run(a.p10_red_dir, "G_P10_RED")
        te = tick0_identity(R2["dump"]["fl"], Rm["dump"]["fl"])
        red = {"n_windows_tick0_unequal": int((~te).sum()), "breaks_tick0_identity": bool((~te).any()), "mutation": "floor_planner returns the prior output + 1e-3 m (harness monkeypatch, gate_p10_red.py)"}
    p10["RED_mutated_floor"] = red
    res["G-P10"] = {"PASS": bool(p10_pass and (red is not None and red["breaks_tick0_identity"])), **p10}
    res["ALL_GATES_PASS"] = bool(all(res[g]["PASS"] for g in ("G-ID", "G-MOV", "G-PHYS", "G-P10")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({g: res[g]["PASS"] for g in ("G-ID", "G-MOV", "G-PHYS", "G-P10")}), "ALL", res["ALL_GATES_PASS"])
    return 0 if res["ALL_GATES_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
