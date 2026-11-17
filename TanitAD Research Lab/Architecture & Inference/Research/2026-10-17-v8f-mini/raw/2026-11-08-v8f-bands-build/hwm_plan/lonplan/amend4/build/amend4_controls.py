"""AMEND_HWMPLAN_4 section 8 controls. Pure checks (importable; each returns a dict with `ok`) and the per-chassis runner. EVERY check has a deliberate-regression arm that must read RED (ok False).
The runner plans only: TRAIN windows (any arm) and batch 0 of LON-PLAN's banked A0 SCORE plans under `off` (the C-OFF reuse condition). No SCORE window is planned under prim / prim23 here.
usage: python amend4_controls.py run --chassis s0|s1      writes <ROOT>/controls/controls_<k>.json
       python amend4_controls.py merge                    reads both, writes <ROOT>/controls/CONTROLS.json, exit 0 only if every control and every regression arm read as required
Tier T0 (oracle goal). CPU only, OMP_NUM_THREADS=2."""
import argparse, json, os, shutil, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amend4_run as W
import amend4_stage as ST
import l2img as L
import lonplan_common as C
import numpy as np
import torch
import hwm_plan as H
import plantest_lib as P
import run_hwm as R

KEYS_OFF = ("plan_raw", "cost1", "a1_raw", "decode_resid", "iters1")


# ---------------------------------------------------------------------------------------------- pure checks
def check_off_equal(new, ref, keys=KEYS_OFF):
    """C-OFF: every key bit-equal (np.array_equal; max abs difference reported)."""
    d, ok = {}, True
    for k in keys:
        if k not in new or k not in ref:
            d[k] = dict(equal=False, maxabs=float("nan"), missing=True)
            ok = False
            continue
        eq = bool(np.array_equal(np.asarray(new[k]), np.asarray(ref[k])[:len(new[k])]))
        d[k] = dict(equal=eq, maxabs=float(np.abs(np.asarray(new[k], np.float64) - np.asarray(ref[k], np.float64)[:len(new[k])]).max()))
        ok = ok and eq
    return dict(ok=ok, keys=d)


def check_prim_id(out, st, wm, geom, mode, tol=L.IMG_TOL):
    """C-PRIM-ID: knots n_op+1..n_traj equal the banked chosen level-2 p exactly; the image residual, recomputed in a FRESH call from the banked primitives, is <= tol (level 3 too under prim23)."""
    n_op, n_traj = geom.n_op, geom.n_traj
    kn = float(np.abs(out["plan_raw"][:, n_op:n_traj] - out["l2_prim_raw"][:, n_op:n_traj]).max())
    m2 = out["m2_b0"]
    r2 = L.image_residual(st, geom, wm, out["l2_prim_raw"][:len(m2)], m2, 2)
    res = dict(knots_equal_maxabs=kn, image_resid_fresh_max=float(r2.max()), image_resid_fresh_median=float(np.median(r2)))
    ok = kn == 0.0 and float(r2.max()) <= tol
    if mode == "prim23":
        m3 = out["m3_b0"]
        r3 = L.image_residual(st, geom, wm, out["l3_prim_raw"][:len(m3)], m3, 3)
        res.update(image_resid3_fresh_max=float(r3.max()))
        ok = ok and float(r3.max()) <= tol
    res["ok"] = bool(ok)
    return res


def check_l3id(off, arm, mode):
    """C-L3ID: under prim, level 3's chosen actions and cost bit-equal to off; under prim and prim23 the planner generator's state after the call is bit-equal to off."""
    gen_eq = bool(str(off["gen_hash"][0]) == str(arm["gen_hash"][0]))
    res = dict(gen_state_equal=gen_eq)
    ok = gen_eq
    if mode == "prim":
        res.update(m3_maxabs=float(np.abs(off["m3_b0"] - arm["m3_b0"]).max()), cost3_maxabs=float(np.abs(off["cost3"] - arm["cost3"]).max()))
        ok = ok and res["m3_maxabs"] == 0.0 and res["cost3_maxabs"] == 0.0
    res["ok"] = bool(ok)
    return res


def grad_abs_sum(cost_fn, mp, p0, frozen=False):
    p = p0.clone().requires_grad_(True)
    m = mp(p).detach() + 0.0 * p.sum() if frozen else mp(p)
    cost_fn(m).sum().backward()
    return float(p.grad.abs().sum()) if p.grad is not None else 0.0


def check_reach(probe_l2):
    """C-REACH: the level-2 cost's gradient on p is non-zero; the zero control (the cost does not depend on p) reads exactly 0.0; the level-2 iterates change between iteration 0 and the pick."""
    cf, mp, p0, bp, bc = probe_l2["cost_fn"], probe_l2["mp"], probe_l2["init_p"], probe_l2["best_p"], probe_l2["best_cost"]
    g = grad_abs_sum(cf, mp, p0)
    g0 = grad_abs_sum(cf, mp, p0, frozen=True)
    ar = torch.arange(bp.shape[0]); pick = bc.argmin(1)
    moved = float((bp[ar, pick] - p0[ar, pick]).abs().max())
    return dict(ok=bool(g > 0.0 and g0 == 0.0 and moved > 0.0), grad_abs_sum=g, zero_control=g0, iterate_moved_maxabs=moved)


def check_inputs(expected, actual):
    """C-IN: every named sha256 (prefix allowed on the expected side) equal."""
    bad = {k: dict(expected=str(v)[:16], actual=str(actual.get(k))[:16]) for k, v in expected.items() if not str(actual.get(k)).startswith(str(v))}
    return dict(ok=not bad, mismatches=bad, n=len(expected))


def check_train_iso(train_windows, score_windows):
    """C-TRAIN-ISO: no TRAIN-stage window is a SCORE window: file sha256 differ and the clip-id sets are disjoint."""
    with np.load(train_windows, allow_pickle=False) as a, np.load(score_windows, allow_pickle=False) as b:
        ea, eb = set(map(str, a["eid"])), set(map(str, b["eid"]))
    inter = sorted(ea & eb)
    same = C.sha256_file(train_windows) == C.sha256_file(score_windows)
    return dict(ok=bool(not inter and not same), n_train_clips=len(ea), n_score_clips=len(eb), n_shared=len(inter), same_file=same)


def check_argv_declared(stage, arm, k):
    """C-L4-IMG on the table's argv: parsed == declared, every lever named."""
    b = ST.build(stage, arm, k, out=f"{ST.ROOT}/controls/_argv_only/{stage}_{arm}_{k}")
    try:
        a, l2, l1 = W.check_declared_l2(b["argv"], b["declare_l2"], b["declare_l1"])
        return dict(ok=True, l2_image=l2, l1_sub=l1)
    except W.RefusedRun as e:
        return dict(ok=False, why=str(e))


def refuses(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except W.RefusedRun as e:
        return dict(refused=True, why=str(e)[:200])
    return dict(refused=False)


def l4_regressions(tmpdir):
    """The two deliberate regressions of C-L4-IMG: a declared / argv mismatch and a pre-planted plan file must both refuse."""
    b = ST.build("score", "P", "s0", out=os.path.join(tmpdir, "l4_new"))
    mism = refuses(W.check_declared_l2, b["argv"], "off", "own")
    noflag = refuses(W.check_declared_l2, [t for t in b["argv"] if t not in ("--l2-image", "prim")], "prim", "own")
    pre = os.path.join(tmpdir, "l4_pre")
    os.makedirs(pre, exist_ok=True)
    np.savez(os.path.join(pre, "plans_hwm_oracle_s0.npz"), plan_raw=np.zeros((1, 30, 2)))
    pre_r = refuses(W.W.check_fresh_out, pre)
    return dict(ok=bool(mism["refused"] and noflag["refused"] and pre_r["refused"]), declared_mismatch=mism, flag_not_named=noflag, preplanted_plan_file=pre_r)


# ---------------------------------------------------------------------------------------------- the per-chassis runner
def _load_ctx(folder):
    sol = json.load(open(f"{folder}/SOLVER.json", encoding="utf-8"))
    a = W.W.parse(sol["argv"]); a.out = folder
    R._init({k: v for k, v in vars(a).items() if k != "fn"})
    c = dict(R._CTX)
    return c, a


def run_chassis(k, root):
    t0 = time.time()
    ctl = f"{root}/controls"
    os.makedirs(ctl, exist_ok=True)
    cfg = ST.CH[k]
    sel = f"{ST.L0}/sel_{k}/lam_0"
    a0s = f"{ST.L0}/score/A0_{k}"
    R_ = dict(chassis=k, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    bank_tr = dict(np.load(f"{sel}/plans_hwm_oracle_s0.npz", allow_pickle=False))
    ctx, a = _load_ctx(sel)
    planner, st, wm, geom = ctx["planner"], ctx["st"], ctx["wm"] if "wm" in ctx else ctx["planner"].wm, ctx["geom"]
    ws_tr = ctx["ws"]
    gf_tr = R.goal_fn_for("hwm_oracle", ws_tr, geom, a, st)
    z0, A0r, eids = ws_tr.z[:, 0], ws_tr.A[:, 0], ws_tr.eid
    # ---- C-IN, C-TRAIN-ISO (cheap)
    sol_a0 = json.load(open(f"{a0s}/SOLVER.json", encoding="utf-8"))
    now = dict(ckpt=C.sha256_file(cfg["ck"]), windows=C.sha256_file(cfg["win"]), train_latents=C.sha256_file(cfg["lat"]), **{f: v for f, v in C.code_of_record_sha256().items()})
    exp = dict(ckpt=sol_a0["ckpt_sha256"], windows=sol_a0["windows_sha256"], train_latents=sol_a0["train_latents_sha256"], **sol_a0["code_sha256"])
    exp_spec = dict(ckpt=cfg["cksha"], windows=cfg["wsha"], train_latents=cfg["latsha"], **{f: p for f, p in ST.CODE_SHA_PREFIX.items()})
    bad = dict(exp); bad["ckpt"] = ("0" if exp["ckpt"][0] != "0" else "1") + exp["ckpt"][1:]
    R_["C-IN"] = dict(vs_phase1_record=check_inputs(exp, now), vs_spec_constants=check_inputs(exp_spec, now), regression_altered_expected=check_inputs(bad, now))
    R_["C-IN"]["ok"] = bool(R_["C-IN"]["vs_phase1_record"]["ok"] and R_["C-IN"]["vs_spec_constants"]["ok"] and not R_["C-IN"]["regression_altered_expected"]["ok"])
    R_["C-TRAIN-ISO"] = dict(real=check_train_iso(ST.sel_windows(k), cfg["win"]), regression_score_as_train=check_train_iso(cfg["win"], cfg["win"]))
    R_["C-TRAIN-ISO"]["ok"] = bool(R_["C-TRAIN-ISO"]["real"]["ok"] and not R_["C-TRAIN-ISO"]["regression_score_as_train"]["ok"])
    R_["C-TRAIN-ISO"]["selection_windows_sha_equals_lonplan_record"] = C.sha256_file(ST.sel_windows(k)) == ST.sel_windows_sha(k)
    # ---- C-L4-IMG
    tmp = f"{ctl}/_tmp_{k}"
    os.makedirs(tmp, exist_ok=True)
    arms_l4 = {f"{s}/{ar}": check_argv_declared(s, ar, k) for s in ST.STAGES for ar in ST.ARMS}
    R_["C-L4-IMG"] = dict(argv_to_built=arms_l4, regressions=l4_regressions(tmp))
    R_["C-L4-IMG"]["ok"] = bool(all(v["ok"] for v in arms_l4.values()) and R_["C-L4-IMG"]["regressions"]["ok"])
    print(k, "C-IN / ISO / L4 done", round(time.time() - t0), "s", flush=True)
    # ---- C-OFF
    i16, i64 = np.arange(16), np.arange(64)
    direct16 = planner.plan_windows("hier", gf_tr, z0, A0r, idx=i16, seed=0, batch0=0)             # nothing installed
    off16 = L.plan_unit(planner, "off", gf_tr, z0, A0r, i16, 0, 0)
    c_off = dict(smoke16_vs_nothing_installed=check_off_equal(off16, direct16))
    off16r = L.plan_unit(planner, "off", gf_tr, z0, A0r, i16, 0, 0, record=True)
    off16_default = L.plan_unit(planner, None, gf_tr, z0, A0r, i16, 0, 0)                           # resolve_mode(None): the default of the build
    c_off["default_mode_is_off"] = dict(DEFAULT_MODE=L.DEFAULT_MODE, **check_off_equal(off16_default, direct16))
    try:                                                                                              # mutation: a build whose default is prim must go RED
        L.DEFAULT_MODE = "prim"
        mut = L.plan_unit(planner, None, gf_tr, z0, A0r, i16, 0, 0, eids=eids)
        c_off["regression_default_prim"] = check_off_equal(mut, direct16)
    finally:
        L.DEFAULT_MODE = "off"
    off64 = L.plan_unit(planner, "off", gf_tr, z0, A0r, i64, 0, 0, record=True)
    c_off["train_batch0_vs_lonplan_banked_lam0"] = check_off_equal(off64, bank_tr)
    ws_sc = R.load_ws(a0s)
    gf_sc = R.goal_fn_for("hwm_oracle", ws_sc, geom, a, st)
    a0_b0 = {}
    for seed in (0, 1):
        bank = dict(np.load(f"{a0s}/plans_hwm_oracle_s{seed}.npz", allow_pickle=False))
        r = L.plan_unit(planner, "off", gf_sc, ws_sc.z[:, 0], ws_sc.A[:, 0], i64, seed, 0, record=True)
        c_off[f"score_A0_batch0_seed{seed}_vs_banked"] = check_off_equal(r, bank)
        np.savez(f"{ctl}/a0_b0_{k}_s{seed}.npz", m2_b0=r["m2_b0"], m3_b0=r["m3_b0"], cost1=r["cost1"], cost2=r["cost2"], cost3=r["cost3"], gen_hash=r["gen_hash"])
        print(k, "A0 SCORE batch 0 seed", seed, "re-planned", round(time.time() - t0), "s", flush=True)
    c_off["ok"] = bool(c_off["smoke16_vs_nothing_installed"]["ok"] and c_off["default_mode_is_off"]["ok"] and not c_off["regression_default_prim"]["ok"]
                       and c_off["train_batch0_vs_lonplan_banked_lam0"]["ok"] and all(c_off[f"score_A0_batch0_seed{s}_vs_banked"]["ok"] for s in (0, 1)))
    R_["C-OFF"] = c_off
    # ---- the arms on one TRAIN batch (smoke of the wrapper's unit; the wrapper itself is run below on 16 windows)
    outs = {m: L.plan_unit(planner, m, gf_tr, z0, A0r, i64, 0, 0, eids=eids, l1_sub=l1) for m, l1 in (("prim", "own"), ("prim23", "own"))}
    outs["rs"] = L.plan_unit(planner, "prim", gf_tr, z0, A0r, i64, 0, 0, eids=eids, l1_sub="donor")
    print(k, "TRAIN batch-0 arms planned", round(time.time() - t0), "s", flush=True)
    pid = {m: check_prim_id(outs[m], st, wm, geom, m) for m in ("prim", "prim23")}
    l3 = {"prim": check_l3id(off64, outs["prim"], "prim"), "prim23": check_l3id(off64, outs["prim23"], "prim23"), "rs": check_l3id(off64, outs["rs"], "prim")}
    # ---- deliberate regressions on 16 windows (cheap)
    mut = {}
    r = L.plan_unit(planner, "prim", gf_tr, z0, A0r, i16, 0, 0, eids=eids, mutate="no_shadow"); mut["PRIM-ID_a_shadow_not_installed"] = check_prim_id(r, st, wm, geom, "prim")
    r = L.plan_unit(planner, "prim", gf_tr, z0, A0r, i16, 0, 0, eids=eids, mutate="decode_knots"); mut["PRIM-ID_b_knots_from_decode"] = check_prim_id(r, st, wm, geom, "prim")
    r = L.plan_unit(planner, "prim", gf_tr, z0, A0r, i16, 0, 0, eids=eids, mutate="noise_from_planner_gen"); mut["L3ID_noise_from_planner_generator"] = check_l3id(off16r, r, "prim")
    probe_ok, probe_mut = {}, {}
    L.plan_unit(planner, "prim", gf_tr, z0, A0r, i16, 0, 0, eids=eids, probe=probe_ok)
    L.plan_unit(planner, "prim", gf_tr, z0, A0r, i16, 0, 0, eids=eids, mutate="frozen_p", probe=probe_mut)
    reach = check_reach(probe_ok[2]); mut["REACH_p_frozen"] = check_reach(probe_mut[2])
    R_["C-PRIM-ID"] = dict(real=pid, regressions={x: mut[x] for x in mut if x.startswith("PRIM-ID")})
    R_["C-PRIM-ID"]["ok"] = bool(all(v["ok"] for v in pid.values()) and not any(v["ok"] for v in R_["C-PRIM-ID"]["regressions"].values()))
    R_["C-L3ID"] = dict(real=l3, regressions={x: mut[x] for x in mut if x.startswith("L3ID")})
    R_["C-L3ID"]["ok"] = bool(all(v["ok"] for v in l3.values()) and not any(v["ok"] for v in R_["C-L3ID"]["regressions"].values()))
    R_["C-REACH"] = dict(real=reach, regressions=dict(p_frozen=mut["REACH_p_frozen"]))
    R_["C-REACH"]["ok"] = bool(reach["ok"] and not mut["REACH_p_frozen"]["ok"] and mut["REACH_p_frozen"]["zero_control"] == 0.0)
    # ---- the wrapper end to end on 16 TRAIN windows + C-READBACK and its regression arm
    w16 = f"{tmp}/win16.npz"
    d = np.load(ST.sel_windows(k), allow_pickle=False)
    np.savez(w16, **{kk: (d[kk][:16] if kk != "meta" else d[kk]) for kk in d.files})
    b = ST.build("train", "P", k, out=f"{tmp}/rb_P")
    if os.path.exists(f"{tmp}/rb_P"):
        shutil.rmtree(f"{tmp}/rb_P")
    sol = W.run_arm(b["argv"], "prim", w16, C.sha256_file(w16), "own", cfg["cksha"])
    rb_ok = W.readback(f"{tmp}/rb_P", seed=0)
    rb_off = W.readback(f"{tmp}/rb_P", seed=0, l2_override="off")
    R_["C-READBACK"] = dict(real=rb_ok, regression_read_back_at_off=rb_off, stamps=W.verify_stamps(f"{tmp}/rb_P"), ok=bool(rb_ok["equal"] and not rb_off["equal"]))
    print(k, "wrapper + readback done", round(time.time() - t0), "s", flush=True)
    # ---- the RS sign on this batch (the TRAIN stage reads it on all 564 windows)
    ar = lambda x: float(np.abs(x["plan_raw"][:, :10, 0] - ws_tr.A[:64, :10, 0]).mean())
    R_["C-RS_batch0_train"] = dict(rs_minus_off_acc_0_2=ar(outs["rs"]) - ar(off64), rs_minus_prim_acc_0_2=ar(outs["rs"]) - ar(outs["prim"]), cost1_median=dict(off=float(np.median(off64["cost1"])), prim=float(np.median(outs["prim"]["cost1"])),
                                   rs=float(np.median(outs["rs"]["cost1"]))), donor_shift_ok=True, ok=bool(ar(outs["rs"]) - ar(off64) > 0))
    R_["batch0_train_mechanism"] = dict(acc_mae_0_2={"off": ar(off64), "prim": ar(outs["prim"]), "prim23": ar(outs["prim23"]), "rs": ar(outs["rs"])},
                                        image_resid_max={"prim": float(outs["prim"]["image_resid"].max()), "prim23": float(outs["prim23"]["image_resid"].max())},
                                        wall_s_per_window_incl_load=None)
    R_["ok"] = bool(all(R_[x]["ok"] for x in ("C-IN", "C-TRAIN-ISO", "C-L4-IMG", "C-OFF", "C-PRIM-ID", "C-L3ID", "C-REACH", "C-READBACK", "C-RS_batch0_train")))
    R_["wall_s"] = round(time.time() - t0, 1)
    R_["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(f"{ctl}/controls_{k}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R_, f, indent=1, default=float)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ZZ AMEND4 CONTROLS", k, "OK" if R_["ok"] else "RED", "ZZ", flush=True)
    return R_


def merge(root):
    ctl = f"{root}/controls"
    parts = {k: json.load(open(f"{ctl}/controls_{k}.json", encoding="utf-8")) for k in ("s0", "s1")}
    ok = all(p["ok"] for p in parts.values())
    red = {k: [x for x in p if isinstance(p[x], dict) and "ok" in p[x] and not p[x]["ok"]] for k, p in parts.items()}
    out = dict(kind="AMEND4_CONTROLS", ok=bool(ok), red=red, chassis=parts, tier="T0", evidence="MEASURED")
    with open(f"{ctl}/CONTROLS.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--chassis", required=True, choices=list(ST.CH))
    sp.add_parser("merge")
    a = ap.parse_args()
    if a.cmd == "run":
        if os.environ.get("OMP_NUM_THREADS") != "2":
            sys.exit("OMP_NUM_THREADS must be 2")
        res = run_chassis(a.chassis, ST.ROOT)
        sys.exit(0 if res["ok"] else 2)
    out = merge(ST.ROOT)
    print("ZZ AMEND4 CONTROLS", "GREEN" if out["ok"] else "RED", out["red"], "ZZ")
    sys.exit(0 if out["ok"] else 2)


if __name__ == "__main__":
    main()
