"""Controls (i)-(iii) of protocol B3 (BESIDE_RULES_SLOWWM.md), each with a red arm that must make it fail. CPU. Writes CONTROLS_B3.json (all_pass).
  (i)   Identity. P = I_128 through the restricted-cost wrapper reproduces the full-width planner BIT FOR BIT on 64 windows (plan, cost1, init1, iters1; context 1, prior seed) and on 8 windows of context 3 (hold seed, lambda 0.3);
        the wrapper leaves nothing behind (a third run after it equals the first). Red arm: P = I with one diagonal entry zeroed must change cost and plan.
  (ii)  Bottom-64. S built from basis columns 192-255 gives a plan different from the full-width plan and from the top-64 plan on the same 64 windows. Red arm: the "bottom" columns mutated to the top columns -> the plan equals the top plan.
  (iii) Jacobian. J-bar by autograd equals a central finite difference (step 1e-3, float64) within 1e-3 relative (Frobenius), for the joint-shift and the newest-frame forms. Red arms: autograd columns shifted by one; the newest-frame
        form against the joint-shift finite difference (K = 3).
All on a PLANTED basis arm (R35H s0's checkpoint with wm_basis = slow, a planted basis, planted token means, planted TRAIN frames), built as control b of ../read/controls_read_slowwm.py builds one.
Usage: python top64_controls.py   (about 6 min; writes the planted projector to <WORK>/proj_SLOW_s0.npz for the smoke run)"""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SLOWWM_READ_WORK", "F:/TanitAD-scratch/slowwm_beside_build/b3")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import top64_lib as T  # noqa: E402
import controls_read_slowwm as C  # noqa: E402  (its planted-arm builders; imported, not edited)

S, LP, SH, SB, LR = T.S, T.LP, T.SH, T.SB, T.LR
WORK = C.WORK
OUT = os.path.join(HERE, "CONTROLS_B3.json")
NAME = "SLOW_s0"


# ---------------------------------------------------------------------------------------------------------------- planted pieces
def plant_train_cache(root, pb, n_clips=40, n_frames=130, seed=3):
    """A planted TRAIN cache in the format of ``slow_basis.py cache``: <root>/<clip>.npz (zbasis, ok) + BASIS_MANIFEST.json. u = the planted basis applied to random token means: planted frames, not real train frames."""
    os.makedirs(root, exist_ok=True)
    g = np.random.default_rng(seed)
    files = {}
    for i in range(n_clips):
        c = f"planted_train_{i:03d}"
        u = SB.apply_basis(g.standard_normal((n_frames, 1024)).astype(np.float32), pb["mean"], pb["proj"])
        ok = np.ones(n_frames, bool)
        ok[:2] = i % 2 == 0
        np.savez(os.path.join(root, f"{c}.npz"), zbasis=u, ok=ok)
        files[c] = T.sha256_file(os.path.join(root, f"{c}.npz"))
    json.dump(dict(basis_sha256=pb["sha"], kind=pb["kind"], files_sha256=files, planted=True), open(os.path.join(root, "BASIS_MANIFEST.json"), "w"))
    return root


def setup():
    os.makedirs(WORK, exist_ok=True)
    fr = LR.load_eval_frames(f"{LP.FRAMES}/frames_s0.npz")
    pb = C.planted_basis("slow", 1)
    nms = C.planted_nm(fr)
    ub = C.planted_ubasis_file("1", fr, nms, pb)
    run = C.planted_run("ok", "slow", "slow", "slow", pb["sha"])
    tc = plant_train_cache(os.path.join(WORK, "planted_train_cache"), pb)
    return dict(pb=pb, ub=ub, run=run, tc=tc)


def _same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _maxdiff(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# ---------------------------------------------------------------------------------------------------------------- (iii)
def control_iii(P):
    t0 = time.time()
    L = S.load_checked(NAME, P["run"], P["pb"]["path"], P["ub"])
    wm = L["model"].wm
    K = T.window_size(wm)
    uw, prov = T.train_windows(P["tc"], P["pb"]["sha"], K, n=256)
    w64 = T.wm64_of(wm)
    cols = np.arange(T.TOPN)
    ja, jf = T.jbar_autograd(w64, uw, cols), T.jbar_fd(w64, uw, cols)
    jca, jcf = T.jbar_autograd(w64, uw, cols, joint=False), T.jbar_fd(w64, uw, cols, joint=False)
    rel_joint, rel_cur = rel_fro(ja, jf), rel_fro(jca, jcf)
    # window form == whole-clip form of z1 (the builder differentiates the window form)
    z = np.load(os.path.join(P["tc"], "planted_train_000.npz"))["zbasis"].astype(np.float64)
    with torch.no_grad():
        full = w64.z1_from_basis(torch.as_tensor(z)[None])[0]
        win = torch.stack([w64.z1_from_basis(torch.as_tensor(z[np.clip(np.arange(j - K + 1, j + 1), 0, len(z) - 1)])[None])[0, -1] for j in range(len(z))])
    win_vs_clip = float((full - win).abs().max())
    # red arms
    ja_shift = T.jbar_autograd(w64, uw, cols + 1)
    red_shift = rel_fro(ja_shift, jf)
    red_cur = rel_fro(jca, jf) if K > 1 else None
    tol = T.FD_REL_TOL
    ok = rel_joint <= tol and rel_cur <= tol and win_vs_clip <= 1e-9 and red_shift > tol and (red_cur is None or red_cur > tol)
    return dict(pass_=bool(ok), n_windows=int(len(uw)), window_K=K, step=T.FD_STEP, tolerance_rel_frobenius=tol, rel_joint_shift=rel_joint, rel_newest_frame=rel_cur,
                max_abs_joint=_maxdiff(ja, jf), jbar_fro_norm=float(np.linalg.norm(ja)), window_form_vs_clip_form_max_abs=win_vs_clip,
                red_arms=dict(autograd_columns_shifted_by_one_rel=red_shift, red_shift_fails=bool(red_shift > tol),
                              newest_frame_form_vs_joint_fd_rel=red_cur, red_current_fails=None if red_cur is None else bool(red_cur > tol)),
                train_frames="PLANTED (planted basis applied to random token means)", wall_s=round(time.time() - t0, 1)), w64, uw


# ---------------------------------------------------------------------------------------------------------------- planner controls
def _unit(ctx, form, idx, lam, seed_mode):
    return LP.unit3(ctx, "flat1", "true", form, idx, lam, seed_mode=seed_mode, collect=False)


def _eq_unit(a, b):
    return {k: _same(a[k], b[k]) for k in ("plan_raw", "cost1", "init1", "iters1")}


def control_i_ii(P, w64, uw, proj_out):
    t0 = time.time()
    ctx = S.prepare_arm_read(NAME, P["run"], P["pb"]["path"], P["ub"])
    D = ctx.D
    idx = LP.sel_arrays(ctx, "c1")["all"][:64]
    idx3 = LP.sel_arrays(ctx, "c3")["all"][:8]
    full64 = _unit(ctx, "c1", idx, 0.0, "prior")
    full3_a = _unit(ctx, "c3", idx3, 0.3, "hold")
    # ---- (i)
    with T.restricted_cost(ctx.st, np.eye(D)):
        id64 = _unit(ctx, "c1", idx, 0.0, "prior")
        id3 = _unit(ctx, "c3", idx3, 0.3, "hold")
    after3 = _unit(ctx, "c3", idx3, 0.3, "hold")
    removed = "goal_cost" not in ctx.st.__dict__
    e64, e3, e3_after = _eq_unit(full64, id64), _eq_unit(full3_a, id3), _eq_unit(full3_a, after3)
    Pm = np.eye(D)
    Pm[0, 0] = 0.0
    with T.restricted_cost(ctx.st, Pm):
        red64 = _unit(ctx, "c1", idx, 0.0, "prior")
    red_cost_differs, red_plan_differs = not _same(full64["cost1"], red64["cost1"]), not _same(full64["plan_raw"], red64["plan_raw"])
    ci = dict(pass_=bool(all(e64.values()) and all(e3.values()) and all(e3_after.values()) and removed and red_cost_differs and red_plan_differs), windows_c1_prior=int(len(idx)), bit_equal_c1_prior_64=e64,
              windows_c3_hold_lam0p3=int(len(idx3)), bit_equal_c3_hold_8=e3, third_run_after_exit_bit_equal=e3_after, wrapper_removed_from_instance=removed,
              red_arm_one_diagonal_zeroed=dict(cost_differs=red_cost_differs, plan_differs=red_plan_differs, max_abs_plan_diff=_maxdiff(full64["plan_raw"], red64["plan_raw"])))
    # ---- (ii) bottom-64 mu columns (192-255); builder-level path on the same planted windows, then the planner
    top_cols, bot_cols = np.arange(0, 64), np.arange(192, 256)
    pr_top = T.projector_from_jbar(T.jbar_autograd(w64, uw, top_cols))
    pr_bot = T.projector_from_jbar(T.jbar_autograd(w64, uw, bot_cols))
    with T.restricted_cost(ctx.st, pr_top["P"]):
        top64 = _unit(ctx, "c1", idx, 0.0, "prior")
    with T.restricted_cost(ctx.st, pr_bot["P"]):
        bot64 = _unit(ctx, "c1", idx, 0.0, "prior")
    mut_cols = top_cols                                                        # the mutation: the "bottom" columns are the top ones
    pr_mut = T.projector_from_jbar(T.jbar_autograd(w64, uw, mut_cols))
    with T.restricted_cost(ctx.st, pr_mut["P"]):
        mut64 = _unit(ctx, "c1", idx, 0.0, "prior")
    d_full, d_top = _maxdiff(bot64["plan_raw"], full64["plan_raw"]), _maxdiff(bot64["plan_raw"], top64["plan_raw"])
    differs = (not _same(bot64["plan_raw"], full64["plan_raw"])) and (not _same(bot64["plan_raw"], top64["plan_raw"])) and d_full > 1e-6 and d_top > 1e-6
    red_equal_top = _same(mut64["plan_raw"], top64["plan_raw"])
    ang = T.principal_angles_deg(pr_top["U"], pr_bot["U"])
    cii = dict(pass_=bool(differs and red_equal_top), windows=int(len(idx)), rank_top=pr_top["rank"], rank_bottom=pr_bot["rank"], bottom_plan_vs_fullwidth_max_abs=d_full, bottom_plan_vs_top64_max_abs=d_top,
               top64_plan_vs_fullwidth_max_abs=_maxdiff(top64["plan_raw"], full64["plan_raw"]), principal_angle_top_vs_bottom_deg=dict(min=float(ang.min()), median=float(np.median(ang)), max=float(ang.max())),
               red_arm_bottom_mutated_to_top_columns=dict(plan_equals_top64_plan=red_equal_top, makes_control_fail=bool(red_equal_top)))
    # ---- the builder end to end on the planted arm (4,096 planted frames) -> the projector the smoke run reads
    res = T.build_arm(NAME, P["run"], P["pb"]["path"], P["ub"], P["tc"], proj_out)
    pj = T.load_projector(proj_out)
    idem = float(np.abs(pj["P"] @ pj["P"] - pj["P"]).max())
    sym = float(np.abs(pj["P"] - pj["P"].T).max())
    shas_ok = all(T.sha256_file(p) == pj["meta"]["inputs_sha256"][k] for k, p in (("basis_npz", P["pb"]["path"]), ("ubasis_npz", P["ub"]), ("ckpt_last", os.path.join(P["run"], "ckpt_last.pt"))))
    T.verify_projector(pj, ctx, P["run"])
    bld = dict(pass_=bool(idem < 1e-10 and sym < 1e-12 and shas_ok and pj["rank"] == res["rank"]), npz=proj_out, rank=pj["rank"], window_K=res["window_K"], n_frames=res["n_frames"], idempotence_max_abs=idem,
               symmetry_max_abs=sym, inputs_sha256_recomputed_equal=shas_ok, beside={k: v for k, v in res["beside"].items() if k != "principal_angles_deg"}, wall_s=round(res["wall_s"], 1),
               train_frames="PLANTED")
    return ci, cii, bld, round(time.time() - t0, 1)


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res.update(protocol="B3", rules="BESIDE_RULES_SLOWWM.md", rules_sha256=T.RULES_SHA, tier=T.TIER, arm="planted SLOW arm (R35H s0 checkpoint, planted basis, planted token means, PLANTED train frames)")
    res.pop("error", None)
    try:
        C._wait_ram(8.0)
        torch.set_num_threads(2)
        P = setup()
        c3, w64, uw = control_iii(P)
        res["iii_jacobian"] = c3
        ci, cii, bld, wall = control_i_ii(P, w64, uw, os.path.join(WORK, f"proj_{NAME}.npz"))
        res["i_identity"], res["ii_bottom64_red_arm"], res["builder_planted"] = ci, cii, bld
        res["wall_s_planner_controls"] = wall
    except BaseException:
        traceback.print_exc()
        res["error"] = traceback.format_exc()[-1500:]
    for v in res.values():
        if isinstance(v, dict) and "pass_" in v:
            v["pass"] = v.pop("pass_")
    keys = ("i_identity", "ii_bottom64_red_arm", "iii_jacobian")
    res["all_pass"] = all(res.get(k, {}).get("pass") is True for k in keys) and res.get("builder_planted", {}).get("pass") is True and "error" not in res
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print({k: res[k].get("pass") for k in keys + ("builder_planted",) if k in res}, "all_pass", res["all_pass"])
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
