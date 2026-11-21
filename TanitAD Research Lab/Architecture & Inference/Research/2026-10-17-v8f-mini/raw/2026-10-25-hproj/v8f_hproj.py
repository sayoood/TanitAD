#!/usr/bin/env python3
"""v8f_hproj.py -- executes PREREG_V8F_HPROJ.md exactly as written (dev box, CPU only, ZERO training, offline, from banked outputs).

  python v8f_hproj.py control --seed M2b_s0     reconstruction control only (rc 2 if a binding control fails)
  python v8f_hproj.py score   --seed M2b_s0     control FIRST (rc 2 -> no read), then HP and every PREREG row
  python v8f_hproj.py read                      both seeds -> the outcome table, verbatim -> READ.json
  python v8f_hproj.py dk                        LONGITUDINAL distance keeping for B / H / HP / B-projected / P10

THE ARM (PREREG, fixed):   tau_HP = project_speed( tau_H60 , cap )[:, 1::2]
   tau_H60 = up(P10) + w60 (.) (tau60 - up(P10)),   in the ego frame at NOW (axis 0 = along/x, axis 1 = cross/y)
   tau60   = up(P10) + Basis c   RECOVERED from B's banked 30-step plan by least squares on the basis sampled at the 30 eval steps ([:, 1::2])
   w60     = HBLEND's banked per-seed weights, linearly interpolated to the 0.1 s grid, w(0.1) = w(0.2).  NO WEIGHT IS RE-FITTED.
   cap     = v8f_mini_eval.py's own: limit_ms_from_onehot(limit, limit_ok), 1e9 where there is no limit (J["limit"][clip], J["limit_ok"][clip]).

BINDING CONTROLS (PREREG; the run stops with rc 2 and NO read if any fails):
   C0  P10's plan equals the model's internal p30 (<= 1e-4 m)             [task step 1: "confirm first"]
   C1  up(P10) + Basis c, subsampled, reproduces B's banked traj30 (max |d| <= 1e-4 m)
   C2  project_speed on B's reconstructed traj60 reproduces B's banked `projected_r1_kamm` ADE6 to 1e-3 m
Everything else in `control` (E*, X1, R*) is a DIAGNOSTIC added by the executor: it is reported, it is not a bar and it cannot change the table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True                      # HB is read-only: do not drop __pycache__ into it
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
HB, OUT = f"{SCR}/hblend", f"{SCR}/hproj"
if HB not in sys.path:
    sys.path.insert(0, HB)
import v8f_hblend as H  # noqa: E402  (read-only reuse: loaders, the eval world, arm_pw, paired rows, jdump)

SEEDS, B_BOOT, DT = H.SEEDS, H.B_BOOT, 0.2
TOL_RECON, TOL_PROJ_ADE6, TOL_P10 = 1e-4, 1e-3, 1e-4        # PREREG / task step 1
BIG = 1e9
CK = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any")
RULES = ("SPEED", "STOP", "TURN", "LEAD")


# ------------------------------------------------------------------------------------------------ pure arithmetic (unit-tested)
def basis_np(nb: int, n60: int = 60, ramp: bool = True) -> np.ndarray:
    """The residual basis of `V8FMini.__init__`: cos(k pi (i+0.5)/60) x the ramp (i+1)/60, i = 0..59, k = 0..nb-1  -> [60, nb]."""
    i = np.arange(n60, dtype=np.float64)
    b = np.stack([np.cos(k * np.pi * (i + 0.5) / n60) for k in range(nb)], 1)
    return b * ((i + 1) / n60)[:, None] if ramp else b


def up_np(p30: np.ndarray) -> np.ndarray:
    """`upsample_prior` in numpy: [N, 30, 2] on the 0.2 s grid -> [N, 60, 2] on the 0.1 s grid (odd indices are the knots, even the midpoints)."""
    p30 = np.asarray(p30, np.float64)
    n = len(p30)
    knots = np.concatenate([np.zeros((n, 1, 2)), p30], 1)
    mid = 0.5 * (knots[:, :-1] + knots[:, 1:])
    return np.stack([mid, knots[:, 1:]], 2).reshape(n, 60, 2)


def recover_c(res30: np.ndarray, basis: np.ndarray):
    """Least squares for the nb coefficients per axis from the 30 residual rows (basis sampled at the eval steps, [1::2]). -> c [N, 2, nb], diag."""
    bk = basis[1::2]
    n = len(res30)
    y = np.asarray(res30, np.float64).transpose(1, 0, 2).reshape(30, n * 2)          # [30, N*2]
    c, _, rank, sv = np.linalg.lstsq(bk, y, rcond=None)                              # [nb, N*2]
    fit = bk @ c
    diag = {"rank": int(rank), "nb": int(bk.shape[1]), "sv_max": float(sv.max()), "sv_min": float(sv.min()), "cond": float(sv.max() / sv.min()),
            "lstsq_residual_max_abs_m": float(np.abs(fit - y).max())}
    return c.reshape(basis.shape[1], n, 2).transpose(1, 2, 0), diag                  # [N, 2, nb]


def recon_traj60(tau30: np.ndarray, p30: np.ndarray, basis: np.ndarray):
    """tau60 = up(P) + Basis c, with c recovered from tau30 - P. -> (tau60 [N, 60, 2], c, diag)."""
    c, diag = recover_c(np.asarray(tau30, np.float64) - np.asarray(p30, np.float64), basis)
    res60 = np.einsum("tk,nck->ntc", basis, c)
    return up_np(p30) + res60, c, diag


def w60_from_w30(w30: np.ndarray) -> np.ndarray:
    """HBLEND's 30-knot weights [30, 2] (knot j at t = 0.2 (j+1)) -> the 0.1 s grid [60, 2] by linear interpolation, w(0.1) = w(0.2)
    (np.interp clamps to the first knot on the left). Independent of `v8f_hblend.w_midpoints`."""
    t30, t60 = 0.2 * np.arange(1, 31), 0.1 * np.arange(1, 61)
    return np.stack([np.interp(t60, t30, np.asarray(w30, np.float64)[:, a]) for a in range(2)], 1)


def blend60(p60: np.ndarray, tau60: np.ndarray, w60: np.ndarray) -> np.ndarray:
    return p60 + w60[None] * (tau60 - p60)


def cap_from_J(J: dict, eid: np.ndarray):
    """cap per window exactly as `v8f_mini_eval.py` builds it: the clip's one-hot limit and its validity flag through `limit_ms_from_onehot`."""
    import torch
    from tanitad.models import v8f_mini as M
    limit = torch.as_tensor(np.asarray(J["limit"])[eid], dtype=torch.float32)
    ok = torch.as_tensor(np.asarray(J["limit_ok"])[eid], dtype=torch.bool)
    lim = M.limit_ms_from_onehot(limit, ok)
    cap = torch.where(torch.isfinite(lim), lim, torch.full_like(lim, BIG))
    return cap.numpy(), lim.numpy()


def project_f32(traj60: np.ndarray, cap: np.ndarray, dtype=None) -> np.ndarray:
    """`project_speed` on [N, 60, 2] in the eval's own precision (float32 torch CPU). -> float64 [N, 60, 2]."""
    import torch
    from tanitad.models import v8f_mini as M
    dt = torch.float32 if dtype is None else dtype
    with torch.no_grad():
        out = M.project_speed(torch.as_tensor(np.asarray(traj60), dtype=dt), None if cap is None else torch.as_tensor(np.asarray(cap), dtype=dt))
    return out.numpy().astype(np.float64)


def ade_pw(pred: np.ndarray, gt: np.ndarray, k: int = 30) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred, np.float64)[:, :k] - gt[:, :k], axis=-1).mean(1)


# ------------------------------------------------------------------------------------------------ context
def load_ctx(seed: str) -> dict:
    import torch
    t0 = time.time()
    grid, J, X = H.eval_world()
    z = np.load(f"{H.BANK}/eval_{seed}-kind-own.npz")
    bj = json.load(open(f"{H.BANK}/eval_{seed}-kind-own.json", encoding="utf-8"))
    model, cfg, ck = H.load_model(seed)
    assert H.sha256_file(H.CKPT[seed]) == H.SHA[seed], "checkpoint sha256 mismatch"
    with torch.no_grad():
        P32 = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy()
    W = json.load(open(f"{H.OUT}/weights_{seed}.json"))
    w30 = np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)
    return {"seed": seed, "grid": grid, "J": J, "X": X, "z": z, "bj": bj, "model": model, "cfg": cfg, "P32": P32, "P": P32.astype(np.float64),
            "P_file": np.load(H.P10_FILE).astype(np.float64), "W": W, "w30": w30, "gt": X["gt"].astype(np.float64), "eid": X["eid"],
            "tau": z["pred"].astype(np.float64), "t_load": time.time() - t0}


# ------------------------------------------------------------------------------------------------ the reconstruction control
def x1_crosscheck(ctx: dict, tau60_rec: np.ndarray, cap_J: np.ndarray, Bp_rec: np.ndarray) -> dict:
    """DIAGNOSTIC (not in the PREREG): on the eval windows whose image features are on this box, run the model's OWN CPU forward and compare its true
    traj60 (all 60 steps, including the 0.1 s midpoints that no banked file carries) and the eval loader's own cap with the reconstruction."""
    import torch
    torch.set_num_threads(4)
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    J, model, cfg, z = ctx["J"], ctx["model"], ctx["cfg"], ctx["z"]
    tracks = D.load_native_tracks(f"{H.WORK}/inputs/native_tracks.pt")
    wins = D.eval_grid_windows(f"{H.WORK}/inputs/eval_grid_windows.json", J)
    n, d60, dknot, dmid, dcap, dproj, dproj_pos = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    d_inter_ade = []
    for cache in (f"{H.WORK}/cache", f"{H.WORK}/cache_viz"):
        have = {p.stem for p in Path(cache, "eval", "clips").glob("*.pt")}
        if not have:
            continue
        store = D.ClipStore(cache, "eval", lru=8, grid="pool4x10", kin_mode="yaw", tracks=tracks)
        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
        idx = [i for i, w in enumerate(wins) if str(J["clip"][w[0]]) in have]
        for b0 in range(0, len(idx), 24):
            ii = idx[b0:b0 + 24]
            items = []
            for i in ii:
                ci, t, _, _ = wins[i]
                a_t = i % 3
                items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
            b = D.collate(items)
            with torch.no_grad():
                o = model.forward_infer(b)
                lim = M.limit_ms_from_onehot(b["limit"], b["limit_ok"])
                cap_b = torch.where(torch.isfinite(lim), lim, torch.full_like(lim, BIG))
                proj_true = M.project_speed(o["traj60"].float(), cap_b)
            tr60 = o["traj60"].numpy().astype(np.float64)
            d60 = max(d60, float(np.abs(tr60 - tau60_rec[ii]).max()))
            dknot = max(dknot, float(np.abs(tr60[:, 1::2] - z["pred"][ii].astype(np.float64)).max()))
            dmid = max(dmid, float(np.abs(tr60[:, 0::2] - tau60_rec[ii][:, 0::2]).max()))
            dcap = max(dcap, float(np.abs(cap_b.numpy().astype(np.float64) - cap_J[ii]).max()))
            pt = proj_true.numpy().astype(np.float64)
            pr = project_f32(tau60_rec[ii], cap_J[ii])
            dproj = max(dproj, float(np.abs(pt - pr).max()))
            dproj_pos = max(dproj_pos, float(np.abs(pt[:, 1::2] - z["traj30_proj"][ii].astype(np.float64)).max()))
            n += len(ii)
    return {"n_windows_checked": n, "label": "DIAGNOSTIC: CPU forward_infer traj60 vs the reconstruction",
            "max_abs_traj60_cpu_vs_recon_all60_m": d60, "max_abs_cpu_traj60_knots_vs_banked_pred_m": dknot,
            "max_abs_traj60_cpu_vs_recon_midpoints_only_m": dmid, "max_abs_cap_eval_loader_vs_J_m_s": dcap,
            "max_abs_projected60_cpu_vs_projected60_recon_m": dproj, "max_abs_cpu_projected_knots_vs_banked_traj30_proj_m": dproj_pos}


def run_control(ctx: dict, nb: int, do_x1: bool = True) -> dict:
    ev = H.load_eval_module()
    X, J, z, bj, cfg, model = ctx["X"], ctx["J"], ctx["z"], ctx["bj"], ctx["cfg"], ctx["model"]
    gt, eid, tau, P, P_file = ctx["gt"], ctx["eid"], ctx["tau"], ctx["P"], ctx["P_file"]
    ctl = {"seed": ctx["seed"], "binding": {}, "diagnostics": {}}
    bd, dg = ctl["binding"], ctl["diagnostics"]
    # ---- C0: P10's plan == the model's internal p30
    bd["C0_P10_file_vs_model_prior_maxabs_m"] = float(np.abs(P_file - P).max())
    bd["C0_pass"] = bool(bd["C0_P10_file_vs_model_prior_maxabs_m"] <= TOL_P10)
    # ---- the basis: built from the formula, checked against the model's own buffer
    basis = basis_np(cfg.nb)
    dg["E1_nb_from_ckpt_cfg"] = int(cfg.nb)
    dg["E1_res_scale_m_from_ckpt_cfg"] = float(cfg.res_scale_m)
    dg["E1_basis_formula_vs_model_buffer_maxabs"] = float(np.abs(basis - model.basis.detach().numpy().astype(np.float64)).max())
    # ---- C1: reconstruct
    tau60, c, ls = recon_traj60(tau, P, basis)
    bd["C1_recon_knots_vs_banked_traj30_maxabs_m"] = float(np.abs(tau60[:, 1::2] - tau).max())
    bd["C1_pass"] = bool(bd["C1_recon_knots_vs_banked_traj30_maxabs_m"] <= TOL_RECON)
    dg["E1_lstsq"] = ls
    dg["E1_c_abs_max"] = float(np.abs(c).max())
    # ---- the cap
    cap, lim = cap_from_J(J, eid)
    lim64 = np.asarray(X["limit_ms"], np.float64)
    fin = np.isfinite(lim)
    dg["E2_cap_nan_pattern_equals_eval_context"] = bool(np.array_equal(fin, np.isfinite(lim64)))
    dg["E2_cap_vs_eval_context_limit_ms_maxabs"] = float(np.abs(lim[fin].astype(np.float64) - lim64[fin]).max()) if fin.any() else 0.0
    dg["E2_n_windows_with_posted_limit"] = int(fin.sum())
    dg["E2_n_windows"] = int(len(cap))
    dg["E2_cap_values_ms_unique"] = sorted({round(float(v), 4) for v in np.unique(cap)})
    # ---- C2: the projection of B's reconstruction reproduces the banked `projected_r1_kamm`
    B60p = project_f32(tau60, cap)
    Bp_rec = B60p[:, 1::2]
    ade6_rec = float(ade_pw(Bp_rec, gt).mean())
    ade6_json = float(bj["projected_r1_kamm"]["traj"]["ade6_m"]["mean"])
    bd["C2_banked_projected_r1_kamm_ade6_json"] = ade6_json
    bd["C2_recon_projected_ade6"] = ade6_rec
    bd["C2_absdiff_m"] = abs(ade6_rec - ade6_json)
    bd["C2_pass"] = bool(bd["C2_absdiff_m"] <= TOL_PROJ_ADE6)
    bd["all_pass"] = bool(bd["C0_pass"] and bd["C1_pass"] and bd["C2_pass"])
    # ---- E3: stricter, per-window reading against the banked projected plan (the eval saved it)
    Bp_banked = z["traj30_proj"].astype(np.float64)
    dg["E3_projected_plan_recon_vs_banked_traj30_proj_maxabs_m"] = float(np.abs(Bp_rec - Bp_banked).max())
    dg["E3_projected_plan_recon_vs_banked_traj30_proj_mean_abs_m"] = float(np.abs(Bp_rec - Bp_banked).mean())
    dg["E3_banked_traj30_proj_ade6_exact_mean"] = float(ade_pw(Bp_banked, gt).mean())
    dg["E3_recon_vs_banked_traj30_proj_ade6_exact_absdiff"] = abs(ade6_rec - dg["E3_banked_traj30_proj_ade6_exact_mean"])
    dg["E3_perwindow_ade6_maxabs_diff"] = float(np.abs(ade_pw(Bp_rec, gt) - ade_pw(Bp_banked, gt)).max())
    dg["E3_n_windows_where_projection_moved_B"] = int((np.abs(Bp_banked - tau).max((1, 2)) > 1e-6).sum())
    rP, _ = ev.evaluate(Bp_rec, X, None, nb, with_families=False)
    bdiff = H.deep_diff(H.jround(rP), bj["projected_r1_kamm"], tol=1e-6)
    dg["E3_projected_block_recon_vs_banked_json"] = {"n_mismatches": len(bdiff), "mismatches": bdiff[:12]}
    # ---- E4: the weights on the 0.1 s grid vs HBLEND's own derived grid
    w60 = w60_from_w30(ctx["w30"])
    dg["E4_w60_vs_hblend_w60_derived_maxabs"] = float(np.abs(w60 - np.asarray(ctx["W"]["w60_derived_unscored"], np.float64)).max())
    dg["E4_w60_at_0.1s_equals_w_at_0.2s"] = bool(np.array_equal(w60[0], w60[1]))
    dg["E4_w60_knots_equal_w30"] = bool(np.abs(w60[1::2] - ctx["w30"]).max() <= 1e-15)
    # ---- E5: H identities
    up = up_np(P)
    H60 = blend60(up, tau60, w60)
    saved = np.load(f"{H.OUT}/score_{ctx['seed']}_perwindow.npz")
    dg["E5_H60_knots_vs_hblend_saved_H_maxabs_m"] = float(np.abs(H60[:, 1::2] - saved["H"]).max())
    dg["E5_w1_gives_tau60_maxabs"] = float(np.abs(blend60(up, tau60, np.ones((60, 2))) - tau60).max())
    dg["E5_w0_gives_up_P10_maxabs"] = float(np.abs(blend60(up, tau60, np.zeros((60, 2))) - up).max())
    dg["E5_up_np_vs_torch_upsample_prior_maxabs"] = _up_check(ctx["P32"])
    # ---- R*: deliberate-regression arms -- each must make its control FAIL, or the control cannot fail
    b_nor = basis_np(cfg.nb, ramp=False)
    t_nor, _, _ = recon_traj60(tau, P, b_nor)
    dg["R1_no_ramp_basis_knots_maxabs_m"] = float(np.abs(t_nor[:, 1::2] - tau).max())
    dg["R1_must_exceed_C1_bar_and_does"] = bool(dg["R1_no_ramp_basis_knots_maxabs_m"] > TOL_RECON)
    Bp_nocap = project_f32(tau60, None)[:, 1::2]
    dg["R2_no_cap_projected_ade6"] = float(ade_pw(Bp_nocap, gt).mean())
    dg["R2_no_cap_absdiff_vs_banked_json"] = abs(dg["R2_no_cap_projected_ade6"] - ade6_json)
    dg["R2_must_exceed_C2_bar_and_does"] = bool(dg["R2_no_cap_absdiff_vs_banked_json"] > TOL_PROJ_ADE6)
    Bp_noproj = tau
    dg["R3_no_projection_ade6"] = float(ade_pw(Bp_noproj, gt).mean())
    dg["R3_no_projection_absdiff_vs_banked_json"] = abs(dg["R3_no_projection_ade6"] - ade6_json)
    dg["R3_must_exceed_C2_bar_and_does"] = bool(dg["R3_no_projection_absdiff_vs_banked_json"] > TOL_PROJ_ADE6)
    dg["precision_float64_projection_vs_float32_maxabs_m"] = float(np.abs(project_f32(tau60, cap, dtype=__import__("torch").float64)[:, 1::2] - Bp_rec).max())
    if do_x1:
        ctl["X1"] = x1_crosscheck(ctx, tau60, cap, Bp_rec)
    ctl["_arrays"] = {"tau60": tau60, "cap": cap, "lim": lim, "w60": w60, "H60": H60, "up": up, "Bp_rec": Bp_rec}
    return ctl


def _up_check(P32: np.ndarray) -> float:
    import torch
    from tanitad.models import v8f_mini as M
    return float(np.abs(M.upsample_prior(torch.as_tensor(P32)).numpy().astype(np.float64) - up_np(P32)).max())


# ------------------------------------------------------------------------------------------------ the score
def trans(a: np.ndarray, b: np.ndarray) -> dict:
    """windows (a) violated / (b) violated cross-tab: how many windows each arm fixes or newly breaks relative to the other."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"a_viol_b_ok": int((a & ~b).sum()), "a_ok_b_viol": int((~a & b).sum()), "both_viol": int((a & b).sum()), "both_ok": int((~a & ~b).sum())}


def speed_anatomy(X, viol_by_arm: dict, D) -> dict:
    """DIAGNOSTIC: which cap makes the SPEED rule active, and where each arm's SPEED violations sit. The SPEED rule's cap is min(posted limit, v_hi_ms);
    the hard projection knows the posted limit only."""
    lim = np.asarray(X["limit_ms"], np.float64)
    vhi = np.asarray(X["cons"][:, D.CONS_IDX["v_hi_ms"]], np.float64)
    vhi = np.where(np.asarray(X["has"], bool), vhi, np.nan)
    cap = np.fmin(lim, vhi)
    act = np.isfinite(cap)
    kinds = {"active_total": act,
             "active_posted_limit_only (v_hi absent)": act & np.isfinite(lim) & ~np.isfinite(vhi),
             "active_v_hi_only (no posted limit)": act & ~np.isfinite(lim) & np.isfinite(vhi),
             "active_both_present": act & np.isfinite(lim) & np.isfinite(vhi),
             "active_both_present_and_v_hi_binding (v_hi < limit)": act & np.isfinite(lim) & np.isfinite(vhi) & (vhi < lim)}
    out = {"n_windows": int(len(act)), "windows_by_kind": {k: int(m.sum()) for k, m in kinds.items()}, "violations_by_arm": {}}
    for arm, v in viol_by_arm.items():
        v = np.asarray(v, bool)
        out["violations_by_arm"][arm] = {"total": int(v.sum()), **{k: int((v & m).sum()) for k, m in kinds.items() if k != "active_total"}}
    return out


def cmd_control(a):
    ctx = load_ctx(a.seed)
    ctl = run_control(ctx, a.n_boot, do_x1=not a.no_x1)
    arrays = ctl.pop("_arrays")
    H.jdump(ctl, f"{OUT}/control_{a.seed}.json")
    b = ctl["binding"]
    print(f"[control {a.seed}] C0 {b['C0_P10_file_vs_model_prior_maxabs_m']:.2e} ({b['C0_pass']})  C1 {b['C1_recon_knots_vs_banked_traj30_maxabs_m']:.2e} ({b['C1_pass']})  "
          f"C2 |{b['C2_recon_projected_ade6']:.4f} - {b['C2_banked_projected_r1_kamm_ade6_json']:.4f}| = {b['C2_absdiff_m']:.2e} ({b['C2_pass']})  ALL {b['all_pass']}", flush=True)
    return 0 if b["all_pass"] else 2


def cmd_score(a):
    t0 = time.time()
    ev = H.load_eval_module()
    from taniteval import ci as CI
    from taniteval import v8f_beta_eval as BE
    from tanitad.data import v8f_mini_data as D
    seed, nb = a.seed, a.n_boot
    ctx = load_ctx(seed)
    # ============ the control FIRST: no HP number exists before this passes
    ctl = run_control(ctx, nb, do_x1=not a.no_x1)
    arr = ctl.pop("_arrays")
    H.jdump(ctl, f"{OUT}/control_{seed}.json")
    b = ctl["binding"]
    print(f"[score {seed}] CONTROL C0 {b['C0_P10_file_vs_model_prior_maxabs_m']:.2e} C1 {b['C1_recon_knots_vs_banked_traj30_maxabs_m']:.2e} "
          f"C2 absdiff {b['C2_absdiff_m']:.2e}  all_pass={b['all_pass']}", flush=True)
    if not b["all_pass"]:
        print(f"[score {seed}] RECONSTRUCTION CONTROL FAILED -- rc 2, NO READ: {json.dumps(b)}", flush=True)
        return 2
    X, J, z, bj, gt, eid, tau = ctx["X"], ctx["J"], ctx["z"], ctx["bj"], ctx["gt"], ctx["eid"], ctx["tau"]
    # ============ the arm
    cap, tau60, w60, H60, up = arr["cap"], arr["tau60"], arr["w60"], arr["H60"], arr["up"]
    HP60 = project_f32(H60, cap)
    HP = HP60[:, 1::2]
    Hs = np.load(f"{H.OUT}/score_{seed}_perwindow.npz")["H"].astype(np.float64)          # HBLEND's own H (reference arm; == H60[:, 1::2] to the E5 tolerance)
    Bp = z["traj30_proj"].astype(np.float64)                                              # B's own projected plan (the eval saved it)
    P10 = ctx["P_file"]
    P10p = project_f32(up_np(P10), cap)[:, 1::2]                                          # DESCRIPTIVE extra: P10 under the same projection
    fl = np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")
    arms = {"B": tau, "H": Hs, "HP": HP, "Bp": Bp, "P10": P10, "P10p": P10p}
    PW = {k: H.arm_pw(v, gt, BE) for k, v in arms.items()}
    out = {"seed": seed, "ckpt_sha256": H.SHA[seed], "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
           "tier": "T1-class open loop (own held outputs; not closed loop)",
           "estimator": f"episode-cluster bootstrap, B={nb}, 141 episodes, seed 0; deltas PAIRED (same resampled episodes). Answers 'another draw of EPISODES' ONLY; "
                        "training-seed variance only via the two seeds side by side and F = |ADE6(s0)-ADE6(s1)|; inference is deterministic (no inference-seed floor).",
           "controls": ctl, "arms_defined": {"B": "B's banked UNPROJECTED plan `pred` (the headline B)", "H": "HBLEND's H (saved array)",
                                              "HP": "project_speed(tau_H60, cap)[:, 1::2]  <- THE ARM",
                                              "Bp": "B's own banked projected plan `traj30_proj` (the eval's --project output)",
                                              "P10": "P10-yaw-4512 plan file (unprojected)", "P10p": "DESCRIPTIVE: P10 under the same projection"},
           "weights_sha256": H.sha256_file(f"{H.OUT}/weights_{seed}.json"),
           "inputs_sha256": {"banked_npz": H.sha256_file(f"{H.BANK}/eval_{seed}-kind-own.npz"), "banked_json": H.sha256_file(f"{H.BANK}/eval_{seed}-kind-own.json"),
                             "p10_4512_pred": H.sha256_file(H.P10_FILE), "ckpt": H.SHA[seed]}}
    keys = list(PW["B"])
    out["single"] = {n: {k: H.single(PW[n][k], eid, CI, nb) for k in keys} for n in ("B", "H", "HP", "Bp", "P10", "P10p")}
    out["paired"] = {"HP_minus_B": H.rows_for(PW["HP"], PW["B"], eid, BE, nb), "HP_minus_H": H.rows_for(PW["HP"], PW["H"], eid, BE, nb),
                     "HP_minus_Bp": H.rows_for(PW["HP"], PW["Bp"], eid, BE, nb), "HP_minus_P10": H.rows_for(PW["HP"], PW["P10"], eid, BE, nb),
                     "H_minus_B": H.rows_for(PW["H"], PW["B"], eid, BE, nb), "Bp_minus_B": H.rows_for(PW["Bp"], PW["B"], eid, BE, nb),
                     "HP_minus_P10p": {k: H.prow(PW["HP"][k], PW["P10p"][k], eid, BE, nb, k) for k in ("ade2", "ade6", "fw_lon_accel_mae")}}
    # ---- the eval's OWN evaluate(): families, consistency (a), self-consistency (b), tactical
    ex = H.extras_of(z, D)
    resHP, pwHP = ev.evaluate(HP, X, ex, nb, with_families=True)
    resH, pwH = ev.evaluate(Hs, X, ex, nb, with_families=True)
    resBp, pwBp = ev.evaluate(Bp, X, ex, nb, with_families=True)
    resP10p, pwP10p = ev.evaluate(P10p, X, None, nb, with_families=False)
    out["blocks"] = {"HP": resHP, "Bp": resBp, "P10p": resP10p}
    hb = json.load(open(f"{H.OUT}/score_{seed}.json"))
    bdiff = H.deep_diff(H.jround(resH), hb["blocks"]["H"], tol=1e-6)
    out["H_block_vs_hblend_saved"] = {"n_mismatches": len(bdiff), "mismatches": bdiff[:12],
                                      "note": "H re-evaluated here through the eval's own evaluate() == the block HBLEND banked (the reference arm is HBLEND's H)"}
    # ---- CONSISTENCY: per constraint, (a) and (b), every pair the PREREG names (+ the transitions)
    viol = {"B": {k: z[k].astype(bool) for k in CK}, "H": {k: pwH[k].astype(bool) for k in CK}, "HP": {k: pwHP[k].astype(bool) for k in CK},
            "Bp": {k: pwBp[k].astype(bool) for k in CK}, "P10": {k: fl[f"P10_refit__{k}"].astype(bool) for k in CK[:5]},
            "P10p": {k: pwP10p[k].astype(bool) for k in CK[:5]}}
    out["consistency"] = {
        "label": "CONSISTENCY(a) violation rate of the PLAN vs the GT-derived constraints (lower is better); (b) = violation of the model's OWN predicted constraints",
        "rate": {n: {k: float(v[k].mean()) for k in v} for n, v in viol.items()},
        "counts_a": {"B": bj["raw"]["consistency_a"], "H": resH["consistency_a"], "HP": resHP["consistency_a"], "Bp": resBp["consistency_a"]},
        "paired": {f"{p}_minus_{q}": {k: BE.paired_row(viol[p][k].astype(float), viol[q][k].astype(float), eid, nb) for k in viol[p] if k in viol[q]}
                   for p, q in (("HP", "B"), ("HP", "H"), ("HP", "Bp"), ("HP", "P10"), ("H", "B"), ("Bp", "B"), ("HP", "P10p"))},
        "transitions": {f"{p}_vs_{q}": {k: trans(viol[p][k], viol[q][k]) for k in viol[p] if k in viol[q]} for p, q in (("HP", "B"), ("HP", "H"), ("H", "B"), ("HP", "Bp"))},
        "speed_anatomy": speed_anatomy(X, {n: viol[n]["viol_SPEED"] for n in ("B", "H", "HP", "Bp")}, D)}
    # ---- the plan itself: how much did the projection move HP vs H, and where
    moved = np.abs(HP - Hs).max((1, 2))
    out["projection_effect"] = {"n_windows_HP_differs_from_H_by_gt_1e-3_m": int((moved > 1e-3).sum()), "max_abs_HP_minus_H_m": float(moved.max()),
                                "mean_abs_HP_minus_H_m": float(np.abs(HP - Hs).mean()),
                                "n_windows_with_posted_limit": int(np.isfinite(arr["lim"]).sum()),
                                "ADE6_mean": {n: float(ade_pw(arms[n], gt).mean()) for n in arms}}
    # ---- structural identity: T's heads and S's checkpoints are untouched => the tactical block (T's heads) and strategic block are B's
    out["structural_identity"] = {"tactical_block_T_heads_HP_equals_banked_B": H.jround(resHP["tactical"]) == H.jround(bj["raw"]["tactical"]),
                                  "strategic": "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19); S is untouched, banked ck_pos_err carried unchanged; NOT a pass",
                                  "banked_B_strategic_block": bj["raw"]["strategic"]}
    out["seconds"] = time.time() - t0
    H.jdump(out, f"{OUT}/score_{seed}.json")
    np.savez_compressed(f"{OUT}/score_{seed}_perwindow.npz", HP=HP, HP60=HP60, H60=H60, tau60=tau60, cap=cap, w60=w60, P10p=P10p, eid=eid,
                        **{f"{n}__{k}": v for n in PW for k, v in PW[n].items()}, **{f"viol_{n}__{k}": v[k] for n, v in viol.items() for k in v})
    pk, pc = out["paired"]["HP_minus_B"], out["consistency"]["paired"]["HP_minus_B"]["viol_any"]
    print(f"[score {seed}] DONE {time.time() - t0:.0f}s  ADE2 HP-B {pk['ade2']['mean']:+.4f} [{pk['ade2']['lo']:+.4f},{pk['ade2']['hi']:+.4f}]  "
          f"ADE6 HP-B {pk['ade6']['mean']:+.4f} [{pk['ade6']['lo']:+.4f},{pk['ade6']['hi']:+.4f}]  CONS(any) HP-B {pc['mean']:+.4f} [{pc['lo']:+.4f},{pc['hi']:+.4f}]", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ read: the outcome table, verbatim
def cmd_read(a):
    S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if not S[s]["controls"]["binding"]["all_pass"]:
            raise SystemExit(f"{s}: the reconstruction control did not pass -- no read")
    rows = {}
    for s in SEEDS:
        P, C = S[s]["paired"]["HP_minus_B"], S[s]["consistency"]["paired"]["HP_minus_B"]["viol_any"]
        a2, a6 = P["ade2"], P["ade6"]
        dom_pt = {"ade2": S[s]["single"]["HP"]["ade2"]["mean"] <= S[s]["single"]["P10"]["ade2"]["mean"],
                  "ade6": S[s]["single"]["HP"]["ade6"]["mean"] <= S[s]["single"]["P10"]["ade6"]["mean"],
                  "accel": S[s]["single"]["HP"]["fw_lon_accel_mae"]["mean"] <= S[s]["single"]["P10"]["fw_lon_accel_mae"]["mean"]}
        dP = S[s]["paired"]["HP_minus_P10"]
        rows[s] = {"PRIMARY_C": {"delta": C["mean"], "lo": C["lo"], "hi": C["hi"], "bar_hi_le": 0.02, "pass": bool(C["hi"] <= 0.02)},
                   "PRIMARY_K": {"ade6": {"delta": a6["mean"], "lo": a6["lo"], "hi": a6["hi"], "separated": a6["separated"], "pass": bool(a6["mean"] < 0 and a6["separated"])},
                                 "ade2": {"delta": a2["mean"], "lo": a2["lo"], "hi": a2["hi"], "separated": a2["separated"], "pass": bool(a2["mean"] < 0 and a2["separated"])}},
                   "DOMINANCE_vs_P10": {"point_estimates_le_P10": dom_pt, "ade2": {k: dP["ade2"][k] for k in ("mean", "lo", "hi", "separated")},
                                        "ade6": {k: dP["ade6"][k] for k in ("mean", "lo", "hi", "separated")},
                                        "accel": {k: dP["fw_lon_accel_mae"][k] for k in ("mean", "lo", "hi", "separated")}}}
        rows[s]["PRIMARY_K"]["pass"] = bool(rows[s]["PRIMARY_K"]["ade6"]["pass"] and rows[s]["PRIMARY_K"]["ade2"]["pass"])
    c_ok = all(rows[s]["PRIMARY_C"]["pass"] for s in SEEDS)
    k_ok = all(rows[s]["PRIMARY_K"]["pass"] for s in SEEDS)
    if c_ok and k_ok:
        outcome, nxt = "PASS", ("HP is the deployed v8F output. Its weights become a model buffer, and the projection becomes part of `forward_infer` in the next trained arm. "
                                "MIXO and FLAT are re-scored with their own fitted HP (descriptive).")
    elif not c_ok:
        outcome, nxt = "CONSISTENCY NOT RESTORED", ("H's violations are not of the posted limit: they are stops, leads or turns, which the cap does not see. Next: a constraint-aware "
                                                    "fit under its own PREREG. It is the same closed form, fitted only on training windows with no active GT constraint, plus O's "
                                                    "predicted stop / lead constraints in the projection.")
    else:
        outcome, nxt = "GAIN LOST", ("the projection and the shrinkage compete. Next: the constraint-aware fit as above (the same closed form, fitted only on training windows "
                                     "with no active GT constraint, plus O's predicted stop / lead constraints in the projection).")
    R = {"outcome": outcome, "committed_next_step_verbatim": nxt, "PRIMARY_C_both_seeds": c_ok, "PRIMARY_K_both_seeds": k_ok,
         "note_both_fail": ("PRIMARY-C and PRIMARY-K both fail: the table row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) matches first; the 'GAIN LOST' row requires "
                            "PRIMARY-C to PASS, so it does not apply. The K result is reported as measured." if (not c_ok and not k_ok) else None),
         "per_seed": rows, "estimator": S["M2b_s0"]["estimator"], "tier": S["M2b_s0"]["tier"]}
    H.jdump(R, f"{OUT}/READ.json")
    print(json.dumps({k: R[k] for k in ("outcome", "PRIMARY_C_both_seeds", "PRIMARY_K_both_seeds")}, indent=1))
    for s in SEEDS:
        print(s, json.dumps({k: rows[s][k] for k in ("PRIMARY_C", "PRIMARY_K")}, default=float))
    return 0


def cmd_dk(a):
    """LONGITUDINAL distance keeping through the programme's OWN reader `v8f_dk_read.read_dk` (-> `taniteval.lead_metrics`), K = 30 block:
    B, H, HP, B-projected (both seeds) and P10; paired HP-B, HP-H, HP-Bp, HP-P10, H-B."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_dk_read", f"{H.OUT}/deps/v8f_dk_read.py")
    dk = importlib.util.module_from_spec(sp)
    sys.modules["v8f_dk_read"] = dk
    sp.loader.exec_module(dk)
    grid, J, X = H.eval_world()
    order = list(grid["order"])
    preds, saved = {}, {}
    for s in SEEDS:
        z = np.load(f"{H.BANK}/eval_{s}-kind-own.npz")
        pw = np.load(f"{OUT}/score_{s}_perwindow.npz")
        k = s.split("_")[1]
        preds[f"B_{k}"], preds[f"H_{k}"], preds[f"HP_{k}"], preds[f"Bp_{k}"] = z["pred"].astype(np.float64), np.load(f"{H.OUT}/score_{s}_perwindow.npz")["H"], pw["HP"], z["traj30_proj"].astype(np.float64)
        saved[f"B_{k}"] = z["ade6"]
    preds["P10"] = np.load(H.P10_FILE).astype(np.float64)
    fl = np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{H.OUT}/inputs/b1_eval_lead_block_k30.npz")
    pairs = []
    for k in ("s0", "s1"):
        pairs += [(f"HP_{k}", f"B_{k}"), (f"HP_{k}", f"H_{k}"), (f"HP_{k}", f"Bp_{k}"), (f"HP_{k}", "P10"), (f"H_{k}", f"B_{k}")]
    res = dk.read_dk(preds=preds, gt=X["gt"].astype(np.float64), eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                     v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=DT, n_boot=a.n_boot, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
    H.jdump(dk._clean(res["read"]), f"{OUT}/dk_read.json")
    np.savez_compressed(f"{OUT}/dk_perwindow.npz", **res["perwindow"])
    r = res["read"]
    print("states", r["join"]["window_states"], "contract", r["human_reference"]["kinematic_contract"]["status"])
    for nm, ab in r["arms"].items():
        m = ab["metrics"]
        print(f"  {nm:6s} headway {m['headway_min_m'].get('mean')} time-gap {m['time_gap_min_s'].get('mean')} TTC {m['min_ttc_s'].get('mean')} LEAD-VIOL at-risk {ab['lead_viol'].get('share_of_at_risk')}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("control", "score"):
        p = sub.add_parser(name)
        p.add_argument("--seed", required=True, choices=SEEDS)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
        p.add_argument("--no-x1", action="store_true", help="skip the 85-window CPU cross-check (diagnostic)")
    for name in ("read", "dk"):
        p = sub.add_parser(name)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
    a = ap.parse_args(argv)
    return {"control": cmd_control, "score": cmd_score, "read": cmd_read, "dk": cmd_dk}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
