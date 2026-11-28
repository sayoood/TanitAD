#!/usr/bin/env python3
"""v8f_hgate.py -- executes PREREG_V8F_HGATE.md exactly as written (dev box, CPU only, ZERO training, offline, from banked outputs).

  python v8f_hgate.py activity                    step 1a: GT STOP / TURN / LEAD (and SPEED) activity of the 9,582 HBLEND fit windows, through the eval's OWN code
  python v8f_hgate.py fit   --seed M2b_s0         step 1b: w^G = HBLEND's closed form + clip + isotonic on the windows with NO active GT STOP / TURN / LEAD
  python v8f_hgate.py score --seed M2b_s0         step 2 + 3: identity controls FIRST (rc 2 -> no read), then HG and every PREREG row
  python v8f_hgate.py read                        both seeds -> the outcome table, verbatim -> READ.json
  python v8f_hgate.py dk                          LONGITUDINAL distance keeping for B / H / HG / P10

THE ARM (PREREG, fixed):
   tau_G  = up(P10) + w^G (.) (tau - up(P10))                         [scored on the 0.2 s knots: up(P10)[1::2] = P10, i.e. HBLEND's H with w^G]
   plan   = tau_B   if ANY of T1..T4 holds, else tau_G
   T1  T's longitudinal argmax is FOLLOW                                   (LON_FOLLOW = 3)
   T2  T's goal probability for STOP >= 0.5                                (STOP_I = 6)
   T3  T's goal probability for TURN_L or TURN_R >= 0.5                    (indices 1, 2)
   T4  speed of tau_G exceeds T's own predicted v_hi_ms at any step with t >= 2.0 s, the SPEED rule's band

DECLARED INTERPRETATIONS (each a choice the PREREG text leaves open; fixed here BEFORE any HG number exists, none can change after):
   I1  'open' / 'closed' follow the PREREG's identity controls: gate CLOSED = the plan is tau_B (a trigger fired, or forced); gate OPEN = the plan is tau_G.
       'the gate's open share' = the share of windows on which tau_G is used.
   I2  'active GT constraint' on a TRAIN window = the eval's own `violations(...)[rule]['active']` for STOP, TURN, LEAD, evaluated by the eval's own
       `gt_constraints` + `v8f_consistency.violations` on the window's join row. SPEED is excluded (PREREG: active on every window).
       Those masks do not depend on the plan (control: identical for three different plans).
   I3  T4 reads the SPEED rule's own speeds: `v8f_consistency.plan_kinematics(plan)['v'][:, BAND_KNOT:]` (BAND_KNOT = 10: the intervals that START at
       tau >= 2.0), compared with T's raw predicted v_hi_ms (cons_pred cell 25, no clamp, no posted-limit min, no 0.5 m/s tolerance) -- the PREREG says
       'exceeds T's own predicted v_hi_ms'. A predicted v_hi <= 0 therefore closes the gate (20 / 3 windows at s0 / s1).
   I4  T2 / T3 use '>= 0.5' (the PREREG's operator), the eval's own self-consistency uses '> 0.5'.
   I5  A train window with no label row (`has` False) has no active rule, exactly as an eval window without labels.
Everything else (w^G closed form, clip [0, 1], unweighted PAV, B = 2000, the eval's own `evaluate()`) is HBLEND's, imported, not re-implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
HB, OUT = f"{SCR}/hblend", f"{SCR}/hgate"
if HB not in sys.path:
    sys.path.insert(0, HB)
import v8f_hblend as H  # noqa: E402  (read-only reuse: loaders, fit_weights, apply_w, arm_pw, paired rows, jdump)

SEEDS, B_BOOT, DT = H.SEEDS, H.B_BOOT, 0.2
# the PREREG's literals; checked against the code tree's constants in `check_constants`
LON_FOLLOW, STOP_I, TURN_L_I, TURN_R_I = 3, 6, 1, 2
P_THR = 0.5
BAND_S = 2.0
FIT_ID_TOL = 1e-12                      # PREREG: the fit on all windows reproduces HBLEND's banked weights to 1e-12
GATE_RULES = ("STOP", "TURN", "LEAD")   # SPEED is active on every window (PREREG), so it cannot define the fit set
CK = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any")
TRIG = ("T1_follow", "T2_stop", "T3_turn", "T4_vhi")


def check_constants(M, C) -> int:
    assert (M.LON_FOLLOW, M.STOP_I, M.TURN_L_I, M.TURN_R_I) == (LON_FOLLOW, STOP_I, TURN_L_I, TURN_R_I), "PREREG literals differ from the code tree"
    band_knot = int(round(BAND_S / DT))
    assert band_knot == C._BAND_KNOT == M.BAND_KNOT, (band_knot, C._BAND_KNOT, getattr(M, "BAND_KNOT", None))
    return band_knot


# ------------------------------------------------------------------------------------------------ pure arithmetic (unit-tested)
def triggers(goal_prob, lon_logits, vhi_pred, v_plan, band_knot: int = 10, thr: float = P_THR) -> dict:
    """The four PREREG triggers, each a bool [N]. ``v_plan`` [N, 30] = the speeds of tau_G on the 0.2 s knots (interval j-1 ends at knot j)."""
    gp = np.asarray(goal_prob, np.float64)
    t1 = np.asarray(lon_logits).argmax(-1) == LON_FOLLOW
    t2 = gp[:, STOP_I] >= thr
    t3 = (gp[:, TURN_L_I] >= thr) | (gp[:, TURN_R_I] >= thr)
    t4 = (np.asarray(v_plan, np.float64)[:, band_knot:] > np.asarray(vhi_pred, np.float64)[:, None]).any(1)
    return {"T1_follow": t1, "T2_stop": t2, "T3_turn": t3, "T4_vhi": t4}


def gate_plan(P, tau, w, goal_prob, lon_logits, vhi_pred, kin_fn, band_knot: int = 10, force=None, use=TRIG):
    """-> (plan [N, 30, 2], info). ``force`` in {None, 'closed', 'open'}; ``use`` = the triggers that may close the gate (all four = the PREREG arm)."""
    P, tau = np.asarray(P, np.float64), np.asarray(tau, np.float64)
    blend = H.apply_w(P, tau, w)                                  # tau_G (the shrunk plan), scored on the 0.2 s knots
    trig = triggers(goal_prob, lon_logits, vhi_pred, kin_fn(blend)["v"], band_knot)
    closed_trig = np.zeros(len(tau), bool)
    for k in use:
        closed_trig |= trig[k]
    if force == "closed":
        closed = np.ones(len(tau), bool)
    elif force == "open":
        closed = np.zeros(len(tau), bool)
    elif force is None:
        closed = closed_trig
    else:
        raise ValueError(force)
    plan = np.where(closed[:, None, None], tau, blend)
    return plan, {"closed": closed, "blend": blend, "triggers": trig, "closed_by_triggers": closed_trig}


# ------------------------------------------------------------------------------------------------ GT constraint activity, through the eval's own code
def window_X(J: dict, ci: np.ndarray, t: np.ndarray, s_h: np.ndarray, D) -> dict:
    """The per-window arrays `ev.gt_constraints(X)` reads, rebuilt from a label join ``J`` and window coordinates (clip index, NOW step).
    It mirrors `v8f_mini_data.eval_context` row by row (jrow = off + t; lead_gap_h = lead_gap[off + min(t + 1 + [0..29], T - 1)])."""
    ci, t = np.asarray(ci, np.int64), np.asarray(t, np.int64)
    off, n = np.asarray(J["off"])[ci], np.asarray(J["n"])[ci]
    r = off + t
    idx = np.minimum(t[:, None] + 1 + np.arange(D.HORIZON)[None], n[:, None] - 1)
    return {"goal_y": J["goal_y"][r], "goal_w": J["goal_w"][r], "cons": J["cons"][r], "lead_valid": J["lead_valid"][r], "has": J["has"][r],
            "lead_gap_h": J["lead_gap"][off[:, None] + idx], "s_h": np.asarray(s_h),
            "limit_ms": np.where(J["limit_ok"][ci], J["limit"][ci] @ (np.array(D.LIMITS_KMH) / 3.6), np.nan), "jrow": r}


def gt_activity(Xw: dict, plan: np.ndarray, ev) -> dict:
    """{rule: active bool [N]} for SPEED / STOP / TURN / LEAD from the eval's own `gt_constraints` and `violations`."""
    gtc = ev.gt_constraints(Xw)
    V = ev.C.violations(np.asarray(plan, np.float64), gtc)
    return {r: np.asarray(V[r]["active"], bool) for r in ev.C.RULE_NAMES}


def nan_equal(a, b) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def s_h_from_cache(ci: np.ndarray, t: np.ndarray, J: dict, cache_dir: str, D) -> np.ndarray:
    """s_h[t, i] for train windows from each clip's cached `pose_grid` via the data module's own `human_arc` (mmap: only the pose grid is read)."""
    import torch
    out = np.full((len(ci), D.HORIZON), np.nan, np.float32)
    for c in np.unique(ci):
        m = ci == c
        o = torch.load(f"{cache_dir}/{str(J['clip'][c])}.pt", map_location="cpu", weights_only=True, mmap=True)
        pg = o["pose_grid"].numpy().astype(np.float64)
        assert len(pg) == int(J["n"][c]), f"clip {J['clip'][c]}: pose_grid {len(pg)} != join n {int(J['n'][c])}"
        out[m] = D.human_arc(pg)[np.asarray(t)[m]]
    return out


def s_h_from_gt(gt: np.ndarray) -> np.ndarray:
    """Independent re-derivation of s_h from the ego-frame GT waypoints (origin + 30 knots): cumulative path length. Used only as a control."""
    g = np.asarray(gt, np.float64)
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(g), 1, 2)), g], 1), axis=1), axis=-1)
    return np.cumsum(seg, 1)


# ------------------------------------------------------------------------------------------------ step 1a: the activity of the fit windows
def cmd_activity(a):
    t0 = time.time()
    ev = H.load_eval_module()
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    band_knot = check_constants(M, ev.C)
    J = D.load_join(f"{HB}/box2/join_train.npz")
    fw = np.load(f"{HB}/fit_windows_M2b_s0.npz")
    ci, t, gt = fw["ci"], fw["t"], fw["gt"].astype(np.float64)
    N = len(ci)
    s_h = s_h_from_cache(ci, t, J, f"{HB}/cache/train/clips", D)
    Xw = window_X(J, ci, t, s_h, D)
    ctl = {}
    # ---- control A: s_h from the cache's pose_grid == s_h re-derived from the GT waypoints (float32 vs float64 only)
    ctl["A_s_h_cache_vs_gt_derived_maxabs_m"] = float(np.nanmax(np.abs(s_h.astype(np.float64) - s_h_from_gt(gt))))
    ctl["A_s_h_all_finite"] = bool(np.isfinite(s_h).all())
    # ---- the activity, on THREE different plans (it must not depend on the plan): human future, B's plan, P10
    acts = {nm: gt_activity(Xw, plan, ev) for nm, plan in (("gt", gt), ("tau_s0", fw["traj30"]), ("p10", fw["p30"]))}
    ctl["B_activity_identical_across_plans"] = bool(all(np.array_equal(acts["gt"][r], acts[nm][r]) for nm in ("tau_s0", "p10") for r in acts["gt"]))
    act = acts["gt"]
    # ---- control C: the SAME builder applied to the EVAL windows reproduces the eval's own activity masks, field by field
    grid, Je, X = H.eval_world()
    Xe = window_X(Je, X["eid"], X["t"], X["s_h"], D)
    fields = {"goal_y": np.array_equal(Xe["goal_y"], X["goal_y"]), "goal_w": np.array_equal(Xe["goal_w"], X["goal_w"]),
              "cons": nan_equal(Xe["cons"], X["cons"]), "lead_valid": np.array_equal(Xe["lead_valid"], X["lead_valid"]),
              "lead_gap_h": nan_equal(Xe["lead_gap_h"], X["lead_gap_h"]), "limit_ms": nan_equal(Xe["limit_ms"], X["limit_ms"]),
              "has": np.array_equal(Xe["has"], X["has"]), "jrow": np.array_equal(Xe["jrow"], X["jrow"])}
    ctl["C_eval_windows_builder_fields_equal"] = {k: bool(v) for k, v in fields.items()}
    z0 = np.load(f"{HB}/inputs/banked/eval_M2b_s0-kind-own.npz")
    Vown = ev.C.violations(z0["pred"].astype(np.float64), ev.gt_constraints(X))
    ev_act_own = {r: np.asarray(Vown[r]["active"], bool) for r in ev.C.RULE_NAMES}
    ev_act_mine = gt_activity(Xe, z0["pred"].astype(np.float64), ev)
    ctl["C_eval_activity_equal_to_evals_own"] = {r: bool(np.array_equal(ev_act_own[r], ev_act_mine[r])) for r in ev.C.RULE_NAMES}
    ctl["C_eval_active_counts"] = {r: int(ev_act_own[r].sum()) for r in ev.C.RULE_NAMES}
    # ---- control D (mutation): breaking the lead-gap indexing must change the LEAD mask, or control C cannot fail
    Xbad = dict(Xw)
    Xbad["lead_gap_h"] = np.full_like(Xw["lead_gap_h"], np.nan)
    ctl["D_mutation_no_lead_gap_changes_LEAD_mask"] = bool(not np.array_equal(gt_activity(Xbad, gt, ev)["LEAD"], act["LEAD"]))
    ok = bool(ctl["A_s_h_all_finite"] and ctl["A_s_h_cache_vs_gt_derived_maxabs_m"] < 1e-3 and ctl["B_activity_identical_across_plans"]
              and all(fields.values()) and all(ctl["C_eval_activity_equal_to_evals_own"].values()) and ctl["D_mutation_no_lead_gap_changes_LEAD_mask"])
    ctl["all_pass"] = ok
    ctrl_any = act["STOP"] | act["TURN"] | act["LEAD"]
    has = Xw["has"]
    rep = {"source": "box-2 label join `join_train.npz` (the trainer's own --join-train, already copied to the dev box: hblend/box2/join_train.npz); "
                     "NO box-2 access was needed. Activity computed by the eval's own gt_constraints + v8f_consistency.violations (hblend/deps/taniteval).",
           "n_windows": int(N), "n_clips": int(len(np.unique(ci))), "k": 32,
           "join_sha256": H.sha256_file(f"{HB}/box2/join_train.npz"), "fit_windows_sha256": H.sha256_file(f"{HB}/fit_windows_M2b_s0.npz"),
           "n_windows_without_label_row": int((~has).sum()),
           "active_counts": {r: int(act[r].sum()) for r in ("SPEED", "STOP", "TURN", "LEAD")},
           "active_share": {r: float(act[r].mean()) for r in ("SPEED", "STOP", "TURN", "LEAD")},
           "n_any_STOP_TURN_LEAD_active": int(ctrl_any.sum()), "n_inactive_all_three": int((~ctrl_any).sum()),
           "share_inactive_all_three": float((~ctrl_any).mean()),
           "pairwise_overlap": {"STOP&TURN": int((act["STOP"] & act["TURN"]).sum()), "STOP&LEAD": int((act["STOP"] & act["LEAD"]).sum()),
                                "TURN&LEAD": int((act["TURN"] & act["LEAD"]).sum())},
           "same_window_set_for_both_seeds": bool(np.array_equal(fw["ci"], np.load(f"{HB}/fit_windows_M2b_s1.npz")["ci"])
                                                  and np.array_equal(fw["t"], np.load(f"{HB}/fit_windows_M2b_s1.npz")["t"])),
           "eval_141_active_counts_for_reference": ctl["C_eval_active_counts"], "controls": ctl, "seconds": time.time() - t0}
    H.jdump(rep, f"{OUT}/train_activity.json")
    np.savez_compressed(f"{OUT}/train_activity.npz", ci=ci, t=t, has=has, **{f"active_{r}": v for r, v in act.items()}, inactive_gate_rules=~ctrl_any)
    print(json.dumps({k: rep[k] for k in ("n_windows", "n_windows_without_label_row", "active_counts", "n_any_STOP_TURN_LEAD_active", "n_inactive_all_three")}, indent=1))
    print("controls all_pass", ok, json.dumps({k: v for k, v in ctl.items() if k != "all_pass"}, default=float)[:900])
    return 0 if ok else 2


# ------------------------------------------------------------------------------------------------ step 1b: the constraint-aware fit
def load_banked_w(seed: str) -> np.ndarray:
    W = json.load(open(f"{HB}/weights_{seed}.json"))
    return np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)


def load_wG(seed: str) -> np.ndarray:
    W = json.load(open(f"{OUT}/weights_G_{seed}.json"))
    return np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)


def fit_identity_all(seed: str) -> dict:
    """PREREG identity (iii): the fit restricted to ALL windows reproduces HBLEND's banked weights to 1e-12."""
    fw = np.load(f"{HB}/fit_windows_{seed}.npz")
    R = H.fit_weights(fw["traj30"], fw["p30"], fw["gt"])
    d = float(np.abs(R["w"] - load_banked_w(seed)).max())
    return {"maxabs_vs_banked_weights": d, "bar": FIT_ID_TOL, "pass": bool(d <= FIT_ID_TOL), "n_windows": int(len(fw["gt"]))}


def cmd_fit(a):
    t0 = time.time()
    seed = a.seed
    A = np.load(f"{OUT}/train_activity.npz")
    fw = np.load(f"{HB}/fit_windows_{seed}.npz")
    assert np.array_equal(A["ci"], fw["ci"]) and np.array_equal(A["t"], fw["t"]), "activity rows are not aligned with this seed's fit windows"
    inact = A["inactive_gate_rules"].astype(bool)
    n = int(inact.sum())
    R = H.fit_weights(fw["traj30"][inact], fw["p30"][inact], fw["gt"][inact])
    Ra = H.fit_weights(fw["traj30"], fw["p30"], fw["gt"])
    wb = load_banked_w(seed)
    ident = fit_identity_all(seed)
    # descriptive: is w^G different from w by more than the sampling noise of an equal-size random subset of the SAME windows?
    rng = np.random.default_rng(0)
    dev_rand = []
    for _ in range(50):
        sel = np.sort(rng.choice(len(fw["gt"]), size=n, replace=False))
        Rr = H.fit_weights(fw["traj30"][sel], fw["p30"][sel], fw["gt"][sel])
        dev_rand.append([float(np.abs(Rr["w"] - wb).mean()), float(np.abs(Rr["w"] - wb).max())])
    dev_rand = np.asarray(dev_rand)
    d = lambda p, g: np.linalg.norm(np.asarray(p, np.float64) - np.asarray(g, np.float64), axis=-1)
    HG_in = H.apply_w(fw["p30"][inact], fw["traj30"][inact], R["w"])
    rec = {"seed": seed, "ckpt_sha256": H.SHA[seed], "n_windows_fit": n, "n_windows_all": int(len(fw["gt"])), "share_of_all": n / len(fw["gt"]),
           "n_clips_in_fit": int(len(np.unique(fw["ci"][inact]))),
           "fit_rule": "HBLEND closed form per step and axis, clip [0, 1], unweighted PAV, on the windows with NO active GT STOP / TURN / LEAD",
           "w_along_by_step": R["w"][:, 0], "w_cross_by_step": R["w"][:, 1], "raw_along": R["raw"][:, 0], "raw_cross": R["raw"][:, 1],
           "clipped_along": R["clipped"][:, 0], "clipped_cross": R["clipped"][:, 1], "sum_r2_along": R["den"][:, 0], "sum_r2_cross": R["den"][:, 1],
           "n_clipped_at_0": int((R["raw"] < 0).sum()), "n_clipped_at_1": int((R["raw"] > 1).sum()),
           "n_changed_by_isotonic": int((np.abs(R["w"] - R["clipped"]) > 1e-12).sum()),
           "w_in_range_and_monotone": bool(((R["w"] >= 0) & (R["w"] <= 1)).all() and (np.diff(R["w"], axis=0) >= -1e-12).all()),
           "identity_fit_all_windows_vs_banked": ident,
           "w_G_minus_w_banked": {"mean_abs": float(np.abs(R["w"] - wb).mean()), "max_abs": float(np.abs(R["w"] - wb).max()),
                                  "mean_signed_along": float((R["w"][:, 0] - wb[:, 0]).mean()), "mean_signed_cross": float((R["w"][:, 1] - wb[:, 1]).mean())},
           "random_subset_same_n_vs_banked_w (50 draws, DESCRIPTIVE)": {"mean_abs_dev_mean": float(dev_rand[:, 0].mean()), "mean_abs_dev_max_over_draws": float(dev_rand[:, 0].max()),
                                                                          "max_abs_dev_mean": float(dev_rand[:, 1].mean()), "max_abs_dev_max_over_draws": float(dev_rand[:, 1].max())},
           "in_sample_inactive_windows_ade6": {"B_tau": float(d(fw["traj30"][inact], fw["gt"][inact]).mean()), "P": float(d(fw["p30"][inact], fw["gt"][inact]).mean()),
                                               "HG_blend": float(d(HG_in, fw["gt"][inact]).mean())},
           "seconds": time.time() - t0}
    H.jdump(rec, f"{OUT}/weights_G_{seed}.json")
    print(f"[fit {seed}] n={n} of {len(fw['gt'])}  identity(all windows vs banked) maxabs {ident['maxabs_vs_banked_weights']:.2e} pass={ident['pass']}  "
          f"|wG - w| mean {rec['w_G_minus_w_banked']['mean_abs']:.4f} max {rec['w_G_minus_w_banked']['max_abs']:.4f}  "
          f"(random equal-n subset: mean {dev_rand[:, 0].mean():.4f} max-over-draws {dev_rand[:, 0].max():.4f})", flush=True)
    return 0 if ident["pass"] else 2


# ------------------------------------------------------------------------------------------------ step 2 + 3: controls, then the arm
def trans(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"a_viol_b_ok": int((a & ~b).sum()), "a_ok_b_viol": int((~a & b).sum()), "both_viol": int((a & b).sum()), "both_ok": int((~a & ~b).sum())}


def crosstab(fired: np.ndarray, truth: np.ndarray) -> dict:
    f, g = np.asarray(fired, bool), np.asarray(truth, bool)
    tp, fp, fn, tn = int((f & g).sum()), int((f & ~g).sum()), int((~f & g).sum()), int((~f & ~g).sum())
    return {"fired_and_active": tp, "fired_not_active": fp, "not_fired_active": fn, "not_fired_not_active": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "n_active": tp + fn, "n_fired": tp + fp}


def load_world(seed: str):
    """Everything `score` and `diag` read for one seed. P is recomputed from the checkpoint exactly as HBLEND did."""
    import torch
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    ev = H.load_eval_module()
    band_knot = check_constants(M, ev.C)
    grid, J, X = H.eval_world()
    z = np.load(f"{H.BANK}/eval_{seed}-kind-own.npz")
    bj = json.load(open(f"{H.BANK}/eval_{seed}-kind-own.json", encoding="utf-8"))
    model, cfg, ck = H.load_model(seed)
    assert H.sha256_file(H.CKPT[seed]) == H.SHA[seed], "checkpoint sha256 mismatch"
    with torch.no_grad():
        P32 = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy()
    ex = H.extras_of(z, D)
    return {"seed": seed, "ev": ev, "D": D, "grid": grid, "J": J, "X": X, "z": z, "bj": bj, "P": P32.astype(np.float64), "P_file": np.load(H.P10_FILE).astype(np.float64),
            "tau": z["pred"].astype(np.float64), "gt": X["gt"].astype(np.float64), "eid": X["eid"], "ex": ex, "band_knot": band_knot,
            "vhi": ex["cons"]["v_hi_ms"].astype(np.float64), "gp": z["goal_prob"], "lon": z["lon_logits"],
            "kin_fn": lambda plan: ev.C.plan_kinematics(plan), "wb": load_banked_w(seed), "wG": load_wG(seed),
            "fl": np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")}


def gate_of(W: dict, w, force=None, use=TRIG):
    return gate_plan(W["P"], W["tau"], w, W["gp"], W["lon"], W["vhi"], W["kin_fn"], W["band_knot"], force=force, use=use)


def run_controls(W: dict, nb: int) -> dict:
    ev, D, X, z, bj, tau, gt = W["ev"], W["D"], W["X"], W["z"], W["bj"], W["tau"], W["gt"]
    seed = W["seed"]
    ctl = {"seed": seed}
    saved = np.load(f"{H.OUT}/score_{seed}_perwindow.npz")
    # ---- (i) gate forced CLOSED everywhere = B exactly
    plan_c, _ = gate_of(W, W["wG"], force="closed")
    ctl["i_closed_vs_B_maxabs_m"] = float(np.abs(plan_c - tau).max())
    ctl["i_closed_ade6_perwindow_maxabs_vs_banked"] = float(np.abs(np.linalg.norm(plan_c - gt, axis=-1).mean(1) - z["ade6"]).max())
    res_c, _ = ev.evaluate(plan_c, X, W["ex"], nb, with_families=True)
    bd = H.deep_diff(H.jround(res_c), {k: bj["raw"][k] for k in res_c if k in bj["raw"]}, tol=1e-6)
    ctl["i_closed_full_block_vs_banked_json"] = {"n_mismatches": len(bd), "mismatches": bd[:12], "compared_keys": sorted(set(res_c) & set(bj["raw"]))}
    # ---- (ii) gate forced OPEN with HBLEND's banked w = H exactly
    plan_o, _ = gate_of(W, W["wb"], force="open")
    ctl["ii_open_banked_w_vs_H_saved_maxabs_m"] = float(np.abs(plan_o - saved["H"]).max())
    ctl["ii_P_recomputed_vs_hblend_saved_P_maxabs_m"] = float(np.abs(W["P"] - saved["P"]).max())
    res_o, _ = ev.evaluate(plan_o, X, W["ex"], nb, with_families=True)
    hb = json.load(open(f"{H.OUT}/score_{seed}.json"))
    bd2 = H.deep_diff(H.jround(res_o), hb["blocks"]["H"], tol=1e-6)
    ctl["ii_open_full_block_vs_hblend_saved_H_block"] = {"n_mismatches": len(bd2), "mismatches": bd2[:12]}
    # ---- (iii) the fit on ALL windows reproduces the banked weights to 1e-12
    ctl["iii_fit_all_windows"] = fit_identity_all(seed)
    # ---- the controls must be able to FAIL (deliberate-regression arms; each must read non-zero)
    plan_wg_open, _ = gate_of(W, W["wG"], force="open")
    ctl["R1_open_with_wG_instead_of_banked_vs_H_maxabs_m"] = float(np.abs(plan_wg_open - saved["H"]).max())
    ctl["R1_must_be_gt_0_and_is"] = bool(ctl["R1_open_with_wG_instead_of_banked_vs_H_maxabs_m"] > 1e-6)
    plan_bad, _ = gate_of(W, W["wG"], force=None)
    ctl["R2_real_gate_vs_B_n_windows_differ"] = int((np.abs(plan_bad - tau).max((1, 2)) > 0).sum())
    ctl["R2_must_be_gt_0_and_is"] = bool(ctl["R2_real_gate_vs_B_n_windows_differ"] > 0)
    # ---- inherited HBLEND route controls
    ctl["banked_pred_eq_traj30"] = bool(np.array_equal(z["pred"], z["traj30"]))
    ctl["model_prior_vs_P10_4512_file_maxabs_m"] = float(np.abs(W["P"] - W["P_file"]).max())
    ctl["wG_in_range_and_monotone"] = bool(((W["wG"] >= 0) & (W["wG"] <= 1)).all() and (np.diff(W["wG"], axis=0) >= -1e-12).all())
    ctl["prereg_literals_equal_source_constants"] = True            # check_constants asserted above
    ctl["all_pass"] = bool(ctl["i_closed_vs_B_maxabs_m"] == 0.0 and ctl["i_closed_full_block_vs_banked_json"]["n_mismatches"] == 0
                           and ctl["ii_open_banked_w_vs_H_saved_maxabs_m"] == 0.0 and ctl["ii_open_full_block_vs_hblend_saved_H_block"]["n_mismatches"] == 0
                           and ctl["iii_fit_all_windows"]["pass"] and ctl["R1_must_be_gt_0_and_is"] and ctl["R2_must_be_gt_0_and_is"]
                           and ctl["banked_pred_eq_traj30"] and ctl["model_prior_vs_P10_4512_file_maxabs_m"] <= 1e-4 and ctl["wG_in_range_and_monotone"])
    return ctl


def cmd_score(a):
    t0 = time.time()
    from taniteval import ci as CI
    from taniteval import v8f_beta_eval as BE
    seed, nb = a.seed, a.n_boot
    W = load_world(seed)
    ev, D, X, z, bj, tau, gt, eid, ex, fl = (W[k] for k in ("ev", "D", "X", "z", "bj", "tau", "gt", "eid", "ex", "fl"))
    # ============ the identity controls FIRST: no HG number exists before they pass
    ctl = run_controls(W, nb)
    print(f"[score {seed}] CONTROLS i {ctl['i_closed_vs_B_maxabs_m']:.1e}/{ctl['i_closed_full_block_vs_banked_json']['n_mismatches']}  "
          f"ii {ctl['ii_open_banked_w_vs_H_saved_maxabs_m']:.1e}/{ctl['ii_open_full_block_vs_hblend_saved_H_block']['n_mismatches']}  "
          f"iii {ctl['iii_fit_all_windows']['maxabs_vs_banked_weights']:.1e}  all_pass={ctl['all_pass']}", flush=True)
    if not ctl["all_pass"]:
        H.jdump({"seed": seed, "identity_controls": ctl}, f"{OUT}/score_{seed}.json")
        print(f"[score {seed}] IDENTITY CONTROL FAILED -- rc 2, NO READ: {json.dumps(ctl, default=float)[:1500]}", flush=True)
        return 2
    # ============ the arm
    HG, info = gate_of(W, W["wG"])
    Hs = np.load(f"{H.OUT}/score_{seed}_perwindow.npz")["H"].astype(np.float64)      # HBLEND's H (reference arm)
    P10 = W["P_file"]
    arms = {"B": tau, "H": Hs, "HG": HG, "P10": P10}
    PW = {k: H.arm_pw(v, gt, BE) for k, v in arms.items()}
    closed, trig = info["closed"], info["triggers"]
    out = {"seed": seed, "ckpt_sha256": H.SHA[seed], "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
           "tier": "T1-class open loop (own held outputs; not closed loop)",
           "estimator": f"episode-cluster bootstrap, B={nb}, 141 episodes, seed 0; deltas PAIRED (same resampled episodes). Answers 'another draw of EPISODES' ONLY; "
                        "training-seed variance only via the two seeds side by side and F = |ADE6(s0)-ADE6(s1)|; inference is deterministic (no inference-seed floor).",
           "identity_controls": ctl, "weights_G_sha256": H.sha256_file(f"{OUT}/weights_G_{seed}.json"), "weights_banked_sha256": H.sha256_file(f"{HB}/weights_{seed}.json"),
           "inputs_sha256": {"banked_npz": H.sha256_file(f"{H.BANK}/eval_{seed}-kind-own.npz"), "banked_json": H.sha256_file(f"{H.BANK}/eval_{seed}-kind-own.json"),
                             "p10_4512_pred": H.sha256_file(H.P10_FILE), "ckpt": H.SHA[seed], "train_activity": H.sha256_file(f"{OUT}/train_activity.json")},
           "arms_defined": {"B": "B's banked plan `pred`", "H": "HBLEND's H (saved array)", "HG": "THE ARM: tau_B where a trigger fires, else up(P10) + w^G (.) (tau - up(P10))",
                            "P10": "P10-yaw-4512 plan file"}}
    # ---- the gate, beside
    def share(v):
        return H.single(np.asarray(v, np.float64), eid, CI, nb)
    out["gate"] = {"interpretation": "OPEN = tau_G used; CLOSED = tau_B used (a trigger fired). PREREG identity controls fix the wording.",
                   "open_share": share(~closed), "closed_share": share(closed),
                   "fires_share": {k: share(v) for k, v in trig.items()},
                   "fires_n": {k: int(v.sum()) for k, v in trig.items()},
                   "closed_by_T1T2T3_share": share(trig["T1_follow"] | trig["T2_stop"] | trig["T3_turn"]),
                   "exclusive_n": {k: int((trig[k] & ~np.any([trig[j] for j in TRIG if j != k], 0)).sum()) for k in TRIG},
                   "n_open": int((~closed).sum()), "n_closed": int(closed.sum()), "n_windows_plan_differs_from_B": int((np.abs(HG - tau).max((1, 2)) > 0).sum()),
                   "predicted_v_hi_ms": {"n_nonpositive": int((W["vhi"] <= 0).sum()), "median": float(np.median(W["vhi"])), "min": float(W["vhi"].min())}}
    # ---- GT-activity cross-tabs (descriptive): do T's triggers fire where the human is constrained?
    Vg = ev.C.violations(tau, ev.gt_constraints(X))
    ga = {r: np.asarray(Vg[r]["active"], bool) for r in ev.C.RULE_NAMES}
    out["gate"]["crosstab_vs_GT_activity"] = {
        "T1_follow_vs_GT_LEAD_active": crosstab(trig["T1_follow"], ga["LEAD"]), "T2_stop_vs_GT_STOP_active": crosstab(trig["T2_stop"], ga["STOP"]),
        "T3_turn_vs_GT_TURN_active": crosstab(trig["T3_turn"], ga["TURN"]),
        "T1T2T3_any_vs_GT_any_of_STOP_TURN_LEAD_active": crosstab(trig["T1_follow"] | trig["T2_stop"] | trig["T3_turn"], ga["STOP"] | ga["TURN"] | ga["LEAD"]),
        "gate_closed_vs_GT_any_of_STOP_TURN_LEAD_active": crosstab(closed, ga["STOP"] | ga["TURN"] | ga["LEAD"]),
        "GT_active_counts_eval141": {r: int(ga[r].sum()) for r in ev.C.RULE_NAMES}}
    # ---- the rows
    keys = list(PW["B"])
    out["single"] = {n: {k: H.single(PW[n][k], eid, CI, nb) for k in keys} for n in PW}
    out["paired"] = {"HG_minus_B": H.rows_for(PW["HG"], PW["B"], eid, BE, nb), "HG_minus_H": H.rows_for(PW["HG"], PW["H"], eid, BE, nb),
                     "HG_minus_P10": H.rows_for(PW["HG"], PW["P10"], eid, BE, nb), "H_minus_B": H.rows_for(PW["H"], PW["B"], eid, BE, nb)}
    # ---- the eval's OWN evaluate(): families, consistency (a), self-consistency (b), tactical
    resHG, pwHG = ev.evaluate(HG, X, ex, nb, with_families=True)
    resH, pwH = ev.evaluate(Hs, X, ex, nb, with_families=False)
    out["blocks"] = {"HG": resHG}
    viol = {"B": {k: z[k].astype(bool) for k in CK}, "H": {k: pwH[k].astype(bool) for k in CK}, "HG": {k: pwHG[k].astype(bool) for k in CK},
            "P10": {k: fl[f"P10_refit__{k}"].astype(bool) for k in CK[:5]}}
    out["consistency"] = {
        "label": "CONSISTENCY(a) violation rate of the PLAN vs the GT-derived constraints (lower is better); (b) = violation of the model's OWN predicted constraints",
        "rate": {n: {k: float(v[k].mean()) for k in v} for n, v in viol.items()},
        "counts_a": {"B": bj["raw"]["consistency_a"], "HG": resHG["consistency_a"]},
        "paired": {f"{p}_minus_{q}": {k: BE.paired_row(viol[p][k].astype(float), viol[q][k].astype(float), eid, nb) for k in viol[p] if k in viol[q]}
                   for p, q in (("HG", "B"), ("HG", "H"), ("HG", "P10"), ("H", "B"))},
        "transitions": {f"{p}_vs_{q}": {k: trans(viol[p][k], viol[q][k]) for k in viol[p] if k in viol[q]} for p, q in (("HG", "B"), ("HG", "H"), ("H", "B"))},
        "by_gate_state": {state: {k: float(viol["HG"][k][m].mean()) if m.any() else None for k in viol["HG"]}
                          for state, m in (("open", ~closed), ("closed", closed))},
        "B_by_gate_state": {state: {k: float(viol["B"][k][m].mean()) if m.any() else None for k in viol["B"]} for state, m in (("open", ~closed), ("closed", closed))}}
    # H's SPEED violations vs T4 (descriptive): T4 is the gate's only SPEED handler
    out["gate"]["crosstab_vs_GT_activity"]["T4_vhi_vs_H_SPEED_violated"] = crosstab(trig["T4_vhi"], viol["H"]["viol_SPEED"])
    out["gate"]["crosstab_vs_GT_activity"]["T4_vhi_vs_B_SPEED_violated"] = crosstab(trig["T4_vhi"], viol["B"]["viol_SPEED"])
    out["structural_identity"] = {"tactical_block_T_heads_HG_equals_banked_B": H.jround(resHG["tactical"]) == H.jround(bj["raw"]["tactical"]),
                                  "strategic": "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19); S is untouched, banked ck_pos_err carried unchanged; NOT a pass",
                                  "banked_B_strategic_block": bj["raw"]["strategic"]}
    out["seconds"] = time.time() - t0
    H.jdump(out, f"{OUT}/score_{seed}.json")
    np.savez_compressed(f"{OUT}/score_{seed}_perwindow.npz", HG=HG, blend=info["blend"], closed=closed, eid=eid, **{f"trig_{k}": v for k, v in trig.items()},
                        **{f"{n}__{k}": v for n in PW for k, v in PW[n].items()}, **{f"viol_{n}__{k}": v[k] for n, v in viol.items() for k in v},
                        **{f"gt_active_{r}": v for r, v in ga.items()})
    pk, pc = out["paired"]["HG_minus_B"], out["consistency"]["paired"]["HG_minus_B"]["viol_any"]
    print(f"[score {seed}] DONE {time.time() - t0:.0f}s  open share {out['gate']['open_share']['mean']:.4f}  ADE2 HG-B {pk['ade2']['mean']:+.4f} [{pk['ade2']['lo']:+.4f},{pk['ade2']['hi']:+.4f}]  "
          f"ADE6 HG-B {pk['ade6']['mean']:+.4f} [{pk['ade6']['lo']:+.4f},{pk['ade6']['hi']:+.4f}]  CONS(any) HG-B {pc['mean']:+.4f} [{pc['lo']:+.4f},{pc['hi']:+.4f}]", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ read: the outcome table, verbatim
NEXT = {"PASS": ("HG is the deployed v8F output. The weights and the gate become part of `forward_infer` in the next trained arm, and MIXO and FLAT are re-scored with "
                 "their own fitted HG (descriptive)"),
        "CONSISTENCY NOT RESTORED": ("T's own constraint predictions are not good enough to tell where the human constrains: the T-side ceiling again (`D-V8F-T-ATTRIB-1`). "
                                     "Next: no further post-processing arm. The lever moves to T, after MIXO and FLAT read"),
        "GAIN LOST": "H's gain lives in the constrained windows. Report it. Next: the same T-side ruling"}


def outcome_of(c_ok: bool, k_ok: bool) -> str:
    """The PREREG's outcome table, verbatim: PASS = PRIMARY-C and PRIMARY-K at both seeds; CONSISTENCY NOT RESTORED = PRIMARY-C fails; GAIN LOST = PRIMARY-C passes, PRIMARY-K fails."""
    return "PASS" if (c_ok and k_ok) else ("CONSISTENCY NOT RESTORED" if not c_ok else "GAIN LOST")


def cmd_read(a):
    S = {s: json.load(open(f"{OUT}/score_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if not S[s]["identity_controls"]["all_pass"]:
            raise SystemExit(f"{s}: identity controls did not pass -- no read")
    rows = {}
    for s in SEEDS:
        P, C = S[s]["paired"]["HG_minus_B"], S[s]["consistency"]["paired"]["HG_minus_B"]["viol_any"]
        a2, a6 = P["ade2"], P["ade6"]
        dom_pt = {"ade2": S[s]["single"]["HG"]["ade2"]["mean"] <= S[s]["single"]["P10"]["ade2"]["mean"],
                  "ade6": S[s]["single"]["HG"]["ade6"]["mean"] <= S[s]["single"]["P10"]["ade6"]["mean"],
                  "accel": S[s]["single"]["HG"]["fw_lon_accel_mae"]["mean"] <= S[s]["single"]["P10"]["fw_lon_accel_mae"]["mean"]}
        dP = S[s]["paired"]["HG_minus_P10"]
        rows[s] = {"PRIMARY_C": {"delta": C["mean"], "lo": C["lo"], "hi": C["hi"], "bar_hi_le": 0.02, "pass": bool(C["hi"] <= 0.02)},
                   "PRIMARY_K": {"ade6": {"delta": a6["mean"], "lo": a6["lo"], "hi": a6["hi"], "separated": a6["separated"], "pass": bool(a6["mean"] < 0 and a6["separated"])},
                                 "ade2": {"delta": a2["mean"], "lo": a2["lo"], "hi": a2["hi"], "separated": a2["separated"], "pass": bool(a2["mean"] < 0 and a2["separated"])}},
                   "DOMINANCE_vs_P10": {"point_estimates_le_P10": dom_pt, "ade2": {k: dP["ade2"][k] for k in ("mean", "lo", "hi", "separated")},
                                        "ade6": {k: dP["ade6"][k] for k in ("mean", "lo", "hi", "separated")},
                                        "accel": {k: dP["fw_lon_accel_mae"][k] for k in ("mean", "lo", "hi", "separated")}}}
        rows[s]["PRIMARY_K"]["pass"] = bool(rows[s]["PRIMARY_K"]["ade6"]["pass"] and rows[s]["PRIMARY_K"]["ade2"]["pass"])
        rows[s]["DOMINANCE_vs_P10"]["HG_dominates_P10"] = bool(all(dom_pt.values()) and dP["ade6"]["separated"])
    c_ok = all(rows[s]["PRIMARY_C"]["pass"] for s in SEEDS)
    k_ok = all(rows[s]["PRIMARY_K"]["pass"] for s in SEEDS)
    outcome = outcome_of(c_ok, k_ok)
    R = {"outcome": outcome, "committed_next_step_verbatim": NEXT[outcome], "PRIMARY_C_both_seeds": c_ok, "PRIMARY_K_both_seeds": k_ok,
         "note_both_fail": ("PRIMARY-C and PRIMARY-K both fail: the table row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) matches first; 'GAIN LOST' requires PRIMARY-C to PASS, so it does "
                            "not apply. The K result is reported as measured." if (not c_ok and not k_ok) else None),
         "per_seed": rows, "estimator": S["M2b_s0"]["estimator"], "tier": S["M2b_s0"]["tier"]}
    H.jdump(R, f"{OUT}/READ.json")
    print(json.dumps({k: R[k] for k in ("outcome", "PRIMARY_C_both_seeds", "PRIMARY_K_both_seeds")}, indent=1))
    for s in SEEDS:
        print(s, json.dumps({k: rows[s][k] for k in ("PRIMARY_C", "PRIMARY_K")}, default=float))
    return 0


def cmd_dk(a):
    """LONGITUDINAL distance keeping through the programme's OWN reader `v8f_dk_read.read_dk` (-> `taniteval.lead_metrics`), K = 30 block."""
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
        k = s.split("_")[1]
        preds[f"B_{k}"], preds[f"H_{k}"] = z["pred"].astype(np.float64), np.load(f"{H.OUT}/score_{s}_perwindow.npz")["H"].astype(np.float64)
        preds[f"HG_{k}"] = np.load(f"{OUT}/score_{s}_perwindow.npz")["HG"].astype(np.float64)
        saved[f"B_{k}"] = z["ade6"]
    preds["P10"] = np.load(H.P10_FILE).astype(np.float64)
    fl = np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{H.OUT}/inputs/b1_eval_lead_block_k30.npz")
    pairs = []
    for k in ("s0", "s1"):
        pairs += [(f"HG_{k}", f"B_{k}"), (f"HG_{k}", f"H_{k}"), (f"HG_{k}", "P10"), (f"H_{k}", f"B_{k}")]
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
    sub.add_parser("activity")
    p = sub.add_parser("fit")
    p.add_argument("--seed", required=True, choices=SEEDS)
    p = sub.add_parser("score")
    p.add_argument("--seed", required=True, choices=SEEDS)
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    for name in ("read", "dk"):
        p = sub.add_parser(name)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
    a = ap.parse_args(argv)
    return {"activity": cmd_activity, "fit": cmd_fit, "score": cmd_score, "read": cmd_read, "dk": cmd_dk}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
