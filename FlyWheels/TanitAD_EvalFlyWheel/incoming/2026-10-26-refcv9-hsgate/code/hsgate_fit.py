"""hsgate_fit.py -- PREREG_HSGATE (sha256 2331e1f3...) PHASE 1: the gate is fitted on the FRESH split only and frozen.  CPU only.  NO eval139 gate
output of any kind is computed here (the only eval-side read is K-A5-PATH: a bit-equality of THIS script's P_F code path against the LANDED
FACTORED_DEPLOY8 package, booleans only -- no feature, decision, rate, metric or cell).

Pipeline (FIT = the L1C FRESH split, S6 compliant rows, 7,147 as l1c_fit_score.load_fresh loads them):
  1. load_fresh (S2 + S6) exactly as h2cn_freeze.py; the decision window + the native plan are captured by wrapping the SAME calls (no second loader).
  2. OOF H2cn: the registered recipe (l1_fit_score.mlp_train, the REGISTERED seed-0 wd / epochs, h2n_fit's per-fold standardisation) refit on 4 of 5
     episode folds (l1_fit_score.folds_of(eid, 0)) and applied to the 5th, per surface; the all-rows refit must reproduce the REGISTERED stage
     (f293b7bd...6f32) bit-for-bit (state_dict + predictions).  P_H = the H2CN_DEPLOY8 construction (h2cn_progress, 8 slots) with the OOF f2 / f6.
  3. P_F = the fresh SELFSHARP roll's plan composed onto the native plan + RETIME_DEPLOY8-staged exactly as Amendment 5.
  4. y = 1[ADE_S6(P_F) < ADE_S6(P_H)] (battery ADE vs the human future G).  15 features (hsgate_common.FEATURE_SPEC).
  5. lambda in LAMBDAS by 5-fold episode CV (folds_of(eid, 1)), criterion = mean staged ADE_S6 of the gated pick on the held-out folds (pooled over
     every FIT row once), ties -> the larger lambda; final fit on all FIT rows.  Frozen to HSGATE_FROZEN.json (+ .sha256).
Controls: K-FRESH-SS, K-JOIN (+ mutation RED), K-OOF (bookkeeping + bit-for-bit all-rows refit), K-PATH, K-PRIOR, K-A5-PATH, K-DISJ, K-DETERM, K-RELOAD.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1 python -X utf8 hsgate_fit.py [--stage oof|all]
        [--smoke-partial-trace]  (DEV ONLY: a non-final fresh trace, rows restricted to those it covers, every output under F:/refcv9_hsgate_scratch/smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hsgate_common as H  # noqa: E402

import numpy as np  # noqa: E402

SCRIPT_SHA = H.sha256_file(__file__)
COMMON_SHA = H.sha256_file(HERE / "hsgate_common.py")
FROZEN = H.WORK / "HSGATE_FROZEN.json"
FROZEN_SHA = H.WORK / "HSGATE_FROZEN.sha256"
FIT_INFO = H.WORK / "HSGATE_FIT_INFO.json"
GATE_SCHEMA = "tanitad.refcv9_hsgate.gate/1"


def bit_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def log(msg: str) -> None:
    print(f"[hsgate_fit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------- loading (fresh)
def load_fit_side(L) -> dict:
    """l1c_fit_score.load_fresh for S2 and S6, verbatim (h2cn_freeze.py's call), with TF.train_surface_inputs / CR2.prepare_seed WRAPPED so the
    decision window and the native plan of the SAME call are kept (load_fresh itself drops them)."""
    C, RT = L["C"], L["RT"]
    import l1c_fit_score as LC
    import retime_trainfit_r8 as TF
    if LC.RT is not RT or TF is not L["TFM"]:
        raise SystemExit("[hsgate_fit] module copies differ (l1c_fit_score.RT / retime_trainfit_r8 vs the lane)")
    tools = RT.tools_load()
    TF.TRAIN_ROOT, TF.TRAIN_TREE = str(C.FRESH_TREE_ROOT), C.FRESH_TREE
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    cap = {}
    orig_tsi, orig_ps = TF.train_surface_inputs, CR2.prepare_seed

    def ps_hook(tree, seed, tools_, rc):
        ctx = orig_ps(tree, seed, tools_, rc)
        cap["_ctx"] = {"plan": ctx["plan"], "lim": ctx["lim"]}
        return ctx

    def tsi_hook(tools_, ctx, surf, first, tree):
        S = orig_tsi(tools_, ctx, surf, first, tree)
        rows = ctx["panels"][surf]["rows"]
        cap[surf] = {"win": np.asarray(S["dec"]["win"]), "plan8_all": np.ascontiguousarray(ctx["plan"][rows]), "HP_all": np.asarray(S["X"]["HP"]),
                     "mask": np.asarray(S["mask"], bool), "slots": [int(x) for x in ctx["panels"][surf]["man"]["grid"]["slots"]]}
        return S
    TF.train_surface_inputs, CR2.prepare_seed = tsi_hook, ps_hook
    try:
        ftr = {}
        trs = {surf: LC.load_fresh(tools, CR2, TF, surf, ftr) for surf in ("S2", "S6")}
    finally:
        TF.train_surface_inputs, CR2.prepare_seed = orig_tsi, orig_ps
    return {"trs": trs, "cap": cap, "tools": tools}


# ------------------------------------------------------------------------------------------------------------------------------- OOF H2cn
def oof_h2cn(L, trs, Z, blob, preds_reg, X6, Z6, eid6) -> dict:
    """Per surface: the all-rows refit (must equal the REGISTERED stage bit-for-bit) and 5 fold refits (folds_of(eid, 0)); OOF f on the S6 FIT rows."""
    import torch
    F, AL = L["F"], L["AL"]
    out = {"f_oof": {}, "f_reg_on_fit": {}, "rec": {}}
    fold6 = F.folds_of(eid6, H.OOF_FOLD_SEED)
    ep_fold = {}
    for e, f in zip(eid6, fold6):
        ep_fold.setdefault(e, int(f))
    for surf in ("S2", "S6"):
        tr = trs[surf]
        X, Zs, eid = tr["X"], Z[surf], tr["eid"]
        Hs = AL.arc(tr["G"]) - AL.arc(tr["HP"])
        b = blob["surfaces"][surf]
        wd, ep = float(b["wd"]), int(b["epochs"])
        foldS = F.folds_of(eid, H.OOF_FOLD_SEED)
        same_eps = bool(set(np.unique(eid)) == set(np.unique(eid6)))
        consistent = bool(all(ep_fold.get(e, f) == f for e, f in zip(eid, foldS)))
        # ---- all-rows refit == the REGISTERED stage (h2n_fit's final call, verbatim arguments)
        t1 = time.time()
        st = {"xmu": X[:, 1:].mean(0), "xsd": X[:, 1:].std(0) + 1e-9, "zmu": Zs.mean(0), "zsd": Zs.std(0) + 1e-9}
        net, _ = F.mlp_train(F.mlp_inputs(X, Zs, st), Hs, wd, ep, 0)
        with torch.no_grad():
            p_ref_rows = net(torch.from_numpy(F.mlp_inputs(X, Zs, st))).numpy()
            p_ref_fit = net(torch.from_numpy(F.mlp_inputs(X6, Z6, st))).numpy()
        sd_r = {k: v.detach().numpy() for k, v in net.state_dict().items()}
        sd_eq = bool(set(sd_r) == set(b["state_dict"]) and all(bit_equal(sd_r[k], b["state_dict"][k].numpy()) for k in sd_r))
        st_eq = bool(all(bit_equal(np.asarray(st[k], np.float64), b["st"][k].numpy()) for k in ("xmu", "xsd", "zmu", "zsd")))
        reg_rows, reg_fit = preds_reg[surf](X, Zs), preds_reg[surf](X6, Z6)
        moved = p_ref_fit.copy()
        moved.ravel()[0] = np.nextafter(moved.ravel()[0], np.inf)
        rec = {"wd": wd, "epochs": ep, "n_rows": int(len(X)), "n_episodes": int(len(np.unique(eid))),
               "allrows_state_dict_bit_equal_registered": sd_eq, "allrows_stats_bit_equal_registered": st_eq,
               "allrows_pred_bit_equal_registered_on_surface_rows": bit_equal(p_ref_rows, reg_rows),
               "allrows_pred_bit_equal_registered_on_S6_FIT_rows": bit_equal(p_ref_fit, reg_fit),
               "comparator_mutation_1ulp_red": bool(not bit_equal(moved, reg_fit)),
               "allrows_wall_s": round(time.time() - t1, 1), "fold_seed": H.OOF_FOLD_SEED, "surface_episode_set_equals_S6_FIT": same_eps,
               "fold_map_consistent_with_S6_FIT_map": consistent}
        # ---- 5 fold refits, OOF predictions on the S6 FIT rows
        f_oof = np.full((len(X6), Hs.shape[1]), np.nan)
        folds = {}
        excl_ok = True
        for j in range(F.NFOLD):
            trm = np.asarray([ep_fold.get(e, int(f)) != j for e, f in zip(eid, foldS)])     # the S6-FIT map decides (== folds_of(eid,0) when consistent)
            te6 = fold6 == j
            stj = {"xmu": X[trm, 1:].mean(0), "xsd": X[trm, 1:].std(0) + 1e-9, "zmu": Zs[trm].mean(0), "zsd": Zs[trm].std(0) + 1e-9}
            netj, _ = F.mlp_train(F.mlp_inputs(X[trm], Zs[trm], stj), Hs[trm], wd, ep, 0)
            with torch.no_grad():
                f_oof[te6] = netj(torch.from_numpy(F.mlp_inputs(X6[te6], Z6[te6], stj))).numpy()
            train_eps = set(np.unique(eid[trm]))
            leak = int(sum(e in train_eps for e in eid6[te6]))
            excl_ok = excl_ok and leak == 0
            folds[str(j)] = {"n_train_rows": int(trm.sum()), "n_train_episodes": len(train_eps), "n_heldout_fit_rows": int(te6.sum()),
                             "n_heldout_fit_episodes": int(len(np.unique(eid6[te6]))), "heldout_rows_whose_episode_is_in_training": leak}
        rec["folds"] = folds
        rec["every_fit_row_predicted_once"] = bool(np.isfinite(f_oof).all())
        rec["every_fit_row_episode_excluded_from_its_fold_model"] = bool(excl_ok)
        rec["oof_vs_allrows_mean_abs_m"] = float(np.abs(f_oof - reg_fit).mean())
        rec["PASS"] = bool(sd_eq and st_eq and rec["allrows_pred_bit_equal_registered_on_surface_rows"] and rec["allrows_pred_bit_equal_registered_on_S6_FIT_rows"]
                           and rec["comparator_mutation_1ulp_red"] and rec["every_fit_row_predicted_once"] and excl_ok and same_eps and consistent)
        out["f_oof"][surf], out["f_reg_on_fit"][surf], out["rec"][surf] = f_oof, reg_fit, rec
        log(f"OOF {surf}: wd={wd} epochs={ep} allrows==registered sd={sd_eq} pred={rec['allrows_pred_bit_equal_registered_on_S6_FIT_rows']} "
            f"excl={excl_ok} PASS={rec['PASS']}")
    out["fold6"] = fold6
    return out


# ------------------------------------------------------------------------------------------------------------------------------- the fresh trace
def load_fresh_trace(smoke: bool) -> dict:
    z = np.load(H.FRESH_TRACE, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    d = {k: z[k] for k in ("e_i", "ws", "raw_lon_logits", "applied_lon_logits", "plan8", "v0_in")}
    d["meta"] = meta
    d["sha256"] = H.sha256_file(H.FRESH_TRACE)
    final = bool(meta.get("final") is True and int(meta.get("n_windows", -1)) == H.FRESH_N_WINDOWS == len(d["ws"]))
    if not final and not smoke:
        raise SystemExit(f"[hsgate_fit] the fresh SELFSHARP trace is not final ({meta.get('final')}, {meta.get('n_windows')} windows) -- refusing")
    d["final"] = final
    return d


def k_fresh_ss(T) -> dict:
    """K-FRESH-SS: FINAL with 10,456 windows; applied lon logits one-hot +-20 on the label on every window; raw LON6 argmax == label on every window."""
    lab_z = np.load(H.FRESH_LABELS, allow_pickle=False)
    lab = {(int(c), int(w)): int(l) for c, w, l in zip(lab_z["clip_index"], lab_z["ws"], lab_z["lon_v7"])}
    n = len(T["ws"])
    labs = np.array([lab.get((int(e), int(w)), -100) for e, w in zip(T["e_i"], T["ws"])])
    L6 = np.array(H.LON6_V7_IDS)
    all_lab = bool(np.isin(labs, L6).all())
    exp = np.full((n, 8), -20.0, np.float32)
    if all_lab:
        exp[np.arange(n), labs.astype(np.int64)] = 20.0
    app_ok = bool(all_lab and np.array_equal(T["applied_lon_logits"], exp))
    rawarg = L6[np.asarray(T["raw_lon_logits"])[:, L6].argmax(1)]
    n_eq = int((rawarg == labs).sum())
    c = T["meta"].get("counters", {})
    r = {"final": bool(T["final"]), "n_windows": int(n), "want": H.FRESH_N_WINDOWS, "all_windows_labelled_lon6": all_lab,
         "applied_onehot_pm20_on_label_every_window": app_ok, "raw_argmax_eq_label": {"n_eq": n_eq, "n": int(n)},
         "raw_argmax_eq_label_every_window": bool(n_eq == n), "keys_unique": bool(len(set(zip(T["e_i"].tolist(), T["ws"].tolist()))) == n),
         "counters": {k: c.get(k) for k in ("calls", "hook_applied", "hook_calls_unarmed", "batch_gt1", "multi_hook_in_call", "oracle_labelled",
                                            "oracle_unlabelled_or_masked", "navnull_calls", "hook_native_fallthrough")},
         "labels_file": str(H.FRESH_LABELS), "labels_sha256": H.sha256_file(H.FRESH_LABELS), "labels_sha256_in_trace_meta": T["meta"].get("labels_sha256"),
         "trace_file": str(H.FRESH_TRACE), "trace_sha256": T["sha256"], "trace_tool": T["meta"].get("tool")}
    r["counters_ok"] = bool(c.get("hook_applied") == n and c.get("calls") == n and c.get("hook_calls_unarmed") == 0 and c.get("batch_gt1") == 0
                            and c.get("multi_hook_in_call") == 0 and c.get("oracle_labelled") == n and c.get("oracle_unlabelled_or_masked") == 0)
    r["PASS"] = bool(r["final"] and n == H.FRESH_N_WINDOWS and all_lab and app_ok and r["raw_argmax_eq_label_every_window"] and r["keys_unique"]
                     and r["counters_ok"] and r["labels_sha256"] == r["labels_sha256_in_trace_meta"])
    return r


def join_trace(T, clip, ws, v0_panel, ws_shift=0) -> tuple:
    idx = {}
    for j, k in enumerate(zip(T["e_i"].tolist(), T["ws"].tolist())):
        idx.setdefault((int(k[0]), int(k[1])), []).append(j)
    hits = [idx.get((int(c), int(w) + ws_shift), []) for c, w in zip(clip, ws)]
    j = np.asarray([h[0] if len(h) == 1 else -1 for h in hits], np.int64)
    ok = j >= 0
    jj = np.where(ok, j, 0)
    rec = {"rows": int(len(j)), "trace_rows": int(len(T["ws"])), "trace_duplicate_keys": int(sum(len(v) - 1 for v in idx.values())),
           "missing": int(sum(len(h) == 0 for h in hits)), "multi": int(sum(len(h) > 1 for h in hits)), "ws_shift": ws_shift,
           "v0_in_vs_panel_v0_maxabs": (float(np.abs(T["v0_in"][jj][ok] - np.asarray(v0_panel, np.float64)[ok]).max()) if ok.any() else None)}
    rec["PASS"] = bool(rec["trace_duplicate_keys"] == 0 and rec["missing"] == 0 and rec["multi"] == 0 and rec["v0_in_vs_panel_v0_maxabs"] is not None
                       and rec["v0_in_vs_panel_v0_maxabs"] <= 1e-5)
    return j, rec


# ------------------------------------------------------------------------------------------------------------------------------- K-A5-PATH (eval: identity only)
def k_a5_path(L, seed: int) -> dict:
    """THIS script's P_F code path (pf_positions) re-applied to the LANDED FACTORED_DEPLOY8 package's own inputs must reproduce its S6 composed plan and
    staged positions bit-for-bit.  Booleans + counts only; no feature / decision / rate / metric / cell is computed on eval rows."""
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_score as S2c
    b1 = S2c.load_l1_base(str(H.EVAL_L1_BANK), seed)
    ts = S2c.load_trace(H.EVAL_SS_TRACE.format(seed=seed))
    z = np.load(str(H.LANDED_FX).format(seed=seed), allow_pickle=False)
    keys = list(zip(z["S6_clip"].tolist(), z["S6_ws"].tolist()))
    jb = np.array([b1["idx"].get(k, -1) for k in keys])
    jt = np.array([ts["idx"].get(k, -1) for k in keys])
    if (jb < 0).any() or (jt < 0).any():
        return {"PASS": False, "join_missing": [int((jb < 0).sum()), int((jt < 0).sum())]}
    P_fx, p32, p64 = H.pf_positions(L, b1["plan8"][jb], ts["plan8"][jt], z["S6_prior8"])
    r = {"n_windows": int(len(keys)), "composed_plan8_bit_equal_landed_S6_plan8": bit_equal(P_fx, z["S6_plan8"]),
         "staged_S6_pos_bit_equal_landed": bit_equal(p32[:, H.S6], z["S6_pos"]), "staged_S6_pos64_bit_equal_landed": bit_equal(p64[:, H.S6], z["S6_pos64"]),
         "staged_pos8_bit_equal_landed": bit_equal(p32, z["S6_pos8"]), "landed_file": str(H.LANDED_FX).format(seed=seed),
         "landed_sha256": H.sha256_file(str(H.LANDED_FX).format(seed=seed))}
    r["PASS"] = bool(r["composed_plan8_bit_equal_landed_S6_plan8"] and r["staged_S6_pos_bit_equal_landed"] and r["staged_S6_pos64_bit_equal_landed"]
                     and r["staged_pos8_bit_equal_landed"])
    return r


# ------------------------------------------------------------------------------------------------------------------------------- the gate fit
def cv_lambda(F, X, y, adeF, adeH, eid6) -> tuple:
    fold = F.folds_of(eid6, H.CV_FOLD_SEED)
    table = []
    for lam in H.LAMBDAS:
        pick = np.full(len(y), np.nan)
        gho = np.zeros(len(y), bool)
        per_fold, conv = [], []
        for j in range(F.NFOLD):
            tr, te = fold != j, fold == j
            mu, sd = H.standardise_fit(X[tr])
            m = H.lr_fit((X[tr] - mu) / sd, y[tr], lam)
            g = H.gate_decide({"mu": mu, "sd": sd, "w": m["w"], "b": m["b"]}, X[te])
            gho[te] = g
            pick[te] = np.where(g, adeF[te], adeH[te])
            per_fold.append(float(pick[te].mean()))
            conv.append({"max_abs_grad": m["max_abs_grad"], "n_iter": m["n_iter"]})
        table.append({"lambda": float(lam), "criterion_mean_staged_ADE_S6_heldout": float(pick.mean()), "per_fold": per_fold,
                      "heldout_F_rate": float(gho.mean()), "heldout_accuracy_vs_label": float((gho == y.astype(bool)).mean()),
                      "max_abs_grad_over_folds": float(max(c["max_abs_grad"] for c in conv)), "n_iter_over_folds": [c["n_iter"] for c in conv]})
        log(f"CV lambda={lam:g}: criterion {table[-1]['criterion_mean_staged_ADE_S6_heldout']:.6f} heldout F-rate {table[-1]['heldout_F_rate']:.4f}")
    crit = np.array([t["criterion_mean_staged_ADE_S6_heldout"] for t in table])
    best = [i for i in range(len(table)) if crit[i] == crit.min()]
    pick_i = max(best, key=lambda i: H.LAMBDAS[i])                     # exact ties -> the larger lambda
    return float(H.LAMBDAS[pick_i]), table, fold, {"rule": "argmin criterion; exact ties -> the larger lambda", "n_tied_at_min": len(best)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("oof", "all"), default="all")
    ap.add_argument("--smoke-partial-trace", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    smoke = bool(a.smoke_partial_trace)
    env = H.env_check()
    prereg = H.check_prereg()
    h2cn = H.check_h2cn_stage()
    out_dir = (H.SCRATCH / "smoke") if smoke else H.WORK
    frozen, frozen_sha, fit_info = out_dir / FROZEN.name, out_dir / FROZEN_SHA.name, out_dir / FIT_INFO.name
    if a.stage == "all" and not smoke:
        for p in (FROZEN, FROZEN_SHA, H.WORK / "REGISTERED_HSGATE.sha256"):
            if p.exists():
                raise SystemExit(f"[hsgate_fit] REFUSING: {p} already exists (the gate is frozen ONCE; registration follows the freeze)")
    log(f"prereg {prereg['sha256'][:16]}.. OK; H2cn stage registered OK; smoke={smoke} stage={a.stage}")
    L = H.lane()
    C, F, AL, RT = L["C"], L["F"], L["AL"], L["RT"]
    import torch
    log(f"lane OK (deploy table {L['rec']['deploy_table_sha256'][:16]}..), torch threads {torch.get_num_threads()}")
    # ---- 1. fresh FIT side
    D = load_fit_side(L)
    trs, cap = D["trs"], D["cap"]
    tr6 = trs["S6"]
    log(f"fresh loaded: S2 {len(trs['S2']['X'])} rows / S6 {len(tr6['X'])} compliant rows (all S6 {cap['S6']['mask'].size})")
    fresh_bank = F.load_bank(H.FRESH_BANK)
    traj_own32 = np.load(H.FRESH_BANK, allow_pickle=False)["traj_own"]
    Z = {s: F.join(fresh_bank, trs[s]["clip"], trs[s]["ws"])[0] for s in ("S2", "S6")}
    _z6, jb6 = F.join(fresh_bank, tr6["clip"], tr6["ws"])
    m6 = cap["S6"]["mask"]
    plan8 = cap["S6"]["plan8_all"][m6]
    P_os8 = np.ascontiguousarray(traj_own32[jb6])
    prior8_all = H.prior8_of(RT, L["SF"], cap["S6"]["win"])
    prior8 = prior8_all[m6]
    k_path = {"fresh_plan8_bit_equal_bank_traj_own": bit_equal(plan8, P_os8), "plan8_dtype": str(plan8.dtype),
              "plan8_at_S6_slots_equal_panel_O": bit_equal(plan8[:, H.S6], np.asarray(tr6["O"])), "S6_slots_from_manifest": cap["S6"]["slots"] == H.S6,
              "n_compliant_mask_equals_rows": bool(int(m6.sum()) == len(tr6["X"]))}
    k_path["PASS"] = bool(all(v for k, v in k_path.items() if isinstance(v, bool)))
    k_prior = {"prior8_at_S6_slots_bit_equal_RT_HP_all_rows": bit_equal(prior8_all[:, H.S6], cap["S6"]["HP_all"]),
               "prior8_at_S6_slots_bit_equal_load_fresh_HP": bit_equal(prior8[:, H.S6], np.asarray(tr6["HP"]))}
    k_prior["PASS"] = bool(all(k_prior.values()))
    log(f"K-PATH {k_path['PASS']}  K-PRIOR {k_prior['PASS']}")
    disj = C.fresh_disjointness({s: np.unique(trs[s]["sha12"]) for s in trs})
    disj_mut = C.fresh_disjointness(None, inject_eval_clip=True)
    k_disj = bool(disj["PASS"] and not disj_mut["PASS"] and disj_mut["intersections"]["fresh12_x_eval139"] == 1)
    # ---- 2. OOF H2cn
    preds_reg, blob, _sha = C.load_stage(H.H2CN_STAGE, registered_sha=H.H2CN_STAGE_SHA)
    eid6 = np.asarray(tr6["eid"])
    O = oof_h2cn(L, trs, Z, blob, preds_reg, tr6["X"], Z["S6"], eid6)
    k_oof = {"S2": O["rec"]["S2"], "S6": O["rec"]["S6"]}
    k_oof_pass = bool(O["rec"]["S2"]["PASS"] and O["rec"]["S6"]["PASS"])
    PH32, PH64, s8H, chH = H.ph_positions(L, plan8, prior8, O["f_oof"]["S2"], O["f_oof"]["S6"])
    PHreg32, _p, _s, _c = H.ph_positions(L, plan8, prior8, O["f_reg_on_fit"]["S2"], O["f_reg_on_fit"]["S6"])
    L2, HL = H.load_l2_common()
    ra = L2.stack()["ra"]
    G6 = np.asarray(tr6["G"])
    adeH = H.ade6(ra, PH32[:, H.S6], G6, tr6["dt"])
    adeH_insample = H.ade6(ra, PHreg32[:, H.S6], G6, tr6["dt"])
    log(f"P_H (OOF) built: running-max changed {int(chH.sum())} windows; K-OOF {k_oof_pass}")
    # ---- extras (features) join: h2cnl_freeze.extras_join (v0 / sha12 / plan cross-checks) + its mutation (next window)
    ex = np.load(H.FRESH_EXTRAS, allow_pickle=False)
    ex = {k: ex[k] for k in ("clip_index", "ws", "v0", "nav_true.traj", H.LOGIT_KEY)}
    ex12 = json.load(open(str(H.FRESH_EXTRAS) + ".json", encoding="utf-8"))["clip_sha12_by_index"]
    jx, k_join_ex = HL.extras_join(ex, ex12, tr6, "S6")
    o = np.lexsort((ex["ws"], ex["clip_index"]))
    stride = int(np.median(np.diff(ex["ws"][o])[np.diff(ex["clip_index"][o]) == 0]))
    _jxm, k_join_ex_mut = HL.extras_join(ex, ex12, tr6, "S6", ws_shift=stride)
    info = {"tool": "hsgate_fit.py", "role": "PREREG_HSGATE PHASE 1 (fit side only; no eval139 gate output)", "smoke_partial_trace": smoke, "stage": a.stage,
            "prereg": prereg, "h2cn_stage": h2cn, "env": env, "lane": L["rec"],
            "code_sha256": {"hsgate_fit.py": SCRIPT_SHA, "hsgate_common.py": COMMON_SHA},
            "inputs": {"fresh_bank": {"file": str(H.FRESH_BANK), "sha256": H.sha256_file(H.FRESH_BANK)},
                       "fresh_extras": {"file": str(H.FRESH_EXTRAS), "sha256": H.sha256_file(H.FRESH_EXTRAS)},
                       "fresh_tree": f"{C.FRESH_TREE_ROOT}/{C.FRESH_TREE}"},
            "controls": {"K-PATH": k_path, "K-PRIOR": k_prior, "K-DISJ": {"PASS": k_disj, "detail": disj, "mutation_arm": disj_mut}, "K-OOF": k_oof,
                         "K-JOIN-EXTRAS": {"join": k_join_ex, "mutation_next_window": k_join_ex_mut}},
            "fit_rows": {"n": int(len(eid6)), "n_episodes": int(len(np.unique(eid6))), "n_clips": int(len(np.unique(tr6["sha12"])))},
            "oof_h2cn": {"running_max_changed_windows": int(chH.sum()), "mean_ADE_S6_H2cn_OOF": float(adeH.mean()),
                         "mean_ADE_S6_H2cn_insample_registered": float(adeH_insample.mean())}}
    if a.stage == "oof":
        for s in ("S2", "S6"):
            np.save(H.SCRATCH / f"oof_f_{s}.npy", O["f_oof"][s])
        info["controls"]["K-A5-PATH"] = {f"s{sd}": k_a5_path(L, sd) for sd in (0, 1)}
        info["wall_s"] = round(time.time() - t0, 1)
        H.jdump(info, H.SCRATCH / "OOF_STAGE_INFO.json")
        log(f"stage oof done: K-OOF {k_oof_pass} K-A5-PATH {[v['PASS'] for v in info['controls']['K-A5-PATH'].values()]} -> {H.SCRATCH / 'OOF_STAGE_INFO.json'}")
        return 0
    # ---- 3. the fresh SELFSHARP trace -> P_F
    st_chain = H.chain_status()
    T = load_fresh_trace(smoke)
    kss = k_fresh_ss(T)
    jt, k_join_tr = join_trace(T, tr6["clip"], tr6["ws"], tr6["v0"])
    _jtm, k_join_tr_mut = join_trace(T, tr6["clip"], tr6["ws"], tr6["v0"], ws_shift=stride)
    k_join_tr_mut["red"] = bool(not k_join_tr_mut["PASS"])
    k_join_tr_mut["red_by_v0_crosscheck_alone"] = bool(k_join_tr_mut["v0_in_vs_panel_v0_maxabs"] is not None and k_join_tr_mut["v0_in_vs_panel_v0_maxabs"] > 1e-5)
    k_join_ex_mut_red = bool(not k_join_ex_mut["PASS"])
    keep = np.ones(len(eid6), bool)
    if smoke:
        keep = jt >= 0
        log(f"SMOKE: {int(keep.sum())} of {len(keep)} FIT rows covered by the partial trace")
    k_join = {"trace": k_join_tr, "trace_mutation_next_window": k_join_tr_mut, "extras": k_join_ex, "extras_mutation_next_window": k_join_ex_mut,
              "PASS": bool(k_join_tr["PASS"] and k_join_ex["PASS"] and k_join_tr_mut["red"] and k_join_tr_mut["red_by_v0_crosscheck_alone"] and k_join_ex_mut_red)}
    if smoke:
        k_join["PASS_smoke_rows_only"] = bool(k_join_ex["PASS"])
    log(f"K-FRESH-SS {kss['PASS']} (final {kss['final']}, raw==label {kss['raw_argmax_eq_label']['n_eq']}/{kss['raw_argmax_eq_label']['n']}); K-JOIN {k_join['PASS']}")
    if not smoke and not (kss["PASS"] and k_join["PASS"]):
        info["controls"].update({"K-FRESH-SS": kss, "K-JOIN": k_join})
        H.jdump(info, H.WORK / "HSGATE_FIT_WITHHELD.json")
        raise SystemExit("[hsgate_fit] WITHHELD: K-FRESH-SS or K-JOIN failed -> HSGATE_FIT_WITHHELD.json")
    sel = np.where(keep)[0]
    P_ss8 = np.ascontiguousarray(T["plan8"][jt[sel]])
    P_fx, PF32, PF64 = H.pf_positions(L, P_os8[sel], P_ss8, prior8[sel])
    PF6, PH6 = PF32[:, H.S6], PH32[sel][:, H.S6]
    adeF = H.ade6(ra, PF6, G6[sel], tr6["dt"])
    aH = adeH[sel]
    y = (adeF < aH).astype(np.float64)
    logits8 = ex[H.LOGIT_KEY][jx[sel]]
    v0x = ex[H.V0_KEY][jx[sel]]
    X, parts = H.features(L2, HL, AL, logits8, v0x, PF6, PH6)
    e6 = eid6[sel]
    log(f"label base rate {y.mean():.4f} on {len(y)} FIT rows; mean ADE_S6 F {adeF.mean():.4f} H(OOF) {aH.mean():.4f}")
    # ---- 4. lambda CV + final fit
    lam, cv_table, cvfold, tie = cv_lambda(F, X, y, adeF, aH, e6)
    mu, sd = H.standardise_fit(X)
    m = H.lr_fit((X - mu) / sd, y, lam)
    gate = {"schema": GATE_SCHEMA, "prereg_sha256": prereg["sha256"], "features": H.FEATURE_SPEC, "decision": H.DECISION, "objective": H.OBJECTIVE,
            "lambda": float(lam), "lambda_grid": [float(x) for x in H.LAMBDAS], "cv_fold_seed": H.CV_FOLD_SEED, "oof_fold_seed": H.OOF_FOLD_SEED,
            "lambda_selection": tie["rule"], "cv_table": cv_table,
            "mu": [float(v) for v in mu], "sd": [float(v) for v in sd], "w": [float(v) for v in m["w"]], "b": float(m["b"]),
            "fit": {"n_rows": int(len(y)), "n_episodes": int(len(np.unique(e6))), "label_base_rate": float(y.mean()), "loss": m["loss"],
                    "max_abs_grad": m["max_abs_grad"], "n_iter": m["n_iter"]},
            "arms": {"F": "FACTORED_DEPLOY8 (Amendment 5: compose(RT, P_os8, P_ss8) then the registered per-frame RETIME_DEPLOY8 table)",
                     "H": "H2CN_DEPLOY8 (Amendment 4: the registered H2cn stage f293b7bd...6f32, 8 slots, running max)"},
            "label": "y = 1[ADE_S6(P_F) < ADE_S6(P_H)] vs the human future G; P_H with OUT-OF-FOLD H2cn (folds_of(eid, 0)); battery ADE (refav1_arm 'ade_m')",
            "inputs_sha256": {"h2cn_stage": H.H2CN_STAGE_SHA, "fresh_trace": T["sha256"], "fresh_labels": kss["labels_sha256"],
                              "fresh_bank": info["inputs"]["fresh_bank"]["sha256"], "fresh_extras": info["inputs"]["fresh_extras"]["sha256"],
                              "deploy_table": L["rec"]["deploy_table_sha256"], "registered_pf_table": L["rec"]["registered_pf_table_sha256"]},
            "code_sha256": {"hsgate_fit.py": SCRIPT_SHA, "hsgate_common.py": COMMON_SHA},
            "use": "x = hsgate_common.features(L2, HL, AL, extras lon logits, extras v0, P_F S6 positions, P_H S6 positions); g = hsgate_common.gate_decide(gate, x); "
                   "emitted positions = FACTORED_DEPLOY8's where g else H2CN_DEPLOY8's"}
    gfit = H.gate_decide(gate, X)
    # K-DETERM: refit + reserialise -> identical bytes
    m2 = H.lr_fit((X - mu) / sd, y, lam)
    gate2 = dict(gate, w=[float(v) for v in m2["w"]], b=float(m2["b"]), fit=dict(gate["fit"], loss=m2["loss"], max_abs_grad=m2["max_abs_grad"], n_iter=m2["n_iter"]))
    b1, b2 = H.canonical_bytes(gate), H.canonical_bytes(gate2)
    k_det = bool(b1 == b2 == H.canonical_bytes(json.loads(b1.decode("utf-8"))))
    # ---- controls summary + write
    k_a5 = {f"s{sd_}": k_a5_path(L, sd_) for sd_ in (0, 1)}
    controls = {"K-FRESH-SS": kss, "K-JOIN": k_join, "K-OOF": k_oof, "K-OOF_PASS": k_oof_pass, "K-PATH": k_path, "K-PRIOR": k_prior,
                "K-DISJ": {"PASS": k_disj, "detail": disj, "mutation_arm": disj_mut}, "K-A5-PATH": k_a5, "K-DETERM": k_det}
    all_pass = bool(kss["PASS"] and k_join["PASS"] and k_oof_pass and k_path["PASS"] and k_prior["PASS"] and k_disj and all(v["PASS"] for v in k_a5.values())
                    and k_det and m["max_abs_grad"] < 1e-6)
    fit_side = {"n_rows": int(len(y)), "label_base_rate": float(y.mean()), "F_rate_final_gate_on_FIT_rows_insample": float(gfit.mean()),
                "mean_ADE_S6": {"H2cn_OOF": float(aH.mean()), "H2cn_insample_registered": float(adeH_insample[sel].mean()), "FACTORED": float(adeF.mean()),
                                "oracle_min": float(np.minimum(adeF, aH).mean()), "gate_final_insample": float(np.where(gfit, adeF, aH).mean())},
                "d_feature": {"mean": float(parts["d"].mean()), "p05": float(np.quantile(parts["d"], 0.05)), "p95": float(np.quantile(parts["d"], 0.95))},
                "argmax_class_counts": {H.LON6[i]: int(c) for i, c in enumerate(np.bincount(np.argmax(L2.lon6(logits8), 1), minlength=6))},
                "lambda": lam, "cv_table": cv_table, "lambda_tie": tie}
    info.update({"controls": {**info["controls"], **controls}, "ALL_PASS": all_pass, "fresh_chain_status": st_chain, "fit_side": fit_side,
                 "trace": {"file": str(H.FRESH_TRACE), "sha256": T["sha256"], "final": T["final"], "n_windows": int(len(T["ws"])), "saved": T["meta"].get("saved")}})
    if not all_pass and not smoke:
        H.jdump(info, H.WORK / "HSGATE_FIT_WITHHELD.json")
        log("WITHHELD: a control failed -> HSGATE_FIT_WITHHELD.json  ZZHSGATEFITWITHHELDZZ")
        return 4
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (frozen.name + ".tmp")
    tmp.write_bytes(b1)
    os.replace(tmp, frozen)
    sha = H.sha256_file(frozen)
    if sha != __import__("hashlib").sha256(b1).hexdigest() or len(sha) != 64:
        raise SystemExit("[hsgate_fit] the written gate file does not hash to its bytes")
    frozen_sha.write_text(f"{sha}  {frozen.name}\n", encoding="utf-8")
    reload = json.loads(frozen.read_text(encoding="utf-8"))                # K-RELOAD: decisions from the FILE == in memory
    k_reload = bool(np.array_equal(H.gate_decide(reload, X), gfit) and np.array_equal(H.gate_logit(reload, X), H.gate_logit(gate, X)))
    info["controls"]["K-RELOAD"] = k_reload
    info["gate_file"] = {"path": str(frozen), "sha256": sha, "bytes": frozen.stat().st_size}
    info["wall_s"] = round(time.time() - t0, 1)
    info["generated"] = time.strftime("%FT%T%z")
    H.jdump(info, fit_info)
    np.savez(H.SCRATCH / ("smoke_fit_rows.npz" if smoke else "FIT_ROWS.npz"), X=X, y=y, adeF=adeF, adeH_oof=aH, adeH_insample=adeH_insample[sel],
             g_final=gfit, cv_fold=cvfold, oof_fold=O["fold6"][sel], episode_code=np.unique(e6, return_inverse=True)[1], clip_index=tr6["clip"][sel],
             ws=tr6["ws"][sel], f2_oof=O["f_oof"]["S2"][sel], f6_oof=O["f_oof"]["S6"][sel], PF6=PF6, PH6=PH6)
    chk = json.loads(fit_info.read_text(encoding="utf-8"))
    assert chk["gate_file"]["sha256"] == H.first_token(frozen_sha) == H.sha256_file(frozen) and (chk["ALL_PASS"] is True or smoke)
    log(f"WROTE {frozen} sha256 {sha} lambda {lam:g} F-rate(FIT) {gfit.mean():.4f} K-RELOAD {k_reload} ALL_PASS {all_pass} wall {info['wall_s']} s  ZZHSGATEFITOKZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
