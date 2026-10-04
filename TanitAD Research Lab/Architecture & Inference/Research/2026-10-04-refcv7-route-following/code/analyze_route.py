"""SPEC.md sec. 4-6: the route-following chain, the decision rule, the selection levers and the fan statistics.

Inputs: the capture packs written by run_route.py (<tag>.npz + <tag>.json): eval_s0 (primary), eval_s1 (inference
replicate), train_s0 (FIT ONLY), reel_s0 (re-forward of the 12 reel clips) and the reel bank rows.jsonl.
CPU only. The four-family rate geometry is the launch tree's own `taniteval.four_families` (imported from the
launch-tree copy; the caller sets PYTHONPATH + TANITEVAL_STACK_OVERRIDE).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_metrics as rm  # noqa: E402

B = 2000
SEED = 0
CLASSES = ("turnL", "turnR", "turn", "straight", "gentle", "all")
TAU_GRID = (0.05, 0.10, 0.18063741505146028, 0.30)
PMIN_GRID = (0.0, 0.5, 0.7, 0.9)
K_GRID = (1, 2, 5, 10, 20, 50, 100)
BETA_GRID = (0.5, 1, 2, 5, 10, 20)
SEAM_CLAMP_E9 = 1.0


def load(p):
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["_json"] = json.loads(Path(str(p).replace(".npz", ".json")).read_text(encoding="utf-8"))
    return d


def e9_blend(s_core, graft, clamp=SEAM_CLAMP_E9):
    """refc_select.apply_seam_clamp in numpy (clamp > 0 path): base + graft * clamp / max(|g|/|b|, clamp)."""
    bn = np.maximum(np.linalg.norm(s_core, axis=1), 1e-9)
    ratio = np.linalg.norm(graft, axis=1) / bn
    scale = clamp / np.maximum(ratio, clamp)
    return s_core + graft * scale[:, None]


def masked_argmax(score, keep, fallback):
    r = np.where(keep, score, -np.inf)
    empty = ~keep.any(1)
    r = np.where(empty[:, None], np.where(fallback, score, -np.inf), r)
    return r.argmax(1), empty


def derive(z):
    W = len(z["sel_idx"])
    d = {"W": W, "ep": np.asarray(z["win_sha12"]).astype(str)}
    cls, th_gt = rm.gt_class(z["gt"], z["gt_valid"])
    d["cls"], d["th_gt"] = cls, th_gt
    d["target"] = np.where(np.isin(cls, ["turnL", "turnR", "straight", "gentle"]), rm.dir_class(th_gt), 9)
    d["th_c"] = rm.terminal_heading(z["fan"])                          # [W, N]
    d["dir_c"] = rm.dir_class(d["th_c"])
    d["ade_c"], d["fde_c"], _ = rm.ade_fde(z["fan"], z["gt"], z["gt_valid"])
    d["reach"] = z["reach"].astype(bool)
    d["ceil"] = z["ceil_keep"].astype(bool)
    d["lat_arg"] = z["p_lat"].argmax(1)
    d["tac_side"] = rm.LAT_SIDE[d["lat_arg"]]
    d["tac_strict"] = np.where(d["lat_arg"] == 6, 1, np.where(d["lat_arg"] == 7, -1, 0))
    pl, pr = z["p_goal"][:, 1], z["p_goal"][:, 2]
    d["goal_side"] = np.where((np.maximum(pl, pr) >= 0.5) & (pl >= pr), 1,
                              np.where((np.maximum(pl, pr) >= 0.5) & (pr > pl), -1, 0))
    d["nav_side"] = np.array([rm.NAV_SIDE[int(n)] for n in z["nav"]])
    lg = z["lat_gt"]
    d["label_side"] = np.where(lg >= 0, rm.LAT_SIDE[np.clip(lg, 0, 7)], 9)
    d["oracle"] = np.nanargmin(np.where(np.isfinite(d["ade_c"]), d["ade_c"], np.inf), axis=1)
    d["oracle_reach"] = np.argmin(np.where(d["reach"] & np.isfinite(d["ade_c"]), d["ade_c"], np.inf), axis=1)
    # E9 with the SPEC_REFCV7 26.1 fix: reach AND ceiling, reach-only where empty
    d["pick_fix"], d["fix_empty"] = masked_argmax(z["s_e9"], d["reach"] & d["ceil"], d["reach"])
    d["navc_term"] = z["rank_terms"][:, 1]
    d["beh_term"] = z["rank_terms"][:, 0]
    return d


# ------------------------------------------------------------------------------------------------- #
# controls C1 / C2 (re-derived from the captured arrays)                                            #
# ------------------------------------------------------------------------------------------------- #
def controls(z, d):
    c = {}
    c["C1_traj_vs_fan_max_abs"] = float(np.abs(z["traj"] - z["fan"][np.arange(d["W"]), z["sel_idx"]]).max())
    a1, _ = masked_argmax(z["s_e9"], d["reach"], d["reach"])
    c["C1_e9_argmax_mismatch"] = int((a1 != z["sel_idx"]).sum())
    a2, _ = masked_argmax(z["s_core"], d["reach"] & d["ceil"], d["reach"])
    c["C1_core_argmax_mismatch"] = int((a2 != z["sel_idx_base"]).sum())
    c["C1_core_score_recon_max_abs"] = float(np.abs(z["rank_res"] - z["s_core"]).max())
    c["C1_rank_sum_recon_max_abs"] = float(np.abs(z["rank_base"] + z["rank_terms"].sum(1) - z["s_core"]).max())
    c["C1_rank_base_is_refined_max_abs"] = float(np.abs(z["rank_base"] - z["refined_res"]).max())
    c["C1_e9_blend_recon_max_abs"] = float(np.abs(e9_blend(z["s_core"], z["e9_graft"]) - z["s_e9"]).max())
    # C2: my predicate == the model's own nav_compliance_prior on informative-nav windows
    inf = d["nav_side"] != 0
    mine = (d["dir_c"] == d["nav_side"][:, None]).astype(np.float32)
    n_inf_cand = int(inf.sum()) * z["fan"].shape[1]
    c["C2_n_informative_windows"] = int(inf.sum())
    c["C2_predicate_mismatch"] = int((mine[inf] != z["navc"][inf]).sum())
    c["C2_n_candidates_compared"] = n_inf_cand
    c["C2_uninformative_term_nonzero"] = int((z["navc"][~inf] != 0).sum())
    gate = z["_json"]["gates"]["navc_gate"]
    c["C2_navc_term_eq_gate_x_pred_max_abs"] = float(np.abs(d["navc_term"] - gate * z["navc"]).max())
    return c


# ------------------------------------------------------------------------------------------------- #
# the chain                                                                                          #
# ------------------------------------------------------------------------------------------------- #
def cls_mask(d, name):
    if name == "turn":
        return np.isin(d["cls"], ["turnL", "turnR"])
    if name == "all":
        return d["target"] != 9
    return d["cls"] == name


def link_indicators(z, d):
    """{link: (correct [W] float, counted [W] bool)} against d['target']; all on the SAME windows."""
    W, t = d["W"], d["target"]
    ar = np.arange(W)
    ok = t != 9
    hag = np.radians(rm.HEAD_AGREE_DEG)
    corr_c = d["dir_c"] == t[:, None]                                # [W, N]
    head_c = np.abs(rm.wrap(d["th_c"] - d["th_gt"][:, None])) <= hag
    L = {}
    L["N_nav"] = (d["nav_side"] == t, ok)
    L["N_nav_informative_only"] = (d["nav_side"] == t, ok & (d["nav_side"] != 0))
    L["T_tac_side"] = (d["tac_side"] == t, ok)
    L["T_tac_strict"] = (d["tac_strict"] == t, ok)
    L["T_label_lat_v7_side"] = (d["label_side"] == t, ok & (d["label_side"] != 9))
    L["G_goal_side"] = (d["goal_side"] == t, ok)
    L["C_fan_all117"] = (corr_c.any(1), ok)
    L["C_fan_reach"] = ((corr_c & d["reach"]).any(1), ok)
    L["C_fan_reach_ceil"] = ((corr_c & d["reach"] & d["ceil"]).any(1), ok)
    L["C_oracle_cand_correct"] = (corr_c[ar, d["oracle"]], ok)
    L["K_core_pick"] = (corr_c[ar, z["sel_idx_base"]], ok)
    L["E_e9_pick"] = (corr_c[ar, z["sel_idx"]], ok)
    L["E_e9_pick_26p1fix"] = (corr_c[ar, d["pick_fix"]], ok)
    L["R_random_reach"] = ((corr_c & d["reach"]).sum(1) / np.maximum(d["reach"].sum(1), 1), ok)
    # heading-agree (secondary)
    L["C_fan_reach_headagree"] = ((head_c & d["reach"]).any(1), ok)
    L["K_core_pick_headagree"] = (head_c[ar, z["sel_idx_base"]], ok)
    L["E_e9_pick_headagree"] = (head_c[ar, z["sel_idx"]], ok)
    L["C_oracle_headagree"] = (head_c[ar, d["oracle"]], ok)
    L["R_random_reach_headagree"] = ((head_c & d["reach"]).sum(1) / np.maximum(d["reach"].sum(1), 1), ok)
    return {k: (np.asarray(v, float), np.asarray(m, bool)) for k, (v, m) in L.items()}


def chain_table(z, d, draws):
    L = link_indicators(z, d)
    out = {}
    for cn in CLASSES:
        m = cls_mask(d, cn)
        row = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(d["ep"][m]))) if m.any() else 0}
        for k, (v, cnt) in L.items():
            mm = m & cnt
            pt, bs, _ = rm.boot_ratio(np.where(mm, v, 0), mm.astype(float), d["ep"], draws=draws)
            row[k] = {"rate": None if not mm.any() else round(float(pt), 4), "ci95": rm.ci95(bs),
                      "n": int(mm.sum())}
        out[cn] = row
    # oracle ADE beside the picks (turn / straight)
    ar = np.arange(d["W"])
    for cn in CLASSES:
        m = cls_mask(d, cn)
        for nm, idx in (("oracle117", d["oracle"]), ("core", z["sel_idx_base"]), ("e9", z["sel_idx"])):
            a = d["ade_c"][ar, idx]
            pt, bs, _ = rm.boot_mean(a, d["ep"], m, draws=draws)
            out[cn][f"ade_{nm}"] = {"mean": None if not m.any() else round(float(pt), 4), "ci95": rm.ci95(bs)}
    return out, L


def decision(L, d, draws):
    m = cls_mask(d, "turn")
    rates = {}
    bss = {}
    for k in ("T_tac_side", "C_fan_reach", "K_core_pick", "E_e9_pick", "N_nav", "G_goal_side"):
        v, cnt = L[k]
        mm = m & cnt
        pt, bs, _ = rm.boot_ratio(np.where(mm, v, 0), mm.astype(float), d["ep"], draws=draws)
        rates[k], bss[k] = float(pt), bs
    drops = {"D_gen (T -> C)": ("T_tac_side", "C_fan_reach"),
             "D_core (C -> K)": ("C_fan_reach", "K_core_pick"),
             "D_E9 (K -> E)": ("K_core_pick", "E_e9_pick")}
    res = {"rates_turn": {k: round(v, 4) for k, v in rates.items()}, "drops": {}}
    for nm, (a, b) in drops.items():
        pt = rates[a] - rates[b]
        bs = bss[a] - bss[b]
        lo, hi = rm.ci95(bs)
        res["drops"][nm] = {"drop": round(pt, 4), "ci95": [lo, hi], "excludes_0": bool(lo is not None and
                                                                                      (lo > 0 or hi < 0))}
    sig = [(v["drop"], k) for k, v in res["drops"].items() if v["excludes_0"] and v["drop"] > 0]
    sig.sort(reverse=True)
    if not sig:
        res["break"] = "no single break (no positive drop with a CI excluding 0)"
    else:
        lead = sig[0]
        names = [lead[1]]
        for v, k in sig[1:]:
            lo, hi = res["drops"][k]["ci95"]
            if hi >= lead[0]:
                names.append(k)
        res["break"] = names
    return res


def confusion(z, d):
    """3 x 3 per link: rows = GT target (L / S / R over turn + straight windows), cols = link side."""
    W = d["W"]
    ar = np.arange(W)
    m = np.isin(d["cls"], ["turnL", "turnR", "straight"])
    sides = {"nav": d["nav_side"], "tac_side": d["tac_side"], "tac_strict": d["tac_strict"],
             "goal": d["goal_side"], "label_lat_v7": d["label_side"],
             "oracle117": d["dir_c"][ar, d["oracle"]], "core_pick": d["dir_c"][ar, z["sel_idx_base"]],
             "e9_pick": d["dir_c"][ar, z["sel_idx"]], "e9_pick_26p1fix": d["dir_c"][ar, d["pick_fix"]]}
    out = {}
    for k, s in sides.items():
        tab = {}
        for tn, tv in (("GT_L", 1), ("GT_S", 0), ("GT_R", -1)):
            r = m & (d["target"] == tv)
            tab[tn] = {"L": int((r & (s == 1)).sum()), "S": int((r & (s == 0)).sum()),
                       "R": int((r & (s == -1)).sum()), "n/a": int((r & (s == 9)).sum())}
        out[k] = tab
    return out


def navc_and_e9(z, d, draws):
    """link (e) and (f)."""
    W = d["W"]
    ar = np.arange(W)
    inf = d["nav_side"] != 0
    out = {"navc_gate": z["_json"]["gates"]["navc_gate"], "goal_gate_e9": z["_json"]["gates"]["goal_gate_e9"]}
    comp = z["navc"] > 0.5
    reach = d["reach"]
    sd_core = np.array([np.std(z["s_core"][w][reach[w]]) if reach[w].sum() > 1 else np.nan for w in range(W)])
    gap_core = np.array([np.sort(z["s_core"][w][reach[w]])[-1] - np.sort(z["s_core"][w][reach[w]])[-2]
                         if reach[w].sum() > 1 else np.nan for w in range(W)])
    for cn in ("turn", "straight", "gentle", "all"):
        m = cls_mask(d, cn) & inf
        if not m.any():
            continue
        r = {"n_informative_windows": int(m.sum())}
        nk = (comp & reach).sum(1)
        r["n_complying_in_reach_median"] = float(np.median(nk[m]))
        r["n_complying_in_reach_mean"] = round(float(nk[m].mean()), 2)
        r["n_reach_median"] = float(np.median(reach.sum(1)[m]))
        r["frac_windows_zero_complying"] = round(float((nk[m] == 0).mean()), 4)
        r["oracle_complies_frac"] = round(float(comp[ar, d["oracle"]][m].mean()), 4)
        agree = m & (d["nav_side"] == d["target"])
        r["n_nav_side_eq_gt"] = int(agree.sum())
        if agree.any():
            r["oracle_complies_frac_when_nav_eq_gt"] = round(float(comp[ar, d["oracle"]][agree].mean()), 4)
            r["e9_pick_complies_frac_when_nav_eq_gt"] = round(float(comp[ar, z["sel_idx"]][agree].mean()), 4)
            r["core_pick_complies_frac_when_nav_eq_gt"] = round(float(comp[ar, z["sel_idx_base"]][agree].mean()), 4)
        r["navc_gate_over_score_std_median"] = round(float(np.nanmedian(out["navc_gate"] / sd_core[m])), 4)
        r["score_top1_top2_gap_median"] = round(float(np.nanmedian(gap_core[m])), 4)
        # counterfactual: remove the navc term (decoder argmax and E9 argmax)
        s_core_wo = z["s_core"] - d["navc_term"]
        k_wo, _ = masked_argmax(s_core_wo, reach & d["ceil"], reach)
        e_wo, _ = masked_argmax(e9_blend(s_core_wo, z["e9_graft"]), reach, reach)
        r["core_pick_changes_without_navc"] = int((k_wo != z["sel_idx_base"])[m].sum())
        r["e9_pick_changes_without_navc"] = int((e_wo != z["sel_idx"])[m].sum())
        out[cn] = r
    # (f) E9 score of the GT-like candidate vs the pick, within reach (GT-turn windows and straight)
    s = np.where(reach, z["s_e9"], -np.inf)
    rank_of = lambda w, i: int((s[w] > s[w, i]).sum()) + 1                           # noqa: E731
    corr_c = d["dir_c"] == d["target"][:, None]
    best_corr = np.argmin(np.where(corr_c & reach & np.isfinite(d["ade_c"]), d["ade_c"], np.inf), axis=1)
    has_corr = (corr_c & reach).any(1)
    f = {}
    for cn in ("turn", "straight", "all"):
        m = cls_mask(d, cn)
        if not m.any():
            continue
        ws = np.nonzero(m)[0]
        ro = np.array([rank_of(w, d["oracle_reach"][w]) for w in ws])
        rb = np.array([rank_of(w, best_corr[w]) for w in ws if has_corr[w]])
        gap = np.array([z["s_e9"][w, z["sel_idx"][w]] - z["s_e9"][w, d["oracle_reach"][w]] for w in ws])
        f[cn] = {"n": int(m.sum()), "oracle_reach_rank_median": float(np.median(ro)),
                 "oracle_reach_rank_p90": float(np.percentile(ro, 90)),
                 "oracle_is_pick_frac": round(float((ro == 1).mean()), 4),
                 "oracle_in_top8_frac": round(float((ro <= 8).mean()), 4),
                 "best_correct_rank_median": float(np.median(rb)) if rb.size else None,
                 "best_correct_in_top8_frac": round(float((rb <= 8).mean()), 4) if rb.size else None,
                 "score_gap_pick_minus_oracle_median": round(float(np.median(gap)), 4),
                 "n_reach_median": float(np.median(reach.sum(1)[m]))}
        # which term prefers the pick over the oracle (all in score units)
        terms = {"sampler_conf": z["refined_base"], "tac8_lat": z["tac8_lat"], "tac8_lon": z["tac8_lon"],
                 "behaviour": d["beh_term"], "navc": d["navc_term"],
                 "e9_goal_graft": z["s_e9"] - z["s_core"]}
        dd = {}
        for tn, tv in terms.items():
            dv = np.array([tv[w, z["sel_idx"][w]] - tv[w, d["oracle_reach"][w]] for w in ws])
            notsame = np.array([z["sel_idx"][w] != d["oracle_reach"][w] for w in ws])
            dd[tn] = {"median_delta_pick_minus_oracle": round(float(np.median(dv[notsame])), 4)
                      if notsame.any() else None,
                      "frac_favours_pick": round(float((dv[notsame] > 0).mean()), 4) if notsame.any() else None}
        f[cn]["term_deltas_when_pick_ne_oracle"] = dd
        f[cn]["n_pick_ne_oracle"] = int(np.sum([z["sel_idx"][w] != d["oracle_reach"][w] for w in ws]))
    out["e9_score_of_gt_like"] = f
    return out


# ------------------------------------------------------------------------------------------------- #
# levers                                                                                             #
# ------------------------------------------------------------------------------------------------- #
def pside(z, d):
    """p_side(c) = sum of p_lat over the classes whose side equals the candidate's direction."""
    P = z["p_lat"]
    pl = P[:, rm.LAT_SIDE == 1].sum(1)
    pr = P[:, rm.LAT_SIDE == -1].sum(1)
    ps = P[:, rm.LAT_SIDE == 0].sum(1)
    dc = d["dir_c"]
    return np.where(dc == 1, pl[:, None], np.where(dc == -1, pr[:, None], ps[:, None]))


def pick(rule, prm, z, d):
    reach = d["reach"]
    if rule == "V0":
        return z["sel_idx"].copy()
    if rule == "V0c":
        return d["pick_fix"].copy()
    if rule == "V5":
        return z["sel_idx_base"].copy()
    if rule in ("V1a", "V1b"):
        tau, pmin = prm
        dc = rm.dir_class(d["th_c"], tau)
        pmax = z["p_lat"].max(1)
        act = (pmax >= pmin) & ((d["tac_side"] != 0) if rule == "V1a" else True)
        keep = reach & np.where(act[:, None], dc == d["tac_side"][:, None], True)
        return masked_argmax(z["s_e9"], keep, reach)[0]
    if rule == "V2":
        tau = prm
        dc = rm.dir_class(d["th_c"], tau)
        act = d["nav_side"] != 0
        keep = reach & np.where(act[:, None], dc == d["nav_side"][:, None], True)
        return masked_argmax(z["s_e9"], keep, reach)[0]
    if rule == "V3":
        k = prm
        s = e9_blend(z["s_core"] + (k - 1) * d["navc_term"], z["e9_graft"])
        return masked_argmax(s, reach, reach)[0]
    if rule == "V4":
        beta = prm
        s = z["s_e9"] + beta * pside(z, d)
        return masked_argmax(s, reach, reach)[0]
    if rule == "ORACLE":
        return d["oracle"].copy()
    # ---- SPEC_ADDENDUM_A4: label-side attribution bounds (NOT levers) ----
    if rule in ("B1", "B1t", "B2", "B3", "B4"):
        t = d["target"][:, None]
        ok = (d["target"] != 9)[:, None]
        r_dir = (d["dir_c"] == t) | ~ok
        r_head = (np.abs(rm.wrap(d["th_c"] - d["th_gt"][:, None])) <= np.radians(rm.HEAD_AGREE_DEG)) | ~ok
        L6c = rm.path_length(z["fan"])
        L6g = rm.path_length(z["gt"])[:, None]
        gv = z["gt_valid"][:, -1][:, None] & (L6g > 0.5)
        r_spd = (np.abs(L6c / np.maximum(L6g, 1e-6) - 1.0) <= 0.10) | ~gv
        if rule == "B1":
            keep = r_dir
        elif rule == "B1t":
            keep = r_dir | ~np.isin(d["cls"], ["turnL", "turnR"])[:, None]
        elif rule == "B2":
            keep = r_head
        elif rule == "B3":
            keep = r_spd
        else:
            keep = r_head & r_spd
        return masked_argmax(z["s_e9"], reach & keep, reach)[0]
    # ---- SPEC_ADDENDUM_A1 (post-hoc, pre-registered before computed) ----
    if rule == "W1":
        T, k = prm
        return mbr_pick(z["s_e9"], reach, pair_ade(z, d), T, k)
    if rule == "W2":
        lam = prm
        return masked_argmax(w2_score(z, lam), reach, reach)[0]
    if rule == "W3":
        k = prm
        return masked_argmax(e9_blend(z["s_core"], k * z["e9_graft"]), reach, reach)[0]
    if rule == "W4":
        (T, k), lam = prm
        return mbr_pick(w2_score(z, lam), reach, pair_ade(z, d), T, k)
    raise ValueError(rule)


def pair_ade(z, d):
    """[W, N, N] mean-over-8-slots L2 between every two candidate paths (cached on d)."""
    if "_pair" not in d:
        F = z["fan"].astype(np.float32)
        d["_pair"] = np.stack([np.linalg.norm(f[:, None] - f[None], axis=-1).mean(-1) for f in F])
    return d["_pair"]


def mbr_pick(score, keep, M, T, k):
    W, N = score.shape
    out = np.zeros(W, np.int64)
    for w in range(W):
        cand = np.nonzero(keep[w])[0]
        if cand.size == 0:
            cand = np.arange(N)
        top = cand[np.argsort(-score[w, cand], kind="mergesort")[:int(k)]]
        s = score[w, top] / float(T)
        p = np.exp(s - s.max())
        p /= p.sum()
        risk = M[w][np.ix_(top, top)] @ p
        out[w] = top[int(np.argmin(risk))]
    return out


def w2_score(z, lam):
    g = z["g_tac"]
    if g.ndim != 3 or not np.isfinite(g).all():
        raise SystemExit("[route] W2 needs the g_tac capture (eval_s0g / train_s0g)")
    F = z["fan"]
    dist = (np.linalg.norm(F[:, :, 5, :] - g[:, None, 1, :2], axis=-1)
            + np.linalg.norm(F[:, :, 7, :] - g[:, None, 2, :2], axis=-1))
    return z["s_e9"] - lam * dist


A1_T = (0.05, 0.1, 0.2, 0.3, 0.5, 1, 2)
A1_K = (8, 16, 117)
A1_LAM = (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2)
A1_KS = (0, 0.5, 2, 5, 10)
GRIDS_A1 = {"W1": [(t, k) for t in A1_T for k in A1_K], "W2": list(A1_LAM), "W3": list(A1_KS),
            "W4": [((t, k), lam) for t in A1_T for k in A1_K for lam in A1_LAM]}
GRIDS = {"V1a": [(t, p) for t in TAU_GRID for p in PMIN_GRID], "V1b": [(t, p) for t in TAU_GRID for p in PMIN_GRID],
         "V2": list(TAU_GRID), "V3": list(K_GRID), "V4": list(BETA_GRID)}


def fit_levers(zt, dt, grids=None):
    grids = GRIDS if grids is None else grids
    ar = np.arange(dt["W"])
    base = np.nanmean(dt["ade_c"][ar, zt["sel_idx"]])
    fit = {"train_V0_ade": round(float(base), 4), "rules": {}}
    for rule, grid in grids.items():
        res = []
        for prm in grid:
            idx = pick(rule, prm, zt, dt)
            res.append((float(np.nanmean(dt["ade_c"][ar, idx])), prm))
        best = min(res, key=lambda r: (r[0], str(r[1])))
        fit["rules"][rule] = {"best_param": best[1], "train_ade": round(best[0], 4),
                              "grid": [[str(p), round(a, 4)] for a, p in res]}
    chosen = min(fit["rules"], key=lambda r: fit["rules"][r]["train_ade"])
    fit["chosen_rule"] = chosen
    fit["chosen_param"] = fit["rules"][chosen]["best_param"]
    return fit


def seq_geom(wp4, dt=0.5):
    """four_families._seq_geometry on the 0-2 s slots (imported from the launch tree)."""
    import torch
    from taniteval import four_families as ff
    return {k: (v.numpy() if hasattr(v, "numpy") else v)
            for k, v in ff._seq_geometry(torch.as_tensor(wp4, dtype=torch.float64), dt=dt).items()}


def plan_metrics(P, z, d):
    """per-window metrics of an emitted plan P [W, 8, 2] (SPEC sec. 5, four families)."""
    gt, gv = z["gt"].astype(np.float64), z["gt_valid"]
    ade, fde, dist = rm.ade_fde(P, gt, gv)
    m = {"ade": ade, "fde": fde}
    for nm, si in (("2s", 3), ("6s", 7)):
        v = gv[:, si]
        m[f"along_signed_{nm}"] = np.where(v, P[:, si, 0] - gt[:, si, 0], np.nan)
        m[f"along_abs_{nm}"] = np.abs(m[f"along_signed_{nm}"])
        m[f"cross_abs_{nm}"] = np.where(v, np.abs(P[:, si, 1] - gt[:, si, 1]), np.nan)
    thp = rm.terminal_heading(P)
    m["term_heading_err_deg"] = np.where(gv[:, -1], np.degrees(np.abs(rm.wrap(thp - d["th_gt"]))), np.nan)
    gp, gg = seq_geom(P[:, :4]), seq_geom(gt[:, :4])
    v4 = gv[:, :4]
    sp = np.abs(gp["speed"] - gg["speed"])
    m["speed_mae_0_2s"] = np.where(v4.any(1), (sp * v4).sum(1) / np.maximum(v4.sum(1), 1), np.nan)
    hv = gp["valid"] & gg["valid"] & v4
    dh = np.abs(rm.wrap(gp["heading"] - gg["heading"]))
    m["heading_mae_0_2s_deg"] = np.where(hv.any(1), np.degrees((dh * hv).sum(1) / np.maximum(hv.sum(1), 1)), np.nan)
    pv = gp["pair_valid"] & gg["pair_valid"] & v4[:, 1:] & v4[:, :-1]
    dk = np.abs(gp["curvature"] - gg["curvature"])
    m["curv_mae_0_2s"] = np.where(pv.any(1), (dk * pv).sum(1) / np.maximum(pv.sum(1), 1), np.nan)
    # POST-HOC (not in SPEC): signed turn ratio theta_plan / theta_gt on GT-turn windows (1 = exact, <1 under-turn)
    is_turn = np.isin(d["cls"], ["turnL", "turnR"])
    m["turn_ratio_posthoc"] = np.where(is_turn, thp * np.sign(d["th_gt"]) / np.maximum(np.abs(d["th_gt"]), 1e-9),
                                       np.nan)
    dp = rm.dir_class(thp)
    m["dir_correct"] = np.where(d["target"] != 9, (dp == d["target"]).astype(float), np.nan)
    comp = (dp == d["nav_side"]) & (d["nav_side"] != 0)
    m["nav_complies"] = np.where(d["nav_side"] != 0, comp.astype(float), np.nan)
    return m


def tactical_ff(P, z):
    import torch
    from taniteval import four_families as ff
    r = ff.tactical_from_trajectory(torch.as_tensor(P[:, :4], dtype=torch.float64),
                                    torch.as_tensor(z["gt"][:, :4], dtype=torch.float64), dt=0.5, n_boot=200)
    keep = {}
    for k, v in r.items():
        if isinstance(v, (int, float, str)):
            keep[k] = v
        elif isinstance(v, dict):
            keep[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, type(None)))}
    return keep


def lever_table(z, d, fit, draws, grids=None):
    grids = GRIDS if grids is None else grids
    ar = np.arange(d["W"])
    arms = [("V0", None), ("V0c", None), ("V5", None), (fit["chosen_rule"], fit["chosen_param"])]
    for r in grids:
        if r != fit["chosen_rule"]:
            arms.append((r, fit["rules"][r]["best_param"]))
    arms.append(("ORACLE", None))
    mets = {}
    for rule, prm in arms:
        idx = pick(rule, prm, z, d)
        P = z["fan"][ar, idx].astype(np.float64)
        mets[(rule, str(prm))] = (idx, plan_metrics(P, z, d))
    base_idx, base = mets[("V0", "None")]
    out = {}
    for (rule, prm), (idx, m) in mets.items():
        row = {"rule": rule, "param": prm, "pick_changed_frac": round(float((idx != base_idx).mean()), 4)}
        for cn in ("turn", "straight", "gentle", "all"):
            msk = cls_mask(d, cn)
            blk = {"n": int(msk.sum())}
            for k, v in m.items():
                pt, bs, _ = rm.boot_mean(v, d["ep"], msk, draws=draws)
                ent = {"mean": None if not np.isfinite(pt) else round(float(pt), 4), "ci95": rm.ci95(bs)}
                if rule != "V0":
                    dv = v - base[k]
                    pt2, bs2, _ = rm.boot_mean(dv, d["ep"], msk, draws=draws)
                    ent["delta_vs_V0"] = None if not np.isfinite(pt2) else round(float(pt2), 4)
                    ent["delta_ci95"] = rm.ci95(bs2)
                blk[k] = ent
            row[cn] = blk
        out[f"{rule}|{prm}"] = row
    return out, mets


def _lt(x, v):
    return x is not None and v is not None and x < v


def _le(x, v):
    return x is not None and v is not None and x <= v


def bar_check(lt, chosen_key, lt_s1=None):
    r = lt[chosen_key]
    c1 = (_lt(r["turn"]["ade"]["delta_ci95"][1], 0) and _lt(0, r["turn"]["dir_correct"]["delta_ci95"][0]))
    st = r["straight"]["ade"]
    c2 = _le(st["delta_vs_V0"], 0.05) and _le(st["delta_ci95"][1], 0.10)
    c3 = _le(r["all"]["ade"]["delta_vs_V0"], 0)
    c4 = None
    if lt_s1 is not None:
        r1 = lt_s1[chosen_key]
        c4 = bool(_lt(r1["turn"]["ade"]["delta_ci95"][1], 0) and _lt(0, r1["turn"]["dir_correct"]["delta_ci95"][0]))
    return {"1_turn_ade_and_dircorrect": bool(c1), "2_straight_no_regression": bool(c2),
            "3_all_ade_not_worse": bool(c3), "4_replicates_seed1": c4,
            "CLEARS": bool(c1 and c2 and c3 and (c4 is not False))}


# ------------------------------------------------------------------------------------------------- #
# fan incoherence                                                                                    #
# ------------------------------------------------------------------------------------------------- #
def fan_stats(z, d, draws):
    W = d["W"]
    reach = d["reach"]
    rng = np.random.default_rng(12345)
    rnd = rng.standard_normal(z["s_e9"].shape)
    per = {k: np.full(W, np.nan) for k in ("e9_top8_circstd_deg", "e9_top8_range_deg", "e9_top8_LR",
                                           "core_top8_circstd_deg", "core_top8_range_deg", "core_top8_LR",
                                           "rho_e9", "rho_core", "rho_random", "rho_sampler_conf",
                                           "top8_has_correct", "n_reach")}
    for w in range(W):
        cand = np.nonzero(reach[w] & np.isfinite(d["ade_c"][w]))[0]
        per["n_reach"][w] = len(cand)
        if len(cand) < 3:
            continue
        for nm, s in (("e9", z["s_e9"][w]), ("core", z["s_core"][w])):
            top = cand[np.argsort(-s[cand], kind="mergesort")[:8]]
            th = d["th_c"][w, top]
            per[f"{nm}_top8_circstd_deg"][w] = math.degrees(rm.circ_std(th))
            per[f"{nm}_top8_range_deg"][w] = math.degrees(th.max() - th.min())
            dc = d["dir_c"][w, top]
            per[f"{nm}_top8_LR"][w] = float((dc == 1).any() and (dc == -1).any())
            if nm == "e9" and d["target"][w] != 9:
                per["top8_has_correct"][w] = float((dc == d["target"][w]).any())
        q = -d["ade_c"][w, cand]
        per["rho_e9"][w] = rm.spearman(z["s_e9"][w, cand], q)
        per["rho_core"][w] = rm.spearman(z["s_core"][w, cand], q)
        per["rho_sampler_conf"][w] = rm.spearman(z["refined_base"][w, cand], q)
        per["rho_random"][w] = rm.spearman(rnd[w, cand], q)
    out = {}
    for cn in ("turn", "straight", "gentle", "all"):
        m = cls_mask(d, cn)
        blk = {"n": int(m.sum())}
        for k, v in per.items():
            pt, bs, _ = rm.boot_mean(v, d["ep"], m, draws=draws)
            blk[k] = {"mean": None if not np.isfinite(pt) else round(float(pt), 4), "ci95": rm.ci95(bs)}
        out[cn] = blk
    return out, per


# ------------------------------------------------------------------------------------------------- #
# reel                                                                                               #
# ------------------------------------------------------------------------------------------------- #
def reel_bank(path):
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    gt = np.array([r["gt_slots"] for r in rows], np.float64)
    gv = np.array([r["slot_valid"] for r in rows], bool)
    traj = np.array([r["traj"] for r in rows], np.float64)
    cls, th = rm.gt_class(gt, gv)
    tgt = np.where(np.isin(cls, ["turnL", "turnR", "straight", "gentle"]), rm.dir_class(th), 9)
    lat_side = rm.LAT_SIDE[np.array([r["tac_lat_pred"] for r in rows])]
    nav = np.array([{"follow": 0, "left": 1, "right": -1}[r["nav"]] for r in rows])
    gs = []
    for r in rows:
        p = {i: pv for i, pv in r["goal_top"]}
        pl, pr = p.get(1, 0.0), p.get(2, 0.0)
        gs.append(1 if (max(pl, pr) >= 0.5 and pl >= pr) else (-1 if max(pl, pr) >= 0.5 else 0))
    gs = np.array(gs)
    e9 = rm.dir_class(rm.terminal_heading(traj))
    fan = [np.array(r["fan"], np.float64) for r in rows]
    lr = np.array([float(((rm.dir_class(rm.terminal_heading(f)) == 1).any() or e9[i] == 1)
                         and ((rm.dir_class(rm.terminal_heading(f)) == -1).any() or e9[i] == -1))
                   for i, f in enumerate(fan)])
    ep = np.array([r["clip_sha12"] for r in rows])
    return {"rows": rows, "cls": cls, "th": th, "target": tgt, "tac_side": lat_side, "nav_side": nav,
            "goal_side": gs, "e9_dir": e9, "top9_LR": lr, "ep": ep, "traj": traj,
            "ade": np.array([r["ade_m"] if r["ade_m"] is not None else np.nan for r in rows], float)}


def reel_table(rb, draws):
    out = {}
    for cn in ("turnL", "turnR", "turn", "straight", "gentle", "all"):
        if cn == "turn":
            m = np.isin(rb["cls"], ["turnL", "turnR"])
        elif cn == "all":
            m = rb["target"] != 9
        else:
            m = rb["cls"] == cn
        blk = {"n": int(m.sum()), "n_clips": int(len(np.unique(rb["ep"][m]))) if m.any() else 0}
        for k, s in (("N_nav", rb["nav_side"]), ("T_tac_side", rb["tac_side"]), ("G_goal_side", rb["goal_side"]),
                     ("E_e9_pick", rb["e9_dir"])):
            v = (s == rb["target"]).astype(float)
            pt, bs, _ = rm.boot_ratio(np.where(m, v, 0), m.astype(float), rb["ep"], draws=draws)
            blk[k] = {"rate": None if not m.any() else round(float(pt), 4), "ci95": rm.ci95(bs)}
        pt, bs, _ = rm.boot_mean(rb["top9_LR"], rb["ep"], m, draws=draws)
        blk["top9_has_L_and_R"] = {"rate": None if not m.any() else round(float(pt), 4), "ci95": rm.ci95(bs)}
        pt, bs, _ = rm.boot_mean(rb["ade"], rb["ep"], m, draws=draws)
        blk["ade_e9"] = {"mean": None if not m.any() else round(float(pt), 4), "ci95": rm.ci95(bs)}
        out[cn] = blk
    return out


def reel_identity(zr, rb):
    """C6: the re-forward reproduces the bank (traj to the bank's 3 dp, sel_idx exactly)."""
    tb = rb["traj"]
    if zr["traj"].shape != tb.shape:
        return {"C6": "FAIL shape", "shape_reforward": list(zr["traj"].shape), "shape_bank": list(tb.shape)}
    dmax = float(np.abs(zr["traj"] - tb).max())
    sel_b = np.array([r["sel_idx"] for r in rb["rows"]])
    core_b = np.array([r["core_sel_idx"] for r in rb["rows"]])
    return {"traj_max_abs_diff_m": dmax, "sel_idx_mismatch": int((zr["sel_idx"] != sel_b).sum()),
            "core_sel_idx_mismatch": int((zr["sel_idx_base"] != core_b).sum()), "n": int(len(sel_b)),
            "C6": "PASS" if (dmax <= 1e-3 and (zr["sel_idx"] == sel_b).all()) else "FAIL"}


def reel_timeline(zr, dr, clip, lo, hi):
    """per-window timeline of one reel clip (sha12) between window positions lo..hi."""
    ws = np.nonzero(np.asarray(zr["win_sha12"]).astype(str) == clip)[0]
    out = []
    ar = np.arange(dr["W"])
    for k, w in enumerate(ws):
        if not (lo <= k <= hi):
            continue
        corr = dr["dir_c"][w] == dr["target"][w]
        out.append({"win": k, "gt_cls": str(dr["cls"][w]), "th_gt_deg": round(math.degrees(dr["th_gt"][w]), 1),
                    "nav": int(dr["nav_side"][w]),
                    "tac": rm.LAT_NAMES[int(dr["lat_arg"][w])], "tac_p": round(float(zr["p_lat"][w].max()), 3),
                    "goal_side": int(dr["goal_side"][w]),
                    "n_reach": int(dr["reach"][w].sum()), "n_correct_in_reach": int((corr & dr["reach"][w]).sum()),
                    "th_e9_deg": round(math.degrees(dr["th_c"][w, zr["sel_idx"][w]]), 1),
                    "th_core_deg": round(math.degrees(dr["th_c"][w, zr["sel_idx_base"][w]]), 1),
                    "th_oracle_deg": round(math.degrees(dr["th_c"][w, dr["oracle"][w]]), 1),
                    "th_prior_deg": (round(math.degrees(float(rm.terminal_heading(zr["prior_path"][w]))), 1)
                                     if "prior_path" in zr else None),
                    "v0": round(float(zr["v0"][w]), 2),
                    "ade_e9": round(float(dr["ade_c"][w, zr["sel_idx"][w]]), 2),
                    "ade_oracle": round(float(dr["ade_c"][w, dr["oracle"][w]]), 2),
                    "oracle_rank_e9": int((np.where(dr["reach"][w], zr["s_e9"][w], -np.inf)
                                           > zr["s_e9"][w, dr["oracle_reach"][w]]).sum()) + 1})
    return out


def along_cross_table(z, d, picks):
    """slot-mean |error| decomposed along / across the GT path's local tangent (post-hoc attribution)."""
    ar = np.arange(d["W"])
    gt = z["gt"].astype(np.float64)
    G = np.concatenate([np.zeros((d["W"], 1, 2)), gt], 1)
    tan = np.diff(G, axis=1)
    tu = tan / np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-6)
    v = z["gt_valid"]
    out = {}
    for name, idx in picks.items():
        e = z["fan"][ar, idx].astype(np.float64) - gt
        al = (e * tu).sum(-1)
        cr = e[..., 0] * (-tu[..., 1]) + e[..., 1] * tu[..., 0]
        blk = {}
        for cn in ("turn", "straight", "gentle", "all"):
            m = cls_mask(d, cn)
            vv = v & m[:, None]
            blk[cn] = {"abs_along_m": round(float(np.abs(al)[vv].mean()), 3),
                       "signed_along_m": round(float(al[vv].mean()), 3),
                       "abs_cross_m": round(float(np.abs(cr)[vv].mean()), 3)}
        out[name] = blk
    return out


def prior_diag(z, d):
    """POST-HOC (not pre-registered): does the emitted pick hug the residual prior (ha0_ext_pose, a hold-action
    extrapolation of the current state) more than the GT-like candidate does? Reported as an attribution diagnostic
    only -- never as a lever result."""
    if "prior_path" not in z or not np.isfinite(z["prior_path"]).all():
        return None
    ar = np.arange(d["W"])
    Pp = z["prior_path"].astype(np.float64)
    thpr = rm.terminal_heading(Pp)
    dpr = rm.dir_class(thpr)
    pick = z["fan"][ar, z["sel_idx"]].astype(np.float64)
    orc = z["fan"][ar, d["oracle"]].astype(np.float64)
    a_pick = np.linalg.norm(pick - Pp, axis=-1).mean(-1)
    a_orc = np.linalg.norm(orc - Pp, axis=-1).mean(-1)
    thp = d["th_c"][ar, z["sel_idx"]]
    out = {}
    for cn in ("turn", "straight", "gentle", "all"):
        m = cls_mask(d, cn)
        if not m.any():
            continue
        wrong = m & (rm.dir_class(thp) != d["target"])
        out[cn] = {"n": int(m.sum()),
                   "prior_dir_correct_frac": round(float((dpr == d["target"])[m].mean()), 4),
                   "prior_abs_theta_deg_median": round(float(np.degrees(np.median(np.abs(thpr[m])))), 2),
                   "pick_abs_theta_deg_median": round(float(np.degrees(np.median(np.abs(thp[m])))), 2),
                   "gt_abs_theta_deg_median": round(float(np.degrees(np.median(np.abs(d["th_gt"][m])))), 2),
                   "ade8_pick_to_prior_median": round(float(np.median(a_pick[m])), 3),
                   "ade8_oracle_to_prior_median": round(float(np.median(a_orc[m])), 3),
                   "frac_pick_closer_to_prior_than_oracle": round(float((a_pick < a_orc)[m].mean()), 4),
                   "n_pick_dir_wrong": int(wrong.sum()),
                   "frac_wrong_picks_with_prior_dir": (round(float((rm.dir_class(thp) == dpr)[wrong].mean()), 4)
                                                       if wrong.any() else None),
                   "spearman_theta_pick_vs_prior": round(rm.spearman(thp[m], thpr[m]), 4),
                   "spearman_theta_pick_vs_gt": round(rm.spearman(thp[m], d["th_gt"][m]), 4)}
    return out


def jdump(p, obj):
    def cv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        raise TypeError(type(o))
    Path(p).write_text(json.dumps(sanitize(obj), indent=1, default=cv, allow_nan=False), encoding="utf-8")


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default=None)
    ap.add_argument("--diag-eval-digest", default=None)
    ap.add_argument("--diag-train-digest", default=None)
    a = ap.parse_args()
    D = Path(a.data)
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    res = {}
    ze = load(D / "eval_s0.npz")
    de = derive(ze)
    draws = rm.make_draws(de["ep"], B=B, seed=SEED)
    res["controls_eval_s0"] = controls(ze, de)
    res["C4_eval_digest"] = {"mine": ze["_json"]["window_sha12_t_sha256"], "diag": a.diag_eval_digest,
                             "PASS": ze["_json"]["window_sha12_t_sha256"] == a.diag_eval_digest}
    res["capture_identity_eval_s0"] = ze["_json"]["identity"]
    res["gates"] = ze["_json"]["gates"]
    res["class_counts_eval"] = {c: int((de["cls"] == c).sum()) for c in
                                ("turnL", "turnR", "straight", "gentle", "unclassified")}
    ct, L = chain_table(ze, de, draws)
    res["chain_eval_s0"] = ct
    res["decision_eval_s0"] = decision(L, de, draws)
    res["confusion_eval_s0"] = confusion(ze, de)
    res["navc_e9_eval_s0"] = navc_and_e9(ze, de, draws)
    print("[route] chain + decision done", flush=True)
    zs1 = None
    if (D / "eval_s1.npz").exists():
        zs1 = load(D / "eval_s1.npz")
        ds1 = derive(zs1)
        if list(ds1["ep"]) != list(de["ep"]):
            raise SystemExit("eval_s1 windows differ from eval_s0")
        res["controls_eval_s1"] = controls(zs1, ds1)
        ct1, L1 = chain_table(zs1, ds1, draws)
        res["chain_eval_s1"] = ct1
        res["decision_eval_s1"] = decision(L1, ds1, draws)
        ar = np.arange(de["W"])
        res["seed_floor"] = {
            "e9_pick_same_frac": round(float((ze["sel_idx"] == zs1["sel_idx"]).mean()), 4),
            "core_pick_same_frac": round(float((ze["sel_idx_base"] == zs1["sel_idx_base"]).mean()), 4),
            "traj_abs_diff_m_median": round(float(np.median(np.abs(ze["traj"] - zs1["traj"]).max((1, 2)))), 4),
            "traj_abs_diff_m_max": round(float(np.abs(ze["traj"] - zs1["traj"]).max()), 4),
            "ade_e9_s0": round(float(np.nanmean(de["ade_c"][ar, ze["sel_idx"]])), 4),
            "ade_e9_s1": round(float(np.nanmean(ds1["ade_c"][ar, zs1["sel_idx"]])), 4),
            "fan_max_abs_diff_m_median": round(float(np.median(np.abs(ze["fan"] - zs1["fan"]).max((1, 2, 3)))), 4)}
    # levers
    zt = load(D / "train_s0.npz")
    dtr = derive(zt)
    res["controls_train_s0"] = controls(zt, dtr)
    res["C4_train_digest"] = {"mine": zt["_json"]["window_sha12_t_sha256"], "diag": a.diag_train_digest,
                              "PASS": zt["_json"]["window_sha12_t_sha256"] == a.diag_train_digest}
    res["class_counts_train"] = {c: int((dtr["cls"] == c).sum()) for c in
                                 ("turnL", "turnR", "straight", "gentle", "unclassified")}
    trd = rm.make_draws(dtr["ep"], B=B, seed=SEED)
    res["chain_train_s0"] = chain_table(zt, dtr, trd)[0]
    fit = fit_levers(zt, dtr)
    res["lever_fit_train"] = fit
    print(f"[route] lever fit: {fit['chosen_rule']} {fit['chosen_param']}", flush=True)
    lt, mets = lever_table(ze, de, fit, draws)
    res["levers_eval_s0"] = lt
    lt1 = None
    if zs1 is not None:
        lt1, _ = lever_table(zs1, ds1, fit, draws)
        res["levers_eval_s1"] = lt1
    ck = f"{fit['chosen_rule']}|{fit['chosen_param']}"
    res["bar_chosen_lever"] = bar_check(lt, ck, lt1)
    res["bar_all_levers"] = {k: bar_check(lt, k, lt1) for k in lt if k not in ("V0|None", "ORACLE|None")}
    # ---- SPEC_ADDENDUM_A1 levers (post-hoc, pre-registered before computed) ----
    zeg = load(D / "eval_s0g.npz") if (D / "eval_s0g.npz").exists() else (ze if "g_tac" in ze else None)
    ztg = load(D / "train_s0g.npz") if (D / "train_s0g.npz").exists() else (zt if "g_tac" in zt else None)
    a1 = {}
    for nm, zg, zb in (("eval", zeg, ze), ("train", ztg, zt)):
        if zg is not None:
            a1[f"fan_identity_{nm}_s0g_vs_s0"] = {
                "fan_max_abs_diff": float(np.abs(zg["fan"] - zb["fan"]).max()),
                "sel_idx_mismatch": int((zg["sel_idx"] != zb["sel_idx"]).sum()),
                "s_e9_max_abs_diff": float(np.abs(zg["s_e9"] - zb["s_e9"]).max())}
    have_g = zeg is not None and ztg is not None
    grids_a1 = GRIDS_A1 if have_g else {k: GRIDS_A1[k] for k in ("W1", "W3")}
    zA, dA = (zeg, derive(zeg)) if have_g else (ze, de)
    zT, dT = (ztg, derive(ztg)) if have_g else (zt, dtr)
    fitA = fit_levers(zT, dT, grids_a1)
    a1["fit_train"] = fitA
    print(f"[route] A1 fit: {fitA['chosen_rule']} {fitA['chosen_param']}", flush=True)
    ltA, metsA = lever_table(zA, dA, fitA, draws, grids_a1)
    a1["levers_eval_s0"] = ltA
    ltA1 = None
    if zs1 is not None and ("g_tac" in zs1 or not have_g):
        ltA1, _ = lever_table(zs1, ds1, fitA, draws, grids_a1)
        a1["levers_eval_s1"] = ltA1
    ckA = f"{fitA['chosen_rule']}|{fitA['chosen_param']}"
    a1["bar_chosen"] = bar_check(ltA, ckA, ltA1)
    a1["bar_all"] = {k: bar_check(ltA, k, ltA1) for k in ltA if k not in ("V0|None", "ORACLE|None")}
    a1["tactical_from_trajectory_chosen"] = tactical_ff(
        zA["fan"][np.arange(dA["W"]), metsA[(fitA["chosen_rule"], str(fitA["chosen_param"]))][0]].astype(np.float64),
        zA)
    res["A1"] = a1
    # ---- SPEC_ADDENDUM_A4 bounds (label-side, attribution only) ----
    a4 = {}
    rows4 = {}
    for rule in ("V0", "B1", "B1t", "B2", "B3", "B4", "ORACLE"):
        rows4[rule] = pick(rule, None, ze, de)
    import rescorer_a2 as _A2
    a4["eval_s0"] = _A2.table_for(ze, de, rows4, draws)
    a4["along_cross_gt_tangent_eval_s0"] = along_cross_table(ze, de, rows4)
    res["A4_bounds"] = a4
    # tactical family from trajectories (four_families), V0 / chosen / oracle
    tff = {}
    ar = np.arange(de["W"])
    for key in ("V0|None", ck, "ORACLE|None", "V0c|None"):
        rule, prm = key.split("|")
        idx = mets[(rule, prm)][0]
        tff[key] = tactical_ff(ze["fan"][ar, idx].astype(np.float64), ze)
    res["tactical_from_trajectory_eval_s0"] = tff
    print("[route] levers done", flush=True)
    zpr, dpr_ = (zeg, dA) if (zeg is not None and "prior_path" in zeg) else (ze, de)
    res["posthoc_prior_diag_eval_s0"] = prior_diag(zpr, dpr_)
    res["posthoc_prior_diag_train_s0"] = prior_diag(zt, dtr)
    fs, _ = fan_stats(ze, de, draws)
    res["fan_eval_s0"] = fs
    print("[route] fan done", flush=True)
    # reel
    if a.bank:
        rb = reel_bank(a.bank)
        rdr = rm.make_draws(rb["ep"], B=B, seed=SEED)
        res["reel_bank"] = reel_table(rb, rdr)
        if (D / "reel_s0.npz").exists():
            zr = load(D / "reel_s0.npz")
            dr = derive(zr)
            res["C6_reel_identity"] = reel_identity(zr, rb)
            res["controls_reel_s0"] = controls(zr, dr)
            rdr2 = rm.make_draws(dr["ep"], B=B, seed=SEED)
            ctr, Lr = chain_table(zr, dr, rdr2)
            res["chain_reel_s0"] = ctr
            res["decision_reel_s0"] = decision(Lr, dr, rdr2)
            res["navc_e9_reel_s0"] = navc_and_e9(zr, dr, rdr2)
            lr_, _ = lever_table(zr, dr, fit, rdr2)
            res["levers_reel_s0"] = {k: v for k, v in lr_.items()}
            if "g_tac" in zr or not have_g:
                res["A1"]["levers_reel_s0"] = lever_table(zr, dr, fitA, rdr2, grids_a1)[0]
            res["fan_reel_s0"] = fan_stats(zr, dr, rdr2)[0]
            res["posthoc_prior_diag_reel_s0"] = prior_diag(zr, dr)
            res["timeline_a78bf2711918_w55_95"] = reel_timeline(zr, dr, "a78bf2711918", 55, 95)
            res["timeline_f6cfd0a20340_w95_175"] = reel_timeline(zr, dr, "f6cfd0a20340", 95, 175)
    jdump(O / "route_analysis.json", res)
    print("[route] wrote", O / "route_analysis.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
