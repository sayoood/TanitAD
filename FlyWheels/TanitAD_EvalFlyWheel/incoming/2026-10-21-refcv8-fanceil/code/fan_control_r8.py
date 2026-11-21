#!/usr/bin/env python
"""fan_control_r8.py -- D-REFCV8-FANCEIL: the refcv8 mode of fan_control_x8.py (refcv7 X8), on the per-frame battery.

QUESTION (decision-grade for refcv9): is refcv8's 149-candidate fan GOOD and the selector WEAK, or is the fan itself weak?

It IMPORTS fan_control_x8.fan_panel UNCHANGED (sha256 recorded) for the 2 s grid, so the estimator is the refcv7 one:
FULL-SET mean; episode-cluster bootstrap (single quantities) and PAIRED episode-cluster bootstrap (differences),
taniteval/taniteval/ci.py, n_boot=2000, seed 0; random pick = analytic mean over all N candidates + 12 seeded draws.
ADDED (refcv8 mode, same estimator): oracle-in-REACH-KEPT set, the selected candidate's ADE rank, the base(117)/extra(32)
split of the oracle, and the S6 grid (1..6 s, model slots 1,3,4,5,6,7, on the battery's own S6 windows).

CONTROLS that must read known values (all asserted into the JSON, never inferred from an exit code):
  C1 sel_idx agreement with the banked battery's sel_idx_nav_true at the SAME inference seed (bar >= 0.99);
  C2 selected ADE (2 s) from the fan == the battery's banked own-pick (analysis_s<seed>.json os ade_dense_m) within 1e-3;
  C3 GT appended as a 150th candidate -> oracle-in-fan reads EXACTLY 0;
  C4 fan[sel_idx] == the model's own traj (max abs diff, metres);
  C5 the whole fan vs the bank: terminal heading recomputed from the rolled fan (refcv3_arm.py:2533-2538 rule) vs the banked
     fan_term_heading_nav_true [N,149], and reach_keep vs the banked reach_keep_nav_true;
  C6 the roll's GT (Thor dump) == the banked GT bit-for-bit (2 s grid).
TIER: oracle-in-fan, random pick and every difference containing them use GT -> T0 DIAGNOSTIC, never driving performance.
`selected` alone is the os pick (T1 stamp, ruling open). CI answers ONLY "another draw of EPISODES"; one training seed;
the inference-seed variance is read by rolling a second seed, not by the CI.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

N_BOOT = 2000


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def load_x8(path, taniteval_root):
    if taniteval_root not in sys.path:
        sys.path.insert(0, taniteval_root)
    os.environ.setdefault("TANITEVAL_ROOT", taniteval_root)
    spec = importlib.util.spec_from_file_location("fan_control_x8", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_bank(dump_dir):
    """{(clip_index, ws): dict} from the banked battery dump (ep*.npz + decisions/ep*.npz)."""
    tab = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        dz = np.load(os.path.join(dump_dir, "decisions", os.path.basename(f)), allow_pickle=True)
        assert np.array_equal(np.asarray(z["ws"]), np.asarray(dz["ws"])), f
        ci = int(np.asarray(z["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(z["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(z["ws"]).ravel()):
            k = (ci, int(w))
            assert k not in tab
            tab[k] = {"g": z["g"][j].astype(np.float64), "os": z["os"][j].astype(np.float64),
                      "ha": z["ha"][j].astype(np.float64), "ha0_ext": z["ha0_ext"][j].astype(np.float64),
                      "eid": eid, "sel": int(dz["sel_idx_nav_true"][j]),
                      "rk": np.asarray(dz["reach_keep_nav_true"][j], np.float64),
                      "th": np.asarray(dz["fan_term_heading_nav_true"][j], np.float64)}
    return tab


def load_s6(s6_dir):
    tab = {}
    for f in sorted(glob.glob(os.path.join(s6_dir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci = int(np.asarray(z["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(z["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(z["ws"]).ravel()):
            tab[(ci, int(w))] = {"g": z["g"][j].astype(np.float64), "os": z["os"][j].astype(np.float64), "eid": eid}
    return tab


def term_heading(fan):
    """refcv3_arm.py:2533-2538, verbatim in numpy: last-minus-second-last slot, atan2, 0 where the step < 0.05 m."""
    d = fan[:, :, -1] - fan[:, :, -2]
    th = np.arctan2(d[..., 1], d[..., 0])
    return np.where(np.hypot(d[..., 0], d[..., 1]) < 0.05, 0.0, th)


def small_panel(C, fan, gt, sel, eid, tag):
    """selected / oracle / random-expectation + paired decomposition + skill, any K (the S6 grid)."""
    B = fan.shape[0]
    de = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
    ar = np.arange(B)
    ds, do, dr = de[ar, sel], de.min(1), de.mean(1)
    b = lambda v, t: dict(C.episode_cluster_bootstrap(v, eid, n_boot=N_BOOT, seed=0), metric=t)
    p = lambda a, c, t: dict(C.paired_episode_cluster_bootstrap(a, c, eid, n_boot=N_BOOT, seed=0), metric=t)
    head, skill = float((dr - do).mean()), float((dr - ds).mean())
    return {"grid": tag, "n_windows": int(B), "n_candidates": int(fan.shape[1]),
            "n_episodes": int(len(set(eid))),
            "selected": b(ds, "selected"), "oracle_in_fan": b(do, "oracle_in_fan"),
            "random_expectation": b(dr, "random_expectation (mean over N)"),
            "sel_gap__selected_minus_best": p(ds, do, "selected - oracle"),
            "best_minus_random_exp": p(do, dr, "oracle - random_expectation"),
            "random_exp_minus_selected": p(dr, ds, "random_expectation - selected"),
            "skill_fraction": {"value": round(skill / head, 4) if head > 0 else None,
                               "defn": "mean(random_exp - selected) / mean(random_exp - oracle); POINT (ratio of means)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", required=True)
    ap.add_argument("--bank-dump", required=True, help="banked battery dump_s<seed> (ep*.npz + decisions/)")
    ap.add_argument("--bank-s6", required=True, help="banked battery s6_s<seed>")
    ap.add_argument("--bank-analysis", required=True, help="banked analysis_s<seed>.json (own-pick ade_dense_m)")
    ap.add_argument("--roll-dump", default=None, help="the fan roll's own dump dir (C6: GT bit-equality)")
    ap.add_argument("--fan-roll-json", default=None)
    ap.add_argument("--x8", required=True, help="fan_control_x8.py (imported, unchanged)")
    ap.add_argument("--taniteval-root", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--bank-is-other-ckpt", action="store_true",
                    help="the bank is ANOTHER checkpoint's battery (GT / model-free refs only): C1/C2/C5 become informative, not pass/fail")
    ap.add_argument("--bank-na-note", default=None, help="like --bank-is-other-ckpt, with this reason text")
    a = ap.parse_args()
    t0 = time.time()
    X8 = load_x8(a.x8, a.taniteval_root)
    from taniteval import ci as C
    z = np.load(a.fan_npz, allow_pickle=True)
    ci, ws = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["ws"]).astype(np.int64)
    fan8 = np.asarray(z["anchor_traj"], np.float64)              # [B, N, 8, 2]
    B, N = fan8.shape[:2]
    sel = np.asarray(z["sel_idx"]).astype(np.int64).ravel()
    traj = np.asarray(z["traj"], np.float64)                      # [B, 8, 2]
    rk = np.asarray(z["reach_keep"], np.float64) if "reach_keep" in z.files else None
    bank = load_bank(a.bank_dump)
    keys = [(int(c), int(w)) for c, w in zip(ci, ws)]
    assert len(set(keys)) == B and B == len(bank), (B, len(bank))
    miss = [k for k in keys if k not in bank]
    assert not miss, miss[:3]
    gt = np.stack([bank[k]["g"] for k in keys])
    eid = [bank[k]["eid"] for k in keys]
    os_b = np.stack([bank[k]["os"] for k in keys])
    sel_b = np.asarray([bank[k]["sel"] for k in keys], np.int64)
    rk_b = np.stack([bank[k]["rk"] for k in keys])
    th_b = np.stack([bank[k]["th"] for k in keys])
    fan = fan8[:, :, :4]                                          # 2 s grid: model slots 0..3 = 0.5/1/1.5/2 s
    ar = np.arange(B)
    # ---------------------------------------------------------------- controls
    ctl = {}
    agree = float((sel == sel_b).mean())
    ctl["C1_sel_idx_agreement_vs_bank"] = {"value": round(agree, 6), "n_disagree": int((sel != sel_b).sum()),
                                           "bar": 0.99, "pass": bool(agree >= 0.99)}
    de_all = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
    sel_ade = float(de_all[ar, sel].mean())
    bank_os_ade = float(np.linalg.norm(os_b - gt, axis=-1).mean(-1).mean())
    aj = json.load(open(a.bank_analysis, encoding="utf-8"))
    banked = float(aj["arms"]["os"]["intervals"]["metrics"]["ade_dense_m"]["mean"])
    ctl["C2_selected_ade_vs_banked_own_pick"] = {
        "fan_selected_ade_2s": round(sel_ade, 6), "banked_analysis_ade_dense_m": banked,
        "banked_dump_os_ade_recomputed": round(bank_os_ade, 6),
        "abs_diff_vs_banked": round(abs(sel_ade - banked), 6), "tol": 1e-3,
        "pass": bool(abs(sel_ade - banked) <= 1e-3)}
    fan_gt = np.concatenate([fan, gt[:, None]], 1)
    or_gt = float(np.linalg.norm(fan_gt - gt[:, None], axis=-1).mean(-1).min(1).max())
    ctl["C3_gt_as_candidate_oracle"] = {"max_over_windows": or_gt, "pass": bool(or_gt == 0.0)}
    ctl["C4_fan_sel_equals_traj_m"] = {"max_abs": float(np.abs(fan8[ar, sel] - traj).max()),
                                       "pass": bool(np.abs(fan8[ar, sel] - traj).max() == 0.0)}
    th = term_heading(fan8)
    dth = np.abs(np.angle(np.exp(1j * (th - th_b))))
    ctl["C5_fan_vs_bank"] = {
        "term_heading_max_abs_rad": float(dth.max()), "term_heading_frac_within_1e-3": round(float((dth < 1e-3).mean()), 6),
        "term_heading_frac_within_1e-2": round(float((dth < 1e-2).mean()), 6),
        "reach_keep_agreement": (round(float((rk == rk_b).mean()), 6) if rk is not None else None),
        "traj2s_vs_banked_os_max_abs_m": float(np.abs(traj[:, :4] - os_b).max()),
        "traj2s_vs_banked_os_frac_windows_within_1e-3_m": round(float((np.abs(traj[:, :4] - os_b).max((1, 2)) < 1e-3).mean()), 6)}
    if a.roll_dump:
        rt = {}
        for f in sorted(glob.glob(os.path.join(a.roll_dump, "ep*.npz"))):
            q = np.load(f, allow_pickle=True)
            c0 = int(np.asarray(q["clip_index"]).ravel()[0])
            for j, w in enumerate(np.asarray(q["ws"]).ravel()):
                rt[(c0, int(w))] = q["g"][j].astype(np.float64)
        g2 = np.stack([rt[k] for k in keys])
        ctl["C6_roll_gt_equals_bank_gt"] = {"max_abs": float(np.abs(g2 - gt).max()),
                                            "pass": bool(np.array_equal(g2, gt))}
    # ---------------------------------------------------------------- the 2 s panel (fan_control_x8.fan_panel, unchanged)
    extra = {}
    ranks = {}
    for nm in ("anchor_logits", "refined_logits", "sel_score"):
        if nm in z.files and np.asarray(z[nm]).shape == (B, N):
            v = np.asarray(z[nm], np.float64)
            if np.isfinite(v).all():
                extra["argmax_" + nm] = v.argmax(1)
                ranks[nm] = v
            else:
                v2 = np.where(np.isfinite(v), v, -1e30)
                extra["argmax_" + nm] = v2.argmax(1)
                ranks[nm + "_finite"] = v2
    refs = {"ha": np.stack([bank[k]["ha"] for k in keys]), "ha0_ext": np.stack([bank[k]["ha0_ext"] for k in keys])}
    panel = X8.fan_panel(fan, gt, sel, eid, n_draws=12, extra_picks=extra, rank_scores=ranks, refs=refs)
    # ---------------------------------------------------------------- refcv8 additions (same estimator)
    b = lambda v, t: dict(C.episode_cluster_bootstrap(np.asarray(v, np.float64), eid, n_boot=N_BOOT, seed=0), metric=t)
    p = lambda x, y, t: dict(C.paired_episode_cluster_bootstrap(np.asarray(x, np.float64), np.asarray(y, np.float64),
                                                                eid, n_boot=N_BOOT, seed=0), metric=t)
    de_sel, de_or, de_rand = de_all[ar, sel], de_all.min(1), de_all.mean(1)
    add = {}
    if rk is not None:
        keep = rk > 0.5
        nk = keep.sum(1)
        de_k = np.where(keep, de_all, np.inf)
        or_k = np.where(nk > 0, de_k.min(1), de_or)
        rand_k = np.where(nk > 0, np.where(keep, de_all, 0).sum(1) / np.maximum(nk, 1), de_rand)
        add["reach_kept"] = {
            "n_kept_per_window": {"mean": round(float(nk.mean()), 2), "min": int(nk.min()), "max": int(nk.max()),
                                  "median": float(np.median(nk))},
            "n_windows_none_kept_fallback_all": int((nk == 0).sum()),
            "frac_sel_in_kept": round(float(keep[ar, sel].mean()), 6),
            "oracle_in_reach_kept": b(or_k, "oracle_in_reach_kept"),
            "random_in_reach_kept_expectation": b(rand_k, "random_in_reach_kept (mean over kept)"),
            "selected_minus_oracle_kept": p(de_sel, or_k, "selected - oracle_kept"),
            "oracle_kept_minus_oracle_all": p(or_k, de_or, "oracle_kept - oracle_all (cost of the reach mask)"),
            "random_kept_minus_selected": p(rand_k, de_sel, "random_kept - selected"),
            "skill_fraction_within_kept": round(float((rand_k - de_sel).mean() / max((rand_k - or_k).mean(), 1e-12)), 4),
            "frac_oracle_all_is_kept": round(float(keep[ar, de_all.argmin(1)].mean()), 6)}
    rank = (de_all < de_sel[:, None]).sum(1)                       # 0 = the selected candidate IS the oracle
    add["selected_rank_by_ade"] = {
        "defn": "number of candidates with strictly lower 2 s ADE than the selected one (0 = selected is the oracle), of N",
        "mean": round(float(rank.mean()), 3), "median": float(np.median(rank)),
        "p90": float(np.percentile(rank, 90)),
        "frac_rank0": round(float((rank == 0).mean()), 4), "frac_top5": round(float((rank < 5).mean()), 4),
        "frac_top10": round(float((rank < 10).mean()), 4), "frac_top30": round(float((rank < 30).mean()), 4),
        "rank_as_fraction_of_N_mean": round(float((rank / (N - 1)).mean()), 4),
        "hist_deciles": np.histogram(rank / (N - 1), bins=10, range=(0, 1))[0].tolist()}
    nb = 117
    if N > nb:
        or_base, or_ext = de_all[:, :nb].min(1), de_all[:, nb:].min(1)
        add["base_vs_extra"] = {
            "n_base": nb, "n_extra": int(N - nb),
            "oracle_in_base117": b(or_base, "oracle_in_base117"),
            "oracle_in_extra": b(or_ext, "oracle_in_extra%d" % (N - nb)),
            "frac_oracle_is_extra": round(float((de_all.argmin(1) >= nb).mean()), 4),
            "frac_sel_is_extra": round(float((sel >= nb).mean()), 4),
            "random_expectation_base": round(float(de_all[:, :nb].mean(1).mean()), 4),
            "random_expectation_extra": round(float(de_all[:, nb:].mean(1).mean()), 4)}
    # ---------------------------------------------------------------- LON / LAT split of the 2 s errors (BESIDE; T0 for oracle/random)
    # error vector at every waypoint, projected on the GT path tangent (along = LONGITUDINAL) and its normal (cross = LATERAL);
    # tangent = g[k] - g[k-1] with g[-1] = the origin; a GT step < 0.05 m (stopped) falls back to the ego x-axis.
    prev = np.concatenate([np.zeros((B, 1, 2)), gt[:, :-1]], 1)
    tan = gt - prev
    nrm = np.linalg.norm(tan, axis=-1, keepdims=True)
    tan = np.where(nrm < 0.05, np.array([1.0, 0.0]), tan / np.maximum(nrm, 1e-9))
    nor = np.stack([-tan[..., 1], tan[..., 0]], -1)

    def lonlat(pl):                                                 # pl [B,4,2] -> (|along| mean over k, |cross| mean over k)
        e = pl - gt
        return np.abs((e * tan).sum(-1)).mean(-1), np.abs((e * nor).sum(-1)).mean(-1)
    or_idx = de_all.argmin(1)
    la_s, lc_s = lonlat(fan[ar, sel])
    la_o, lc_o = lonlat(fan[ar, or_idx])
    e_all = fan - gt[:, None]
    la_r = np.abs((e_all * tan[:, None]).sum(-1)).mean(-1).mean(1)
    lc_r = np.abs((e_all * nor[:, None]).sum(-1)).mean(-1).mean(1)
    add["lon_lat_split_2s"] = {
        "defn": "mean over the 4 waypoints of |error . GT tangent| (LONGITUDINAL, along) and |error . GT normal| (LATERAL, cross); "
                "oracle = the ADE-oracle candidate; random = expectation over all N",
        "selected": {"along_m": b(la_s, "selected along"), "cross_m": b(lc_s, "selected cross")},
        "oracle_in_fan": {"along_m": b(la_o, "oracle along"), "cross_m": b(lc_o, "oracle cross")},
        "random_expectation": {"along_m": round(float(la_r.mean()), 4), "cross_m": round(float(lc_r.mean()), 4)},
        "selected_minus_oracle_along": p(la_s, la_o, "selected - oracle (along)"),
        "selected_minus_oracle_cross": p(lc_s, lc_o, "selected - oracle (cross)")}
    # ---------------------------------------------------------------- S6 grid (1..6 s; the battery's own S6 windows)
    s6 = load_s6(a.bank_s6)
    idx = [i for i, k in enumerate(keys) if k in s6]
    k6 = [keys[i] for i in idx]
    g6 = np.stack([s6[k]["g"] for k in k6])
    os6 = np.stack([s6[k]["os"] for k in k6])
    e6 = [s6[k]["eid"] for k in k6]
    f6 = fan8[idx][:, :, [1, 3, 4, 5, 6, 7]]
    sel6 = sel[idx]
    s6p = small_panel(C, f6, g6, sel6, e6, "S6 1..6 s (model slots 1,3,4,5,6,7)")
    s6p["control_selected_vs_banked_s6_os"] = {
        "fan_selected_ade_s6": round(float(np.linalg.norm(f6[np.arange(len(idx)), sel6] - g6, axis=-1).mean(-1).mean()), 6),
        "banked_s6_os_ade": round(float(np.linalg.norm(os6 - g6, axis=-1).mean(-1).mean()), 6),
        "max_abs_traj_vs_banked_s6_os_m": float(np.abs(traj[idx][:, [1, 3, 4, 5, 6, 7]] - os6).max())}
    if rk is not None:
        keep6 = (rk[idx] > 0.5)
        de6 = np.linalg.norm(f6 - g6[:, None], axis=-1).mean(-1)
        or6k = np.where(keep6.sum(1) > 0, np.where(keep6, de6, np.inf).min(1), de6.min(1))
        s6p["oracle_in_reach_kept"] = dict(C.episode_cluster_bootstrap(or6k, e6, n_boot=N_BOOT, seed=0),
                                           metric="oracle_in_reach_kept S6")
    if a.bank_is_other_ckpt or a.bank_na_note:
        for k in ("C1_sel_idx_agreement_vs_bank", "C2_selected_ade_vs_banked_own_pick", "C5_fan_vs_bank"):
            ctl[k]["pass"] = None
            ctl[k]["note"] = a.bank_na_note or ("NOT APPLICABLE as a control: the bank is a DIFFERENT checkpoint's battery (no "
                                                "per-frame battery bank exists for this checkpoint); read as an informative "
                                                "cross-checkpoint comparison only")
    res = {"arm": a.label, "mode": "refcv8-pf", "inference_seed": a.seed, "controls": ctl, "panel_2s": panel,
           "refcv8_additions_2s": add, "panel_s6": s6p,
           "inputs": {"fan_npz": a.fan_npz, "fan_npz_sha256": _sha256(a.fan_npz), "bank_dump": a.bank_dump,
                      "bank_s6": a.bank_s6, "bank_analysis": a.bank_analysis,
                      "bank_analysis_sha256": _sha256(a.bank_analysis), "roll_dump": a.roll_dump},
           "stamp": {"tier": "T0 DIAGNOSTIC for oracle / random / every difference containing them (they use GT); "
                             "`selected` alone is the os pick (T1 stamp, ruling open)",
                     "estimator": "FULL-SET mean; episode_cluster_bootstrap / paired_episode_cluster_bootstrap "
                                  "(taniteval/taniteval/ci.py) n_boot=%d seed=0" % N_BOOT,
                     "ci_answers": "ONLY another draw of EPISODES (139 eval clips). NOT another training run "
                                   "(one training seed, refcv8-r101-s0). NOT another inference run (the DDIM draw is "
                                   "stochastic): that is read by comparing inference seeds, not by this CI.",
                     "random_pick": "uniform over all N candidates; analytic expectation + 12 draws seeds 20261011..20261022",
                     "fan_control_x8": a.x8, "fan_control_x8_sha256": _sha256(a.x8),
                     "script": os.path.abspath(__file__), "script_sha256": _sha256(os.path.abspath(__file__)),
                     "wall_s": round(time.time() - t0, 1)}}
    if a.fan_roll_json:
        res["fan_roll"] = json.load(open(a.fan_roll_json, encoding="utf-8"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    pp = panel
    print("ZZFANCTLZZ seed=%d n=%d N=%d sel=%.4f or=%.4f rand=%.4f skill=%s C1=%.4f C2=%s C3=%s" % (
        a.seed, pp["n_windows"], pp["n_candidates"], pp["selected"]["mean"], pp["oracle_in_fan"]["mean"],
        pp["random_expectation"]["mean"], pp["skill_fraction"]["value"], agree,
        ctl["C2_selected_ade_vs_banked_own_pick"]["pass"], ctl["C3_gt_as_candidate_oracle"]["pass"]), flush=True)


if __name__ == "__main__":
    main()
