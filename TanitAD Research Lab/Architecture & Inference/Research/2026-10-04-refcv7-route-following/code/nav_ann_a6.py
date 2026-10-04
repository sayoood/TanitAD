"""SPEC_ADDENDUM_A6 (+ A7): score T3a (REPORTED) / T3 (foil) / T2a / T3a-c (control) on the DENSE capture of the held-out eval139.

Pre-registration: SPEC_ADDENDUM_A6.md (sha256 b4099578...) with A7's G1' replacing G1 (sha256 8e846a84...). Nothing here is
fitted on EVAL; every definition is A5's (`nav_tl_a5.py`), with `nav_ann` = A5's nav_tl rule over the ANNOUNCED entries only
(a non-announced entry is skipped, the next announced one is considered). Announced flags come from `raw/a6_announced_table.json`
(written by `announced_a6.py`, which imports the nav_command builder; this scoring process must NOT, because it imports the
launch-tree `taniteval` and the two resolve different `tanitad` trees).

Run (CPU only, launch-tree import; WINDOWS paths):
  PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" OMP_NUM_THREADS=4 python nav_ann_a6.py
Order: capture controls and A5 controls first; the arms are scored only if every control passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_route as A  # noqa: E402
import nav_tl_a5 as N  # noqa: E402
import rescorer_a2 as R2  # noqa: E402
import route_metrics as rm  # noqa: E402

RAW = HERE.parent / "raw"
H = N.H_DEFAULT
T3_K = N.T3_K
T2_TAU = N.T2_TAU
ARMS = ("T3a", "T3", "T2a", "T3a-c")


def nav_side(z, recs, flags, t_rel, donor=None, announced_only=True):
    """per-window nav side. announced_only=True: nav_ann (skip non-announced entries); False: A5's nav_tl (all entries).
    donor: clip -> donor clip (T3a-c): the donor's entries (and its announced flags) evaluated at the RECIPIENT's t_rel."""
    ws = np.asarray(z["win_sha12"]).astype(str)
    side = np.zeros(len(ws), np.int64)
    for i, s in enumerate(ws):
        src = donor[s] if donor is not None else s
        ents = recs[src]["entries"]
        if announced_only:
            ents = [e for e, f in zip(ents, flags[src]) if f]
        side[i] = N.nav_tl_state(ents, float(t_rel[i]), H)[0]
    return side


def reading_rule(bars):
    """A6 reading rule, fixed in the spec (and coded here BEFORE any arm was scored)."""
    if bars["T3a-c"]["CLEARS"]:
        return {"branch": "T3a-c PASSES", "text": "the instrument is broken; nothing from A6 is quotable"}
    if bars["T3a"]["CLEARS"]:
        return {"branch": "T3a PASSES, T3a-c FAILS",
                "text": "a deployable time-localised nav lifts route following at inference on refcv7 as trained (confirmed on denser "
                        "windows of the SAME episodes); it ships as an opt-in inference rule and L2 enters refcv8"}
    if bars["T3"]["CLEARS"]:
        return {"branch": "T3a FAILS, T3 PASSES",
                "text": "the inference gain needs non-announced turns (curves, obstacle passes) -- information deployment lacks; nothing ships "
                        "at inference; L2 still enters refcv8 as a training input with announced entries only"}
    return {"branch": "BOTH FAIL", "text": "A5's T3 result was within-episode noise; route following needs the training levers (L1 + selector)"}


def subset_view(d, keep):
    """A copy of derive()'s dict in which windows outside `keep` belong to no class (cls 'unclassified', target 9)."""
    dd = dict(d)
    dd["cls"] = np.where(keep, d["cls"], "unclassified").astype(object)
    dd["target"] = np.where(keep, d["target"], 9)
    return dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/refcv7_route_bin/2026-10-04/a6")
    ap.add_argument("--a5data", default="D:/refcv7_route_bin/2026-10-04")
    ap.add_argument("--labels-eval", default="D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    ap.add_argument("--sidecar", default="D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
    ap.add_argument("--k1-cols", default=str(RAW / "a5_k1_bank_cols.tsv"))
    ap.add_argument("--window-list", default=str(RAW / "a6_window_list.json"))
    ap.add_argument("--out", default=str(RAW / "a6_dense_score.json"))
    ap.add_argument("--s1-tag", default="a6_s1")
    ap.add_argument("--debug-no-stop", action="store_true", help="MECHANICS DEBUGGING ONLY: do not stop on a failed control (output must be discarded)")
    a = ap.parse_args()
    D = Path(a.data)
    res = {"spec": {"A6_sha256_expected": "b409957816248d333886ad694c5e3061cc26c9a29d115b246de4c91b1c5fd736",
                    "A6_sha256_on_disk": hashlib.sha256((HERE.parent / "SPEC_ADDENDUM_A6.md").read_bytes()).hexdigest(),
                    "A7_sha256_expected": "8e846a842018ad8c21a953ae8db3827a6be6e6b6343bce97fb8b1ac67164e5a5",
                    "A7_sha256_on_disk": hashlib.sha256((HERE.parent / "SPEC_ADDENDUM_A7.md").read_bytes()).hexdigest()}}
    if res["spec"]["A6_sha256_on_disk"] != res["spec"]["A6_sha256_expected"] or res["spec"]["A7_sha256_on_disk"] != res["spec"]["A7_sha256_expected"]:
        raise SystemExit("[A6] a SPEC file differs from its registered sha256")
    g1p = json.loads((RAW / "a7_g1prime.json").read_text(encoding="utf-8"))
    if not g1p["G1prime_PASS"]:
        raise SystemExit("[A6] G1' did not pass -- refusing to score")
    res["G1prime"] = {"PASS": True, "file": "raw/a7_g1prime.json"}
    recs = N.load_records(a.labels_eval)
    _g = sorted(f for f in {getattr(m, "__file__", None) for m in sys.modules.values()} if f and f.upper().startswith("G:"))
    if _g:
        raise SystemExit("[A6] a module was imported from G: " + str(_g[:3]))
    tab = json.loads((RAW / "a6_announced_table.json").read_text(encoding="utf-8"))["eval"]
    sidecar = N.read_sidecar(a.sidecar)
    z0 = np.load(D / "a6_s0.npz", allow_pickle=True)
    z0 = {k: z0[k] for k in z0.files}
    z0["_json"] = json.loads((RAW / "run_a6_s0.json").read_text(encoding="utf-8"))
    z1 = np.load(D / f"{a.s1_tag}.npz", allow_pickle=True)
    z1 = {k: z1[k] for k in z1.files}
    z1["_json"] = json.loads((RAW / f"run_{a.s1_tag}.json").read_text(encoding="utf-8"))
    ze = N.load_pack(a.a5data, RAW, "eval_s0g")
    ze1 = N.load_pack(a.a5data, RAW, "eval_s1")
    wl = json.loads(Path(a.window_list).read_text(encoding="utf-8"))
    ctl = {}

    # ---------------------------------------------------------------- capture controls
    ws0 = np.asarray(z0["win_sha12"]).astype(str)
    ws1 = np.asarray(z1["win_sha12"]).astype(str)
    key0 = list(zip(ws0, [int(t) for t in z0["win_t"]]))
    ctl["C_windows_seed0_eq_seed1_eq_window_list"] = {
        "n": len(key0), "PASS": bool(key0 == list(zip(ws1, [int(t) for t in z1["win_t"]]))
                                     and key0 == [(s, int(t)) for s, t, _f in wl["windows"]])}
    d0, d1 = A.derive(z0), A.derive(z1)
    gtsel = np.load(Path(a.window_list).with_suffix(".gt.npz"))
    ctl["C_selection_GT_eq_captured_GT"] = {
        "max_abs_diff_m": float(np.abs(gtsel["gt"] - z0["gt"]).max()),
        "valid_mismatch": int((gtsel["gt_valid"] != z0["gt_valid"]).sum()), "n": len(key0)}
    ctl["C_selection_GT_eq_captured_GT"]["PASS"] = bool(ctl["C_selection_GT_eq_captured_GT"]["max_abs_diff_m"] <= 1e-4
                                                        and ctl["C_selection_GT_eq_captured_GT"]["valid_mismatch"] == 0)
    n_turn_cap = int(np.isin(d0["cls"], ["turnL", "turnR"]).sum())
    sel_flag = np.array([f == "turn" for _s, _t, f in wl["windows"]])
    ctl["C_turn_windows_in_capture_eq_selected"] = {
        "captured_GT_turn": n_turn_cap, "selected_turn": int(sel_flag.sum()),
        "PASS": bool(n_turn_cap == int(sel_flag.sum()) == wl["n_turn_windows"]
                     and bool(np.isin(d0["cls"], ["turnL", "turnR"])[sel_flag].all()))}
    # identity vs A5's eval_s0g / eval_s1 on the overlapping windows (same per-window seed, same model)
    key5 = {(s, int(t)): i for i, (s, t) in enumerate(zip(np.asarray(ze["win_sha12"]).astype(str), ze["win_t"]))}
    ov = [(i, key5[k]) for i, k in enumerate(key0) if k in key5]
    ia, ib = np.array([p[0] for p in ov]), np.array([p[1] for p in ov])
    ident = {}
    for nm, zn, zr in (("seed0_vs_eval_s0g", z0, ze), ("seed1_vs_eval_s1", z1, ze1)):
        ident[nm] = {"n_overlap_windows": len(ov),
                     "fan_max_abs_diff": float(np.abs(zn["fan"][ia] - zr["fan"][ib]).max()),
                     "s_e9_max_abs_diff": float(np.abs(zn["s_e9"][ia] - zr["s_e9"][ib]).max()),
                     "sel_idx_mismatch": int((zn["sel_idx"][ia] != zr["sel_idx"][ib]).sum()),
                     "gt_max_abs_diff": float(np.abs(zn["gt"][ia] - zr["gt"][ib]).max())}
        ident[nm]["PASS"] = bool(ident[nm]["sel_idx_mismatch"] == 0 and ident[nm]["fan_max_abs_diff"] <= 1e-4)
    ctl["C_overlap_identity_with_A5_capture"] = {"PASS": all(v["PASS"] for v in ident.values()), **ident}
    # per-capture identity controls from the run records (C1 of the route package)
    for nm, z in (("a6_s0", z0), ("a6_s1", z1)):
        i_ = z["_json"]["identity"]
        ctl[f"C1_capture_identity_{nm}"] = {**i_, "PASS": bool(i_["traj_vs_fan_max_abs"] == 0.0 and i_["n_e9_argmax_mismatch"] == 0
                                                             and i_["n_core_argmax_mismatch"] == 0 and i_["max_model_forwards_per_window"] == 1)}
    # ---------------------------------------------------------------- A5 controls, re-run
    for nm, z in (("a6_s0", z0), ("a6_s1", z1)):
        bank = np.array([rm.NAV_SIDE[int(n)] for n in z["nav"]])
        rec = np.array([recs[s]["token_side"] for s in np.asarray(z["win_sha12"]).astype(str)])
        ctl[f"K0_join_bank_nav_eq_record_clip_token_{nm}"] = {"n": len(bank), "n_equal": int((bank == rec).sum()), "PASS": bool((bank == rec).all())}
    ctl["K1_clock_vs_replay_bank_t_label_s"] = N.k1_control(a.k1_cols, recs, sidecar)
    t_rel, src = N.window_clock(z0, recs, sidecar)
    k2 = N.k2_control(z0, recs, t_rel)
    k2["PASS"] = bool(k2["n_windows"] > 0 and k2["n_unequal"] == 0)
    ctl["K2_nav_tl_eq_clip_token_at_t_rel_0_dense_windows"] = k2
    ctl["K2_supplementary_dense_one_t_per_eval_clip"] = N.k2_dense(recs, sidecar)
    res["controls"] = ctl
    ok = all(v.get("PASS") for v in ctl.values())
    res["controls_all_pass"] = bool(ok)
    print("[A6] controls:", json.dumps({k: v.get("PASS") for k, v in ctl.items()}), flush=True)
    if not ok and not a.debug_no_stop:
        A.jdump(RAW / "a6_dense_FAILED_controls.json", res)
        print("[A6] A CONTROL FAILED -- the arms are NOT scored", flush=True)
        return 2

    # ---------------------------------------------------------------- nav series
    gate = z0["_json"]["gates"]["navc_gate"]
    keys5 = sorted(set(np.asarray(ze["win_sha12"]).astype(str)))
    der = N.derangement(keys5, seed=0)
    res["derangement"] = {"n_clips": len(der), "fixed_points": int(sum(1 for k, v in der.items() if k == v)),
                          "digest_sha256": hashlib.sha256(json.dumps(sorted(der.items())).encode()).hexdigest(),
                          "same_as_A5_T2c": hashlib.sha256(json.dumps(sorted(der.items())).encode()).hexdigest() ==
                          "968842331f9c8f1f058a024e90df1acec14e32e44b2594a9d3eb70b4097ff4b7"}
    side_ann = nav_side(z0, recs, tab, t_rel, announced_only=True)
    side_tl = nav_side(z0, recs, tab, t_rel, announced_only=False)
    side_ann_c = nav_side(z0, recs, tab, t_rel, donor=der, announced_only=True)
    clip_side = np.array([recs[s]["token_side"] for s in ws0])
    M = {"GT-turn": A.cls_mask(d0, "turn"), "GT-straight": A.cls_mask(d0, "straight"), "gentle": A.cls_mask(d0, "gentle"),
         "unclassified": d0["cls"] == "unclassified", "every window": np.ones(len(ws0), bool)}
    res["window_counts_dense"] = {nm: {"n": int(m.sum()), "nav_ann_active": int((m & (side_ann != 0)).sum()),
                                       "nav_tl_active": int((m & (side_tl != 0)).sum()), "clip_token_active": int((m & (clip_side != 0)).sum()),
                                       "nav_ann_differs_from_nav_tl": int((m & (side_ann != side_tl)).sum()),
                                       "nav_ann_eq_GT_dir": int((m & (side_ann == d0["target"]) & (side_ann != 0)).sum()),
                                       "nav_ann_opposite_of_GT": int((m & (side_ann != 0) & (side_ann == -d0["target"])).sum())}
                                  for nm, m in M.items()}
    res["capture"] = {"n_windows": len(ws0), "n_episodes": int(len(set(ws0))), "window_list": {k: wl[k] for k in (
        "n_windows_total_eval", "n_turn_windows", "n_rest_sampled", "class_counts_total", "class_counts_selected", "select_seconds")},
        "compute_seed0": z0["_json"]["compute"], "compute_seed1": z1["_json"]["compute"],
        "window_digest_sha256": {"seed0": z0["_json"]["window_sha12_t_sha256"], "seed1": z1["_json"]["window_sha12_t_sha256"]}}

    draws = rm.make_draws(d0["ep"], B=A.B, seed=A.SEED)
    clip_arr = clip_side

    def picks_for(z, d):
        return {"V0": z["sel_idx"].copy(),
                "T3a": N.pick_T3(z, d, side_ann, gate),
                "T3": N.pick_T3(z, d, side_tl, gate),
                "T2a": N.pick_T2(z, d, side_ann),
                "T3a-c": N.pick_T3(z, d, side_ann_c, gate),
                "B1t": A.pick("B1t", None, z, d),
                "ORACLE": d["oracle"].copy()}
    P0, P1 = picks_for(z0, d0), picks_for(z1, d1)
    # ---- the registered scoring: dense set
    L0, L1 = R2.table_for(z0, d0, P0, draws), R2.table_for(z1, d1, P1, draws)
    bars = {k: A.bar_check(L0, k, L1) for k in ARMS}
    res["bar_dense"] = bars
    res["reading_rule"] = reading_rule(bars)
    res["dense_seed0"], res["dense_seed1"] = L0, L1
    print("[A6] bars:", json.dumps({k: v["CLEARS"] for k, v in bars.items()}), "| reading:", res["reading_rule"]["branch"], flush=True)
    # ---- B1t fraction
    ar = np.arange(d0["W"])
    mt = A.cls_mask(d0, "turn")
    ade = {k: d0["ade_c"][ar, v] for k, v in P0.items()}
    dl = {k: ade[k] - ade["V0"] for k in ade}
    pt_a, bs_a, _ = rm.boot_mean(dl["T3a"], d0["ep"], mt, draws=draws)
    pt_b, bs_b, _ = rm.boot_mean(dl["B1t"], d0["ep"], mt, draws=draws)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = bs_a / bs_b
    res["B1t_fraction"] = {"T3a_dADE_turn": round(float(pt_a), 4), "B1t_dADE_turn_recomputed": round(float(pt_b), 4),
                           "fraction_recomputed": round(float(pt_a / pt_b), 4), "fraction_of_A5_registered_B1t_-0.566": round(float(pt_a / -0.566), 4),
                           "per_draw_ratio_ci95": rm.ci95(fr)}
    for arm in ("T3", "T2a"):
        p_, _b, _ = rm.boot_mean(dl[arm], d0["ep"], mt, draws=draws)
        res["B1t_fraction"][f"{arm}_fraction_recomputed"] = round(float(p_ / pt_b), 4)
    # ---- four families
    res["tactical_from_trajectory_dense_seed0"] = {k: A.tactical_ff(z0["fan"][ar, P0[k]].astype(np.float64), z0)
                                                  for k in ("V0", "T3a", "T3", "T2a", "T3a-c", "ORACLE")}
    # ---- A5's windows as a SUBSET (reported separately)
    keep = np.array([k in key5 for k in key0])
    ds0, ds1 = subset_view(d0, keep), subset_view(d1, keep)
    S0, S1 = R2.table_for(z0, ds0, P0, draws), R2.table_for(z1, ds1, P1, draws)
    res["subset_A5_windows"] = {"n_windows": int(keep.sum()),
                                "class_counts": {nm: int((M_ & keep).sum()) for nm, M_ in M.items()},
                                "bar": {k: A.bar_check(S0, k, S1) for k in ARMS}, "seed0": S0, "seed1": S1,
                                "note": "windows of the A5 EVAL grid that the dense capture contains: all 107 GT-turn windows + the random-sample members"}
    # ---- POST-HOC: natural-frequency re-weighting of the non-turn classes, both-draws pooling
    tot, sel = wl["class_counts_total"], wl["class_counts_selected"]
    w = np.ones(d0["W"])
    wts = {}
    for c in ("straight", "gentle"):
        wts[c] = tot[c] / max(sel[c], 1)
        w[d0["cls"] == c] = wts[c]
    allm = A.cls_mask(d0, "all")

    def wmean(x):
        pt, bs, _ = rm.boot_ratio(np.where(allm & np.isfinite(x), w * x, 0.0), np.where(allm & np.isfinite(x), w, 0.0), d0["ep"], draws=draws)
        return {"mean": round(float(pt), 4), "ci95": rm.ci95(bs)}
    dl1 = {k: (d1["ade_c"][ar, P1[k]] - d1["ade_c"][ar, P1["V0"]]) for k in P1}
    res["POSTHOC"] = {"label": "POST-HOC descriptive; not part of the bar or the reading rule",
                      "natural_frequency_weights_(non-turn_classes)": {k: round(v, 3) for k, v in wts.items()},
                      "all_window_dADE_natural_frequency_reweighted_seed0": {k: wmean(dl[k]) for k in ("T3a", "T3", "T2a", "T3a-c")},
                      "all_window_dADE_natural_frequency_reweighted_seed1": {k: wmean(dl1[k]) for k in ("T3a", "T3", "T2a", "T3a-c")},
                      "pooled_two_draws_dADE": {k: {cn: dict(zip(("mean", "ci95"), (lambda r: (round(float(r[0]), 4), rm.ci95(r[1])))(
                          rm.boot_mean(0.5 * (dl[k] + dl1[k]), d0["ep"], A.cls_mask(d0, cm), draws=draws)[:2])))
                          for cn, cm in (("turn", "turn"), ("straight", "straight"), ("all", "all"))} for k in ("T3a", "T3", "T2a", "T3a-c")}}
    # picks changed & where the T3a gain sits (seed 0)
    t = M["GT-turn"]
    for arm in ("T3a", "T3"):
        ch = t & (P0[arm] != P0["V0"])
        g = dl[arm]
        gg = np.sort(g[ch])
        res["POSTHOC"][f"{arm}_turn_windows_seed0"] = {
            "n_turn_windows": int(t.sum()), "n_pick_changed": int(ch.sum()), "n_changed_improved": int((g[ch] < 0).sum()),
            "n_changed_worse": int((g[ch] > 0).sum()), "n_episodes_with_a_changed_turn_window": int(len(set(d0["ep"][ch]))),
            "sum_dADE_over_changed": round(float(g[ch].sum()), 3), "sum_of_5_largest_gains": round(float(gg[:5].sum()), 3) if gg.size else None}
    # is the TRUE announced series better than the donor series on the same windows? (paired ADE, T3a - T3a-c)
    res["POSTHOC"]["paired_T3a_minus_T3a-c_ADE_(negative = the true series is better)"] = {
        tag: {cn: dict(zip(("mean", "ci95"), (lambda r: (round(float(r[0]), 4), rm.ci95(r[1])))(
            rm.boot_mean(dd["ade_c"][ar, PP["T3a"]] - dd["ade_c"][ar, PP["T3a-c"]], dd["ep"], A.cls_mask(dd, cm), draws=draws)[:2])))
            for cn, cm in (("turn", "turn"), ("straight", "straight"), ("all", "all"))}
        for tag, dd, PP in (("seed0", d0, P0), ("seed1", d1, P1))}
    # robustness across EPISODES (the estimator's cluster): per-episode mean dADE of T3a on GT-turn windows, and leave-one-episode-out
    for arm in ("T3a", "T2a"):
        eps_t = sorted(set(d0["ep"][t]))
        pe = np.array([dl[arm][t & (d0["ep"] == e)].mean() for e in eps_t])
        tot_sum, n_t = float(dl[arm][t].sum()), int(t.sum())
        loo = np.array([(tot_sum - dl[arm][t & (d0["ep"] == e)].sum()) / (n_t - int((t & (d0["ep"] == e)).sum())) for e in eps_t])
        res["POSTHOC"][f"{arm}_per_episode_turn_dADE_seed0"] = {
            "n_turn_episodes": len(eps_t), "n_episodes_dADE_lt_0": int((pe < -1e-12).sum()), "n_episodes_dADE_gt_0": int((pe > 1e-12).sum()),
            "n_episodes_unchanged": int((np.abs(pe) <= 1e-12).sum()),
            "leave_one_episode_out_turn_dADE_min_max": [round(float(loo.min()), 4), round(float(loo.max()), 4)],
            "largest_single_episode_share_of_total_gain": round(float(min(dl[arm][t & (d0["ep"] == e)].sum() for e in eps_t) / tot_sum), 3)}
    res["POSTHOC"]["T3a_vs_T3_picks_differ"] = {"windows": int((P0["T3a"] != P0["T3"]).sum()),
                                                "GT_turn_windows": int((t & (P0["T3a"] != P0["T3"])).sum())}
    res["picks_changed_vs_V0_seed0"] = {k: L0[k]["pick_changed_frac"] for k in L0}
    res["picks_changed_vs_V0_seed1"] = {k: L1[k]["pick_changed_frac"] for k in L1}
    res["nav_oracle_caveat"] = ("nav_ann is built from the v8 nav_30s ORACLE (ego-future) restricted to ANNOUNCED entries; an inference-time "
                                "route-following gain under that caveat; the 38 turn episodes are the SAME held-out episodes as A5 "
                                "(denser windows, not new episodes); one training seed")
    on_g = sorted(f for f in {getattr(m, "__file__", None) for m in sys.modules.values()} if f and f.upper().startswith("G:"))
    res["modules_imported_from_G_mount"] = on_g
    if on_g:
        raise SystemExit("[A6] a module was imported from G: " + str(on_g[:3]))
    A.jdump(Path(a.out), res)
    print("[A6] wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
