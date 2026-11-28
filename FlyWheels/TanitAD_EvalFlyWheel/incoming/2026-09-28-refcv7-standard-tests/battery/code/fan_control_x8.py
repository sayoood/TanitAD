#!/usr/bin/env python
"""fan_control_x8.py -- RANDOM-PICK CONTROL for the oracle-in-fan vs selected gap (D-REFCV7-FANCEIL).

WHY: best-actual is LARGE for ANY wide fan, so "oracle 0.1914 vs selected 0.4728 m" says
nothing until it is read against a RANDOM pick from the same fan on the same windows.
Per arm this reports:
    selected   = ADE of the model own pick                           (de_sel)
    oracle     = min over the N candidates, uses GT -> T0 DIAGNOSTIC  (de_or)
    random     = uniform pick over the N candidates                   (de_rand)
                   analytic expectation = mean over all N candidates (infinite-draw limit)
                   plus K >= 10 explicit draws, one fixed seed each
    best - random      = oracle - random     (<= 0: headroom a PERFECT selector has over chance)
    random - selected  = random - selected   (> 0: the selector beats chance by this much)
    selected - best    = sel_gap             (what refc_scale_ab.py published)
    skill fraction     = (random - selected) / (random - oracle)   (0 = chance, 1 = oracle)

ESTIMATOR: FULL-SET mean; episode-cluster bootstrap (single quantities) and PAIRED
episode-cluster bootstrap (differences), taniteval/taniteval/ci.py, n_boot=2000, seed 0.
WHICH VARIANCE: the CI answers ONLY "would another draw of EPISODES say this?". It does NOT
answer another TRAINING run, and for a fan with stochastic refinement NOT another INFERENCE
run either (that needs a second fan roll, same script).
TIER: oracle-in-fan and every difference containing it use GT => T0 diagnostic, never driving
performance. `selected` alone is the T1 open-loop self-action pick.
Modes:  refcv7 --fan-npz F --gt-dump D   |   oldpt --pt fan_refc-*.pt (reproduction check)
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

for _p in (os.environ.get("TANITEVAL_ROOT"), "C:/Users/Admin/ev7/taniteval",
           "E:/Projects/TanitAD/taniteval"):
    if _p and os.path.isdir(os.path.join(_p, "taniteval")) and _p not in sys.path:
        sys.path.insert(0, _p)
from taniteval import ci as C  # noqa: E402

N_BOOT = 2000
N_DRAWS_DEFAULT = 12
DRAW_SEED_BASE = 20261011      # draw k uses seed DRAW_SEED_BASE + k (fixed, recorded)


def _boot(v, eid, tag):
    b = C.episode_cluster_bootstrap(np.asarray(v, np.float64), eid, n_boot=N_BOOT, seed=0)
    b["metric"] = tag
    return b


def _paired(a, b, eid, tag):
    r = C.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                           eid, n_boot=N_BOOT, seed=0)
    r["metric"] = tag
    return r


def _de_all(fan, gt):
    """[B,N] per-candidate ADE over the 4 waypoints (mirrors refc_scale_ab._de_all)."""
    return np.linalg.norm(fan.astype(np.float64) - gt.astype(np.float64)[:, None], axis=-1).mean(-1)


TOPK = (2, 4, 8, 16, 32)


def fan_panel(fan, gt, sel, eid, n_draws=N_DRAWS_DEFAULT, extra_picks=None,
              rank_scores=None, refs=None):
    """The control on ONE arm. fan [B,N,4,2], gt [B,4,2], sel [B] int, eid [B]."""
    fan = np.asarray(fan)
    B, N = fan.shape[:2]
    assert fan.shape[2:] == (4, 2) and gt.shape == (B, 4, 2) and len(eid) == B and len(sel) == B
    sel = np.asarray(sel).astype(np.int64)
    assert sel.min() >= 0 and sel.max() < N, "sel out of range"
    de_all = _de_all(fan, gt)                       # [B,N]
    ar = np.arange(B)
    de_sel = de_all[ar, sel]
    de_or = de_all.min(1)
    de_rand_exp = de_all.mean(1)                    # analytic E[uniform pick] per window
    out = {"n_windows": int(B), "n_candidates": int(N),
           "n_episodes": int(len(set(map(str, eid)))),
           "selected": _boot(de_sel, eid, "selected"),
           "oracle_in_fan": _boot(de_or, eid, "oracle_in_fan"),
           "random_expectation": _boot(de_rand_exp, eid, "random_expectation (mean over N)"),
           "frac_sel_2x_worse": round(float((de_sel > 2 * de_or).mean()), 4),
           "frac_randexp_2x_worse": round(float((de_rand_exp > 2 * de_or).mean()), 4),
           "frac_sel_better_than_randexp": round(float((de_sel < de_rand_exp).mean()), 4),
           "sel_idx_n_unique": int(np.unique(sel).size),
           "fan_spread": {"median_over_windows_of_candidate_std":
                          round(float(np.median(de_all.std(1))), 4),
                          "mean_worst_candidate_ade": round(float(de_all.max(1).mean()), 4)}}
    # --- headline decomposition, all PAIRED on identical windows ---
    out["sel_gap__selected_minus_best"] = _paired(de_sel, de_or, eid, "selected - oracle (sel_gap)")
    out["best_minus_random_exp"] = _paired(de_or, de_rand_exp, eid, "oracle - random_expectation")
    out["random_exp_minus_selected"] = _paired(de_rand_exp, de_sel, eid,
                                               "random_expectation - selected")
    head = float((de_rand_exp - de_or).mean())
    skill = float((de_rand_exp - de_sel).mean())
    out["skill_fraction"] = {"value": round(skill / head, 4) if head > 0 else None,
                             "defn": "mean(random_exp - selected) / mean(random_exp - oracle); "
                                     "0 = chance, 1 = oracle; POINT estimate (ratio of means)"}
    # --- explicit uniform draws ---
    draws = []
    rd_all = np.zeros(B)
    for k in range(n_draws):
        rng = np.random.default_rng(DRAW_SEED_BASE + k)
        pick = rng.integers(0, N, size=B)
        d = de_all[ar, pick]
        rd_all += d
        draws.append({"draw": k, "seed": DRAW_SEED_BASE + k,
                      "random_pick": _boot(d, eid, "random_draw_%d" % k),
                      "best_minus_random": _paired(de_or, d, eid, "oracle - random_draw_%d" % k),
                      "random_minus_selected": _paired(d, de_sel, eid,
                                                       "random_draw_%d - selected" % k)})
    out["draws"] = draws
    m = [x["random_pick"]["mean"] for x in draws]
    bm = [x["best_minus_random"]["delta"] for x in draws]
    rm = [x["random_minus_selected"]["delta"] for x in draws]
    npos = sum(1 for x in draws if x["random_minus_selected"]["separated"]
               and x["random_minus_selected"]["delta"] > 0)
    nneg = sum(1 for x in draws if x["random_minus_selected"]["separated"]
               and x["random_minus_selected"]["delta"] < 0)
    out["draws_summary"] = {
        "n_draws": n_draws, "seed_base": DRAW_SEED_BASE,
        "random_pick_mean_over_draws": round(float(np.mean(m)), 4),
        "random_pick_min_max": [round(float(min(m)), 4), round(float(max(m)), 4)],
        "best_minus_random_range": [round(float(min(bm)), 4), round(float(max(bm)), 4)],
        "random_minus_selected_range": [round(float(min(rm)), 4), round(float(max(rm)), 4)],
        "random_minus_selected_n_separated_pos": int(npos),
        "random_minus_selected_n_separated_neg": int(nneg),
        "analytic_expectation_mean": out["random_expectation"]["mean"]}
    out["avgK_random_minus_selected"] = _paired(rd_all / n_draws, de_sel, eid,
                                                "mean_of_%d_draws - selected" % n_draws)
    # --- extra (NOT requested): deterministic alternative pickers on the same windows ---
    for name, idx in (extra_picks or {}).items():
        idx = np.asarray(idx).astype(np.int64)
        d = de_all[ar, idx]
        out.setdefault("extra_pickers", {})[name] = {
            "ade": _boot(d, eid, name),
            "minus_selected": _paired(d, de_sel, eid, name + " - selected"),
            "agree_with_sel_frac": round(float((idx == sel).mean()), 4)}
    # --- STRONGER NULL (extra): uniform-over-vocabulary is a straw man when the vocabulary spans all
    # speeds/curvatures. Random pick among the model own top-K (analytic expectation = mean of the
    # K best-ranked candidates ADE), per score head. K=1 is that head argmax. ---
    for sname, sc in (rank_scores or {}).items():
        sc = np.asarray(sc, np.float64)
        assert sc.shape == (B, N), (sname, sc.shape)
        order = np.argsort(-sc, axis=1, kind="stable")
        rows = {}
        for K in TOPK:
            if K > N:
                continue
            d = np.take_along_axis(de_all, order[:, :K], axis=1).mean(1)
            rows["K%d" % K] = {"ade": _boot(d, eid, "%s_top%d_random" % (sname, K)),
                               "minus_selected": _paired(d, de_sel, eid,
                                                         "%s_top%d_random - selected" % (sname, K)),
                               "minus_oracle": _paired(d, de_or, eid,
                                                       "%s_top%d_random - oracle" % (sname, K))}
        rows["oracle_rank_in_top"] = {"frac_oracle_in_top%d" % K: round(float(
            (order[:, :K] == de_all.argmin(1)[:, None]).any(1).mean()), 4) for K in TOPK if K <= N}
        out.setdefault("topk_random_by_score", {})[sname] = rows
    # --- reference trajectories on the same windows (hold-action / constant-velocity) ---
    for rname, tr in (refs or {}).items():
        tr = np.asarray(tr, np.float64)
        assert tr.shape == (B, 4, 2), (rname, tr.shape)
        d = np.linalg.norm(tr - gt.astype(np.float64), axis=-1).mean(-1)
        out.setdefault("reference_plans", {})[rname] = {
            "ade": _boot(d, eid, rname), "minus_selected": _paired(d, de_sel, eid, rname + " - selected")}
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def load_gt_dump(dump_dir):
    """{(clip_index, ws): (g[4,2], eid, os[4,2])} from a banked dump dir of epNNN.npz."""
    fs = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    assert fs, "no epNNN.npz under " + dump_dir
    tab = {}
    for f in fs:
        z = np.load(f, allow_pickle=True)
        ci = int(np.asarray(z["clip_index"]).ravel()[0])
        eid = str(int(np.asarray(z["eid"]).ravel()[0]))
        for j, w in enumerate(np.asarray(z["ws"]).ravel()):
            key = (ci, int(w))
            assert key not in tab, "duplicate window %s in %s" % (key, dump_dir)
            tab[key] = (z["g"][j].astype(np.float64), eid, z["os"][j].astype(np.float64))
    return tab, len(fs)


def load_ha(dump_dir):
    """{(clip_index, ws): ha[4,2]} -- the hold-action plan, banked beside the GT."""
    tab = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ci = int(np.asarray(z["clip_index"]).ravel()[0])
        for j, w in enumerate(np.asarray(z["ws"]).ravel()):
            tab[(ci, int(w))] = z["ha"][j].astype(np.float64)
    return tab


def run_refcv7(a):
    z = np.load(a.fan_npz, allow_pickle=True)
    ci, ws = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["ws"]).astype(np.int64)
    fan = z["anchor_traj"]
    sel = np.asarray(z["sel_idx"]).astype(np.int64).ravel()
    B = fan.shape[0]
    assert fan.ndim == 4 and fan.shape[2:] == (4, 2), "fan shape %s" % (fan.shape,)
    assert len(ci) == len(ws) == len(sel) == B
    tab, n_ep_files = load_gt_dump(a.gt_dump)
    keys = [(int(c), int(w)) for c, w in zip(ci, ws)]
    assert len(set(keys)) == B, "duplicate (clip_index, ws) in the fan file"
    miss = [k for k in keys if k not in tab]
    assert not miss, "%d fan windows have no GT row, e.g. %s" % (len(miss), miss[:3])
    assert B == len(tab), "fan windows %d != dump windows %d" % (B, len(tab))
    tab_ha = load_ha(a.gt_dump)
    gt = np.stack([tab[k][0] for k in keys])
    eid = [tab[k][1] for k in keys]
    os_bank = np.stack([tab[k][2] for k in keys])
    # selected plan inside the fan must equal traj[:4] (the model own 2 s plan) -- exact
    traj4 = np.asarray(z["traj"])[:, :4, :].astype(np.float64)
    fan_sel = fan[np.arange(B), sel].astype(np.float64)
    sel_eq_traj = float(np.abs(fan_sel - traj4).max())
    extra = {}
    for nm in ("anchor_logits", "refined_logits"):
        if nm in z.files and np.asarray(z[nm]).shape == (B, fan.shape[1]):
            extra["argmax_" + nm] = np.asarray(z[nm]).argmax(1)
    ranks = {nm: np.asarray(z[nm]) for nm in ("anchor_logits", "refined_logits", "sel_score")
             if nm in z.files and np.asarray(z[nm]).shape == (B, fan.shape[1])}
    refs = {}
    for rn in ("ha",):
        refs[rn] = np.stack([tab_ha[k] for k in keys])
    panel = fan_panel(fan, gt, sel, eid, n_draws=a.n_draws, extra_picks=extra,
                      rank_scores=ranks, refs=refs)
    ade_fan_sel = np.linalg.norm(fan_sel - gt, axis=-1).mean(-1)
    xc = {"fan_selected_ade_4wp_mean": round(float(ade_fan_sel.mean()), 4),
          "max_abs_fan_sel_minus_traj4_m": round(sel_eq_traj, 6)}
    for d in (a.xcheck_dump or []):
        t2, _ = load_gt_dump(d)
        ks = [k for k in keys if k in t2]
        assert len(ks) == B, "xcheck dump %s covers %d/%d" % (d, len(ks), B)
        o2 = np.stack([t2[k][2] for k in keys])
        ade2 = np.linalg.norm(o2 - gt, axis=-1).mean(-1)
        xc[os.path.basename(os.path.normpath(d))] = {
            "banked_os_ade_4wp_mean": round(float(ade2.mean()), 4),
            "fan_minus_banked_mean": round(float(ade_fan_sel.mean() - ade2.mean()), 4),
            "paired_fan_minus_banked": _paired(ade_fan_sel, ade2, eid,
                                               "fan_selected - banked_os (same windows)")}
    return {"arm": "refcv7-r101-s0", "mode": "refcv7", "panel": panel, "crosscheck": xc,
            "n_dump_episode_files": n_ep_files, "fan_npz": a.fan_npz,
            "fan_npz_sha256": _sha256(a.fan_npz)}


def run_oldpt(a):
    import torch
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    fan, gt = d["fan"].numpy(), d["gt"].numpy()
    sel = d["sel"].numpy()
    extra = {"argmax_logits": d["logits"].numpy().argmax(1)}
    panel = fan_panel(fan, gt, sel, d["eid"], n_draws=a.n_draws, extra_picks=extra,
                      rank_scores={"logits": d["logits"].numpy()},
                      refs={"cv": d["cv"].numpy()})
    return {"arm": os.path.basename(a.pt), "mode": "oldpt", "panel": panel,
            "ckpt": d.get("ckpt"), "ckpt_step": d.get("ckpt_step"),
            "n_anchors": d.get("n_anchors"), "pt_sha256": _sha256(a.pt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["refcv7", "oldpt"])
    ap.add_argument("--fan-npz")
    ap.add_argument("--gt-dump")
    ap.add_argument("--xcheck-dump", nargs="*")
    ap.add_argument("--pt")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--ckpt-md5", default=None)
    ap.add_argument("--roll-json", default=None)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    assert a.n_draws >= 10, "the brief requires at least 10 uniform draws"
    t0 = time.time()
    res = run_refcv7(a) if a.mode == "refcv7" else run_oldpt(a)
    res["stamp"] = {
        "tier": "T0 diagnostic (oracle and every difference containing it use GT); "
                "selected alone is the T1 open-loop pick",
        "estimator": "FULL-SET mean; episode_cluster_bootstrap / paired_episode_cluster_bootstrap "
                     "(taniteval/taniteval/ci.py) n_boot=%d seed=0" % N_BOOT,
        "ci_answers": "ONLY: would another draw of EPISODES say this? NOT another training run; "
                      "NOT another inference run (stochastic refinement) unless a second fan roll is run",
        "random_pick": "uniform over all N candidates in the fan (not prior-weighted); "
                       "%d explicit draws seeds %d..%d plus the analytic expectation"
                       % (a.n_draws, DRAW_SEED_BASE, DRAW_SEED_BASE + a.n_draws - 1),
        "ckpt_md5_declared": a.ckpt_md5, "script": os.path.abspath(__file__),
        "wall_s": round(time.time() - t0, 1)}
    if a.roll_json:
        rj = json.load(open(a.roll_json))
        res["roll_meta"] = {k: rj.get(k) for k in rj if k != "roll_record"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    p = res["panel"]
    print("ZZ%sZZ n=%d N=%d ep=%d sel=%.4f or=%.4f rand=%.4f skill=%s" % (
        res["arm"], p["n_windows"], p["n_candidates"], p["n_episodes"], p["selected"]["mean"],
        p["oracle_in_fan"]["mean"], p["random_expectation"]["mean"], p["skill_fraction"]["value"]))


if __name__ == "__main__":
    main()
