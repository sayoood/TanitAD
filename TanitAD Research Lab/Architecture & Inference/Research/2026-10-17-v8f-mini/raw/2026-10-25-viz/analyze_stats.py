"""analyze_stats.py -- merge the per-clip chunks of stats_pass.py, compute the CONTROL numbers and the aggregate attention statistics over ALL
2,399 eval windows, write fig_attention_stats.json / .png and controls.json.

Plotted aggregation (declared): attention weight averaged over HEADS and QUERIES, and over the LAYERS of that Reader (S, T: 1 layer; O: 2).
Per-layer numbers are kept in the JSON.  CIs: episode-cluster bootstrap (resample the 141 episodes, window-weighted mean, 2,000 draws, 95 % percentile).
These intervals answer 'would another draw of EPISODES say this?' only -- NOT 'would another training run say this?' (that is why both seeds are shown).
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_extract import *  # noqa: E402,F401,F403

CHUNKS = f"{WORK}/stats/chunks"
SEEDS = ("M2b_s0", "M2b_s1")
NB = 2000


def merge():
    files = sorted(glob.glob(f"{CHUNKS}/*.npz"))
    parts = [dict(np.load(f)) for f in files]
    keys = parts[0].keys()
    M_ = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    o = np.argsort(M_["win"], kind="stable")
    return {k: v[o] for k, v in M_.items()}, len(files)


def boot_mean(V: np.ndarray, grp: np.ndarray, seed: int = 0):
    """V [N, K] (or [N]); grp [N] episode id.  Returns (mean, lo, hi) of the window-weighted mean under episode-cluster resampling."""
    V = V.reshape(len(V), -1).astype(np.float64)
    ug, inv = np.unique(grp, return_inverse=True)
    G = len(ug)
    Es = np.zeros((G, V.shape[1]))
    np.add.at(Es, inv, V)
    n = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=NB).astype(np.float64)
    bm = (cnt @ Es) / (cnt @ n)[:, None]
    return V.mean(0), np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)


def pack(V, grp):
    m, lo, hi = boot_mean(V, grp)
    return {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def level_arrays(Mg, seed, level, layers="mean"):
    """-> cond [N, nc], scene [N, 4, 40] (mean over heads, queries, and layers)."""
    keys = sorted(k for k in Mg if k.startswith(f"{seed}__att_{level}"))
    cs, ss = [], []
    for k in keys:
        a = Mg[k].mean(axis=(1, 2))                         # [N, S]
        nc = len(COND_NAMES[level])
        cs.append(a[:, :nc])
        ss.append(a[:, nc:].reshape(-1, HIST, NTOK))
    if layers == "mean":
        return np.mean(cs, 0), np.mean(ss, 0)
    return cs[int(layers)], ss[int(layers)]


def stats_of(cond, scene, grp, strata=None):
    N = len(cond)
    scene_mass = scene.sum((1, 2))
    sm = np.maximum(scene_mass, 1e-12)
    frame = scene.sum(2)                                    # [N, 4] absolute mass per history frame (0 oldest .. 3 tick)
    rows = scene.reshape(N, HIST, GH, GW).sum((1, 3))       # [N, 4] mass per token row over all history frames
    cols = scene.reshape(N, HIST, GH, GW).sum((1, 2))       # [N, 10]
    rows_new = scene[:, -1].reshape(N, GH, GW).sum(2)       # tick frame only
    p = scene.reshape(N, -1) / sm[:, None]
    ent = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
    out = {
        "n_windows": N,
        "cond_mass": pack(cond.sum(1), grp),
        "scene_mass": pack(scene_mass, grp),
        "cond_token_mass": pack(cond, grp),                              # absolute mass per cond token
        "frame_mass_abs": pack(frame, grp),                              # absolute, 4 history frames
        "frame_share_of_scene": pack(frame / sm[:, None], grp),
        "row_share_of_scene": pack(rows / sm[:, None], grp),             # far (row 0, top of image) -> near (row 3)
        "col_share_of_scene": pack(cols / sm[:, None], grp),
        "row_share_tick_frame": pack(rows_new / np.maximum(rows_new.sum(1, keepdims=True), 1e-12), grp),
        "scene_entropy_nats": pack(ent, grp),
        "effective_tokens_exp_entropy": pack(np.exp(ent), grp),
        "max_token_share_of_scene": pack(p.max(1), grp),
        "uniform_baseline": {"cond_mass_if_uniform": cond.shape[1] / (cond.shape[1] + HIST * NTOK), "frame_share": 0.25, "row_share": 0.25,
                             "effective_tokens_if_uniform": HIST * NTOK},
    }
    return out


def main():
    Mg, nf = merge()
    N = len(Mg["win"])
    assert N == 2399 and (Mg["win"] == np.arange(N)).all(), (N, nf)
    ew_grid = json.load(open(f"{INPUTS}/eval_grid_windows.json"))
    clips = [c for c in ew_grid["order"] for _ in ew_grid["windows"][c]]
    eid = np.array([ew_grid["order"].index(c) for c in clips])
    st = np.load(STRATA_NPZ, allow_pickle=True)["stratum"]
    SN = ("JUNCTION", "CURVE", "NO-TURN")

    # ------------------------------------------------------------------ controls
    ctl = {"n_windows": N, "n_clips": nf, "seeds": {}}
    for s in SEEDS:
        ctl["seeds"][s] = {
            "hooked_vs_unhooked_traj60_maxabs": float(Mg[f"{s}__ctl_hook_vs_plain_traj60"].max()),
            "needweights_mode_vs_unhooked_traj60_maxabs": float(Mg[f"{s}__ctl_needweights_vs_plain_traj60"].max()),
            "explicit_weights_vs_torch_needweights_maxabs": float(Mg[f"{s}__ctl_explicit_vs_needweights_attn"].max()),
            "hooked_traj30_vs_banked_pred_maxabs": float(Mg[f"{s}__ctl_vs_banked_pred"].max()),
            "hooked_ade6_vs_banked_ade6_maxabs": float(Mg[f"{s}__ctl_vs_banked_ade6"].max()),
            "n_windows_over_1e-5_vs_banked_pred": int((Mg[f"{s}__ctl_vs_banked_pred"] > 1e-5).sum()),
            "ade6_mean_mine": float(Mg[f"{s}__ade6"].mean()),
            "rowsum_check": {k.split("__")[1]: [float(Mg[k].sum(-1).min()), float(Mg[k].sum(-1).max())] for k in Mg if k.startswith(f"{s}__att_")},
        }
    ctl["PASS_hook_vs_plain_le_1e-5"] = bool(all(v["hooked_vs_unhooked_traj60_maxabs"] <= 1e-5 for v in ctl["seeds"].values()))
    ctl["randomisation"] = {}
    for lv in LEVELS:
        r = {k: Mg[f"rand_{lv}__{k}"] for k in ("rho_all", "rho_new", "dother", "dtraj", "cond_trained", "cond_random")}
        ctl["randomisation"][lv] = {
            "spearman_all_160_tokens": pack(r["rho_all"], eid), "spearman_tick_frame_40_tokens": pack(r["rho_new"], eid),
            "frac_windows_rho_all_gt_0.5": float((r["rho_all"] > 0.5).mean()),
            "median_rho_all": float(np.median(r["rho_all"])),
            "other_levels_attention_maxabs_delta": float(r["dother"].max()),
            "traj60_maxabs_delta_trained_vs_randomised_run": float(r["dtraj"].max()),
            "cond_mass_trained": pack(r["cond_trained"], eid), "cond_mass_randomised": pack(r["cond_random"], eid)}
    json.dump(ctl, open(f"{OUT}/controls.json", "w"), indent=1)

    # ------------------------------------------------------------------ statistics
    res = {"aggregation": "mean over heads, queries and the Reader's layers", "n_windows": N, "n_episodes": int(eid.max() + 1),
           "estimator": "episode-cluster bootstrap, 141 episodes, 2000 draws, 95% percentile (answers the EPISODE-draw question only)",
           "cond_names": {lv: list(COND_NAMES[lv]) for lv in LEVELS}, "seeds": {}, "contrasts": {}, "seed_agreement": {}}
    arrs = {}
    for s in SEEDS:
        res["seeds"][s] = {}
        for lv in LEVELS:
            cond, scene = level_arrays(Mg, s, lv)
            arrs[(s, lv)] = (cond, scene)
            d = stats_of(cond, scene, eid)
            d["per_layer_cond_mass"] = {f"layer{li}": pack(level_arrays(Mg, s, lv, li)[0].sum(1), eid) for li in range(len(
                [k for k in Mg if k.startswith(f"{s}__att_{lv}")]))}
            # per-head cond mass (heads are not interchangeable across levels; kept for transparency)
            ks = sorted(k for k in Mg if k.startswith(f"{s}__att_{lv}"))
            nc = len(COND_NAMES[lv])
            ph = np.mean([Mg[k][..., :nc].sum(-1).mean(2) for k in ks], 0)       # [N, H]
            d["per_head_cond_mass"] = pack(ph, eid)
            d["by_stratum"] = {}
            for i, n in enumerate(SN):
                m = st == i
                dd = stats_of(cond[m], scene[m], eid[m])
                d["by_stratum"][n] = {k: dd[k] for k in ("n_windows", "cond_mass", "cond_token_mass", "frame_share_of_scene", "row_share_of_scene",
                                                          "effective_tokens_exp_entropy")}
            d["mean_tick_frame_map_4x10"] = (scene[:, -1].mean(0) / np.maximum(scene[:, -1].sum(1).mean(), 1e-12)).reshape(GH, GW).tolist()
            d["mean_all_history_map_4x10"] = (scene.sum(1).mean(0) / np.maximum(scene.sum((1, 2)).mean(), 1e-12)).reshape(GH, GW).tolist()
            res["seeds"][s][lv] = d
        # paired contrasts between levels on the SAME windows
        res["contrasts"][s] = {}
        for a_, b_ in (("T", "S"), ("O", "T"), ("O", "S")):
            ca, sa = arrs[(s, a_)]
            cb, sb = arrs[(s, b_)]
            def feats(c, sc):
                N_ = len(c)
                mass = sc.sum((1, 2))
                sm = np.maximum(mass, 1e-12)
                fr = sc.sum(2) / sm[:, None]
                rw = sc.reshape(N_, HIST, GH, GW).sum((1, 3)) / sm[:, None]
                p = sc.reshape(N_, -1) / sm[:, None]
                return np.column_stack([c.sum(1), fr[:, -1], rw[:, 0], rw[:, 3], np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum(1))])
            D_ = feats(ca, sa) - feats(cb, sb)
            names = ["cond_mass", "tick_frame_share_of_scene", "row0_far_share", "row3_near_share", "effective_tokens"]
            m, lo, hi = boot_mean(D_, eid)
            res["contrasts"][s][f"{a_}-{b_}"] = {n: {"mean": float(m[i]), "lo": float(lo[i]), "hi": float(hi[i]), "separated_from_0": bool(lo[i] > 0 or hi[i] < 0)}
                                                 for i, n in enumerate(names)}
    # head peakedness: how close to one-hot is each (head, query) attention row?
    res["head_peakedness"] = {}
    for s in SEEDS:
        res["head_peakedness"][s] = {}
        for lv in LEVELS:
            for k in sorted(k for k in Mg if k.startswith(f"{s}__att_{lv}")):
                A = Mg[k]                                              # [N, H, Q, S]
                mx = A.max(-1)                                         # [N, H, Q]
                am = A.argmax(-1).reshape(len(A), -1)                  # [N, H*Q]
                nd = np.array([len(np.unique(r)) for r in am])
                res["head_peakedness"][s][f"{lv}_{k.split('att_')[1]}"] = {
                    "mean_max_weight_per_head_query": float(mx.mean()), "frac_rows_max_gt_0.9": float((mx > 0.9).mean()),
                    "frac_rows_max_gt_0.5": float((mx > 0.5).mean()), "mean_distinct_argmax_tokens_of_16": float(nd.mean()),
                    "frac_argmax_on_cond_token": float((am < len(COND_NAMES[lv])).mean())}
    # seed agreement: same contrast, both seeds
    for k in res["contrasts"][SEEDS[0]]:
        res["seed_agreement"][k] = {n: {"s0": res["contrasts"][SEEDS[0]][k][n]["mean"], "s1": res["contrasts"][SEEDS[1]][k][n]["mean"],
                                        "same_sign": bool(np.sign(res["contrasts"][SEEDS[0]][k][n]["mean"]) == np.sign(res["contrasts"][SEEDS[1]][k][n]["mean"]))}
                                    for n in res["contrasts"][SEEDS[0]][k]}
    # spatial-map agreement between seeds (Spearman over 40 cells of the mean tick-frame map)
    from scipy.stats import spearmanr
    res["seed_map_spearman_tick_frame"] = {lv: float(spearmanr(np.ravel(res["seeds"][SEEDS[0]][lv]["mean_tick_frame_map_4x10"]),
                                                              np.ravel(res["seeds"][SEEDS[1]][lv]["mean_tick_frame_map_4x10"]))[0]) for lv in LEVELS}
    res["level_map_spearman_s0_tick_frame"] = {f"{a_}~{b_}": float(spearmanr(np.ravel(res["seeds"][SEEDS[0]][a_]["mean_tick_frame_map_4x10"]),
                                                                           np.ravel(res["seeds"][SEEDS[0]][b_]["mean_tick_frame_map_4x10"]))[0])
                                               for a_, b_ in (("S", "T"), ("T", "O"), ("S", "O"))}
    # eval-level sanity
    res["ade6_mean"] = {s: float(Mg[f"{s}__ade6"].mean()) for s in SEEDS}
    json.dump(res, open(f"{OUT}/fig_attention_stats.json", "w"), indent=1)
    np.savez_compressed(f"{WORK}/stats/merged_summary.npz", eid=eid, stratum=st,
                        **{f"{s}__{lv}__cond": arrs[(s, lv)][0] for s in SEEDS for lv in LEVELS},
                        **{f"{s}__{lv}__scene": arrs[(s, lv)][1].astype(np.float32) for s in SEEDS for lv in LEVELS},
                        **{f"{s}__ade6": Mg[f"{s}__ade6"] for s in SEEDS})
    print(json.dumps(ctl["seeds"], indent=1)[:1500])
    for lv in LEVELS:
        print(lv, "rand rho_all", ctl["randomisation"][lv]["spearman_all_160_tokens"]["mean"], "other-level maxabs", ctl["randomisation"][lv]["other_levels_attention_maxabs_delta"])
    return res


if __name__ == "__main__":
    main()
