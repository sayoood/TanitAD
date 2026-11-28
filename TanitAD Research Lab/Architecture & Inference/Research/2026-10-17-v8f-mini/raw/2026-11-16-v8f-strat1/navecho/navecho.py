"""navecho.py -- D-STRAT1-TACLAT-F1: is C0's higher lateral-head macro-F1 a nav echo, or lateral skill the text nav costs?

CPU only, no training, nothing re-dumped; reads the banked T1-kin dumps in place.  Pure numpy except the landed `taniteval.ci` bootstrap.
Reuses (never edits) `fam/fam_extra.py` (confusion, conf_stats, conf_block, conf_pair, v9_rows, load_dump) and `strat1_lib`.

Tier of every number: T1-kin tick 0 (the lateral head's argmax at each window's first tick, dead-reckoned nav held from t0, logged scene) scored against the
v9 label lat_cls (5-way; labels may use ego, the heads and the nav-only predictor see only what the arm is fed).
Estimator: paired / unpaired episode-cluster bootstrap (taniteval.ci), B 2,000, seed 0 ("another draw of EPISODES" only); training-run variance = floor F.

nav_in layout (v8f_mini_data.py:219-222 for v9, nav_text_arrays :390-394 for text):
  channels 0-2 one-hot (FOLLOW, TURN_L, TURN_R), all-zero row = no nav row; channel 3 = distance/100 clipped [0, 3]; channel 4 = distance valid.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAM = os.environ.get("NAVECHO_FAM_DIR", "F:/TanitAD-scratch/strat1/fam")
READ = os.environ.get("STRAT1_READ_DIR", "F:/TanitAD-scratch/strat1/read")
for _p in (FAM, READ, READ + "/tree/taniteval", READ + "/tree/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import fam_extra as F  # noqa: E402
import strat1_lib as L  # noqa: E402

LAT9, IGNORE, NB, SEED = F.LAT9, F.IGNORE, L.NB, L.SEED
TURN_L, TURN_R = 1, 2
NO_ROW = 3                                              # key of an all-zero nav row
DIST_EDGES_M = (30.0, 60.0, 100.0)                      # extended predictor: distance bins (m) of channel 3 x 100
TIER = ("T1-kin tick 0 (lateral head argmax at the window's first tick under the T1 feed: dead-reckoned nav, logged scene) vs the v9 label lat_cls (5-way); "
        "the nav-only predictor sees only nav_in at that tick")
EST = f"episode-cluster bootstrap (taniteval.ci, stat / paired stat), B={NB}, seed {SEED}, 141 episodes; 'another draw of EPISODES' only; training-run variance = floor F = |C0_rep - C0_s0|"
FAMILIES = ("C0", "N1", "S1", "S1d")
SEEDS = ("s0", "s1", "s10", "s11")
GAP_PAIRS = (("S1", "C0"), ("N1", "C0"), ("S1d", "C0"))
ECHO_RATIO = 0.90
N_RANDOM_SPLITS = 20


# ------------------------------------------------------------------------------------------------------------------ nav keys
def nav_key(nav):
    """[N,5] nav_in -> int key: 0 FOLLOW, 1 TURN_L, 2 TURN_R, 3 no nav row (all-zero one-hot)."""
    nav = np.asarray(nav, np.float64)
    k = nav[:, :3].argmax(1)
    return np.where(nav[:, :3].sum(1) == 0, NO_ROW, k).astype(np.int64)


def nav_key_dist(nav):
    """extended key: token key x distance bin (invalid distance = bin 0; then 1 + searchsorted over DIST_EDGES_M)."""
    nav = np.asarray(nav, np.float64)
    d = nav[:, 3] * 100.0
    b = np.where(nav[:, 4] == 1, 1 + np.searchsorted(np.asarray(DIST_EDGES_M), d, side="right"), 0)
    return nav_key(nav) * (len(DIST_EDGES_M) + 2) + b.astype(np.int64)


def informative(nav):
    """the nav token at the window's first tick names a turn"""
    return np.isin(nav_key(nav), (TURN_L, TURN_R))


# ------------------------------------------------------------------------------------------------------------------ nav-only predictor
def fit_map(key, y, K=5, ignore=IGNORE):
    """majority label per key among labelled windows (ties -> lowest class); returns (dict key -> class, global majority)"""
    key, y = np.asarray(key), np.asarray(y)
    m = (y != ignore) & (y >= 0) & (y < K)
    if not m.any():
        return {}, 0
    glob = int(np.bincount(y[m], minlength=K).argmax())
    mp = {}
    for k in np.unique(key[m]):
        mp[int(k)] = int(np.bincount(y[m & (key == k)], minlength=K).argmax())
    return mp, glob


def apply_map(mp, glob, key):
    return np.asarray([mp.get(int(k), glob) for k in np.asarray(key)], np.int64)


def episode_halves(eid, seed=0):
    """two disjoint halves of the EPISODES: boolean mask of half A (random permutation of the unique ids, seed)"""
    u = np.unique(np.asarray(eid))
    rng = np.random.default_rng(seed)
    a = set(rng.permutation(u)[: len(u) // 2].tolist())
    return np.asarray([int(e) in a for e in np.asarray(eid)])


def macro_f1(y, p, K=5):
    return float(F.conf_stats(F.confusion(y, p, K))["macro_f1"])


def crossfit(key, y, eid, seed=0, K=5):
    """fit on one half of the episodes, score on the other, and swap.  -> out-of-fold predictions (every window predicted by a map fit WITHOUT its own episode) and the per-half F1s"""
    key, y = np.asarray(key), np.asarray(y)
    A = episode_halves(eid, seed)
    pred = np.zeros(len(y), np.int64)
    out = {}
    for name, fit, sc in (("fit_A_score_B", A, ~A), ("fit_B_score_A", ~A, A)):
        mp, glob = fit_map(key[fit], y[fit], K)
        pred[sc] = apply_map(mp, glob, key[sc])
        out[name] = {"macro_f1": macro_f1(np.where(sc, y, IGNORE), pred, K), "n_scored": int(((y != IGNORE) & sc).sum()),
                     "map": {str(k): LAT9[v] for k, v in sorted(mp.items())}}
    out["mean"] = 0.5 * (out["fit_A_score_B"]["macro_f1"] + out["fit_B_score_A"]["macro_f1"])
    out["pooled_oof_macro_f1"] = macro_f1(y, pred, K)
    return out, pred, A


def random_split_spread(key, y, eid, n=N_RANDOM_SPLITS, K=5):
    v = [crossfit(key, y, eid, seed=s, K=K)[0]["mean"] for s in range(1, n + 1)]
    return {"n_splits": n, "min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))}


def shuffled_nav(key, y, eid, n=N_RANDOM_SPLITS, K=5):
    """nav permuted across windows (so across episodes): the cross-fit F1 must read near the majority baseline"""
    rng = np.random.default_rng(SEED)
    v = [crossfit(rng.permutation(np.asarray(key)), y, eid, seed=0, K=K)[0]["mean"] for _ in range(n)]
    return {"n_permutations": n, "mean": float(np.mean(v)), "max": float(np.max(v)), "min": float(np.min(v))}


def majority_f1(y, K=5):
    y = np.asarray(y)
    m = (y != IGNORE) & (y >= 0) & (y < K)
    maj = int(np.bincount(y[m], minlength=K).argmax())
    return macro_f1(y, np.full(len(y), maj), K)


# ------------------------------------------------------------------------------------------------------------------ loading
def tick0(ticks_npz, clip_index, ws):
    """lateral logits [N,5] and nav_in [N,5] at each window's FIRST tick, in the dump's window order (refuses a missing / duplicate tick-0 row)"""
    z = np.load(ticks_npz)
    m = np.asarray(z["tick"]) == 0
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(z["ci"])[m], np.asarray(z["t0"])[m]))}
    if len(key) != int(m.sum()):
        raise RuntimeError(f"{ticks_npz}: duplicate (ci, t0) among tick-0 rows")
    try:
        rows = np.asarray([key[(int(c), int(t))] for c, t in zip(clip_index, ws)])
    except KeyError as e:
        raise RuntimeError(f"{ticks_npz}: a dump window has no tick-0 row: {e}") from None
    if len(rows) != int(m.sum()):
        raise RuntimeError(f"{ticks_npz}: {int(m.sum())} tick-0 rows for {len(rows)} dump windows")
    return (np.asarray(z["t_lat_logits"])[m][rows].astype(np.float64), np.asarray(z["nav_in"])[m][rows].astype(np.float64))


# ------------------------------------------------------------------------------------------------------------------ splits and the head gap
def masked(y, mask):
    return np.where(np.asarray(mask), np.asarray(y), IGNORE)


def split_masks(nav_v9, nav_text):
    """windows by the nav feeds at tick 0.  UNINF_BOTH = neither the v9 nor the text feed names a turn (the committed read); UNINF_V9 = C0's own feed names none."""
    iv, it = informative(nav_v9), informative(nav_text)
    return {"UNINF_BOTH": ~iv & ~it, "INF_ANY": iv | it, "UNINF_V9": ~iv, "INF_V9": iv}


def split_counts(y, eid, masks):
    y, eid = np.asarray(y), np.asarray(eid)
    out = {}
    for nme, m in masks.items():
        lab = m & (y != IGNORE)
        out[nme] = {"n_windows": int(m.sum()), "n_labelled": int(lab.sum()), "n_episodes_labelled": int(len(np.unique(eid[lab]))),
                    "label_support": {LAT9[c]: int((y[lab] == c).sum()) for c in range(5)}}
    return out


def net_correct(y, p_a, p_b, mask):
    """windows where a is right and b wrong minus the reverse, on labelled windows of the mask (additive, so the splits sum to the whole)"""
    y = np.asarray(y)
    m = np.asarray(mask) & (y != IGNORE)
    ra, rb = np.asarray(p_a) == y, np.asarray(p_b) == y
    return int((m & ra & ~rb).sum()) - int((m & ~ra & rb).sum())


def reading(ratios, gap_uninf):
    """the committed rules.  ratios: {s0, s1} nav-only / head F1 on C0's feed.  gap_uninf: {s0, s1} -> paired dict of S1 - C0 on UNINF_BOTH (needs 'separated', 'delta')."""
    echo_nav = all(ratios[s] >= ECHO_RATIO for s in ("s0", "s1"))
    sep = {s: bool(gap_uninf[s]["separated"]) for s in ("s0", "s1")}
    same_sign = np.sign(gap_uninf["s0"]["delta"]) == np.sign(gap_uninf["s1"]["delta"])
    if echo_nav and not any(sep.values()):
        return "ECHO"
    if all(sep.values()) and same_sign:
        return "SKILL-COST"
    return "MIXED"


# ------------------------------------------------------------------------------------------------------------------ run
def run(dumps, out_dir, nb=NB, log=print):
    from taniteval import ci as CI
    t_start = time.time()
    join = dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False))
    ref = F.load_dump(F.dump_dir(dumps, "C0_s0"))
    eid, ci_, ws = ref["eid"], ref["clip_index"], ref["ws"]
    V = F.v9_rows(join, ci_, ws)
    y = V["lat"]
    landed = json.load(open(os.path.join(FAM, "out", "RESULT_FAM_EXTRA.json"), encoding="utf-8"))
    arms = [f"{f}_{s}" for f in FAMILIES for s in SEEDS] + ["C0_rep"]
    A = {}
    for a in arms:
        lg, nav = tick0(F.ticks_path(dumps, a), ci_, ws)
        A[a] = {"pred": lg.argmax(1), "nav": nav}
    res = {"tool": "navecho.py", "question": "D-STRAT1-TACLAT-F1: nav echo or skill cost", "evidence_class": "MEASURED (ours; banked T1-kin dumps, zero training, CPU)",
           "tier": TIER, "estimator": EST, "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "controls": {"problems": []}}
    prob = res["controls"]["problems"]
    # ---- feeds
    feed_of = {a: ("v9" if a.startswith("C0") else "text") for a in arms}
    for fam in ("C0",):
        for a in arms:
            if a.startswith("C0") and not np.array_equal(A[a]["nav"], A["C0_s0"]["nav"]):
                prob.append(f"K-FEED: {a} nav_in differs from C0_s0")
    for a in arms:
        if not a.startswith("C0") and not np.array_equal(A[a]["nav"], A["N1_s0"]["nav"]):
            prob.append(f"K-FEED: {a} nav_in differs from N1_s0")
    nav_v9, nav_txt = A["C0_s0"]["nav"], A["N1_s0"]["nav"]
    res["feeds"] = {"v9_nav (C0 arms)": {"n_FOLLOW": int((nav_key(nav_v9) == 0).sum()), "n_TURN_L": int((nav_key(nav_v9) == 1).sum()), "n_TURN_R": int((nav_key(nav_v9) == 2).sum()),
                                         "n_no_row": int((nav_key(nav_v9) == NO_ROW).sum())},
                    "text_nav (N1, S1, S1d)": {"n_FOLLOW": int((nav_key(nav_txt) == 0).sum()), "n_TURN_L": int((nav_key(nav_txt) == 1).sum()), "n_TURN_R": int((nav_key(nav_txt) == 2).sum()),
                                               "n_no_row": int((nav_key(nav_txt) == NO_ROW).sum())},
                    "n_windows_feeds_differ_in_token": int((nav_key(nav_v9) != nav_key(nav_txt)).sum())}
    masks = split_masks(nav_v9, nav_txt)
    res["splits"] = {"definition": {"UNINF_BOTH": "neither the v9 nor the text nav token at tick 0 names a turn (FOLLOW or no row): the committed read, the only set where no arm has route information",
                                    "INF_ANY": "at least one feed names a turn (complement of UNINF_BOTH)",
                                    "UNINF_V9": "C0's own v9 token names no turn", "INF_V9": "C0's own v9 token names a turn"},
                     "counts": split_counts(y, eid, masks)}
    # ---- head macro-F1 (full set) against the landed number
    head = {}
    for a in arms:
        head[a] = macro_f1(y, A[a]["pred"])
        lf = landed["families"]["TACTICAL"][a]["TAC-SEL-LAT"]["macro_f1"]["point"]
        if abs(head[a] - lf) > 1e-12:
            prob.append(f"K-LANDED: {a} macro-F1 {head[a]!r} != landed {lf!r}")
    # ---- union control: the split confusions sum to the whole
    ok_union = True
    for a in arms:
        full = F.confusion(y, A[a]["pred"], 5)
        for p, q in (("UNINF_BOTH", "INF_ANY"), ("UNINF_V9", "INF_V9")):
            s = F.confusion(masked(y, masks[p]), A[a]["pred"], 5) + F.confusion(masked(y, masks[q]), A[a]["pred"], 5)
            ok_union &= bool(np.array_equal(s, full))
    res["controls"]["K-UNION"] = {"pass": ok_union, "reading": "confusion(split) + confusion(complement) equals the full-set confusion for every arm; full-set macro-F1 equals the landed RESULT_FAM_EXTRA.json to 1e-12"}
    if not ok_union:
        prob.append("K-UNION: the split confusions do not sum to the full confusion")
    # ---- the nav-only predictor, per feed
    nop = {}
    for feed, nav in (("v9", nav_v9), ("text", nav_txt)):
        kt, kd = nav_key(nav), nav_key_dist(nav)
        cf, _, _ = crossfit(kt, y, eid, seed=0)
        cfd, _, _ = crossfit(kd, y, eid, seed=0)
        nop[feed] = {"token_map": dict(cf, random_split_spread=random_split_spread(kt, y, eid)),
                     "token_x_distance_map (sensitivity)": dict(cfd, random_split_spread=random_split_spread(kd, y, eid)),
                     "shuffled_nav_control": shuffled_nav(kt, y, eid), "majority_baseline_macro_f1": majority_f1(y)}
    # nav-only out-of-fold predictions scored on each split (its own feed), beside the heads' split F1
    for feed, nav in (("v9", nav_v9), ("text", nav_txt)):
        _, oof, _ = crossfit(nav_key(nav), y, eid, seed=0)
        nop[feed]["token_map_oof_macro_f1_on_splits"] = {sp: macro_f1(masked(y, m), oof) for sp, m in masks.items()}
    res["nav_only_predictor"] = {"definition": "best fixed map nav key -> lat class (majority label per key); fit on one half of the EPISODES, scored on the other, halves swapped; "
                                               "key = token (FOLLOW / TURN_L / TURN_R / no row); the sensitivity adds a distance bin (channel 3 x 100 m, edges 30 / 60 / 100, invalid = own bin)",
                                 "tier": TIER, "by_feed": nop}
    # ---- head vs nav-only table, per arm
    tab = {}
    for a in arms:
        feed = feed_of[a]
        Ah = episode_halves(eid, 0)
        hh = {"half_A": macro_f1(masked(y, Ah), A[a]["pred"]), "half_B": macro_f1(masked(y, ~Ah), A[a]["pred"])}
        n1, nd = nop[feed]["token_map"], nop[feed]["token_x_distance_map (sensitivity)"]
        tab[a] = {"feed": feed, "head_macro_f1": head[a], "head_macro_f1_on_halves": hh, "nav_only_token_cross_fit_mean": n1["mean"], "nav_only_token_x_distance_cross_fit_mean": nd["mean"],
                  "ratio_nav_only_token_to_head": n1["mean"] / head[a], "ratio_nav_only_token_x_distance_to_head": nd["mean"] / head[a]}
    res["head_vs_nav_only"] = tab
    ratios = {s: tab[f"C0_{s}"]["ratio_nav_only_token_to_head"] for s in ("s0", "s1")}
    # ---- the split read: head F1 and recall, then paired gaps
    names = list(LAT9)
    blocks, gaps = {}, {}
    for sp in ("UNINF_BOTH", "INF_ANY", "UNINF_V9", "INF_V9"):
        ys = masked(y, masks[sp])
        blocks[sp] = {}
        for a in arms:
            b = F.conf_block(CI, ys, A[a]["pred"], 5, names, eid, nb)
            blocks[sp][a] = {k: b[k] for k in ("n", "n_episodes", "macro_f1", "accuracy", "support", "confusion_rows_GT_cols_pred")}
            blocks[sp][a]["recall_TURN_L"], blocks[sp][a]["recall_TURN_R"] = b["recall"]["TURN_L"], b["recall"]["TURN_R"]
        Fl = {}
        pr = F.conf_pair(CI, ys, A["C0_rep"]["pred"], A["C0_s0"]["pred"], 5, names, eid, None, nb)
        Fl = {k: abs(pr[k]["delta"]) for k in ("macro_f1", "recall_TURN_L", "recall_TURN_R") if k in pr and np.isfinite(pr[k].get("delta", np.nan))}
        gaps[sp] = {"floor_F_abs_C0rep_minus_C0s0": Fl}
        for fam, ref_ in GAP_PAIRS:
            for s in SEEDS:
                x, r = f"{fam}_{s}", f"{ref_}_{s}"
                pp = F.conf_pair(CI, ys, A[x]["pred"], A[r]["pred"], 5, names, eid, Fl, nb)
                gaps[sp][f"{x}-{r}"] = {k: pp[k] for k in ("macro_f1", "recall_TURN_L", "recall_TURN_R") if k in pp}
                gaps[sp][f"{x}-{r}"]["net_correct_windows_x_minus_ref"] = net_correct(y, A[x]["pred"], A[r]["pred"], masks[sp])
        log(f"split {sp} {time.time() - t_start:.0f}s", flush=True)
    res["split_head_blocks"] = blocks
    res["split_gaps"] = gaps
    # ---- controls: self pair reads exactly 0.0
    k1 = F.conf_pair(CI, masked(y, masks["UNINF_BOTH"]), A["C0_s0"]["pred"], A["C0_s0"]["pred"], 5, names, eid, None, nb)
    rd = lambda v: v.get("delta") == 0.0 and v.get("lo") == 0.0 and v.get("hi") == 0.0  # noqa: E731
    k1ok = bool(all(rd(v) for v in k1.values() if "delta" in v and np.isfinite(v["delta"])))
    res["controls"]["K1_self_pair"] = {"pass": k1ok, "reading": "the head against itself on UNINF_BOTH reads exactly 0.0 [0.0, 0.0] on every cell"}
    if not k1ok:
        prob.append("K1: self-pair not exactly 0.0")
    sh = nop["v9"]["shuffled_nav_control"]["mean"]
    mj = nop["v9"]["majority_baseline_macro_f1"]
    res["controls"]["K-SHUFFLE"] = {"shuffled_mean_macro_f1": sh, "majority_baseline_macro_f1": mj, "pass": bool(abs(sh - mj) <= 0.08),
                                    "reading": "a nav permuted across windows must read near the majority baseline (|diff| <= 0.08, fixed before the run)"}
    if not res["controls"]["K-SHUFFLE"]["pass"]:
        prob.append("K-SHUFFLE: shuffled nav is not near the majority baseline")
    # ---- the committed reading
    gu = {s: gaps["UNINF_BOTH"][f"S1_{s}-C0_{s}"]["macro_f1"] for s in ("s0", "s1")}
    rd_ = reading(ratios, gu)
    share = {}
    for s in ("s0", "s1"):
        tot = net_correct(y, A[f"C0_{s}"]["pred"], A[f"S1_{s}"]["pred"], np.ones(len(y), bool))
        share[s] = {"net_correct_windows_C0_minus_S1": {sp: net_correct(y, A[f"C0_{s}"]["pred"], A[f"S1_{s}"]["pred"], masks[sp]) for sp in masks}, "total": tot}
    res["reading"] = {"rule": "ECHO: nav-only >= 90 % of C0's head macro-F1 at s0 and s1 AND the S1 - C0 gap on UNINF_BOTH not separated at either; SKILL-COST: gap separated, same sign, at both; else MIXED",
                      "nav_only_ratio_C0_feed": ratios, "S1_minus_C0_on_UNINF_BOTH": gu, "verdict": rd_, "additive_share_of_gap": share,
                      "n_turn_windows_by_split": {sp: res["splits"]["counts"][sp]["label_support"] for sp in masks}}
    res["elapsed_s"] = round(time.time() - t_start, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "RESULT_NAVECHO.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=READ + "/t1")
    ap.add_argument("--out", default=HERE + "/out")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    r = run(a.dumps, a.out, a.n_boot)
    print(json.dumps({"verdict": r["reading"]["verdict"], "problems": r["controls"]["problems"], "elapsed_s": r["elapsed_s"]}))
