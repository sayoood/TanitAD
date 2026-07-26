"""Situation classifier — STEP 4 (dev box, CPU): the held-out evaluation. Read ONCE.

Every interval is a **paired episode-cluster bootstrap** (B=2000, seed 0), resampling CLIPS with
replacement, both arms recomputed inside the same draw. The estimator machinery is IMPORTED from
`2026-07-26-h2-classifier/scripts/h2c_stats.py`, which itself imports `taniteval/taniteval/ci.py`
(`episode_index`, `_draws`) — never re-implemented here, because two independent re-implementations
produced the nulls that were overturned on 2026-07-26. ⛔ `overlapping_holdout_se` appears nowhere.

Produces `artifacts/sc_results.json` with, per situation:
  * discrimination + the PAIRED above-chance test (vs a constant score, whose AP equals the base
    rate INSIDE every draw — the correct test, not "does the AP interval clear the full-sample rate")
  * the operating point at the TRAIN-fixed theta*, and every baseline (a)-(e)
  * ⭐ the LEAD-TIME distribution — the requirement a high-AP zero-lead trigger must fail
  * the per-camera NEED, with a matched non-situation baseline
  * the efficiency curve: recall at a fixed camera budget, beside (never instead of) the saving
  * C-POS / C-NEG / C-CHANCE / C-BLIND / C-POW, each with its MDE

usage:  python sc_eval.py --run <run_dir> --bundle <bundle> --cross <cross_dir> --out <artifacts>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time



import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..",
                                                "2026-07-26-h2-classifier", "scripts")))
sys.path.insert(0, HERE)
from h2c_stats import ESTIMATOR, average_precision, roc_auc  # noqa: E402
import sc_situations as S  # noqa: E402
from taniteval.ci import _draws, episode_index  # noqa: E402

SITS = ("lane_change", "roundabout", "intersection")
B_STAR = 0.05
BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
N_BOOT = 2000
BLIND_MAX_ROWS = 40_000       # C-BLIND subsamples whole clip clusters; see the call site
EXTRA_CAMS = ["camera_cross_left_120fov", "camera_cross_right_120fov",
              "camera_front_tele_30fov", "camera_rear_left_70fov",
              "camera_rear_right_70fov", "camera_rear_tele_30fov"]


# ------------------------------------------------------------------------------- bootstrap driver
_DRAWS: dict = {}


def set_draws(eid, n_boot=N_BOOT, seed=0):
    """Materialise the episode-cluster draws ONCE per situation; every statistic reuses them.

    ⚠️ This is a performance detail with no statistical content: the draws come from
    `taniteval.ci._draws` (imported, never re-implemented) with the same `uniq`, `idx_by_ep` and
    seed a fresh call would use. Reusing one set is also exactly what makes every interval in this
    document **paired** — across arms, across metrics and across budgets.

    Two costs are avoided here that made the first version unusable: `taniteval.ci.episode_index`
    is O(n_clips x n_frames) (1,610 x 165 k = 266 M comparisons) so it is called once, not once per
    statistic; and the draw indices are stored as **int32**, halving 2.6 GB to 1.3 GB.
    """
    _DRAWS.clear()
    uniq, idx_by_ep = episode_index(eid)
    _DRAWS["d"] = [d.astype(np.int32) for d in _draws(uniq, idx_by_ep, n_boot, seed)]
    _DRAWS["n"] = n_boot
    return _DRAWS["d"]


def draws_for(eid=None, n_boot=None, seed=0):
    return _DRAWS["d"]


def paired(fn, eid, arms: dict, n_boot=N_BOOT, seed=0, alpha=0.05):
    """fn(**arrays)->float on each arm inside the SAME draw. -> {arm: (point, lo, hi)} + boots."""
    pts = {k: float(fn(**v)) for k, v in arms.items()}
    boots = {k: [] for k in arms}
    for sel in draws_for(eid, n_boot, seed):
        for k, v in arms.items():
            boots[k].append(fn(**{kk: a[sel] for kk, a in v.items()}))
    out = {}
    for k in arms:
        b = np.array([x for x in boots[k] if np.isfinite(x)], float)
        out[k] = dict(point=pts[k], lo=float(np.quantile(b, alpha / 2)) if len(b) else None,
                      hi=float(np.quantile(b, 1 - alpha / 2)) if len(b) else None,
                      n_boot=int(len(b)))
    return out, boots


def delta(boots, a, b, alpha=0.05):
    d = np.array(boots[a], float) - np.array(boots[b], float)
    d = d[np.isfinite(d)]
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    return dict(delta=float(np.mean(d)), lo=lo, hi=hi, separated=bool(lo > 0 or hi < 0))


def _ap(y, s):
    return average_precision(y, s)


# ------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--cross", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Z = np.load(os.path.join(a.run, "scores.npz"), allow_pickle=True)
    L = np.load(os.path.join(a.bundle, "sc_labels.npz"))
    meta = json.load(open(os.path.join(a.bundle, "sc_meta.json")))
    tsum = json.load(open(os.path.join(a.run, "train_summary.json")))
    uni = json.load(open(os.path.join(a.bundle, "universe.json")))

    te, tr = Z["te_idx"], Z["tr_idx"]
    clip, Y, V, E = Z["clip"], Z["Y"], Z["V"], Z["E"]
    off, T_, kk = Z["off"], Z["T"], Z["k"]
    eid_te = clip[te]
    arms = [k for k in Z.files if k.startswith(("head_", "ridge_")) and "__trainoof" not in k]
    print(f"[eval] {len(te):,} held-out windows, {len(np.unique(eid_te))} clip clusters, "
          f"arms: {arms}")

    # ---------- baseline (e): the one-line kinematic rules, PRE-REG Sec 5.3 ----------
    v, alon_pre, omega_pre = E[:, 0], E[:, 1], E[:, 2]
    lat_rate = np.abs(omega_pre) * np.maximum(v, 1e-3)     # |v * yaw-rate| = lateral drift rate
    HEUR = {"lane_change": lat_rate, "roundabout": -v, "intersection": -alon_pre}

    res = {"estimator": ESTIMATOR, "n_boot": N_BOOT, "B_star": B_STAR,
           "min_useful_lead_s": S.MIN_USEFUL_LEAD_S, "lead_s": S.LEAD_S,
           "train_selected": tsum["selected"], "situations": {}}

    for si, sit in enumerate(SITS):
        m = V[te][:, si]
        y = Y[te][:, si][m].astype(float)
        eid = eid_te[m]
        n_clu = int(len(np.unique(eid[y > 0])))
        base = float(y.mean())
        R = {"n_windows": int(m.sum()), "n_pos": int(y.sum()), "n_pos_clusters": n_clu,
             "base_rate": base, "n_clusters": int(len(np.unique(eid))),
             "C_POW": "OK" if n_clu >= 40 else "UNDERPOWERED"}

        t_s = time.time()
        set_draws(eid, N_BOOT, 0)
        print(f"[eval] {sit}: {N_BOOT} episode-cluster draws built in "
              f"{time.time()-t_s:.0f}s over {len(np.unique(eid))} clusters", flush=True)

        sc = {arm: Z[arm][m][:, si] for arm in arms}
        sc["heur_kin"] = HEUR[SITS[si]][te][m]
        sc["constant"] = np.zeros(m.sum())                 # C-CHANCE
        # all three one-line rules against every situation, so the choice cannot flatter the head
        for nm, hv in HEUR.items():
            sc[f"heur_{nm}"] = hv[te][m]

        pack = {k: dict(y=y, s=v_) for k, v_ in sc.items()}
        t_s = time.time()
        disc, boots = paired(lambda y, s: _ap(y, s), eid, pack)
        print(f"[eval] {sit}: discrimination over {len(pack)} arms in {time.time()-t_s:.0f}s",
              flush=True)
        R["AP"] = {k: dict(**disc[k], ap_over_base=round(disc[k]["point"] / max(base, 1e-12), 4),
                           auroc=round(float(roc_auc(y, sc[k])), 4)) for k in sc}
        R["above_chance"] = {k: delta(boots, k, "constant") for k in sc if k != "constant"}
        R["vs_head_ego"] = {k: delta(boots, k, "head_ego") for k in sc
                            if k not in ("head_ego", "constant")}
        R["vs_ridge_ego"] = {k: delta(boots, k, "ridge_ego") for k in sc
                             if k in ("ridge_img_ego", "ridge_img", "head_img_ego")} \
            if "ridge_ego" in sc else {}
        R["vs_heur_kin"] = {k: delta(boots, k, "heur_kin") for k in sc
                            if k.startswith(("head_", "ridge_"))}

        # ---------- operating point: theta* fixed on TRAIN OUT-OF-FOLD only ----------
        R["operating_point"] = {}
        R["efficiency_curve"] = {}
        R["lead_time"] = {}
        for arm in list(arms) + ["heur_kin"]:
            t_s = time.time()
            oof = (Z[arm + "__trainoof"][:, si] if arm + "__trainoof" in Z.files
                   else HEUR[SITS[si]][tr])
            mt = V[tr][:, si]
            thr = float(np.quantile(oof[mt], 1.0 - B_STAR / 2.0))   # 2 cams per firing frame
            fire = sc[arm] >= thr
            R["operating_point"][arm] = _op(y, fire, eid, thr)
            R["lead_time"][arm] = _lead(Z, L, meta, sc[arm], thr, te, m, si, sit)
            R["efficiency_curve"][arm] = [
                _op(y, sc[arm] >= float(np.quantile(oof[mt], 1.0 - b / 2.0)), eid,
                    float(np.quantile(oof[mt], 1.0 - b / 2.0)), budget=b) for b in BUDGETS]
            print(f"[eval]   {sit}/{arm}: operating point + curve in {time.time()-t_s:.0f}s",
                  flush=True)

        # ---------- baselines (a) (b) (c) ----------
        rate = R["operating_point"]["head_img_ego"]["firing_rate"]["point"]
        rng = np.random.default_rng(0)
        rr = []
        for _ in range(200):
            f = rng.random(len(y)) < rate
            rr.append(dict(recall=float(y[f].sum() / max(y.sum(), 1)),
                           precision=float(y[f].mean()) if f.any() else 0.0))
        R["baselines"] = {
            "always_escalate": {"recall": 1.0, "firing_rate": 1.0, "precision": base},
            "never_escalate": {"recall": 0.0, "firing_rate": 0.0, "precision": None},
            "random_at_matched_rate": {
                "n_seeds": 200, "matched_rate": rate,
                "recall_mean": float(np.mean([x["recall"] for x in rr])),
                "recall_p2.5": float(np.quantile([x["recall"] for x in rr], .025)),
                "recall_p97.5": float(np.quantile([x["recall"] for x in rr], .975)),
                "precision_mean": float(np.mean([x["precision"] for x in rr]))},
            "oracle": {"recall": 1.0, "firing_rate": base, "precision": 1.0}}
        # paired recall delta vs a random score at the matched rate, inside the same draws
        rs = rng.random(len(y))
        rthr = float(np.quantile(rs, 1.0 - rate))
        pk = {"head": dict(y=y, f=(sc["head_img_ego"] >= R["operating_point"]
                                   ["head_img_ego"]["theta"]).astype(float)),
              "rand": dict(y=y, f=(rs >= rthr).astype(float)),
              "ego": dict(y=y, f=(sc["head_ego"] >= R["operating_point"]
                                  ["head_ego"]["theta"]).astype(float))}
        if "ridge_img_ego" in sc:
            pk["ridge"] = dict(y=y, f=(sc["ridge_img_ego"] >= R["operating_point"]
                                       ["ridge_img_ego"]["theta"]).astype(float))
        _r, rb = paired(lambda y, f: float((y * f).sum() / max(y.sum(), 1e-9)), eid, pk)
        R["paired_recall_delta"] = {
            "head_img_ego - random_at_rate": delta(rb, "head", "rand"),
            "head_img_ego - head_ego": delta(rb, "head", "ego"),
            **({"ridge_img_ego - random_at_rate": delta(rb, "ridge", "rand"),
                "ridge_img_ego - head_ego": delta(rb, "ridge", "ego")} if "ridge" in pk else {})}

        R["controls"] = {
            "C_POS_head_priv": R["above_chance"].get("head_priv"),
            "C_NEG_head_img_shuf": R["above_chance"].get("head_img_shuf"),
            "C_NEG_ridge_img_shuf": R["above_chance"].get("ridge_img_shuf"),
            "MDE_from_C_NEG": max(
                (R["above_chance"].get(k, {}).get("hi") or 0.0)
                for k in ("head_img_shuf", "ridge_img_shuf") if k in R["above_chance"]),
        }
        # ---------- C-BLIND: the PACKAGED firewall, IMPORTED, never re-implemented ----------
        try:
            from taniteval.blind_baseline import blind_conditioning_baseline
            ee = E[te][m]
            ctx_all = {"v_bin": np.digitize(ee[:, 0], np.arange(0, 30, 2.0)).astype(np.int64),
                       "alon_bin": np.digitize(ee[:, 1], np.arange(-4, 4, 0.5)).astype(np.int64),
                       "omega_bin": np.digitize(ee[:, 2], np.arange(-0.6, 0.6, 0.05)).astype(np.int64)}
            # The firewall fits 8 MLPs + its own bootstrap on CPU; at 165 k rows that dominates the
            # whole evaluation. Subsample WHOLE CLIP CLUSTERS (never frames) to keep the episode-
            # clustered split meaningful, and declare the subsample in the record.
            keep = np.ones(len(y), bool)
            cl = np.unique(eid)
            if len(y) > BLIND_MAX_ROWS:
                rg = np.random.default_rng(0)
                take = set(rg.choice(cl, size=max(40, int(len(cl) * BLIND_MAX_ROWS / len(y))),
                                     replace=False).tolist())
                keep = np.array([e in take for e in eid])
            fw = blind_conditioning_baseline(
                {k_: v_[keep] for k_, v_ in ctx_all.items()}, y[keep].astype(np.int64), eid[keep],
                real_pred=(sc["head_img_ego"][keep] > 0.5).astype(np.int64),
                problem=f"situation:{sit}", n_boot=400)
            R["C_BLIND"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in fw.items() if not isinstance(v, np.ndarray)}
            R["C_BLIND"]["subsample"] = {"rows": int(keep.sum()), "of": int(len(y)),
                                         "clusters": int(len(np.unique(eid[keep]))),
                                         "unit": "whole clip clusters, never frames"}
            # ⭐ The binding rule: state a control's MDE against the effect it exists to catch, and
            # prove it CAN fail. The firewall's CIRCULAR branch fires at
            # blind_accuracy >= 1 - deterministic_eps. On a target whose positive rate is p, the
            # majority-class predictor already scores 1 - p, so whenever p < deterministic_eps the
            # branch fires for ANY context, including one with zero information.
            pr = float(y[keep].mean())
            eps = float(fw["thresholds"]["deterministic_eps"])
            R["C_BLIND"]["MDE_AUDIT"] = {
                "positive_rate": round(pr, 5),
                "majority_accuracy": round(1 - pr, 5),
                "deterministic_eps": eps,
                "max_possible_accuracy_gain_over_majority": round(pr, 5),
                "branch_fires_for_any_context": bool((1 - pr) >= 1 - eps),
                "blind_skill_over_majority": fw.get("blind_skill_over_majority"),
                "reading": ("DEGENERATE — the CIRCULAR branch cannot fail on this target: the "
                            "majority-class predictor alone clears 1 - deterministic_eps, and the "
                            "largest accuracy gain any context could add is the positive rate "
                            "itself. Read `blind_skill_over_majority` and `context_leaks` instead; "
                            "the informative form of this question on a rare-positive target is "
                            "the AP-based `vs_head_ego` contrast, which is the pre-registered "
                            "primary comparison.")
                if (1 - pr) >= 1 - eps else "the branch can fail on this target; verdict is readable"}
        except Exception as exc:                                   # noqa: BLE001
            R["C_BLIND"] = {"status": f"NOT RUN: {type(exc).__name__}: {exc}"}

        res["situations"][sit] = R
        print(f"[eval] {sit}: base {base:.5f}, {int(y.sum())} pos, {n_clu} clusters -> "
              f"{R['C_POW']} | C-BLIND {R['C_BLIND'].get('verdict', R['C_BLIND'].get('status'))}",
              flush=True)

    # ---------- multi-camera need + the turn-vs-curve validation ----------
    if a.cross and os.path.isdir(a.cross):
        res["camera_need"] = _camera_need(a.cross, L, meta, Z)
    # ⚠️ The Sec 6.2 turn-vs-curve validation lives in `sc_validate_labels.py` (V4) and is NOT
    # recomputed here. It has its own population (turn frames vs curve frames), therefore its own
    # episode-cluster draws — reusing this evaluator's cached per-situation draws against it was a
    # real bug (IndexError, caught in the harness smoke) and, silently, would have been a wrong
    # resampling unit. One measurement, one place.
    res["turn_is_junction_see"] = "artifacts/label_validation.json :: V4_turn_is_junction"
    res["universe"] = uni["per_situation"]
    res["roundabout_ccw_purity"] = uni["roundabout_ccw_purity"]
    json.dump(res, open(os.path.join(a.out, "sc_results.json"), "w"), indent=2)

    # ---- per-frame dump so EVERY bar in this study is recomputable with NO GPU ----
    # (`*.parquet` is git-ignored in this repo; `.npz` is not, and is exact rather than rounded.)
    np.savez_compressed(
        os.path.join(a.out, "heldout_frames.npz"),
        clip_cluster=eid_te.astype(np.int32),
        situations=np.array(SITS), arms=np.array(arms),
        y=Y[te].astype(np.uint8), valid=V[te].astype(np.uint8),
        ego=E[te].astype(np.float32),
        heur_kin=np.stack([HEUR[s][te] for s in SITS], 1).astype(np.float32),
        **{arm: Z[arm].astype(np.float32) for arm in arms})
    print(f"[eval] -> {os.path.join(a.out, 'sc_results.json')} + heldout_frames.npz")


def _op(y, fire, eid, thr, budget=B_STAR):
    """Operating-point metrics. All three are computed in ONE pass over the shared draws — three
    separate `paired` calls would recompute the same resampling 3x for no statistical gain."""
    f = fire.astype(float)
    tot, hit, n = y.sum(), (y * f).sum(), float(len(y))
    b = {"recall": [], "precision": [], "firing_rate": []}
    for sel in draws_for():
        ys, fs = y[sel], f[sel]
        s = (ys * fs).sum()
        b["recall"].append(s / max(ys.sum(), 1e-9))
        b["precision"].append(s / max(fs.sum(), 1e-9))
        b["firing_rate"].append(fs.mean())
    pts = {"recall": hit / max(tot, 1e-9), "precision": hit / max(f.sum(), 1e-9),
           "firing_rate": f.sum() / n}
    o = {}
    for k, v in b.items():
        arr = np.array([x for x in v if np.isfinite(x)], float)
        o[k] = dict(point=float(pts[k]), lo=float(np.quantile(arr, .025)),
                    hi=float(np.quantile(arr, .975)), n_boot=int(len(arr)))
    o["theta"] = thr
    o["budget"] = budget
    o["extra_cams_per_frame"] = 2.0 * o["firing_rate"]["point"]
    o["precision_lift"] = (o["precision"]["point"] / max(float(y.mean()), 1e-12))
    o["n_fired"] = int(fire.sum())
    o["n_caught"] = int((y * fire).sum())
    return o


def _lead(Z, L, meta, score, thr, te, m, si, sit):
    """⭐ LEAD TIME. For every held-out event onset, the EARLIEST frame inside the label's
    positive window (o-LEAD, o] at which the score clears theta*. Events with no firing frame in
    that window are misses and contribute no lead time (they are counted, not imputed)."""
    idx = te[m]
    full = np.full(len(Z["clip"]), -np.inf)
    full[idx] = score
    Lw = int(round(S.LEAD_S * S.HZ))
    leads, n_ev, n_hit = [], 0, 0
    off, kk = Z["off"], Z["k"]
    side = Z["side"]
    for ci in range(len(off)):
        if side[off[ci]] != "HELDOUT":
            continue
        ons = L[f"c{int(kk[ci])}_onset_{sit}"]
        for o in np.asarray(ons).ravel():
            o = int(o)
            lo = max(0, o - Lw)
            if o <= lo:
                continue
            n_ev += 1
            w = full[off[ci] + lo:off[ci] + o]
            hit = np.nonzero(w >= thr)[0]
            if len(hit):
                n_hit += 1
                leads.append((o - (lo + int(hit[0]))) / S.HZ)
    leads = np.array(leads, float)
    q = {"0.1": None, "0.25": None, "0.5": None, "0.75": None, "0.9": None}
    if len(leads):
        q = {k: round(float(np.quantile(leads, float(k))), 3) for k in q}
    return {"n_events": n_ev, "n_detected": n_hit,
            "event_recall": round(n_hit / max(n_ev, 1), 4),
            "median_lead_s": round(float(np.median(leads)), 3) if len(leads) else None,
            "mean_lead_s": round(float(leads.mean()), 3) if len(leads) else None,
            "lead_quantiles_s": q,
            "frac_events_at_or_above_min_lead": round(
                float((leads >= S.MIN_USEFUL_LEAD_S).sum() / max(n_ev, 1)), 4),
            "min_useful_lead_s": S.MIN_USEFUL_LEAD_S,
            "PASS_min_lead": bool(len(leads) and np.median(leads) >= S.MIN_USEFUL_LEAD_S)}


def _camera_need(cross_dir, L, meta, Z):
    """For each situation, P(an agent projects into camera X but NOT into the front crop | in S),
    against the matched NOT-in-S baseline on the same clips."""
    C = np.load(os.path.join(cross_dir, "sc_cross.npz"))
    kk, off, T_ = Z["k"], Z["off"], Z["T"]
    out = {}
    for sit in SITS:
        rows = {c: [[], [], [], []] for c in list(EXTRA_CAMS) + ["any_off_front"]}
        n_cl = 0
        for ci in range(len(off)):
            k = int(kk[ci])
            if f"c{k}_cross" not in C.files:
                continue
            ong = L[f"c{k}_ongoing_{sit}"].astype(bool)
            if not ong.any():
                continue
            n_cl += 1
            for c in list(EXTRA_CAMS) + ["any_off_front"]:
                key = f"c{k}_need_{c}" if c != "any_off_front" else f"c{k}_any_off_front"
                if key not in C.files:
                    continue
                nd = C[key].astype(bool)
                n = min(len(nd), len(ong))
                rows[c][0].append(nd[:n][ong[:n]])
                rows[c][1].append(nd[:n][~ong[:n]])
                rows[c][2].append(np.full(int(ong[:n].sum()), k))
                rows[c][3].append(np.full(int((~ong[:n]).sum()), k))
        o = {}
        for c, (a_, b_, ae, be) in rows.items():
            if not a_:
                continue
            A = np.concatenate(a_).astype(float)
            Bv = np.concatenate(b_).astype(float) if b_ else np.zeros(0)
            # ⚠️ A bare lift is not admissible in this program. The interval is the same paired
            # episode-cluster bootstrap used everywhere else, over the clips that carry the
            # situation — the in- and not-in-situation rates are recomputed inside the SAME draw,
            # so the clip's own scene difficulty cancels.
            eid = np.concatenate([np.concatenate(ae), np.concatenate(be)])
            yv = np.concatenate([A, Bv])
            gv = np.concatenate([np.ones(len(A)), np.zeros(len(Bv))])
            set_draws(eid, 400, 0)
            bl = []
            for sel in draws_for():
                ys, gs = yv[sel], gv[sel]
                den = ys[gs < 1].mean() if (gs < 1).any() else np.nan
                num = ys[gs > 0].mean() if (gs > 0).any() else np.nan
                if np.isfinite(den) and den > 0 and np.isfinite(num):
                    bl.append(num / den)
            bl = np.array(bl, float)
            o[c] = {"in_situation": round(float(A.mean()), 5), "n_in": int(len(A)),
                    "not_in_situation": round(float(Bv.mean()), 5) if len(Bv) else None,
                    "lift": round(float(A.mean() / max(Bv.mean(), 1e-9)), 3) if len(Bv) else None,
                    "lift_ci95": ([round(float(np.quantile(bl, .025)), 3),
                                   round(float(np.quantile(bl, .975)), 3)] if len(bl) else None),
                    "separated_from_1": (bool(float(np.quantile(bl, .025)) > 1.0) if len(bl)
                                         else None),
                    "n_clusters": int(len(np.unique(eid))), "n_boot": int(len(bl))}
        out[sit] = {"n_clips": n_cl, "per_camera": o,
                    "estimator": "paired episode-cluster bootstrap (taniteval.ci._draws, B=400)"}
    return out



if __name__ == "__main__":
    main()
