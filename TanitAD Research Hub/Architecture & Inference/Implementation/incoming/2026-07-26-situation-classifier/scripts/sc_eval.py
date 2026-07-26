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
EXTRA_CAMS = ["camera_cross_left_120fov", "camera_cross_right_120fov",
              "camera_front_tele_30fov", "camera_rear_left_70fov",
              "camera_rear_right_70fov", "camera_rear_tele_30fov"]


# ------------------------------------------------------------------------------- bootstrap driver
def paired(fn, eid, arms: dict, n_boot=N_BOOT, seed=0, alpha=0.05):
    """fn(**arrays)->float on each arm inside the SAME draw. -> {arm: (point, lo, hi)} + deltas."""
    uniq, idx_by_ep = episode_index(eid)
    pts = {k: float(fn(**v)) for k, v in arms.items()}
    boots = {k: [] for k in arms}
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
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

        sc = {arm: Z[arm][m][:, si] for arm in arms}
        sc["heur_kin"] = HEUR[SITS[si]][te][m]
        sc["constant"] = np.zeros(m.sum())                 # C-CHANCE
        # all three one-line rules against every situation, so the choice cannot flatter the head
        for nm, hv in HEUR.items():
            sc[f"heur_{nm}"] = hv[te][m]

        pack = {k: dict(y=y, s=v_) for k, v_ in sc.items()}
        disc, boots = paired(lambda y, s: _ap(y, s), eid, pack)
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
            ctx = {"v_bin": np.digitize(ee[:, 0], np.arange(0, 30, 2.0)).astype(np.int64),
                   "alon_bin": np.digitize(ee[:, 1], np.arange(-4, 4, 0.5)).astype(np.int64),
                   "omega_bin": np.digitize(ee[:, 2], np.arange(-0.6, 0.6, 0.05)).astype(np.int64)}
            fw = blind_conditioning_baseline(
                ctx, y.astype(np.int64), eid, real_pred=(sc["head_img_ego"] > 0.5).astype(np.int64),
                problem=f"situation:{sit}", n_boot=400)
            R["C_BLIND"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in fw.items() if not isinstance(v, np.ndarray)}
        except Exception as exc:                                   # noqa: BLE001
            R["C_BLIND"] = {"status": f"NOT RUN: {type(exc).__name__}: {exc}"}

        res["situations"][sit] = R
        print(f"[eval] {sit}: base {base:.5f}, {int(y.sum())} pos, {n_clu} clusters -> "
              f"{R['C_POW']} | C-BLIND {R['C_BLIND'].get('verdict', R['C_BLIND'].get('status'))}",
              flush=True)

    # ---------- multi-camera need + the turn-vs-curve validation ----------
    if a.cross and os.path.isdir(a.cross):
        res["camera_need"] = _camera_need(a.cross, L, meta, Z)
        res["turn_is_junction"] = _turn_validation(a.cross, L, meta, Z)
    res["universe"] = uni["per_situation"]
    res["roundabout_ccw_purity"] = uni["roundabout_ccw_purity"]
    json.dump(res, open(os.path.join(a.out, "sc_results.json"), "w"), indent=2)

    # ---- per-frame dump so EVERY bar in the report is recomputable with no GPU ----
    # (`*.parquet` is git-ignored in this repo, so the dump is gzipped CSV.)
    import gzip
    cols = ["clip", "chunk_side"] + [f"y_{s}" for s in SITS] + [f"valid_{s}" for s in SITS] \
        + ["v", "alon_pre", "omega_pre"] \
        + [f"{arm}_{s}" for arm in arms for s in SITS]
    with gzip.open(os.path.join(a.out, "heldout_frames.csv.gz"), "wt", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        Ys, Vs, Es = Y[te], V[te], E[te]
        A = {arm: Z[arm] for arm in arms}
        for i in range(len(te)):
            row = [str(int(eid_te[i])), "HELDOUT"]
            row += [str(int(Ys[i, j])) for j in range(3)]
            row += [str(int(Vs[i, j])) for j in range(3)]
            row += [f"{Es[i, j]:.5f}" for j in range(3)]
            row += [f"{A[arm][i, j]:.6f}" for arm in arms for j in range(3)]
            f.write(",".join(row) + "\n")
    print(f"[eval] -> {os.path.join(a.out, 'sc_results.json')} + heldout_frames.csv.gz")


def _op(y, fire, eid, thr, budget=B_STAR):
    def rec(y, f):
        return float((y * f).sum() / max(y.sum(), 1e-9))

    def pre(y, f):
        return float((y * f).sum() / max(f.sum(), 1e-9))

    def rt(y, f):
        return float(f.mean())
    pk = dict(y=y, f=fire.astype(float))
    o = {}
    for nm, fn in (("recall", rec), ("precision", pre), ("firing_rate", rt)):
        d, _ = paired(lambda y, f, _fn=fn: _fn(y, f), eid, {"a": pk}, n_boot=600)
        o[nm] = d["a"]
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
        rows = {c: [[], []] for c in EXTRA_CAMS}
        rows["any_off_front"] = [[], []]
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
        o = {}
        for c, (a_, b_) in rows.items():
            if not a_:
                continue
            A, Bv = np.concatenate(a_), np.concatenate(b_) if b_ else np.zeros(0, bool)
            o[c] = {"in_situation": round(float(A.mean()), 5), "n_in": int(len(A)),
                    "not_in_situation": round(float(Bv.mean()), 5) if len(Bv) else None,
                    "lift": round(float(A.mean() / max(Bv.mean(), 1e-9)), 3) if len(Bv) else None}
        out[sit] = {"n_clips": n_cl, "per_camera": o}
    return out


def _turn_validation(cross_dir, L, meta, Z):
    """⭐ PRE-REG Sec 6.2 — does the TURN half actually mark junctions, or is it a curve detector?

    P(perpendicular cross traffic | tight-radius TURN) vs P(... | matched-heading-change LARGE-radius
    curve). If the ratio's CI includes 1.0, the turn half is NOT a junction detector."""
    C = np.load(os.path.join(cross_dir, "sc_cross.npz"))
    kk, off = Z["k"], Z["off"]
    a_hit, a_eid, b_hit, b_eid = [], [], [], []
    poses = None
    for ci in range(len(off)):
        k = int(kk[ci])
        if f"c{k}_cross" not in C.files:
            continue
        cr = C[f"c{k}_cross"].astype(bool)
        tn = np.asarray(L[f"c{k}_turn_ab"]).reshape(-1, 2)
        ong = np.zeros(len(cr), bool)
        for a_, b_ in tn:
            ong[int(a_):int(b_) + 1] = True
        cu = np.asarray(L.get(f"c{k}_curve_ab", np.zeros((0, 2)))).reshape(-1, 2)
        cmask = np.zeros(len(cr), bool)
        for a_, b_ in cu:
            cmask[int(a_):int(b_) + 1] = True
        if ong.any():
            a_hit.append(cr[ong])
            a_eid.append(np.full(int(ong.sum()), ci))
        if cmask.any():
            b_hit.append(cr[cmask])
            b_eid.append(np.full(int(cmask.sum()), ci))
    if not a_hit:
        return {"status": "no turn frames on the cross subset"}
    A = np.concatenate(a_hit)
    Ae = np.concatenate(a_eid)
    out = {"P_cross_given_turn": round(float(A.mean()), 5), "n_turn_frames": int(len(A))}
    if b_hit:
        Bv = np.concatenate(b_hit)
        Be = np.concatenate(b_eid)
        y = np.concatenate([A, Bv]).astype(float)
        g = np.concatenate([np.ones(len(A)), np.zeros(len(Bv))])
        eid = np.concatenate([Ae, Be])
        d, _ = paired(lambda y, g: float(y[g > 0].mean() / max(y[g < 1].mean(), 1e-9)),
                      eid, {"ratio": dict(y=y, g=g)})
        out["P_cross_given_large_radius_curve"] = round(float(Bv.mean()), 5)
        out["n_curve_frames"] = int(len(Bv))
        out["ratio"] = d["ratio"]
        out["separated_from_1"] = bool(d["ratio"]["lo"] > 1.0)
    return out


if __name__ == "__main__":
    main()
