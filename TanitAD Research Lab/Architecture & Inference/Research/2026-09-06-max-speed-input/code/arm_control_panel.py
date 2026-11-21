#!/usr/bin/env python3
"""THE MANDATORY EGO-ONLY ARM CONTROL, across the four metric families.

⛔⛔ WHAT THIS IS, STATED BEFORE ANY NUMBER. NO MODEL RUNS HERE. These "arms" are
LINEAR READOUTS on scalar inputs, so:
  * a T-tier stamp would be a CATEGORY ERROR. T0 is a world-model diagnostic and
    T1 is self-action open loop; neither describes a least-squares fit on two
    scalars. Every number below is an INPUT-CHANNEL INFORMATION PROBE.
  * it measures ORACLE TRANSFER -- how much of each family's target the channel
    hands over for free -- not driving performance, and not whether the wired
    model gets better. The real arm (train refcv3 with the flag ON vs OFF) is a
    GPU job and is named as the integration item, not silently substituted for.
  * ⚠️ THE LONGITUDINAL TARGET IS CONTAMINATED BY CONSTRUCTION FOR THE RAW ARM:
    ``v_hi`` IS the max of the ego's future speed over 2-6 s, so "raw beats ego"
    on future speed is a tautology. That is exactly the point -- the size of the
    tautology IS the leak -- but it must never be read as a capability claim.

THE CONTROL THE BRIEF REQUIRES. ARM_EGO uses ``v0`` and nothing else. If the
max-speed arms match it, the channel is an echo. Precedent: the linear
vision->target-speed readout separates WORSE than repeating ``v0`` (0.2379 vs
0.4078), so "adds a channel" and "adds information" are different claims.

FOUR FAMILIES, NEVER POOLED. Each is reported separately with its estimator, its
n, and its CI. A family that cannot be computed is reported WITH ITS REASON.
  LONGITUDINAL  future speed at tau = 2/4/6 s; along-track displacement at tau.
                DISTANCE-KEEPING / TTC: NOT COMPUTABLE here -- it needs the
                `obstacle.offline` agent boxes, which are not in the v8 label
                record; reported as such with n = 0.
  LATERAL       |dyaw| at tau, curvature at tau, yaw-rate at tau, cross-track
                (lateral offset in the t0 ego frame) at tau.
  TACTICAL      a_tac.lon (7-way) and a_tac.lat (5-way) token accuracy.
  STRATEGIC     a_str.token accuracy over the 8-30 s band.

ESTIMATOR. Paired episode-cluster bootstrap (`taniteval.ci`), eid = clip_id.
⛔ NEVER `overlapping_holdout_se`.
⛔⛔ NAME THE VARIANCE QUESTION: one window per clip, so the cluster IS the clip
and the interval answers **"how much would this difference move if we drew a
different sample of CLIPS from this corpus?"** It says NOTHING about seed
variance or split variance, and a separated CI from a one-seed arm is NECESSARY,
NOT SUFFICIENT -- a pure replicate produced "separated" differences on 6 of 42
cells (14.3 %) in this programme.

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, "C:/Users/Admin/tanitad-wt/taniteval")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/taniteval")

from tanitad.data import egomotion_source as ES   # noqa: E402
from tanitad.refs import max_speed_input as MSI   # noqa: E402
from taniteval import ci as CI                    # noqa: E402

HZ = 10.0
TAUS = (2.0, 4.0, 6.0)


# --------------------------------------------------------------- target build

def build_targets(labels_gz: Path, join_json: Path) -> dict:
    join = json.load(open(join_json, encoding="utf-8"))
    assert join["units"] == "m_s"
    by_clip = {r["clip_id"]: r for r in join["rows"]}
    lab = {}
    for line in gzip.open(labels_gz, "rt", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            lab[r["clip_id"]] = r
    cols: dict[str, list] = {k: [] for k in
                             ("clip", "v0", "v_hi", "lon_tok", "lat_tok", "str_tok")}
    for t in TAUS:
        for k in ("spd", "dx", "dyaw", "kappa", "yawrate", "cross"):
            cols[f"{k}_{int(t)}s"] = []
    for cid, jr in sorted(by_clip.items()):
        r = lab.get(cid)
        if r is None:
            continue
        tr = ES.load(cid)
        p, key = tr.poses, tr.key_index
        need = key + int(round(max(TAUS) * HZ))
        if need >= len(p):
            continue
        c, s = np.cos(-p[key, 2]), np.sin(-p[key, 2])
        dx_all = p[:, 0] - p[key, 0]
        dy_all = p[:, 1] - p[key, 1]
        ex = c * dx_all - s * dy_all
        ey = s * dx_all + c * dy_all
        cols["clip"].append(cid)
        cols["v0"].append(float(p[key, 3]))
        cols["v_hi"].append(float(jr["v_hi_ms"]))
        at = r.get("a_tac") or {}
        cols["lon_tok"].append(at.get("lon") or "?")
        cols["lat_tok"].append(at.get("lat") or "?")
        cols["str_tok"].append(((r.get("a_str") or {}).get("token")) or "?")
        for t in TAUS:
            i = key + int(round(t * HZ))
            cols[f"spd_{int(t)}s"].append(float(p[i, 3]))
            cols[f"dx_{int(t)}s"].append(float(ex[i]))
            cols[f"dyaw_{int(t)}s"].append(float(p[i, 2] - p[key, 2]))
            cols[f"kappa_{int(t)}s"].append(float(tr.curvature[i]))
            cols[f"yawrate_{int(t)}s"].append(float(p[i, 3] * tr.curvature[i]))
            cols[f"cross_{int(t)}s"].append(float(ey[i]))
    out = {k: (np.array(v) if k != "clip" else np.array(v, dtype=object))
           for k, v in cols.items()}
    return out


# ----------------------------------------------------------------- the arms

def arm_features(v0, v_hi):
    """d is printed with every arm: an arm comparison without d is a comparison
    of two different model capacities wearing the same name."""
    q, over = MSI.quantize_up_array(v_hi)
    qn = q / MSI.V_SCALE_MS
    rng = np.random.default_rng(20260906)
    perm = rng.permutation(len(v_hi))
    return {
        "ARM_EGO_only_v0":        np.column_stack([v0]),
        "ARM_EGO_plus_QUANTIZED": np.column_stack([v0, qn, over.astype(float)]),
        "ARM_EGO_plus_RAW":       np.column_stack([v0, v_hi]),
        "CONTROL_shuffled_quant": np.column_stack([v0, qn[perm],
                                                   over.astype(float)[perm]]),
        "CONTROL_constant_only":  np.zeros((len(v0), 0)),
    }


def _kfold_reg(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    p = np.empty(len(y))
    for f in np.array_split(order, k):
        tr = np.setdiff1d(order, f)
        A = np.column_stack([X[tr], np.ones(len(tr))])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[f] = np.column_stack([X[f], np.ones(len(f))]) @ beta
    return p


def _kfold_clf(X, y_idx, n_cls, k=5, seed=0):
    """Multinomial one-vs-rest ridge on the same features -- deliberately the
    SAME capacity as the regressor so an accuracy gap is about the CHANNEL, not
    about a fancier head."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_idx))
    p = np.empty(len(y_idx), int)
    for f in np.array_split(order, k):
        tr = np.setdiff1d(order, f)
        A = np.column_stack([X[tr], np.ones(len(tr))])
        S = np.zeros((len(f), n_cls))
        Atest = np.column_stack([X[f], np.ones(len(f))])
        for c in range(n_cls):
            t = (y_idx[tr] == c).astype(float)
            beta = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ t)
            S[:, c] = Atest @ beta
        p[f] = S.argmax(axis=1)
    return p


def run(labels_gz: Path, join_json: Path, out_json: Path) -> dict:
    T = build_targets(labels_gz, join_json)
    n = len(T["clip"])
    eid = np.array([str(c) for c in T["clip"]])          # cluster == clip
    feats = arm_features(T["v0"], T["v_hi"])
    doc: dict = {"units": "m_s", "n_clips": int(n),
                 "tier": ("NONE -- no model runs. A T0/T1 stamp would be a "
                          "category error; these are linear readouts on scalars, "
                          "an input-channel information probe."),
                 "estimator": "paired_episode_cluster_bootstrap (taniteval.ci)",
                 "cluster_unit": "clip (one window per clip)",
                 "variance_question": ("how much would this difference move if a "
                                       "different sample of CLIPS were drawn from "
                                       "this corpus? NOT seed variance, NOT split "
                                       "variance."),
                 "arm_dims": {k: int(v.shape[1]) for k, v in feats.items()},
                 "families": {}}

    # ---- regression families: LONGITUDINAL and LATERAL ---------------------
    reg_targets = {
        "LONGITUDINAL": [("future_speed_%ds_ms" % int(t), f"spd_{int(t)}s")
                         for t in TAUS]
                        + [("alongtrack_disp_%ds_m" % int(t), f"dx_{int(t)}s")
                           for t in TAUS],
        "LATERAL": [("dyaw_%ds_rad" % int(t), f"dyaw_{int(t)}s") for t in TAUS]
                   + [("curvature_%ds_invm" % int(t), f"kappa_{int(t)}s") for t in TAUS]
                   + [("yawrate_%ds_rads" % int(t), f"yawrate_{int(t)}s") for t in TAUS]
                   + [("crosstrack_%ds_m" % int(t), f"cross_{int(t)}s") for t in TAUS],
    }
    for fam, targets in reg_targets.items():
        doc["families"][fam] = {"metric": "absolute error (lower better)",
                                "n": int(n), "targets": {}}
        for label, col in targets:
            y = T[col]
            errs = {a: np.abs(_kfold_reg(X, y) - y) for a, X in feats.items()}
            ent = {"per_arm": {}, "vs_ego": {}}
            for a, e in errs.items():
                b = CI.episode_cluster_bootstrap(e, eid, reduce="mean")
                ent["per_arm"][a] = {"mean_abs_err": float(e.mean()),
                                     "ci": [b["lo"], b["hi"]],
                                     "n_clusters": b["n_episodes"],
                                     "d": int(feats[a].shape[1])}
            for a in ("ARM_EGO_plus_QUANTIZED", "ARM_EGO_plus_RAW",
                      "CONTROL_shuffled_quant"):
                pb = CI.paired_episode_cluster_bootstrap(
                    errs[a], errs["ARM_EGO_only_v0"], eid)
                ent["vs_ego"][a] = {"delta": pb["delta"], "ci": [pb["lo"], pb["hi"]],
                                    "separated": bool(pb["separated"])}
            doc["families"][fam]["targets"][label] = ent

    # DISTANCE-KEEPING / TTC: reported WITH ITS REASON and n, never omitted.
    doc["families"]["LONGITUDINAL"]["distance_keeping_ttc"] = {
        "computable": False, "n": 0,
        "reason": ("needs `obstacle.offline` agent boxes to form a gap; those "
                   "boxes are not in the v8 label record and the join is a "
                   "separate artifact (`taniteval/lead_source.py`). "
                   "`tanitad.eval.constraints` computes frac_over / "
                   "frac_under_when_allowed and the clearance-derived ceiling "
                   "from a MODEL trajectory, which no arm here produces.")}

    # ---- classification families: TACTICAL and STRATEGIC -------------------
    for fam, col in (("TACTICAL_lon", "lon_tok"), ("TACTICAL_lat", "lat_tok"),
                     ("STRATEGIC", "str_tok")):
        toks = sorted(set(T[col].tolist()))
        idx = np.array([toks.index(t) for t in T[col]])
        maj = float(np.bincount(idx).max() / n)
        ent = {"metric": "accuracy (higher better)", "n": int(n),
               "n_classes": len(toks), "majority_baseline": maj,
               "classes": toks, "per_arm": {}, "vs_ego": {}}
        hits = {}
        for a, X in feats.items():
            if X.shape[1] == 0:
                hits[a] = (idx == np.bincount(idx).argmax()).astype(float)
            else:
                hits[a] = (_kfold_clf(X, idx, len(toks)) == idx).astype(float)
            b = CI.episode_cluster_bootstrap(hits[a], eid, reduce="mean")
            ent["per_arm"][a] = {"accuracy": float(hits[a].mean()),
                                 "ci": [b["lo"], b["hi"]],
                                 "n_clusters": b["n_episodes"],
                                 "d": int(X.shape[1])}
        for a in ("ARM_EGO_plus_QUANTIZED", "ARM_EGO_plus_RAW",
                  "CONTROL_shuffled_quant"):
            pb = CI.paired_episode_cluster_bootstrap(hits[a],
                                                     hits["ARM_EGO_only_v0"], eid)
            ent["vs_ego"][a] = {"delta": pb["delta"], "ci": [pb["lo"], pb["hi"]],
                                "separated": bool(pb["separated"])}
        doc["families"][fam] = ent

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def _p(d):
    print("n_clips = %d   TIER: %s" % (d["n_clips"], d["tier"]))
    print("arm dims d: %s" % json.dumps(d["arm_dims"]))
    print("estimator: %s  (cluster = %s)" % (d["estimator"], d["cluster_unit"]))
    for fam, blk in d["families"].items():
        print()
        print("=" * 78)
        print("FAMILY %s   n=%d   metric=%s" % (fam, blk["n"], blk["metric"]))
        if "targets" in blk:
            for label, e in blk["targets"].items():
                ego = e["per_arm"]["ARM_EGO_only_v0"]
                print("  %-26s EGO d=%d  %.4f [%.4f, %.4f]"
                      % (label, ego["d"], ego["mean_abs_err"], ego["ci"][0], ego["ci"][1]))
                for a, v in e["vs_ego"].items():
                    print("      %-24s delta %+.4f [%+.4f, %+.4f] %s"
                          % (a.replace("ARM_EGO_plus_", "+"), v["delta"],
                             v["ci"][0], v["ci"][1],
                             "SEPARATED" if v["separated"] else "not separated"))
            dk = blk.get("distance_keeping_ttc")
            if dk:
                print("  distance-keeping/TTC: NOT COMPUTABLE  n=%d  reason: %s"
                      % (dk["n"], dk["reason"][:120]))
        else:
            print("  classes=%d  majority baseline %.4f" % (blk["n_classes"],
                                                            blk["majority_baseline"]))
            for a, v in blk["per_arm"].items():
                print("    %-26s d=%d acc %.4f [%.4f, %.4f]"
                      % (a, v["d"], v["accuracy"], v["ci"][0], v["ci"][1]))
            for a, v in blk["vs_ego"].items():
                print("      %-24s delta %+.4f [%+.4f, %+.4f] %s"
                      % (a.replace("ARM_EGO_plus_", "+"), v["delta"], v["ci"][0],
                         v["ci"][1], "SEPARATED" if v["separated"] else "not separated"))


if __name__ == "__main__":
    _p(run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
