#!/usr/bin/env python
"""Separate the four tactical-decoder mechanisms with ONE forward pass per arm.

`decode_audit.py` reads the ARGMAX off the banked dumps and shows the incumbent's
lateral head is EXACTLY the majority-class predictor. That does not yet say WHY.
This tool opens the head and the latent it reads:

  * **the loss share** — the exact `cross_entropy` the tactical-label term
    carries on these windows, weighted by `w_tac_label`, against the three
    feature terms measured on the SAME windows. (⛔ the trainer NEVER logs
    `loss_lat_label` / `loss_lon_label`: `refa_v1_train.py:699-734` has no such
    key, so this is the only way to see the number at all.)
  * **RANK vs ARGMAX** — AUC of the head's own `P(TURN_L)+P(TURN_R)` and
    `1 - P(LANE_KEEP)` against (i) the geometric turn stratum (all windows) and
    (ii) the v7.2 lateral label (in-band). A collapsed ARGMAX with an intact
    RANKING is a decision-rule/prior problem; no ranking at all is deeper.
  * **the head's own geometry** — per-class bias and weight norms, the
    bias-only counterfactual prediction, logit margins, max softmax.
  * **the latent** — an episode-disjoint linear probe on the frozen `intent`
    (PCA fit on the FIT split ONLY), with a constant-only control that must read
    the base rate exactly and a shuffled-label control.
  * **nav vs vision** — every quantity is computed under `nav_true` AND
    `nav_zero`. The nav-shuffle obligation (`refa_v1.py:395`) is discharged by
    `decode_audit.py`; `nav_zero` here isolates what VISION alone buys.

⚠️ n and d are printed everywhere. n=40 labelled rows against d_intent=256 is
underpowered BY CONSTRUCTION; the geometric stratum (n=140) is the powered one.

TIER: T0 (open-loop, banked checkpoints). EVIDENCE CLASS: MEASURED.
GPU: one forward per window per arm. Yields nothing; run when the 4060 is free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

GT_TURN_DEG = 5.0
IGNORE_ID = -100


# --------------------------------------------------------------------------- #
def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney). Ties get the average rank."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = np.asarray(score)[order]
    r = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    n1 = int(y.sum())
    n0 = int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(score, y, eid, n_boot=10000, seed=0) -> dict:
    """Episode-cluster bootstrap on the AUC (CLAUDE.md decision-grade interval).
    Point estimate is the FULL-SET AUC; the bootstrap only supplies the interval."""
    score, y, eid = np.asarray(score), np.asarray(y), np.asarray(eid)
    uniq = np.unique(eid)
    idx = {e: np.where(eid == e)[0] for e in uniq}
    rng = np.random.default_rng(seed)
    pt = auc(score, y)
    b = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        v = auc(score[sel], y[sel])
        if np.isfinite(v):
            b.append(v)
    b = np.asarray(b)
    return {"auc": round(float(pt), 6),
            "lo": round(float(np.percentile(b, 2.5)), 6) if b.size else None,
            "hi": round(float(np.percentile(b, 97.5)), 6) if b.size else None,
            "n": int(y.size), "n_pos": int(np.asarray(y).astype(bool).sum()),
            "n_episodes": int(uniq.size), "n_boot_kept": int(b.size),
            "no_information_value": 0.5,
            "estimator": "episode-cluster bootstrap (percentile, full-set point)"}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce(logits: np.ndarray, y: np.ndarray) -> float:
    """`F.cross_entropy` on the rows given (already masked). nats."""
    p = softmax(logits)
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean())


def linear_probe(X, y, eid, n_pca: int, seed: int, l2: float = 1.0) -> dict:
    """Episode-disjoint leave-one-episode-out logistic probe on `intent`.

    ⛔ EVERY hyper-parameter (the PCA basis, the standardiser) is fit on the FIT
    fold ONLY — the held-out episode never touches the basis. Multinomial for
    a multi-class y, binary otherwise. Controls are the caller's job.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y, eid = np.asarray(X, float), np.asarray(y), np.asarray(eid)
    uniq = np.unique(eid)
    pred = np.full(len(y), -1, dtype=np.int64)
    prob = np.zeros(len(y), dtype=float)
    for e in uniq:
        te = eid == e
        fit = ~te
        if len(np.unique(y[fit])) < 2:
            continue
        k = int(min(n_pca, fit.sum() - 1, X.shape[1]))
        sc = StandardScaler().fit(X[fit])
        pc = PCA(n_components=k, random_state=seed).fit(sc.transform(X[fit]))
        Zf = pc.transform(sc.transform(X[fit]))
        Zt = pc.transform(sc.transform(X[te]))
        clf = LogisticRegression(C=1.0 / l2, max_iter=2000,
                                 random_state=seed).fit(Zf, y[fit])
        pred[te] = clf.predict(Zt)
        pp = clf.predict_proba(Zt)
        if pp.shape[1] == 2:
            prob[te] = pp[:, list(clf.classes_).index(1)] \
                if 1 in clf.classes_ else pp[:, -1]
        else:
            prob[te] = pp.max(axis=1)
    ok = pred >= 0
    out = {"n": int(len(y)), "d_in": int(X.shape[1]), "n_pca": int(n_pca),
           "l2": l2, "n_scored": int(ok.sum()),
           "n_episodes_folds": int(uniq.size),
           "accuracy": round(float((pred[ok] == y[ok]).mean()), 6) if ok.any() else None,
           "note": ("leave-one-EPISODE-out; PCA + standardiser fit on the FIT "
                    "fold only; n_pca <= n_fit-1")}
    if len(np.unique(y)) == 2 and ok.any():
        out["auc"] = round(auc(prob[ok], y[ok] == 1), 6)
    # per-class recall for the multi-class case
    if ok.any():
        cls = sorted(set(y.tolist()))
        out["recall_by_class"] = {
            int(c): (round(float((pred[ok][y[ok] == c] == c).mean()), 6)
                     if (y[ok] == c).any() else None) for c in cls}
        out["support_by_class"] = {int(c): int((y[ok] == c).sum()) for c in cls}
    return out, pred, prob


# --------------------------------------------------------------------------- #
def collect(a) -> dict:
    import torch
    sys.path.insert(0, a.arm_tool_dir)
    import refav1_arm as ARM                                  # noqa: N812
    from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v

    dev = a.device
    model, cfg, prov = ARM.load_model(a.ckpt, a.config, dev, False)
    k_wm = int(cfg.op_steps)
    names = ARM.episode_names(a.cache)[:int(a.episodes_n)] if a.episodes_n \
        else ARM.episode_names(a.cache)
    ld = ARM.build_loader(a, cfg, k_wm, names)
    vv = getattr(cfg, "tac_vocab_version", "v6.0")
    lat_names = list(tactical_lat_actions(vv))
    lon_names = list(tactical_lon_actions_v(vv))
    stride = max(1, int(a.window_stride))
    sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
           if (t - (ld.W - 1)) % stride == 0]
    print(f"[probe] {a.name}: {len(sel)} windows / {len(ld.names)} episodes; "
          f"vocab={vv}; d_intent={model.lat_head[0].normalized_shape[0]}",
          flush=True)

    rows: dict[str, list] = {kk: [] for kk in (
        "eid", "t", "nav", "v0", "lat_label", "lon_label",
        "lat_logits", "lon_logits", "intent",
        "lat_logits_nav0", "lon_logits_nav0", "intent_nav0",
        "loss_feat_op", "loss_feat_tac", "loss_feat_str", "gt_turn_deg")}
    by_ep: dict[int, list] = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))

    t0 = time.time()
    for fi, ei in enumerate(sorted(by_ep)):
        nm = ld.names[ei]
        o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        poses = o["poses"].float()
        _F, v_ep, _k = ld._episode(nm)
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(dev)
            fut = b["future_feats"].to(dev)
            act = b["actions"].to(dev)
            v0 = float(v_ep[2 * t])
            v0_t = torch.tensor([v0], dtype=torch.float32, device=dev)
            nav_i = int(b["nav_cmd"][0]) if b.get("nav_cmd") is not None else 0
            nav_t = torch.tensor([nav_i], device=dev)
            nav_z = torch.zeros(1, dtype=torch.long, device=dev)
            with torch.no_grad():
                out = model(feats, act[:, :k_wm], future_feats=fut[:, :k_wm],
                            nav_cmd=nav_t, v0=v0_t)
                out0 = model(feats, act[:, :k_wm], future_feats=fut[:, :k_wm],
                             nav_cmd=nav_z, v0=v0_t)
            rows["eid"].append(fi)
            rows["t"].append(int(t))
            rows["nav"].append(nav_i)
            rows["v0"].append(v0)
            rows["lat_label"].append(int(b["lat_label"][0]))
            rows["lon_label"].append(int(b["lon_label"][0]))
            for tag, oo in (("", out), ("_nav0", out0)):
                rows[f"lat_logits{tag}"].append(
                    out_np := oo["lat_logits"][0].float().cpu().numpy())
                rows[f"lon_logits{tag}"].append(
                    oo["lon_logits"][0].float().cpu().numpy())
                rows[f"intent{tag}"].append(oo["intent"][0].float().cpu().numpy())
            del out_np
            for kk in ("loss_feat_op", "loss_feat_tac", "loss_feat_str"):
                rows[kk].append(float(out[kk]))
            rows["gt_turn_deg"].append(_gt_turn_deg(poses, t, int(a.horizon_k)))
        print(f"  ep{fi:03d} {nm} done ({time.time()-t0:.1f}s)", flush=True)

    d = {kk: np.asarray(v) for kk, v in rows.items()}
    d["_meta"] = {"ckpt": a.ckpt, "step": prov.get("step"),
                  "config_source": prov.get("config_source"),
                  "lat_names": lat_names, "lon_names": lon_names,
                  "w_tac_label": float(getattr(cfg, "w_tac_label", 0.0)),
                  "w_str_label": float(getattr(cfg, "w_str_label", 0.0)),
                  "w_feat_op": float(cfg.w_feat_op),
                  "w_feat_tac": float(cfg.w_feat_tac),
                  "w_feat_str": float(cfg.w_feat_str),
                  "nav_inject": bool(getattr(cfg, "nav_inject", False)),
                  "d_intent": int(cfg.tactical_cfg["d_intent"]
                                  if isinstance(cfg.tactical_cfg, dict)
                                  else cfg.tactical_cfg.d_intent),
                  "wallclock_s": round(time.time() - t0, 1)}
    # the head's own geometry, straight off the weights
    W_lat = model.lat_head[-1].weight.detach().float().cpu().numpy()
    b_lat = model.lat_head[-1].bias.detach().float().cpu().numpy()
    W_lon = model.lon_head[-1].weight.detach().float().cpu().numpy()
    b_lon = model.lon_head[-1].bias.detach().float().cpu().numpy()
    d["_head"] = {
        "lat": {"bias": b_lat.tolist(),
                "row_norms": np.linalg.norm(W_lat, axis=1).tolist(),
                "argmax_bias_only": lat_names[int(b_lat.argmax())],
                "W_fro": float(np.linalg.norm(W_lat))},
        "lon": {"bias": b_lon.tolist(),
                "row_norms": np.linalg.norm(W_lon, axis=1).tolist(),
                "argmax_bias_only": lon_names[int(b_lon.argmax())],
                "W_fro": float(np.linalg.norm(W_lon))},
    }
    return d


def _gt_turn_deg(poses, t: int, k: int) -> float:
    """Verbatim from `cost_surface_probe._gt_turn_deg` — the same stratum."""
    import math
    j0, j1 = 2 * t, 2 * (t + k)
    if j1 >= poses.shape[0]:
        return float("nan")
    y0, y1 = float(poses[j0, 2]), float(poses[j1, 2])
    return math.degrees(abs(math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))))


# --------------------------------------------------------------------------- #
def analyse(d: dict, n_boot: int, seed: int, n_pca: int) -> dict:
    meta, head = d["_meta"], d["_head"]
    lat_names, lon_names = meta["lat_names"], meta["lon_names"]
    eid = d["eid"]
    y_lat, y_lon = d["lat_label"], d["lon_label"]
    m = y_lat != IGNORE_ID
    gt = d["gt_turn_deg"]
    turns = np.isfinite(gt) & (gt >= GT_TURN_DEG)
    i_keep = lat_names.index("LANE_KEEP")
    i_tl, i_tr = lat_names.index("TURN_L"), lat_names.index("TURN_R")

    out: dict = {"meta": meta, "head_geometry": head,
                 "n_windows": int(eid.size), "n_in_band": int(m.sum()),
                 "d_intent": meta["d_intent"],
                 "underpowered_note": (
                     f"in-band n={int(m.sum())} vs d_intent={meta['d_intent']}: "
                     f"n << d BY CONSTRUCTION. The powered panel is the "
                     f"geometric stratum (n={int(eid.size)}, "
                     f"n_pos={int(turns.sum())})."),
                 }

    # ---- 1. THE LOSS SHARE (the number no training log carries) ----------- #
    share = {}
    for cond, tag in (("nav_true", ""), ("nav_zero", "_nav0")):
        ce_lat = ce(d[f"lat_logits{tag}"][m], y_lat[m])
        ce_lon = ce(d[f"lon_logits{tag}"][m], y_lon[m])
        term = meta["w_tac_label"] * 0.5 * (ce_lat + ce_lon)
        feat = (meta["w_feat_op"] * d["loss_feat_op"].mean()
                + meta["w_feat_tac"] * d["loss_feat_tac"].mean()
                + meta["w_feat_str"] * d["loss_feat_str"].mean())
        share[cond] = {
            "ce_lat_nats": round(ce_lat, 6), "ce_lon_nats": round(ce_lon, 6),
            "w_tac_label": meta["w_tac_label"],
            "weighted_tactical_label_term": round(float(term), 8),
            "weighted_feature_terms_sum": round(float(feat), 8),
            "components": {
                "w_feat_op*loss_feat_op":
                    round(float(meta["w_feat_op"] * d["loss_feat_op"].mean()), 8),
                "w_feat_tac*loss_feat_tac":
                    round(float(meta["w_feat_tac"] * d["loss_feat_tac"].mean()), 8),
                "w_feat_str*loss_feat_str":
                    round(float(meta["w_feat_str"] * d["loss_feat_str"].mean()), 8),
            },
            "tactical_label_share_of_those_terms_pct":
                round(float(100.0 * term / (term + feat)), 4),
            "n_in_band_rows_scored": int(m.sum()),
            "caveat": ("the trainer logs NO loss_lat_label/loss_lon_label key "
                       "(refa_v1_train.py:699-734), so no run's telemetry "
                       "carries this; feature terms are the mean over THESE "
                       "windows, not the training stream"),
        }
    out["loss_share"] = share

    # ---- 2. label entropy: what a marginal-matching head would pay -------- #
    def _H(y):
        c = np.bincount(y, minlength=len(lat_names)).astype(float)
        p = c[c > 0] / c.sum()
        return float(-(p * np.log(p)).sum())
    out["label_entropy_nats_in_band"] = {
        "lat": round(_H(y_lat[m]), 6), "lon": round(_H(y_lon[m]), 6),
        "note": ("the CE a head that outputs the MARGINAL and nothing else "
                 "would attain; CE at or above this = no information used")}

    # ---- 3. RANK vs ARGMAX ------------------------------------------------ #
    rank = {}
    for cond, tag in (("nav_true", ""), ("nav_zero", "_nav0")):
        P = softmax(d[f"lat_logits{tag}"])
        p_turn = P[:, i_tl] + P[:, i_tr]
        p_nonkeep = 1.0 - P[:, i_keep]
        blk = {
            "argmax_histogram": {lat_names[i]: int(c) for i, c in enumerate(
                np.bincount(P.argmax(1), minlength=len(lat_names))) if c},
            "max_softmax_median": round(float(np.median(P.max(1))), 6),
            "P_LANE_KEEP_median": round(float(np.median(P[:, i_keep])), 6),
            "P_TURN_sum_median": round(float(np.median(p_turn)), 8),
            "auc_P_TURN_vs_geometric_turn":
                boot_auc(p_turn, turns, eid, n_boot, seed),
            "auc_1minusP_LANE_KEEP_vs_geometric_turn":
                boot_auc(p_nonkeep, turns, eid, n_boot, seed),
        }
        lab_turn = np.isin(y_lat, [i_tl, i_tr])
        if m.sum() and lab_turn[m].any() and not lab_turn[m].all():
            blk["auc_P_TURN_vs_label_TURN_in_band"] = boot_auc(
                p_turn[m], lab_turn[m], eid[m], n_boot, seed)
        lab_nonkeep = (y_lat != i_keep) & m
        if m.sum() and lab_nonkeep[m].any() and not lab_nonkeep[m].all():
            blk["auc_1minusP_LANE_KEEP_vs_label_nonLANE_KEEP_in_band"] = boot_auc(
                p_nonkeep[m], lab_nonkeep[m], eid[m], n_boot, seed)
        rank[cond] = blk
    out["rank_vs_argmax"] = rank

    # ---- 4. the latent: episode-disjoint linear probe ---------------------- #
    probes = {}
    for cond, tag in (("nav_true", ""), ("nav_zero", "_nav0")):
        X = d[f"intent{tag}"]
        p_geo, pred_geo, prob_geo = linear_probe(
            X, turns.astype(int), eid, n_pca, seed)
        base = float(turns.mean())
        p_geo["constant_only_base_rate"] = round(base, 6)
        p_geo["constant_only_accuracy"] = round(max(base, 1 - base), 6)
        rng = np.random.default_rng(seed)
        shuf = [linear_probe(X, rng.permutation(turns.astype(int)), eid,
                             n_pca, seed)[0]["accuracy"] for _ in range(5)]
        p_geo["shuffled_label_accuracy_5draws"] = [round(float(s), 6)
                                                   for s in shuf]
        blk = {"geometric_turn_binary": p_geo}
        if m.sum() >= 10:
            p_lab, _, _ = linear_probe(X[m], y_lat[m], eid[m], n_pca, seed)
            c = np.bincount(y_lat[m], minlength=len(lat_names))
            p_lab["constant_only_base_rate"] = round(float(c.max() / c.sum()), 6)
            p_lab["classes"] = {int(i): lat_names[i] for i in range(len(c)) if c[i]}
            blk["v72_lat_label_multiclass"] = p_lab
        probes[cond] = blk
    out["intent_linear_probe"] = probes

    # ---- 5. nav-only ceiling ---------------------------------------------- #
    nav = d["nav"]
    tab = {}
    for nv in sorted(set(nav.tolist())):
        sub = m & (nav == nv)
        if sub.any():
            c = np.bincount(y_lat[sub], minlength=len(lat_names))
            tab[int(nv)] = {"n": int(sub.sum()),
                            "lat_label_hist": {lat_names[i]: int(x)
                                               for i, x in enumerate(c) if x},
                            "majority": lat_names[int(c.argmax())]}
    # leave-one-episode-out nav-only classifier
    correct, n = 0, 0
    for e in np.unique(eid[m]):
        te = m & (eid == e)
        fit = m & (eid != e)
        for i in np.where(te)[0]:
            f2 = fit & (nav == nav[i])
            if not f2.any():
                continue
            c = np.bincount(y_lat[f2], minlength=len(lat_names))
            n += 1
            correct += int(int(c.argmax()) == int(y_lat[i]))
    out["nav_only_ceiling"] = {
        "per_nav_label_table_in_band": tab,
        "leave_one_episode_out_accuracy": round(correct / n, 6) if n else None,
        "n_scored": int(n),
        "note": ("nav is an ORACLE (ego-future) input. This is the accuracy a "
                 "predictor that reads NOTHING BUT nav attains on the same "
                 "rows — the ceiling any nav-echo can reach.")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes-n", type=int, default=20)
    ap.add_argument("--window-stride", type=int, default=10)
    ap.add_argument("--horizon-k", type=int, default=10)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-pca", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm-tool-dir", default=None,
                    help="dir holding refav1_arm.py (taniteval/tools)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None)
    a = ap.parse_args()
    if a.arm_tool_dir is None:
        raise SystemExit("--arm-tool-dir is required (taniteval/tools)")

    d = collect(a)
    if a.npz:
        np.savez_compressed(a.npz, **{k: v for k, v in d.items()
                                      if not k.startswith("_")})
    res = analyse(d, a.n_boot, a.seed, a.n_pca)
    res["tool"] = "intent_logit_probe.py"
    res["tier"] = "T0"
    res["evidence_class"] = "MEASURED"
    res["arm"] = a.name
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["inputs"] = {"ckpt": a.ckpt, "config": a.config, "cache": a.cache,
                     "labels": a.labels, "nav": a.nav,
                     "episodes_n": a.episodes_n, "window_stride": a.window_stride}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in
                      ("loss_share", "label_entropy_nats_in_band",
                       "rank_vs_argmax", "nav_only_ceiling")},
                     indent=1)[:6000])
    print(json.dumps(res["intent_linear_probe"], indent=1)[:3000])
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
