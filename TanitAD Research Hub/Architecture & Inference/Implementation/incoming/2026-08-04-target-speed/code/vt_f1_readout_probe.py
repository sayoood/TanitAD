#!/usr/bin/env python3
"""D-VT1 step 3 — F1 (`--tactical-speed-input`) and the PREDICTED target speed,
measured on the banked fan. 0 GPU-days, no re-inference, no training launched.

SUBSTRATE. `…/2026-08-03-dtac1-tactical-head/dtac1_substrate_refc-base-30k.pt`
— REF-C-base step 29999's frozen `pooled` (704-d), `v0`, and the (lat, lon, man5)
labels for **1364 windows / 39 val episodes**. Its window grid is
`lead_source.window_last_indices(T, stride=5)` and the join to the pose track is
proved by CONTENT: `max |v0_substrate - v_pose[l]| = 0` over all 1364 rows —
emitted as `_join_proof` in this script's own output, and asserted at runtime so a
future substrate that does not align FAILS LOUD instead of scoring the wrong rows.
⚠️ 39 of 40 episodes: eid 809056053 is absent from the substrate, so every number
here is on 39 clusters, not 40.

THE FOUR ARMS. All read the SAME frozen `pooled`; they differ only in what is
concatenated beside it — which is exactly what `tactical_speed_input` changes
(`refc.py`: `tac_in = cat([pooled, v] + [keep])` vs `pooled`).

  A  img               the SHIPPED head's input — image embedding alone
  B  img+v0            **F1**, `--tactical-speed-input`, +384 params
  C  img+v0+vt_pred    F1 + a PREDICTED guarded target speed (deployable: the
                       predictor reads only `pooled` and `v0`, fit inside the
                       training fold, so no held-out episode informs its own goal)
  D  img+v0+vt_label   F1 + the SUPPLIED guarded label ⛔ INADMISSIBLE at
                       inference — reported only as the ceiling C is measured against

⛔ WHAT THIS IS NOT. The trunk is FROZEN. This is a READOUT-level estimate of
whether the speed channel carries decision-relevant information the image
embedding lacks — it is not a retrained F1 arm, and it cannot see a trunk that
co-adapts. It is the cheapest discriminating measurement that exists today, and
it is the evidence a GPU-day should be argued from, not a substitute for one.
⚠️ The real head trains under `ego_dropout = 0.5` (v0 zeroed on half the samples);
the probe does not. That asymmetry makes this an OPTIMISTIC read of F1.

⛔ Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, unit = val episode.
Every per-class row carries precision AND recall AND both denominators.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval.ci import paired_episode_cluster_bootstrap        # noqa: E402
from taniteval.lead_source import window_last_indices            # noqa: E402
from tanitad.lake.vtarget import VT_GUARD_STEPS, vtarget_guarded  # noqa: E402

MAN5 = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
#: the classes whose decision is LONGITUDINAL. The programme's defect lives here.
LON5 = (3, 4)
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0),
         (15.0, np.inf)]
STEPS = 400
LR = 3e-3
WD = 1e-4
SEEDS = (0, 1, 2)
RIDGE = 1e-2


# --------------------------------------------------------------------------- #
def _z(train, *rest):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return [((a - mu) / sd).astype(np.float32) for a in (train, *rest)]


def ridge_fit_predict(xtr, ytr, xte, lam=RIDGE):
    xtr, xte = _z(xtr, xte)
    a = np.column_stack([xtr, np.ones(len(xtr), dtype=np.float32)])
    b = np.column_stack([xte, np.ones(len(xte), dtype=np.float32)])
    m = a.T @ a + lam * np.eye(a.shape[1], dtype=np.float32)
    m[-1, -1] -= lam
    w = np.linalg.solve(m, a.T @ ytr.astype(np.float32))
    return b @ w


def fit_head(xtr, ytr, xte, *, hidden: int, seed: int):
    """The REAL head shape when hidden > 0 (`Linear(d,384)-ReLU-Linear(384,5)`,
    `aux_hidden = 384`), a bare multinomial logistic when hidden == 0."""
    torch.manual_seed(seed)
    xtr_z, xte_z = _z(xtr, xte)
    xt = torch.from_numpy(xtr_z)
    yt = torch.from_numpy(ytr.astype(np.int64))
    net = (torch.nn.Sequential(torch.nn.Linear(xt.shape[1], hidden),
                               torch.nn.ReLU(),
                               torch.nn.Linear(hidden, len(MAN5)))
           if hidden else torch.nn.Linear(xt.shape[1], len(MAN5)))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(STEPS):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        return net(torch.from_numpy(xte_z)).numpy()


def macro_recall(y, p, k=len(MAN5)) -> float:
    r = [float((p[y == c] == c).mean()) for c in range(k) if (y == c).any()]
    return float(np.mean(r)) if r else float("nan")


def macro_f1(y, p, k=len(MAN5)) -> float:
    f = []
    for c in range(k):
        tp = float(((p == c) & (y == c)).sum())
        pr = tp / max(float((p == c).sum()), 1e-9)
        rc = tp / max(float((y == c).sum()), 1e-9)
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def per_class(y, p) -> dict:
    out = {}
    for c, name in enumerate(MAN5):
        tp = int(((p == c) & (y == c)).sum())
        n_true, n_pred = int((y == c).sum()), int((p == c).sum())
        out[name] = {
            "n_true": n_true, "n_pred": n_pred, "tp": tp,
            "recall": round(tp / n_true, 4) if n_true else None,
            "precision": round(tp / n_pred, 4) if n_pred else None,
            "_denominators": "recall over n_true, precision over n_pred",
        }
    return out


# --------------------------------------------------------------------------- #
def main(substrate: Path, poses: Path, out_json: Path):
    t0 = time.time()
    d = torch.load(substrate, map_location="cpu", weights_only=False)
    pooled = d["pooled"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    man5 = d["man5"].numpy().astype(np.int64)
    eid = np.array(d["eid"])

    z = np.load(poses, allow_pickle=True)
    meta = json.loads(bytes(z["_meta_json"]).decode())
    P = {m["episode_id"]: z[m["file"].replace('.pt', '')
                            + "__poses"].astype(np.float64) for m in meta}

    vt, vt_ok, dv, align = (np.zeros(len(eid)), np.zeros(len(eid), bool),
                            np.zeros(len(eid)), 0.0)
    for e in dict.fromkeys(eid.tolist()):
        sel = np.where(eid == e)[0]
        pose = P[e]
        last = window_last_indices(pose.shape[0], stride=5)
        assert len(last) == len(sel), f"grid mismatch on eid {e}"
        align = max(align, float(np.abs(v0[sel] - pose[last, 3]).max()))
        a, ok, _look, _ = vtarget_guarded(pose[:, 3], last,
                                          guard_steps=VT_GUARD_STEPS,
                                          min_lookahead=50)
        vt[sel], vt_ok[sel] = a, ok
        j = np.minimum(last + VT_GUARD_STEPS, pose.shape[0] - 1)
        dv[sel] = pose[j, 3] - pose[last, 3]
    assert align == 0.0, f"substrate/pose join is not bit-exact: {align}"

    eps = list(dict.fromkeys(eid.tolist()))
    logits = {k: np.zeros((len(eid), len(MAN5))) for k in
              ("A_img", "B_img_v0", "C_img_v0_vtpred", "D_img_v0_vtlabel")}
    logits_lin = {k: np.zeros_like(v) for k, v in logits.items()}
    vt_pred = np.zeros(len(eid))
    seed_spread = {k: [] for k in logits}

    for e in eps:
        te = eid == e
        tr = ~te
        # --- nested: the goal predictor NEVER sees the held-out episode -------
        f_tr = np.column_stack([pooled[tr], v0[tr]])
        f_te = np.column_stack([pooled[te], v0[te]])
        m = vt_ok[tr]
        vt_pred[te] = ridge_fit_predict(f_tr[m], vt[tr][m], f_te)
        feats = {
            "A_img": (pooled[tr], pooled[te]),
            "B_img_v0": (f_tr, f_te),
            "C_img_v0_vtpred": (
                np.column_stack([f_tr, ridge_fit_predict(f_tr[m], vt[tr][m],
                                                         f_tr)]),
                np.column_stack([f_te, vt_pred[te]])),
            "D_img_v0_vtlabel": (np.column_stack([f_tr, vt[tr]]),
                                 np.column_stack([f_te, vt[te]])),
        }
        for k, (xtr, xte) in feats.items():
            per_seed = [fit_head(xtr, man5[tr], xte, hidden=384, seed=s)
                        for s in SEEDS]
            logits[k][te] = np.mean(per_seed, axis=0)
            seed_spread[k].append([float(macro_recall(man5[te],
                                                      p.argmax(1)))
                                   for p in per_seed])
            logits_lin[k][te] = fit_head(xtr, man5[tr], xte, hidden=0, seed=0)
        print(f"  fold eid={e} done ({time.time() - t0:.0f}s)", flush=True)

    def block(lg, tag):
        pred = {k: v.argmax(1) for k, v in lg.items()}
        res = {"_head": tag, "arms": {}, "paired_vs_A": {}}
        for k, p in pred.items():
            res["arms"][k] = {
                "macro_recall_5way": round(macro_recall(man5, p), 4),
                "macro_f1_5way": round(macro_f1(man5, p), 4),
                "accuracy": round(float((p == man5).mean()), 4),
                "lon_recall_pooled": round(
                    float(np.isin(p[np.isin(man5, LON5)],
                                  LON5).mean()), 4),
                "lon_fires": int(np.isin(p, LON5).sum()),
                "lon_true": int(np.isin(man5, LON5).sum()),
                "per_class": per_class(man5, p),
            }
        for k, p in pred.items():
            if k == "A_img":
                continue
            hit_a = (pred["A_img"] == man5).astype(float)
            hit_k = (p == man5).astype(float)
            ci = paired_episode_cluster_bootstrap(hit_k, hit_a, eid,
                                                 n_boot=2000, seed=0)
            lon = np.isin(man5, LON5)
            ci_lon = paired_episode_cluster_bootstrap(
                np.isin(p, LON5)[lon].astype(float),
                np.isin(pred["A_img"], LON5)[lon].astype(float),
                eid[lon], n_boot=2000, seed=0)
            res["paired_vs_A"][k] = {"accuracy": ci,
                                     "lon_class_recall": ci_lon}
        return res

    out = {
        "_what": ("F1 (`--tactical-speed-input`) and a PREDICTED target speed, "
                  "measured as a READOUT on REF-C-base's frozen fan"),
        "_substrate": str(substrate.name),
        "_join_proof": {"max_abs_v0_minus_pose_speed": align,
                        "_reads": "0.0 => bit-exact; the join is by CONTENT"},
        "_estimator": ("paired_episode_cluster_bootstrap (taniteval.ci), unit = "
                       "val episode, B=2000. NEVER overlapping_holdout_se"),
        "_frozen_trunk_caveat": (
            "readout-level: the trunk is FROZEN and ego_dropout is NOT applied, "
            "so this is an OPTIMISTIC estimate of what a retrained F1 arm buys, "
            "and it cannot observe trunk co-adaptation"),
        "n_windows": int(len(eid)), "n_episodes": len(eps),
        "guard_steps": VT_GUARD_STEPS,
        "vt_label_valid": int(vt_ok.sum()),
        "vt_pred_quality": {
            "r2_out_of_fold": round(
                float(1 - ((vt[vt_ok] - vt_pred[vt_ok]) ** 2).sum()
                      / ((vt[vt_ok] - vt[vt_ok].mean()) ** 2).sum()), 4),
            "mae_mps": round(float(np.abs(vt[vt_ok] - vt_pred[vt_ok]).mean()), 4),
            "r2_of_v0_alone": round(
                float(1 - ((vt[vt_ok] - v0[vt_ok]) ** 2).sum()
                      / ((vt[vt_ok] - vt[vt_ok].mean()) ** 2).sum()), 4),
            "n": int(vt_ok.sum()),
            "_reads": ("r2_of_v0_alone is the free baseline: if predicting the "
                       "target speed is no better than repeating the current "
                       "speed, the goal head is a dead parameter"),
        },
        "mlp_head_384": block(logits, "Linear(d,384)-ReLU-Linear(384,5), "
                                      "3-seed mean logits"),
        "linear_probe": block(logits_lin, "multinomial logistic (variance-free "
                                          "corroboration)"),
        "seed_spread_macro_recall_per_fold": {
            k: {"mean_within_fold_range": round(float(np.mean(
                [max(r) - min(r) for r in v])), 4)} for k, v in
            seed_spread.items()},
        "by_speed": {},
        "_runtime_s": round(time.time() - t0, 1),
    }

    pred_mlp = {k: v.argmax(1) for k, v in logits.items()}
    for lo, hi in BANDS:
        m = (v0 >= lo) & (v0 < hi)
        name = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}"
        n_ep = len(set(eid[m].tolist()))
        if m.sum() < 30 or n_ep < 5:
            out["by_speed"][name] = {
                "status": "UNPOWERED", "n": int(m.sum()), "n_episodes": n_ep,
                "reason": "<30 windows or <5 episode clusters"}
            continue
        out["by_speed"][name] = {
            "n": int(m.sum()), "n_episodes": n_ep,
            "n_true_longitudinal": int(np.isin(man5[m], LON5).sum()),
            "arms": {k: {"macro_recall_5way": round(macro_recall(man5[m],
                                                                 p[m]), 4),
                         "macro_f1_5way": round(macro_f1(man5[m], p[m]), 4),
                         "lon_recall": (round(float(np.isin(
                             p[m][np.isin(man5[m], LON5)], LON5).mean()), 4)
                             if np.isin(man5[m], LON5).any() else None),
                         "n_lon_true": int(np.isin(man5[m], LON5).sum())}
                     for k, p in pred_mlp.items()},
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for tag in ("mlp_head_384", "linear_probe"):
        print(f"[{tag}]")
        for k, a in out[tag]["arms"].items():
            print(f"  {k:20s} BA={a['macro_recall_5way']:.4f} "
                  f"F1={a['macro_f1_5way']:.4f} acc={a['accuracy']:.4f} "
                  f"lon_rec={a['lon_recall_pooled']:.4f} "
                  f"lon_fires={a['lon_fires']}/{a['lon_true']}")
    print(f"vt_pred R2={out['vt_pred_quality']['r2_out_of_fold']} "
          f"(v0-alone {out['vt_pred_quality']['r2_of_v0_alone']})")
    print(f"wrote {out_json} in {out['_runtime_s']}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
