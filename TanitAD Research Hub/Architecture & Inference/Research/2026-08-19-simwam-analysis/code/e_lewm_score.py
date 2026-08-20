"""E-LEWM-1 scorer — run the UNCHANGED E-TRUNK-2 probe on each ablation arm.

⭐ WHY THIS IS AN EXACT COMPARISON. The arms are scored on **the same 5,617
(clip_id, frame_idx) keys, in the same order, with the same episode-disjoint
folds** that `v6_cells`, `dino_pooled`, `C-EGO` and `C-PIXEL` were scored on. So
an arm's number sits in the same table as v6's, not merely beside it.

The probe itself is imported, never re-implemented: dual/Gram ridge with
scale-normalised Gram, inner episode-disjoint lambda selection, and the
episode-cluster bootstrap of the POOLED statistic.

Also reported per arm, as the pre-registration requires:
  * final SIGReg value and its TRAJECTORY (does it converge, or stall as v6's
    does at 7.83 since step 11k?)
  * the between/within-episode variance ratio (v6 4.56x, dino_pooled 2.47x)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

import e_trunk2_probe as P  # noqa: E402
import e_lewm_ablate as E   # noqa: E402
from tanitad.models.sigreg import SigReg  # noqa: E402

TARGETS = P.REGRESSION + P.BINARY


def frame_index_map():
    """(clip_id, frame_idx) -> row in the lewm frame cache."""
    clips = json.loads((E.CACHE / "clips.json").read_text(encoding="utf-8"))
    return {(c["clip_id"], f): c["start"] + f
            for c in clips for f in range(c["n"])}


def between_within(Z, ep):
    tot = Z.var(axis=0).sum()
    within = 0.0
    for e in set(ep):
        m = np.fromiter((x == e for x in ep), bool, len(ep))
        within += m.sum() * Z[m].var(axis=0).sum()
    within /= len(Z)
    return float((tot - within) / max(within, 1e-12))


def sigreg_of(Z, n=48, reps=20, slices=512):
    """SIGReg at v6's LIVE batch geometry, so the values are comparable to the
    trainer's own o6_sigreg (7.83) and to the reference ladder in LEWM_VS_OURS."""
    sig = SigReg(n_slices=slices)
    t = torch.from_numpy(Z.astype(np.float32))
    g = torch.Generator().manual_seed(0)
    v = [float(sig(t[torch.randperm(t.shape[0], generator=g)[:n]], generator=g))
         for _ in range(reps)]
    return float(np.mean(v))


def main() -> None:
    ref_keys = [tuple(k) for k in
                json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in ref_keys]
    folds = P.episode_folds(ep)
    fmap = frame_index_map()
    rows = np.array([fmap[k] for k in ref_keys])      # raises if a key is absent

    tgt = {}
    for line in P.TARGETS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tgt[(r["clip_id"], int(r["frame_idx"]))] = r
    eg = P.ego_features(ref_keys)
    for i, k in enumerate(ref_keys):
        row = tgt.setdefault(k, {})
        row["ego_speed"] = float(eg[i, 0])
        row["ego_accel"] = float(eg[i, 1])
        row["ego_yawrate"] = float(eg[i, 2])

    train = json.loads((SP / "e_lewm_train.json").read_text(encoding="utf-8"))
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "prereg": "PREREG_E_LEWM_1.md (e4d58be, committed BEFORE any number)",
           "n_frames": len(ref_keys), "n_episodes": len(set(ep)),
           "note": "same keys/order/folds as E-TRUNK-2 — directly comparable to "
                   "v6_cells, dino_pooled, C-EGO, C-PIXEL",
           "arms": []}

    for rec in train:
        zp = E.CACHE / rec["latents"]
        if not zp.exists():
            print(f"  [skip] {rec['arm']} s{rec['seed']}: {zp.name} absent")
            continue
        Z = np.load(zp)[rows].astype(np.float64)
        G = Z @ Z.T
        G = G / float(np.mean(np.diag(G)))
        res = {}
        for name in TARGETS:
            y = np.array([tgt.get(k, {}).get(name, np.nan) for k in ref_keys],
                         dtype=np.float64)
            ok = np.isfinite(y)
            if ok.sum() < 500:
                continue
            idx = np.nonzero(ok)[0]
            remap = {v: i for i, v in enumerate(idx)}
            sub = [np.array([remap[i] for i in f if i in remap]) for f in folds]
            pred, lams = P.dual_ridge_oof(G[np.ix_(idx, idx)], y[idx],
                                          [ep[i] for i in idx], sub)
            binary = name in P.BINARY
            pt, lo, hi = P.episode_cluster_bootstrap(
                pred, y[idx], [ep[i] for i in idx], binary)
            res[name] = {"metric": "AUC" if binary else "R2",
                         "point": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "n": int(ok.sum()),
                         "lambda_at_grid_edge": sum(
                             l in (P.LAMBDAS[0], P.LAMBDAS[-1]) for l in lams)}
        h = rec["history"]
        arm = {"arm": rec["arm"], "seed": rec["seed"],
               "n_params": rec["n_params"], "config": rec["config"],
               "sigreg_final": round(h[-1]["sigreg"], 4),
               "sigreg_first": round(h[0]["sigreg"], 4),
               "sigreg_trajectory": [(r["step"], round(r["sigreg"], 3)) for r in h],
               "pred_final": round(h[-1]["pred"], 6),
               "sigreg_on_encoded_probe_set": round(sigreg_of(Z), 4),
               "between_within": round(between_within(Z, ep), 3),
               "targets": res}
        out["arms"].append(arm)
        lg = res.get("lead_gap_m", {})
        lo_ = res.get("left_occupied", {})
        ro = res.get("right_occupied", {})
        print(f"{rec['arm']:<8} s{rec['seed']}  "
              f"sigreg {arm['sigreg_first']:7.3f} -> {arm['sigreg_final']:7.3f}  "
              f"b/w {arm['between_within']:5.2f}  |  "
              f"lead_gap {lg.get('point'):+.4f} {lg.get('ci95')}  "
              f"L {lo_.get('point'):.4f}  R {ro.get('point'):.4f}", flush=True)

    dest = SP / "e_lewm_ablate.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
