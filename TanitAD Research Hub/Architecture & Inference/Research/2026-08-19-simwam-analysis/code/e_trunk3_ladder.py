"""E-TRUNK-3 — run the E-TRUNK-2 battery across the S-W TRAINING LADDER.

⛔ THE QUESTION, pre-registered in `PREREG_E_TRUNK_3.md` BEFORE any number here
was computed: was the environment information **LOST** to the action-conditioned
prediction objective, or **NEVER ACQUIRED**? The two demand opposite fixes —
anchor/regularise the encoder, versus freeze a strong pretrained one.

Nine banked cell caches sit on the IDENTICAL 5,617 frames, stride 4, 130
episodes. Only the checkpoint moves. 0 GPU, 0 retraining.

FALSIFIERS (from the pre-registration, enforced here):
  1. every cache must carry the SAME (clip_id, frame_idx) keys IN THE SAME ORDER
     as the reference — otherwise folds differ and steps are not comparable.
     This script REFUSES rather than reporting an incomparable ladder.
  2. C-EGO and C-PIXEL never see the checkpoint, so they are computed ONCE and
     are constant lines by construction; (1) is what makes that valid.
  3. C-EGO -> ego_speed must read ~1.0 (identity map). It read -1.81 under an
     absolute lambda grid and +0.9845 with the Gram scale-normalised, which is
     the check that caught that defect.

TIER: T0-DIAGNOSTIC. Decodability is a representation property, never driving
performance.
"""
from __future__ import annotations

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
from tanitad.models.v6 import spectrum_report, o6_rank_verdict  # noqa: E402

STEPS = (2000, 9000, 9250, 10000, 11250, 12000, 14000, 16000, 18000)
TARGETS = P.REGRESSION + P.BINARY


def load_cells(cache_dir: Path):
    obj = torch.load(cache_dir / "latents.pt", map_location="cpu",
                     weights_only=False)
    keys, vecs = [], []
    for r in obj["rows"]:
        if r.get("cells") is None:
            continue
        keys.append((r["clip_id"], int(r["frame_idx"])))
        vecs.append(r["cells"].reshape(-1).float())
    X = torch.stack(vecs).numpy()
    del vecs, obj
    return keys, X


def score(G, keys, tgt, ep, folds):
    out = {}
    for name in TARGETS:
        y_all = np.array([tgt.get(k, {}).get(name, np.nan) for k in keys],
                         dtype=np.float64)
        ok = np.isfinite(y_all)
        if ok.sum() < 500:
            continue
        idx = np.nonzero(ok)[0]
        remap = {v: i for i, v in enumerate(idx)}
        sub = [np.array([remap[i] for i in f if i in remap]) for f in folds]
        pred, lams = P.dual_ridge_oof(G[np.ix_(idx, idx)], y_all[idx],
                                      [ep[i] for i in idx], sub)
        binary = name in P.BINARY
        pt, lo, hi = P.episode_cluster_bootstrap(
            pred, y_all[idx], [ep[i] for i in idx], binary)
        out[name] = {"metric": "AUC" if binary else "R2",
                     "point": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "n": int(ok.sum()),
                     "lambda_at_grid_edge": sum(
                         l in (P.LAMBDAS[0], P.LAMBDAS[-1]) for l in lams)}
    return out


def main() -> None:
    ref_keys = [tuple(k) for k in
                json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in ref_keys]
    folds = P.episode_folds(ep)

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

    out = {"_evidence_class": "MEASURED (ours; dev-box CPU dual ridge)",
           "eval_tier": "T0-DIAGNOSTIC",
           "prereg": "PREREG_E_TRUNK_3.md (committed 548548f, BEFORE any "
                     "ladder number was computed)",
           "n_frames": len(ref_keys), "n_episodes": len(set(ep)),
           "controls": {}, "ladder": {}}

    # falsifier 2/3 — controls, computed ONCE (they never see a checkpoint)
    for arm, fn in (("C-EGO", "c_ego.npy"), ("C-PIXEL", "c_pixel.npy")):
        G = P.gram_memmap(P.FEAT / fn)
        G = G / float(np.mean(np.diag(G)))
        out["controls"][arm] = score(G, ref_keys, tgt, ep, folds)
        print(f"[control] {arm}: "
              f"lead_gap {out['controls'][arm].get('lead_gap_m',{}).get('point')} "
              f"ego_speed {out['controls'][arm].get('ego_speed',{}).get('point')}",
              flush=True)

    idm = out["controls"]["C-EGO"].get("ego_speed", {}).get("point")
    if idm is None or abs(idm - 1.0) > 0.05:
        raise RuntimeError(f"FALSIFIER 3 TRIPPED: C-EGO->ego_speed is an "
                           f"identity map and must read ~1.0, got {idm}")
    print(f"[falsifier 3] identity map reads {idm:.4f} — OK\n", flush=True)

    for st in STEPS + (20000,):
        cd = (SP / "sp2/cache_tok20000_s4" if st == 20000
              else SP / f"sp2/cache_s{st:05d}")
        keys, X = load_cells(cd)
        if keys != ref_keys:                       # falsifier 1
            raise RuntimeError(
                f"FALSIFIER 1 TRIPPED at step {st}: cache keys differ from the "
                f"reference ({len(keys)} vs {len(ref_keys)}); folds would not be "
                f"comparable across the ladder")
        Z = torch.from_numpy(X)
        rep = spectrum_report(Z, top_k=8)
        ver = o6_rank_verdict(rep)
        G = X.astype(np.float32) @ X.astype(np.float32).T
        G = G / float(np.mean(np.diag(G)))
        res = score(G, ref_keys, tgt, ep, folds)
        out["ladder"][str(st)] = {
            "spectrum": {"participation_ratio": rep["participation_ratio"],
                         "effective_rank": rep["effective_rank"],
                         "top_k_share": rep["top_k_share"],
                         "rank_ceiling": ver["rank_ceiling"],
                         "verdict": ver["status"]},
            "targets": res}
        lg = res.get("lead_gap_m", {})
        lo = res.get("left_occupied", {})
        ro = res.get("right_occupied", {})
        print(f"step {st:>6}  PR {rep['participation_ratio']:6.2f}  "
              f"top8 {rep['top_k_share']:.3f}  |  "
              f"lead_gap {lg.get('point'):+.4f} {lg.get('ci95')}  "
              f"left_occ {lo.get('point'):.4f}  right_occ {ro.get('point'):.4f}",
              flush=True)
        del G, X, Z

    dest = SP / "e_trunk3_ladder.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
