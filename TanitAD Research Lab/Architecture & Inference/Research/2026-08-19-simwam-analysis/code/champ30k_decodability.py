"""E-TRUNK-3 ladder, CHAMP30K RUNG — does the 30k champion SEE anything?

⛔ THE QUESTION. champ30k (two-term + k1 + subspace32, 30,000 steps) raised VAL
participation 4.052 (k1@6k) -> 6.499. E-TRUNK-3 already established that this
statistic RISES while decodability stays pinned at zero:

    "An anti-collapse regulariser guarantees the dimensions are USED, never that
     they are INFORMATIVE. A variance/isotropy constraint can be satisfied by
     noise."  (E_TRUNK_3_LADDER.md 5b)

So the participation gain is NOT evidence of a better representation. The
decisive measurement is decodability, and this script runs exactly the banked
E-TRUNK-2 battery on the champ30k trunk so the number is comparable rung-for-rung
with steps 2000 / 16000 / 18000 / 20000 of v6F-SW-30k.

BOTH OUTCOMES, committed here before the number exists (the pre-registration's
own decision rule, E_TRUNK_3_LADDER.md §2/§3):
  * `lead_gap_m` R² remains at/below zero with a CI straddling or below 0
      -> WORLD B holds for the NEW objective too. Two-term+k1 buys rank and NOT
         perception; the participation lever is exhausted as a route to a useful
         trunk, and the v7 encoder question (E-ENC-3WAY) becomes the only lever
         left. champ30k is a rank result and must never be quoted as progress.
  * `lead_gap_m` R² separates ABOVE zero and above the C-PIXEL floor
      -> the FIRST v6/v7 checkpoint to acquire environment information. That
         would make two-term+k1 a genuine representational advance, would
         partially rehabilitate the participation statistic as a progress
         signal, and would make a scaled champ30k the v7 candidate.

FALSIFIERS (inherited, enforced):
  1. the champ30k cache must carry the SAME (clip_id, frame_idx) keys IN THE
     SAME ORDER as the reference, else folds differ and rungs are incomparable.
  2. C-EGO / C-PIXEL never see the checkpoint -> computed once, constant lines.
  3. C-EGO -> ego_speed is an identity map and MUST read ~1.0.

⚠️ ONE STATED DIFFERENCE FROM THE OTHER RUNGS: the banked rungs were built from
fp16 weight snapshots ("fp16->fp32 (lossy)"); champ30k is loaded from its full
fp32 `ckpt.pt`. That is a quality improvement, not a matched condition, and it is
stamped in the output.

TIER: T0-DIAGNOSTIC. Decodability is a representation property, NEVER driving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# the banked E-TRUNK-2/3 instrument (⚠️ lives in a TEMP scratchpad — see the
# escalation in the TrainingFlyWheel report; this must be banked into the repo)
SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

import e_trunk2_probe as P  # noqa: E402
from tanitad.models.v6 import spectrum_report, o6_rank_verdict  # noqa: E402

CACHE = SP / "sp2" / "cache_champ30k"
OUT = Path(__file__).with_name("champ30k_decodability.json")
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
           "_arm": "champ30k = two-term(o5+o6) + k1 + sigreg-subspaces 32, "
                   "S-W, 30,000 steps, parity cache e438721ae894",
           "_precision_note": "champ30k loaded from FULL fp32 ckpt.pt; the "
                              "banked v6F rungs were fp16->fp32 (lossy). Stated, "
                              "not matched.",
           "_prereg": "both outcomes committed in this file's docstring before "
                      "the number existed",
           "n_frames": len(ref_keys), "n_episodes": len(set(ep)),
           "controls": {}, "reference": {}, "arm": {}}

    for name, fn in (("C-EGO", "c_ego.npy"), ("C-PIXEL", "c_pixel.npy")):
        G = P.gram_memmap(P.FEAT / fn)
        G = G / float(np.mean(np.diag(G)))
        out["controls"][name] = score(G, ref_keys, tgt, ep, folds)
        lg = out["controls"][name].get("lead_gap_m", {}).get("point")
        print(f"[control] {name}: lead_gap {lg}", flush=True)

    idm = out["controls"]["C-EGO"].get("ego_speed", {}).get("point")
    if idm is None or abs(idm - 1.0) > 0.05:
        raise RuntimeError(f"FALSIFIER 3 TRIPPED: C-EGO->ego_speed must read "
                           f"~1.0, got {idm}")
    print(f"[falsifier 3] identity map reads {idm:.4f} — OK", flush=True)

    # the DINOv3 reference rung, same folds (this is the 40.77 / +0.3792 line)
    Gd = P.gram_memmap(P.FEAT / "gram_dino_pooled.npy")
    Gd = Gd / float(np.mean(np.diag(Gd)))
    out["reference"]["dino_pooled"] = score(Gd, ref_keys, tgt, ep, folds)
    print(f"[reference] dino_pooled lead_gap "
          f"{out['reference']['dino_pooled'].get('lead_gap_m', {}).get('point')}",
          flush=True)
    del Gd

    keys, X = load_cells(CACHE)
    if keys != ref_keys:                                    # falsifier 1
        raise RuntimeError(
            f"FALSIFIER 1 TRIPPED: champ30k cache keys differ from the reference "
            f"({len(keys)} vs {len(ref_keys)}); folds would not be comparable")
    print(f"[falsifier 1] {len(keys)} keys match the reference exactly — OK",
          flush=True)

    Z = torch.from_numpy(X)
    rep = spectrum_report(Z, top_k=8)
    ver = o6_rank_verdict(rep)
    G = X.astype(np.float32) @ X.astype(np.float32).T
    G = G / float(np.mean(np.diag(G)))
    res = score(G, ref_keys, tgt, ep, folds)
    out["arm"]["champ30k"] = {
        "spectrum": {"participation_ratio": rep["participation_ratio"],
                     "effective_rank": rep["effective_rank"],
                     "top_k_share": rep["top_k_share"],
                     "rank_ceiling": ver["rank_ceiling"],
                     "verdict": ver["status"]},
        "targets": res}

    lg = res.get("lead_gap_m", {})
    lo = res.get("left_occupied", {})
    ro = res.get("right_occupied", {})
    print(f"\nchamp30k  PR {rep['participation_ratio']:6.2f}  "
          f"top8 {rep['top_k_share']:.3f}  |  "
          f"lead_gap {lg.get('point'):+.4f} {lg.get('ci95')}  "
          f"left_occ {lo.get('point'):.4f}  right_occ {ro.get('point'):.4f}",
          flush=True)

    # the committed decision rule, evaluated mechanically
    ci = lg.get("ci95") or [None, None]
    pix = out["controls"]["C-PIXEL"].get("lead_gap_m", {}).get("point")
    if ci[0] is not None and ci[0] > 0 and pix is not None and lg["point"] > pix:
        verdict = ("SEPARATED ABOVE ZERO AND ABOVE THE PIXEL FLOOR — the first "
                   "v6/v7 checkpoint to acquire environment information")
    else:
        verdict = ("NOT SEPARATED — WORLD B holds for two-term+k1 as well; the "
                   "participation gain did NOT buy perception")
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
