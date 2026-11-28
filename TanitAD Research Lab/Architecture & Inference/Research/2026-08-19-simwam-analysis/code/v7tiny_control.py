"""⭐ POSITIVE CONTROL for the "the latent is 98 % noise" finding.

⛔ WHY THIS RUNS BEFORE ANY MORE TRAINING. E-DENSE-1 was a tiny rig whose
POSITIVE CONTROL FAILED, and every other arm in it became uninterpretable. The
oracle probe now claims that the per-tick movement of v6F's / v7-tiny's latent
is ~98 % unpredictable. That claim is only admissible if the SAME probe, on the
SAME held-out clips, with the SAME ridge / bootstrap / episode-disjoint
protocol, CAN detect predictable movement where predictable movement certainly
exists.

Three representations of the identical 24 held-out clips:

  ego     [x, y, yaw, speed] from `poses`. Predicting d(ego) from (ego, action)
          is KINEMATICS. ⭐ If this does not score high, THE PROBE IS BROKEN and
          every latent number from it must be withdrawn.

  pixel   the raw frame, greyscaled and area-pooled to a small grid. Camera
          motion is smooth, so consecutive frames are strongly related. This is
          the "raw perception" reference: whatever structure a trained encoder
          adds, it should not have LESS temporal predictability than pixels.

  ego_hi  ego lifted with the products a learned model could trivially form
          (v*sin(yaw), v*cos(yaw), v*steer). A linear map cannot express the
          rotation that turns body-frame velocity into world-frame displacement,
          so plain `ego` UNDERSTATES what is learnable; this bounds that gap and
          keeps the control from being unfairly weak.

⚠️ NO GPU. Pure numpy on already-banked bytes, so it does not contend with the
o-term ladder training on the same box.

⚠️ Same estimator as the latent measurement: ridge fit on one half of the clips,
scored on the DISJOINT other half, episode-cluster bootstrap of the POOLED
statistic, selection on the CI LOWER BOUND (never the point estimate).

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
HELD = SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"
MATERIAL = 0.01
LAMBDAS = (1e-4, 1e-2, 1.0, 1e2, 1e4)


def load_clip(path: Path, max_frames: int, want_pixels: bool):
    d = torch.load(path, map_location="cpu", weights_only=False)
    poses = d["poses"].numpy().astype(np.float64)
    acts = d["actions"].numpy().astype(np.float64)
    n = min(len(poses), len(acts), max_frames)
    px = None
    if want_pixels:
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
            np.int64)
        rows = []
        for i in range(n):
            im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("L")
            rows.append(np.asarray(im.resize((40, 16), Image.BOX),
                                   dtype=np.float64).ravel() / 255.0)
        px = np.stack(rows)
        if float(np.abs(px).mean()) == 0.0:
            raise SystemExit(f"[FATAL] {path.name} decoded to all-zero frames")
    return poses[:n], acts[:n], px


def representation(kind, poses, acts, px):
    """-> (state, action) arrays for this representation."""
    if kind == "ego":
        return poses, acts
    if kind == "ego_hi":
        x, y, yaw, v = poses.T
        extra = np.stack([v * np.sin(yaw), v * np.cos(yaw), v * acts[:, 0],
                          np.sin(yaw), np.cos(yaw), v ** 2], 1)
        return np.concatenate([poses, extra], 1), acts
    if kind == "pixel":
        return px, acts
    raise ValueError(kind)


def oracle(states, actions, cids, seed=0):
    """Identical protocol to v7tiny_oracle.py: episode-disjoint ridge + CI."""
    k = len(states)
    half = k // 2
    def build(ix):
        X, Y = [], []
        for i in ix:
            s, a = states[i], actions[i]
            n = min(len(s), len(a)) - 1
            X.append(np.concatenate([s[:n], a[:n]], 1))
            Y.append(s[1:n + 1] - s[:n])
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = build(range(half))
    mu = Ytr.mean(0, keepdims=True)
    xm = Xtr.mean(0, keepdims=True)
    Xc, Yc = Xtr - xm, Ytr - mu
    G = Xc.T @ Xc
    C = Xc.T @ Yc
    te = list(range(half, k))
    rng = np.random.default_rng(seed)
    best, best_lam, best_ci = None, None, (-9e9, -9e9)
    rows = {}
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), C)
        errs, movs = [], []
        for i in te:
            s, a = states[i], actions[i]
            n = min(len(s), len(a)) - 1
            Xi = np.concatenate([s[:n], a[:n]], 1)
            Yi = s[1:n + 1] - s[:n]
            pi = (Xi - xm) @ W + mu
            errs.append(float(((Yi - pi) ** 2).sum()))
            movs.append(float((Yi ** 2).sum()))
        errs, movs = np.array(errs), np.array(movs)
        e = 1.0 - errs.sum() / movs.sum()
        bs = np.empty(4000)
        for b in range(4000):
            j = rng.integers(0, len(errs), len(errs))
            bs[b] = 1.0 - errs[j].sum() / movs[j].sum()
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        rows[f"ridge_{lam:g}"] = {"em": round(e, 6),
                                  "ci95": [round(lo, 6), round(hi, 6)]}
        if lo > best_ci[0]:
            best, best_lam, best_ci = e, lam, (lo, hi)
    return best, best_lam, best_ci, rows, len(te)


def autocorr(states):
    c1, c5 = [], []
    for s in states:
        dz = s[1:] - s[:-1]
        nz = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
        c1.append(float((nz[:-1] * nz[1:]).sum(1).mean()))
        if len(nz) > 5:
            c5.append(float((nz[:-5] * nz[5:]).sum(1).mean()))
    return float(np.mean(c1)), float(np.mean(c5))


def main() -> int:
    ap = argparse.ArgumentParser(description="positive control for the oracle")
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames-per-clip", type=int, default=120)
    ap.add_argument("--out", default=str(SP / "v7tiny_control.json"))
    a = ap.parse_args()

    paths = sorted(HELD.glob("*.v2ep.pt"))[:a.clips]
    print(f"  {len(paths)} held-out clips · NO GPU (numpy only)\n", flush=True)
    P, A, PX, cids = [], [], [], []
    for n, p in enumerate(paths, 1):
        poses, acts, px = load_clip(p, a.frames_per_clip, want_pixels=True)
        P.append(poses); A.append(acts); PX.append(px); cids.append(p.name[:8])
        print(f"    [{n}/{len(paths)}] {p.name[:10]} {len(poses)} frames",
              flush=True)

    res = {"_evidence_class": "MEASURED (ours; dev-box, numpy)",
           "eval_tier": "T0-DIAGNOSTIC", "parity": False,
           "purpose": "does the oracle probe detect predictable movement where "
                      "it certainly exists? if not, the latent numbers are void",
           "n_clips": len(paths), "materiality_floor": MATERIAL,
           "representations": {}}
    print(f"\n  {'representation':<12}{'dim':>6}{'lag1 cos':>11}"
          f"{'best EM':>11}  {'CI95':<24}{'verdict'}")
    print("  " + "-" * 84)
    for kind in ("ego", "ego_hi", "pixel"):
        S, Ac = [], []
        for i in range(len(P)):
            s, ac = representation(kind, P[i], A[i], PX[i])
            S.append(s); Ac.append(ac)
        c1, c5 = autocorr(S)
        best, lam, ci, rows, nte = oracle(S, Ac, cids)
        verdict = ("PREDICTABLE" if ci[0] > 0.10 else
                   "weakly predictable" if ci[0] > MATERIAL else
                   "NOT predictable (== hold)")
        res["representations"][kind] = {
            "dim": int(S[0].shape[1]), "lag1_cos": round(c1, 4),
            "lag5_cos": round(c5, 4), "best_em": round(float(best), 6),
            "best_lambda": lam, "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "n_test_clips": nte, "ridge": rows, "verdict": verdict}
        print(f"  {kind:<12}{S[0].shape[1]:>6}{c1:>+11.4f}{best:>+11.4f}  "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]   {verdict}")

    ego_ok = res["representations"]["ego_hi"]["ci95"][0] > 0.10
    res["probe_valid"] = bool(ego_ok)
    res["verdict"] = (
        "PROBE VALID: it detects predictable movement on ego kinematics, so its "
        "reading that the LEARNED LATENT's per-tick movement is ~98 % "
        "unpredictable is a statement about the representation, not an artifact "
        "of the estimator."
        if ego_ok else
        "⛔ PROBE INVALID: it cannot detect predictability even on ego "
        "kinematics, where the relation is physics. EVERY latent number from "
        "this probe must be WITHDRAWN until the estimator is fixed.")
    print(f"\n  {res['verdict']}")
    print(f"\n  for comparison, the LEARNED latents on these same clips:")
    print(f"    v6F @20k      best EM +0.0203  [+0.0056, +0.0317]")
    print(f"    v7-tiny fixed best EM +0.0018  [-0.0015, +0.0030]")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
