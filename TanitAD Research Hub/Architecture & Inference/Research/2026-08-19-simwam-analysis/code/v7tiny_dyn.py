"""⭐⭐ Is the DYNAMICS in the latent, and is the G2 target the wrong one?

⛔ WHAT THE CONTROL JUST ESTABLISHED (v7tiny_control.json, 24 held-out clips):

    ego   [x,y,yaw,v]      best EM +0.9871   the probe DETECTS physics
    pixel raw 640-dim      best EM +0.0010   raw pixels are NOT predictable
    v6F   latent 2048-dim  best EM +0.0203   our latent is 20x BETTER than pixels

So "per-tick change is ~98 % unpredictable" is NOT a defect of our encoder -- it
is a property of ANY high-dimensional visual representation at 10 Hz, and raw
pixels are worse. Meanwhile the underlying vehicle dynamics are 98.7 %
predictable. The predictable part exists; it is simply a tiny fraction of the
VARIANCE of the per-tick change.

⇒ That makes "beat HOLD on the full latent" (G2) a badly posed gate: it scores a
predictor mostly on how well it reproduces unpredictable local texture. This
file measures the two directions that decide what the RIGHT target is.

  A. z -> d(ego)   DOES THE LATENT CARRY THE DYNAMICS?
     Ridge from the latent (and its one-tick change) to the ego displacement.
     ⭐ If this is high, the information IS in the latent and only the OBJECTIVE
     is wrong -- the fix is to predict in a dynamics-bearing subspace, not to
     retrain the encoder. If it is low, the encoder genuinely is not seeing
     motion and that is a much deeper problem.

  B. d(ego) -> dz  HOW MUCH OF THE LATENT'S MOVEMENT IS DYNAMICS AT ALL?
     The share of dz's variance that ego motion can explain. This is the
     CEILING that any action-conditioned predictor could reach on the full
     latent, and it is what makes G2's ~2 % interpretable.

  C. SUBSPACE G2. Project dz onto the top-k principal components of the
     ego-explainable part and re-score. ⭐ If EM is high in that subspace while
     ~0 on the full latent, the finding is precise and actionable: the target
     should be the projection, not the raw latent.

⚠️ Every fit is EPISODE-DISJOINT (fit on half the clips, scored on the other
half), episode-cluster bootstrap of the POOLED statistic, selection on the CI
LOWER BOUND. Same protocol as the numbers it is compared against.

⚠️ `pixel` is carried through every panel as the floor, because "our latent
beats hold by 2 %" only means something next to what raw perception scores.

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
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects"
                            r"\TanitAD\stack")))
HELD = SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"
LAMBDAS = (1e-2, 1.0, 1e2, 1e4, 1e6)


def ridge_em(Xs, Ys, seed=0, baseline="zero"):
    """Episode-disjoint ridge; -> (best_em, lam, ci, rows).

    ⛔ `baseline` DECIDES WHAT THE SCORE MEANS, and getting it wrong manufactures
    a result. MEASURED 2026-08-22: with `baseline="zero"` the panel-A targets
    (d(ego), whose mean is ~0.47 m/tick of forward motion) scored +0.5420 from
    the LATENT, +0.5412 from [z, dz] and +0.5421 from RAW PIXELS -- identical to
    three decimals, because all three merely reproduced the CONSTANT mean
    displacement and none added anything. The pixel floor matching exactly is
    what exposed it.

      "zero"  the target is a DELTA and the baseline is HOLD (predict no
              change), which is genuinely the zero vector. Correct for dz.
      "mean"  the target has a large constant component, so the baseline is
              the TRAINING mean applied to test. The score is then a skill
              score OVER the constant predictor -- the only thing that can
              show a representation contributing information.

    ⛔ LAMBDA IS CHOSEN ON A VALIDATION SPLIT OF THE *FIT* CLIPS, NEVER ON THE
    SCORED CLIPS. Selecting lambda on the test set is selection bias, and it
    DEGENERATES: MEASURED 2026-08-22, picking the best CI lower bound on test
    chose lambda=1e6 -- which shrinks the ridge to the constant mean predictor,
    scoring EXACTLY +0.0000 with a zero-width CI, and therefore beating every
    noisy positive estimate. The latent, [z, dz] and the constant CONTROL all
    then read +0.0000 and the panel reported "the latent does not carry the
    dynamics", which was an artifact of the SELECTOR, not a measurement.
    """
    k = len(Xs)
    half = k // 2
    fit = list(range(half))
    te = list(range(half, k))
    nv = max(1, len(fit) // 4)
    inner, val = fit[:-nv], fit[-nv:]
    Xtr = np.concatenate([Xs[i] for i in inner])
    Ytr = np.concatenate([Ys[i] for i in inner])
    xm, mu = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - xm, Ytr - mu
    G = Xc.T @ Xc
    C = Xc.T @ Yc
    rng = np.random.default_rng(seed)
    rows, Ws, val_sse = {}, {}, {}
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), C)
        Ws[lam] = W
        val_sse[lam] = sum(
            float((((Ys[i]) - ((Xs[i] - xm) @ W + mu)) ** 2).sum())
            for i in val)
    best_lam = min(val_sse, key=val_sse.get)          # <- chosen on VAL only
    for lam in LAMBDAS:
        W = Ws[lam]
        errs, tots = [], []
        for i in te:
            p = (Xs[i] - xm) @ W + mu
            errs.append(float(((Ys[i] - p) ** 2).sum()))
            ref = np.zeros_like(Ys[i]) if baseline == "zero" else mu
            tots.append(float(((Ys[i] - ref) ** 2).sum()))
        errs, tots = np.array(errs), np.array(tots)
        e = 1.0 - errs.sum() / tots.sum()
        bs = np.empty(2000)
        for b in range(2000):
            j = rng.integers(0, len(errs), len(errs))
            bs[b] = 1.0 - errs[j].sum() / tots[j].sum()
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        rows[f"ridge_{lam:g}"] = {"em": round(e, 6),
                                  "ci95": [round(lo, 6), round(hi, 6)],
                                  "selected": bool(lam == best_lam)}
        if lam == best_lam:
            best, best_ci = e, (lo, hi)
    return best, best_lam, best_ci, rows


def load_pixels(path, n):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
        np.int64)
    rows = []
    for i in range(min(n, len(off) - 1)):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("L")
        rows.append(np.asarray(im.resize((40, 16), Image.BOX),
                               dtype=np.float64).ravel() / 255.0)
    return np.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="is the dynamics in the latent?")
    ap.add_argument("--arm", default="fixed")
    ap.add_argument("--v6f", action="store_true")
    ap.add_argument("--v6f-ckpt", default=str(SP / "ckpt/v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--v6f-config", default=str(SP / "sp2/v6F_config.json"))
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames-per-clip", type=int, default=120)
    ap.add_argument("--out", default=str(SP / "v7tiny_dyn.json"))
    a = ap.parse_args()

    import v7tiny_g2 as G
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.v6f:
        import e_pred_probe as E
        world, step = E.load_world(Path(a.v6f_ckpt), Path(a.v6f_config), dev)
        name = f"v6F@{step}"
    else:
        world, step = G.load_arm(a.arm, dev)
        name = f"v7tiny-{a.arm}@{step}"

    paths = sorted(HELD.glob("*.v2ep.pt"))[:a.clips]
    Z, PO, PX = [], [], []
    for n, p in enumerate(paths, 1):
        z, _act, _spd = G.encode_clip(world, p, dev, a.frames_per_clip)
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = min(len(z), len(d["poses"]))
        Z.append(z.numpy()[:m].astype(np.float64))
        PO.append(d["poses"].numpy()[:m].astype(np.float64))
        PX.append(load_pixels(p, m)[:m])
        print(f"    [{n}/{len(paths)}] {p.name[:10]} {m} frames", flush=True)

    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "model": name, "parity": False,
           "n_clips": len(Z), "panels": {}}

    # ---- A. does the latent carry the dynamics? -----------------------------
    print(f"\n  A. z -> d(ego):  DOES THE LATENT CARRY THE DYNAMICS?")
    print(f"     {'source':<26}{'EM':>10}  {'CI95':<24}{'verdict'}")
    print("     " + "-" * 74)
    print("     (skill score OVER the constant mean-displacement predictor; "
          "a feature set adding nothing scores 0.0000 by construction)")
    for label, feats in (("latent z", Z), ("latent [z, dz]", None),
                         ("pixel (floor)", PX), ("constant (control)", "CONST")):
        if isinstance(feats, str) and feats == "CONST":
            # ⭐ the control that would have caught the panel-A bug on sight:
            # a single constant column carries NO information, so any correct
            # skill score must read ~0.0000 here.
            X = [np.ones((len(z) - 1, 1)) for z in Z]
            Y = [p[1:] - p[:-1] for p in PO]
            X = [x[:len(y)] for x, y in zip(X, Y)]
            Y = [y[:len(x)] for x, y in zip(X, Y)]
        elif feats is None:
            X = [np.concatenate([z[1:], z[1:] - z[:-1]], 1) for z in Z]
            Y = [p[2:] - p[1:-1] for p in PO]
            X = [x[:len(y)] for x, y in zip(X, Y)]
        else:
            X = [f[:-1] for f in feats]
            Y = [p[1:] - p[:-1] for p in PO]
            X = [x[:len(y)] for x, y in zip(X, Y)]
            Y = [y[:len(x)] for x, y in zip(X, Y)]
        e, lam, ci, rows = ridge_em(X, Y, baseline="mean")
        v = ("CARRIES dynamics" if ci[0] > 0.30 else
             "partial" if ci[0] > 0.05 else "does NOT carry it")
        res["panels"].setdefault("A_z_to_dego", {})[label] = {
            "em": round(float(e), 6), "lambda": lam,
            "ci95": [round(ci[0], 6), round(ci[1], 6)], "verdict": v,
            "ridge": rows}
        print(f"     {label:<26}{e:>+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]   {v}")

    # ---- B. how much of dz is ego-driven at all? ----------------------------
    print(f"\n  B. d(ego) -> dz:  WHAT SHARE OF THE LATENT'S MOVEMENT IS DYNAMICS?")
    dego = [np.concatenate([p[1:] - p[:-1], p[:-1]], 1) for p in PO]
    dz = [z[1:] - z[:-1] for z in Z]
    m = [min(len(x), len(y)) for x, y in zip(dego, dz)]
    Xb = [x[:k] for x, k in zip(dego, m)]
    Yb = [y[:k] for y, k in zip(dz, m)]
    eb, lamb, cib, _ = ridge_em(Xb, Yb)
    res["panels"]["B_dego_to_dz"] = {
        "em": round(float(eb), 6), "lambda": lamb,
        "ci95": [round(cib[0], 6), round(cib[1], 6)],
        "meaning": "the share of the latent's per-tick movement that ego motion "
                   "can explain -- the CEILING for any action-conditioned "
                   "predictor scored on the FULL latent"}
    print(f"     ego motion explains {eb:+.4f} of dz  "
          f"[{cib[0]:+.4f}, {cib[1]:+.4f}]")

    # ---- C. subspace G2 -----------------------------------------------------
    print(f"\n  C. SUBSPACE G2: is the dynamics CONCENTRATED in a few directions?")
    half = len(Z) // 2
    Xtr = np.concatenate(Xb[:half]); Ytr = np.concatenate(Yb[:half])
    xm, mu = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    W = np.linalg.solve((Xtr - xm).T @ (Xtr - xm)
                        + 1.0 * np.eye(Xtr.shape[1]), (Xtr - xm).T @ (Ytr - mu))
    fit = (Xtr - xm) @ W                       # the ego-explainable part of dz
    U, S, Vt = np.linalg.svd(fit - fit.mean(0, keepdims=True),
                             full_matrices=False)
    res["panels"]["C_subspace"] = {}
    print(f"     {'k dims':<10}{'EM in subspace':>16}  {'CI95':<24}"
          f"{'share of dz energy':>20}")
    print("     " + "-" * 74)
    for k in (2, 4, 8, 16, 32):
        B = Vt[:k].T                                        # [d, k]
        Xs = [x for x in Xb]
        Ys = [y @ B for y in Yb]
        e, lam, ci, _ = ridge_em(Xs, Ys)
        tot = sum(float((y ** 2).sum()) for y in Yb)
        sh = sum(float(((y @ B) ** 2).sum()) for y in Yb) / tot
        res["panels"]["C_subspace"][f"k{k}"] = {
            "em": round(float(e), 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "share_of_dz_energy": round(sh, 6)}
        print(f"     k={k:<8}{e:>+16.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
              f"{sh:>20.4%}")

    A = res["panels"]["A_z_to_dego"]
    best_a = max(v["em"] for v in A.values() if isinstance(v, dict))
    res["verdict"] = (
        f"THE LATENT CARRIES THE DYNAMICS (z -> d(ego) EM {best_a:+.4f}) while "
        f"ego motion explains only {eb:+.4f} of the latent's per-tick movement. "
        f"⇒ G2 on the FULL latent scores a predictor mostly on unpredictable "
        f"local variation. The target is wrong, not the encoder."
        if best_a > 0.30 else
        f"THE LATENT DOES NOT CARRY THE DYNAMICS (best z -> d(ego) EM "
        f"{best_a:+.4f}). This is an ENCODER problem and no change of "
        f"prediction target will fix it.")
    print(f"\n  VERDICT: {res['verdict']}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
