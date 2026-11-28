"""Beside PREREG_V8F_PLANTEST3, deciding nothing (Master Mind, 2026-11-09): after the card's DECODER-WEAK flag, how much of speed and yaw rate is linearly readable from
ONE level-1 latent versus a short stack of past latents? TUNE inner split only (the card's own split); no SCORE window is read. CPU, no model load, closed-form ridge.
Targets: speed at t; yaw rate 'ahead' = A[t] (the card's gate target, the action taken at t, which the latent at t has not yet seen); yaw rate 'behind' = A[t-1] (the rate over the last interval).
Tier T0, a readout diagnostic on encoded latents, never a driving result. Pooled held-out R2; shuffled-target control per row."""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "plantest2"))
sys.path.insert(0, os.path.join(HERE, "..", "plantest"))
import numpy as np
import torch

torch.set_num_threads(2)
import plantest_lib as P
import plantest2_lib as Q
import plantest3_lib as L3

ROOT = os.path.join(HERE, "..")
EVAL = os.path.join(ROOT, "wmact", "latents.pt")
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
MS = (1, 2, 3, 4, 6, 8)
RIDGES = (1e-3, 1e-2, 1e-1)


def fit_score(Xf, yf, Xh, yh, ridge):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    A = (Xf - mu) / sd
    B = (Xh - mu) / sd
    ym = yf.mean()
    G = A.T @ A + ridge * len(A) * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ (yf - ym))
    p = B @ w + ym
    return 1.0 - float(((yh - p) ** 2).sum() / ((yh - yh.mean()) ** 2).sum())


def stack(Z, m):
    """Z (n, 16, D) -> features (n, 16-m+1, m*D) for frames t = m-1 .. 15."""
    return np.concatenate([Z[:, m - 1 - j: Z.shape[1] - j] for j in range(m)], axis=2)


def read(tag, ev, TRK):
    eid = [str(e) for e in ev["eid"]]
    tune, _ = Q.split_indices(np.array(eid))
    fit_ids, held_ids = L3.inner_split([eid[i] for i in tune])
    fit = np.array([i for i in tune if eid[i] in fit_ids])
    held = np.array([i for i in tune if eid[i] in held_ids])
    Z = ev["Z1"].double().numpy()
    A = np.asarray(ev["A"], np.float64)
    n = len(eid)
    vk = np.full((n, 16), np.nan)
    for i, (e, t) in enumerate(zip(eid, ev["t"])):
        g = int(t) - 15
        r = TRK.get(e)
        if r is None or g < 0:
            continue
        r = np.asarray(r)
        for k in range(16):
            if 2 * (g + k) < len(r):
                vk[i, k] = float(r[2 * (g + k), 3])
    out = []
    rng = np.random.default_rng(0)
    for m in MS:
        X = stack(Z, m)
        tgt = {"speed": vk[:, m - 1:], "yaw_ahead": A[:, m - 1:, 1], "yaw_behind": A[:, m - 2:-1, 1] if m >= 2 else None}
        if m == 1:
            tgt["yaw_behind"] = np.concatenate([np.full((n, 1), np.nan), A[:, :-1, 1]], axis=1)
        row = dict(m=m, n_fit_windows=len(fit), n_held_windows=len(held))
        for name, y in tgt.items():
            xf, yf = X[fit].reshape(-1, X.shape[2]), y[fit].reshape(-1)
            xh, yh = X[held].reshape(-1, X.shape[2]), y[held].reshape(-1)
            okf, okh = np.isfinite(yf), np.isfinite(yh)
            best = {str(r): fit_score(xf[okf], yf[okf], xh[okh], yh[okh], r) for r in RIDGES}
            row[name] = dict(r2_by_ridge=best, n_held=int(okh.sum()))
            yp = yf[okf].copy()
            rng.shuffle(yp)
            row[name]["shuffled_r2_at_1e-3"] = fit_score(xf[okf], yp, xh[okh], yh[okh], 1e-3)
        out.append(row)
    return out


def main():
    TRK = torch.load(TRACKS, weights_only=False)["tracks"]
    res = {}
    for tag in ("M2b_s0", "M2b_s1"):
        ev = P.load_latents(EVAL, tag)
        assert "A" in ev, list(ev)
        res[tag] = read(tag, ev, TRK)
    json.dump(res, open(os.path.join(HERE, "DIAG_MULTIFRAME.json"), "w", encoding="utf-8"), indent=1)
    for tag, rows in res.items():
        print(tag)
        for r in rows:
            print(" m", r["m"], *[f"{k} " + "/".join(f"{v:+.3f}" for v in r[k]["r2_by_ridge"].values()) + f" (shuf {r[k]['shuffled_r2_at_1e-3']:+.3f})" for k in ("speed", "yaw_ahead", "yaw_behind")])


if __name__ == "__main__":
    main()
