"""E-PROBE-POWER — is the DECODER the problem, or the representation?

⛔ THE PI'S QUESTION (2026-08-21): "Could our decoder be the problem? or how we
decode?" Every null so far — v6_cells, cls192, v6shape, lewm, wsig, aux — came
from ONE probe: a LINEAR (dual/Gram ridge) read of a SINGLE frame's latent. That
is a real limitation and it has never been tested.

⭐ IT IS ALSO LeWM'S OWN CAVEAT. Their §5.1 reports BOTH linear and MLP probes,
and the gap is large where it matters: Block Angle linear r 0.902 -> MLP r 0.990,
MSE 0.187 -> 0.021. If our content is non-linearly encoded, a linear probe reads
it as ABSENT.

THREE SUSPECTS, each with its own arm:

  linear   the incumbent — ridge on z_t                    (what every null used)
  mlp      non-linear head on z_t                          (LeWM's second column)
  window   linear on [z_{t-2}, z_{t-1}, z_t] concatenated  (the model is a
           SEQUENCE model; a single frame may not be where the state lives)

⛔⛔ AND THE CONTROL THAT DECIDES IT. Every probe runs on the RANDOM UNTRAINED
encoder too. If the MLP lifts `random` as much as it lifts a trained arm, the
MLP is doing the work and the lift is NOT a property of the representation —
that is the same "winner's curse" shape as SEL-1, in a probe costume.

PROTOCOL, unchanged from E-TRUNK-2 so the numbers stay comparable: the same
5,617 keys, the same EPISODE-DISJOINT folds, the same episode-cluster bootstrap
of the POOLED statistic. Only the decoder changes.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path.cwd() / "stack"))

import e_trunk2_probe as P  # noqa: E402
import e_lewm_ablate as L   # noqa: E402

TARGETS = ("lead_gap_m", "left_occupied", "right_occupied", "ego_speed")
#: arms that live in the lewm frame-cache (26,108 rows -> index to probe keys)
LEWM_ARMS = ("cls192", "v6shape", "random-CONTROL", "supervised-CONTROL")
#: arms already stored at the 5,617 probe rows
FEAT_ARMS = {"v6_cells": "v6_cells.npy", "dino_pooled": "dino_pooled.npy"}


def load_arm(name: str, rows: np.ndarray) -> np.ndarray | None:
    if name in FEAT_ARMS:
        p = P.FEAT / FEAT_ARMS[name]
        return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32) if p.exists() else None
    # ⚠️ the capacity control was saved as `z_supervised_s0.npy`, not
    # `z_supervised-CONTROL_s0.npy` — a naming mismatch that SILENTLY skipped
    # the one arm that calibrates the whole comparison. Try both.
    for stem in (name, name.replace("-CONTROL", "")):
        p = L.CACHE / f"z_{stem}_s0.npy"
        if p.exists():
            return np.load(p)[rows].astype(np.float32)
    return None


def window_stack(Z: np.ndarray, keys, k: int = 3) -> np.ndarray:
    """[z_{t-k+1} … z_t] concatenated, clamped at clip starts.

    ⚠️ Clamping (not dropping) keeps the row set IDENTICAL to every other probe,
    so the folds and the bootstrap stay comparable. A different row set would
    make the comparison to the linear numbers invalid."""
    by = {}
    for i, (cid, f) in enumerate(keys):
        by.setdefault(cid, {})[f] = i
    out = np.zeros((len(keys), Z.shape[1] * k), dtype=np.float32)
    for i, (cid, f) in enumerate(keys):
        m = by[cid]
        idx = [m.get(f - j * 4, m.get(f, i)) for j in range(k - 1, -1, -1)]
        out[i] = np.concatenate([Z[j] for j in idx])
    return out


def mlp_oof(X, y, ep, folds, *, binary, hidden=256, epochs=120, seed=0):
    """Out-of-fold MLP predictions, EPISODE-DISJOINT, standardised per fold."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = np.full(len(y), np.nan, dtype=np.float64)
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.from_numpy((X[tr] - mu) / sd).to(dev)
        Xte = torch.from_numpy((X[te] - mu) / sd).to(dev)
        ym, ys = (y[tr].mean(), y[tr].std() + 1e-6) if not binary else (0.0, 1.0)
        ytr = torch.from_numpy(((y[tr] - ym) / ys).astype(np.float32)).to(dev)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, hidden), nn.GELU(),
                            nn.Linear(hidden, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), 256):
                j = perm[i:i + 256]
                loss = lossf(net(Xtr[j]).squeeze(-1), ytr[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            o = net(Xte).squeeze(-1).cpu().numpy().astype(np.float64)
        pred[te] = o if binary else o * ys + ym
    return pred


def main() -> None:
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    folds = P.episode_folds(ep)
    fmap = {(c["clip_id"], f): c["start"] + f
            for c in json.loads((L.CACHE / "clips.json").read_text(encoding="utf-8"))
            for f in range(c["n"])}
    rows = np.array([fmap[k] for k in keys])

    tgt = {}
    for line in P.TARGETS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tgt[(r["clip_id"], int(r["frame_idx"]))] = r
    eg = P.ego_features(keys)
    for i, k in enumerate(keys):
        tgt.setdefault(k, {})["ego_speed"] = float(eg[i, 0])

    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "question": "is the DECODER the limit, or the representation?",
           "protocol": "same 5,617 keys / episode-disjoint folds / "
                       "episode-cluster bootstrap as E-TRUNK-2; only the "
                       "decoder changes",
           "arms": {}}

    for arm in list(LEWM_ARMS) + list(FEAT_ARMS):
        Z = load_arm(arm, rows)
        if Z is None:
            print(f"  [skip] {arm}: latents absent", flush=True)
            continue
        Zw = window_stack(Z, keys)
        out["arms"][arm] = {"d": int(Z.shape[1]), "d_window": int(Zw.shape[1]),
                            "targets": {}}
        for t in TARGETS:
            y = np.array([tgt.get(k, {}).get(t, np.nan) for k in keys],
                         dtype=np.float64)
            ok = np.isfinite(y)
            if ok.sum() < 500:
                continue
            idx = np.nonzero(ok)[0]
            remap = {v: i for i, v in enumerate(idx)}
            sub = [np.array([remap[i] for i in f if i in remap]) for f in folds]
            eps = [ep[i] for i in idx]
            binary = t in P.BINARY
            res = {}
            for probe, feats in (("mlp", Z[idx]), ("window", Zw[idx])):
                if probe == "window":
                    G = feats @ feats.T
                    G = G / float(np.mean(np.diag(G)))
                    pr, _ = P.dual_ridge_oof(G, y[idx], eps, sub)
                else:
                    pr = mlp_oof(feats, y[idx], eps, sub, binary=binary)
                pt, lo, hi = P.episode_cluster_bootstrap(pr, y[idx], eps, binary)
                res[probe] = {"point": round(pt, 4),
                              "ci95": [round(lo, 4), round(hi, 4)]}
            out["arms"][arm]["targets"][t] = res
            print(f"  {arm:<20}{t:<16} mlp {res['mlp']['point']:+.4f} "
                  f"{res['mlp']['ci95']}   window {res['window']['point']:+.4f} "
                  f"{res['window']['ci95']}", flush=True)
        (SP / "e_probe_power.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    print(f"\n-> {SP / 'e_probe_power.json'}")


if __name__ == "__main__":
    main()
