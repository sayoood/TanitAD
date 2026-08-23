"""IDM-v2 shared substrate: window building, honest per-channel metrics
(incl. scale-free ones), and episode-cluster-bootstrap wrappers.

Estimator policy (CLAUDE.md): every interval here is
`taniteval.ci.episode_cluster_bootstrap` (or its PAIRED form) over EPISODES.
`overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

from taniteval import ci as tci  # noqa: E402
import idm_head as ih            # noqa: E402

LAT = Path("/root/idm2/lat")
DT = 0.1
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
UNITS = ("m/s", "rad/s", "norm", "m/s^2")
# Physical admissibility limits for a road vehicle (used ONLY to COUNT
# impossible labels; nothing is silently dropped without being reported).
PHYS_LIMIT = {"speed": 60.0, "yaw_rate": 1.5, "steer": 1.0, "long_accel": 12.0}


# --------------------------------------------------------------------------- #
# episodes / split                                                             #
# --------------------------------------------------------------------------- #
def all_tags():
    return sorted(p.stem for p in LAT.glob("*.pt"))


def split_tags(val_every: int = 3):
    """Episode-disjoint, domain-stratified, deterministic. Every `val_every`-th
    episode of each domain goes to VAL."""
    tr, va = [], []
    for dom in ("pai", "cm"):
        tags = sorted(t for t in all_tags() if t.startswith(dom + "_"))
        for i, t in enumerate(tags):
            (va if i % val_every == 0 else tr).append(t)
    return tr, va


def load_ep(tag):
    d = torch.load(LAT / f"{tag}.pt", weights_only=False)
    return d


def build_set(tags, k=4, stride=1, horizons=ih.DEFAULT_HORIZONS,
              want_seq=False, want_z=True):
    """-> dict with Z [N,2k+1,D] (fp16), S [N,4], Traj [N,H,2], eid [N] str,
    dom [N] str, tcen [N] int, and (optional) Vseq [N,2k+1] speed sequence."""
    Zs, Ss, Ts, eids, doms, tcs, Vs = [], [], [], [], [], [], []
    for tag in tags:
        d = load_ep(tag)
        z = d["z"].float()
        po = d["poses"].float()
        ac = d["actions"].float()
        zw, sc, tj = ih.build_windows(z, po, ac, k=k, horizons=horizons,
                                      stride=stride)
        if zw.shape[0] == 0:
            continue
        t = ih.valid_centers(z.shape[0], k, horizons, stride)
        if want_z:
            Zs.append(zw.half())
        Ss.append(sc)
        Ts.append(tj)
        eids += [tag] * sc.shape[0]
        doms += [d.get("domain", tag.split("_")[0])] * sc.shape[0]
        tcs.append(t)
        if want_seq:
            offs = torch.arange(-k, k + 1)
            Vs.append(po[t[:, None] + offs[None, :], 3])
    out = {"S": torch.cat(Ss), "Traj": torch.cat(Ts),
           "eid": np.array(eids), "dom": np.array(doms),
           "tcen": torch.cat(tcs)}
    if want_z:
        out["Z"] = torch.cat(Zs)
    if want_seq:
        out["Vseq"] = torch.cat(Vs)
    return out


# --------------------------------------------------------------------------- #
# metrics — R2 is NOT enough for the low-variance channels                     #
# --------------------------------------------------------------------------- #
def spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def chan_metrics(pred, gt, label=None):
    """Per-channel honest metric bundle.

    R2               classic, MISLEADING on low-variance / heavy-tailed channels
    rho              Spearman rank corr  (scale-free, outlier-robust)
    mae / medae      location errors
    nmedae           medae / MAD(gt)  -- SCALE-FREE skill: <1 beats "predict the
                     median"; this is the number to quote for yaw_rate/long_accel
    rmse
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    err = p - g
    mad = float(np.median(np.abs(g - np.median(g))))
    ss_res = float((err ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rho": spearman(p, g),
            "mae": float(np.abs(err).mean()),
            "medae": float(np.median(np.abs(err))),
            "nmedae": float(np.median(np.abs(err)) / max(mad, 1e-12)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "gt_std": float(g.std()), "gt_mad": mad, "n": int(g.size)}


def boot_r2(pred, gt, eid, n_boot=2000, seed=0):
    """Episode-cluster bootstrap CI on R2, using ci.py's documented callable
    reducer path (per_window carries the window INDEX; the reducer recomputes
    R2 on the resampled index set, so the resampling unit is the EPISODE)."""
    p = np.asarray(pred, dtype=np.float64); g = np.asarray(gt, dtype=np.float64)

    def _r2(idx):
        i = idx.astype(np.int64)
        gg, pp = g[i], p[i]
        ssr = ((pp - gg) ** 2).sum()
        sst = ((gg - gg.mean()) ** 2).sum()
        return float(1.0 - ssr / max(sst, 1e-12))
    _r2.__name__ = "r2"
    return tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                         eid, reduce=_r2, n_boot=n_boot,
                                         seed=seed)


def boot_mae(pred, gt, eid, n_boot=2000, seed=0, reduce="mean"):
    return tci.episode_cluster_bootstrap(np.abs(np.asarray(pred, np.float64) -
                                                np.asarray(gt, np.float64)),
                                         eid, reduce=reduce, n_boot=n_boot,
                                         seed=seed)


def paired_mae(pred_a, pred_b, gt, eid, n_boot=2000, seed=0, reduce="mean"):
    """delta = MAE(a) - MAE(b) on the SAME windows.  Negative => a is better."""
    a = np.abs(np.asarray(pred_a, np.float64) - np.asarray(gt, np.float64))
    b = np.abs(np.asarray(pred_b, np.float64) - np.asarray(gt, np.float64))
    return tci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot,
                                                seed=seed, reduce=reduce)


def jdump(obj, path):
    def _d(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_d))
    print("WROTE", path, flush=True)
