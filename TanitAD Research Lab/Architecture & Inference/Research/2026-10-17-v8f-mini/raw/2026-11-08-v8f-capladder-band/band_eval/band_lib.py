"""band_lib.py -- band slices of the existing four_families geometry (zero training, CPU only).

Re-uses taniteval.four_families._seq_geometry (tip 26fb0bc1) on the WHOLE 30-knot track, then restricts the PER-STEP arrays to a knot band
[a, b) (knot i is time (i+1)*0.2 s; operative = knots 0..9 = 0.2..2.0 s, tactical = knots 10..29 = 2.2..6.0 s).
Why the geometry is computed on the full track and sliced afterwards: a slice cut BEFORE the geometry would restart the origin at the band edge
and turn the first tactical step into a chord from the ego position at NOW.  The per-window means follow t1k_read.comps (masked mean, NaN when empty).

Step i (speed, heading, valid, along, cross) belongs to a band when a <= i < b.
Pair j (yaw_rate, curvature, accel; between step j and j+1) belongs to a band when a <= j and j+1 < b.  The pair that straddles the seam
(j = 9 for 10/20 knots) is in neither band; a full band (0, 30) holds all 29 pairs, so a full-band slice equals comps() exactly.
"""
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
B = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/band_eval/tree"
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", B + "/stack")
sys.path.insert(0, B + "/taniteval")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from taniteval import four_families as FF  # noqa: E402
from taniteval import ci as CI  # noqa: E402
from taniteval import lead_metrics as LM  # noqa: E402

DT = 0.2
K = 30
BANDS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30), "full_0_6s": (0, 30)}
METRICS = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps",
           "curvature_mae_1pm")
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL",
          "cross_mae_m": "LATERAL", "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL"}


def _mm(x, m):
    s, c = (x * m).sum(1), m.sum(1)
    return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))


def band_rows(X, G, a, b):
    """Per-window [n] rows of track X against GT G (both [n, 30, 2]) inside knot band [a, b)."""
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    n = Xt.shape[0]
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    e = torch.linalg.norm(Xt - Gt, dim=-1)
    sm = torch.zeros(K, dtype=torch.bool)               # step membership
    sm[a:b] = True
    pm = torch.zeros(K - 1, dtype=torch.bool)           # pair membership: both steps inside the band
    if b - a >= 2:
        pm[a:b - 1] = True
    smf, pmf = sm[None, :].expand(n, K), pm[None, :].expand(n, K - 1)
    both = P["valid"] & Q["valid"] & smf
    bp = P["pair_valid"] & Q["pair_valid"] & pmf
    dh = P["heading"] - Q["heading"]
    dh = ((dh + math.pi) % (2 * math.pi) - math.pi).abs()
    fs, fp = smf.to(torch.float64), pmf.to(torch.float64)

    def mean_all(x, f):
        return (x * f).sum(1) / f.sum(1)
    out = {"ade_m": mean_all(e, fs), "fde_m": e[:, b - 1],
           "speed_mae_mps": mean_all((P["speed"] - Q["speed"]).abs(), fs),
           "along_mae_m": mean_all((P["along"] - Q["along"]).abs(), fs),
           "accel_mae_mps2": mean_all((P["accel"] - Q["accel"]).abs(), fp) if pm.any() else torch.full((n,), float("nan"), dtype=torch.float64),
           "cross_mae_m": mean_all((P["cross"] - Q["cross"]).abs(), fs),
           "heading_mae_deg": torch.rad2deg(_mm(dh, both)),
           "yaw_rate_mae_degps": torch.rad2deg(_mm((P["yaw_rate"] - Q["yaw_rate"]).abs(), bp)),
           "curvature_mae_1pm": _mm((P["curvature"] - Q["curvature"]).abs(), bp)}
    return {k: v.numpy().astype(np.float64) for k, v in out.items()}


def band_dk(X, Jd, a, b):
    """distance_keeping (headway / time gap / TTC) of track X over knots [a, b), per window (NaN where no lead or not in the corridor)."""
    return LM.distance_keeping(np.asarray(X, np.float64)[:, a:b], Jd["leads"][:, a:b], Jd["lead_lens"], Jd["speeds"], DT)


def load_dir(d):
    import glob
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == 141, (d, len(fs))
    out = {}
    zs = [np.load(f) for f in fs]
    for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws"):
        out[k] = np.concatenate([z[k] for z in zs])
    return out
