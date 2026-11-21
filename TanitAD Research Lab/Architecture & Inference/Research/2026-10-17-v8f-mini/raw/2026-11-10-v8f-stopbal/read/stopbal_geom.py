"""stopbal_geom.py -- per-window geometry rows for the STOPBAL read (CPU, zero training). Needs torch and the read tree (env STOPBAL_TREE).

The geometry, the lead join and the manoeuvre labels are theory/otband/read/read_otband.py's, unchanged (join, labels, arm_arrays), cut to the two bands of
the STOPBAL card (operative knots 1-10 = [0,10), tactical knots 11-30 = [10,30)) and without the DK-VIOL block (distance keeping is read as the absolute
errors against the human's own minimum headway and time gap, OTBAND reading choice r2).
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import read_lib_stopbal as BL  # noqa: E402
from read_lib_stopbal import LM, FF, DT  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
RAW = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
LEAD = ART + "/dagoacc_t1/inputs/b1_eval_lead_block_k30.npz"
LEAD_SHA = "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
BANDS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30)}
METRICS = list(BL.METRICS) + ["dk_hw_abs_err_m", "dk_tg_abs_err_s"]
FAMILY = dict(BL.FAMILY, dk_hw_abs_err_m="LONGITUDINAL", dk_tg_abs_err_s="LONGITUDINAL")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_BLK = {}


def _lead_block():
    if not _BLK:
        assert sha(LEAD) == LEAD_SHA, "lead block sha mismatch"
        blk = dict(np.load(LEAD, allow_pickle=True))
        idx = {(c, int(f)): i for i, (c, f) in enumerate(zip(np.asarray(blk["clip_id"]).astype(str).tolist(), np.asarray(blk["frame"]).astype(np.int64).tolist()))}
        assert len(idx) == len(blk["leads"]), "duplicate lead rows"
        _BLK.update(blk=blk, idx=idx, clip_of=np.load(RAW + "/join_eval141.npz", allow_pickle=False)["clip"].astype(str))
    return _BLK["blk"], _BLK["idx"], _BLK["clip_of"]


def join(d):
    """read_otband.join, unchanged: the lead track and speeds of the eval windows of dump dict d."""
    blk, idx, clip_of = _lead_block()
    clip = clip_of[d["clip_index"]]
    frame = 2 * d["ws"].astype(np.int64)
    rows = np.array([idx.get((c, int(f)), -1) for c, f in zip(clip.tolist(), frame.tolist())])
    assert (rows >= 0).all(), "grid mismatch"
    bsp = np.asarray(blk["speeds"], np.float64).reshape(-1)[rows]
    assert float(np.max(np.abs(d["v0"] - bsp))) <= 1e-3, "speed proof failed"
    st = np.asarray(blk["state"]).astype(str).reshape(-1)[rows]
    is_lead = st == "LEAD"
    leads = np.full((rows.size, 30, 2), np.nan)
    lens = np.full(rows.size, np.nan)
    leads[is_lead] = np.asarray(blk["leads"], np.float64)[rows[is_lead]][:, :30]
    lens[is_lead] = np.asarray(blk["lead_lens"], np.float64).reshape(-1)[rows[is_lead]]
    return {"leads": leads, "lead_lens": lens, "speeds": d["v0"].astype(np.float64), "is_lead": is_lead}


def labels(X):
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def arm_arrays(X, G, J):
    """{band: {metric: [N]}} for track X against GT G, plus _lat/_lon plan manoeuvre labels (read_otband.arm_arrays without DK-VIOL)."""
    out = {}
    for band, (a, b) in BANDS.items():
        r = BL.band_rows(X, G, a, b)
        dk, H = BL.band_dk(X, J, a, b), BL.band_dk(G, J, a, b)
        r["dk_hw_abs_err_m"] = np.abs(np.asarray(dk["headway_min_m"], np.float64) - np.asarray(H["headway_min_m"], np.float64))
        r["dk_tg_abs_err_s"] = np.abs(np.asarray(dk["time_gap_min_s"], np.float64) - np.asarray(H["time_gap_min_s"], np.float64))
        out[band] = r
    out["_lat"], out["_lon"] = labels(X)
    return out
