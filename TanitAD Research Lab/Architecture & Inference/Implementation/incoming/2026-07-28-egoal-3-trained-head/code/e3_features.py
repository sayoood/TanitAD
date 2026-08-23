#!/usr/bin/env python3
"""E-GOAL-3 S1 -- PAST-ONLY ego-kinematic features + the 2 s along-track target,
built from `poses[T,4]` on the PARITY caches, plus the C23 future-content audit
and the content-based leak fingerprint.

⛔ PRIORITY 1 IS THE AUDIT, NOT THE FEATURES. A leaked head is worse than no
head. `head_deg` in every fan dump is the FUTURE net heading change
(`driving_diagnostic.net_heading_change_deg` reads `poses[last + horizon, 2]`)
and sits beside `v0`; a TRAINED head is far more exposed to that than a
resampler was. Three instruments, all run BEFORE anything is fitted:

  1. BY DEFINITION -- every column's index expression is <= L (the window's last
     OBSERVED frame). The table is in PRE_REGISTRATION §3.
  2. ⭐ THE `future_blind` TEST -- recompute every feature from a pose array whose
     rows > L have been overwritten with large random garbage. A feature that
     reads the future CHANGES. Required: max |Δ| == 0.0 over every window.
     ⛔ Its FAILING HALF: the TARGET `y_long` must CHANGE under the identical
     corruption, or the instrument has no power and proves nothing.
  3. NEGATIVE-INDEX GUARD -- a negative Python index wraps to the END of the
     array, i.e. it silently reads the FUTURE. `v_lag_1p0` at the first window
     of every episode is `L-10 = -3`. Every index is CLAMPED at 0 and the clamp
     count is reported.

WINDOWS. Reconstructed from `refc_rerank.dump()` verbatim:
    starts = range(0, T - WINDOW - K_MAX, STRIDE),  WINDOW=8, STRIDE=8, K_MAX=20
    L      = start + WINDOW - 1                      (the last OBSERVED frame)
`eid` in the dump is `data.load_frames`'s FILE INDEX (`RawEp(..., i)`), not the
cache's stored `episode_id` -- so the join is by sorted-file position and is
unambiguous. Proven by gates F-2/F-4 in `e3_place.py`, not assumed.

LEAK FINGERPRINT. sha256 over the raw `poses[T,4]` float32 bytes -- computed on
THE SAME in-memory tensor the features are derived from, so the fingerprint is
of the bytes actually used, not of a second read of a possibly different copy.
(E-GOAL-2's own gap was that a sibling audited `_epcache` while the dump read
`/root/valdata` on the same pod.)

PARITY: read-only. `_epcache` is never written, no episode is re-selected.
PRIVACY: no clip UUID or raw content reaches any artifact.

Run on pod2 (the only host with both the 600-episode val build and the parity
train corpus):
    OMP_NUM_THREADS=6 nice -n 10 python3 e3_features.py --which val
    OMP_NUM_THREADS=6 nice -n 10 python3 e3_features.py --which train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

# --- the canonical val protocol, copied from refc_rerank.py --------------- #
WINDOW, STRIDE, K_MAX, DT = 8, 8, 20, 0.1
V_FLOOR = 0.5                      # m/s; curvature is meaningless below this

VAL_READ = "/root/valdata/physicalai-val-0c5f7dac3b11"          # the dump reads THIS
TRAIN = ("/workspace/data/physicalai_phase0/_epcache/"
         "physicalai-train-e438721ae894")

#: the fed columns, in this order. Mirrors `lead_state_gate.EGO_COLS`.
EGO3_COLS = ["v", "ax", "ay", "curv", "abs_curv", "yawrate",
             "dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
HIST_COLS = ["dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]

#: every index offset a feature may read, relative to L. ALL <= 0 BY DEFINITION.
FEATURE_OFFSETS = (0, -1, -2, -5, -10)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def features(poses: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T,4] poses + [n] last-observed indices -> ([n,10] features, [n] clamped).

    EVERY index is `max(L + offset, 0)`. There is no positive offset anywhere in
    this function; that is the property gate `future_blind` verifies empirically
    rather than by reading this docstring.
    """
    s = poses[:, 3]
    yaw = poses[:, 2]

    def at(off, arr):
        assert off <= 0, f"POSITIVE OFFSET {off} IS FUTURE CONTENT"
        return arr[np.maximum(L + off, 0)]

    clamped = np.zeros(len(L), bool)
    for off in FEATURE_OFFSETS:
        clamped |= (L + off) < 0

    v = at(0, s)
    ax = (v - at(-1, s)) / DT
    yawrate = wrap(at(0, yaw) - at(-2, yaw)) / (2 * DT)
    curv = yawrate / np.maximum(v, V_FLOOR)
    ay = v * yawrate
    v_l05, v_l10 = at(-5, s), at(-10, s)
    X = np.stack([v, ax, ay, curv, np.abs(curv), yawrate,
                  v - v_l05, v - v_l10, v_l05, v_l10], 1)
    return X.astype(np.float64), clamped


def target(poses: np.ndarray, L: np.ndarray) -> np.ndarray:
    """[n,2] ego-frame displacement over K_MAX steps -- `gt_ego_waypoints`'s last
    horizon, re-derived: (along, cross). ⛔ THE TARGET IS FUTURE BY DEFINITION."""
    p0 = poses[L, :2]
    p1 = poses[L + K_MAX, :2]
    d = p1 - p0
    yaw0 = poses[L, 2]
    c, sn = np.cos(yaw0), np.sin(yaw0)
    return np.stack([d[:, 0] * c + d[:, 1] * sn,
                     -d[:, 0] * sn + d[:, 1] * c], 1)


def starts_of(T: int) -> np.ndarray:
    return np.arange(0, max(T - WINDOW - K_MAX, 0), STRIDE)


def read_poses(f: Path) -> tuple[np.ndarray, str]:
    """poses[T,4] float32 + the sha256 of ITS OWN raw bytes (the leak key)."""
    ep = torch.load(str(f), map_location="cpu", mmap=True, weights_only=False)
    p = ep["poses"] if isinstance(ep, dict) else ep.poses
    t = torch.as_tensor(p).float().contiguous()
    return t.numpy().astype(np.float64), hashlib.sha256(
        t.numpy().tobytes()).hexdigest()


# ======================================================================== #
# ⭐ THE AUDIT                                                              #
# ======================================================================== #
def future_blind_audit(poses: np.ndarray, L: np.ndarray, rng) -> dict:
    """Recompute every feature from a pose array whose rows > L are GARBAGE.

    Per window, because L differs per window. Required: features bit-identical
    (max |Δ| == 0.0). Required for POWER: the target must CHANGE.
    """
    X0, _ = features(poses, L)
    Y0 = target(poses, L)
    dX = 0.0
    dY = 0.0
    n_target_changed = 0
    for i, li in enumerate(L):
        q = poses.copy()
        q[li + 1:] = rng.normal(1e4, 1e4, q[li + 1:].shape)
        xi, _ = features(q, np.array([li]))
        dX = max(dX, float(np.max(np.abs(xi[0] - X0[i]))))
        # the target reads poses[L + 20] -> it MUST move
        yi = target(q, np.array([li]))
        dy = float(np.max(np.abs(yi[0] - Y0[i])))
        dY = max(dY, dy)
        n_target_changed += int(dy > 0)
    return {"max_abs_feature_delta": dX, "features_unchanged": bool(dX == 0.0),
            "max_abs_target_delta": dY,
            "target_changed_on_n_windows": int(n_target_changed),
            "n_windows_audited": int(len(L)),
            "power": bool(n_target_changed == len(L))}


# ======================================================================== #
def build(root: str, label: str, stride: int, audit_eps: int, out: Path) -> dict:
    files = sorted(Path(root).glob("ep_*.pt"))
    print(f"[{label}] {len(files)} episodes under {root}", flush=True)
    rng = np.random.default_rng(20260728)

    Xs, Ys, EPI, LS, CL, SHA, TT = [], [], [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        poses, sha = read_poses(f)
        T = len(poses)
        st = starts_of(T) if stride == STRIDE else np.arange(
            0, max(T - WINDOW - K_MAX, 0), stride)
        if not len(st):
            SHA.append(sha)
            TT.append(T)
            continue
        L = st + WINDOW - 1
        assert (L + K_MAX < T).all(), "target index past the array"
        X, cl = features(poses, L)
        Y = target(poses, L)
        Xs.append(X)
        Ys.append(Y)
        EPI.append(np.full(len(L), i))
        LS.append(L)
        CL.append(cl)
        SHA.append(sha)
        TT.append(T)
        if (i + 1) % 200 == 0:
            print(f"  [{label}] {i+1}/{len(files)} ({time.time()-t0:.0f}s)",
                  flush=True)

    # ⭐ the audit, over the FIRST `audit_eps` episodes' every window
    aud = {"max_abs_feature_delta": 0.0, "max_abs_target_delta": 0.0,
           "target_changed_on_n_windows": 0, "n_windows_audited": 0}
    for i, f in enumerate(files[:audit_eps]):
        poses, _ = read_poses(f)
        st = starts_of(len(poses)) if stride == STRIDE else np.arange(
            0, max(len(poses) - WINDOW - K_MAX, 0), stride)
        if not len(st):
            continue
        a = future_blind_audit(poses, st + WINDOW - 1, rng)
        aud["max_abs_feature_delta"] = max(aud["max_abs_feature_delta"],
                                           a["max_abs_feature_delta"])
        aud["max_abs_target_delta"] = max(aud["max_abs_target_delta"],
                                          a["max_abs_target_delta"])
        aud["target_changed_on_n_windows"] += a["target_changed_on_n_windows"]
        aud["n_windows_audited"] += a["n_windows_audited"]
    aud["features_unchanged"] = bool(aud["max_abs_feature_delta"] == 0.0)
    aud["power"] = bool(aud["target_changed_on_n_windows"]
                        == aud["n_windows_audited"] > 0)

    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    epi = np.concatenate(EPI)
    L = np.concatenate(LS)
    cl = np.concatenate(CL)
    meta = {
        "_label": label, "_root": root, "_stride": stride,
        "_n_episodes": len(files), "_n_windows": int(len(X)),
        "_cols": EGO3_COLS,
        "_protocol": {"WINDOW": WINDOW, "STRIDE": STRIDE, "K_MAX": K_MAX,
                      "DT": DT, "L": "start + WINDOW - 1 (last OBSERVED frame)"},
        "C23_future_blind_audit": aud,
        "C23_negative_index_clamp": {
            "what": ("a negative index wraps to the END of the array = FUTURE "
                     "content. Every index is max(L+off, 0)."),
            "offsets_read": list(FEATURE_OFFSETS),
            "max_offset": int(max(FEATURE_OFFSETS)),
            "all_offsets_non_positive": bool(max(FEATURE_OFFSETS) <= 0),
            "n_windows_clamped": int(cl.sum()),
            "frac_windows_clamped": round(float(cl.mean()), 6)},
        "pose_sha256_per_episode": {f.name: {"sha256": s, "T": t}
                                    for f, s, t in zip(files, SHA, TT)},
        "_wall_s": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, Y=Y, epi=epi, L=L, clamped=cl,
                        cols=np.array(EGO3_COLS))
    print(f"[{label}] {len(X)} windows / {len(files)} episodes -> {out} "
          f"({meta['_wall_s']}s)", flush=True)
    print(f"[{label}] C23 audit: {json.dumps(aud)}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["val", "train"], required=True)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--audit-eps", type=int, default=8)
    ap.add_argument("--outdir", default="/workspace/_egoal3")
    a = ap.parse_args()
    od = Path(a.outdir)
    if a.which == "val":
        meta = build(VAL_READ, "val600", a.stride or STRIDE, a.audit_eps,
                     od / "e3_val600_windows.npz")
    else:
        meta = build(TRAIN, "train2376", a.stride or 1, a.audit_eps,
                     od / "e3_train2376_windows.npz")
    (od / f"e3_features_{a.which}.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {od / f'e3_features_{a.which}.json'}", flush=True)


if __name__ == "__main__":
    main()
