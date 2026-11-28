"""X3's LEAK instrument (SPEC_REFCV8 sec. 8.3 (a)(1)): how much FUTURE-speed information a speed INPUT carries beyond
the current speed v0 -- D4's method (`TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/
D4_usage_and_design/code/d4_vmax_nonoracle.py`), restated here so the trainer's own tree can run it on the channel it
actually FEEDS (through the model's own 4-way binning), not only on the 8-step value D4 measured.

The estimator, exactly D4's:
* windows: provider rows of the v2ep manifest, window 8 (NOW = provider row t + 7), only windows with a FULL 6-s
  future (row NOW + 60 inside the clip);
* target ``y`` = the max speed over provider rows [NOW + 20, NOW + 60] (= [NOW + 2 s, NOW + 6 s]);
* 5-fold CLIP-GROUPED out-of-fold ordinary least squares, fold = ``int(sha256(clip_id)[:12], 16) % 5``;
* base design ``[1, v0]``; a feature block is appended as columns;
* ``recovered = (R2_feat - R2_base) / (1 - R2_base)`` -- the share of the information v0 lacks that the input supplies.

Known-value controls: a constant block recovers exactly 0; the target itself exactly 1 (pinned in
`tests/test_refcv8_speed_input.py`). ⛔ No clip id is ever written by this module -- only counts and shares.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

#: D4's literals (`d4_lib.py`), re-typed
W, MAXH, FUT = 8, 20, 60
PAST_ROWS = 200                                   # 20 s at 10 Hz
LADDER8_KMH = (20, 30, 50, 70, 80, 100, 120, 130)
LADDER4_KMH = (30, 50, 100, 120)
URBAN_FLOOR_KMH = 50.0


def fold_of(clip_id: str) -> int:
    return int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12], 16) % 5


def snap_up(v_ms, ladder_kmh) -> np.ndarray:
    """Containing-window ladder index: the lowest step >= v (km/h); the top step caps (D4's ``snap_up``)."""
    v = np.asarray(v_ms, np.float64) * 3.6
    idx = np.searchsorted(np.asarray(ladder_kmh, np.float64), v - 1e-9, side="left")
    return np.minimum(idx, len(ladder_kmh) - 1)


def onehot(idx, n: int, valid=None) -> np.ndarray:
    idx = np.asarray(idx, np.int64)
    o = np.zeros((len(idx), n))
    o[np.arange(len(idx)), np.clip(idx, 0, n - 1)] = 1.0
    if valid is not None:
        o *= np.asarray(valid, bool)[:, None]
    return o


def oof_r2(y: np.ndarray, X: np.ndarray, folds: np.ndarray, k: int = 5) -> float:
    yh = np.zeros_like(y)
    for f in range(k):
        tr = folds != f
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        yh[folds == f] = X[folds == f] @ w
    return float(1.0 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def recovered(y: np.ndarray, v0: np.ndarray, folds: np.ndarray, blocks: Mapping[str, np.ndarray]) -> dict:
    """{name: {oof_r2, recovered}} for each feature block, plus the base row."""
    base = np.hstack([np.ones((len(y), 1)), np.asarray(v0, np.float64)[:, None]])
    r0 = oof_r2(y, base, folds)
    out = {"_base": {"oof_r2": round(r0, 6), "n": int(len(y))}}
    for name, blk in blocks.items():
        blk = np.asarray(blk, np.float64)
        X = base if blk.size == 0 else np.hstack([base, blk.reshape(len(y), -1)])
        rr = oof_r2(y, X, folds)
        out[name] = {"oof_r2": round(rr, 6), "recovered": round((rr - r0) / (1.0 - r0), 6)}
    return out


def clip_windows(poses: np.ndarray) -> dict:
    """Per-clip window arrays from provider-row poses ``[T, 4]`` = (x, y, yaw, v): the D4 window grid."""
    v = np.asarray(poses, np.float64)[:, 3]
    T = len(v)
    n = T - (W + MAXH)
    if n <= 0:
        return {"n": 0}
    r = np.arange(n) + W - 1                                       # provider row of NOW
    K = np.arange(FUT + 1)
    idx = r[:, None] + K[None, :]
    valid = idx <= T - 1
    Vv = np.where(valid, v[np.minimum(idx, T - 1)], -np.inf)
    full = valid[:, FUT]
    y = np.where(full, Vv[:, 20:FUT + 1].max(1), np.nan)
    f6 = Vv.max(1)                                                 # [NOW, NOW + 6 s] (the oracle control)
    p20 = np.array([v[max(0, rr - PAST_ROWS):rr + 1].max() for rr in r])
    return {"n": n, "r": r, "y": y, "v0": v[r], "f6": f6, "p20": p20}


def fed_bin4(v_ms, known) -> np.ndarray:
    """The 4-way block the TRAINER feeds: the model's own `speed_max_bin_tensor` on float32 m/s (CONTAINING-WINDOW
    {30, 50, 100, 120} km/h, > 120 to the top step), zeroed where unknown -- `MaxSpeedOneHotEncoder`'s input."""
    import torch
    from tanitad.refs import refcv6_max_speed as v6ms
    v = torch.tensor(np.where(np.asarray(known, bool), np.asarray(v_ms, np.float64), 0.0), dtype=torch.float32)
    idx, _over = v6ms.speed_max_bin_tensor(v)
    return onehot(idx.numpy(), v6ms.N_SPEED_MAX_BINS_V6, known)


def fed_enc8(v_ms, known) -> np.ndarray:
    """refcv8 (B) `--r8-speed-enc8`: the 9-dim block the model's own `refcv8_conditioning.speed_enc8` feeds (8-step
    one-hot + known bit; an unknown row is all zeros)."""
    import torch
    from tanitad.refs import refcv8_conditioning as r8c
    k = np.asarray(known, bool)
    v = torch.tensor(np.where(k, np.asarray(v_ms, np.float64), 0.0), dtype=torch.float32)
    return r8c.speed_enc8(v, torch.tensor(k, dtype=torch.float32)).numpy().astype(np.float64)


def x3_leak_table(manifest_path: str, v9_npz: str, v8_sidecar: str | None = None, *, unknown_p: float = 0.45,
                  seed: int = 2, roll_block: int = 8) -> dict:
    """The X3 LEAK table on the REAL artifacts. Rows:

    * ``D4_N2_from_poses_bin8`` -- D4's own N2 recomputed from the manifest poses (the instrument's identity with D4);
    * ``v9_N2_bin8`` -- WP-A's shipped ``speed_n2_kmh`` (their artifact), 8-step one-hot (D4: 0.0346);
    * ``v9_N2_fed_bin4`` / ``..._unknown_p0.45`` -- what refcv8 FEEDS (eval / training);
    * ``v9_N2_fed_bin4_ROLLED`` -- the V-VSHUF information control: each window takes the N2 of the window
      ``roll_block`` positions away inside a random permutation (another clip with probability ~1) -> ~0;
    * ``v9_N3_fed_bin4``;
    * ``v8_sidecar_bin4_D4`` -- refcv7's per-clip v8 value, 4-way, an invalid row binned as 0 m/s exactly as D4 did
      (USAGE_AUDIT: 0.238); ``v8_sidecar_bin4_as_fed`` -- the same with the invalid row ALL-ZERO, as the trainer feeds it;
    * ``ORACLE_window_0_6s_bin8`` (D4: 0.5698), ``CONTROL_constant`` (0 exactly), ``CONTROL_target_itself`` (1).
    """
    import torch
    from tanitad.data import v9_labels as V9
    from tanitad.data.v2_dataset import stable_episode_id
    man = torch.load(manifest_path, map_location="cpu", weights_only=False)
    rel = V9.load_v9_release(v9_npz)
    side = None
    if v8_sidecar:
        from tanitad.refs import refcv6_max_speed as v6ms
        side, _meta = v6ms.read_speed_max_sidecar_v6(str(v8_sidecar))
    cols = {"y": [], "v0": [], "f6": [], "p20": [], "fold": [], "n2": [], "n3": [], "v9_hit": [], "v8": [], "v8ok": []}
    n_clip_v9 = 0
    for i, cid in enumerate(man["clip_id"]):
        B = clip_windows(man["poses"][i].numpy())
        if B["n"] == 0:
            continue
        sid = int(stable_episode_id(cid))
        c = rel.clips.get(sid)
        n2 = np.full(B["n"], -1, np.int64)
        n3 = np.full(B["n"], -1, np.int64)
        hit = np.zeros(B["n"], bool)
        if c is not None:
            n_clip_v9 += 1
            r0, nr, k0 = c
            j = B["r"] + 2 - k0                                        # raw row k = provider row + 2
            ok = (j >= 0) & (j < nr)
            rows = r0 + j[ok]
            n2[ok] = np.asarray(rel.rows["speed_n2_kmh"])[rows]
            n3[ok] = np.asarray(rel.rows["speed_n3"])[rows]
            hit[ok] = True
        vs = side.get(sid) if side is not None else None
        cols["y"].append(B["y"])
        cols["v0"].append(B["v0"])
        cols["f6"].append(B["f6"])
        cols["p20"].append(B["p20"])
        cols["fold"].append(np.full(B["n"], fold_of(cid)))
        cols["n2"].append(n2)
        cols["n3"].append(n3)
        cols["v9_hit"].append(hit)
        cols["v8"].append(np.full(B["n"], 0.0 if vs is None else float(vs[0])))
        cols["v8ok"].append(np.full(B["n"], vs is not None and float(vs[1]) > 0.5))
    A = {k: np.concatenate(v) for k, v in cols.items()}
    m = np.isfinite(A["y"])
    A = {k: v[m] for k, v in A.items()}
    y, v0, folds = A["y"], A["v0"], A["fold"]
    n = len(y)
    known2 = A["n2"] >= 0
    rng = np.random.default_rng(seed)
    unk = rng.random(n) < float(unknown_p)
    # V-VSHUF information control: within a random permutation, each window takes the value `roll_block` slots on
    perm = rng.permutation(n)
    rolled = np.empty(n, np.int64)
    rolled[perm] = A["n2"][np.roll(perm, roll_block)]
    lad8 = np.asarray(LADDER8_KMH, np.float64) / 3.6
    b_d4 = snap_up(np.maximum(A["p20"], URBAN_FLOOR_KMH / 3.6), LADDER8_KMH)
    b_v9 = snap_up(np.where(known2, A["n2"], 0) / 3.6, LADDER8_KMH)
    n3_kmh = np.where(A["n3"] == 0, 50.0, np.where(A["n3"] == 1, 100.0, 130.0))
    blocks = {
        "CONTROL_constant": np.ones((n, 1)),
        "CONTROL_target_itself": y[:, None],
        "D4_N2_from_poses_bin8": onehot(b_d4, 8),
        "v9_N2_bin8": onehot(b_v9, 8, known2),
        "v9_N2u_bin8_unknown_p%.2f" % unknown_p: onehot(b_v9, 8, known2 & ~unk),       # D4's N2u (0.0093)
        "v9_N2_fed_bin4": fed_bin4(A["n2"] / 3.6, known2),
        "v9_N2_fed_bin4_unknown_p%.2f" % unknown_p: fed_bin4(A["n2"] / 3.6, known2 & ~unk),
        "v9_N2_fed_bin4_ROLLED": fed_bin4(rolled / 3.6, rolled >= 0),
        "v9_N3_fed_bin4": fed_bin4(n3_kmh / 3.6, A["n3"] >= 0),
        # refcv8 (B): the 4-way block AND the 8-step seam, as `--r8-speed-enc8` feeds them together
        "v9_N2_fed_bin4+enc8": np.hstack([fed_bin4(A["n2"] / 3.6, known2), fed_enc8(A["n2"] / 3.6, known2)]),
        "v9_N2_fed_bin4+enc8_unknown_p%.2f" % unknown_p: np.hstack([fed_bin4(A["n2"] / 3.6, known2 & ~unk),
                                                                     fed_enc8(A["n2"] / 3.6, known2 & ~unk)]),
        "v9_N2_fed_bin4+enc8_ROLLED": np.hstack([fed_bin4(rolled / 3.6, rolled >= 0),
                                                fed_enc8(rolled / 3.6, rolled >= 0)]),
        "ORACLE_window_0_6s_bin8": onehot(snap_up(A["f6"], LADDER8_KMH), 8),
    }
    if side is not None:
        blocks["v8_sidecar_bin4_D4"] = onehot(snap_up(np.where(A["v8ok"], A["v8"], 0.0), LADDER4_KMH), 4)
        blocks["v8_sidecar_bin4_as_fed"] = fed_bin4(A["v8"], A["v8ok"])
    rows = recovered(y, v0, folds, blocks)
    agree = known2 & (b_v9 == b_d4)
    return {
        "_evidence": "MEASURED (WP-B X3 leak instrument = D4's estimator; dev-box CPU)",
        "n_windows": int(n), "n_clips": int(len(man["clip_id"])), "n_clips_in_v9": int(n_clip_v9),
        "v9_join_frac": round(float(A["v9_hit"].mean()), 6), "n2_known_frac": round(float(known2.mean()), 6),
        "v9_N2_equals_D4_N2_frac": round(float(agree.sum()) / max(int(known2.sum()), 1), 6),
        "unknown_p": float(unknown_p), "seed": int(seed), "roll_block": int(roll_block),
        "rows": rows,
    }
