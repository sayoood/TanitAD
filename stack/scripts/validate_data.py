"""Axis-1 data-consistency GATE for the flagship run (PRE_FLIGHT_VALIDATION.md).

Evidence-based, repeatable pre-flight gate over the CORRECTED episode caches
(comma2k19 + PhysicalAI-AV, D-015 9-channel contract). No model, no training —
only the episode loader + the geom_sanity temporal/ego machinery (reused, one
source of truth). Every verdict is a measurement on the actual `ep_*.pt` shards
the flagship dataloader will read.

Five checks (each -> PASS/FAIL + the number that decides it):

  1 CACHE INTEGRITY   every ep_*.pt loads (torch.load weights_only), size sane
                      (>100 KB), D-015 contract shapes, counts match
                      (comma 410/90, physicalai 400/100), no non-finite proprio.
                      Uses mmap load: torch.load parses the zip central directory
                      (written LAST), so a truncated shard -- the bug that
                      recurred twice -- raises here; a tail-frame touch pages the
                      end of the frames storage as a second truncation probe.
  2 TEMPORAL ALIGN    x-correlation lag between per-step visual motion and pose
                      speed == 0 (no off-by-one); action-sign sanity (steer vs
                      yaw-rate, accel vs d-speed); on turn windows (dyaw over 2 s)
                      sign(dyaw)==sign(ego-y) under the repo `_ego` convention,
                      with a concrete strongest-left-turn example per corpus.
  3 UNITS / SCALE     steer in rad (|p99| < ~1), accel m/s^2, poses metres (speed
                      plausible); no NaN/Inf; comma-vs-physicalai range compare
                      (units mismatch vs mere regime difference).
  4 MIX & SPLIT       realmix physicalai share ~= 0.6 (the real MixedWindowDataset
                      mix_report); I3 train/val episode-id split DISJOINT for both
                      corpora (zero episode-id overlap).
  5 CORRECTED==OLD    for ~5 clips in both old and corrected physicalai caches,
                      actions & poses are byte-identical and only frames differ
                      (the f-theta fix touched geometry, not proprioceptive
                      targets).

Usage (pod-side, from the stack/ dir):
  python3 scripts/validate_data.py \
      --comma-cache  /opt/comma_epcache \
      --pai-cache    /workspace/data/physicalai/_epcache \
      --pai-old-cache /workspace/data/physicalai/_epcache_old \
      --out /workspace/experiments/validate_data.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# tanitad package (stack/) and geom_sanity (scripts/) both importable regardless
# of how the script is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # stack/  -> tanitad
sys.path.insert(0, _HERE)                            # scripts/ -> geom_sanity

import geom_sanity as gs                             # noqa: E402  (reuse machinery)
from tanitad.data.mixing import (MixedWindowDataset,  # noqa: E402
                                 load_episode)
from tanitad.data._contract import EpisodeWindowDataset  # noqa: E402

# ---- pass thresholds (documented, tunable) ---------------------------------
MIN_BYTES = 100_000                 # a shard below this is truncated/empty
EXPECT = {"comma_train": 410, "comma_val": 90,
          "pai_train": 400, "pai_val": 100}
STEER_P99_MAX = 1.0                 # radians (task: |p99| < ~1)
ACCEL_P99_MAX = 6.0                 # m/s^2 (plausible driving longitudinal accel)
SPEED_P99_MAX = 45.0                # m/s (~162 km/h; above => not metres)
MIX_TARGET = 0.6                    # physicalai realmix share (--sim-frac 0.6)
MIX_TOL = 0.02
EGO_H = 20                          # 2 s @ 10 Hz for the sign check (task spec)
TURN_RAD = 0.15                     # deliberate turn over 2 s (refb YAW_TURN_RAD)
STRAIGHT_RAD = math.radians(2.0)    # |dyaw| below this = straight window
SIGN_FRAC_MIN = 0.90                # min sign(dyaw)==sign(ego-y) agreement
MOVE_SPEED = 3.0                    # m/s; below this the GPS-derived heading (yaw)
                                    # is ill-defined -> a "180 deg turn" over 2 s with
                                    # ~0 displacement is a stationary-noise artifact,
                                    # NOT a real turn; exclude such windows.
XCORR_LAGS = range(-5, 6)           # frame<->pose lag search (wide enough to see a
                                    # peak, not just a monotone scene-autocorr drift)
PEAK_GAP_MAX = 0.05                 # frame<->pose xcorr counts as "peakless / lag~0"
                                    # when (max_over_lags - corr_at_lag0) < this; a
                                    # true off-by-one is a SHARP peak (gap >> this).
ACT_POSE_CORR_MIN = 0.30            # min lag-0 corr for steer<->yawrate & accel<->dspeed
SIGN_CORR_MIN = 0.50                # min corr(dyaw, ego_y) on moving turn windows
FWD_POS_FRAC_MIN = 0.95             # min fraction of straight moving windows going fwd
MIX_LEN = 100_000                   # windows drawn for the mix_report measurement
IDENTITY_CLIPS = 5                  # clips compared old-vs-corrected


def _fdir(parent: str, want: str) -> Path:
    """The single train/val subdir under a cache parent (glob on key hash)."""
    hits = sorted(p for p in Path(parent).glob(f"*{want}*") if p.is_dir())
    assert hits, f"no *{want}* dir under {parent}"
    assert len(hits) == 1, f"ambiguous *{want}* under {parent}: {hits}"
    return hits[0]


# --------------------------------------------------------------------------- #
# 1  cache integrity  (scan EVERY shard)                                       #
# --------------------------------------------------------------------------- #
def scan_dir(d: Path) -> dict:
    """Load & validate every ep_*.pt in d. Returns per-dir summary + episodes.

    The returned mmap ToyEpisode list is reused downstream (window build / temporal
    sampling) — mmap keeps them disk-backed so holding all of them is cheap."""
    files = sorted(d.glob("ep_*.pt"))
    corrupt, undersized, bad_shape, nonfinite = [], [], [], []
    eids: list[int] = []
    eps = []
    Ts = []
    for f in files:
        try:
            sz = f.stat().st_size
            if sz < MIN_BYTES:
                undersized.append((f.name, sz))
            # mmap load -> parses the zip central directory (catches truncation)
            ep = load_episode(str(f), mmap=True)
            fr, ac, po = ep.frames, ep.actions, ep.poses
            T = fr.shape[0]
            ok = (fr.ndim == 4 and fr.shape[1] == 9 and fr.dtype == torch.uint8
                  and ac.shape == (T, 2) and po.shape == (T, 4)
                  and ac.dtype == torch.float32 and po.dtype == torch.float32)
            if not ok:
                bad_shape.append((f.name, list(fr.shape), list(ac.shape),
                                  list(po.shape), str(fr.dtype)))
            # tail-frame touch: page the END of the largest storage (2nd
            # truncation probe); proprio finiteness (small, fully materialized).
            _ = float(fr[-1].float().mean())
            if not (torch.isfinite(ac).all() and torch.isfinite(po).all()):
                nonfinite.append(f.name)
            eids.append(int(ep.episode_id))
            eps.append(ep)
            Ts.append(int(T))
        except Exception as e:                       # noqa: BLE001 (report, continue)
            corrupt.append((f.name, f"{type(e).__name__}: {e}"))
    return {
        "dir": str(d), "n_files": len(files), "n_loaded": len(eps),
        "corrupt": corrupt, "undersized": undersized, "bad_shape": bad_shape,
        "nonfinite": nonfinite, "episode_ids": eids,
        "T_min": min(Ts) if Ts else None, "T_max": max(Ts) if Ts else None,
        "size_min_bytes": min((f.stat().st_size for f in files), default=None),
        "size_max_bytes": max((f.stat().st_size for f in files), default=None),
        "_eps": eps,
    }


def check1_integrity(scans: dict) -> dict:
    counts = {k: scans[k]["n_files"] for k in EXPECT}
    counts_ok = counts == EXPECT
    corrupt = {k: scans[k]["corrupt"] for k in scans if scans[k]["corrupt"]}
    undersized = {k: scans[k]["undersized"] for k in scans if scans[k]["undersized"]}
    bad_shape = {k: scans[k]["bad_shape"] for k in scans if scans[k]["bad_shape"]}
    nonfinite = {k: scans[k]["nonfinite"] for k in scans if scans[k]["nonfinite"]}
    n_scanned = sum(scans[k]["n_files"] for k in scans)
    ok = (counts_ok and not corrupt and not undersized and not bad_shape
          and not nonfinite)
    return {
        "status": "PASS" if ok else "FAIL",
        "n_shards_scanned": n_scanned,
        "counts": counts, "counts_expected": EXPECT, "counts_ok": counts_ok,
        "n_corrupt": sum(len(v) for v in corrupt.values()),
        "corrupt": corrupt,
        "n_undersized": sum(len(v) for v in undersized.values()),
        "undersized": undersized,
        "n_bad_shape": sum(len(v) for v in bad_shape.values()),
        "bad_shape": bad_shape,
        "n_nonfinite_proprio_files": sum(len(v) for v in nonfinite.values()),
        "nonfinite": nonfinite,
        "size_bytes": {k: [scans[k]["size_min_bytes"], scans[k]["size_max_bytes"]]
                       for k in scans},
        "T_range": {k: [scans[k]["T_min"], scans[k]["T_max"]] for k in scans},
    }


# --------------------------------------------------------------------------- #
# 2  temporal alignment                                                        #
# --------------------------------------------------------------------------- #
def frame_pose_xcorr(eps, lags=XCORR_LAGS, move_speed=MOVE_SPEED) -> dict:
    """Frame<->pose temporal lag: x-correlate per-step visual motion with pose
    speed on MOVING windows, over a wide lag range.

    Visual motion = mean |luma[t+1]-luma[t]| of the current frame (last 3 of the
    9ch stack). A genuine off-by-one is a SHARP peak at lag +-1; a temporally
    aligned pair (frames & poses share the resampling query timeline by
    construction) gives a flat/peakless curve since frame-diff magnitude is a
    weak, scene-modulated proxy for speed. We therefore report the whole curve
    and the (peak - lag0) gap, not just argmax."""
    corr_by_lag = {L: [] for L in lags}
    for ep in eps:
        luma = gs.cur_luma(ep)
        T = luma.shape[0]
        if T < 15:
            continue
        vis = (luma[1:] - luma[:-1]).abs().mean(dim=(1, 2)).numpy()   # [T-1]
        sp = ep.poses[:, 3].numpy()
        spd = sp[1:]                                                  # speed@t+1
        mov = (sp[:-1] > move_speed) & (sp[1:] > move_speed)         # moving window
        n = len(vis)
        for L in lags:
            if L >= 0:
                a, b, ma, mb = vis[:n - L], spd[L:], mov[:n - L], mov[L:]
            else:
                a, b, ma, mb = vis[-L:], spd[:n + L], mov[-L:], mov[:n + L]
            m = min(len(a), len(b))
            keep = (ma[:m] & mb[:m])
            a, b = a[:m][keep], b[:m][keep]
            if len(a) > 5:
                corr_by_lag[L].append(gs._pearson(a, b))
    curve = {int(L): round(float(np.mean(v)), 4) for L, v in corr_by_lag.items() if v}
    best = max(curve, key=lambda k: curve[k])
    peak = max(curve.values())
    lag0 = curve.get(0, 0.0)
    # An off-by-one is a +-1-step shift: it makes an IMMEDIATE neighbor beat lag 0
    # by a lot. A benign low-frequency scene-autocorrelation drift raises a FAR
    # lag slightly but leaves the +-1 neighbours ~= lag 0. So decide on the
    # neighbour gap; report the global-peak gap for transparency.
    neigh = max(curve.get(-1, lag0), curve.get(1, lag0))
    return {"xcorr_by_lag_moving": curve, "best_lag": best,
            "corr_at_lag0": lag0, "peak_minus_lag0": round(peak - lag0, 4),
            "neighbor_minus_lag0": round(neigh - lag0, 4),
            "peakless_lag0": bool(neigh - lag0 < PEAK_GAP_MAX),
            "move_speed_thr_mps": move_speed}


def ego_sign_check(eps, H=EGO_H, turn_rad=TURN_RAD, move_speed=MOVE_SPEED) -> dict:
    """sign(dyaw over 2 s) == sign(ego-y) on genuinely-MOVING turn windows.

    Reuses gs._ego / gs._wrap (exact repo `_ego`: +x fwd, +y left, CCW-positive
    yaw). Only windows moving (> move_speed) at BOTH ends are classified -- at
    near-zero speed the GPS heading is undefined and yields spurious ~180 deg
    'turns' with ~0 displacement. Adaptively lowers the turn threshold until a
    real left turn is found so a concrete example is always reported."""
    def sweep(thr):
        turn_ok = turn_n = 0
        lefts, rights, straight_fwd, pairs = [], [], [], []
        for ep in eps:
            P = ep.poses.numpy()
            T = P.shape[0]
            for t in range(1, T - H):
                if not (P[t, 3] > move_speed and P[t + H, 3] > move_speed):
                    continue                                    # yaw ill-defined
                dyaw = float(gs._wrap(P[t + H, 2] - P[t, 2]))
                e = gs._ego(P[t + H, :2] - P[t, :2], P[t, 2])   # [fwd, left]
                if abs(dyaw) > thr:
                    turn_n += 1
                    if np.sign(dyaw) == np.sign(e[1]):
                        turn_ok += 1
                    pairs.append((dyaw, float(e[1])))
                    (lefts if dyaw > 0 else rights).append(
                        {"dyaw_rad": round(dyaw, 4), "ego_y_m": round(float(e[1]), 3),
                         "ego_x_fwd_m": round(float(e[0]), 3),
                         "sign_ok": bool(np.sign(dyaw) == np.sign(e[1]))})
                elif abs(dyaw) < STRAIGHT_RAD:
                    straight_fwd.append(float(e[0]))
        return turn_ok, turn_n, lefts, rights, straight_fwd, pairs

    thr = turn_rad
    for _ in range(4):                       # lower thr until a left turn appears
        turn_ok, turn_n, lefts, rights, straight_fwd, pairs = sweep(thr)
        if lefts:
            break
        thr *= 0.5
    corr = 0.0
    if len(pairs) > 5:
        dy = np.array([p[0] for p in pairs]); el = np.array([p[1] for p in pairs])
        corr = gs._pearson(dy, el)
    strongest_left = max(lefts, key=lambda d: d["dyaw_rad"]) if lefts else None
    return {
        "turn_threshold_rad_used": round(thr, 4),
        "horizon_s": H * gs.DT, "move_speed_thr_mps": move_speed,
        "turn_windows": turn_n,
        "frac_sign(dyaw)==sign(ego_y)": round(turn_ok / max(1, turn_n), 4),
        "corr_dyaw_vs_ego_y": round(corr, 4),
        "n_left_turn_windows": len(lefts),
        "frac_left_ego_y_positive": round(
            float(np.mean([l["sign_ok"] for l in lefts])), 4) if lefts else None,
        "strongest_left_turn": strongest_left,
        "n_right_turn_windows": len(rights),
        "straight_windows": len(straight_fwd),
        "frac_forward_positive": round(float(np.mean(np.array(straight_fwd) > 0)), 4)
        if straight_fwd else None,
    }


def check2_temporal(eps_by_corpus: dict) -> dict:
    """Temporal alignment, decided on three fronts:

      (a) actions<->poses  (DECISIVE, sample-exact): steer<->yaw-rate and
          accel<->d-speed lag-0 correlations strongly positive => no off-by-one
          between the proprioceptive/action stream and the pose stream.
      (b) frames<->poses: visual-motion<->speed x-correlation is peakless around
          lag 0 (no sharp competing peak) on moving windows => no off-by-one
          (frames & poses share the resampling timeline by construction; the
          frame-diff proxy is too weak to resolve sub-frame timing but a real
          shift would still show a sharp peak).
      (c) handedness: sign(dyaw over 2 s) == sign(ego-y) on genuinely moving
          turn windows, with a concrete strongest-left-turn example; straight
          moving windows go forward (ego-x > 0)."""
    per = {}
    ok = True
    for corp, eps in eps_by_corpus.items():
        tmp = gs.check_temporal(eps, lags=XCORR_LAGS)     # reused (steer/yaw etc.)
        fp = frame_pose_xcorr(eps)                        # moving-filtered curve
        ego = ego_sign_check(eps)
        # (a) actions<->poses
        act_pose = (tmp["steer_vs_yawrate_corr_mean"] >= ACT_POSE_CORR_MIN
                    and tmp["accel_vs_dspeed_corr_mean"] >= ACT_POSE_CORR_MIN)
        # (b) frames<->poses: no sharp competing peak away from lag 0
        frame_pose = bool(fp["peakless_lag0"])
        # (c) handedness on real (moving) turns
        sl = ego["strongest_left_turn"]
        egook = (ego["frac_sign(dyaw)==sign(ego_y)"] >= SIGN_FRAC_MIN
                 and ego["corr_dyaw_vs_ego_y"] >= SIGN_CORR_MIN
                 and ego["straight_windows"] >= 15
                 and (ego["frac_forward_positive"] or 0) >= FWD_POS_FRAC_MIN
                 and sl is not None and sl["sign_ok"])
        cok = act_pose and frame_pose and egook
        ok = ok and cok
        per[corp] = {"status": "PASS" if cok else "FAIL",
                     "actions_poses_lag0_ok": act_pose,
                     "frames_poses_peakless_ok": frame_pose,
                     "handedness_ok": egook,
                     "actions_poses_xcorr": tmp,
                     "frames_poses_xcorr_moving": fp,
                     "ego_2s": ego}
    return {"status": "PASS" if ok else "FAIL", "by_corpus": per}


# --------------------------------------------------------------------------- #
# 3  units / scale / finiteness                                                #
# --------------------------------------------------------------------------- #
def check3_units(eps_by_corpus: dict, integrity: dict) -> dict:
    per = {}
    ok = True
    for corp, eps in eps_by_corpus.items():
        sc = gs.check_action_scale(eps)
        steer_p99 = sc["abs_steer_rad_pct"]["p99"]
        accel_p99 = max(abs(sc["accel_mps2_pct"]["p99"]), abs(sc["accel_mps2_pct"]["p1"]))
        speed_p99 = sc["speed_mps_pct"]["p99"]
        cok = (steer_p99 < STEER_P99_MAX and accel_p99 < ACCEL_P99_MAX
               and speed_p99 < SPEED_P99_MAX)
        ok = ok and cok
        per[corp] = {"status": "PASS" if cok else "FAIL", "scale": sc,
                     "abs_steer_p99_rad": steer_p99, "abs_accel_p99_mps2": accel_p99,
                     "speed_p99_mps": speed_p99,
                     "steer_in_rad_range": bool(steer_p99 < STEER_P99_MAX),
                     "accel_plausible_mps2": bool(accel_p99 < ACCEL_P99_MAX),
                     "poses_in_metres": bool(speed_p99 < SPEED_P99_MAX)}
    # cross-corpus compare (units mismatch vs regime difference)
    corps = list(per)
    mismatch = None
    if len(corps) == 2:
        a, b = corps
        def ratio(x, y): return round(x / y, 3) if y else None
        both_rad = per[a]["steer_in_rad_range"] and per[b]["steer_in_rad_range"]
        mismatch = {
            "corpora": corps,
            "abs_steer_p99_ratio": ratio(per[a]["abs_steer_p99_rad"],
                                         per[b]["abs_steer_p99_rad"]),
            "abs_accel_p99_ratio": ratio(per[a]["abs_accel_p99_mps2"],
                                         per[b]["abs_accel_p99_mps2"]),
            "speed_p99_ratio": ratio(per[a]["speed_p99_mps"], per[b]["speed_p99_mps"]),
            "both_steer_in_rad_range": bool(both_rad),
            "units_mismatch_flag": bool(not both_rad),
            "note": "large ratios with both corpora in-range = regime difference "
                    "(comma highway vs physicalai urban), NOT a units mismatch",
        }
    nonfinite = integrity["n_nonfinite_proprio_files"]
    ok = ok and nonfinite == 0 and (mismatch is None or not mismatch["units_mismatch_flag"])
    return {"status": "PASS" if ok else "FAIL", "by_corpus": per,
            "cross_corpus": mismatch,
            "n_nonfinite_proprio_files_all_shards": nonfinite}


# --------------------------------------------------------------------------- #
# 4  mix ratio & split disjointness                                            #
# --------------------------------------------------------------------------- #
def check4_mix_split(scans: dict, comma_train_eps, pai_train_eps,
                     window: int = 8, seed: int = 0) -> dict:
    # split disjointness (episode-id sets; a shared clip MUST share episode_id,
    # so zero overlap => clip-disjoint regardless of any id hash collisions).
    def split_report(corp):
        tr = scans[f"{corp}_train"]["episode_ids"]
        va = scans[f"{corp}_val"]["episode_ids"]
        inter = sorted(set(tr) & set(va))
        return {"n_train": len(tr), "n_val": len(va),
                "n_train_distinct_ids": len(set(tr)),
                "n_val_distinct_ids": len(set(va)),
                "n_id_overlap": len(inter), "overlap_ids": inter[:20],
                "disjoint": len(inter) == 0}
    comma_split = split_report("comma")
    pai_split = split_report("pai")
    split_ok = comma_split["disjoint"] and pai_split["disjoint"]

    # realmix physicalai share — the REAL MixedWindowDataset, weights (0.4, 0.6).
    c_ds = EpisodeWindowDataset(comma_train_eps, window=window, max_horizon=4)
    p_ds = EpisodeWindowDataset(pai_train_eps, window=window, max_horizon=4)
    mix = MixedWindowDataset([(c_ds, 1.0 - MIX_TARGET), (p_ds, MIX_TARGET)],
                             length=MIX_LEN, seed=seed)
    rep = mix.mix_report()
    pai_share = rep["domain_1_frac"]                 # domain 1 = physicalai
    mix_ok = abs(pai_share - MIX_TARGET) <= MIX_TOL
    ok = split_ok and mix_ok
    return {
        "status": "PASS" if ok else "FAIL",
        "split_disjoint": {"comma": comma_split, "physicalai": pai_split,
                           "ok": split_ok},
        "mix_ratio": {"target_physicalai_share": MIX_TARGET,
                      "measured_physicalai_share": round(pai_share, 4),
                      "measured_comma_share": round(rep["domain_0_frac"], 4),
                      "n_windows_drawn": MIX_LEN,
                      "comma_train_windows": len(c_ds),
                      "pai_train_windows": len(p_ds),
                      "within_tol": mix_ok, "tol": MIX_TOL},
    }


# --------------------------------------------------------------------------- #
# 5  corrected-vs-old identity                                                 #
# --------------------------------------------------------------------------- #
def _id_to_file(*dirs: Path) -> dict:
    """episode_id -> file, keeping only ids that are UNIQUE across the given dirs
    (unambiguous match; physicalai episode_id uses 4 clip-id bytes so guard)."""
    seen: dict[int, Path] = {}
    dup: set[int] = set()
    for d in dirs:
        for f in sorted(d.glob("ep_*.pt")):
            eid = int(load_episode(str(f), mmap=True).episode_id)
            if eid in seen:
                dup.add(eid)
            seen[eid] = f
    return {k: v for k, v in seen.items() if k not in dup}


def check5_identity(pai_old_cache: str | None, pai_cache: str,
                    n_clips: int = IDENTITY_CLIPS) -> dict:
    if not pai_old_cache or not Path(pai_old_cache).exists():
        return {"status": "SKIP",
                "reason": f"old cache not present at {pai_old_cache}"}
    old = _id_to_file(_fdir(pai_old_cache, "train"), _fdir(pai_old_cache, "val"))
    new = _id_to_file(_fdir(pai_cache, "train"), _fdir(pai_cache, "val"))
    common = sorted(set(old) & set(new))
    if not common:
        return {"status": "FAIL", "reason": "no shared episode_id between old and "
                "corrected caches", "n_old": len(old), "n_new": len(new)}
    sel = common[:n_clips]
    clips = []
    all_ok = True
    dframes = []
    for eid in sel:
        eo = load_episode(str(old[eid]))              # full load (byte compare)
        en = load_episode(str(new[eid]))
        same_shape = (eo.actions.shape == en.actions.shape
                      and eo.poses.shape == en.poses.shape
                      and eo.frames.shape == en.frames.shape)
        act_id = bool(same_shape and torch.equal(eo.actions, en.actions))
        pose_id = bool(same_shape and torch.equal(eo.poses, en.poses))
        frames_diff = bool((not same_shape) or (not torch.equal(eo.frames, en.frames)))
        mad = (float((eo.frames.float() - en.frames.float()).abs().mean())
               if same_shape else None)
        if mad is not None:
            dframes.append(mad)
        cok = act_id and pose_id and frames_diff
        all_ok = all_ok and cok
        clips.append({"episode_id": eid, "old": old[eid].name, "new": new[eid].name,
                      "same_shape": same_shape, "actions_byte_identical": act_id,
                      "poses_byte_identical": pose_id, "frames_differ": frames_diff,
                      "mean_abs_dframe": round(mad, 3) if mad is not None else None,
                      "status": "PASS" if cok else "FAIL"})
    return {"status": "PASS" if all_ok else "FAIL",
            "n_common_ids": len(common), "n_compared": len(sel),
            "mean_abs_dframe_overall": round(float(np.mean(dframes)), 3)
            if dframes else None, "clips": clips}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comma-cache", required=True,
                    help="parent with comma2k19-{train,val}-<key>/")
    ap.add_argument("--pai-cache", required=True,
                    help="parent with physicalai-{train,val}-<key>/ (CORRECTED)")
    ap.add_argument("--pai-old-cache", default=None,
                    help="parent with the OLD physicalai caches (check 5)")
    ap.add_argument("--sample", type=int, default=30,
                    help="val episodes/corpus for temporal+units checks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = {
        "comma_train": _fdir(args.comma_cache, "train"),
        "comma_val": _fdir(args.comma_cache, "val"),
        "pai_train": _fdir(args.pai_cache, "train"),
        "pai_val": _fdir(args.pai_cache, "val"),
    }
    print("[validate] resolved cache dirs:", flush=True)
    for k, v in dirs.items():
        print(f"    {k:12s} {v}", flush=True)

    print("[validate] check 1: scanning every shard (mmap load + contract) ...",
          flush=True)
    scans = {k: scan_dir(v) for k, v in dirs.items()}
    for k in dirs:
        s = scans[k]
        print(f"    {k:12s} files={s['n_files']} loaded={s['n_loaded']} "
              f"corrupt={len(s['corrupt'])} undersized={len(s['undersized'])} "
              f"bad_shape={len(s['bad_shape'])} nonfinite={len(s['nonfinite'])}",
              flush=True)
    integrity = check1_integrity(scans)
    print(f"    -> check1 {integrity['status']}", flush=True)

    # temporal + units sample: first N val episodes per corpus (mmap, held-out).
    eps_by_corpus = {
        "comma2k19": scans["comma_val"]["_eps"][:args.sample],
        "physicalai": scans["pai_val"]["_eps"][:args.sample],
    }
    print(f"[validate] check 2/3: temporal+units on {args.sample} val eps/corpus ...",
          flush=True)
    temporal = check2_temporal(eps_by_corpus)
    for corp, r in temporal["by_corpus"].items():
        x = r["actions_poses_xcorr"]; fp = r["frames_poses_xcorr_moving"]
        e = r["ego_2s"]
        print(f"    {corp:11s} steer/yaw={x['steer_vs_yawrate_corr_mean']} "
              f"accel/dv={x['accel_vs_dspeed_corr_mean']} (act<->pose lag0) | "
              f"frame<->pose peak-lag0={fp['peak_minus_lag0']} "
              f"(best_lag={fp['best_lag']}, peakless={fp['peakless_lag0']}) | "
              f"turn_sign={e['frac_sign(dyaw)==sign(ego_y)']} "
              f"corr_dyaw_egoy={e['corr_dyaw_vs_ego_y']} "
              f"straight_fwd={e['frac_forward_positive']} "
              f"strongest_left={e['strongest_left_turn']} -> {r['status']}",
              flush=True)
    units = check3_units(eps_by_corpus, integrity)
    for corp, r in units["by_corpus"].items():
        print(f"    {corp:11s} |steer|p99={r['abs_steer_p99_rad']}rad "
              f"|accel|p99={r['abs_accel_p99_mps2']}m/s2 "
              f"speed_p99={r['speed_p99_mps']}m/s -> {r['status']}", flush=True)

    print("[validate] check 4: mix ratio + split disjointness ...", flush=True)
    mix_split = check4_mix_split(scans, scans["comma_train"]["_eps"],
                                 scans["pai_train"]["_eps"],
                                 window=args.window, seed=args.seed)
    ms = mix_split
    print(f"    split comma disjoint={ms['split_disjoint']['comma']['disjoint']} "
          f"(overlap={ms['split_disjoint']['comma']['n_id_overlap']}) "
          f"physicalai disjoint={ms['split_disjoint']['physicalai']['disjoint']} "
          f"(overlap={ms['split_disjoint']['physicalai']['n_id_overlap']})",
          flush=True)
    print(f"    physicalai mix share={ms['mix_ratio']['measured_physicalai_share']} "
          f"(target {MIX_TARGET}) -> {ms['status']}", flush=True)

    print("[validate] check 5: corrected-vs-old identity ...", flush=True)
    identity = check5_identity(args.pai_old_cache, args.pai_cache)
    print(f"    -> check5 {identity['status']} "
          f"mean|dframe|={identity.get('mean_abs_dframe_overall')}", flush=True)

    checks = {"1_cache_integrity": integrity, "2_temporal_alignment": temporal,
              "3_units_scale_finite": units, "4_mix_ratio_split": mix_split,
              "5_corrected_vs_old_identity": identity}
    statuses = {k: v["status"] for k, v in checks.items()}
    hard = [k for k, s in statuses.items() if s == "FAIL"]
    skipped = [k for k, s in statuses.items() if s == "SKIP"]
    if hard:
        verdict = "RED"
    elif skipped:
        verdict = "AMBER"
    else:
        verdict = "GREEN"

    report = {"axis": "Axis 1 - Data consistency",
              "gate": "PRE_FLIGHT_VALIDATION.md",
              "verdict": verdict, "statuses": statuses,
              "hard_fails": hard, "skipped": skipped,
              "config": {"seed": args.seed, "window": args.window,
                         "sample": args.sample},
              "cache_dirs": {k: str(v) for k, v in dirs.items()},
              "checks": checks}

    # strip the heavy mmap episode lists before serializing
    for k in scans:
        scans[k].pop("_eps", None)
        scans[k].pop("episode_ids", None)
    report["shard_scan"] = scans

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[validate] verdict = {verdict}", flush=True)
    print(f"[validate] statuses = {json.dumps(statuses)}", flush=True)
    print(f"[validate] report -> {args.out}\nVALIDATE_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
