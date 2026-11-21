#!/usr/bin/env python3
"""Build the ``win["lead"]`` block for an episode corpus from staged `obstacle.offline`.

⛔ WHY. `four_families.longitudinal` reports its **distance-keeping half UNAVAILABLE** without a
lead track, and the binding rule (Sayed 2026-08-02) says a missing family is a **WORK ITEM, not a
pass**. Distance-keeping is not a nicety here: **88.7 % of the oracle gap is longitudinal**, and
headway / time-gap / TTC are the only metrics in the block that can see whether the arm keeps a
safe distance rather than merely tracing the right shape.

This is the INGEST half. :mod:`taniteval.lead_source` is the pure join (selection, frame chain,
span guard) and :mod:`taniteval.lead_metrics` is the pure metric; this file only turns parquet into
the arrays they take. It is a sibling of the dev-box
`Architecture & Inference/…/2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`, which
hardcodes a Windows data root and so cannot run on a pod.

⛔ THE THREE THINGS THAT MAKE OR BREAK IT, each already a measured defect elsewhere:

1. **Window grid identity.** The lead rows must line up with the arm's ``pred`` rows one-for-one.
   ``lead_source.window_last_indices`` reproduces ``rollout.collect``'s grid exactly, so the block
   attaches to an ALREADY-SCORED dump with no re-inference. Recomputing the grid by hand here would
   be a second implementation that can drift.
2. **Registration, not an assumed 0.1 s tick.** The episode grid is an affine reparametrisation of
   the clip clock with spacing ~0.1007 s, not 0.1 — assuming 0.1 drifts ~0.13 s over 200 steps,
   about 1.8 m of lead displacement at 13.6 m/s.
3. **NO_LABEL is never free flow.** A clip with no `obstacle.offline`, or a window whose horizon
   leaves the labelled span, is counted as NO_LABEL and excluded — never as an empty road, which
   would manufacture a safe-looking headway out of missing data.

⚠️ Coverage is a MEASURED denominator, reported per state, never absorbed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import torch


def _p(*a):
    print(*a, flush=True)


def _quat_yaw(qx, qy, qz, qw):
    """Yaw from a quaternion. Imported from the stack when available so there is ONE
    implementation; the inline form is the documented fallback for a pod whose
    `stack/scripts` is not on the path."""
    try:
        from lead_state_gate import quaternion_yaw
        return quaternion_yaw(qx, qy, qz, qw)
    except Exception:
        return np.arctan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))


def _member(z: zipfile.ZipFile, clip: str):
    for n in z.namelist():
        if n.endswith(".parquet") and n.rsplit("/", 1)[-1].startswith(clip):
            return n
    return None


def _read(zpath: str, clip: str):
    if not os.path.exists(zpath):
        return None
    with zipfile.ZipFile(zpath) as z:
        n = _member(z, clip)
        if n is None:
            return None
        import io
        return pd.read_parquet(io.BytesIO(z.read(n)))


def ego_series(df: pd.DataFrame) -> dict:
    t = df["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: df[c].to_numpy(np.float64)[o]                       # noqa: E731
    yaw = np.unwrap(_quat_yaw(g("qx"), g("qy"), g("qz"), g("qw")))
    return {"t": t[o], "x": g("x"), "y": g("y"), "yaw": yaw,
            "v": np.hypot(g("vx"), g("vy"))}


def obs_series(df: pd.DataFrame | None, vehicle_classes) -> dict | None:
    """⚠️ `obstacle.offline` names its clock ``timestamp_us``, `egomotion` names
    its ``timestamp``. They are NOT the same column name and assuming one for the
    other raises here rather than silently mis-joining — which is the good case.

    ⭐ It also ships ``reference_frame_timestamp_us``: the time of the ego frame
    each cuboid is expressed in. The rig->world->t0 chain assumes that equals the
    row's own timestamp; with the column present that is CHECKED, not assumed,
    and the max disagreement is returned so a violation cannot pass silently."""
    if df is None or df.empty:
        return None
    tcol = "timestamp_us" if "timestamp_us" in df.columns else "timestamp"
    t = df[tcol].to_numpy(np.float64) / 1e6
    frame_skew_s = 0.0
    if "reference_frame_timestamp_us" in df.columns:
        rt = df["reference_frame_timestamp_us"].to_numpy(np.float64) / 1e6
        frame_skew_s = float(np.nanmax(np.abs(rt - t))) if rt.size else 0.0
    o = np.argsort(t)
    g = lambda c: df[c].to_numpy()[o]                                 # noqa: E731
    cls = g("label_class").astype(str)
    return {"t": t[o], "track": g("track_id").astype(str),
            "center_x": g("center_x").astype(np.float64),
            "center_y": g("center_y").astype(np.float64),
            "size_x": g("size_x").astype(np.float64),
            "is_vehicle": np.isin(cls, np.asarray(vehicle_classes)),
            "frame_skew_s": frame_skew_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--order", required=True,
                    help="TSV whose row i names the clip in ep_{i:05d}.pt. ⛔ the "
                         "episode<->clip identity; a wrong order silently puts "
                         "another clip's traffic on every window.")
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense-k", type=int, default=20,
                    help="horizon steps; must match the arm's dense grid")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=0)
    a = ap.parse_args()
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")

    from taniteval import lead_source as ls
    from tanitad.data.mixing import load_episode

    # ⛔ Pick the clip column BY SHAPE, not by position. The order file here has
    # three columns (index, clip_id, chunk) and a positional `[-1]` silently
    # returns the CHUNK NUMBER — which then matches no clip at all and the whole
    # corpus reports NO_LABEL. Caught only because the coverage line prints the
    # miss count; a build that just wrote zeros would have looked like a corpus
    # with no traffic in it.
    def clip_of(line: str) -> str:
        for c in line.rstrip("\n").split("\t"):
            c = c.strip()
            if len(c) == 36 and c.count("-") == 4:      # UUID shape
                return c
        raise ValueError(f"no clip-id (UUID) column in order row: {line[:120]!r}")

    order = [clip_of(l) for l in open(a.order)
             if l.strip() and not l.startswith("#")]
    n_eps = a.episodes or len(order)
    _p(f"[order] {len(order)} clips, scoring {n_eps}")

    # clip -> containing zip, built ONCE. Scanning every zip per episode would be
    # 221 x 290 zip opens; it is also how a clip present in two chunks would get
    # picked inconsistently between the ego and obs passes.
    def index(d):
        idx = {}
        for f in sorted(os.listdir(d)):
            if not f.endswith(".zip"):
                continue
            p = os.path.join(d, f)
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith(".parquet"):
                        idx.setdefault(n.rsplit("/", 1)[-1].split(".")[0], p)
        return idx

    ego_idx, obs_idx = index(a.ego_dir), index(a.obs_dir)
    _p(f"[index] egomotion {len(ego_idx)} clips · obstacle.offline {len(obs_idx)} clips")
    missing_obs = [c for c in order[:n_eps] if c not in obs_idx]
    _p(f"[index] clips WITHOUT obstacle.offline: {len(missing_obs)} of {n_eps} "
       f"({100.0*len(missing_obs)/max(n_eps,1):.2f} %) — these become NO_LABEL, "
       f"never free flow")

    ts_rel = np.arange(1, a.dense_k + 1) * a.dt
    LEADS, LENS, SPD, STATE, EID, GAP0 = [], [], [], [], [], []
    n_reg_fail, n_no_ego, n_no_obs, spans = 0, 0, 0, []
    max_skew = 0.0
    n_ep_emitted = 0

    def _bank(blk: dict, i: int, n_win: int):
        nonlocal n_ep_emitted
        if blk["leads"].shape[0] != n_win:
            raise AssertionError(
                f"ep{i:05d}: lead block has {blk['leads'].shape[0]} rows for "
                f"{n_win} windows — the positional join would silently shift")
        LEADS.append(blk["leads"]); LENS.append(blk["lead_lens"])
        SPD.append(blk["speeds"]); STATE.append(blk["state"])
        GAP0.append(blk["gap0_m"]); EID.extend([i] * n_win)
        if blk["label_span_s"]:
            spans.append(blk["label_span_s"])
        n_ep_emitted += 1

    def blank(n_win: int) -> dict:
        """⛔ A dropped episode still emits its ROWS, as NO_LABEL.

        The block is joined POSITIONALLY to an already-scored ``pred`` dump, so
        skipping an episode would shift every later row onto another episode's
        windows — the arm's trajectory scored against a different clip's traffic,
        and the metric would still return a perfectly plausible number. Same
        class as the registration failure this guards: a silent misalignment is
        worse than a loud gap."""
        return {"leads": np.full((n_win, ts_rel.size, 2), np.nan),
                "lead_lens": np.full(n_win, np.nan),
                "speeds": np.full(n_win, np.nan),
                "state": np.array([ls.NO_LABEL] * n_win, dtype=object),
                "gap0_m": np.full(n_win, np.nan), "label_span_s": None}

    for i in range(n_eps):
        clip = order[i]
        ep = load_episode(os.path.join(a.corpus, f"ep_{i:05d}.pt"), mmap=True)
        poses = ep.poses.float().numpy()
        last = ls.window_last_indices(poses.shape[0])
        n_win = len(last)
        ego_df = _read(ego_idx[clip], clip) if clip in ego_idx else None
        if ego_df is None:
            n_no_ego += 1
            _bank(blank(n_win), i, n_win)
            continue
        ego = ego_series(ego_df)
        try:
            reg = ls.register_poses_to_time(poses[:, :2], ego["t"], ego["x"], ego["y"])
        except ls.RegistrationError as e:
            # ⛔ LOUD, and the episode's rows become NO_LABEL rather than being
            # scored on a guessed clock. A mis-registered window puts the lead in
            # the wrong place and the metric still returns a plausible number.
            n_reg_fail += 1
            _p(f"  [reg-fail] ep{i:05d} {clip[:8]}: {e}")
            _bank(blank(n_win), i, n_win)
            continue
        t0s = np.asarray(reg["t_s"])[last]

        obs_df = _read(obs_idx[clip], clip) if clip in obs_idx else None
        obs = obs_series(obs_df, ls.VEHICLE_CLASSES)
        if obs is None:
            n_no_obs += 1
        else:
            max_skew = max(max_skew, obs.pop("frame_skew_s", 0.0))
        _bank(ls.lead_block(t0s, ts_rel, obs, ego), i, n_win)
        if (i + 1) % 50 == 0:
            _p(f"  {i+1}/{n_eps} episodes")

    state = np.concatenate(STATE)
    out = {
        "leads": np.concatenate(LEADS), "lead_lens": np.concatenate(LENS),
        "speeds": np.concatenate(SPD), "state": state,
        "gap0_m": np.concatenate(GAP0), "eid": EID,
        "has_lead": state == ls.LEAD, "ts_rel_s": ts_rel,
        "counts": {s: int((state == s).sum())
                   for s in (ls.LEAD, ls.NO_LEAD, ls.NO_LABEL)},
        "episodes_scored": n_ep_emitted, "episodes_requested": n_eps,
        "episodes_dropped_no_egomotion": n_no_ego,
        "episodes_dropped_registration": n_reg_fail,
        "episodes_without_obstacle_offline": n_no_obs,
        "max_reference_frame_skew_s": max_skew,
        "_frame_skew_check": (
            "obstacle.offline cuboids are expressed in the ego frame at their own\n"
            "            timestamp; the rig->world->t0 chain depends on that. This is the\n"
            "            MEASURED max |reference_frame_timestamp_us - timestamp_us|, so the\n"
            "            assumption is checked rather than asserted in prose."),
        "_denominator_note": (
            "NO_LABEL is NOT free flow. A clip with no obstacle.offline, or a "
            "window whose horizon leaves the labelled span, is excluded — "
            "counting it as an empty road would manufacture a safe headway out "
            "of missing data. Every drop above is reported, never absorbed."),
        "_grid": ("window origins from lead_source.window_last_indices, which "
                  "reproduces rollout.collect's grid exactly, so this block "
                  "attaches row-for-row to an already-scored dump."),
        "_registration": ("affine pose-index -> clip-time fit (Theil-Sen + inlier "
                          "refit). The realised episode spacing is ~0.1007 s, not "
                          "0.1; assuming 0.1 drifts ~0.13 s over 200 steps, about "
                          "1.8 m of lead displacement at 13.6 m/s."),
    }
    torch.save(out, a.out)
    tot = len(state)
    _p(f"\n[out] {a.out}")
    _p(f"  windows {tot}  episodes {n_ep_emitted}/{n_eps} (every episode emits its rows; drops become NO_LABEL)")
    for k, v in out["counts"].items():
        _p(f"  {k:9s} {v:>6}  ({100.0*v/max(tot,1):.2f} %)")
    _p(f"  dropped: no_egomotion {n_no_ego}  registration {n_reg_fail}  "
       f"no_obstacle_offline {n_no_obs}")
    json.dump({k: v for k, v in out.items()
               if isinstance(v, (int, str, dict))},
              open(a.out + ".report.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
