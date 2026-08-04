#!/usr/bin/env python3
"""EVAL-HOST runner: turn the canonical val40 windows into a populated LONGITUDINAL family.

This is the step the 2026-08-03 distance-keeping package named as its open work item:

    "Wiring it into the *eval* path (val40 windows -> win["lead"]) needs the obstacle.offline
     chunks for the 40 val episodes on the eval host. Until that lands, arm evals will still
     report the family UNAVAILABLE."

Run it on the host that holds `/workspace/val40cache`. 0 GPU in `--pred-npz` mode: it re-scores a
BANKED per-window prediction dump, so no arm is re-run and no training pod is touched.

    python score_val40_lead.py --val-dir /workspace/val40cache \
        --data-root /workspace/physicalai --pred-npz banked/refc-base.npz --out refc_base.json

⛔ WHAT IT WILL REFUSE TO DO
* register an episode it cannot locate on the clip's egomotion track (`RegistrationError`) —
  a mis-registered window puts the lead in the wrong place and still returns a plausible number;
* treat a clip with no `obstacle.offline`, or a window whose horizon leaves the ~20 s labelled
  span, as free flow. Those are NO_LABEL and are reported as their own denominator.

⛔ PARITY. Read-only: it opens `val40cache/ep_*.pt` and the label zips, and writes one JSON. It
never re-selects a clip and never writes into an episode cache, so `physicalai-val-0c5f7dac3b11`
and skip-hash `f09e44db` are untouched. It ASSERTS the val40 window grid reproduces the canonical
881 windows before scoring anything.

PRIVACY: PhysicalAI-AV is gated — the output carries `clip_<sha256[:8]>`, never a clip UUID.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_VAL40_WINDOWS = 881
HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
WP_REL_S = np.array([0.5, 1.0, 1.5, 2.0])        # taniteval.rollout.WP_STEPS at 10 Hz


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def anon(cid: str) -> str:
    return "clip_" + hashlib.sha256(str(cid).encode()).hexdigest()[:8]


def _bootstrap(repo: Path):
    sys.path.insert(0, str(repo / "stack"))
    sys.path.insert(0, str(repo / "taniteval"))
    sys.path.insert(0, str(repo / "stack" / "scripts"))


def clip_of_episode(episode_id: int, ids: list[str]) -> str:
    """Recover the clip UUID from the cached episode's ``episode_id``.

    `physicalai.build_episode` sets ``episode_id = int.from_bytes(clip_id[:4], "big")`` — the first
    four characters of the UUID. That prefix is not globally unique (2376 episodes over 65 536
    4-hex-char prefixes ⇒ ~43 expected collisions corpus-wide), so this REFUSES an ambiguous match
    rather than guessing. MEASURED on the canonical val40: all 40 prefixes resolve uniquely inside
    the 3 000-clip phase0 selection.
    """
    pref = int(episode_id).to_bytes(4, "big").decode("ascii", "replace")
    hit = [c for c in ids if c.startswith(pref)]
    if len(hit) != 1:
        raise KeyError(f"episode_id {episode_id} -> prefix {pref!r} matched {len(hit)} clips; "
                       f"cannot register unambiguously")
    return hit[0]


def ensure_chunk(root: Path, kind: str, chunk: int, download: bool) -> Path | None:
    rel = f"labels/{kind}/{kind}.chunk_{chunk:04d}.zip"
    p = root / rel
    if p.exists():
        return p
    if not download:
        return None
    from tanitad.keys import enable_tls, load_keys
    try:
        enable_tls()
        load_keys()
    except Exception as e:                                   # noqa: BLE001
        log(f"  keys helper unavailable ({e!r}); relying on env HF_TOKEN")
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=str(root)))
    except Exception as e:                                   # noqa: BLE001
        log(f"  could not fetch {rel}: {e!r}")
        return None


def _member(z: zipfile.ZipFile, clip: str) -> str | None:
    for n in z.namelist():
        if n.endswith(".parquet") and n.split("/")[-1].startswith(clip):
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[6]))
    ap.add_argument("--val-dir", default="/workspace/val40cache")
    ap.add_argument("--data-root", default="/workspace/physicalai")
    ap.add_argument("--selection", default=None,
                    help="phase0_selection.parquet; defaults to <data-root>/r0/")
    ap.add_argument("--pred-npz", default=None,
                    help="banked per-window predictions: arrays 'pred' [N,4,2] (+optional "
                         "'pred_dense' [N,20,2]) in the SAME order rollout.collect emits")
    ap.add_argument("--arm", default="GT_vs_CV_floor")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _bootstrap(Path(a.repo))
    from tanitad.data.mixing import load_episode
    from taniteval.lead_metrics import (distance_keeping, distance_keeping_by_speed,
                                        paired_distance_keeping)
    from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD, RegistrationError,
                                       lead_block, register_poses_to_time,
                                       window_last_indices)
    from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw

    root = Path(a.data_root)
    sel_path = Path(a.selection) if a.selection else root / "r0" / "phase0_selection.parquet"
    sel = pd.read_parquet(sel_path)
    sel["clip_id"] = sel["clip_id"].astype(str)
    ids = sel["clip_id"].tolist()
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))

    eps = sorted(Path(a.val_dir).glob("ep_*.pt"))
    if not eps:
        raise SystemExit(f"no ep_*.pt under {a.val_dir}")
    log(f"{len(eps)} val episodes under {a.val_dir}")

    # ---- window grid, asserted against the published canonical count ------- #
    loaded = [load_episode(p, mmap=True) for p in eps]
    T = [int(e.poses.shape[0]) for e in loaded]
    grid = [window_last_indices(t) for t in T]
    n_win = sum(g.size for g in grid)
    log(f"window grid: {n_win} windows over {len(eps)} episodes")
    if len(eps) == 40 and n_win != CANONICAL_VAL40_WINDOWS:
        raise SystemExit(f"REFUSING: 40 val episodes gave {n_win} windows, not the canonical "
                         f"{CANONICAL_VAL40_WINDOWS}. The grid does not match rollout.collect's, "
                         f"so a lead block built from it would be misaligned with every pred row.")

    LEADS, LENS, SPD, ST, EID = [], [], [], [], []
    GT, CV = [], []
    per_ep, refused = [], []
    for p, ep, last in zip(eps, loaded, grid):
        poses = np.asarray(ep.poses, dtype=np.float64)
        try:
            cid = clip_of_episode(int(ep.episode_id), ids)
        except KeyError as e:
            refused.append({"file": p.name, "reason": str(e)})
            continue
        ch = chunk_of[cid]
        ezp = ensure_chunk(root, "egomotion", ch, not a.no_download)
        ozp = ensure_chunk(root, "obstacle.offline", ch, not a.no_download)
        if ezp is None:
            refused.append({"file": p.name, "clip": anon(cid), "chunk": int(ch),
                            "reason": "egomotion chunk unavailable"})
            continue
        with zipfile.ZipFile(ezp) as ez:
            m = _member(ez, cid)
            if m is None:
                refused.append({"file": p.name, "clip": anon(cid), "chunk": int(ch),
                                "reason": "clip not in the egomotion chunk"})
                continue
            ego_df = pd.read_parquet(io.BytesIO(ez.read(m)))
        t = ego_df["timestamp"].to_numpy(np.float64) / 1e6
        o = np.argsort(t)
        g = lambda c: ego_df[c].to_numpy(np.float64)[o]          # noqa: E731
        ego = {"t": t[o], "x": g("x"), "y": g("y"),
               "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
               "v": np.hypot(g("vx"), g("vy"))}
        try:
            reg = register_poses_to_time(poses[:, :2], ego["t"], ego["x"], ego["y"])
        except RegistrationError as e:
            refused.append({"file": p.name, "clip": anon(cid), "chunk": int(ch),
                            "reason": f"RegistrationError: {e}"})
            continue
        t0s = reg["t_s"][last]

        obs = None
        if ozp is not None:
            with zipfile.ZipFile(ozp) as oz:
                mo = _member(oz, cid)
                if mo is not None:
                    df = pd.read_parquet(io.BytesIO(oz.read(mo)))
                    obs = {"t": df["timestamp_us"].to_numpy(np.float64) / 1e6,
                           "track": df["track_id"].astype(str).to_numpy(object),
                           "center_x": df["center_x"].to_numpy(np.float64),
                           "center_y": df["center_y"].to_numpy(np.float64),
                           "size_x": df["size_x"].to_numpy(np.float64),
                           "is_vehicle": df["label_class"].astype(str)
                           .isin(VEHICLE_CLASSES).to_numpy()}
        blk = lead_block(t0s, WP_REL_S, obs, ego)
        LEADS.append(blk["leads"])
        LENS.append(blk["lead_lens"])
        SPD.append(blk["speeds"])
        ST.append(blk["state"])
        EID.extend([anon(cid)] * last.size)

        steps = np.round(WP_REL_S / 0.1).astype(int)
        x, y, yaw, v = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        gt = np.full((last.size, steps.size, 2), np.nan)
        cv = np.full_like(gt, np.nan)
        for i, l in enumerate(last):
            j = np.clip(l + steps, 0, poses.shape[0] - 1)
            c, s = np.cos(yaw[l]), np.sin(yaw[l])
            dx, dy = x[j] - x[l], y[j] - y[l]
            gt[i, :, 0], gt[i, :, 1] = dx * c + dy * s, -dx * s + dy * c
            cv[i, :, 0], cv[i, :, 1] = v[l] * WP_REL_S, 0.0
        GT.append(gt)
        CV.append(cv)
        per_ep.append({"file": p.name, "clip": anon(cid), "chunk": int(ch), "T": int(poses.shape[0]),
                       "n_windows": int(last.size), "has_obstacle": bool(obs is not None),
                       "registration_residual_m": reg["residual_m"]["median"],
                       "grid_dt_s": reg["grid_dt_s"],
                       **{k: int(v) for k, v in blk["counts"].items()}})
        log(f"  {p.name} {anon(cid)} chunk {ch:04d}: {last.size} windows, "
            f"lead {blk['counts'][LEAD]}, no-lead {blk['counts'][NO_LEAD]}, "
            f"no-label {blk['counts'][NO_LABEL]}")

    if not per_ep:
        raise SystemExit("no episode registered; nothing to score")
    LEADS = np.concatenate(LEADS)
    LENS = np.concatenate(LENS)
    SPD = np.concatenate(SPD)
    ST = np.concatenate(ST)
    EID = np.array(EID, dtype=object)
    GT, CV = np.concatenate(GT), np.concatenate(CV)
    dt = float(WP_REL_S[1] - WP_REL_S[0])

    arms = {"GT": GT, "CV": CV}
    if a.pred_npz:
        z = np.load(a.pred_npz, allow_pickle=True)
        pred = np.asarray(z["pred"], dtype=np.float64)
        if pred.shape[0] != GT.shape[0]:
            raise SystemExit(f"REFUSING: banked pred has {pred.shape[0]} rows but this run built "
                             f"{GT.shape[0]} windows. They are not the same window set.")
        arms[a.arm] = pred

    out = {"_what": "LONGITUDINAL distance-keeping on the canonical val40 window grid",
           "_binding": "Sayed 2026-08-02 — per family, never pooled; a missing family is a work item",
           "_estimator": "episode-cluster bootstrap; paired for deltas. NEVER overlapping_holdout_se",
           "n_windows": int(GT.shape[0]), "n_episodes": len(per_ep),
           "canonical_881_asserted": bool(len(eps) == 40),
           "window_states": {k: int((ST == k).sum()) for k in (LEAD, NO_LEAD, NO_LABEL)},
           "episodes": per_ep, "refused": refused, "arms": {}}
    dks = {}
    for name, paths in arms.items():
        dk = distance_keeping(paths, LEADS, LENS, SPD, dt)
        dks[name] = dk
        out["arms"][name] = {
            "pooled": {k: v for k, v in dk.items() if not isinstance(v, np.ndarray)},
            "by_speed": distance_keeping_by_speed(dk, SPD, EID, states=ST, n_boot=a.n_boot),
        }
        log(f"{name}: n={dk['n']}/{dk['n_windows']} headway={dk.get('mean_headway_min_m')} "
            f"ttc={dk.get('mean_min_ttc_s')}")
    out["paired"] = {}
    for name in arms:
        if name == "CV":
            continue
        out["paired"][f"{name}_minus_CV"] = paired_distance_keeping(
            dks[name], dks["CV"], EID, names=(name, "CV"), n_boot=a.n_boot)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
