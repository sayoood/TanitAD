#!/usr/bin/env python3
"""The FOUR-FAMILY panel for `refc-base-30k` and `flagship-v1` on the canonical val40 — with
LONGITUDINAL distance-keeping populated for the first time on real arms.

Builds the lead block ONCE (so every arm is scored on the identical window set and every delta is
paired) and then reports, per family, never pooled into one score:

  LONGITUDINAL  target-speed (speed_mae / bias) AND distance-keeping (headway / time-gap / min-TTC)
  LATERAL       heading, curvature, yaw-rate, cross-track
  TACTICAL      manoeuvre decision + goal setting  -> availability decided from the dump's own keys
  STRATEGIC     route/goal                          -> ditto

⛔ Estimator: paired episode-cluster bootstrap (`taniteval.ci`). NEVER `overlapping_holdout_se`.
⛔ 0 GPU. Reads banked per-window dumps + the poses-only val40 view. No training pod touched.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

from tanitad.data.mixing import load_episode                                       # noqa: E402
from taniteval import ci as _ci                                                    # noqa: E402
from taniteval import four_families as ff                                          # noqa: E402
from taniteval.lead_metrics import (distance_keeping, distance_keeping_by_speed,   # noqa: E402
                                    paired_distance_keeping)
from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD, lead_block,            # noqa: E402
                                   register_poses_to_time, window_last_indices)
from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw                        # noqa: E402

VIEW = Path(sys.argv[1])
DATA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
N_BOOT = 2000
WP_REL_S = np.array([0.5, 1.0, 1.5, 2.0])
DT = 0.5                       # the SPARSE 4-waypoint cadence of these dumps
ARMS = ["refc-base-30k", "flagship-30k"]
FLOOR = "cv"                   # the canonical constant_velocity floor, carried in every dump

# ---------------------------------------------------------------- lead block --
sel = pd.read_parquet(DATA / "r0" / "phase0_selection.parquet")
sel["clip_id"] = sel["clip_id"].astype(str)
ids = sel["clip_id"].tolist()
chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))


def _member(z, clip):
    for n in z.namelist():
        if n.endswith(".parquet") and n.split("/")[-1].startswith(clip):
            return n
    return None


eps = sorted(VIEW.glob("ep_*.pt"))
LEADS, LENS, SPD, ST, EID, GAP0 = [], [], [], [], [], []
per_ep = []
for p in eps:
    ep = load_episode(p, mmap=False)
    poses = np.asarray(ep.poses, dtype=np.float64)
    last = window_last_indices(int(poses.shape[0]))
    pref = int(ep.episode_id).to_bytes(4, "big").decode("ascii", "replace")
    hit = [c for c in ids if c.startswith(pref)]
    assert len(hit) == 1, (p.name, len(hit))
    cid, ch = hit[0], chunk_of[hit[0]]
    with zipfile.ZipFile(DATA / f"labels/egomotion/egomotion.chunk_{ch:04d}.zip") as ez:
        ego_df = pd.read_parquet(io.BytesIO(ez.read(_member(ez, cid))))
    t = ego_df["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: ego_df[c].to_numpy(np.float64)[o]                       # noqa: E731
    ego = {"t": t[o], "x": g("x"), "y": g("y"),
           "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
           "v": np.hypot(g("vx"), g("vy"))}
    reg = register_poses_to_time(poses[:, :2], ego["t"], ego["x"], ego["y"])
    obs = None
    ozp = DATA / f"labels/obstacle.offline/obstacle.offline.chunk_{ch:04d}.zip"
    if ozp.exists():
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
    blk = lead_block(reg["t_s"][last], WP_REL_S, obs, ego)
    LEADS.append(blk["leads"]); LENS.append(blk["lead_lens"]); SPD.append(blk["speeds"])
    ST.append(blk["state"]); GAP0.append(blk["gap0_m"]); EID.extend([p.stem] * last.size)
    per_ep.append({"file": p.name, "n_windows": int(last.size),
                   "has_obstacle": obs is not None,
                   "registration_residual_m": reg["residual_m"]["median"],
                   "grid_dt_s": reg["grid_dt_s"],
                   **{k: int(v) for k, v in blk["counts"].items()}})

LEADS = np.concatenate(LEADS); LENS = np.concatenate(LENS); SPD = np.concatenate(SPD)
ST = np.concatenate(ST); GAP0 = np.concatenate(GAP0); EID = np.array(EID, dtype=object)
W = LEADS.shape[0]
print(f"lead block: {W} windows, states "
      f"{ {k: int((ST == k).sum()) for k in (LEAD, NO_LEAD, NO_LABEL)} }", flush=True)

# ---------------------------------------------------------------- arm tensors --
dumps = {a: torch.load(REPO / "taniteval" / "results" / f"windows_{a}.pt",
                       map_location="cpu", weights_only=False) for a in ARMS}
gt = dumps[ARMS[0]]["gt"].double()
for a in ARMS[1:]:
    assert torch.equal(dumps[a]["gt"], dumps[ARMS[0]]["gt"]), f"{a} gt differs — not paired"
paths = {a: dumps[a]["pred"].double() for a in ARMS}
paths["cv"] = dumps[ARMS[0]]["cv"].double()
paths["gt_oracle"] = gt.clone()

lead_kw = dict(leads=LEADS, lead_lens=LENS, speeds=SPD, state=ST, eid=EID, n_boot=N_BOOT)

out = {
    "_what": "FOUR-FAMILY panel, canonical val40 (881 windows / 40 episodes), with LONGITUDINAL "
             "distance-keeping populated on real arms for the first time",
    "_binding": "Sayed 2026-08-02 — per family, never pooled; a missing family is a work item",
    "_estimator": "paired episode-cluster bootstrap (taniteval.ci). NEVER overlapping_holdout_se",
    "_cadence": {"dt_s": DT, "wp_rel_s": WP_REL_S.tolist(),
                 "note": "SPARSE 4-waypoint view. speed ~ 1/dt, accel ~ 1/dt^2, yaw_rate ~ 1/dt; "
                         "along_*, heading, curvature, cross_* are dt-invariant. NOT comparable "
                         "to a dense 10 Hz run of the same module."},
    "_floor": {"cv": "baseline_waypoints()['constant_velocity'] as banked by rollout.collect — "
                     "the floor every driving_*.json headline uses"},
    "n_windows": W, "n_episodes": len(eps),
    "window_states": {k: int((ST == k).sum()) for k in (LEAD, NO_LEAD, NO_LABEL)},
    "episodes": per_ep,
    "families": {}, "paired": {},
}

ff.DT_S = DT
FAM = out["families"]
for name, P in paths.items():
    lo = ff.longitudinal(P, gt, dt=DT, lead=dict(lead_kw))
    la = ff.lateral(P, gt, dt=DT)
    FAM[name] = {"LONGITUDINAL": lo, "LATERAL": la}
    dk = lo["distance_keeping"]
    print(f"{name}: speed_mae={lo['speed_mae_mps']} | headway={dk.get('mean_headway_min_m')} "
          f"ttc={dk.get('mean_min_ttc_s')} n={dk.get('n')}", flush=True)

# ---------------------------------------------------------------- paired deltas --
per_win = {}
for name, P in paths.items():
    dk = distance_keeping(P.numpy(), LEADS, LENS, SPD, DT)
    per_win[name] = dk


def _fam_per_window(P):
    """Per-window forms of the family scalars, verified against the module before use."""
    Pg, Gg = ff._seq_geometry(P, DT), ff._seq_geometry(gt, DT)
    both, both_pair = Pg["valid"] & Gg["valid"], Pg["pair_valid"] & Gg["pair_valid"]
    deg = 180.0 / np.pi

    def _rm(x, m):
        n = m.sum(1)
        s = torch.where(m, x, torch.zeros_like(x)).sum(1)
        return torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float("nan")))

    dh = Pg["heading"] - Gg["heading"]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ct = Pg["cross"] - Gg["cross"]
    K = Pg["speed"].shape[1]
    full = torch.full((Pg["speed"].shape[0],), float(K))
    vals = {
        "speed_mae_mps": ((Pg["speed"] - Gg["speed"]).abs().mean(1), full),
        "speed_bias_mps": ((Pg["speed"] - Gg["speed"]).mean(1), full),
        "along_mae_m": ((Pg["along"] - Gg["along"]).abs().mean(1), full),
        "heading_mae_deg": (_rm(dh.abs() * deg, both), both.sum(1).double()),
        "yaw_rate_mae_degps": (_rm((Pg["yaw_rate"] - Gg["yaw_rate"]).abs() * deg, both_pair),
                               both_pair.sum(1).double()),
        "curvature_mae_1pm": (_rm((Pg["curvature"] - Gg["curvature"]).abs(), both_pair),
                              both_pair.sum(1).double()),
        "cross_mae_m": (ct.abs().mean(1), full),
    }
    return {k: v for k, (v, _) in vals.items()}, {k: n for k, (_, n) in vals.items()}


_pwn = {n: _fam_per_window(P) for n, P in paths.items()}
PW = {n: v[0] for n, v in _pwn.items()}
NW = {n: v[1] for n, v in _pwn.items()}
# ⛔ the per-window reimplementation is admissible ONLY where it reproduces the module. The module
# aggregates ELEMENT-weighted (`x[mask].mean()`); the per-window form is a row-mean. They differ
# whenever rows carry different valid counts, so the check reconstructs the element-weighted
# aggregate exactly rather than comparing a mean-of-row-means (which is a DIFFERENT statistic).
agree = {}
for n in paths:
    for k, v in PW[n].items():
        mod = (FAM[n]["LONGITUDINAL"] if k in ("speed_mae_mps", "speed_bias_mps", "along_mae_m")
               else FAM[n]["LATERAL"]).get(k)
        if mod is None:
            continue
        x, cnt = v.numpy(), NW[n][k].numpy()
        ok = np.isfinite(x) & (cnt > 0)
        elemw = float((x[ok] * cnt[ok]).sum() / cnt[ok].sum())
        d = abs(elemw - float(mod))
        agree[f"{n}.{k}"] = round(d, 8)
        assert d < 1e-3, f"per-window {n}.{k} disagrees with the module by {d}"
out["_per_window_agreement_max"] = max(agree.values())
out["_per_window_weighting_note"] = (
    "the module's scalars are ELEMENT-weighted; the bootstrapped per-window series is "
    "ROW-weighted (each window one observation, which is what an episode-cluster bootstrap "
    "needs). Verified identical after re-weighting — max |diff| in _per_window_agreement_max. "
    "A paired delta is unaffected: both arms use the same weighting.")

PAIRS = [("refc-base-30k", "cv"), ("flagship-30k", "cv"),
         ("refc-base-30k", "flagship-30k"), ("gt_oracle", "cv"),
         ("gt_oracle", "refc-base-30k"), ("gt_oracle", "flagship-30k")]
for a, b in PAIRS:
    key = f"{a}_minus_{b}"
    blk = {"distance_keeping": paired_distance_keeping(
        per_win[a], per_win[b], EID, names=(a, b), n_boot=N_BOOT)["metrics"]}
    fam = {}
    for k in PW[a]:
        x, y = PW[a][k].numpy(), PW[b][k].numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        fam[k] = {"n_used": int(ok.sum()),
                  **_ci.paired_episode_cluster_bootstrap(
                      x[ok], y[ok], list(EID[ok]), n_boot=N_BOOT, seed=0)}
    blk["families"] = fam
    out["paired"][key] = blk
    print(f"paired {key}: headway {blk['distance_keeping']['headway_min_m'].get('mean')} "
          f"ttc {blk['distance_keeping']['min_ttc_s'].get('mean')}", flush=True)

# ---------------------------------------------------------------- stratified --
out["by_speed"] = {n: distance_keeping_by_speed(per_win[n], SPD, EID, states=ST, n_boot=N_BOOT)
                   for n in paths}

# ---------------------------------------------------------------- T / S families --
probe = dumps[ARMS[0]]
out["families_unavailable"] = {}
for lvl, keys in (("TACTICAL", ("maneuver_pred", "maneuver_gt")),
                  ("STRATEGIC", ("route_pred", "route_gt"))):
    fam = ff.tactical(probe) if lvl == "TACTICAL" else ff.strategic(probe)
    out["families_unavailable"][lvl] = {
        "status": fam.get("status"), "n": fam.get("n"),
        "reason": fam.get("reason"),
        "dump_keys_present": sorted(k for k in probe if not k.startswith("_")),
        "required_keys": list(keys),
    }
    for a in ARMS:
        FAM[a][lvl] = fam

Path(OUT).write_text(json.dumps(out, indent=1, default=str))
print("wrote", OUT)
