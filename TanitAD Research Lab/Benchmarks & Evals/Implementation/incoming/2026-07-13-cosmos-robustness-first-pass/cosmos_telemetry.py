"""Cosmos-Drive-Dreams annotation tars -> ScenarioTelemetry (Benchmarks & Eval backlog #3).

DATA-ONLY first pass -- no simulator, no model, dev-box/4060 only. Derives the occlusion /
telemetry contract the custom metric suite (LAL / TMS / OKRI / CNCE / LOPS,
``stack/tanitad/eval/metrics.py``) consumes, from the per-clip RDS-HQ annotations that
Cosmos-Drive-Dreams ships. These are all small per-clip tars -- **no 43 GB video shard is
needed** because the metric suite is pixel-free (it operates on telemetry, not RGB):

  vehicle_pose/<clip>.tar      ~297 per-frame 4x4 ego-to-world (float32)     -> ego kinematics
  all_object_info/<clip>.tar   ~297 per-frame JSON {oid: object_to_world 4x4,
                               object_lwh, object_is_moving, object_type}     -> 3D object tracks
  pinhole_intrinsic/<clip>.tar per-camera [fx,fy,cx,cy,W,H]                   -> FoV (front_wide=120)

Ego kinematics REUSE the integrated loader
``stack/tanitad/data/cosmos_drive.poses_to_signals`` (identical steer/accel/v derivation as the
training contract). Occlusion geometry is a bird's-eye ray test against the 3D boxes: an object
is occluded when a closer box subtends its bearing (documented approximation, see ``occluded_by``).

HONEST SCOPE (P8). These numbers characterize the **logged synthetic ego trajectory + scene
occlusion geometry** of the corpus. They are NOT a TanitAD model claim:
  * OKRI / TMS / LAL(-v1/-v2) are computed on the *logged* ego -> corpus/baseline characterization.
  * LOPS needs a world-model latent estimate ``wm_hazard_xy`` -> **0.0 data-only** baseline
    (the E2E-no-latent-track case). ``oracle_lops`` (perfect-perception wm = gt + noise) is included
    ONLY to validate the LOPS path end-to-end (returns ~E[exp(-gamma*noise)] ~ 0.83); not a claim.
  * CNCE needs inference latency + active-param count -> filled with a LABELLED TanitAD-4B stub
    (a compute-normalization constant, not measured here).

This is glue for the "available NOW, no simulator" first pass (STATE backlog #3). It does NOT touch
the running metric contract; it only *consumes* it. Proposed target if triaged: ``stack/scripts/``.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Locate the stack (so tanitad.* imports resolve when run from the intake dir) #
# --------------------------------------------------------------------------- #
def _ensure_stack_on_path() -> None:
    import os
    env = os.environ.get("TANITAD_STACK")
    cands = [Path(env)] if env else []
    here = Path(__file__).resolve()
    cands += [p / "stack" for p in here.parents]      # walk up looking for a stack/ dir
    for c in cands:
        if (c / "tanitad" / "eval" / "metrics.py").is_file():
            sys.path.insert(0, str(c))
            return
    raise RuntimeError("could not locate the TanitAD stack; set TANITAD_STACK=<repo>/stack")


_ensure_stack_on_path()
from tanitad.data.cosmos_drive import poses_to_signals          # noqa: E402
from tanitad.eval.metrics import (ScenarioTelemetry,            # noqa: E402
                                  compute_lops, run_scenario_suite)

TARGET_HZ = 10.0
SRC_FPS = 30.0
FRONT_HFOV_DEG = 120.0      # Cosmos front_wide_120fov


# --------------------------------------------------------------------------- #
# Tar IO                                                                       #
# --------------------------------------------------------------------------- #
def load_pose_tar(p: Path) -> np.ndarray:
    """vehicle_pose tar -> [N,4,4] ego-to-world, frame-ordered."""
    frames: dict[int, np.ndarray] = {}
    with tarfile.open(p) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".vehicle_pose.npy"):
                idx = int(m.name.split(".")[-3])
                frames[idx] = np.load(io.BytesIO(tf.extractfile(m).read())).reshape(4, 4)
    return np.stack([frames[i] for i in sorted(frames)]).astype(np.float64)


def load_objects_tar(p: Path) -> list[dict]:
    """all_object_info tar -> per-frame list of {oid: {c[3], lwh[3], moving, type}}."""
    per_frame: dict[int, dict] = {}
    with tarfile.open(p) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".all_object_info.json"):
                idx = int(m.name.split(".")[-3])
                raw = json.loads(tf.extractfile(m).read())
                objs = {}
                for oid, rec in raw.items():
                    o2w = np.asarray(rec["object_to_world"], dtype=float)
                    objs[oid] = {"c": o2w[:3, 3],
                                 "lwh": np.asarray(rec["object_lwh"], dtype=float),
                                 "moving": bool(rec.get("object_is_moving", False)),
                                 "type": rec.get("object_type", "")}
                per_frame[idx] = objs
    return [per_frame[i] for i in sorted(per_frame)]


def _ego_frame_xy(ego_to_world: np.ndarray, c_world: np.ndarray) -> tuple[float, float]:
    """World point -> ego-frame (x_forward, y_left) via inv(ego_to_world). FLU convention."""
    R = ego_to_world[:3, :3]
    t = ego_to_world[:3, 3]
    p = R.T @ (c_world - t)
    return float(p[0]), float(p[1])


# --------------------------------------------------------------------------- #
# Derivation                                                                   #
# --------------------------------------------------------------------------- #
def derive_telemetry(clip: str, root: str | Path, stride: int | None = None,
                     hfov_deg: float = FRONT_HFOV_DEG) -> dict:
    """One Cosmos clip -> {tel: ScenarioTelemetry, prov, gt_xy, is_occl}."""
    root = Path(root)
    ego4_full = load_pose_tar(root / "vehicle_pose" / f"{clip}.tar")
    objs_full = load_objects_tar(root / "all_object_info" / f"{clip}.tar")
    n = min(len(ego4_full), len(objs_full))
    ego4_full, objs_full = ego4_full[:n], objs_full[:n]

    if stride is None:
        stride = max(1, int(round(SRC_FPS / TARGET_HZ)))          # 3 -> 10 Hz
    dt = stride / SRC_FPS
    ego4 = ego4_full[::stride]
    objs = objs_full[::stride]
    T = len(ego4)

    # ego kinematics (reuse the integrated loader's derivation)
    actions, poses = poses_to_signals(ego4, dt)
    steer, accel = actions[:, 0].astype(float), actions[:, 1].astype(float)
    ego_v = poses[:, 3].astype(float)
    ego_jerk = np.gradient(accel, dt)
    steer_rate = np.abs(np.gradient(steer, dt))

    half_fov = np.radians(hfov_deg) / 2.0

    # per-frame ego-frame geometry for every object id
    oids = sorted({oid for fr in objs for oid in fr})
    geo = {oid: {"present": np.zeros(T, bool), "r": np.full(T, np.inf),
                 "th": np.zeros(T), "x": np.full(T, np.nan), "y": np.full(T, np.nan),
                 "foot": np.zeros(T), "moving": False, "type": ""} for oid in oids}
    for ti in range(T):
        for oid, rec in objs[ti].items():
            x, y = _ego_frame_xy(ego4[ti], rec["c"])
            g = geo[oid]
            g["present"][ti] = True
            g["r"][ti] = float(np.hypot(x, y))
            g["th"][ti] = float(np.arctan2(y, x))
            g["x"][ti], g["y"][ti] = x, y
            g["foot"][ti] = 0.5 * float(np.hypot(rec["lwh"][0], rec["lwh"][1]))
            g["moving"] = g["moving"] or rec["moving"]
            g["type"] = rec["type"]

    def occluded_by(oid: str, ti: int) -> float:
        """Range of the nearest box occluding oid at frame ti (bird's-eye), else inf."""
        g = geo[oid]
        if not g["present"][ti] or not np.isfinite(g["r"][ti]):
            return np.inf
        r_h, th_h = g["r"][ti], g["th"][ti]
        best = np.inf
        for oj in oids:
            if oj == oid:
                continue
            gj = geo[oj]
            if not gj["present"][ti]:
                continue
            r_j = gj["r"][ti]
            if not np.isfinite(r_j) or r_j >= r_h or r_j < 0.1:
                continue
            alpha = np.arctan2(gj["foot"][ti], r_j)               # angular half-width
            if abs(th_h - gj["th"][ti]) < alpha:
                best = min(best, r_j)
        return best

    def ahead_infov(oid):
        g = geo[oid]
        return g["present"] & (g["x"] > 0) & (np.abs(g["th"]) <= half_fov)

    cands = [oid for oid in oids if geo[oid]["moving"] and ahead_infov(oid).any()]

    # hazard: a moving forward agent; prefer one with an occlusion->reveal, then closest approach
    scored = []
    for oid in cands:
        occ = np.array([np.isfinite(occluded_by(oid, ti)) for ti in range(T)]) & geo[oid]["present"]
        los = ahead_infov(oid) & ~occ
        reveal = (occ.any() and los.any()
                  and np.flatnonzero(occ).min() < np.flatnonzero(los).max())
        min_r = float(np.nanmin(np.where(geo[oid]["present"], geo[oid]["r"], np.inf)))
        scored.append((reveal, -min_r, oid))
    hazard, hazard_kind = None, "none"
    if scored:
        scored.sort(reverse=True)
        hazard = scored[0][2]
        hazard_kind = "occlusion_reveal" if scored[0][0] else "nearest_forward_moving"

    hazard_los = np.zeros(T, bool)
    is_occl = np.zeros(T, bool)
    dist_blind = np.full(T, 100.0)
    gt_xy = np.full((T, 2), np.nan)
    if hazard is not None:
        g = geo[hazard]
        infov = ahead_infov(hazard)
        for ti in range(T):
            if not g["present"][ti]:
                continue
            occ_r = occluded_by(hazard, ti)
            occluded = np.isfinite(occ_r)
            is_occl[ti] = occluded
            hazard_los[ti] = bool(infov[ti] and not occluded)
            if occluded:
                dist_blind[ti] = occ_r
            gt_xy[ti] = (g["x"][ti], g["y"][ti])

    tel = ScenarioTelemetry(
        ego_v=ego_v, ego_jerk=ego_jerk, steer_rate=steer_rate,
        latency_ms=np.full(T, 18.0),               # STUB (TanitAD-4B) -> feeds CNCE only
        hazard_los_flag=hazard_los, dist_to_blind_spot=dist_blind,
        is_occluded_flag=is_occl,
        wm_hazard_xy=np.full((T, 2), np.nan),      # no model -> LOPS 0.0 baseline
        gt_hazard_xy=gt_xy,
        dt=dt, collisions=0, ego_mass_kg=1500.0, params_billions=4.0,
    )
    prov = {
        "clip": clip, "T": T, "dt": round(dt, 4), "n_objects_total": len(oids),
        "n_moving_forward_cands": len(cands),
        "hazard_id": hazard, "hazard_kind": hazard_kind,
        "hazard_type": geo[hazard]["type"] if hazard else None,
        "frac_occluded": round(float(is_occl.mean()), 3),
        "frac_los": round(float(hazard_los.mean()), 3),
        "min_dist_blind": round(float(dist_blind.min()), 2),
        "ego_v_mean": round(float(ego_v.mean()), 2),
        "ego_v_max": round(float(ego_v.max()), 2),
    }
    return {"tel": tel, "prov": prov, "gt_xy": gt_xy, "is_occl": is_occl}


def oracle_lops(built: dict, noise_m: float = 0.3, seed: int = 0) -> float:
    """Pipeline-validation LOPS: perfect-perception oracle wm = gt + N(0,noise). NOT a model claim
    -- proves LOPS end-to-end returns ~E[exp(-gamma*|N(0,0.3)|_2)] ~ 0.8325 on real occluded frames."""
    tel = built["tel"]
    gt, occ = built["gt_xy"], built["is_occl"]
    wm = gt + np.random.default_rng(seed).normal(0, noise_m, gt.shape)
    wm[~occ] = np.nan
    t2 = ScenarioTelemetry(
        ego_v=tel.ego_v, ego_jerk=tel.ego_jerk, steer_rate=tel.steer_rate,
        latency_ms=tel.latency_ms, hazard_los_flag=tel.hazard_los_flag,
        dist_to_blind_spot=tel.dist_to_blind_spot, is_occluded_flag=occ,
        wm_hazard_xy=wm, gt_hazard_xy=gt, dt=tel.dt)
    return round(float(compute_lops(t2)), 4)


def score_clip(clip: str, root: str | Path) -> dict:
    built = derive_telemetry(clip, root)
    suite = run_scenario_suite(built["tel"], model_name=f"cosmos:{clip[:8]}")
    suite["LOPS_oracle"] = oracle_lops(built)
    suite.update({k: built["prov"][k] for k in
                  ("hazard_kind", "hazard_type", "frac_occluded", "frac_los",
                   "min_dist_blind", "ego_v_mean", "n_objects_total")})
    return {"suite": suite, "prov": built["prov"]}


def main():
    import argparse
    import glob
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="dir holding vehicle_pose/ + all_object_info/ + pinhole_intrinsic/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    clips = sorted({Path(p).name[:-4]
                    for p in glob.glob(str(Path(args.root) / "vehicle_pose" / "*.tar"))})
    rows, provs = [], []
    for clip in clips:
        try:
            r = score_clip(clip, args.root)
            s, p = r["suite"], r["prov"]
            rows.append(s)
            provs.append(p)
            print(f"{clip[:8]} v~{p['ego_v_mean']:5.1f} obj={p['n_objects_total']:3d} "
                  f"haz={p['hazard_kind']:22s} occ={p['frac_occluded']:.2f} "
                  f"LAL={s['LAL_s']:7.2f} LALv2={s['LAL_v2_s']:7.2f} TMS={s['TMS']:.3f} "
                  f"OKRI={s['OKRI']:8.2f} CNCE={s['CNCE']:7.1f} LOPS={s['LOPS']} orc={s['LOPS_oracle']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{clip[:8]} ERR {type(e).__name__}: {e}")
    if args.out:
        json.dump({"rows": rows, "prov": provs}, open(args.out, "w"), indent=1, default=str)
    print(f"[done] {len(rows)} clips scored")


if __name__ == "__main__":
    main()
