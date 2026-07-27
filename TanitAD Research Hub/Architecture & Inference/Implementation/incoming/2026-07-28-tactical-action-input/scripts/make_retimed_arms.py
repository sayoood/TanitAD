#!/usr/bin/env python3
"""Build the Block-B arms — give v1's tactical plan the speed it never saw.

⭐ NO GPU, NO CHECKPOINT, NO CORPUS. Every arm here is a pure arithmetic
transform of the pseudo-simulation arm panel's committed per-window dumps
(``pw_v1_tactical_follow.npz`` etc.), which is the panel's own VERIFIED
no-GPU recompute path. Row identity is therefore automatic and exact: the
derived arms inherit ``ep_i`` / ``anchor`` / ``pt_*`` / ``eid`` / ``ref_path`` /
``ref_yaw`` / ``v0`` unchanged, and only ``traj`` moves.

⚠️ NOTHING IN ``taniteval/pseudosim.py`` IS REIMPLEMENTED. Scoring is done by
``panel_combine.py`` (which imports ``PS.score_windows`` / the estimator
verbatim). This file only writes new ``pw_*.npz`` dumps.

THE FACTORISATION
-----------------
A plan is a CURVE ``gamma(s)`` plus a SCHEDULE ``s_t``. v1's tactical head
emits both, having never seen ``v0`` (verified: ``TacticalPolicy.forward``
takes ``ego=`` but ``ego_emb is None`` on every panel arm). An ego-speed
input's entire job is the SCHEDULE. So:

                | shape = v1's (it steers)  | shape = straight (no steering)
  schedule v1's | v1_tactical_follow  (pub) | v1_lat_straight
  schedule v0*t | v1_ego_v0        <-- THE  | cv_holdv0           (pub)
                |                      ARM  |

⭐ Two of the four cells are ALREADY-PUBLISHED arms, so the construction
validates itself in BOTH directions: this script asserts that
``gamma_v1 o s_v1`` reproduces ``pw_v1_tactical_follow`` and
``gamma_straight o s_cv`` reproduces ``pw_cv_holdv0`` — bit-exactly, up to
float32 round-off — and REFUSES to write anything if either fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# tanitad.ego_plan holds the geometry + the ego seam; taniteval.pseudosim holds
# DT and the grid contract. Both are IMPORTED, never copied.
for p in ("/root/TanitAD/stack", "/workspace/TanitAD/stack", "/root/taniteval"):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

from tanitad.ego_plan import (arc_length, constant_speed_schedule,  # noqa: E402
                              retime, straight_plan)
from taniteval import pseudosim as PS  # noqa: E402

PW_KEYS = ("traj", "ref_path", "ref_yaw", "v0", "pt_dlat", "pt_dyaw",
           "pt_dlon", "anchor", "ep_i", "eid")


def load_pw(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in PW_KEYS}


def save_pw(pw: dict, traj: np.ndarray, path: Path) -> None:
    out = dict(pw)
    out["traj"] = np.ascontiguousarray(traj.astype(np.float32))
    np.savez_compressed(path, **out)


def logged_arclength(ref_path: torch.Tensor) -> torch.Tensor:
    """TRUE distance the human travelled, per step. ``ref_path`` is [n, H+1, 2]
    in WORLD coordinates starting at the reference pose; arc length is frame
    invariant so no transform is needed.

    ⚠️ ORACLE — it reads the ego's OWN FUTURE. Upper bound only, never
    deployable, stamped on every node that uses it."""
    seg = ref_path[:, 1:] - ref_path[:, :-1]
    return seg.norm(dim=-1).cumsum(dim=-1)                    # [n, H]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", required=True,
                    help="directory holding the published pw_*.npz dumps")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="v1_tactical_follow",
                    help="the no-ego control arm the new arms are derived from")
    ap.add_argument("--prefix", default="",
                    help="prefix for the derived arm names (replication runs)")
    a = ap.parse_args()

    panel, out_dir = Path(a.panel_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = float(PS.DT)

    src = load_pw(panel / f"pw_{a.source}.npz")
    cv = load_pw(panel / "pw_cv_holdv0.npz")
    traj = torch.as_tensor(src["traj"])                       # [n, H, 2]
    v0 = torch.as_tensor(src["v0"])                           # [n]
    ref_path = torch.as_tensor(src["ref_path"])               # [n, H+1, 2]
    n, H, _ = traj.shape
    report: dict = {
        "_what": "Block B -- the tactical plan's SHAPE re-timed onto a SCHEDULE "
                 "that knows the ego speed. Zero fitted parameters.",
        "_evidence_class": "MEASURED (ours; artifact = the pw_*.npz written here)",
        "_no_gpu": True, "source_arm": a.source, "n_rows": int(n),
        "horizon_steps": int(H), "dt": dt,
        "_estimator_note": "scoring is done by panel_combine.py, which imports "
                           "taniteval.pseudosim verbatim; NOTHING is reimplemented",
    }

    # ------------------------------------------------------------------ #
    # SELF-VALIDATION, BOTH DIRECTIONS. Refuse to write if either fails.  #
    # ------------------------------------------------------------------ #
    s_own = arc_length(traj)[:, 1:]                           # v1's own schedule
    ident = retime(traj, s_own)["traj"]
    d_ident = float((ident - traj).abs().max())

    s_cv = constant_speed_schedule(v0, H, dt)
    cv_rebuilt = straight_plan(s_cv)
    d_cv = float((cv_rebuilt - torch.as_tensor(cv["traj"])).abs().max())

    # float32 round-off floor: the re-timing does a divide and a multiply, so
    # exact bitwise equality is not physical. 1e-3 m = 1 mm on a 25 m plan.
    TOL = 1e-3
    report["self_validation"] = {
        "gamma_v1_o_s_v1_reproduces_the_published_v1_arm": {
            "max_abs_err_m": d_ident, "tol_m": TOL, "pass": d_ident < TOL},
        "gamma_straight_o_s_cv_reproduces_the_published_cv_arm": {
            "max_abs_err_m": d_cv, "tol_m": TOL, "pass": d_cv < TOL},
    }
    if not (d_ident < TOL and d_cv < TOL):
        print(f"⛔ SELF-VALIDATION FAILED: identity {d_ident:.3e} m, "
              f"cv {d_cv:.3e} m (tol {TOL}). REFUSING to write arms.")
        (out_dir / "SELF_VALIDATION_FAILED.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return 2
    print(f"[blockB] self-validation PASS  identity {d_ident:.3e} m  "
          f"cv {d_cv:.3e} m", flush=True)

    # ------------------------------------------------------------------ #
    # the arms                                                            #
    # ------------------------------------------------------------------ #
    s_oracle = logged_arclength(ref_path)
    arms = {
        # ⭐ THE ARM: v1's own curve, walked at the speed the planner never saw
        "v1_ego_v0": (traj, s_cv,
                      "v1's tactical CURVE, re-timed to v0*t*dt -- the plan now "
                      "knows the ego speed. ZERO fitted parameters."),
        # the ceiling of ANY longitudinal input, including a perfect predictor
        "v1_ego_oracle_lon": (traj, s_oracle,
                              "ORACLE: v1's curve re-timed to the TRUE logged "
                              "arc length. Upper bound of any longitudinal "
                              "input. NOT DEPLOYABLE."),
        # ⭐ the pure longitudinal ceiling: NO steering at all, but the TRUE
        # distance. If this matches v1_ego_oracle_lon, the entire oracle gain is
        # longitudinal and v1's tactical SHAPE contributes nothing.
        "oracle_lon_straight": (straight_plan(s_oracle), s_oracle,
                                "ORACLE: straight-ahead shape walked to the TRUE "
                                "logged distance -- the pure longitudinal "
                                "ceiling, zero steering. NOT DEPLOYABLE."),
        # the other factorial cell: v1's extent, no steering
        "v1_lat_straight": (straight_plan(s_own), s_own,
                            "straight-ahead shape on v1's OWN schedule -- "
                            "isolates what v1's steering contributes."),
        # ⛔ the deliberate degradation. MUST score worse.
        "v1_ego_half": (traj, constant_speed_schedule(v0, H, dt, scale=0.5),
                        "⛔ DELIBERATE DEGRADATION (half speed). MUST score "
                        "WORSE -- the composite is known to RISE for a slowed "
                        "planner because a barely-moving plan is scored NaN."),
        # ⛔ the mirror degradation, so the guard is exercised in BOTH directions
        "v1_ego_double": (traj, constant_speed_schedule(v0, H, dt, scale=2.0),
                          "⛔ DELIBERATE DEGRADATION (double speed). MUST also "
                          "score worse -- a control that only fails in one "
                          "direction cannot detect a monotone artefact."),
    }

    report["arms"] = {}
    for name, (shape, sched, what) in arms.items():
        out = retime(shape, sched)
        key = f"{a.prefix}{name}"
        save_pw(src, out["traj"].numpy(), out_dir / f"pw_{key}.npz")
        end = out["traj"][:, -1]
        node = {
            "what": what,
            "frac_rows_extrapolated_past_path_end":
                round(float(out["frac_extrapolated"]), 6),
            "frac_rows_degenerate_no_tangent":
                round(float(out["degenerate"].float().mean()), 6),
            "mean_plan_endpoint_x_m": round(float(end[:, 0].mean()), 4),
            "mean_plan_endpoint_y_m": round(float(end[:, 1].mean()), 4),
            "mean_plan_path_length_m":
                round(float(arc_length(out["traj"])[:, -1].mean()), 4),
            "_derived_from": a.source,
            "_no_gpu_no_checkpoint": True,
        }
        if "oracle" in name:
            node["⚠️_ORACLE"] = ("reads the ego's OWN FUTURE -- an upper bound, "
                                 "never a deployable arm")
        report["arms"][key] = node
        print(f"[blockB] wrote pw_{key}.npz  end_x {node['mean_plan_endpoint_x_m']:7.3f} m  "
              f"len {node['mean_plan_path_length_m']:7.3f} m  "
              f"extrap {node['frac_rows_extrapolated_past_path_end']:.4f}",
              flush=True)

    # the reference numbers, for the report's decomposition table
    report["reference_arms"] = {
        a.source: {
            "mean_plan_endpoint_x_m": round(float(traj[:, -1, 0].mean()), 4),
            "mean_plan_endpoint_y_m": round(float(traj[:, -1, 1].mean()), 4),
            "mean_plan_path_length_m": round(float(s_own[:, -1].mean()), 4)},
        "cv_holdv0": {
            "mean_plan_endpoint_x_m":
                round(float(torch.as_tensor(cv["traj"])[:, -1, 0].mean()), 4),
            "mean_plan_path_length_m": round(float(s_cv[:, -1].mean()), 4)},
        "logged_human_oracle": {
            "mean_path_length_m": round(float(s_oracle[:, -1].mean()), 4)},
    }
    (out_dir / f"{a.prefix}blockB_build.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[blockB] build report -> {out_dir}/{a.prefix}blockB_build.json")
    print("BLOCK_B_BUILD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
