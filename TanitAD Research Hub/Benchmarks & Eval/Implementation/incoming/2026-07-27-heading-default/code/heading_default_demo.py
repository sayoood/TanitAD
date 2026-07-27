"""Machine-readable evidence for the comma2k19 heading DEFAULT FLIP.

Emits `raw/heading_default_guard.json`: every claim in HEADING_DEFAULT.md §1-§3
as a number or a caught exception, produced by RUNNING the shipped code — not
by restating a test name. Three blocks:

  `defect`       the legacy label reproduced on a standstill fixture (the
                 FAILING DIRECTION — a guard that cannot fire is worse than none)
  `guard`        every accident mode and every half-acknowledgement, with the
                 exception each one actually raises
  `cache_keys`   the legacy dir name is UNCHANGED, the repaired one CANNOT
                 collide, and acknowledged-legacy is BIT-IDENTICAL to an
                 independent reimplementation of the pre-flip formula

Run (dev box, project venv):
    cd stack && python "<this file>"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "stack"))

from tanitad.data.comma2k19 import (DEFAULT_HEADING_MODE,  # noqa: E402
                                    HEADING_MODE_HOLD, HEADING_MODE_LEGACY,
                                    HEADING_OBSERVABLE_V_MPS,
                                    LEGACY_HEADING_REASON,
                                    LegacyHeadingRefused, actions_and_poses,
                                    cache_build_params, ecef_to_enu,
                                    label_params, resolve_heading_mode)
from tanitad.data.epcache import cache_key  # noqa: E402

REF_ECEF = np.array([4278000.0, 635000.0, 4672000.0])
EAST = np.array([-0.147, 0.989, 0.0])
N_STILL, N_MOVING, DT = 6, 24, 0.05
ACK = dict(allow_legacy_heading=True, legacy_heading_reason=LEGACY_HEADING_REASON)


def fixture():
    n = N_STILL + N_MOVING
    ft = np.arange(n) * DT
    rng = np.random.default_rng(0)
    vel = np.zeros((n, 3))
    d = rng.normal(size=(N_STILL, 3))
    vel[:N_STILL] = 0.01 * d / np.linalg.norm(d, axis=1, keepdims=True)
    vel[N_STILL:] = 10.0 * EAST
    pos = REF_ECEF[None] + np.cumsum(vel * DT, axis=0)
    can = (ft, np.full(n, 10.0))
    return ft, pos, vel, can, (ft, np.full(n, 15.3))


def yaw_rate(poses):
    dy = np.diff(poses[:, 2].astype(np.float64))
    return np.arctan2(np.sin(dy), np.cos(dy)) / DT


def refused(fn, *a, **kw):
    """Return the exception class+message a refusal actually produced."""
    try:
        fn(*a, **kw)
    except Exception as e:                                   # noqa: BLE001
        return {"raised": type(e).__name__, "msg": str(e)[:160]}
    return {"raised": None, "msg": "NOT REFUSED — the guard did not fire"}


def main():
    ft, pos, vel, can, steer = fixture()
    _, p_leg = actions_and_poses(ft, pos, vel, can, steer, 1,
                                 heading_mode=HEADING_MODE_LEGACY, **ACK)
    _, p_def = actions_and_poses(ft, pos, vel, can, steer, 1)
    still = slice(0, N_STILL - 1)
    obs = p_leg[:, 3] >= HEADING_OBSERVABLE_V_MPS

    # independent reimplementation of the PRE-FLIP default
    enu_p, enu_v = ecef_to_enu(pos, vel)
    ref = np.column_stack([enu_p[:, 0], enu_p[:, 1],
                           np.arctan2(enu_v[:, 1], enu_v[:, 0]),
                           np.linalg.norm(enu_v, axis=1)]).astype(np.float32)

    base = {"size": 256, "n_stack": 3, "stride": 2, "max_steps": 300}
    srcs = [f"Chunk_1/route{i}/seg{i}" for i in range(50)]

    out = {
        "what": "comma2k19 heading DEFAULT FLIP — guard + reproducibility pin",
        "date": "2026-07-27",
        "evidence_class": "MEASURED (ours; this script, run on the dev box)",
        "tier": "instrument-grade (a synthetic fixture that REPRODUCES the "
                "measured corpus defect; it is not a corpus measurement)",
        "label_protocol": {
            "heading_repair_default_before": HEADING_MODE_LEGACY,
            "heading_repair_default_after": DEFAULT_HEADING_MODE,
            "v_min": HEADING_OBSERVABLE_V_MPS,
        },
        "defect": {
            "fixture": f"{N_STILL} standstill frames (|v| = 0.01 m/s, random "
                       f"direction) then {N_MOVING} frames at 10 m/s, dt={DT}s",
            "max_abs_yaw_rate_rad_s_LEGACY": float(
                np.abs(yaw_rate(p_leg)[still]).max()),
            "max_abs_yaw_rate_rad_s_REPAIRED": float(
                np.abs(yaw_rate(p_def)[still]).max()),
            "physically_impossible_threshold_rad_s": 1.5,
            "legacy_is_impossible": bool(
                np.abs(yaw_rate(p_leg)[still]).max() > 1.5),
            "moving_part_bit_identical": bool(
                np.array_equal(p_leg[N_STILL:, 2], p_def[N_STILL:, 2])),
            "speed_channel_bit_identical": bool(
                np.array_equal(p_leg[:, 3], p_def[:, 3])),
            "n_frames_changed": int((p_leg[:, 2] != p_def[:, 2]).sum()),
            "n_frames_changed_above_v_min": int(
                (p_leg[obs, 2] != p_def[obs, 2]).sum()),
        },
        "guard": {
            "legacy_bare": refused(resolve_heading_mode, HEADING_MODE_LEGACY),
            "legacy_flag_no_reason": refused(
                resolve_heading_mode, HEADING_MODE_LEGACY, allow_legacy=True),
            "legacy_flag_blank_reason": refused(
                resolve_heading_mode, HEADING_MODE_LEGACY, allow_legacy=True,
                reason="   "),
            "legacy_reason_no_flag": refused(
                resolve_heading_mode, HEADING_MODE_LEGACY,
                reason="I want the old labels"),
            "unknown_mode": refused(resolve_heading_mode, "enu_velocty"),
            "cache_build_params_legacy_bare": refused(
                cache_build_params, base, HEADING_MODE_LEGACY),
            "actions_and_poses_legacy_bare": refused(
                actions_and_poses, ft, pos, vel, can, steer, 1,
                heading_mode=HEADING_MODE_LEGACY),
            "accident_None_resolves_to": resolve_heading_mode(None),
            "full_acknowledgement_resolves_to": resolve_heading_mode(
                HEADING_MODE_LEGACY, allow_legacy=True,
                reason=LEGACY_HEADING_REASON),
        },
        "cache_keys": {
            "label_params_legacy": label_params(HEADING_MODE_LEGACY),
            "label_params_default": label_params(),
            "key_base_preflip": cache_key(srcs, base),
            "key_legacy": cache_key(srcs, {**base,
                                           **label_params(HEADING_MODE_LEGACY)}),
            "key_repaired": cache_key(srcs, {**base,
                                             **label_params(HEADING_MODE_HOLD)}),
            "existing_cache_dir_name_unchanged": cache_key(
                srcs, {**base, **label_params(HEADING_MODE_LEGACY)})
            == cache_key(srcs, base),
            "repaired_cannot_collide_with_legacy": cache_key(
                srcs, {**base, **label_params(HEADING_MODE_HOLD)})
            != cache_key(srcs, base),
        },
        "reproducibility_pin": {
            "claim": "acknowledged LEGACY == an INDEPENDENT reimplementation of "
                     "the pre-flip formula, exactly",
            "bit_identical": bool(np.array_equal(p_leg, ref)),
            "max_abs_delta": float(np.abs(p_leg - ref).max()),
        },
    }
    p = HERE.parent / "raw" / "heading_default_guard.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWROTE {p}")


if __name__ == "__main__":
    main()
