"""Synthetic TanitResim session bundle — a pod-free, full-viz-standard demo.

Builds a self-contained bundle exercising EVERY element of THE STANDARD
(``taniteval.corpus_overlay``) without a checkpoint, a corpus, or a GPU:

  * camera projection    — per-arm trajectory fans on a road-like frame;
  * metric BEV inset      — all arms + GT in metres (the master panel);
  * decoded intent text   — each arm's tactical maneuver (maneuver_probs argmax)
                            + strategic route/goal (nav_cmd), rendered as the
                            camera HUD;
  * ADE + v0              — per-arm error and ego speed;
  * formal-gate panel     — a synthetic D1-D3 + Phase-0 GO verdict;
  * BEV-only fallback     — one episode on an "uncalibrated" corpus so the
                            camera pane shows the fallback note and the BEV
                            carries the comparison.

Used by ``scripts/resim_app.py --demo`` (one-command demo) and by
``tests/test_resim.py``. Deterministic (seeded) so the bundle is reproducible.

    from tanitad.resim.sample import make_sample_bundle
    make_sample_bundle("/tmp/resim-demo/demo")   # then serve the PARENT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tanitad.replay.engine import WAYPOINT_STEPS, ArmOutput, TimestepRecord
from tanitad.resim.export import export_bundle

# Kinematic maneuver + strategic route/goal class names (mirror
# tanitad.refs.refb / scripts.refb_labels — kept local so the demo needs no
# torch/refb import beyond what export_bundle's maneuver labelling already
# pulls). Order is canonical.
MANEUVER_CLASSES = ("lane_keep", "turn_left", "turn_right",
                    "accelerate", "brake_stop")
LANE_KEEP, TURN_LEFT, TURN_RIGHT, ACCELERATE, BRAKE_STOP = range(5)

# Per-episode scenario: (corpus tag, kinematic maneuver the ego executes,
# nav command index into export.NAV_COMMANDS, human label for the picker).
_SCENARIOS = [
    ("physicalai-val-demo", TURN_LEFT, 1, "urban left turn"),
    ("comma2k19-val-demo", LANE_KEEP, 0, "highway cruise"),
    ("physicalai-val-demo", BRAKE_STOP, 3, "braking for lead"),
    ("cosmos-ood-demo", TURN_RIGHT, 2, "OOD (BEV-only)"),   # uncalibrated
]
_UNCALIBRATED = "cosmos-ood-demo"


def _road_frame(t: int, curve: float, h: int = 256, w: int = 256) -> np.ndarray:
    """A cheap but legible driving-scene frame: sky gradient, tarmac, a
    dashed lane line that bends with ``curve`` and scrolls with ``t`` so
    playback shows motion. Overlays read far better on this than on noise."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = h // 2
    # sky: deep slate -> lighter near the horizon (per-row column vector)
    ramp = np.linspace(0.0, 1.0, horizon)[:, None]           # [horizon, 1]
    img[:horizon, :, 0] = (22 + ramp * 22).astype(np.uint8)
    img[:horizon, :, 1] = (30 + ramp * 34).astype(np.uint8)
    img[:horizon, :, 2] = (52 + ramp * 60).astype(np.uint8)
    # road: grey tarmac below the horizon
    img[horizon:] = (56, 60, 66)
    # perspective lane centre line (dashed), bending by `curve`
    for row in range(horizon, h):
        depth = (row - horizon) / max(1, h - horizon)          # 0..1 near->far
        cx = int(w / 2 + curve * (1 - depth) * 90)
        half = max(1, int(2 + depth * 6))
        if ((row + t * 3) // 14) % 2 == 0:                     # dashes scroll
            img[row, max(0, cx - half):min(w, cx + half)] = (210, 200, 120)
        # road edges
        edge = int(depth * (w // 2 - 8))
        img[row, max(0, w // 2 - edge - 2):w // 2 - edge] = (120, 126, 134)
        img[row, w // 2 + edge:min(w, w // 2 + edge + 2)] = (120, 126, 134)
    return img


def _poses(T: int, maneuver: int, v0: float = 9.0) -> np.ndarray:
    """Ego poses [T,4]=(x,y,yaw,v) that classify to ``maneuver`` under the
    kinematic labeller (yaw for turns, speed slope for accel/brake)."""
    dt = 0.1
    yaw_rate = {TURN_LEFT: 0.22, TURN_RIGHT: -0.22}.get(maneuver, 0.0)
    dv = {ACCELERATE: 0.9, BRAKE_STOP: -0.9}.get(maneuver, 0.0)   # m/s per s
    t = np.arange(T)
    yaw = yaw_rate * dt * t
    v = np.clip(v0 + dv * dt * t, 0.0, None)
    # integrate a rough path (display-only; BEV uses per-step waypoints)
    x = np.cumsum(v * np.cos(yaw) * dt)
    y = np.cumsum(v * np.sin(yaw) * dt)
    return np.stack([x, y, yaw, v], axis=1).astype(np.float64)


def _fan(v0: float, curve: float, bias: float, jitter: float,
         rng: np.random.Generator) -> np.ndarray:
    """A plausible [H,2] ego-metric waypoint fan: forward ~ v0, lateral bends
    with ``curve``; ``bias``/``jitter`` differentiate the arms from GT."""
    out = []
    for k in WAYPOINT_STEPS:
        fwd = v0 * (k * 0.1) * (1.0 + 0.02 * bias)
        lat = curve * (k * 0.1) ** 2 * 1.4 + bias * 0.15 \
            + rng.normal(0, jitter)
        out.append([fwd, lat])
    return np.array(out, dtype=np.float64)


def _man_probs(true_man: int, sharpness: float,
               rng: np.random.Generator) -> np.ndarray:
    """A softmax-like maneuver distribution peaked (imperfectly) on the true
    class — the arm's DECODED tactical intent."""
    logits = rng.normal(0, 1.0, size=len(MANEUVER_CLASSES))
    logits[true_man] += sharpness
    p = np.exp(logits - logits.max())
    return (p / p.sum()).astype(np.float64)


def _arm_output(name: str, k_step: int, v0: float, curve: float,
                true_man: int, nav_cmd: int,
                rng: np.random.Generator) -> ArmOutput:
    """One arm's synthetic output for one step, with per-arm character:
    main = accurate + imagination/belief heads; refa = frozen-encoder-ish
    (higher error, imagination only); refb = maneuver/route policy heads."""
    prof = {
        "main": dict(bias=0.3, jit=0.05, sharp=3.2, lat=11.0),
        "refa": dict(bias=1.6, jit=0.16, sharp=1.4, lat=9.0),
        "refb": dict(bias=0.9, jit=0.10, sharp=2.4, lat=5.0),
    }[name]
    wp = _fan(v0, curve, prof["bias"], prof["jit"], rng)
    steer = float(np.clip(curve * 0.8 + rng.normal(0, 0.03), -1, 1))
    accel = float({ACCELERATE: 0.6, BRAKE_STOP: -0.7}.get(true_man, 0.0)
                  + rng.normal(0, 0.05))
    out = ArmOutput(latency_ms=prof["lat"] + rng.normal(0, 0.4),
                    waypoints=wp, action=np.array([steer, accel]))
    if name == "main":
        out.imag_rel = {1: float(0.35 + 0.03 * k_step),
                        4: float(0.7 + 0.05 * k_step)}
        out.sigma = float(0.25 + 0.01 * k_step)
        out.maneuver_probs = _man_probs(true_man, prof["sharp"], rng)
        out.maneuver_gt = true_man
        out.nav_cmd = nav_cmd
    elif name == "refa":
        out.imag_rel = {1: float(0.6 + 0.04 * k_step),
                        4: float(1.1 + 0.06 * k_step)}
        out.maneuver_probs = _man_probs(true_man, prof["sharp"], rng)
        out.maneuver_gt = true_man
        out.nav_cmd = nav_cmd
    else:  # refb
        out.maneuver_probs = _man_probs(true_man, prof["sharp"], rng)
        out.maneuver_gt = true_man
        out.nav_cmd = nav_cmd
        out.conf = float(np.clip(0.8 - 0.02 * k_step, 0, 1))
        out.ood = float(np.clip(0.15 + 0.03 * k_step, 0, 1))
    return out


def _records(arms=("main", "refa", "refb"), n_step: int = 14, seed: int = 7):
    """Synthetic replay stream over the _SCENARIOS episodes."""
    rng = np.random.default_rng(seed)
    step = 0
    for ep, (corpus, man, nav, _label) in enumerate(_SCENARIOS):
        curve = {TURN_LEFT: 0.16, TURN_RIGHT: -0.16}.get(man, 0.01)
        v0 = 9.0 if man != BRAKE_STOP else 7.0
        for j in range(n_step):
            t = 3 + j
            vv = v0 - (0.3 * j if man == BRAKE_STOP else 0.0)
            gt = _fan(vv, curve, 0.0, 0.0, rng)
            yield TimestepRecord(
                step=step, corpus=corpus, episode_id=ep, ep_index=ep, t=t,
                gt_waypoints=gt,
                gt_action=np.array([curve * 0.8,
                                    {ACCELERATE: 0.6,
                                     BRAKE_STOP: -0.7}.get(man, 0.0)]),
                speed=float(vv), yaw_rate=float(curve * 0.2),
                arms={a: _arm_output(a, j, vv, curve, man, nav, rng)
                      for a in arms},
                frame=_road_frame(j, curve))
            step += 1


def _demo_gates() -> tuple[dict, dict]:
    """Synthetic per-arm D1-D3 gate blocks + Phase-0 GO summary (shape matches
    compare_arms.compact_gate_blocks so the UI panel/banner populate)."""
    arm_gates = {
        "main": {"D1": "PASS", "d1_ade_0_2s": 0.44,
                 "oracle_ceiling_ade_0_2s": 0.30, "D2": "PASS",
                 "d2_dir_acc": 0.89, "D3": "PASS", "d3_ratio": 1.12,
                 "grounded_ade_0_2s": 0.51, "grounded_beats_cv": True,
                 "maneuver_balacc": 0.71, "route_balacc": 0.66},
        "refa": {"D1": "FAIL", "d1_ade_0_2s": 1.28,
                 "oracle_ceiling_ade_0_2s": 0.33, "D2": "PASS",
                 "d2_dir_acc": 0.81, "D3": "N/A", "d3_ratio": None,
                 "grounded_ade_0_2s": 1.40, "grounded_beats_cv": False,
                 "maneuver_balacc": 0.52, "route_balacc": 0.58},
        "refb": {"D1": "FAIL", "d1_ade_0_2s": 1.05,
                 "oracle_ceiling_ade_0_2s": 0.31, "D2": "PASS",
                 "d2_dir_acc": 0.84, "D3": "N/A", "d3_ratio": None,
                 "grounded_ade_0_2s": 1.12, "grounded_beats_cv": False,
                 "maneuver_balacc": 0.63, "route_balacc": 0.61},
    }
    gates_summary = {
        "baselines": {"constant_velocity": 0.58, "go_straight": 0.63,
                      "constant_yaw_rate": 0.60},
        "verdict": {
            "per_metric": {"d1_decode_ade_0_2s": {"winner_lowest": "main"}},
            "hierarchy_edge_necessary_conditions": {
                "flagship_beats_refs_on_d1_decode": True,
                "flagship_grounded_beats_cv_floor": True}},
        "n_val_episodes": 4, "n_windows": 56,
        "camera_ade_max_m": 1.0, "oracle_ceiling_target_m": 1.65}
    return arm_gates, gates_summary


def make_sample_bundle(out_dir: str | Path, *,
                       session_name: str = "TanitResim demo — synthetic",
                       n_step: int = 14, seed: int = 7,
                       with_gates: bool = True) -> dict:
    """Write a full-viz-standard synthetic bundle to ``out_dir``; return the
    session dict. Serve the PARENT directory with ``resim_app.py``."""
    recs = list(_records(n_step=n_step, seed=seed))
    ego_poses = {ep: _poses(3 + n_step + 25, man)
                 for ep, (_c, man, _n, _l) in enumerate(_SCENARIOS)}
    arm_gates, gates_summary = _demo_gates() if with_gates else (None, None)
    return export_bundle(
        recs, out_dir, session_name,
        corpora=sorted({c for c, *_ in _SCENARIOS}),
        arm_ckpts={"main": "flagship4b-speedjerk-30k.pt",
                   "refa": "refa-dinov2-4b-30k.pt",
                   "refb": "refb-speed-30k.pt"},
        maneuver_classes=MANEUVER_CLASSES,
        ego_poses=ego_poses,
        arm_gates=arm_gates, gates_summary=gates_summary,
        uncalibrated_corpora={_UNCALIBRATED})


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="write a TanitResim demo bundle")
    ap.add_argument("out", help="bundle dir (serve its PARENT)")
    args = ap.parse_args()
    s = make_sample_bundle(args.out)
    n = sum(len(e["steps"]) for e in s["episodes"])
    print(f"[sample] {len(s['episodes'])} episodes, {n} steps -> {args.out}")
