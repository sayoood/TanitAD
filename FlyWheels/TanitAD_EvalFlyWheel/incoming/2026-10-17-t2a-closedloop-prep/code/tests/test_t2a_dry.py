"""T2-A CPU dry-path guards: the closed-loop harness, a STUB renderer, and the known answers.

CPU ONLY (CUDA_VISIBLE_DEVICES=-1 is set before torch loads; a test asserts no GPU is visible).

Every test is a KNOWN-VALUE assertion; the ones that matter come with a DELIBERATE-REGRESSION arm that must go RED
(the mutated run is fed through the SAME checker, and the checker must report a problem):

* the hold-speed floor `cl_ha0` drives a STRAIGHT line a KNOWN distance (15 m/s * 49 decision intervals * 0.1 s = 73.5 m between the first and the last decision), with
  steer and accel commands that read EXACTLY 0.0 -- and a bias in the steer, an offset in the accel or a wrong DT
  each turn the checker RED;
* the loop is closed on PERCEPTION: the pose handed to the renderer is the EXECUTED pose (a policy that veers off
  the log is rendered where it IS, not where the log was);
* the recorded action channel is a STEER ANGLE: a circle of known radius is reproduced with `action_units="steer"`
  and over-rotated ~2.9x with `"kappa"` (the R46 class);
* P10 reproduces the banked P10 predictions from banked kinematics (max |diff| < 1e-4 m on 300 windows) and the
  `"kappa"` mutation does not.

Run (from the kit or the repo):   T2A_TREE=<dir with stack/ and taniteval/> pytest -q tests/test_t2a_dry.py
Slow (CPU forward of the real checkpoint, ~minutes):   T2A_SLOW=1 ... -k refcv7
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PKG = CODE.parent
REPO = Path(os.environ.get("T2A_REPO") or PKG.parents[3])
if os.environ.get("T2A_TREE"):
    TREE = Path(os.environ["T2A_TREE"])
elif (CODE.parent / "tree").is_dir():
    TREE = CODE.parent / "tree"
else:
    TREE = REPO
HARNESS = (TREE / "stack/experiments/alpasim-gsplat")
if not (HARNESS / "closedloop_drive.py").exists():
    HARNESS = REPO / "stack/experiments/alpasim-gsplat"
for p in (TREE / "taniteval" / "tools", TREE / "taniteval", TREE / "stack" / "scripts", TREE / "stack",
          TREE / "stack/experiments/nurec-gsplat", HARNESS, CODE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import closedloop_drive as cd  # noqa: E402
import stub_gsplat_renderer as stub  # noqa: E402
import t2a_policies as tp  # noqa: E402

V, N = stub.STUB_V, 50
TICK = 0.1          # the 10 Hz control tick, written out HERE: a known value must not be read from the thing under test
P10_W = next((p for p in (PKG / "assets" / "p10_S2.pt", CODE.parent / "p10" / "p10_S2.pt") if p.exists()),
             PKG / "assets" / "p10_S2.pt")
FIX = PKG / "assets" / "p10_known_value_fixture.npz"


def test_no_gpu_is_visible():
    assert not torch.cuda.is_available()


# --------------------------------------------------------------------------------------------- #
# helpers                                                                                         #
# --------------------------------------------------------------------------------------------- #
def _scene():
    r = stub.NuRecGsplatRenderer("stub")
    tr = cd.InProcTransport(r)
    intr = cd.build_intr(tr.camera())
    n = r.n_frames() // 3
    gt_T = [r.gt_rig_to_world(f * 3) for f in range(n)]
    gt_ts = [r.frame_timestamps_us(f * 3)[1] for f in range(n)]
    return r, tr, intr, gt_T, gt_ts


def _rollout(policy, start=0, n=N):
    r, tr, intr, gt_T, gt_ts = _scene()
    rec = cd.run_rollout(tr, r, policy, intr, start, n, gt_T, gt_ts, gt_stride=3)
    return rec, r, gt_T


def check_straight(rec, v=V, n=N, tol=1e-9):
    """The analytic answer for a hold-speed floor on a straight, constant-speed log. -> list of problems."""
    bad = []
    st = rec["steps"]
    x0 = st[0]["ego"][0]
    if len(st) != n:
        bad.append(f"n_steps {len(st)} != {n}")
    if any(s["steer"] != 0.0 for s in st):
        bad.append("steer command is not EXACTLY 0.0")
    if any(s["accel"] != 0.0 for s in st):
        bad.append("accel command is not EXACTLY 0.0")
    for k, s in enumerate(st):
        want = x0 + k * v * TICK
        if abs(s["ego"][0] - want) > tol:
            bad.append(f"x[{k}] {s['ego'][0]} != {want}")
            break
        if abs(s["ego"][1]) > tol or abs(s["ego"][3]) > tol:
            bad.append(f"y/yaw[{k}] = {s['ego'][1]}, {s['ego'][3]}")
            break
        if abs(s["v"] - v) > 1e-9:
            bad.append(f"v[{k}] {s['v']} != {v}")
            break
    return bad


# --------------------------------------------------------------------------------------------- #
# 1 . the straight-line known distance, and the deliberate-regression arms                         #
# --------------------------------------------------------------------------------------------- #
def test_hold_speed_floor_drives_a_known_straight_distance():
    pol = cd.KinematicFloorPolicy("cl_ha0")
    rec, r, gt_T = _rollout(pol)
    assert check_straight(rec) == []
    drove = rec["steps"][-1]["ego"][0] - rec["steps"][0]["ego"][0]
    assert abs(drove - V * (N - 1) * TICK) < 1e-9          # 15 * 49 * 0.1 = 73.5 m between the first and last DECISION
    # and the car ENDS where the analytic answer says: one more step past the last decision
    T_end = r.render_calls[-1]
    x_end = np.linalg.inv(r.cam.T_sensor_rig)[0, 3]  # noqa: F841  (sanity: the matrix is invertible)
    assert abs(T_end[0, 3] - (rec["steps"][0]["ego"][0] + N * V * TICK + 1.0)) < 1e-9   # +1.0 m camera mount


def test_deliberate_regression_steer_bias_goes_red(monkeypatch):
    real = cd.wp_to_control
    monkeypatch.setattr(cd, "wp_to_control", lambda w, v, **k: (lambda s, a, vt, kp: (s + 0.01, a, vt, kp))(*real(w, v, **k)))
    rec, _, _ = _rollout(cd.KinematicFloorPolicy("cl_ha0"))
    bad = check_straight(rec)
    assert bad and any("steer" in b or "y/yaw" in b for b in bad), bad


def test_deliberate_regression_accel_offset_goes_red(monkeypatch):
    real = cd.wp_to_control
    monkeypatch.setattr(cd, "wp_to_control", lambda w, v, **k: (lambda s, a, vt, kp: (s, a + 0.5, vt, kp))(*real(w, v, **k)))
    bad = check_straight(_rollout(cd.KinematicFloorPolicy("cl_ha0"))[0])
    assert bad and any("accel" in b or "x[" in b or "v[" in b for b in bad), bad


def test_deliberate_regression_wrong_dt_goes_red(monkeypatch):
    monkeypatch.setattr(cd, "DT", 0.2)
    bad = check_straight(_rollout(cd.KinematicFloorPolicy("cl_ha0"))[0], tol=1e-9)
    assert bad, "a wrong tick length must not pass the known-distance check"


# --------------------------------------------------------------------------------------------- #
# 2 . the loop is closed on perception                                                            #
# --------------------------------------------------------------------------------------------- #
class _Veer:
    """A policy that does NOT follow the log: it always asks for a point 1 m to the left at 0.5 s."""
    window = cd.WINDOW
    consumes_nav_known = False
    canon_mode = None
    name = "veer"
    f_eff = None
    canon_provenance = {"mode": "none"}

    def plan(self, frames, intr, v0, nav_cmd, nav_known=None):
        t = np.array([[v0 * 0.5, 0.35], [v0 * 1.0, 1.4], [v0 * 1.5, 3.1], [v0 * 2.0, 5.6]], np.float64)
        return t, {}


def test_loop_is_closed_on_perception_render_pose_is_the_executed_pose():
    rec, r, gt_T = _rollout(_Veer())
    last_exec = np.array(rec["steps"][-1]["ego"])
    # the last render was requested from the camera at the pose AFTER the last step: it must have drifted off the log
    T = r.render_calls[-1]
    assert abs(T[1, 3]) > 0.05, "the car veered but the renderer was asked for a camera still on the logged line"
    # and the second-to-last render pose IS the pose the last decision was taken from (executed, not logged)
    ex = last_exec
    T_ego = np.eye(4)
    T_ego[:3, :3] = cd._rz(ex[3])
    T_ego[:3, 3] = ex[:3]
    assert np.abs(r.render_calls[-2] - T_ego @ r.cam.T_sensor_rig).max() < 1e-9
    # CONTROL: had the loop replayed the log, the camera y would be exactly 0 at every render
    assert max(abs(c[1, 3]) for c in r.render_calls) > 0.05
    # and the frame the policy sees genuinely depends on the pose (the stub's one honest property)
    pose_a, pose_b = r.render_calls[0].copy(), r.render_calls[-1].copy()    # (render() appends to the list)
    img_a, _, _ = r.render(pose_a)
    img_b, _, _ = r.render(pose_b)
    assert np.any(img_a != img_b)


# --------------------------------------------------------------------------------------------- #
# 3 . the hook is inert for every historical arm                                                  #
# --------------------------------------------------------------------------------------------- #
def test_history_hook_is_inert_for_historical_arms():
    for c in (cd.FlagshipV1Policy, cd.RefCPolicy, cd.RefCV3Policy, cd.KinematicFloorPolicy):
        assert not hasattr(c, "set_history"), c
    assert cd.EXTRA_ARMS == {} or all(k in tp.LIVE_ARMS or k == "refcv7" for k in cd.EXTRA_ARMS)


# --------------------------------------------------------------------------------------------- #
# 4 . the unit trap: the recorded channel is a STEER ANGLE                                         #
# --------------------------------------------------------------------------------------------- #
def _circle_history(kappa, v=15.0, n=10, dt=cd.DT):
    """Unicycle-exact (x, y, yaw, v) history on a circle of curvature `kappa` (rows = poses; v constant)."""
    x = y = yaw = 0.0
    rows = []
    for _ in range(n):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v * kappa * dt
    return np.array(rows)


def test_kin_from_history_recovers_the_encoder_steer_of_a_known_curvature():
    k = tp.kin_from_history(_circle_history(0.02))
    assert abs(k["kappa"] - 0.02) < 1e-12
    assert abs(k["steer0"] - math.atan(2.9 * 0.02)) < 1e-12
    assert abs(k["a0"]) < 1e-12 and k["v0"] == 15.0


def _lateral_at_2s(units, kappa=0.005):
    hist = _circle_history(kappa)
    pol = tp.LiveKinematicPolicy("cl_ha0_ext_live", action_units=units)
    pol.set_history(hist)
    traj, _ = pol.plan(None, None, 15.0, 0)
    return traj


def _euler_circle(kappa, v, steps):
    """INDEPENDENT known answer: the discrete unicycle (position advances on the yaw at the START of the step)."""
    x = y = yaw = 0.0
    for _ in range(steps):
        x += v * math.cos(yaw) * TICK
        y += v * math.sin(yaw) * TICK
        yaw += v * kappa * TICK
    return x, y


def test_live_echo_reproduces_a_circle_with_steer_units_and_overrotates_with_kappa_units():
    kappa, v = 0.005, 15.0                                       # R = 200 m: small angle, so the ratio is ~2.9 cleanly
    x_true, y_true = _euler_circle(kappa, v, 20)                 # 2.0 s
    good = _lateral_at_2s("steer")[-1]
    assert abs(good[1] - y_true) < 1e-9 and abs(good[0] - x_true) < 1e-9, (good, x_true, y_true)
    bad = _lateral_at_2s("kappa")[-1]                      # DELIBERATE REGRESSION: steer angle integrated as 1/m
    assert 2.7 < bad[1] / y_true < 3.1, (bad, y_true)            # ~2.9x over-rotation -- the R46 defect, reproduced
    assert abs(bad[1] - y_true) > 1.0, "the regression arm must be visibly wrong"


def test_live_echo_on_a_straight_constant_speed_log_equals_the_hold_speed_floor_exactly():
    rec, _, _ = _rollout(_mk("cl_ha0_ext_live"))
    assert check_straight(rec) == []


def _mk(mode):
    return tp.LiveKinematicPolicy(mode, p10_weights=str(P10_W) if mode == "cl_p10" else None)


# --------------------------------------------------------------------------------------------- #
# 5 . P10 known value                                                                            #
# --------------------------------------------------------------------------------------------- #
def test_p10_reproduces_the_banked_predictions_from_banked_kinematics():
    z = np.load(FIX)
    core = tp._P10Core(str(P10_W), "steer")
    out = core.paths(z["kin"])
    assert np.abs(out["ha0"] - z["ha0"]).max() < 2e-4                 # float32 dump vs float64 rebuild
    assert np.abs(out["ha0_ext"] - z["ha0_ext"]).max() < 2e-4
    assert np.abs(out["kd_x"] - z["kd_x"]).max() < 2e-4
    assert np.abs(out["P10"] - z["P10"]).max() < 1e-3, np.abs(out["P10"] - z["P10"]).max()


def test_p10_deliberate_regression_kappa_units_is_not_the_banked_floor():
    z = np.load(FIX)
    core = tp._P10Core(str(P10_W), "kappa")
    assert np.abs(core.paths(z["kin"])["ha0_ext"] - z["ha0_ext"]).max() > 1.0     # the unit trap moves a 2 s path by metres


def test_p10_live_drift_is_the_floors_own_bias_integrated_by_a_unit_gain_loop():
    # CONTROL (must read exactly straight): with the MLP residual zeroed, P10 == kd_x, and kd_x on a straight constant-
    # speed state is the hold-speed line -- so the live loop must stay on the log to 1e-9.
    pol = _mk("cl_p10")
    for prm in pol.core.net[2].parameters():
        torch.nn.init.zeros_(prm)
    pol.core.stats = (pol.core.stats[0], pol.core.stats[1], np.zeros_like(pol.core.stats[2]))
    rec0, _, _ = _rollout(pol)
    assert np.abs(np.array([s["ego"][1] for s in rec0["steps"]])).max() < 1e-9
    # MEASURED behaviour of the real weights: the lateral residual P10 emits at the zero state is fed back as a
    # curvature each tick and integrates (loop gain ~1) -- it drifts, boundedly, and keeps the speed.
    rec, _, _ = _rollout(_mk("cl_p10"))
    ego = np.array([s["ego"] for s in rec["steps"]])
    assert ego[-1, 0] - ego[0, 0] > 0.8 * V * (N - 1) * TICK
    drift = float(np.abs(ego[:, 1]).max())
    assert 0.05 < drift < 5.0, drift                          # 2.35 m on 2026-10-17 (stub straight log, V=15)
    assert all(set(s["extra"]) >= {"floor_v0_ms", "floor_steer0_enc_rad"} for s in rec["steps"])


def test_p10_held_at_t0_equals_p10_of_the_t0_state_and_is_reissued_unchanged():
    pol = tp.HeldKinematicPolicy(str(P10_W))
    rec, r, gt_T = _rollout(pol)
    plans = np.array([s["plan"] for s in rec["steps"]])
    assert np.array_equal(plans, np.broadcast_to(plans[0], plans.shape))          # held: identical every tick
    f0 = 9
    kin = np.array([[rec["steps"][0]["v"], pol.t0["a0_mps2"], pol.t0["steer0_enc_rad"]]])
    assert np.abs(pol.core.paths(kin)["P10"][0] - plans[0]).max() < 1e-12
    assert abs(pol.t0["a0_mps2"]) < 1e-9 and abs(pol.t0["steer0_enc_rad"]) < 1e-9   # straight, constant-speed log
    assert pol.t0["pose_indices_read"] == [f0 - 2, f0 - 1, f0]                        # nothing after t0


# --------------------------------------------------------------------------------------------- #
# 6 . extrinsics convention (the lift bank's camera mount)                                        #
# --------------------------------------------------------------------------------------------- #
def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = tp.quat_xyzw_to_rot(*q)
        qx, qy, qz, qw = tp.rot_to_quat_xyzw(R)
        assert np.abs(tp.quat_xyzw_to_rot(qx, qy, qz, qw) - R).max() < 1e-12


@pytest.mark.skipif(not Path("D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json").exists()
                    or not Path("E:/thor_archive/2026-09-14/nurec_scenes").exists(), reason="dev-box assets")
def test_nurec_extrinsics_match_the_physicalai_convention():
    tab = json.loads(Path("D:/refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json").read_text())
    Rc = [tp.quat_xyzw_to_rot(r["qx"], r["qy"], r["qz"], r["qw"]) for r in tab.values() if "qx" in r]
    sc = Path("E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release")
    for sub in ("00040136-e651-4abd-991d-0655ccda9430", "7c72937c-c620-4776-9555-d57222c0081f"):
        rig = tp.read_rig_json(str(sc / sub))
        T = tp.front_wide_T_sensor_rig(rig["json"])
        d = min(np.linalg.norm(R - T[:3, :3]) for R in Rc)
        assert d < 0.02, (sub, d)                                      # MEASURED 0.0033 / 0.0052
        ex = tp.extrinsics_from_T_sensor_rig(T)
        assert np.abs(tp.quat_xyzw_to_rot(ex.qx, ex.qy, ex.qz, ex.qw) - T[:3, :3]).max() < 1e-6   # T is float32-orthonormal
        assert 1.2 < ex.z < 1.7 and 1.5 < ex.x < 2.6


# --------------------------------------------------------------------------------------------- #
# 7 . the REAL harness main(), the REAL scorer, stub renderer                                       #
# --------------------------------------------------------------------------------------------- #
def _run_main(tmp_path, arm, extra=(), monkeypatch=None, scene_dir=None, starts="0,17", steps=N):
    argv = ["t2a_run.py", "--scene-dir", str(scene_dir or (tmp_path / "scene")), "--arm", arm, "--starts", starts,
            "--steps", str(steps), "--out", str(tmp_path / arm), *extra]
    (tmp_path / "scene").mkdir(exist_ok=True)
    stub.install()
    tp.register()
    old = sys.argv
    sys.argv = argv
    try:
        cd.main()
    finally:
        sys.argv = old
    return json.loads((tmp_path / arm / f"rollouts_{arm}_empty.json").read_text())


def test_main_end_to_end_cl_ha0_then_score_reads_the_known_zeros(tmp_path):
    from cl_metrics import collect, families
    payload = _run_main(tmp_path, "cl_ha0")
    assert payload["arm"] == "cl_ha0" and len(payload["rollouts"]) == 2
    for rec in payload["rollouts"]:
        assert check_straight(rec) == []
    d, rows, eids, summ = collect(str(tmp_path / "cl_ha0" / "rollouts_cl_ha0_empty.json"))
    fam = families(rows, eids, summ)
    # a straight log driven straight: zero path error, zero cross-track, full progress
    assert abs(fam["ADE"]["dist_to_gt_traj_m"]["mean"]) < 1e-9
    assert all(abs(s["max_cross_track"]) < 1e-9 for s in summ)
    assert all(abs(s["progress_rel"] - 1.0) < 0.05 for s in summ)


def test_box2_render_flags_reach_the_renderer_through_main(tmp_path):
    """The banked render flags (`--cull-scale-quantile 0.95 --sky-gain 0.3`) are what makes box-2 renders comparable with Thor's.
    Reachability from the actual caller: argv -> main() -> the renderer's `cull_by_scale` / `attach_sky`. run_on_box2.RENDER is the
    list the grid passes; it is fed here verbatim."""
    import run_on_box2 as rb
    stub.NuRecGsplatRenderer.instances.clear()
    _run_main(tmp_path, "cl_ha0", extra=tuple(rb.RENDER), starts="0", steps=3)
    (r,) = stub.NuRecGsplatRenderer.instances[-1:]
    assert r.cull_calls == [0.95], r.cull_calls
    assert len(r.sky_calls) == 1 and r.sky_calls[0]["gain"] == 0.3 and r.sky_calls[0]["lo_deg"] == 0.0 and r.sky_calls[0]["hi_deg"] == 6.0
    # control: WITHOUT the flags neither call happens (the check can fail)
    stub.NuRecGsplatRenderer.instances.clear()
    plain = tmp_path / "plain"
    plain.mkdir()
    _run_main(plain, "cl_ha0", starts="0", steps=3)
    (r2,) = stub.NuRecGsplatRenderer.instances[-1:]
    assert r2.cull_calls == [] and r2.sky_calls == []


def test_box2_runner_zero_control_check_reads_exact_zero_and_goes_red_on_a_1e_minus_9_steer(tmp_path):
    """`run_on_box2.zero_control` is the guard that marks the cl_ha0 arm FAILED on box 2. It must read EXACT zero on a real
    harness output and go RED on the smallest mutation (a 1e-9 steer, a 1e-9 accel)."""
    import run_on_box2 as rb
    _run_main(tmp_path, "cl_ha0")
    f = tmp_path / "cl_ha0" / "rollouts_cl_ha0_empty.json"
    z = rb.zero_control(f)
    assert z["exact_zero"] is True and z["n_steps"] == 2 * 50 and z["max_abs_steer"] == 0.0 and z["max_abs_accel"] == 0.0
    for key in ("steer", "accel"):
        d = json.loads(f.read_text())
        d["rollouts"][1]["steps"][7][key] = 1e-9
        bad = tmp_path / f"mut_{key}.json"
        bad.write_text(json.dumps(d))
        zb = rb.zero_control(bad)
        assert zb["exact_zero"] is False and zb["max_abs_" + key] == 1e-9, zb


def test_main_registered_live_arm_refuses_a_ckpt_it_would_not_read(tmp_path):
    with pytest.raises(SystemExit):
        _run_main(tmp_path, "cl_ha0_ext_live", extra=("--ckpt", "x.pt"))


# --------------------------------------------------------------------------------------------- #
# 7b. refcv7 pieces that need NO model: the canon cache and the per-rollout ceiling                #
# --------------------------------------------------------------------------------------------- #
def _bare_refcv7():
    """A RefCV7Policy shell with only the attributes `canon` / `set_t0` touch (no 100 M-parameter build)."""
    from tanitad.data.comma2k19 import stack_frames
    from tanitad.models.trunk_shapes import FRAME_416x1024
    p = object.__new__(tp.RefCV7Policy)
    p.torch, p.device, p.frame, p.window = torch, "cpu", FRAME_416x1024, 8
    p._stack, p.f_eff, p.canon_provenance, p._canon_cache = stack_frames, None, None, {}
    p.vmax_mode, p.vmax_ms, p._tick, p.t0 = "logged_2_6s", None, 3, None
    return p


def test_refcv7_canon_cache_is_value_neutral_and_resamples_one_frame_per_tick():
    import tanitad.data.calib as C
    r, tr, intr, gt_T, gt_ts = _scene()
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(11)]
    real, calls = C.cylindrical_rectify, {"n": 0}

    def wrapped(vid, intr_, frame, *a, **k):
        o = real(vid, intr_, frame, *a, **k)
        # (the real body sets last_* on the module-global NAME, which is `wrapped` while patched)
        calls["n"] += 1
        return o
    cached = _bare_refcv7()
    C.cylindrical_rectify = wrapped
    try:
        out1 = cached.canon(frames[:10], intr)
        n1 = calls["n"]
        out2 = cached.canon(frames[1:11], intr)           # the window slides by one frame
        n2 = calls["n"] - n1
    finally:
        C.cylindrical_rectify = real
    assert (n1, n2) == (10, 1), (n1, n2)                  # tick 1 resamples 10 frames, tick 2 exactly ONE
    ref2 = _bare_refcv7().canon(frames[1:11], intr)       # no cache history: every frame resampled from scratch
    assert torch.equal(out2, ref2), "the cache changed a value"
    assert out1.shape == (1, 8, 9, 416, 1024) and float(out2.max()) <= 1.0
    assert 0.5 < cached.canon_provenance["observed_frac"] <= 1.0


def _logged_speed_poses(speeds):
    """gt_T whose 10 Hz forward-difference speeds are `speeds` (m/s)."""
    x, T = 0.0, []
    for v in list(speeds) + [speeds[-1]]:
        M = np.eye(4)
        M[0, 3] = x
        T.append(M)
        x += v * TICK
    return T


def test_refcv7_ceiling_is_the_logged_max_over_2_to_6_s_after_t0_per_rollout():
    sp = [0.0] * 40 + [3.0] * 30 + [12.0] * 30 + [15.0] * 10          # stands, then creeps, then moves, then fast at the end
    gt = _logged_speed_poses(sp)
    p = _bare_refcv7()
    p.set_t0(gt, 9)                                                    # window = ticks 29..69 -> max 3.0
    assert p.t0["window_ticks"] == [29, 69] and abs(p.vmax_ms - 3.0) < 1e-9 and p._tick == 0 and not p.t0["window_truncated"]
    p.set_t0(gt, 30)                                                   # window = ticks 50..90 -> reaches the 12 m/s block
    assert abs(p.vmax_ms - 12.0) < 1e-9
    p.set_t0(gt, 95)                                                   # window 115..109(end) -> empty: ceiling WITHHELD
    assert p.vmax_ms is None and "WITHHELD" in p.t0["why"]
    p.set_t0(gt, 60)                                                   # window 80..120 but the log ends at 110 -> truncated
    assert p.t0["window_truncated"] is True and abs(p.vmax_ms - 15.0) < 1e-9
    p.vmax_mode = "withheld"
    p.vmax_ms = None
    p.set_t0(gt, 9)
    assert p.vmax_ms is None                                           # a withheld arm never reads the log


# --------------------------------------------------------------------------------------------- #
# 8 . SLOW: the real refcv7-50,400 on CPU, one tick                                               #
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(os.environ.get("T2A_SLOW") != "1", reason="set T2A_SLOW=1 (CPU forward of the real checkpoint)")
def test_refcv7_one_cpu_tick_through_the_stub_loop(tmp_path):
    ck = os.environ.get("T2A_CKPT", "E:/refcv7_eval_kit/ckpt/ckpt_50400.pt")
    cfg = os.environ.get("T2A_CONFIG", "E:/refcv7_eval_kit/ckpt/config.json")
    rig = tp.read_rig_json("E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/"
                           "00040136-e651-4abd-991d-0655ccda9430")
    pol = tp.RefCV7Policy(ck, cfg, rig, vmax_ms=None, device="cpu", scene_id="00040136-e651-4abd-991d-0655ccda9430")
    assert pol.prov["horizons"][:4] == list(cd.WP_STEPS) and pol.window == 8
    assert pol.prov["state_dict_strict"] and not pol.prov["missing"] and not pol.prov["unexpected"]
    r, tr, intr, gt_T, gt_ts = _scene()
    rec = cd.run_rollout(tr, r, pol, intr, 0, 1, gt_T, gt_ts, gt_stride=3)
    st = rec["steps"][0]
    plan = np.array(st["plan"])
    assert plan.shape == (4, 2) and np.isfinite(plan).all()
    full = np.array(st["extra"]["traj_full_6s"])
    assert full.shape == (8, 2) and np.isfinite(full).all()
    assert st["extra"]["vmax_valid"] == 0.0
    (tmp_path / "tick.json").write_text(json.dumps({"plan": plan.tolist(), "full": full.tolist(),
                                                    "plan_ms": st["plan_ms"], "canon": pol.canon_provenance}))
    out = os.environ.get("T2A_TICK_OUT")
    if out:
        Path(out).write_text((tmp_path / "tick.json").read_text())


@pytest.mark.skipif(os.environ.get("T2A_SLOW") != "1", reason="set T2A_SLOW=1 (CPU forwards of the real checkpoint, ~3.5 min each)")
def test_refcv7_argv_flags_reach_the_model_through_main(tmp_path):
    """REACHABILITY FROM THE ACTUAL CALLER (CLAUDE.md: 'built, tested, and unreachable from its caller'). The slow test above builds the
    policy directly; this one goes argv -> cd.main() -> EXTRA_ARMS -> RefCV7Policy -> the model call, for the levers box 2 varies.
    Reference = raw/refcv7_cpu_tick.json (seed 0, ceiling withheld, cpu, the same single stub-world tick at 15 m/s).

      off_s3  `--t2a-infer-seed 3`            -> plan MUST differ from the reference: the seed reaches the sampler.        (ASSERTED)
      log_s0  `--t2a-vmax-mode logged_2_6s`    -> v_max_valid == 1.0 and the stub log's 15.0 m/s reach the model call.      (ASSERTED)
      fixed5  `--t2a-vmax-mode fixed 5.0 m/s`  -> v_max_valid == 1.0 and 5.0 reach the model call.                          (ASSERTED)
                                                  Whether the PLAN moves is RECORDED, NOT ASSERTED -- see below.

    ⛔ WHY THE CEILING'S EFFECT ON THE PLAN IS NOT ASSERTED (a mis-specified pre-registration, kept visible). The first version asserted that
    a 5 m/s ceiling MUST change and slow the plan, with 'bit-identical' read as 'the ceiling is dead at inference'. Measured: the plan is
    bit-identical (max |diff| 0.0, peak 15.13 m/s) on BOTH the launch tree fec3a0dc and the R4 tree 0fcb1c1c (which carries the SPEC 26.1
    fix). That reading was wrong, because the pre-registration omitted the documented empty-survivor fallback: `SpeedCeilingFilter`
    (refcv6_selection.py:187-192) keeps the WHOLE fan when no candidate is compliant, and a 15 m/s world with a 8.33 m/s limit has none
    (every candidate is rolled from v0 and `planned_max_speed` includes the first segment). So 'identical' is predicted by the SPEC 26.1 defect
    AND by the fallback: it does not discriminate. A discriminating control needs a SATISFIABLE binding ceiling; none exists in the 15 m/s
    stub world (limits are 8.33 / 13.89 / 27.78 / 33.33 m/s). Evidence: raw/refcv7_argv_reach_*.json.
    `T2A_REACH_TAGS` selects a subset (default all three)."""
    ref = json.loads((PKG / "raw" / "refcv7_cpu_tick.json").read_text())
    ref_plan = np.array(ref["plan"])
    scene = ("E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430")
    base = ("--ckpt", os.environ.get("T2A_CKPT", "E:/refcv7_eval_kit/ckpt/ckpt_50400.pt"),
            "--t2a-config", os.environ.get("T2A_CONFIG", "E:/refcv7_eval_kit/ckpt/config.json"))
    table = {"off_s3": ("--t2a-vmax-mode", "withheld", "--t2a-infer-seed", "3"),
             "fixed5": ("--t2a-vmax-mode", "fixed", "--t2a-vmax-ms", "5.0", "--t2a-infer-seed", "0"),
             "log_s0": ("--t2a-vmax-mode", "logged_2_6s", "--t2a-infer-seed", "0")}
    tags = [t for t in os.environ.get("T2A_REACH_TAGS", ",".join(table)).split(",") if t]
    got = {}
    for tag in tags:
        d = tmp_path / tag
        d.mkdir()
        pay = _run_main(d, "refcv7", extra=base + table[tag], scene_dir=scene, starts="0", steps=1)
        st = pay["rollouts"][0]["steps"][0]
        got[tag] = {"plan": st["plan"], "vmax_valid": st["extra"]["vmax_valid"], "vmax_fed_ms": st["extra"]["vmax_fed_ms"],
                    "plan_peak_speed_ms": st["extra"]["plan_peak_speed_ms"], "infer_seed": pay["model_provenance"]["infer_seed"],
                    "vmax_mode": pay["model_provenance"]["vmax_mode"]}
        got[tag]["max_abs_plan_diff_vs_reference"] = float(np.abs(np.array(st["plan"]) - ref_plan).max())
        out = os.environ.get("T2A_REACH_OUT")
        if out:
            Path(out).write_text(json.dumps({"reference_plan_seed0_withheld": ref["plan"], **got}, indent=1))
    if "off_s3" in got:
        a = got["off_s3"]
        assert a["vmax_valid"] == 0.0 and a["infer_seed"] == 3
        assert a["max_abs_plan_diff_vs_reference"] > 1e-6, "the inference seed did not reach the sampler"
    if "log_s0" in got:
        b = got["log_s0"]
        assert b["vmax_valid"] == 1.0 and abs(b["vmax_fed_ms"] - 15.0) < 1e-3 and b["infer_seed"] == 0
    if "fixed5" in got:
        c = got["fixed5"]
        assert c["vmax_valid"] == 1.0 and c["vmax_fed_ms"] == 5.0
