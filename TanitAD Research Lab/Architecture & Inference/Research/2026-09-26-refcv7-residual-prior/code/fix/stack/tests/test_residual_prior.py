"""refcv7 NEW-1 -- the plan is a RESIDUAL on a causal kinematic prior (SPEC_REFCV7 section 1).

Module under test: ``tanitad/models/kinematic_prior.py`` and its wiring in ``refs/refc.py``
(decoder + ``RefCModel``), ``refs/refc_v3.py`` and ``scripts/refc_v3_train.py``.
Package: ``TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior/``.

What is pinned, by the letters of the brief:
  (a) ``P`` on ANALYTIC cases, with LITERAL expected values: a straight line at 10 m/s is
      ``x = 10 t, y = 0``; a constant yaw rate traces the forward-Euler circle whose radius and
      centre are derived here from chord geometry (``math``, not the integrator); the stop clamp.
  (b) ``P`` equals the battery's echo ``ha0_ext`` on REAL eval windows -- against the BANKED
      battery arrays and against the battery's own functions -- or a documented skip.
  (c) ``--residual-prior off`` is BIT-IDENTICAL to refcv6: a digest over every loss scalar,
      every parameter gradient and every planner output, recorded on the TIP tree (literal), and
      a mechanism test that an off build never reaches ``kinematic_prior`` at all.
  (d) With the prior on and ``Delta == 0`` the plan IS ``P`` exactly (module and model level).
  (e) MUTATION arms that hand a consumer ``Delta`` instead of ``P + Delta`` go RED.
CPU only; no GPU, no checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve().parent            # <repo>/stack/tests
_STACK = _HERE.parent
_REPO = _STACK.parent
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import kinematic_prior as KP  # noqa: E402
from tanitad.refs import refc_sampler as rs  # noqa: E402

HZ = (5, 10, 15, 20, 30, 40, 50, 60)               # V3_HORIZONS, literal
F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
         "--f5-emitting-conf", "--f6-w-u0-zero"]
#: 5 x 4 alat grid INCLUDING (0, 0) at index 10 -- the F3 reach test's ladder.
A_LON = (-2.0, -1.0, 0.0, 1.0, 2.0)
A_LAT = (-1.5, -0.5, 0.0, 0.5)
ZERO_IDX = 10


def _grid() -> "torch.Tensor":
    return torch.cartesian_prod(torch.tensor(A_LON), torch.tensor(A_LAT))


# =========================================================================== #
# (a) analytic cases -- literals                                              #
# =========================================================================== #
def test_a_straight_at_10_mps_is_x_equals_10t_y_zero() -> None:
    p = KP.prior_path(torch.zeros(1), torch.zeros(1), torch.tensor([10.0]), HZ)[0]
    want_x = torch.tensor([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert torch.equal(p[:, 0], want_x), p[:, 0]
    assert torch.equal(p[:, 1], torch.zeros(8)), p[:, 1]


def test_a_constant_yaw_rate_traces_the_euler_circle_of_radius_v_over_omega() -> None:
    """v = 10 m/s, kappa = 0.02 1/m => omega = 0.2 rad/s, R = v/omega = 50 m. Forward Euler at
    dt = 0.1 moves along equal CHORDS (s = v dt) turning by omega dt each step, so its points lie
    on the circumcircle of that polygon: radius v dt / (2 sin(omega dt / 2)) = 50.00083334305565,
    centre (v dt / 2, r cos(omega dt / 2)) = (0.5, 49.99833332222211). Literals, from geometry."""
    v, k = 10.0, 0.02
    r_d, cx, cy = 50.00083334305565, 0.5, 49.99833332222211
    assert abs(r_d - (v * 0.1) / (2 * math.sin(v * k * 0.1 / 2))) < 1e-9     # the derivation
    assert abs(r_d - v / (v * k)) < 1e-3                                      # ~ v / omega
    p = KP.prior_path(torch.zeros(1), torch.tensor([k]), torch.tensor([v]),
                      tuple(range(1, 61)))[0].double()
    d = torch.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2)
    assert float((d - r_d).abs().max()) < 1e-3, float((d - r_d).abs().max())
    # and it is NOT the continuous circle centred at (0, R): the O(dt) offset is real
    d_cont = torch.sqrt(p[:, 0] ** 2 + (p[:, 1] - 50.0) ** 2)
    assert float((d_cont - 50.0).abs().max()) > 0.1


def test_a_deceleration_stops_the_car_it_never_reverses() -> None:
    """a0 = -2 m/s^2 from 3 m/s: 15 Euler ticks of v = 3.0, 2.8, ..., 0.2 then v = 0.
    x = 0.1 * sum(v) = 1.3 m (5 ticks), 2.1 m (10), 2.4 m (15 and after)."""
    p = KP.prior_path(torch.tensor([-2.0]), torch.zeros(1), torch.tensor([3.0]), HZ)[0]
    want = torch.tensor([1.3, 2.1, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4])
    assert float((p[:, 0] - want).abs().max()) < 1e-5, p[:, 0]
    assert float(p[:, 1].abs().max()) == 0.0


def _window(a: float = 1.5, omega: float = 0.3, v_start: float = 5.0, kappa_true: float = 0.04,
            n: int = 8) -> tuple:
    """A synthetic OBSERVED window: speed ramps at ``a``, heading turns at ``omega``, and the
    recorded steer encodes ``kappa_true`` exactly the way physicalai.py:621 writes it."""
    t = torch.arange(n, dtype=torch.float32)
    v = v_start + a * 0.1 * t
    yaw = omega * 0.1 * t
    poses = torch.stack([t, torch.zeros(n), yaw, v], dim=-1)[None]          # [1, n, 4]
    steer = torch.full((n,), math.atan(2.9 * kappa_true), dtype=torch.float32)
    actions = torch.stack([steer, torch.full((n,), 9.9)], dim=-1)[None]     # accel col unread
    return poses, actions


def test_a_prior_controls_read_the_window_the_way_each_mode_says() -> None:
    poses, actions = _window()
    v0 = float(poses[0, -1, 3])                                             # 6.05
    a0, k0 = KP.prior_controls("ha0_ext_pose", poses)
    assert abs(float(a0) - 1.5) < 1e-4 and abs(float(k0) - 0.3 / v0) < 1e-4
    a0c, k0c = KP.prior_controls("cv_yawrate", poses)
    assert float(a0c) == 0.0 and torch.equal(k0c, k0)
    a0e, k0e = KP.prior_controls("ha0_ext", poses, actions=actions)
    assert torch.equal(a0e, a0)                    # the SAME backward difference
    assert abs(float(k0e) - 0.04) < 1e-6           # the recorded curvature, not the pose one
    # the accel COLUMN of the actions is never read (ha0_ext's a0 is the speed difference)
    assert float(a0e) != 9.9


def test_a_pose_curvature_floor_and_cap_are_the_declared_literals() -> None:
    assert KP.POSE_KAPPA_V_FLOOR == 2.0 and KP.POSE_KAPPA_CAP == 0.3
    slow, _ = _window(a=0.0, omega=0.2, v_start=0.5)          # v0 = 0.5 < floor
    assert abs(float(KP.prior_controls("ha0_ext_pose", slow)[1]) - 0.2 / 2.0) < 1e-5
    sharp, _ = _window(a=0.0, omega=1.0, v_start=3.0)         # 1.0 / 3.0 > cap
    assert float(KP.prior_controls("ha0_ext_pose", sharp)[1]) == pytest.approx(0.3)


def test_a_every_mode_refuses_the_input_it_does_not_read() -> None:
    poses, actions = _window()
    with pytest.raises(KP.ResidualPriorError, match="RECORDED STEER"):
        KP.prior_controls("ha0_ext", poses)
    for m in ("ha0_ext_pose", "cv_yawrate"):
        with pytest.raises(KP.ResidualPriorError, match="silently dropped"):
            KP.prior_controls(m, poses, actions=actions)
    with pytest.raises(KP.ResidualPriorError):
        KP.prior_controls("off", poses)
    with pytest.raises(KP.ResidualPriorError):
        KP.check_mode("cv")                        # no fuzzy match, no default


def test_a_the_prior_reads_nothing_after_t0() -> None:
    """Mutate every step AFTER n_past; the prior must be bit-identical (the ego_history rule)."""
    poses, actions = _window(n=10)
    p2, a2 = poses.clone(), actions.clone()
    p2[:, 8:] = 1e6
    a2[:, 8:] = 0.7
    for m in ("ha0_ext_pose", "cv_yawrate"):
        assert all(torch.equal(x, y) for x, y in zip(KP.prior_controls(m, poses, 8),
                                                     KP.prior_controls(m, p2, 8)))
    assert all(torch.equal(x, y) for x, y in
               zip(KP.prior_controls("ha0_ext", poses, 8, actions),
                   KP.prior_controls("ha0_ext", p2, 8, a2)))


# =========================================================================== #
# (d) Delta = 0 IS the prior -- module level                                  #
# =========================================================================== #
@pytest.mark.parametrize("units", ["alat", "kappa"])
def test_d_zero_residual_rolls_to_the_prior_bit_for_bit(units: str) -> None:
    a0 = torch.tensor([0.0, 1.2, -2.5, 0.4])
    k0 = torch.tensor([0.0, 0.05, -0.02, 0.25])            # 0.25 > the vocabulary cap 0.12
    v = torch.tensor([10.0, 3.0, 20.0, 1.0])
    plan = KP.roll_plan(torch.zeros(4, 7, 8, 2), a0, k0, v, HZ, control_units=units)
    p = KP.prior_path(a0, k0, v, HZ)
    for i in range(7):
        assert torch.equal(plan[:, i], p), (units, i)
    # the prior is NOT clipped by the vocabulary's kappa_cap: row 3 turns at 0.25, and a prior
    # squeezed through alat_to_curvature would turn at 0.12
    clipped = KP.prior_path(a0[3:], torch.tensor([0.12]), v[3:], HZ)
    assert float((p[3] - clipped[0]).abs().max()) > 0.05


def test_d_a_zero_prior_is_the_vocabularys_own_roll() -> None:
    """P = 0 reduces roll_plan to refc_sampler.roll_controls (the refcv6 fan), bit for bit."""
    torch.manual_seed(0)
    u = torch.randn(3, 5, 8, 2)
    v = torch.tensor([0.5, 8.0, 27.0])
    z = torch.zeros(3)
    for units in ("alat", "kappa"):
        assert torch.equal(KP.roll_plan(u, z, z, v, HZ, control_units=units),
                           rs.roll_controls(u, v, HZ, control_units=units))


def test_d_absolute_and_residual_controls_are_inverse() -> None:
    torch.manual_seed(1)
    d = torch.randn(3, 5, 8, 2)
    a0, k0, v = torch.randn(3), 0.05 * torch.randn(3), torch.tensor([2.0, 9.0, 30.0])
    for units in ("alat", "kappa"):
        back = KP.residual_controls(KP.absolute_controls(d, a0, k0, v, control_units=units),
                                    a0, k0, v, control_units=units)
        assert float((back - d).abs().max()) < 1e-3


def test_d_withheld_rows_get_the_no_information_prior() -> None:
    a0, k0 = torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.2])
    a, k = KP.withhold(a0, k0, torch.tensor([True, False]))
    assert a.tolist() == [1.0, 0.0] and k.tolist() == [pytest.approx(0.1), 0.0]


# =========================================================================== #
# (b) P == the battery's ha0_ext on REAL windows                              #
# =========================================================================== #
DUMP = os.environ.get("REFCV7_ECHO_DUMP", "C:/Users/Admin/ev6_battery/raw/step30000/dump_s0")
CACHE = os.environ.get("REFCV7_ECHO_CACHE",
                       "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt")
N_EP = 5


def _real_windows(need_dump: bool):
    if not os.path.isfile(CACHE):
        pytest.skip(f"eval kit not reachable: {CACHE} (set REFCV7_ECHO_CACHE); the analytic "
                    f"tests above still pin P")
    if need_dump and not os.path.isfile(os.path.join(DUMP, "manifest.json")):
        pytest.skip(f"banked battery dump not reachable: {DUMP} (set REFCV7_ECHO_DUMP)")
    import numpy as np
    cm = torch.load(CACHE, map_location="cpu", weights_only=False)
    by_clip = {str(c): i for i, c in enumerate(cm["clip_id"])}
    rows = []
    if need_dump:
        man = json.load(open(os.path.join(DUMP, "manifest.json"), encoding="utf-8"))
        W = int(man["grid"]["obs_window"])
        for e in man["episodes"][:N_EP]:
            z = np.load(os.path.join(DUMP, f"ep{int(e['file_index']):03d}.npz"))
            rows.append((by_clip[str(e["clip_id"])], torch.as_tensor(z["ws"]).long(),
                         torch.as_tensor(z["v0"]).float(), torch.as_tensor(z["ha0_ext"]).float()))
    else:
        W = 8
        for ci in range(N_EP):
            T_ = int(cm["poses"][ci].shape[0])
            ws = torch.arange(W - 1, T_ - 20, 7)
            rows.append((ci, ws, cm["poses"][ci][ws, 3].float(), None))
    out = []
    for ci, ws, v0, echo in rows:
        poses, acts = cm["poses"][ci].float(), cm["actions"][ci].float()
        idx = ws[:, None] - (W - 1) + torch.arange(W)[None]
        out.append((poses, acts, ws, poses[idx], acts[idx], v0, echo, W))
    return out


def test_b_P_equals_the_BANKED_battery_echo_on_real_windows() -> None:
    """MEASURED on the full surface (4,754 windows / 139 clips, step-30k dump): max |P - echo|
    = 0.0 m on every window. Here: the first 5 clips, same assertion, with a control that must
    DIFFER (the pose-curvature prior is a different prior)."""
    n = 0
    diff_pose = 0.0
    for poses, acts, ws, ph, ah, v0, echo, W in _real_windows(need_dump=True):
        assert torch.equal(ph[:, -1, 3], v0), "the window does not end at the banked v0"
        a0, k0 = KP.prior_controls("ha0_ext", ph, W, ah)
        p = KP.prior_path(a0, k0, v0, HZ)[:, :4]                 # the 2 s grid = slots 0..3
        assert float((p - echo).abs().max()) <= 1e-5, float((p - echo).abs().max())
        a1, k1 = KP.prior_controls("ha0_ext_pose", ph, W)
        diff_pose = max(diff_pose, float((KP.prior_path(a1, k1, v0, HZ)[:, :4] - echo)
                                         .abs().max()))
        n += int(ws.shape[0])
    assert n >= 50, n
    assert diff_pose > 1e-3, "control: the POSE prior read the same as the echo -- unpowered"


def test_b_P_equals_the_battery_FUNCTIONS_on_real_windows() -> None:
    """The battery's own call chain (refcv3_arm.py:2125-2128), per window, on real clips."""
    tools = _REPO / "taniteval" / "tools"
    if not (tools / "refcv3_arm.py").is_file():
        pytest.skip(f"taniteval tools not in this tree: {tools}")
    spec = importlib.util.spec_from_file_location("refcv3_arm_rp_test", tools / "refcv3_arm.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    grid = rc.grid_slots(HZ, "2s")
    worst, n = 0.0, 0
    for poses, acts, ws, ph, ah, v0, _e, W in _real_windows(need_dump=False):
        a0, k0 = KP.prior_controls("ha0_ext", ph, W, ah)
        p = KP.prior_path(a0, k0, v0, HZ)[:, grid["slots"]]
        for j, t0 in enumerate(ws.tolist()):
            ext = rc.ra.hold_ext_controls(None, poses[:, 3], acts[:, 0], int(t0), dt=0.1,
                                          stride=1)
            hx = rc.integrate_select(ext[None].expand(grid["n_frames"], 2), float(v0[j]), grid,
                                     action_units="steer")
            worst = max(worst, float((torch.as_tensor(hx[0]) - p[j]).abs().max()))
            n += 1
    assert n >= 50 and worst <= 1e-5, (n, worst)


# =========================================================================== #
# the model: the trainer's OWN build + loss (smoke width)                     #
# =========================================================================== #
@pytest.fixture(scope="module")
def T():
    path = _STACK / "scripts" / "refc_v3_train.py"
    spec = importlib.util.spec_from_file_location("refc_v3_train_rp_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(T, mode: str | None, extra: list | None = None, seed: int = 0):
    argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
            "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
            "--out", str(_REPO / "_rp_test_unused_out")] + F1_F6 + list(extra or [])
    if mode is not None:
        argv += ["--residual-prior", mode]
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(seed)
    model = T.v3.RefCV3Model(cfg)
    with torch.no_grad():
        model.core.decoder.anchor_controls.copy_(_grid())
    return args, cfg, model


def _batch(T, cfg, turning: bool = True):
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    b = torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])
    if turning:
        # a window that ACCELERATES and TURNS, so the prior is far from zero on every row
        W = int(b["pose_hist"].shape[1])
        t = torch.arange(W, dtype=torch.float32)
        for r, (v_s, a, om) in enumerate([(6.0, 1.0, 0.2), (12.0, -1.5, -0.1),
                                          (3.0, 0.5, 0.3), (20.0, 0.0, 0.05)]):
            b["pose_hist"][r, :, 3] = v_s + a * 0.1 * t
            b["pose_hist"][r, :, 2] = om * 0.1 * t
            b["actions"][r, :, 0] = math.atan(2.9 * om / (v_s + a * 0.1 * (W - 1)))
        b["pose_last"] = b["pose_hist"][:, -1].clone()
    return b


def _forward(T, cfg, model, batch, mode: str | None, seed: int = 2):
    model.eval()
    torch.manual_seed(seed)
    ph = batch["pose_hist"]
    acts = batch["actions"] if (mode and mode != "off" and KP.needs_actions(mode)) else None
    with torch.no_grad():
        model.core.set_ego_window(ph, int(ph.shape[1]), actions=acts)
        return model(T.frames_to_device(batch["frames"], "cpu"), nav_cmd=batch["nav_cmd"],
                     v0=batch["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)


# ---- (c) off is refcv6, bit for bit -------------------------------------- #
#: MEASURED 2026-09-26 by ``code/offmode_digest.py`` on a clean ``git archive`` of the TIP
#: (59f0d46; the stack is byte-identical at 5de9363) -- i.e. refcv6's own code, which has no
#: residual-prior flag at all -- and reproduced by the candidate tree with the flag at its
#: default. REPRODUCED on the landed fixes batch ab436ee (tip AND tip + NEW-1), so the fixes
#: did not move the refcv6 smoke path. One intra-op thread (the digest is thread-count
#: dependent otherwise, MEASURED). Re-record with that script on the tip whenever the refcv6
#: smoke path itself changes.
TIP_OFF_DIGEST = "ebc4db4599e6c3f0fcdd9e077113a5ac789dfeff19349f391b65057ef3064ef3"
DIGEST_PLATFORM = ("2.11.0+cu128", "win32")


def _off_digest(extra_argv: list) -> dict:
    """``code/offmode_digest.py::build_and_digest``, VERBATIM in substance (the package script
    is the cross-tree instrument that ran on the tip; this copy runs in any tree). A fresh
    trainer module per call, one intra-op thread, three fixed seeds."""
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec = importlib.util.spec_from_file_location(
            "rv3t_digest_rp_test", _STACK / "scripts" / "refc_v3_train.py")
        T = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(T)
        argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
                "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
                "--out", str(_REPO / "_digest_unused_out")] + F1_F6 + list(extra_argv)
        args = T.build_parser().parse_args(argv)
        cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
        torch.manual_seed(0)
        model = T.v3.RefCV3Model(cfg)
        with torch.no_grad():
            model.core.decoder.anchor_controls.copy_(_grid())
        eps = T._synth_episodes(2, cfg.core, seed=0)
        ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                         channels=cfg.core.encoder.in_channels)
        ds.ego_history = True
        batch = torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])
        h = hashlib.sha256()

        def feed(name, t):
            h.update(name.encode())
            if t is None:
                h.update(b"<None>")
                return
            if isinstance(t, (list, tuple)):
                for i, x in enumerate(t):
                    feed(f"{name}[{i}]", x)
                return
            x = t.detach().cpu().contiguous()
            h.update(str(tuple(x.shape)).encode() + str(x.dtype).encode())
            h.update(x.numpy().tobytes())

        model.train()
        torch.manual_seed(1)
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
        for k in sorted(losses):
            v = losses[k]
            if torch.is_tensor(v) and v.dim() == 0:
                feed(f"loss:{k}", v)
        losses["loss"].backward()
        for name, p in sorted(model.named_parameters()):
            if p.grad is not None:
                feed(f"grad:{name}", p.grad)
        model.zero_grad(set_to_none=True)
        model.eval()
        torch.manual_seed(2)
        with torch.no_grad():
            ph = batch["pose_hist"]
            model.core.set_ego_window(ph, int(ph.shape[1]))
            out = model(T.frames_to_device(batch["frames"], "cpu"), nav_cmd=batch["nav_cmd"],
                        v0=batch["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)
        for k in ("anchor_bank", "anchor_traj", "traj", "sel_idx", "anchor_logits",
                  "refined_logits", "sel_score", "u0_hat", "layer_u0_hat", "layer_logits"):
            feed(f"out:{k}", out.get(k))
        return {"digest": h.hexdigest(),
                "n_params": sum(p.numel() for p in model.parameters()),
                "keys": sorted(model.state_dict().keys()),
                "residual_keys_in_out": sorted(k for k in out if k.startswith("residual_prior"))}
    finally:
        torch.set_num_threads(nt)


def test_c_off_is_BIT_IDENTICAL_to_refcv6_digest_recorded_on_the_tip() -> None:
    if (torch.__version__, sys.platform) != DIGEST_PLATFORM:
        pytest.skip(f"the literal digest was recorded on torch {DIGEST_PLATFORM[0]} / "
                    f"{DIGEST_PLATFORM[1]}; this is {torch.__version__} / {sys.platform} "
                    f"({platform.machine()}) -- float kernels differ across builds. The "
                    f"mechanism tests below still run.")
    for extra in ([], ["--residual-prior", "off"]):
        got = _off_digest(extra)
        assert got["residual_keys_in_out"] == []
        assert got["digest"] == TIP_OFF_DIGEST, (extra, got["digest"])
    # the DISCRIMINATING control: a residual build moves the digest (and adds NO parameter)
    on = _off_digest(["--residual-prior", "ha0_ext_pose"])
    assert on["digest"] != TIP_OFF_DIGEST
    assert on["n_params"] == got["n_params"] and on["keys"] == got["keys"]


def test_c_an_OFF_build_never_reaches_the_prior_module(T, monkeypatch) -> None:
    """Every kinematic_prior entry the decoder can call is made to RAISE. The off build must
    train and run untouched; the SAME patch on a residual build must fail (discriminating)."""
    from tanitad.refs import refc

    def boom(*a, **k):
        raise AssertionError("an OFF build reached kinematic_prior")
    for name in ("roll_plan", "prior_path", "absolute_controls", "residual_controls",
                 "prior_controls", "withhold"):
        monkeypatch.setattr(refc._kp, name, boom)
    args, cfg, model = _arm(T, None)
    b = _batch(T, cfg)
    model.train()
    losses = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    losses["loss"].backward()
    out = _forward(T, cfg, model, b, None)
    assert not any(k in out for k in KP.OUT_KEYS)
    _a, cfg2, model2 = _arm(T, "ha0_ext_pose")
    with pytest.raises(AssertionError, match="OFF build reached"):
        _forward(T, cfg2, model2, _batch(T, cfg2), "ha0_ext_pose")


def test_c_a_ZERO_prior_reproduces_the_off_build_when_no_row_is_withheld(T, monkeypatch) -> None:
    """The composition is refcv6's own integration when P = 0: a residual build whose prior is
    forced to zero emits EXACTLY the off build's bank, fan, pick and controls (eval: no row is
    withheld). Same seed, same weights."""
    _a, cfg, off = _arm(T, None)
    b = _batch(T, cfg)
    o_off = _forward(T, cfg, off, b, None)
    _a, cfg2, on = _arm(T, "ha0_ext_pose")
    on.load_state_dict(off.state_dict())
    monkeypatch.setattr(KP, "prior_controls", lambda m, p, n=None, a=None, **k:
                        (torch.zeros(p.shape[0]), torch.zeros(p.shape[0])))
    o_on = _forward(T, cfg2, on, b, "ha0_ext_pose")
    for k in ("anchor_bank", "anchor_traj", "traj", "sel_idx", "u0_hat", "anchor_logits"):
        assert torch.equal(o_off[k], o_on[k]), k
    for x, y in zip(o_off["layer_u0_hat"], o_on["layer_u0_hat"]):
        assert torch.equal(x, y)


# ---- (d) + the consumer check on the REAL model output ------------------- #
@pytest.mark.parametrize("mode", ["ha0_ext", "ha0_ext_pose", "cv_yawrate"])
def test_d_the_model_emits_P_plus_Delta_and_the_zero_anchor_IS_P(T, mode: str) -> None:
    _a, cfg, model = _arm(T, mode)
    b = _batch(T, cfg)
    out = _forward(T, cfg, model, b, mode)
    # the keys reach the MODEL-level output (RefCV3Model), not only the decoder (the A16 lesson)
    assert all(k in out for k in KP.OUT_KEYS), sorted(out)
    rep = KP.plan_check(out, model.core.decoder.anchor_controls, decoder=model.core.decoder)
    assert rep["ok"], rep
    assert rep["zero_residual_anchor"] == ZERO_IDX
    assert rep["max_abs_fan_minus_reroll_m"] < 1e-3                  # the FAN is P + Delta too
    assert rep["max_abs_zero_anchor_minus_prior_m"] == 0.0            # bit-exact, not approx
    assert rep["rows_v_gt_0_prior_nonzero"] == rep["rows_v_gt_0"] == 4  # the G-LIVE clause
    # every row's prior really is the declared (a0, kappa0) rolled from the forward's v0
    a0, k0 = KP.prior_controls(mode, b["pose_hist"], None,
                               b["actions"] if KP.needs_actions(mode) else None)
    assert torch.equal(out["residual_prior_ctrl"][:, 0], a0)
    assert torch.equal(out["residual_prior_v"], b["pose_last"][:, 3])
    if mode == "cv_yawrate":
        assert float(out["residual_prior_ctrl"][:, 0].abs().max()) == 0.0


def test_d_the_trainers_OWN_loss_runs_with_every_term_live(T) -> None:
    args, cfg, model = _arm(T, "ha0_ext")
    b = _batch(T, cfg)
    model.train()
    losses = T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    for k in ("loss", "traj", "cascade"):
        assert k in losses and math.isfinite(float(losses[k].detach())), k
    losses["loss"].backward()
    # the emitted fan's head. ⚠️ Under F3 it is the CASCADE's last stage, NOT
    # `decoder.control_head`, which receives no gradient at all on the refcv6 flag set --
    # MEASURED on the TIP with the prior absent (`p.grad is None`), so it is a refcv6
    # finding, not a NEW-1 one (package RESULT.md, "findings").
    heads = model.core.decoder.cascade.control_heads
    for i, h in enumerate(heads):
        g = h.weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0, i
    # the run record names the prior, and the built decoder agrees with it (G-DVB)
    stamp = T._seam_stamp(cfg, args)
    assert stamp["residual_prior"]["residual_prior"] == "ha0_ext"
    assert stamp["residual_prior"]["equals_battery_echo"] is True
    assert model.core.decoder.residual_prior == args.residual_prior == "ha0_ext"
    T.assert_seams_are_built(model, stamp)
    bad = dict(stamp, residual_prior=KP.prior_stamp("off"))
    with pytest.raises((SystemExit, ValueError)):
        T.assert_seams_are_built(model, bad)


def test_d_a_residual_checkpoint_reloads_STRICTLY_and_replays_bit_identically(T) -> None:
    """G-EVAL / G-CKPT's NEW-1 slice: the prior adds NO state, so a residual checkpoint loads
    strictly (0 missing / 0 unexpected) into a model REBUILT from the same argv (the eval
    loaders' route: build_parser + _pin_trainer_cfg), and the rebuilt model emits the SAME plan
    and the SAME prior on a fixed batch and seed."""
    import io
    _a, cfg, trained = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    with torch.no_grad():                          # make the weights non-trivial
        for p in trained.parameters():
            p.add_(0.01 * torch.randn_like(p))
    buf = io.BytesIO()
    torch.save(trained.state_dict(), buf)
    buf.seek(0)
    _a2, cfg2, rebuilt = _arm(T, "ha0_ext_pose", seed=123)       # different init
    res = rebuilt.load_state_dict(torch.load(buf, weights_only=False), strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    o1 = _forward(T, cfg, trained, b, "ha0_ext_pose")
    o2 = _forward(T, cfg2, rebuilt, b, "ha0_ext_pose")
    for k in ("traj", "anchor_traj", "anchor_bank", "u0_hat", *KP.OUT_KEYS):
        assert torch.equal(o1[k], o2[k]), k


def test_d_withheld_rows_compose_on_the_no_information_prior_at_the_banks_speed(T) -> None:
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    dec = model.core.decoder
    a0, k0 = torch.tensor([1.0, 1.0]), torch.tensor([0.05, 0.05])
    keep = torch.tensor([True, False])
    rp = dec._residual_prior((a0, k0), torch.tensor([7.0, 7.0]), keep, 2, None)
    assert rp[0].tolist() == [1.0, 0.0] and rp[1].tolist() == [pytest.approx(0.05), 0.0]
    assert rp[2].tolist() == [7.0, 10.0]                   # withheld row at ref_speed_ms
    bank = dec.roll_bank(torch.tensor([7.0, 7.0]), keep, 2, torch.float32, prior=rp)
    # the withheld row's (0, 0) anchor is the straight line at 10 m/s: x = 10 t, literal
    assert torch.equal(bank[1, ZERO_IDX, :, 0], torch.tensor(
        [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
    assert torch.equal(bank[1, ZERO_IDX, :, 1], torch.zeros(8))


def test_d_TRAINING_forwards_with_withheld_rows_still_emit_P_plus_Delta(T) -> None:
    """Ego-dropout (0.5 on the refcv6 flag set) withholds rows in TRAINING. A withheld row's
    bank is rolled at the reference speed; its prior must be zero and its SAMPLED fan must be
    rolled at that same speed -- the refcv6 sampler rolls it at the raw v0 instead
    (`refc.py` `_sample`), which a residual on a prior cannot tolerate. The consumer check
    re-rolls the exported controls on every row, withheld ones included."""
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    dec = model.core.decoder
    model.train()
    seen_withheld = 0
    for seed in range(6):
        torch.manual_seed(seed)
        ph = b["pose_hist"]
        model.core.set_ego_window(ph, int(ph.shape[1]))
        with torch.no_grad():
            out = model(T.frames_to_device(b["frames"], "cpu"), nav_cmd=b["nav_cmd"],
                        v0=b["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)
        rep = KP.plan_check(out, dec.anchor_controls, decoder=dec)
        assert rep["ok"], (seed, rep)
        withheld = out["residual_prior_v"] != b["pose_last"][:, 3]
        if bool(withheld.any()):
            seen_withheld += int(withheld.sum())
            assert torch.equal(out["residual_prior_v"][withheld],
                               torch.full((int(withheld.sum()),), 10.0))
            assert float(out["residual_prior_ctrl"][withheld].abs().max()) == 0.0
    assert seen_withheld > 0, "no row was withheld in 6 seeds -- the probe is unpowered"


def test_gdvb_entry_reads_the_BUILT_prior_and_goes_RED_when_it_is_lost(T) -> None:
    """SPEC_REFCV7 section 2, G-DVB: every argv lever equals the BUILT model's value. The entry
    lives in the fixes batch's registry (`tanitad/train/declared_vs_built.py`); this test runs
    once that module exists and FAILS -- not skips -- if the registry lacks the entry."""
    dvb = pytest.importorskip("tanitad.train.declared_vs_built",
                              reason="G-DVB lands with the fixes batch (FIX-3/4/5)")
    assert "residual_prior" in dvb.REGISTRY, (
        "G-DVB has no `residual_prior` entry: a trainer flag without one is refused "
        "(SPEC_REFCV7 section 2) -- apply the package's apply_residual_prior_edits.py")
    chk = dvb.REGISTRY["residual_prior"].check
    args, _cfg, model = _arm(T, "ha0_ext_pose")
    assert chk(model, args) == []
    model.core.decoder.residual_prior = "off"               # REGRESSION ARM: lost in the build
    bad = chk(model, args)
    assert bad and "residual-prior" in str(bad[0]), bad
    args0, _c0, model0 = _arm(T, None)                      # an off launch reads clean
    assert chk(model0, args0) == []


def test_passthrough_carries_the_prior_and_a_dropped_key_goes_RED(T, monkeypatch) -> None:
    """`RefCModel.DECODER_PASSTHROUGH` is the ONE tuple the forward copies and G-DVB reads. NEW-1's
    three keys must be in it: the trainer's F3 cascade re-roll READS two of them. REGRESSION ARM:
    drop `residual_prior_v` -> the trainer's own loss REFUSES (a partial key set, the A16 class),
    and G-DVB reports the missing key."""
    from tanitad.refs import refc
    assert set(KP.OUT_KEYS) == {"residual_prior_ctrl", "residual_prior_v",
                                "residual_prior_path"}                       # literal
    assert set(KP.OUT_KEYS) <= set(refc.RefCModel.DECODER_PASSTHROUGH)
    args, cfg, model = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    model.train()
    T.compute_losses_v3(model, b, "cpu", mode="diffusion")["loss"].backward()   # green control
    dropped = tuple(k for k in refc.RefCModel.DECODER_PASSTHROUGH if k != "residual_prior_v")
    monkeypatch.setattr(refc.RefCModel, "DECODER_PASSTHROUGH", dropped)
    with pytest.raises(KP.ResidualPriorError, match="pass-through dropped"):
        T.compute_losses_v3(model, b, "cpu", mode="diffusion")
    dvb = pytest.importorskip("tanitad.train.declared_vs_built",
                              reason="G-DVB lands with the fixes batch (FIX-3/4/5)")
    bad = dvb.REGISTRY["residual_prior"].check(model, args)
    assert any("residual_prior_v" in str(x) for x in bad), bad


# ---- refusals: every input the prior needs, and every silent drop -------- #
def test_refusals_at_build_and_at_forward(T) -> None:
    with pytest.raises(SystemExit, match="ego-history"):
        args = T.build_parser().parse_args(
            ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
             "--residual-prior", "ha0_ext", "--out", "x"])
        T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    with pytest.raises(ValueError, match="control-space DDIM"):
        T.v3.RefCV3Model(_cfg_without(T, "--sampler"))
    _a, cfg, model = _arm(T, "ha0_ext")
    b = _batch(T, cfg)
    with pytest.raises(KP.ResidualPriorError, match="RECORDED STEER"):
        _forward(T, cfg, model, b, "ha0_ext_pose")          # actions withheld on an ha0_ext build
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    with pytest.raises(ValueError, match="SILENTLY DROPPED"):
        _forward(T, cfg, model, b, "ha0_ext")               # actions handed to a pose build
    b2 = dict(b)
    b2["pose_last"] = b["pose_last"].clone()
    b2["pose_last"][:, 3] += 1.0                            # a window ending at another frame
    with pytest.raises(ValueError, match="wrong t0"):
        _forward(T, cfg, model, b2, "ha0_ext_pose")


def _cfg_without(T, flag: str):
    argv = ["--arm", "hier", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
            "--n-anchors", "20", "--ego-history", "--residual-prior", "ha0_ext_pose",
            "--out", "x"]
    if flag != "--sampler":
        argv += ["--sampler", "ddim"]
    return T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), T.build_parser().parse_args(argv))


# ---- the RL channel contract: `ego_actions` is DECLARED by the seam that owns it --------- #
# The Thor full-suite gate of 2026-09-27 went RED in six RL test files on the undeclared
# channel -- correctly: both adapters derive their channel set from the live signatures and
# REFUSE an undeclared one. The declaration lives in `kinematic_prior.FORWARD_EXCLUSIONS`.
def _rl_required_minus_plumbed():
    """``refc_adapter``'s own comparison: signature - declared exclusions - FORWARD_KEYS."""
    import inspect

    from tanitad.channel_admissibility import excluded_channels
    from tanitad.refs.refc_v3 import RefCV3Model
    from tanitad.rl import refc_adapter as ra
    sig = set(inspect.signature(RefCV3Model.forward).parameters) - {"self", "frames", "steps"}
    return (sig - set(excluded_channels())) - set(ra.FORWARD_KEYS)


def test_rl_ego_actions_is_declared_must_not_be_plumbed_by_the_seam_that_owns_it() -> None:
    import tanitad.channel_admissibility as ca
    from tanitad.rl import refc_adapter as ra
    assert "tanitad.models.kinematic_prior" in ca.SEAM_MODULES
    d = ca.exclusion_by_channel()["ego_actions"]
    assert d == KP.FORWARD_EXCLUSIONS[0] and "kinematic_prior" in d.owner
    assert d.permanent is False                        # TEMPORARY: a PI ruling lifts it
    assert "ha0_ext" in d.reason and "ha0_ext" in d.unblock
    assert KP.ACTION_MODES == ("ha0_ext",)             # the one mode the reason names
    assert "ego_actions" not in ra.FORWARD_KEYS        # honoured, not merely declared
    assert _rl_required_minus_plumbed() == set()


def test_rl_MUTATION_without_the_declaration_BOTH_adapters_go_RED(monkeypatch) -> None:
    """Remove this seam from SEAM_MODULES: `refc_adapter`'s two-way comparison is blind to
    `ego_actions` and `refcv3_adapter` REFUSES to build its contract -- the two failures the
    Thor gate reported. The CONTROL (declaration restored) is quiet on both."""
    import types

    import tanitad.channel_admissibility as ca
    from tanitad.refs import refc
    from tanitad.rl import refc_adapter as ra
    from tanitad.rl import refcv3_adapter as ad
    pilot = types.SimpleNamespace(forward=refc.RefCModel.forward)
    monkeypatch.setattr(ca, "SEAM_MODULES", tuple(
        m for m in ca.SEAM_MODULES if m != "tanitad.models.kinematic_prior"))
    assert _rl_required_minus_plumbed() == {"ego_actions"}
    with pytest.raises(ra.RequirementDeclarationError, match="ego_actions"):
        ad.forward_conditioning_channels(pilot)
    monkeypatch.undo()
    assert _rl_required_minus_plumbed() == set()
    assert "ego_actions" not in ad.forward_conditioning_channels(pilot)


# =========================================================================== #
# (e) MUTATIONS -- a consumer handed Delta instead of P + Delta must go RED   #
# =========================================================================== #
def test_e_MUTATION_a_roll_that_drops_the_prior_is_caught(T, monkeypatch) -> None:
    from tanitad.refs import refc
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    dec = model.core.decoder
    good = KP.plan_check(_forward(T, cfg, model, b, "ha0_ext_pose"), dec.anchor_controls,
                         decoder=dec)
    assert good["ok"], good
    real = refc._kp.roll_plan

    def delta_only(delta, a0, k0, v, *a, **k):        # the defect: Delta emitted as the plan
        return real(delta, torch.zeros_like(a0), torch.zeros_like(k0), v, *a, **k)
    monkeypatch.setattr(refc._kp, "roll_plan", delta_only)
    bad = KP.plan_check(_forward(T, cfg, model, b, "ha0_ext_pose"), dec.anchor_controls,
                        decoder=dec)
    assert not bad["ok"], bad                           # RED
    assert bad["max_abs_zero_anchor_minus_prior_m"] > 0.5


def test_e_MUTATION_a_FAN_rolled_without_the_prior_is_caught(T, monkeypatch) -> None:
    """Only the SAMPLED fan loses the prior (the bank keeps it): check (1) alone would pass,
    so the consumer check re-rolls the exported controls -- (2) must go RED."""
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    dec = model.core.decoder
    real = type(dec)._roll_state

    def fan_without_prior(self, z, v, metre, prior):
        if prior is None:
            return real(self, z, v, metre, prior)
        a0, k0, pv = prior
        return real(self, z, v, metre, (torch.zeros_like(a0), torch.zeros_like(k0), pv))
    monkeypatch.setattr(type(dec), "_roll_state", fan_without_prior)
    bad = KP.plan_check(_forward(T, cfg, model, b, "ha0_ext_pose"), dec.anchor_controls,
                        decoder=dec)
    assert bad["max_abs_zero_anchor_minus_prior_m"] == 0.0      # (1) is blind to it ...
    assert not bad["ok"] and bad["max_abs_fan_minus_reroll_m"] > 0.5, bad   # ... (2) is RED


def test_e_MUTATION_exporting_the_residual_controls_is_caught(T, monkeypatch) -> None:
    """The consumer that re-rolls exported controls (the trainer's F3 cascade) must get the
    emitted fan back. Exporting Delta as `u0_hat` breaks that; so does forgetting the prior."""
    _a, cfg, model = _arm(T, "ha0_ext_pose")
    b = _batch(T, cfg)
    dec = model.core.decoder

    def reroll_gap(out):
        rp = KP.prior_from_out(out)
        p = dec._state_to_path(out["u0_hat"], rp[2], False, prior=rp)
        return float((p - out["anchor_traj"]).abs().max())
    out = _forward(T, cfg, model, b, "ha0_ext_pose")
    assert reroll_gap(out) < 1e-3, reroll_gap(out)
    with pytest.raises(ValueError, match="needs `prior=`"):
        dec._state_to_path(out["u0_hat"], out["residual_prior_v"], False)
    monkeypatch.setattr(type(dec), "_export_controls", lambda self, z, prior: z)
    out_bad = _forward(T, cfg, model, b, "ha0_ext_pose")
    assert reroll_gap(out_bad) > 0.5, reroll_gap(out_bad)   # RED
    assert not KP.plan_check(out_bad, dec.anchor_controls, decoder=dec)["ok"]


# =========================================================================== #
# SPEC_REFCV7 section 7 (A2, PI): nav compliance and the max-speed ceiling    #
# rank the ABSOLUTE candidates (P + Delta), never Delta                       #
# =========================================================================== #
#: nav index of `right` in refc.NAV_COMMANDS = ("follow", "left", "right", "straight")
NAV_RIGHT = 2
#: the vocabulary index of (a_lon 0, a_lat -1.5): a residual that turns RIGHT
RIGHT_RESIDUAL_IDX = 8


def _selection_arm(T, monkeypatch, *, drop_prior_in_fan: bool):
    """A residual build with the nav-compliance term and the ceiling filter ON, one EVAL forward
    on a crafted window whose prior P ACCELERATES (v0 12 m/s, a0 +1.5 m/s^2) and turns LEFT
    (omega0 0.3 rad/s), under a RIGHT command and a 50 km/h ceiling. The sampler's noise is
    zeroed and its heads are at their zero init, so the (0, 0) residual candidate of the fan is
    EXACTLY P (Delta = 0). Both mechanisms' INPUTS and OUTPUTS are recorded where the decoder
    calls them."""
    from tanitad.refs import refcv6_selection as v6sel
    args, cfg, _m = _arm(T, "ha0_ext_pose")
    cfg.core.graft_nav_compliance = True
    cfg.core.nav_compliance_tau_rad = 0.05
    cfg.core.speed_ceiling_filter = True
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    with torch.no_grad():
        model.core.decoder.anchor_controls.copy_(_grid())
    dec = model.core.decoder
    assert dec.navc_gate is not None and dec.speed_ceiling_filter
    b = _batch(T, cfg, turning=False)
    W = int(b["pose_hist"].shape[1])
    t = torch.arange(W, dtype=torch.float32)
    b["pose_hist"][:, :, 3] = 12.0 - 1.5 * 0.1 * (W - 1) + 1.5 * 0.1 * t   # ends at v0 = 12.0
    b["pose_hist"][:, :, 2] = 0.3 * 0.1 * t
    b["pose_last"] = b["pose_hist"][:, -1].clone()
    rec: dict = {}
    real_nc = v6sel.nav_compliance_prior

    def rec_nc(cand, nav_cmd, **kw):
        o = real_nc(cand, nav_cmd, **kw)
        rec["nc_cand"], rec["nc"] = cand.detach().clone(), o[0].detach().clone()
        return o
    real_f = v6sel.SpeedCeilingFilter.forward

    def rec_f(self, cand, v_limit_ms):
        keep, tele = real_f(self, cand, v_limit_ms)
        rec["ceil_cand"], rec["keep"] = cand.detach().clone(), keep.detach().clone()
        return keep, tele
    monkeypatch.setattr(v6sel, "nav_compliance_prior", rec_nc)
    monkeypatch.setattr(v6sel.SpeedCeilingFilter, "forward", rec_f)
    v_lim = torch.full((4,), 50.0 / 3.6)
    real_fwd = type(dec).forward
    monkeypatch.setattr(type(dec), "forward",
                        lambda self, *a, **k: real_fwd(self, *a, **{**k, "v_limit_ms": v_lim}))
    monkeypatch.setattr(torch, "randn_like", lambda x, *a, **k: torch.zeros_like(x))
    if drop_prior_in_fan:                      # the RED arm: Delta reaches selection, not P + Delta
        real_roll = type(dec)._roll_state

        def fan_without_prior(self, z, v, metre, prior):
            if prior is None:
                return real_roll(self, z, v, metre, prior)
            a0, k0, pv = prior
            return real_roll(self, z, v, metre, (torch.zeros_like(a0), torch.zeros_like(k0), pv))
        monkeypatch.setattr(type(dec), "_roll_state", fan_without_prior)
    model.eval()
    with torch.no_grad():
        model.core.set_ego_window(b["pose_hist"], W)
        out = model(T.frames_to_device(b["frames"], "cpu"),
                    nav_cmd=torch.full((4,), NAV_RIGHT, dtype=torch.long),
                    v0=b["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)
    return out, rec


def test_A2_nav_compliance_and_the_ceiling_rank_the_ABSOLUTE_plan(T, monkeypatch) -> None:
    out, rec = _selection_arm(T, monkeypatch, drop_prior_in_fan=False)
    fan, p = out["anchor_traj"], out["residual_prior_path"]
    # (1) both mechanisms were handed the EMITTED fan -- the composed candidates, bit for bit
    assert torch.equal(rec["nc_cand"], fan) and torch.equal(rec["ceil_cand"], fan)
    # (2) Delta = 0 on the (0, 0) residual: that candidate IS the prior, exactly
    assert torch.equal(fan[:, ZERO_IDX], p)
    # (3) the ceiling acts on P: P accelerates past 50 km/h and is MASKED ...
    assert not bool(rec["keep"][:, ZERO_IDX].any()), rec["keep"][:, ZERO_IDX]
    assert bool(rec["keep"].any(dim=1).all())               # ... while braking residuals survive
    # (4) nav compliance acts on P + Delta: P turns LEFT against a RIGHT command -> 0, and the
    #     RIGHT-turning residual composed on it still turns left -> 0 (not rewarded)
    assert float(rec["nc"][:, ZERO_IDX].abs().max()) == 0.0
    assert float(rec["nc"][:, RIGHT_RESIDUAL_IDX].abs().max()) == 0.0


def test_A2_MUTATION_Delta_fed_to_selection_goes_RED(T, monkeypatch) -> None:
    """The RED arm the coordinator asked for: the SAME window with the fan rolled WITHOUT the
    prior (Delta alone reaches selection). Both mechanisms now read the wrong plan and their
    verdicts on the two probe candidates FLIP -- so the assertions of the test above fail."""
    out, rec = _selection_arm(T, monkeypatch, drop_prior_in_fan=True)
    # the (0, 0) candidate is now a constant-speed straight line at 12 m/s: KEPT under 50 km/h
    assert bool(rec["keep"][:, ZERO_IDX].all())
    # the right residual alone now turns right and "complies": REWARDED
    assert float(rec["nc"][:, RIGHT_RESIDUAL_IDX].min()) == 1.0
    # and the fan's (0, 0) candidate is no longer P
    assert not torch.equal(out["anchor_traj"][:, ZERO_IDX], out["residual_prior_path"])
