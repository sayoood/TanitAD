"""Unit tests for TRK-SCORE (PREREG_TRKSCORE.md section 6, "pre-run unit tests"). Run in the SEAM's venv (driverl-eval: torch + the DriveRL
nuplan fork) with the navsim devkit available, either under pytest or directly:

    TRK_PKG=<PK of the tip export> TRK_NAV=<navsim @ 3e8291b> CUDA_VISIBLE_DEVICES=-1 python test_trk_score.py

identity (+ its mutation), frame round-trip, grid, heading branch, RED (start state reaches the tracker; a set-reading scorer changes
the pick), completeness, submitted = planned (the patched infer), batch independence of the tracker, RED subset rate."""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("TRK_PKG", "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
NAV = os.environ.get("TRK_NAV", "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896")
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, HERE)
import trk_lib as TL  # noqa: E402

TINY = dict(backbone="vits16", width=32, depth=1, heads=2, img_h=32, img_w=32, n_cameras=1, cameras=("CAM_F0",), dec_depth=1,
            n_proposals=64, horizon_steps=20, reg_heads=2, n_registers=2, lora_rank=2)


def _tiny():
    import torch
    from model import REFe, REFeConfig
    torch.manual_seed(3)
    m = REFe(REFeConfig(**TINY)).eval()
    with torch.no_grad():                       # move every tensor off its init so the heads are not degenerate
        g = torch.Generator().manual_seed(5)
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    cap = {}
    m.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))
    return m, cap


def _forward(m):
    import torch
    g = torch.Generator().manual_seed(9)
    img = torch.randn(1, 1, 3, 32, 32, generator=g)
    ego = torch.randn(1, m.cfg.ego_dim, generator=g)
    goal = torch.randn(1, 2 * m.cfg.n_goal_points, generator=g)
    with torch.no_grad():
        return m(img, ego, goal)


def _stub_planner(model):
    import planner as PL
    st = types.SimpleNamespace(model=model, rule="navsim_v1", V1_W=PL.REFePlanner.V1_W, PDM_W=PL.REFePlanner.PDM_W, select="best",
                               repair_last_heading=True)
    st.aggregate = types.MethodType(PL.REFePlanner.aggregate, st)
    st.executed = types.MethodType(PL.REFePlanner.executed, st)
    st._pick = types.MethodType(PL.REFePlanner._pick, st)
    return st


def _seam():
    sys.path.insert(0, os.path.join(PKG, "eval"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("seam_t", os.path.join(PKG, "eval", "refe_navtest_seam.py"))
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


def _ego(v=8.0, x=120.0, y=-40.0, h=0.7):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    return EgoState.build_from_rear_axle(StateSE2(x, y, h), StateVector2D(v, 0.0), StateVector2D(0.3, 0.0), 0.0, TimePoint(1_600_000_000_000_000),
                                         get_pacifica_parameters(), angular_vel=0.0)


def _plans20(v=8.0):
    """64 smooth plans: constant speed v, lateral end offsets (j - 32) * 0.08 m via a quadratic ramp, heading = path tangent."""
    t = 0.2 * np.arange(1, 21)
    P = np.zeros((64, 20, 3))
    for j in range(64):
        yend = (j - 32) * 0.08
        x = v * t
        y = yend * (t / 4.0) ** 2
        dy = yend * 2 * t / 16.0
        P[j, :, 0], P[j, :, 1], P[j, :, 2] = x, y, np.arctan2(dy, v)
    return P


# ------------------------------------------------------------------ pure numpy
def test_frame_roundtrip_and_grid():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.normal(size=(64, 20, 3)) * [10.0, 3.0, 0.8]
        x0, y0, h0 = rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-math.pi, math.pi)
        w = TL.refe20_to_world(p, x0, y0, h0)
        back = TL.world_to_refe20(np.concatenate([w, np.zeros((64, 41, 8))], -1), x0, y0, h0, ref20=p)
        assert np.abs(back - p).max() < 1e-9, np.abs(back - p).max()
    sim = np.zeros((1, 41, 11))
    sim[0, :, 0] = 10.0 * 0.1 * np.arange(41)            # x = 10 t
    out = TL.world_to_refe20(sim, 0.0, 0.0, 0.0)
    assert np.allclose(out[0, :, 0], 10.0 * 0.2 * np.arange(1, 21)), out[0, :, 0]   # REFe step i <-> t = 0.2 (i+1)


def test_heading_branch():
    sim = np.zeros((1, 41, 11))
    sim[0, :, 2] = -3.1                                   # world heading -3.1 rad with h0 = 0
    ref = np.zeros((1, 20, 3))
    ref[0, :, 2] = 3.1                                    # the planned set's branch near +pi
    out = TL.world_to_refe20(sim, 0.0, 0.0, 0.0, ref20=ref)
    assert np.allclose(out[0, :, 2], 2 * math.pi - 3.1), out[0, 0, 2]
    out2 = TL.world_to_refe20(sim, 0.0, 0.0, 0.0)          # no reference: wrapped into [-pi, pi)
    assert np.allclose(out2[0, :, 2], -3.1)


def test_red_subset_rate():
    toks = [f"{i:016x}" for i in range(20000)]
    r = sum(TL.is_red_token(t) for t in toks) / len(toks)
    assert 0.09 < r < 0.11, r


# ------------------------------------------------------------------ the scoring path
def test_identity_and_its_mutation():
    import torch
    m, cap = _tiny()
    traj, score = _forward(m)
    with torch.no_grad():                    # as in the seam: planner.infer and trk_step both run under no_grad
        s = m.score_trajectories(traj, cap["sctx"])
    assert torch.equal(s, score), float((s - score).abs().max())       # bit for bit: same modules, same tensor
    pl = _stub_planner(m)
    assert TL.pick(pl, s[0]) == pl._pick(score)
    # MUTATION: a wrong context must break the identity (the check can fail)
    with torch.no_grad():
        bad = m.score_trajectories(traj, torch.zeros_like(cap["sctx"]))
    assert not torch.equal(bad, score)


def test_completeness_refused():
    import torch
    m, cap = _tiny()
    traj, score = _forward(m)
    pl = _stub_planner(m)
    try:
        TL.trk_step(pl, None, traj[0, :63], score[0, :63], 0, cap["sctx"], to_navsim=None)
    except ValueError as e:
        assert "COMPLETE" in str(e)
    else:
        raise AssertionError("an incomplete set was not refused")
    try:
        TL.check_set(torch.zeros(64, 20, 3), types.SimpleNamespace(n_proposals=32, horizon_steps=20, traj_dim=3))
    except ValueError:
        pass
    else:
        raise AssertionError("a 32-proposal model config was not refused")


# ------------------------------------------------------------------ the tracker (navsim)
def test_tracker_batch_independence_and_start_row():
    TL.tracker_setup(NAV)
    ego = _ego()
    seam = _seam()
    props8 = np.stack([seam.to_navsim(p) for p in _plans20()]).astype(np.float32)
    st = TL.proposal_states(props8, ego)
    sim = TL.simulate(st, ego)
    assert sim.shape == (64, 41, 11)
    s0 = TL._T["state_array"](ego)
    assert np.abs(sim[:, 0] - s0).max() == 0.0             # row 0 is the start state
    sub = TL.simulate(st[[7, 40]], ego)
    assert np.abs(sub - sim[[7, 40]]).max() == 0.0, np.abs(sub - sim[[7, 40]]).max()
    # the tracked plan follows the plan (a sanity bound, not a parity claim: parity is CTRL-PARITY on real tokens)
    trk = TL.world_to_refe20(sim, ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading)
    assert np.abs(trk[..., :2] - _plans20()[..., :2]).max() < 1.0


def test_red_start_state_reaches_tracker_and_changes_pick():
    import torch
    TL.tracker_setup(NAV)
    ego = _ego()
    red = TL.red_ego(ego)
    dx, dy = red.rear_axle.x - ego.rear_axle.x, red.rear_axle.y - ego.rear_axle.y
    h = ego.rear_axle.heading
    assert abs((-math.sin(h)) * dx + math.cos(h) * dy - TL.RED_DY_M) < 1e-12 and abs(math.cos(h) * dx + math.sin(h) * dy) < 1e-12
    seam = _seam()
    plans = _plans20()
    m, cap = _tiny()
    traj, score = _forward(m)
    traj = torch.from_numpy(plans.astype(np.float32)).unsqueeze(0)      # a planned set with known geometry
    pl = _stub_planner(m)

    def scorer(tt, sctx):                                                 # a set-READING scorer: best = mean lateral position nearest 0
        y = tt[0, :, :, 1].mean(-1)
        lg = torch.full((tt.shape[1], 6), 10.0)
        lg[:, 2] = -10.0 * y.abs()
        return lg.unsqueeze(0)
    m.score_trajectories = scorer
    rec = TL.trk_step(pl, ego, traj[0], score[0], 0, cap["sctx"], seam.to_navsim, red=True)
    assert rec["red_d0"] >= 0.5, rec["red_d0"]
    assert rec["k_red"] != rec["k_trk"], (rec["k_red"], rec["k_trk"])
    assert rec["trk20"].shape == (64, 20, 3) and rec["ship8"].shape == (8, 3)
    # with the TRUE start state the set-reading scorer picks the proposal whose TRACKED mean y is nearest 0
    assert rec["k_trk"] == int(np.abs(rec["trk20"][:, :, 1].mean(-1)).argmin())


def test_submitted_is_planned_via_patched_infer():
    """The launcher's patched infer returns the ORIGINAL traj object and k_trk; the seam then executes the planned proposal k_trk."""
    import torch
    import trk_select as TS
    m, cap = _tiny()
    seen = []

    class FakePlanner:
        V1_W, PDM_W = _stub_planner(m).V1_W, _stub_planner(m).PDM_W

        def __init__(self, model):
            self.model, self.rule, self.select, self.repair_last_heading = model, "navsim_v1", "best", True
            self._scenario = types.SimpleNamespace(_initial_lidar_token="00000000000000aa")

        def infer(self, ego, img):
            with torch.no_grad():
                t, score = self.model(*img)
            v = t[0]
            seen.append(v)
            return v, score[0], int(self.aggregate(score)[0].argmax())

    import planner as PL
    FakePlanner.aggregate = PL.REFePlanner.aggregate
    FakePlanner.executed = PL.REFePlanner.executed
    fake_pl = types.SimpleNamespace(REFePlanner=FakePlanner)
    a = types.SimpleNamespace(red_all=True, red_mod=10, keep_sim=False)
    st = TS.State(a)
    TS.install(a, st, fake_pl, TL, _seam().to_navsim)
    TL.tracker_setup(NAV)
    p = FakePlanner(m)
    g = torch.Generator().manual_seed(9)
    img = (torch.randn(1, 1, 3, 32, 32, generator=g), torch.randn(1, m.cfg.ego_dim, generator=g), torch.randn(1, 2 * m.cfg.n_goal_points, generator=g))
    out_traj, out_score, k = p.infer(_ego(), img)
    assert out_traj is seen[0]                                            # the planned set itself, never the tracked one
    rec = st.recs["00000000000000aa"]
    assert k == rec["k_trk"] and rec["k_id"] == rec["k_ship"]
    assert torch.equal(p.executed(out_traj, k), PL.repair_last_heading(seen[0][k]))   # the PLANNED proposal, A7-repaired as the planner executes it


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print(f"ZZTRK_TESTS {'PASS' if fails == 0 else 'FAIL'} failed={fails}", flush=True)
    sys.exit(1 if fails else 0)
