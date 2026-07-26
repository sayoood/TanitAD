"""``taniteval.clhorizon`` — the de-stranded horizon-capable closed loop.

WHY THIS MODULE EXISTS (the stranding, MEASURED 2026-07-26)
-----------------------------------------------------------
The gate card registers ``corridor_departure_rate`` at **K=185 (18.5 s)** on the
closed-loop surface. ``taniteval/closedloop.py`` fixed ``K_MAX = 20`` — the
*blind* horizon — and its ``run_and_save`` refused any arm without
``traj_capable`` **and** ``model.tactical_policy``, which a v4
``FlagshipV4Head`` checkpoint has **neither** of. **So the registered co-primary
was reachable only through a one-off driver in ``incoming/``.**

THE TWO LOAD-BEARING TESTS
--------------------------
1. :func:`test_port_is_tensor_identical_to_the_driver` — the ported rollout and
   the ``incoming/`` driver's rollout produce **bit-identical** tensors on the
   same inputs. "Keeping the driver's measured behaviour" is asserted, not
   asserted-about.
2. :func:`test_reproduces_the_gate_coprimary_numbers` — on the COMMITTED
   per-window tensors the port reproduces **v4 K=185 overall 0.6388 / junction
   0.8432** and **REF-C base 0.5833**, the numbers in ``GATE_30K_RESULTS.md``
   6.3, exactly.

CPU only: a synthetic 3-episode corpus + a deterministic stub planner. No
checkpoint, no GPU.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from taniteval import clhorizon as CH
from taniteval import closedloop as CL
from taniteval import ood as _ood

_REPO = Path(__file__).resolve().parents[2]
_COPRIMARY = (_REPO / "TanitAD Research Hub" / "Benchmarks & Eval"
              / "Implementation" / "incoming" / "2026-07-26-v4-30k-gate"
              / "coprimary")
_PW_V4 = _COPRIMARY / "corridor_v4_30k_K185_perwindow_K185.pt"
_PW_REFC = _COPRIMARY / "corridor_refcbase_30k_K185_perwindow_K185.pt"

# MEASURED — GATE_30K_RESULTS.md 6.3. Named so a drift goes red.
V4_K185_OVERALL = 0.6388
V4_K185_JUNCTION = 0.8432
REFC_K185_OVERALL = 0.5833


# =========================================================================== #
# a synthetic corpus + a deterministic stub planner                           #
# =========================================================================== #
def _episode(T=60, seed=0, H=16, Wd=16):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=torch.float32)
    yaw = 0.02 * torch.sin(t * 0.05) + 0.001 * seed
    v = 8.0 + 0.5 * torch.sin(t * 0.03)
    x = torch.cumsum(v * torch.cos(yaw) * CH.DT, 0)
    y = torch.cumsum(v * torch.sin(yaw) * CH.DT, 0)
    poses = torch.stack([x, y, yaw, v], dim=-1)
    frames = torch.rand(T, 3, H, Wd, generator=g)
    return SimpleNamespace(poses=poses, frames=frames, episode_id=f"ep{seed}")


class _StubPlanner:
    """Deterministic, frame-dependent, and NOT constant — so a difference in the
    warped window (the thing the loop actually feeds the model) shows up in the
    trajectory and therefore in every downstream tensor."""

    def __init__(self):
        self.calls = 0

    def traj(self, fw, v0, goal_batch):
        self.calls += 1
        b = fw.shape[0]
        sig = fw.reshape(b, -1).mean(dim=1).float()
        steps = torch.arange(1, 21, dtype=torch.float32)[None]
        x = v0.float().cpu()[:, None] * steps * CH.DT
        y = 0.4 * torch.sin(sig.cpu()[:, None] * 12.0) * steps * CH.DT
        return torch.stack([x, y], dim=-1)


def _driver():
    """The ``incoming/`` driver, imported for the identity check."""
    for p in (_REPO / "stack", _REPO / "stack" / "scripts", _COPRIMARY):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return pytest.importorskip("v4_corridor_cl")


# =========================================================================== #
# ⭐ 1 — the port keeps the driver's MEASURED behaviour, tensor for tensor     #
# =========================================================================== #
def test_port_is_tensor_identical_to_the_driver():
    V = _driver()
    eps = [_episode(T=60, seed=s) for s in range(3)]
    K = 30

    def frames_of(ep, a, b):
        fr = ep.frames[a:b]
        return fr.float() if fr.dtype != torch.uint8 else fr.float().div(255.0)

    mine = CH.corridor_rollout(_StubPlanner(), eps, None, "cpu", K,
                               stride=8, batch=4, frames_of=frames_of)
    # The driver's `_frames` is module-level; point it at the same accessor so
    # only the ROLLOUT differs between the two runs.
    old = V._frames
    V._frames = frames_of
    try:
        class _G:
            def get(self, *a, **k):
                return None
        theirs = V.rollout(_StubPlanner(), eps, _G(), "cpu", K, stride=8,
                           batch=4, verbose=False)
    finally:
        V._frames = old

    assert mine is not None and theirs is not None
    assert mine["eid"] == theirs["eid"]
    assert mine["fixed_steps"] == theirs["fixed_steps"]
    assert mine["_rollout_steps_executed"] == theirs["_rollout_steps_executed"]
    for key in ("lat", "yaw", "ade2s", "hd2s", "hdK", "speed", "t0", "epi",
                "de_fixed"):
        assert torch.equal(mine[key], theirs[key]), f"{key} differs from the driver"


def test_the_rollout_actually_advances_K_steps():
    eps = [_episode(T=80, seed=s) for s in range(2)]
    pw = CH.corridor_rollout(_StubPlanner(), eps, None, "cpu", 50, stride=8,
                             batch=4)
    assert pw["lat"].shape[1] == 50
    # `_rollout_steps_executed` counts loop ticks per BATCH chunk (the driver's
    # own semantics, kept): 2 episodes x 1 chunk x K.
    assert pw["_rollout_steps_executed"] == 50 * 2


class TestHorizonIsFree:
    def test_K_far_beyond_20_is_accepted(self):
        """The whole defect: 20 was a CAP, and the co-primary needs 185."""
        eps = [_episode(T=210, seed=s) for s in range(2)]
        pw = CH.corridor_rollout(_StubPlanner(), eps, None, "cpu", 185,
                                 stride=32, batch=4)
        assert pw is not None and pw["lat"].shape[1] == 185

    def test_the_structural_ceiling_is_refused(self):
        with pytest.raises(ValueError, match="structural ceiling"):
            CH.corridor_rollout(_StubPlanner(), [_episode()], None, "cpu", 200)

    def test_n_collapses_with_K_and_that_is_reported(self):
        """`starts = range(0, T - W - K, stride)` — the reason a K=185 block has
        ~1 window per episode. A corridor number without its n is inadmissible."""
        assert CH.horizon_windows(200, 20, stride=8) > CH.horizon_windows(200, 185, stride=8)
        assert CH.horizon_windows(200, 185, stride=8) == 1
        assert CH.horizon_windows(190, 185, stride=8) == 0

    def test_no_surviving_window_is_a_NOT_MEASURED_not_a_pass(self):
        assert CH.corridor_rollout(_StubPlanner(), [_episode(T=30)], None,
                                   "cpu", 25) is None


# =========================================================================== #
# ⭐ 2 — the gate co-primary numbers, reproduced                               #
# =========================================================================== #
@pytest.mark.skipif(not _PW_V4.exists() or not _PW_REFC.exists(),
                    reason="committed per-window tensors absent")
def test_reproduces_the_gate_coprimary_numbers():
    """On the COMMITTED per-window tensors from the 30 k gate."""
    v4 = CH.corridor_from_perwindow(_PW_V4, K=185)
    assert v4["overall"]["corridor_departure_rate"]["mean"] == V4_K185_OVERALL
    assert v4["junction"]["corridor_departure_rate"]["mean"] == V4_K185_JUNCTION
    assert v4["overall"]["n_windows"] == 41 and v4["overall"]["n_episodes"] == 40
    assert v4["overall"]["corridor_primary_m"] == 1.75
    assert v4["horizon_K"] == 185 and v4["horizon_s"] == 18.5
    assert (v4["overall"]["corridor_departure_rate"]["estimator"]
            == "episode_cluster_bootstrap")

    refc = CH.corridor_from_perwindow(_PW_REFC, K=185)
    assert refc["overall"]["corridor_departure_rate"]["mean"] == REFC_K185_OVERALL
    assert refc["overall"]["n_windows"] == 41


@pytest.mark.skipif(not _PW_V4.exists(), reason="committed per-window tensors absent")
def test_the_emitted_OOD_block_is_the_FIXED_one():
    """The co-primary's own artifact said "within the measured envelope on
    average" at K=185. The ported emitter cannot say that."""
    v4 = CH.corridor_from_perwindow(_PW_V4, K=185)
    o = v4["ood"]["overall"]
    assert o["EXTRAPOLATION_VERDICT"] == _ood.VERDICT_EXTRAPOLATION
    assert o["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] > 0.9
    assert o["ratio_is_lower_bound"] is True
    for stratum in ("overall", "junction", "longitudinal", "other"):
        node = v4["ood"][stratum]
        assert _ood.verdict_class(node["EXTRAPOLATION_VERDICT"]) != _ood.CLASS_MEASUREMENT


@pytest.mark.skipif(not _PW_V4.exists(), reason="committed per-window tensors absent")
def test_reaggregation_is_arithmetic_only():
    """The point of persisting per-window paths: a different half-width is a
    recomputation, not a GPU re-run."""
    out = CH.corridor_from_perwindow(_PW_V4, K=185, thresholds=(1.0, 1.75, 2.5),
                                     primary=1.0)
    assert out["overall"]["corridor_primary_m"] == 1.0
    assert out["overall"]["corridor_departure_rate"]["mean"] == 0.7048  # 6.3


# =========================================================================== #
# closedloop.py — the two lines that stranded the co-primary                  #
# =========================================================================== #
class TestClosedLoopNoLongerRefusesAV4Arm:
    def test_K_MAX_is_a_default_not_a_cap(self):
        assert CL.K_MAX == CL.K_ADE2S == 20
        assert CL.HORIZON_CEILING_K == 190
        # every rollout takes k
        import inspect
        for fn in (CL.closed_loop_rollout, CL.collect, CL.run_and_save):
            assert "k" in inspect.signature(fn).parameters, fn.__name__

    def test_collect_refuses_only_the_STRUCTURAL_ceiling(self):
        with pytest.raises(ValueError, match="structural ceiling"):
            CL.collect(None, None, [], "cpu", k=191)

    def test_run_and_save_accepts_an_injected_plan_step(self):
        """The refusal at the old line 897 rejected every v4 checkpoint. It now
        refuses only when there is genuinely no way to plan."""
        import inspect
        p = inspect.signature(CL.run_and_save).parameters
        assert "plan_fn" in p and "model" in p
        assert p["save_per_window"].default is True     # persisted BY DEFAULT

    def test_it_still_refuses_when_there_is_no_planner_at_all(self):
        out = CL.run_and_save("definitely-not-an-arm", device="cpu")
        assert "skipped" in out

    def test_closed_loop_rollout_uses_the_injected_plan_fn(self):
        """A v4 arm has no `tactical_policy`; the loop must never touch it."""
        seen = {"n": 0}

        class _NoHierarchy:
            """Any access to strategic/tactical is a bug — a v4 arm has none."""

            def __getattr__(self, name):
                if name in ("strategic_policy", "tactical_policy"):
                    raise AssertionError(
                        f"the injected plan_fn path must not touch {name}")
                raise AttributeError(name)

            def predictor(self, win_s, win_a):
                return None, win_s[:, -1] * 1.0

        def plan_fn(model, win_s, v):
            seen["n"] += 1
            b = win_s.shape[0]
            return torch.stack([v.float(), torch.zeros(b)], dim=-1)

        b, Wn, S, A = 3, 4, 6, 2
        out = CL.closed_loop_rollout(
            _NoHierarchy(), lambda a, bb: torch.zeros(a.shape[0], 3),
            torch.zeros(b, Wn, S), torch.zeros(b, Wn, A),
            torch.full((b,), 5.0), False, k=7, plan_fn=plan_fn)
        assert seen["n"] == 7
        assert out["closed_bike"].shape == (b, 7, 2)


class TestArmAIsNotFaked:
    def test_arm_A_is_NOT_MEASURED_rather_than_zero_filled(self):
        """A zero path is a *plausible* path and would quietly produce a
        real-looking imagination A/B. It must be NaN and then stated."""
        n = 6
        gt = torch.rand(n, 4, 2)
        # eids must be int-castable: `split_by_episode` (legacy block) casts.
        win = {"eid": [str(i // 2) for i in range(n)], "gt": gt,
               "cv": gt + 0.1, "closed_bike": gt + 0.2, "closed_grnd": gt + 0.2,
               "open_grnd": gt + 0.1, "open_bike": gt + 0.1,
               "open_plan_bike": torch.full_like(gt, float("nan")),
               "plan_direct": torch.full_like(gt, float("nan")),
               "speed": torch.full((n,), 8.0),
               "head_deg": torch.zeros(n),
               "steer": torch.zeros(n, 20), "accel": torch.zeros(n, 20),
               "vseq": torch.full((n, 20), 8.0)}
        res = CL.analyze(win, n_boot=50)
        ic = res["imagination_comparison"]
        assert ic["measured"] is False
        assert "NOT MEASURED" in ic["verdict"]
        assert res["summary"]["imagination_B_minus_A_ade@2s"] is None
        # ...and the closed-loop arm (B) is still reported.
        assert ic["B_closed_bike_ade@2s"]["mean"] > 0
