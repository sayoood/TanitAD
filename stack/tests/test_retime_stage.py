"""Tests for ``tanitad.retime_stage`` -- refcv8's frozen post-planner RE-TIMING stage (RUNTIME side).

D-R8-SHIP-RETIME + SHIP-R8 AMENDMENT 1 (PI 2026-10-19): refcv8 ships with this stage iff SHIP-R8 passes at step 48,852, scored on the
DEPLOYED gain table.  What these tests pin, and how each guard is shown to be able to fail (every expected value is a LITERAL, an
analytic closed form or an independently authored reference -- never an expression over the code under test):

  * ``gain_at``            knots / midpoints / outside the range, as literals; the deploy table's eight knots as exact doubles typed here
                           AND as the Amendment's three-decimal values.
  * identity (g == 1)      the plan comes back bit for bit.   MUTATION: plan and prior swapped -> the identity assertion goes RED.
  * g == 0                 each point lies on the PLAN's path at the PRIOR's arc length (straight line and an L-shape, closed form),
                           including the end extension, the clip at s' >= 0 and the stationary-plan +x fallback.
  * on-path                a re-timed point is at distance ~0 from the plan polyline / its extension ray (an independent point-to-
                           segment distance written here).
  * table integrity        a flipped digit, a missing sidecar, a wrong sidecar, a wrong pinned digest are all REFUSED; a CRLF checkout
                           of the same bytes is accepted.  The deployed file is byte-identical to what the builder regenerates from
                           the frozen train file (MUTATION: a retyped float whose sidecar was recomputed is caught by that byte compare).
  * EQUIVALENCE            on the banked step30000ema_unc S2 and S6 windows, ``retime_plan`` reproduces the eval package's
                           ``retime_apply(..., cast=False)`` to <= 1e-9 m (it is 0.0: the geometry is a port).  MUTATION: gains
                           mapped g -> 1 - g must exceed that tolerance, so the comparison has power.  Skips, with the reason, when the
                           package / archive / torch is absent (env ``TANITAD_R8_BATTERY_PKG``, ``TANITAD_R8_ARCHIVE``).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import tanitad.retime_stage as RS

HERE = Path(__file__).resolve().parent
STACK = HERE.parent
PKG_REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"

#: the deploy table's gains as exact doubles, typed here independently of the file (they are the S2 frozen train gains at 0.5/1/1.5/2 s
#: and the S6 frozen train gains at 3/4/5/6 s, FROZEN_GAINS_RETIME_train.json sha256 b208c9c1...)
KNOTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
GAINS = [0.9673903790217738, 0.6192916835387219, 0.5511341845460037, 0.530539399347386,
         0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]
AMENDMENT_3DP = [0.967, 0.619, 0.551, 0.531, 0.655, 0.699, 0.722, 0.745]   # as printed in SHIP-R8 AMENDMENT 1


def _const(g, t0=0.0):
    """A one-knot table = a constant gain at every instant."""
    return {"knots_s": [t0], "gains": [g]}


def _zigzag(n, T, seed):
    """Deterministic curvy forward plans [n, T, 2] (float64): speed 2-12 m per step, heading random walk."""
    rng = np.random.default_rng(seed)
    step = rng.uniform(2.0, 12.0, (n, T))
    yaw = np.cumsum(rng.normal(0.0, 0.25, (n, T)), axis=1)
    return np.cumsum(np.stack([step * np.cos(yaw), step * np.sin(yaw)], -1), axis=1)


def _dist_to_polyline(p, plan):
    """Distance from ONE point to the polyline (0,0) -> plan[0] -> ... -> plan[-1] unioned with the final extension ray along the last
    positive-length segment (written independently of the module under test)."""
    V = np.vstack([np.zeros((1, 2)), plan]).astype(np.float64)
    best = np.inf
    for a, b in zip(V[:-1], V[1:]):
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den == 0.0 else min(1.0, max(0.0, float((p - a) @ ab) / den))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    segs = V[1:] - V[:-1]
    ln = np.hypot(segs[:, 0], segs[:, 1])
    if (ln > 0).any():
        j = int(np.flatnonzero(ln > 0)[-1])
        d = segs[j] / ln[j]
        t = max(0.0, float((p - V[-1]) @ d))
        best = min(best, float(np.linalg.norm(p - (V[-1] + t * d))))
    return best


# ------------------------------------------------------------------------------------------------------------ gain_at
def test_gain_at_knots_midpoints_and_outside_as_literals():
    tab = {"knots_s": [1.0, 2.0, 4.0], "gains": [0.2, 0.6, 1.0]}
    assert RS.gain_at(1.0, tab) == 0.2 and RS.gain_at(2.0, tab) == 0.6 and RS.gain_at(4.0, tab) == 1.0   # knots: exact
    assert RS.gain_at(1.5, tab) == pytest.approx(0.4, abs=1e-15)      # midpoint of (1, .2)-(2, .6)
    assert RS.gain_at(3.0, tab) == pytest.approx(0.8, abs=1e-15)      # midpoint of (2, .6)-(4, 1.)
    assert RS.gain_at(2.5, tab) == pytest.approx(0.7, abs=1e-15)      # quarter way on the second segment: 0.6 + 0.25 * 0.4
    assert RS.gain_at(0.5, tab) == 0.2 and RS.gain_at(0.0, tab) == 0.2 and RS.gain_at(-3.0, tab) == 0.2   # held before the first knot
    assert RS.gain_at(5.0, tab) == 1.0 and RS.gain_at(100.0, tab) == 1.0                                  # held after the last knot
    out = RS.gain_at(np.array([[1.0, 3.0], [0.0, 9.0]]), tab)
    assert out.shape == (2, 2) and out.ravel().tolist() == pytest.approx([0.2, 0.8, 0.2, 1.0], abs=1e-15)


def test_gain_at_refuses_a_malformed_table():
    for bad in ({"knots_s": [1.0, 1.0], "gains": [0.1, 0.2]},          # not strictly increasing
                {"knots_s": [2.0, 1.0], "gains": [0.1, 0.2]},
                {"knots_s": [1.0, 2.0], "gains": [0.1]},               # length mismatch
                {"knots_s": [], "gains": []},
                {"knots_s": [1.0], "gains": [float("nan")]}):
        with pytest.raises(RS.RetimeStageError):
            RS.gain_at(1.0, bad)


def test_deploy_table_is_the_amendment_table_exactly():
    tab = RS.load_deploy_table()
    assert tab["knots_s"].tolist() == KNOTS
    assert tab["gains"].tolist() == GAINS                                   # exact doubles: no rounding, no retyping
    assert np.allclose(tab["gains"], AMENDMENT_3DP, atol=5e-4, rtol=0)     # and the Amendment's printed 3-decimal values
    assert tab["meta"]["name"] == "RETIME_DEPLOY"
    assert tab["meta"]["source"]["sha256"] == "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
    # S2 is IDENTICAL to RETIME_TRAINFIT (every S2 instant is a knot) ...
    assert RS.gain_at([0.5, 1.0, 1.5, 2.0], tab).tolist() == GAINS[:4]
    # ... and on S6 only the 1.0 s and 2.0 s gains change (0.639 -> 0.619, 0.597 -> 0.531): the S6 TRAIN gains were
    # 0.6393548580383511 / 0.5969720822621201 at those instants, the deployed values are the S2 knots
    s6 = RS.gain_at([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tab)
    assert s6.tolist() == [0.6192916835387219, 0.530539399347386, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424,
                           0.7453601677670005]
    # between the knots 2.0 and 3.0 it is the straight line, held constant outside [0.5, 6.0]
    assert RS.gain_at(2.5, tab) == pytest.approx((0.530539399347386 + 0.6546139969520486) / 2.0, abs=1e-15)
    assert RS.gain_at(0.1, tab) == 0.9673903790217738 and RS.gain_at(9.0, tab) == 0.7453601677670005


# ------------------------------------------------------------------------------------------------- identity (g == 1)
def _assert_identity(call, plan, prior, t):
    """The g == 1 contract: whatever ``call(plan, prior, t, table)`` is, it returns ``plan`` unchanged, bit for bit."""
    out = call(plan, prior, t, _const(1.0))
    assert out.dtype == plan.dtype and out.shape == plan.shape
    assert out.tobytes() == plan.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_g_equal_one_returns_the_plan_bit_for_bit(dtype):
    t = np.array([0.5, 1.0, 1.5, 2.0])
    plan = _zigzag(64, 4, 1).astype(dtype)
    prior = _zigzag(64, 4, 2).astype(dtype)            # a DIFFERENT path: it must not leak into a g == 1 result
    _assert_identity(RS.retime_plan, plan, prior, t)
    lead = _zigzag(6, 4, 3).astype(dtype).reshape(2, 3, 4, 2)   # leading batch dims are carried through
    _assert_identity(RS.retime_plan, lead, lead[::-1].copy(), t)
    _assert_identity(RS.retime_plan, plan[0], prior[0], t)       # an unbatched [T, 2] plan


def test_g_equal_one_on_the_stationary_and_degenerate_plans():
    t = np.array([1.0, 2.0])
    still = np.zeros((1, 2, 2))
    moving = np.array([[[2.0, 0.0], [4.0, 0.0]]])
    _assert_identity(RS.retime_plan, still, moving, t)                          # a zero-length polyline
    stop_go = np.array([[[0.0, 0.0], [3.0, 4.0]]])                              # zero-length FIRST segment
    _assert_identity(RS.retime_plan, stop_go, moving, t)
    go_stop = np.array([[[3.0, 4.0], [3.0, 4.0]]])                              # zero-length LAST segment
    _assert_identity(RS.retime_plan, go_stop, moving, t)


def test_MUTATION_swapping_plan_and_prior_makes_the_identity_test_RED():
    t = np.array([0.5, 1.0, 1.5, 2.0])
    plan, prior = _zigzag(16, 4, 1), _zigzag(16, 4, 2)
    swapped = lambda p, b, tt, tab: RS.retime_plan(b, p, tt, tab)               # noqa: E731  the deliberate regression
    with pytest.raises(AssertionError):
        _assert_identity(swapped, plan, prior, t)
    # and the swapped stage is exactly "the PRIOR comes back", which is what makes the identity assertion able to see it
    assert RS.retime_plan(prior, plan, t, _const(1.0)).tobytes() == prior.tobytes()


# -------------------------------------------------------------------------------- g == 0 and the closed forms
def test_g_zero_straight_line_puts_each_point_at_the_priors_arc_length():
    t = np.array([0.5, 1.0, 1.5, 2.0])
    plan = np.stack([10.0 * t, np.zeros(4)], -1)[None]            # 10 m/s along +x: arc lengths 5, 10, 15, 20
    slow = np.stack([6.0 * t, np.zeros(4)], -1)[None]             # prior 6 m/s: arc lengths 3, 6, 9, 12
    out = RS.retime_plan(plan, slow, t, _const(0.0))
    assert out[0].tolist() == [[3.0, 0.0], [6.0, 0.0], [9.0, 0.0], [12.0, 0.0]]
    fast = np.stack([12.0 * t, np.zeros(4)], -1)[None]            # prior 12 m/s: 6, 12, 18, 24 -- 24 is beyond the plan's end (20)
    out = RS.retime_plan(plan, fast, t, _const(0.0))
    assert out[0].tolist() == [[6.0, 0.0], [12.0, 0.0], [18.0, 0.0], [24.0, 0.0]]   # the extension continues along +x from (20, 0)


def test_g_zero_and_half_on_an_L_shaped_plan_closed_form():
    t = np.array([1.0, 2.0])
    plan = np.array([[[3.0, 0.0], [3.0, 4.0]]])                   # L = [3, 7]; second leg runs +y from (3, 0)
    prior = np.array([[[2.0, 0.0], [4.0, 0.0]]])                  # M = [2, 4]
    g0 = RS.retime_plan(plan, prior, t, _const(0.0))
    assert g0[0].tolist() == [[2.0, 0.0], [3.0, 1.0]]             # s' = M = [2, 4] -> (2, 0) and 1 m up the second leg
    g_half = RS.retime_plan(plan, prior, t, _const(0.5))
    assert g_half[0].tolist() == [[2.5, 0.0], [3.0, 2.5]]         # s' = [2.5, 5.5]
    g_two = RS.retime_plan(plan, prior, t, _const(2.0))           # s' = L - (M - L) = [4, 10]: the second is beyond the end (7)
    assert g_two[0].tolist() == [[3.0, 1.0], [3.0, 7.0]]          # extension continues +y from (3, 4)


def test_progress_is_clipped_at_zero_and_a_stationary_plan_extends_along_plus_x():
    t = np.array([1.0, 2.0])
    plan = np.array([[[1.0, 0.0], [2.0, 0.0]]])
    prior = np.array([[[5.0, 0.0], [10.0, 0.0]]])
    out = RS.retime_plan(plan, prior, t, _const(2.0))             # s' = L + (1 - 2)(M - L) = [-3, -6] -> clipped to 0
    assert out[0].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    still = np.zeros((1, 2, 2))
    out = RS.retime_plan(still, prior, t, _const(0.0))            # polyline of zero length: the documented +x fallback
    assert out[0].tolist() == [[5.0, 0.0], [10.0, 0.0]]


def test_every_retimed_point_lies_on_the_plans_own_path():
    t = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    plan, prior = _zigzag(24, 8, 5), _zigzag(24, 8, 6)
    tab = RS.load_deploy_table()
    out = RS.retime_plan(plan, prior, t, tab, cast=False)
    for i in range(plan.shape[0]):
        for k in range(8):
            assert _dist_to_polyline(out[i, k], plan[i]) < 1e-9
    assert not np.allclose(out, plan)                              # ... and the stage did something


def test_dtype_cast_and_shape_contract():
    t = np.array([1.0, 2.0])
    plan = _zigzag(5, 2, 7).astype(np.float32)
    prior = _zigzag(5, 2, 8).astype(np.float32)
    tab = _const(0.6)
    out32, out64 = RS.retime_plan(plan, prior, t, tab), RS.retime_plan(plan, prior, t, tab, cast=False)
    assert out32.dtype == np.float32 and out64.dtype == np.float64 and out32.shape == out64.shape == plan.shape
    assert np.array_equal(out32, out64.astype(np.float32))
    assert RS.retime_plan(plan[:0], prior[:0], t, tab).shape == (0, 2, 2)      # an empty batch is fine


def test_refusals_on_a_malformed_call():
    t = np.array([1.0, 2.0])
    plan = np.ones((3, 2, 2))
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(plan, np.ones((3, 3, 2)), t, _const(1.0))              # prior on different instants
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(plan, plan, np.array([1.0, 2.0, 3.0]), _const(1.0))    # t_s of the wrong length
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(plan, plan, np.array([2.0, 1.0]), _const(1.0))         # instants not increasing
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(np.ones((3, 2, 3)), np.ones((3, 2, 3)), t, _const(1.0))   # not xy
    bad = plan.copy()
    bad[1, 1, 0] = np.nan
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(bad, plan, t, _const(1.0))                             # non-finite plan
    with pytest.raises(RS.RetimeStageError):
        RS.retime_plan(plan, bad, t, _const(1.0))                             # non-finite prior


# --------------------------------------------------------------------------------------------- table integrity
def _copy_deploy(tmp_path):
    src = RS.DEPLOY_TABLE_PATH
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    shutil.copyfile(str(src) + ".sha256", str(dst) + ".sha256")
    return dst


def test_load_gains_table_verifies_its_sha256_sidecar(tmp_path):
    p = _copy_deploy(tmp_path)
    ok = RS.load_gains_table(p)
    assert ok["gains"].tolist() == GAINS
    assert RS.load_gains_table(p, expected_sha256=ok["sha256"].upper())["sha256"] == ok["sha256"]


def test_a_sha_mismatch_is_refused(tmp_path):
    p = _copy_deploy(tmp_path)
    text = p.read_text(encoding="utf-8")
    assert "0.9673903790217738" in text
    # MUTATION: a one-digit edit that leaves a perfectly valid JSON table -- only the sidecar can catch it
    p.write_text(text.replace("0.9673903790217738", "0.9673903790217739"), encoding="utf-8", newline="\n")
    assert json.loads(p.read_text(encoding="utf-8"))["gains"][0] == 0.9673903790217739
    with pytest.raises(RS.RetimeStageError, match="sha256"):
        RS.load_gains_table(p)


def test_missing_wrong_and_unpinned_sidecars_are_refused(tmp_path):
    p = _copy_deploy(tmp_path)
    side = Path(str(p) + ".sha256")
    good = side.read_text(encoding="utf-8")
    side.write_text("0" * 64 + "  x.json\n", encoding="utf-8")
    with pytest.raises(RS.RetimeStageError, match="sha256"):
        RS.load_gains_table(p)                                                # a well-formed but WRONG digest
    side.write_text("not a digest\n", encoding="utf-8")
    with pytest.raises(RS.RetimeStageError):
        RS.load_gains_table(p)                                                # a malformed sidecar
    side.unlink()
    with pytest.raises(RS.RetimeStageError, match="sidecar"):
        RS.load_gains_table(p)                                                # no sidecar: a table without a pinned hash is refused
    side.write_text(good, encoding="utf-8")
    with pytest.raises(RS.RetimeStageError, match="pinned"):
        RS.load_gains_table(p, expected_sha256="1" * 64)                      # the caller's own pin disagrees
    assert RS.load_gains_table(p)["gains"].tolist() == GAINS                  # restored: loads again


def test_a_crlf_checkout_of_the_same_bytes_still_verifies(tmp_path):
    """core.autocrlf=true turns the committed LF into CRLF on the Windows dev box (and the sidecar with it); a digest over raw bytes
    would then refuse a faithful copy.  The canonical form folds CRLF to LF, a one-digit edit still changes it."""
    p = _copy_deploy(tmp_path)
    raw = p.read_bytes()
    assert b"\r" not in raw
    p.write_bytes(raw.replace(b"\n", b"\r\n"))
    side = Path(str(p) + ".sha256")
    side.write_bytes(side.read_bytes().replace(b"\n", b"\r\n"))
    assert RS.load_gains_table(p)["gains"].tolist() == GAINS


# ----------------------------------------------------------------------------------- the forward-out adapter
def _fake_out(B=4, S=8, seed=11):
    plan, prior = _zigzag(B, S, seed).astype(np.float32), _zigzag(B, S, seed + 1).astype(np.float32)
    return {"traj": plan, "residual_prior_path": prior, "residual_prior_ctrl": np.zeros((B, 2), np.float32)}, plan, prior


MODEL_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])     # anchor_horizons (5, 10, 15, 20, 30, 40, 50, 60) * anchor_dt (0.1)


def test_retime_from_forward_out_selects_the_surface_slots_and_leaves_out_alone():
    out, plan, prior = _fake_out()
    snap = {k: v.copy() for k, v in out.items()}
    tab = RS.load_deploy_table()
    for slots in ([0, 1, 2, 3], [1, 3, 4, 5, 6, 7]):                       # S2 and S6 (model slots of those instants)
        got = RS.retime_from_forward_out(out, MODEL_T, tab, slots)
        want = RS.retime_plan(plan[:, slots], prior[:, slots], MODEL_T[slots], tab)
        assert got.shape == (4, len(slots), 2) and got.tobytes() == want.tobytes()
    assert RS.retime_from_forward_out(out, MODEL_T, tab).shape == (4, 8, 2)  # slots=None: all eight (NOT a scored function)
    for k in out:
        assert out[k].tobytes() == snap[k].tobytes()                         # the planner's own plan is never mutated


def test_retime_from_forward_out_refuses_an_off_prior_build_and_a_wrong_grid():
    out, _, _ = _fake_out()
    tab = RS.load_deploy_table()
    off = {"traj": out["traj"]}
    with pytest.raises(RS.RetimeStageError, match="residual_prior_path"):
        RS.retime_from_forward_out(off, MODEL_T, tab)
    with pytest.raises(RS.RetimeStageError):
        RS.retime_from_forward_out(out, MODEL_T[:5], tab)                    # instants do not match the model's slot axis


def test_retime_from_forward_out_reads_torch_tensors_without_importing_torch_itself():
    torch = pytest.importorskip("torch")
    out, plan, prior = _fake_out()
    tout = {k: torch.from_numpy(v).requires_grad_(k == "traj") for k, v in out.items()}   # a graph-attached plan must not break it
    tab = RS.load_deploy_table()
    got = RS.retime_from_forward_out(tout, MODEL_T, tab, [0, 1, 2, 3])
    assert isinstance(got, np.ndarray)
    assert got.tobytes() == RS.retime_plan(plan[:, :4], prior[:, :4], MODEL_T[:4], tab).tobytes()


def test_the_stage_module_does_not_import_torch():
    src = (STACK / "tanitad" / "retime_stage.py").read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src


# ------------------------------------------------------------ the deployed file == what the builder regenerates
def _pkg_dir() -> Path | None:
    cands = [os.environ.get("TANITAD_R8_BATTERY_PKG"), str(STACK.parent / PKG_REL)]
    for c in cands:
        if c and (Path(c) / "code" / "a0k0_headroom_retime_r8.py").is_file() and (Path(c) / "raw" / "retime_trainfit").is_dir():
            return Path(c)
    return None


def _need_pkg() -> Path:
    p = _pkg_dir()
    if p is None:
        pytest.skip("the refcv8 milestone-battery eval package is not reachable (set TANITAD_R8_BATTERY_PKG to its directory)")
    return p


def _builder():
    spec = importlib.util.spec_from_file_location("_build_retime_deploy_table", STACK / "scripts" / "build_retime_deploy_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_deployed_file_is_byte_identical_to_the_builders_regeneration():
    frozen = _need_pkg() / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
    b = _builder()
    assert b.main(["--src", str(frozen), "--out", str(RS.DEPLOY_TABLE_PATH), "--check"]) == 0


def test_MUTATION_a_retyped_float_with_a_recomputed_sidecar_is_caught_by_the_regeneration_check(tmp_path):
    frozen = _need_pkg() / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
    b = _builder()
    p = _copy_deploy(tmp_path)
    text = p.read_text(encoding="utf-8").replace("0.6192916835387219", "0.619291683538722")   # "rounded" by hand
    p.write_text(text, encoding="utf-8", newline="\n")
    Path(str(p) + ".sha256").write_text(f"{b.sha256_lf(p.read_bytes())}  {p.name}\n", encoding="utf-8", newline="\n")
    assert RS.load_gains_table(p)["gains"][1] == 0.619291683538722            # the sidecar is consistent: load_gains_table accepts it ...
    assert b.main(["--src", str(frozen), "--out", str(p), "--check"]) == 2    # ... and the byte compare against the frozen source does not


def test_the_builder_refuses_a_frozen_file_that_is_not_the_pinned_one(tmp_path):
    frozen = _need_pkg() / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
    b = _builder()
    bad = tmp_path / "FROZEN_GAINS_RETIME_train.json"
    bad.write_bytes(frozen.read_bytes().replace(b"0.9673903790217738", b"0.9673903790217739"))
    with pytest.raises(SystemExit, match="frozen digest"):
        b.main(["--src", str(bad), "--out", str(tmp_path / "o.json")])


# ----------------------------------------------------------------------------------- EQUIVALENCE on banked windows
ARCHIVE = Path(os.environ.get("TANITAD_R8_ARCHIVE", "C:/Users/Admin/refcv8_eval_archive/battery"))
TREE = "step30000ema_unc"
W_OBS = 8
_BANK: dict = {}


def _reference_module():
    """Import the eval package's reference module WITHOUT its import-time side effects leaking into the rest of the suite: it
    ``setdefault``s CUDA_VISIBLE_DEVICES=-1 / OMP_NUM_THREADS=4 and puts its own code directory first on ``sys.path``."""
    pkg = _need_pkg()
    saved_env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    saved_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_a0k0_headroom_retime_r8_ref", pkg / "code" / "a0k0_headroom_retime_r8.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path[:] = saved_path
    return mod


def _banked_surface(sub: str):
    """(plan `os` [n, T, 2] float32, prior ha0_ext_pose path [n, T, 2] float32, instants_s [T]) of one banked panel, built the way the
    reference builds them: `os` from the panel's ep*.npz, the prior from `kinematic_prior.prior_controls('ha0_ext_pose', window)` /
    `prior_path` over the observed pose window ws-7..ws of each window (decisions/ep*.npz)."""
    if sub in _BANK:
        return _BANK[sub]
    torch = pytest.importorskip("torch")
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    d = ARCHIVE / TREE / sub
    if not (d / "manifest.json").is_file():
        pytest.skip(f"banked panel {d} not present (set TANITAD_R8_ARCHIVE)")
    grid = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["grid"]
    hz = [int(h) for h in grid["horizons_steps"]]
    files = sorted(d.glob("ep*.npz"))
    if not files:
        pytest.skip(f"no ep*.npz under {d}")
    plans, wins = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        dd = np.load(d / "decisions" / f.name, allow_pickle=True)
        ep, ws = dd["ep_poses"], dd["ws"].astype(np.int64)
        idx = ws[:, None] + np.arange(-(W_OBS - 1), 1)[None, :]
        assert idx.min() >= 0 and idx.max() < ep.shape[0]
        plans.append(np.asarray(z["os"]))
        wins.append(ep[idx])
    O = np.concatenate(plans)
    pw = torch.from_numpy(np.ascontiguousarray(np.concatenate(wins), dtype=np.float32))
    a0, kp = KP.prior_controls("ha0_ext_pose", pw)
    v0 = ego_channels_from_poses(pw, W_OBS, dt=0.1)[:, -1, 0]
    B = KP.prior_path(a0, kp, v0, hz).numpy().astype(np.float32)
    assert O.shape == B.shape and O.shape[1] == len(hz) and O.dtype == np.float32
    _BANK[sub] = (O, B, np.asarray(grid["instants_s"], np.float64))
    return _BANK[sub]


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def test_EQUIVALENCE_S2_matches_the_reference_retime_apply_on_banked_windows():
    ref = _reference_module()
    O, B, t = _banked_surface("panel_s0")
    assert t.tolist() == [0.5, 1.0, 1.5, 2.0] and O.shape[0] > 1000
    tab = RS.load_deploy_table()
    mine = RS.retime_plan(O, B, t, tab, cast=False)
    gains = np.asarray(GAINS[:4], np.float64)                                  # RETIME_TRAINFIT's S2 gains == the deploy knots there
    theirs = ref.retime_apply(O, B, gains, cast=False)
    err = _max_err(mine, theirs)
    print(f"[equivalence S2] n={O.shape[0]} max|mine - reference| = {err!r} m")
    assert err <= 1e-9
    assert _max_err(RS.retime_plan(O, B, t, tab), ref.retime_apply(O, B, gains, cast=True)) <= 1e-9     # the float32-cast path too
    assert np.abs(mine - O.astype(np.float64)).max() > 0.1                      # the comparison is not vacuous: the stage moved points
    # MUTATION: the same gains read as 1 - g (a sign slip in the gain direction) must leave the tolerance by a wide margin
    mutated = ref.retime_apply(O, B, 1.0 - gains, cast=False)
    assert _max_err(mine, mutated) > 1e-3


def test_EQUIVALENCE_S6_matches_the_reference_retime_apply_on_banked_windows():
    ref = _reference_module()
    O, B, t = _banked_surface("s6_s0")
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] and O.shape[0] > 1000
    tab = RS.load_deploy_table()
    mine = RS.retime_plan(O, B, t, tab, cast=False)
    # the DEPLOYED S6 gains: S2 knots at 1 s / 2 s, S6 knots at 3..6 s (typed literal vector, see GAINS)
    gains = np.asarray([GAINS[1], GAINS[3], GAINS[4], GAINS[5], GAINS[6], GAINS[7]], np.float64)
    err = _max_err(mine, ref.retime_apply(O, B, gains, cast=False))
    print(f"[equivalence S6] n={O.shape[0]} max|mine - reference| = {err!r} m")
    assert err <= 1e-9
    assert _max_err(mine, ref.retime_apply(O, B, 1.0 - gains, cast=False)) > 1e-3
    # and with the REFERENCE's own frozen per-surface S6 gains the geometry still agrees (the table is the only thing that differs)
    frozen = json.loads((_need_pkg() / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json").read_text(encoding="utf-8"))
    g6 = np.asarray(frozen["surfaces"]["S6"]["all"]["RETIME"], np.float64)
    assert _max_err(RS.retime_plan(O, B, t, _per_instant(t, g6), cast=False), ref.retime_apply(O, B, g6, cast=False)) <= 1e-9


def _per_instant(t, g):
    return {"knots_s": t.tolist(), "gains": np.asarray(g).tolist()}


def test_EQUIVALENCE_the_stage_keeps_the_banked_path_and_changes_only_the_progress():
    O, B, t = _banked_surface("panel_s0")
    out = RS.retime_plan(O, B, t, RS.load_deploy_table(), cast=False)
    rng = np.random.default_rng(0)
    for i in rng.choice(O.shape[0], 200, replace=False):                       # a sample: the point-to-polyline loop is pure Python
        for k in range(4):
            assert _dist_to_polyline(out[i, k], O[i].astype(np.float64)) < 1e-6
