"""⭐⭐ ``ha0_ext`` IS ONE IMPLEMENTATION, SHARED BY BOTH HARNESSES (Rung A1).

WHY THIS FILE EXISTS. refcv5's acceptance bar is *"beat BOTH ``ha`` and
``ha0_ext``"*. MEASURED 2026-09-05, with a same-breath control so the reading is
about the file and not about the flapping G: mount::

    grep -c ha0_ext      taniteval/tools/refcv3_arm.py  ->  0   (the target)
    grep -c add_argument taniteval/tools/refcv3_arm.py  -> 28   (the control)
    grep -c ha0_ext      taniteval/tools/refav1_arm.py  -> 12   (the sibling)

so half the bar was UNREADABLE on the REF-C surface: the harness that produces
that surface never computed the control. The fix is a PORT, not a second
implementation — ``stack/tanitad/eval/echo_gate.py::ha0_ext``'s own docstring
states the rule this file enforces:

    *"ONE implementation, shared with the model's own E14 base — deliberately.
    A control re-implemented beside the thing it controls is a control that can
    drift away from it, and then the gate measures the drift instead of the
    model."*

WHAT IS PINNED (each assertion is a failure mode, not a formality)

  (1) THE FUNCTION refcv3_arm CALLS IS **DEFINED IN refav1_arm.py**. Checked by
      ``__code__.co_filename``, i.e. by where the code actually lives — not by a
      name that could be shadowed by a local copy.
  (2) refcv3_arm.py DEFINES NO ha0_ext DERIVATION OF ITS OWN. A port that
      "temporarily" inlines the two lines is exactly how the second
      implementation appears.
  (3) THE LEGACY refav1 READ IS BIT-UNCHANGED by the ``stride`` generalisation:
      the default ``stride=2`` reproduces the pre-Rung-A1 formula EXACTLY, so no
      banked refav1 number moves.
  (4) BOTH HARNESSES PRODUCE THE SAME ``ha0_ext`` ON THE SAME PHYSICAL INPUT —
      the assertion that makes "same shared call" true rather than claimed. The
      curvature read is bit-identical (same index); the acceleration agrees to
      float tolerance because one divides by 0.2 and the other by 0.1.
  (5) THE WHOLE PATH AGREES, not only the controls: integrating those controls
      through the programme's ONE unicycle gives the same trajectory at the
      shared instants.
  (6) THE ARM IS REGISTERED — tier stamp T1, a meaning, the default arm list,
      and the paired block that makes the bar readable. An arm computed but not
      emitted is the same absence in a new costume.

⛔ CPU only; no checkpoint, no pod, no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "taniteval" / "tools"
sys.path.insert(0, str(_REPO / "stack" / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("refcv3_arm_ha0ext_test", _TOOLS / "refcv3_arm.py")
ra = _load("refav1_arm_ha0ext_test", _TOOLS / "refav1_arm.py")

DT_REFCV3 = 0.1          # the corpus's native tick — refcv3 windows on it
DT_REFAV1 = 0.2          # refav1's operative tick — it windows every 2nd frame


@pytest.fixture(scope="module")
def series():
    """A 0.1 s speed/steer series both harnesses can index.

    The speed is a CONSTANT-ACCELERATION ramp on purpose: only then do the two
    backward differences — refav1's over 2 frames / 0.2 s and refcv3's over
    1 frame / 0.1 s — describe the SAME physical acceleration, which is the
    precondition for comparing them at all.
    """
    n = 41
    i = torch.arange(n, dtype=torch.float32)
    v = 8.0 + 0.5 * i                       # a = 5.0 m/s^2, exact in binary
    kap = 0.03 * torch.sin(i / 7.0) + 0.004
    return v, kap


# =========================================================================== #
# (1)+(2) one implementation, and it does not live in refcv3_arm.py            #
# =========================================================================== #
def test_the_function_refcv3_calls_is_defined_in_refav1_arm():
    fn = rc.ra.hold_ext_controls
    where = Path(fn.__code__.co_filename).name
    assert where == "refav1_arm.py", (
        f"refcv3_arm's ha0_ext derivation is defined in {where}, not in "
        f"refav1_arm.py — a second implementation of a shared kinematic is how "
        f"two 'independent' checks come to agree on a wrong answer")


def test_refcv3_arm_defines_no_ha0_ext_derivation_of_its_own():
    src = (_TOOLS / "refcv3_arm.py").read_text(encoding="utf-8")
    # same-breath control: a token we KNOW is in the file, so a 0 below is a
    # fact about the file and not about a failed read.
    assert src.count("def hold_v0_controls") == 1, "control read failed"
    for banned in ("def hold_ext_controls", "def ha0_ext", "def _ha0_ext"):
        assert banned not in src, (
            f"refcv3_arm.py defines {banned!r} — the port must CALL the shared "
            f"derivation, never restate it")
    assert src.count("ra.hold_ext_controls(") == 1, (
        "expected exactly one call to the shared derivation in the roll")
    assert "ra.hold_ext_controls.__doc__" in src, (
        "the manifest must stamp the SHARED function's own docstring, not a "
        "paraphrase of it")


# =========================================================================== #
# (3) the stride generalisation moved no banked refav1 number                  #
# =========================================================================== #
def test_default_stride_reproduces_the_legacy_refav1_formula_bit_exactly(series):
    v, kap = series
    for t in (1, 5, 9, 15):
        got = ra.hold_ext_controls(None, v, kap, t, dt=DT_REFAV1)
        want = torch.stack([(v[2 * t] - v[2 * t - 2]) / DT_REFAV1,
                            kap[2 * t]]).float()
        assert torch.equal(got, want), f"legacy refav1 read moved at t={t}"


def test_the_guard_still_refuses_a_window_with_no_closed_step(series):
    v, kap = series
    with pytest.raises(ValueError):
        ra.hold_ext_controls(None, v, kap, 0, dt=DT_REFAV1)
    with pytest.raises(ValueError):
        ra.hold_ext_controls(None, v, kap, 0, dt=DT_REFCV3, stride=1)


# =========================================================================== #
# (4) THE CROSS-HARNESS ASSERTION — same input, same ha0_ext                   #
# =========================================================================== #
def test_both_harnesses_read_the_same_ha0_ext_on_the_same_input(series):
    """refav1 at (stride 2, dt 0.2) and refcv3 at (stride 1, dt 0.1) index the
    SAME origin frame ``i0 = 2t`` and must describe the same state there."""
    v, kap = series
    for t in (1, 4, 9, 17):
        i0 = 2 * t
        av1 = ra.hold_ext_controls(None, v, kap, t, dt=DT_REFAV1)
        cv3 = ra.hold_ext_controls(None, v, kap, i0, dt=DT_REFCV3, stride=1)
        # the curvature channel is the SAME recorded value at the SAME index:
        # nothing to round, so bit-exact or the port read the wrong frame.
        assert torch.equal(cv3[1], av1[1]), (
            f"the two harnesses read different curvature at i0={i0}")
        assert torch.equal(cv3[1], kap[i0].float()), "kappa is not read AT t0"
        # the acceleration is the same physical quantity through two different
        # divisors (0.2 vs 0.1), so equality is to float tolerance, not bits.
        assert torch.allclose(cv3[0], av1[0], rtol=1e-6, atol=1e-6), (
            f"a0 disagrees at i0={i0}: refcv3 {float(cv3[0])} vs refav1 "
            f"{float(av1[0])}")
        assert torch.allclose(cv3[0], torch.tensor(5.0), rtol=1e-5), (
            "the ramp's own acceleration is not recovered — the test's premise "
            "is broken, not the code")


def test_the_read_consumes_nothing_after_t0(series):
    """The whole point of ``ha0_ext`` as a T1 control: perturbing every frame
    STRICTLY AFTER the origin leaves it bit-identical."""
    v, kap = series
    i0 = 20
    before = ra.hold_ext_controls(None, v, kap, i0, dt=DT_REFCV3, stride=1)
    v2, kap2 = v.clone(), kap.clone()
    v2[i0 + 1:] += 7.0
    kap2[i0 + 1:] += 0.5
    after = ra.hold_ext_controls(None, v2, kap2, i0, dt=DT_REFCV3, stride=1)
    assert torch.equal(before, after), "a recorded FUTURE value reached ha0_ext"


# =========================================================================== #
# (5) the trajectory, not only the controls                                    #
# =========================================================================== #
def test_the_integrated_path_is_the_same_through_the_one_unicycle(series):
    v, kap = series
    i0, v0, n = 20, 11.5, 20
    grid = {"horizons_steps": [5, 10, 15, 20]}
    ext = ra.hold_ext_controls(None, v, kap, i0, dt=DT_REFCV3, stride=1)
    got = rc.integrate_select(ext[None].expand(n, 2), v0, grid,
                              action_units="steer")
    ref = ra.paths_from_controls(ext[None].expand(n, 2), v0, rc.DT_FRAME, n,
                                 action_units="steer")
    idx = torch.tensor([h - 1 for h in grid["horizons_steps"]])
    assert torch.equal(torch.as_tensor(got), ref[:, idx].float().cpu()), (
        "refcv3's index-select of the shared unicycle is not the refav1 path")


def test_ha0_ext_is_not_the_constant_velocity_floor(series):
    """If it were, the bar would have ONE control in it, not two."""
    v, kap = series
    i0, v0, n = 20, 11.5, 20
    grid = {"horizons_steps": [5, 10, 15, 20]}
    ext = ra.hold_ext_controls(None, v, kap, i0, dt=DT_REFCV3, stride=1)
    ha0_ext = rc.integrate_select(ext[None].expand(n, 2), v0, grid,
                                  action_units="steer")
    ha0 = rc.integrate_select(rc.hold_v0_controls(n), v0, grid,
                              action_units="steer")
    assert not torch.equal(torch.as_tensor(ha0_ext), torch.as_tensor(ha0)), (
        "ha0_ext collapsed onto ha0 on a series with non-zero a0 AND k0")


# =========================================================================== #
# (6) computed IS NOT emitted — the arm has to reach the record                 #
# =========================================================================== #
def test_the_arm_is_registered_stamped_and_paired():
    assert rc.ARM_TIERS.get("ha0_ext") == "T1"
    assert "ha0_ext" in rc.ARM_MEANING
    src = (_TOOLS / "refcv3_arm.py").read_text(encoding="utf-8")
    assert 'arms = ["os", "ha", "ha0", "ha0_ext"]' in src, (
        "ha0_ext is not in the DEFAULT arm list — an optional control is one "
        "that will be missing from the run that needs it")
    assert '"paired_os_minus_ha0ext"' in src, (
        "the os - ha0_ext paired block is the half of refcv5's bar this rung "
        "exists to make readable")
    assert '"ha0_ext_rule"' in src and '"ha0_ext_call"' in src, (
        "the manifest must carry the derivation rule and name the shared call")


def test_the_retracted_line_about_ha0_being_the_bar_is_gone():
    """``echo_gate.py`` RETRACTS this file's line by name; the retraction had
    never been applied to the string a reader actually sees."""
    src = (_TOOLS / "refcv3_arm.py").read_text(encoding="utf-8")
    # same-breath control: the phrase below must be searched in a file we can
    # demonstrably read, or its absence is a claim about the read.
    assert src.count("STRONGEST TRIVIAL BASELINE") >= 1, "control read failed"
    ha0_meaning = rc.ARM_MEANING["ha0"]
    assert "the echo test's real bar, and" not in ha0_meaning, (
        "ha0 is still described as 'the echo test's real bar' — "
        "stack/tanitad/eval/echo_gate.py retracts exactly that phrase")
    assert "RETRACT" in ha0_meaning


def test_the_shared_kinematic_module_the_provenance_names_really_exists():
    """The manifest cites ``echo_gate.ha0_ext``; a citation to a module that
    does not import is a claim about the internet, not about a file we hold."""
    from tanitad.eval.echo_gate import ha0_ext          # noqa: F401
    out = ha0_ext(torch.tensor([10.0]), torch.tensor([0.5]),
                  torch.tensor([0.01]), [1.0, 2.0])
    assert out.shape[0] == 1 and out.shape[1] == 2
