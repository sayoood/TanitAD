"""The trainer side of the per-clip camera, and the two refusals it adds.

Every gate is shown to FAIL its defect.

* **R1** ``--agent-rig-extrinsics`` now reads a PER-CLIP table as well as a
  single mount pose, and ``config.json`` states which one
  (``mount_pose_scope``). Without that word a reader cannot tell a one-camera
  arm from a per-clip one -- retraction class **C28** ("a constant where the
  quantity is per-clip") in a config-record costume.
* **R2** a per-clip table that does not cover the run's episodes REFUSES.
  The uncovered rows would get no camera, both monocular terms would skip
  them (counted only in the log row), and ``config.json`` would still read
  ``mount_pose_scope: PER-CLIP``.
* **R3** ``--agent-w-ground > 0`` REFUSES, because the term is a MEASURED
  tautology (loss 2.6e-08, gradient 8.7e-11) -- and the refusal is itself a
  measurement, so it lifts the day the term becomes real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_v3_train as t                                    # noqa: E402
from tanitad.data import calib                               # noqa: E402
from tanitad.data.rig_projection import RigCamera            # noqa: E402
from tanitad.refs import refc_agents as ra                   # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

FRAME = calib.PHYSICALAI_WIDE120_256x640

#: three clip_ids and three DIFFERENT mount poses -- the real values from the
#: repo's own banked table, which straddles the old guard band (1.43, 1.56).
CLIPS = {
    "d85682b8-f612-4aa3-90a2-ee2bb86151e6": dict(
        qx=0.5046319078166642, qy=-0.5000661210092555,
        qz=0.4978822491501181, qw=-0.4973869501839488,
        x=2.1286449432373047, y=-0.02760353498160839, z=1.575823426246643),
    "ca11a2a2-1021-475c-9907-e6dbe658ad12": dict(
        qx=-0.4950537810225835, qy=0.5005944140937533,
        qz=-0.5012295706851077, qw=0.5030863782137199,
        x=1.9963752031326294, y=-0.0052909464575350285, z=1.296807885169983),
    "6ed4ef7a-1a70-48c5-8dc1-3c6a48ec3ecc": dict(
        qx=-0.5046292082912891, qy=0.5065287522508706,
        qz=-0.4979041719124624, qw=0.4907844953486693,
        x=2.0102603435516357, y=0.0010222510900348425, z=1.2922037839889526),
}
SINGLE = dict(CLIPS["ca11a2a2-1021-475c-9907-e6dbe658ad12"])


def _args(*extra):
    return t.build_parser().parse_args(
        ["--out", "x", "--arm", "hier", "--image-hw", "256", "640", *extra])


def _cfg(args):
    return t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), args)


def _write(tmp, obj, name):
    p = tmp / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


class _M:
    pass


class _Ep:
    def __init__(self, i):
        self.episode_id = i


class _DS:
    def __init__(self, ids):
        self.episodes = [_Ep(i) for i in ids]


def _bank_over(ids):
    cam = RigCamera.nominal(FRAME, height_m=1.4)
    return ra.RigCameraBank(by_episode={int(i): cam for i in ids})


def _model_with(cam):
    m = _M()
    m._rig_camera = cam
    return m


# =========================================================================
# R1 -- the table is read, and the record SAYS which shape it was
# =========================================================================

def test_a_PER_CLIP_table_builds_a_bank_and_stamps_the_scope(tmp_path):
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-rig-extrinsics", _write(tmp_path, CLIPS, "tab.json"),
              "--agent-w-project", "0.2")
    cam, stamp = t._build_rig_camera(_cfg(a), a)
    assert isinstance(cam, ra.RigCameraBank)
    assert len(cam) == 3
    assert stamp["mount_pose_scope"] == "PER-CLIP"
    assert stamp["n_clips"] == 3
    assert stamp["key"] == "stable_episode_id(clip_id)"
    # the spread is IN the record, so a reader sees it was never a constant
    assert stamp["height_m_min"] < 1.43
    assert stamp["height_m_max"] > 1.56


def test_a_SINGLE_extrinsic_is_still_accepted_and_says_so(tmp_path):
    """The converse control. One camera is a legal coarse arm; what was
    missing was the WORD that distinguishes it, not the capability."""
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-rig-extrinsics", _write(tmp_path, SINGLE, "one.json"),
              "--agent-w-project", "0.2")
    cam, stamp = t._build_rig_camera(_cfg(a), a)
    assert not isinstance(cam, ra.RigCameraBank)
    assert stamp["mount_pose_scope"] == "SINGLE-CAMERA-WHOLE-CORPUS"
    assert stamp["n_clips"] == 1


def test_every_camera_source_declares_a_scope():
    """A key present on only some branches rots the first time it is grepped."""
    for extra, want in (
            ((), "NONE"),
            (("--agents", "oracle", "--agent-rig-camera", "nominal",
              "--agent-w-project", "0.2"), "SINGLE-CAMERA-WHOLE-CORPUS")):
        a = _args(*extra)
        _, stamp = t._build_rig_camera(_cfg(a), a)
        assert stamp["mount_pose_scope"] == want


def test_the_batch_carries_the_episode_id_the_bank_is_keyed_on():
    """Without `agent_ep` in the batch a per-clip camera cannot reach the
    loss, and the run silently falls back to one mount pose."""
    src = Path(t.__file__).read_text(encoding="utf-8")
    assert '"agent_ep": torch.tensor(eid, dtype=torch.long)' in src
    assert "_resolve_rig_cameras" in src
    # and the ids are selected with the SAME mask as the targets
    assert "ep_ag.index_select(" in src


# =========================================================================
# R2 -- an uncovered corpus REFUSES
# =========================================================================

def test_a_PARTIAL_bank_REFUSES_and_names_the_gap():
    with pytest.raises(SystemExit, match="covers 2/4 episodes"):
        t.assert_rig_camera_covers(_model_with(_bank_over([1, 2])),
                                   _DS([1, 2, 3, 4]), _args())


def test_the_gap_can_be_accepted_BY_NAME_and_is_then_stamped():
    out = t.assert_rig_camera_covers(
        _model_with(_bank_over([1, 2])), _DS([1, 2, 3, 4]),
        _args("--agent-rig-extrinsics-allow-partial"))
    cov = out["coverage"]
    assert cov["n_covered"] == 2 and cov["n"] == 4 and cov["frac"] == 0.5
    assert cov["allow_partial"] is True


def test_a_FULLY_covering_bank_passes_and_still_reports_its_coverage():
    """The converse control: the refusal must fire on the gap only."""
    out = t.assert_rig_camera_covers(_model_with(_bank_over([1, 2, 3])),
                                     _DS([1, 2, 3]), _args())
    assert out["coverage"]["frac"] == 1.0
    assert out["coverage"]["n_missing"] == 0


def test_a_single_camera_run_is_untouched_by_the_coverage_check():
    """Every banked arm passed one camera. The check must be a no-op there."""
    m = _model_with(RigCamera.nominal(FRAME))
    assert t.assert_rig_camera_covers(m, _DS([1, 2, 3]), _args()) == {}


# =========================================================================
# R3 -- the ground prior refuses, and the refusal is a MEASUREMENT
# =========================================================================

def test_w_ground_REFUSES_because_the_term_trains_nothing():
    """THE DELIBERATE REGRESSION, and it is the config an arm would actually
    have run: a camera IS built (M18 satisfied) and the term is still
    identically zero."""
    a = _args("--agents", "oracle", "--agent-rig-camera", "nominal",
              "--agent-w-ground", "0.5")
    with pytest.raises(SystemExit, match="TAUTOLOGY"):
        t.assert_ground_prior_is_supervised(
            _model_with(RigCamera.nominal(FRAME, height_m=1.5)), a)


def test_the_ground_refusal_does_NOT_fire_on_the_arms_that_ran():
    """The converse control, in both directions that matter."""
    a = _args("--agents", "oracle", "--agent-rig-camera", "nominal",
              "--agent-w-project", "0.2")
    cam = RigCamera.nominal(FRAME, height_m=1.5)
    assert t.assert_ground_prior_is_supervised(
        _model_with(cam), a)["checked"] is False
    assert t.assert_ground_prior_is_supervised(
        _model_with(None), a)["checked"] is False


def test_the_ground_probe_reports_its_own_positive_CONTROL():
    """A flat gradient is indistinguishable from a probe that never ran, so
    the control must be shown to be alive."""
    src = Path(t.__file__).read_text(encoding="utf-8")
    fn = src.split("def assert_ground_prior_is_supervised", 1)[1]
    fn = fn.split("\ndef ", 1)[0]
    assert "monocular_projection_loss" in fn, "no positive control in the probe"
    assert "INCONCLUSIVE" in fn, "a flat control must refuse, not pass"
    assert "control_grad_absmax_project" in fn


def test_a_per_clip_bank_is_probed_through_one_of_its_own_cameras(tmp_path):
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-rig-extrinsics", _write(tmp_path, CLIPS, "tab.json"),
              "--agent-w-ground", "0.5")
    cam, _ = t._build_rig_camera(_cfg(a), a)
    with pytest.raises(SystemExit, match="TAUTOLOGY"):
        t.assert_ground_prior_is_supervised(_model_with(cam), a)
