"""The three refcv5 provenance defects, pinned — mm-decisions M17 / M18.

Each test names the MEASURED defect it exists to prevent, and every gate here
is shown to FAIL the defect: a control that has never read the wrong value
certifies nothing.

* **P1** ``--agent-w-project`` / ``--agent-w-ground`` were SILENT NO-OPS —
  ``model._rig_camera`` was set to ``None`` and never assigned, while both
  weights were stamped into ``config.json``. The run record stated a training
  configuration that did not happen.
* **P2** a run must be able to state every ``--agent-*`` / ``--w-*`` knob it
  trained at, and the assertion is DERIVED FROM ARGPARSE so it cannot rot.
* **P3** ``--agent-queries`` is 100, not 32; 32 came from val40 (max 24) and
  drops 41,362 boxes on train with the nearest sacrifice at 13.1 m.
* **P4** a recorded hash carries the ARTIFACT it covers, or it is a number.
"""
from __future__ import annotations

import ast
import json
import lzma
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_v3_train as t                                    # noqa: E402
from tanitad.data import join_meta as jm                     # noqa: E402
from tanitad.refs import refc_agents as ra                   # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

TRAINER_SRC = Path(t.__file__)


def _args(*extra):
    return t.build_parser().parse_args(["--out", "x", "--arm", "hier",
                                        *extra])


def _cfg_at(hw=(256, 640), args=None):
    """A config at a DECLARED corpus geometry, pinned through the trainer's own
    helper so the test exercises the shipped path, not a parallel one.

    ⚠️ ``_pin_trainer_cfg`` is where the rig-camera refusals fire, so a test
    that expects a refusal wraps THIS call — the point of P1 being that the
    failure happens at config-pin time, before a model and before the GPU.
    """
    a = args if args is not None else _args("--image-hw", str(hw[0]),
                                            str(hw[1]))
    return t._pin_trainer_cfg(v3.refc_v3_sized_config("tiny", hier=True), a), a


# =========================================================================
# P1 — the flag REFUSES; it does not parse, stamp, and do nothing
# =========================================================================

@pytest.mark.parametrize("flag", ["--agent-w-project", "--agent-w-ground"])
def test_P1_weight_without_a_camera_REFUSES(flag):
    """⛔⛔ THE DELIBERATE-REGRESSION CONTROL. MEASURED 2026-09-05 (M18):
    ``refc_v3_train`` set ``model._rig_camera = None`` and never assigned it,
    ``agent_losses`` guards both monocular terms on ``cam is not None``, and
    both weights were written into ``config.json`` regardless. A run could
    therefore stamp ``--agent-w-project 0.2`` and train NOTHING.

    This is the config that used to run silently. It must now raise."""
    a = _args("--agents", "oracle", flag, "0.2")
    with pytest.raises(SystemExit) as e:
        t._pin_refcv5_seams(v3.refc_v3_smoke_config(True), a)
    msg = str(e.value)
    assert "--agent-rig-camera" in msg
    assert "config.json" in msg          # the refusal names WHY it matters


def test_P1_zero_weights_without_a_camera_are_fine():
    """The converse control: the refusal must fire on the LYING config only.
    Every existing arm runs with both weights 0 and must be unaffected."""
    a = _args("--agents", "oracle")
    cfg = v3.refc_v3_smoke_config(True)
    t._pin_refcv5_seams(cfg, a)                       # no raise
    cam, stamp = t._build_rig_camera(cfg, a)
    assert cam is None and stamp["source"] == "off"


def test_P1_a_camera_with_agents_off_REFUSES():
    """⛔ NO THIRD STATE IN THE OTHER DIRECTION EITHER. The camera is consumed
    only by ``agent_losses``, which is not called without an agent seam, so a
    camera built under ``--agents off`` would itself be a flag that parses and
    does nothing."""
    a = _args("--agent-rig-camera", "nominal")
    with pytest.raises(SystemExit, match="--agents off"):
        t._pin_refcv5_seams(v3.refc_v3_smoke_config(True), a)


@pytest.mark.parametrize("argv,needle", [
    (["--w-agent", "1.0", "--agent-join", "j.jsonl"], "w_agent"),
    (["--agent-w-project", "0.2"], "agent_w_project"),
    (["--agent-w-ground", "0.2"], "agent_w_ground"),
    (["--agent-rig-camera", "nominal"], "--agent-rig-camera"),
])
def test_P1_every_agent_weight_under_agents_off_REFUSES(argv, needle):
    """\u26d4\u26d4 THE FOURTH DEAD-FLAG DIRECTION, AND ONE OF THEM WAS SILENT IN A
    WAY THE M18 AUDIT DID NOT NAME. With ``--agents off`` no ``AgentSeamConfig``
    is built, so the model emits no ``agent_slots`` and ``agent_losses`` is
    never called \u2014 every agent weight parses, is stamped into ``config.json``
    and trains nothing.

    \u26a0\ufe0f ``--w-agent 1.0`` WITH a join is the dangerous one: the join puts
    ``agent_box`` into the batch, so ``compute_losses_v3``'s existing guard
    (which fires only when ``agent_box`` is ABSENT) does not fire, and the
    detection loss is simply skipped. That is the `--agent-w-project` defect
    one flag over, and it is now a refusal."""
    a = _args("--agents", "off", *argv)
    with pytest.raises(SystemExit, match="--agents off") as e:
        t._pin_refcv5_seams(v3.refc_v3_smoke_config(True), a)
    assert needle in str(e.value)


def test_P1_agents_off_with_DEFAULT_weights_is_untouched():
    """The converse control: `--agents off` is the default arm, and it must
    stay exactly as green as it was for every banked run."""
    a = _args("--agents", "off")
    cfg = v3.refc_v3_smoke_config(True)
    t._pin_refcv5_seams(cfg, a)                        # no raise
    assert getattr(cfg.core, "agents", None) is None or \
        not cfg.core.agents.enable


def test_P1_unknown_geometry_REFUSES_rather_than_inventing_an_f_ref():
    """⛔ A CanonicalFrame is not its pixel count. This corpus is CYLINDRICAL
    (column linear in azimuth); the pinhole formula reads 92.6 deg for a
    120 deg camera and looks entirely plausible (`CLAUDE.md`). The 64x64 tiny
    rig has no declared frame, so the camera must refuse, not guess."""
    a = _args("--agents", "oracle", "--agent-w-project", "0.2",
              "--agent-rig-camera", "nominal")
    cfg = v3.refc_v3_smoke_config(True)
    assert tuple(cfg.core.encoder.image_hw()) == (64, 64)
    with pytest.raises(SystemExit, match="no canonical frame is DECLARED"):
        t._build_rig_camera(cfg, a)


def test_P1_nominal_camera_is_BUILT_at_a_declared_geometry():
    a = _args("--agents", "oracle", "--agent-w-project", "0.2",
              "--agent-rig-camera", "nominal", "--image-hw", "256", "640")
    cfg, a = _cfg_at(args=a)
    cam, stamp = t._build_rig_camera(cfg, a)
    assert cam is not None
    assert stamp["source"] == "nominal"
    # ⚠️ the provenance, not just the value: `ground_range_prior` back-projects
    # through the road plane, so a pitch-free mount biases its range directly.
    assert stamp["mount_pose"] == "NOMINAL-no-pitch"
    assert stamp["frame"]["projection"] == "cylindrical"
    assert stamp["frame"]["f_ref"] == pytest.approx(305.5774907364391)
    assert stamp["in_measured_height_band"] is True


def test_P1_extrinsics_source_needs_a_FILE():
    """⭐ And note WHERE this raises: inside `_pin_trainer_cfg`, i.e. at
    config-pin time, before a model is built and before the GPU."""
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-w-ground", "0.1", "--image-hw", "256", "640")
    with pytest.raises(SystemExit, match="--agent-rig-extrinsics"):
        _cfg_at(args=a)


def _quat_cam_to_vehicle(pitch_rad: float) -> dict:
    """The ``sensor_extrinsics`` quaternion for a nominal mount pitched by
    ``pitch_rad``: ``R_cam_to_rig = R_nominal @ Rx(theta)``.

    ⚠️ Composed with the axis permutation on purpose. A bare
    "rotate by theta" quaternion is NOT a small pitch here — with
    identity the boresight maps to vehicle UP and the pitch reads -pi/2. Same
    scope error as the rest of this file, in a rotation costume, so the test
    derives it rather than assuming it.
    """
    import math
    import numpy as np
    from tanitad.data.rig_projection import RIG_TO_CAM
    th = float(pitch_rad)
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(th), -math.sin(th)],
                   [0.0, math.sin(th), math.cos(th)]])
    R = np.array(RIG_TO_CAM).T @ rx
    tr = float(np.trace(R))
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw, qx = 0.25 * S, (R[2, 1] - R[1, 2]) / S
        qy, qz = (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx = (R[2, 1] - R[1, 2]) / S, 0.25 * S
        qy, qz = (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S
        qy, qz = 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S
        qy, qz = (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return {"qx": qx, "qy": qy, "qz": qz, "qw": qw}


@pytest.mark.parametrize("pitch", [0.0, 0.05])
def test_P1_extrinsics_json_builds_a_pitched_camera(tmp_path, pitch):
    """The corpus constructor. A quaternion with a real pitch must produce a
    camera whose stamped pitch is that pitch — the number that puts the
    horizon on the right row, and the whole difference from `nominal`.

    ⛔ ``pitch = 0`` is the CONTROL THAT MUST READ A KNOWN VALUE: an
    unpitched extrinsic must reproduce ``RigCamera.nominal`` exactly, so the
    non-zero reading at 0.05 is the pitch and not an offset in the
    instrument."""
    q = dict(_quat_cam_to_vehicle(pitch), x=1.5, y=0.0, z=1.47)
    p = tmp_path / "extr.json"
    p.write_text(json.dumps(q), encoding="utf-8")
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-rig-extrinsics", str(p), "--agent-w-ground", "0.1",
              "--image-hw", "256", "640")
    cfg, a = _cfg_at(args=a)
    cam, stamp = t._build_rig_camera(cfg, a)
    assert cam is not None
    assert stamp["source"] == "extrinsics"
    assert stamp["height_m"] == pytest.approx(1.47)
    assert abs(stamp["optical_axis_pitch_rad"]) == pytest.approx(pitch,
                                                                 abs=1e-9)
    assert stamp["extrinsics_path"] == str(p)
    from tanitad.data.rig_projection import RigCamera
    nominal = RigCamera.nominal(cam.frame, height_m=1.47)
    same = bool(torch.allclose(cam.R_cam_to_rig, nominal.R_cam_to_rig,
                               atol=1e-9))
    assert same is (pitch == 0.0)


def test_P1_extrinsics_json_without_a_quaternion_REFUSES(tmp_path):
    p = tmp_path / "extr.json"
    p.write_text(json.dumps({"z": 1.5}), encoding="utf-8")
    a = _args("--agents", "oracle", "--agent-rig-camera", "extrinsics",
              "--agent-rig-extrinsics", str(p), "--agent-w-ground", "0.1",
              "--image-hw", "256", "640")
    with pytest.raises(SystemExit, match="declares no"):
        _cfg_at(args=a)


def test_P1_the_camera_the_TRAINER_builds_makes_the_terms_COMPUTE():
    """⭐ THE POSITIVE ASSERTION, end to end through the trainer's own builder.
    Without a camera `agent_losses` emits neither term and `total` is the set
    loss alone; with the camera the trainer builds, both terms are present,
    finite, computed over n > 0 and ADD to the total. That is what "the flag
    is no longer a no-op" means, and it FAILS if `_build_rig_camera` ever
    returns None here."""
    torch.manual_seed(0)
    a = _args("--agents", "head", "--w-agent", "1.0", "--agent-join", "j",
              "--agent-w-project", "1.0", "--agent-w-ground", "0.5",
              "--agent-rig-camera", "nominal", "--image-hw", "256", "640")
    cfg, a = _cfg_at(args=a)
    cam, _ = t._build_rig_camera(cfg, a)
    assert cam is not None

    acfg = ra.AgentSeamConfig(enable=True, queries=2, d_model=64, depth=2,
                              enforce_band=False, w_project=1.0, w_ground=0.5)
    box = torch.tensor([[[18.0, 2.0, 4.5, 1.9], [30.0, -3.0, 4.2, 1.8]],
                        [[12.0, 0.5, 4.4, 1.9], [25.0, 4.0, 4.1, 1.8]]])
    # ⭐ DETERMINISTIC slots, offset in RANGE from the targets: a random
    # untrained head puts most boxes outside the frame, and the term then
    # reads a legitimate 0.0 over n = 0 — a no-information value, not evidence
    # that the term works. This test asserts the opposite.
    pred = box.clone()
    pred[..., 0] += 3.0                                # +3 m range error
    slots = {"box": pred, "yaw": torch.zeros(2, 2),
             "yaw_vec": torch.tensor([[[1.0, 0.0]] * 2] * 2),
             "cls_logits": torch.zeros(2, 2, ra.N_AGENT_CLASSES),
             "presence_logit": torch.full((2, 2), 4.0),
             "rates": torch.zeros(2, 2, 3),
             "valid": torch.ones(2, 2, dtype=torch.bool),
             "occ_logit": torch.zeros(2, 2)}
    tgt = {"box": box, "yaw": torch.zeros(2, 2),
           "cls": torch.zeros(2, 2, dtype=torch.long),
           "valid": torch.ones(2, 2, dtype=torch.bool),
           "occ": torch.full((2, 2), -1.0),
           "rates": torch.zeros(2, 2, 3),
           "rates_mask": torch.zeros(2, 2, dtype=torch.bool)}
    blind = ra.agent_losses(slots, tgt, acfg, cam=None)
    seeing = ra.agent_losses(slots, tgt, acfg, cam=cam)
    assert "loss_project" not in blind and "loss_ground" not in blind
    for k in ("loss_project", "loss_ground"):
        assert k in seeing and torch.isfinite(seeing[k]), k
        # n FIRST: a term computed over ZERO items reads 0.0 and is not
        # evidence that it ran.
        assert seeing["n"][k.replace("loss_", "")] > 0, k
        assert float(seeing[k]) > 0.0, k
    assert float(seeing["total"]) > float(blind["total"])
    # ⛔ CONTROL THAT MUST READ A KNOWN VALUE: a PERFECT prediction makes the
    # image-plane term exactly zero, so the non-zero above is an error signal
    # and not an offset baked into the instrument.
    exact = {**slots, "box": box.clone()}
    zero = ra.agent_losses(exact, tgt, acfg, cam=cam)
    assert float(zero["loss_project"]) == 0.0 and zero["n"]["project"] > 0


def test_P1_the_trainer_never_assigns_a_CONSTANT_None_rig_camera():
    """⛔ THE REGRESSION PIN FOR THE ORIGINAL DEFECT, asserted structurally so
    it cannot be reintroduced by a merge: the literal line that caused it was
    ``model._rig_camera = None``. Parsed, not grepped, so formatting cannot
    hide it."""
    tree = ast.parse(TRAINER_SRC.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt_ in node.targets:
            if isinstance(tgt_, ast.Attribute) and tgt_.attr == "_rig_camera":
                if isinstance(node.value, ast.Constant) and \
                        node.value.value is None:
                    bad.append(node.lineno)
    assert not bad, (f"`_rig_camera` is assigned a constant None at lines "
                     f"{bad} — that is the M18 silent-no-op defect.")


# =========================================================================
# P2 — every knob reaches the record, and the check is DERIVED
# =========================================================================

def test_P1_the_two_weights_SURVIVE_the_trip_from_argv_to_the_LOSS_CONFIG():
    """⛔ THE LEG OF THE TRIP NOTHING ELSE WATCHES.

    Every other P1 gate reads ONE end of the wire. The refusals read ``args``;
    ``agent_rig_camera`` in the stamp reads ``args``; the "terms COMPUTE" test
    hand-builds an ``AgentSeamConfig(w_project=1.0)``. But the LOSS reads
    ``cfg.core.agents`` -- and nothing asserted that argv reaches it.

    So dropping the weight in ``_pin_refcv5_seams`` alone would re-open M18 in
    its exact original shape: the camera still builds, the refusal still fires
    correctly, ``config.json['seams']['agent_rig_camera']['w_project']`` still
    reads 0.2 -- and the addend is simply gone. MEASURED 2026-09-05 as UNPINNED
    (the plumbing was correct; the coverage was not).

    The two weights are given DIFFERENT values on purpose: equal ones cannot
    catch a copy-paste that reads ``agent_w_project`` into both.
    """
    a = _args("--agents", "head", "--w-agent", "1.0", "--agent-join", "j",
              "--agent-w-project", "0.2", "--agent-w-ground", "0.1",
              "--agent-rig-camera", "nominal", "--image-hw", "256", "640")
    cfg, a = _cfg_at(args=a)
    ag = cfg.core.agents
    assert ag is not None, "no AgentSeamConfig was built under --agents head"
    assert float(ag.w_project) == 0.2, f"w_project reached the loss config as {ag.w_project}"
    assert float(ag.w_ground) == 0.1, f"w_ground reached the loss config as {ag.w_ground}"
    d = ag.as_dict()
    assert float(d["w_project"]) == 0.2 and float(d["w_ground"]) == 0.1

    # ⭐ THE TWO STAMPS ARE INDEPENDENT READS AND MUST AGREE. `agents`
    # comes from the config the loss uses; `agent_rig_camera` comes from argv.
    # Their agreement is the only thing in config.json that can distinguish a
    # run that trained the term from one that recorded it.
    st = _seam_stamp_of(cfg, a)
    assert float(st["agents"]["w_project"]) == \
        float(st["agent_rig_camera"]["w_project"]), (
            "config.json would state two different w_project values: the loss "
            "config says {} and argv says {}".format(
                st["agents"]["w_project"],
                st["agent_rig_camera"]["w_project"]))
    assert float(st["agents"]["w_ground"]) == \
        float(st["agent_rig_camera"]["w_ground"])

    # ⛔ CONTROL THAT MUST READ A KNOWN VALUE: with the weights at zero the
    # same path must read zero, so the assertions above are reading argv and
    # not a constant baked into the config builder.
    z = _args("--agents", "head", "--w-agent", "1.0", "--agent-join", "j",
              "--image-hw", "256", "640")
    zcfg, _ = _cfg_at(args=z)
    assert float(zcfg.core.agents.w_project) == 0.0
    assert float(zcfg.core.agents.w_ground) == 0.0


def _seam_stamp_of(cfg, a):
    """`_seam_stamp` needs the parser's defaults present on the namespace."""
    return t._seam_stamp(cfg, a)


def test_P2_knob_set_is_derived_from_the_parser_not_listed():
    p = t.build_parser()
    dests = t.agent_knob_dests(p)
    # ⛔ same-breath control: the parser really did yield options (a zero-count
    # "pass" from a broken read is indistinguishable from a genuine absence).
    assert len(p._actions) > 40
    assert len(dests) >= 15
    for must in ("w_agent", "w_u0", "agent_w_project", "agent_w_ground",
                 "agent_queries"):
        assert must in dests, must
    # and nothing that is not an --agent-*/--w-* option leaked in
    assert "lr" not in dests and "seed" not in dests


def _leaf_values(obj) -> list:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _leaf_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _leaf_values(v)
    else:
        out.append(obj)
    return out


def test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE():
    """⭐ DERIVED FROM ARGPARSE, ASSERTED BY VALUE. Each knob is set to a
    distinctive value and must be findable in the seam stamp — so the test
    survives any renaming of the stamp's keys and cannot be satisfied by a
    hand-written list that has rotted."""
    p = t.build_parser()
    # The enabling context, so a knob under test is never refused for a reason
    # that has nothing to do with the knob.
    # ⭐ WP-D's seam is enabled here for exactly the reason `--agents oracle`
    # is: `--w-bev-aux` REFUSES without `--bev-aux col|xcol` (a weight with no
    # head is the `w_agent` defect verbatim — a term stamped into config.json
    # and silently skipped), so without this the probe would have no admissible
    # value and would report the knob unstampable for a reason that has nothing
    # to do with the knob.
    # ⭐ WP-B's seam is enabled here for exactly the same reason WP-D's is:
    # every `--wp-index-*` knob REFUSES under `--wp-index off` (a knob with no
    # seam is the `w_agent` defect verbatim -- parsed, stamped into
    # config.json, and silently inert), so without this the probe would find NO
    # admissible value for `--wp-index-mode` and report the knob unstampable
    # for a reason that has nothing to do with the knob.
    base = ["--out", "x", "--arm", "hier", "--image-hw", "256", "640",
            "--agents", "oracle", "--agent-rig-camera", "nominal",
            "--agent-join", "j.jsonl",
            "--bev-aux", "col", "--w-bev-aux", "0.1",
            "--wp-index", "on"]
    # Candidates, tried in order: a knob with a DOMAIN (a mount height must be
    # a plausible height) takes the first admissible one. A per-knob table of
    # values would be the rotting list this test exists to avoid.
    cand = {float: (0.137, 1.37), int: (61, 7)}
    checked = 0
    for a in p._actions:
        if a.dest not in t.agent_knob_dests(p) or not a.option_strings:
            continue
        opt = a.option_strings[0]
        vals: list = []
        if a.type in cand and a.nargs is None:
            vals = [a.type(v) for v in cand[a.type]]
        elif a.choices:
            # a choice that is NOT the default, so a stamp that ignores argv
            # and echoes the default cannot pass
            vals = [c for c in a.choices if c != a.default]
        elif isinstance(a.const, bool) or a.nargs == 0:
            vals = [True]
        if not vals:
            continue
        last = None
        for v in vals:
            argv = list(base) + ([opt] if v is True else [opt, str(v)])
            try:
                args = p.parse_args(argv)
                cfg, _ = _cfg_at(args=args)
                stamp = t._seam_stamp(cfg, args)
            except SystemExit as exc:          # inadmissible value, try next
                last = exc
                continue
            assert v in _leaf_values(stamp), (
                f"{opt} = {v!r} does not reach config.json by value; a run "
                f"cannot state the weight it trained at (M18).")
            t.assert_knobs_stamped(args, stamp)
            checked += 1
            break
        else:
            raise AssertionError(f"no admissible probe value for {opt}: "
                                 f"{last}")
    # the same-breath control: a loop that checked nothing passes vacuously.
    assert checked >= 12, checked


def test_P2_assert_knobs_stamped_FAILS_a_stamp_that_lost_a_knob():
    """⛔ DELIBERATE REGRESSION: a gate never shown to fail the defect
    certifies nothing. Drop one knob from the stamp; the run must refuse."""
    a = _args("--agents", "oracle", "--w-agent", "0.25")
    cfg, a = _cfg_at(args=a)
    stamp = t._seam_stamp(cfg, a)
    t.assert_knobs_stamped(a, stamp)                          # green first
    broken = dict(stamp)
    broken["agent_knobs"] = {k: v for k, v in stamp["agent_knobs"].items()
                             if k != "w_agent"}
    with pytest.raises(SystemExit, match="do NOT reach config.json"):
        t.assert_knobs_stamped(a, broken)
    with pytest.raises(SystemExit, match="no `agent_knobs` block"):
        t.assert_knobs_stamped(a, {"hier": True})


def test_P2_config_json_writes_the_CHECKED_stamp():
    """The link between what this file asserts (`_seam_stamp`) and what a run
    writes (`config.json["seams"]`), asserted on the PARSED trainer: the value
    written under "seams" is the same object `assert_knobs_stamped` was called
    on, and the check happens BEFORE the write."""
    src = TRAINER_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    lines_assign = [n.lineno for n in ast.walk(fn)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(x, ast.Name) and x.id == "_seams"
                            for x in n.targets)
                    and isinstance(n.value, ast.Call)
                    and getattr(n.value.func, "id", "") == "_seam_stamp"]
    lines_check = [n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "assert_knobs_stamped"]
    written = [k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
               for k, v in zip(n.keys, n.values)
               if isinstance(k, ast.Constant) and k.value == "seams"
               and isinstance(v, ast.Name) and v.id == "_seams"]
    assert lines_assign and lines_check and written == ["seams"]
    assert min(lines_check) > min(lines_assign)


# =========================================================================
# P3 — M17: 100 queries, and the help text no longer asserts the val40 claim
# =========================================================================

def test_P3_agent_queries_default_is_100():
    assert t.AGENT_QUERIES_DEFAULT == 100
    assert t.build_parser().get_default("agent_queries") == 100


def test_P3_the_default_reaches_the_seam_config():
    a = _args("--agents", "oracle")
    cfg = v3.refc_v3_smoke_config(True)
    t._pin_refcv5_seams(cfg, a)
    assert cfg.core.agents.queries == 100


def test_P3_the_help_text_no_longer_asserts_the_val40_claim():
    """⛔ THE NUMBER WAS WRONG IN THE PLACE A READER WOULD CHECK IT. The old
    help said '32 drops ZERO targets' — true on val40 (max 24), false on train
    (max 94, 41,362 boxes dropped, nearest sacrifice 13.1 m)."""
    act = next(a for a in t.build_parser()._actions
               if a.dest == "agent_queries")
    h = act.help
    assert "32 drops ZERO targets" not in h
    assert "13.1 m" in h and "94" in h and "match_slots" in h
    # same-breath control: the help really was read
    assert len(h) > 200


# =========================================================================
# P4 — M18: a hash carries its artifact scope, or the consumer refuses
# =========================================================================

def _fixture_join(tmp_path, body=b'{"clip":"a"}\n{"clip":"b"}\n'):
    raw = tmp_path / "toy_agents.jsonl"
    raw.write_bytes(body)
    xz = tmp_path / "toy_agents.jsonl.xz"
    xz.write_bytes(lzma.compress(body))
    return raw, xz


def test_P4_a_sidecar_without_a_declared_scope_REFUSES(tmp_path):
    """⛔ THE RULE. MEASURED: the train sidecar's md5 covers the .xz, val40's
    covers the .jsonl, and NEITHER says so — so a checker inherited from one
    refuses the other's perfectly good file."""
    _, xz = _fixture_join(tmp_path)
    meta = {"summary": {"md5": jm.file_digest(xz, scope="compressed"),
                        "out": str(xz)}}
    with pytest.raises(jm.JoinDigestScopeMissing, match="digest_scope"):
        jm.verify(xz, meta, where="toy")


def test_P4_the_extension_heuristic_is_MEASURED_wrong_so_it_is_not_offered():
    """The repair everyone reaches for is to read the extension off
    `summary.out`. On the REAL val40 .xz sidecar that names the .jsonl, so the
    heuristic is wrong in the other direction. This test pins that the module
    exposes no such inference."""
    assert not any("infer" in n or "guess" in n for n in dir(jm))


@pytest.mark.parametrize("scope", ["compressed", "decompressed"])
def test_P4_a_declared_scope_verifies_in_BOTH_shapes(tmp_path, scope):
    raw, xz = _fixture_join(tmp_path)
    d = jm.file_digest(xz, scope=scope)
    fname = xz.name if scope == "compressed" else raw.name
    meta = jm.attach({"summary": {"md5": d}}, d, scope=scope, filename=fname)
    ev = jm.verify(xz, meta, where="toy")
    assert ev["verified"] and ev["scope"] == scope and ev["digest"] == d


def test_P4_the_WRONG_scope_declaration_is_caught(tmp_path):
    """⛔ THE DELIBERATE REGRESSION: the exact live failure, reproduced. A
    digest over the compressed bytes, declared as covering the decompressed
    ones, must be REFUSED — not quietly accepted."""
    raw, xz = _fixture_join(tmp_path)
    d_xz = jm.file_digest(xz, scope="compressed")
    meta = jm.attach({}, d_xz, scope="decompressed", filename=raw.name)
    with pytest.raises(jm.JoinDigestMismatch, match="different bytes"):
        jm.verify(xz, meta, where="toy")


def test_P4_a_digest_about_ANOTHER_FILE_is_caught(tmp_path):
    _, xz = _fixture_join(tmp_path)
    meta = jm.attach({}, jm.file_digest(xz, scope="compressed"),
                     scope="compressed", filename="val40_agents.jsonl.xz")
    with pytest.raises(jm.JoinDigestMismatch, match="DIFFERENT file"):
        jm.verify(xz, meta, where="toy")


@pytest.mark.parametrize("scope", ["compressed", "decompressed"])
def test_P4_backfill_MEASURES_the_scope_of_a_legacy_digest(tmp_path, scope):
    """The migration for every sidecar that predates the rule: hash BOTH
    artifacts and report which one the recorded digest matched."""
    raw, xz = _fixture_join(tmp_path)
    rec = jm.file_digest(xz, scope=scope)
    meta, ev = jm.backfill({"summary": {"md5": rec}}, xz)
    assert ev["matched"] == [scope]
    assert set(ev["candidates"]) == set(jm.ARTIFACT_SCOPES)
    assert meta[jm.BLOCK]["scope"] == scope
    assert meta[jm.BLOCK]["filename"] == (xz.name if scope == "compressed"
                                          else raw.name)
    assert jm.verify(xz, meta, where="toy")["verified"]


def test_P4_backfill_REFUSES_when_neither_artifact_matches(tmp_path):
    _, xz = _fixture_join(tmp_path)
    with pytest.raises(jm.JoinDigestMismatch, match="NEITHER"):
        jm.backfill({"summary": {"md5": "0" * 32}}, xz)


def test_P4_backfill_REFUSES_an_ambiguous_uncompressed_file(tmp_path):
    """⚠️ An uncompressed .jsonl hashes the same under both scopes, so a
    declaration derived from it would later be quoted about an .xz it does not
    cover. Ambiguity is refused, not resolved by preference."""
    raw, _ = _fixture_join(tmp_path)
    rec = jm.file_digest(raw, scope="compressed")
    with pytest.raises(jm.JoinDigestError, match="both scopes hash"):
        jm.backfill({"summary": {"md5": rec}}, raw)


def test_P4_sidecar_path_probes_more_than_one_name(tmp_path):
    """⚠️ 'Absence found at ONE location is not absence'. MEASURED: the train
    sidecar is `<file>.xz.meta.json`, not the `<stem>.meta.json` the package
    doc named — a fetch of the documented name 404s."""
    _, xz = _fixture_join(tmp_path)
    assert jm.sidecar_path(xz) is None
    side = tmp_path / (xz.name + ".meta.json")
    side.write_text("{}", encoding="utf-8")
    assert jm.sidecar_path(xz) == side
    side.unlink()
    alt = tmp_path / "toy_agents.jsonl.meta.json"
    alt.write_text("{}", encoding="utf-8")
    assert jm.sidecar_path(xz) == alt


def test_P4_the_trainer_REFUSES_an_undeclared_sidecar(tmp_path):
    """The consumer side, through the trainer's own helper."""
    _, xz = _fixture_join(tmp_path)
    side = tmp_path / (xz.name + ".meta.json")
    side.write_text(json.dumps(
        {"summary": {"md5": jm.file_digest(xz, scope="compressed")}}),
        encoding="utf-8")
    a = _args("--agent-join", str(xz))
    with pytest.raises(SystemExit, match="MIGRATION"):
        t._verify_agent_join(a)
    # ...and the declared form passes, through the same helper
    meta, _ = jm.backfill(json.loads(side.read_text(encoding="utf-8")), xz)
    side.write_text(json.dumps(meta), encoding="utf-8")
    ev = t._verify_agent_join(a)
    assert ev["verified"] and ev["scope"] == "compressed"


def test_P4_the_trainer_records_an_UNVERIFIED_join_rather_than_passing(
        tmp_path):
    """A join with no sidecar warns and is STAMPED unverified — an absent
    sidecar is a missing check, not a lying one, and the record must say
    which of the two it was."""
    _, xz = _fixture_join(tmp_path)
    ev = t._verify_agent_join(_args("--agent-join", str(xz)))
    assert ev["verified"] is False and "no sidecar" in ev["reason"]
    off = t._verify_agent_join(_args("--agent-join", str(xz),
                                     "--agent-join-verify", "off"))
    assert off["verified"] is False and off["mode"] == "off"


def test_P4_the_two_REAL_sidecars_declare_nothing_today():
    """The incident, pinned against the actual banked artifacts. When the
    DataFlyWheel backfills them this test SHOULD start failing — that is the
    signal the migration landed, and the assertion is then inverted."""
    root = Path(__file__).resolve().parents[2] / "TanitAD Research Lab"
    cands = [
        root / "Data Engineering/Implementation/incoming"
             / "2026-08-17-train-obstacle-join/raw/train2400_agents.meta.json",
        root / "Benchmarks & Evals/Implementation/incoming"
             / "2026-08-18-val40-lead-join/raw/val40_agents.jsonl.xz.meta.json",
    ]
    seen = 0
    for p in cands:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except OSError:
            continue                       # a flapping mount is not evidence
        seen += 1
        s = jm.summarise(meta)
        assert s["declared"] is False
        assert s["recorded_digest"]        # it DOES record a digest…
    if seen == 0:
        pytest.skip("neither banked sidecar was readable (mount)")
