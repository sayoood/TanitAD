"""K2DC FINAL GATE (MM 4a3d05, 2026-11-13) -- two GATE-side defects refused a K2DC argv whose real smoke built AND fired the map-dose
share controller (refcv9 E2 chain, commit df49fd59, ``r9gate3_thor_gate.sh final K2DC``, 22:48:56Z, token verdict FAIL):

  G-LIVE  "102 training loss calls observed, expected 100 + 1 fp32 gradient-share replay(s) at [50]" -- the controller's OWN fp32 replay
          (``map_dose_ctl.probe_before_forward``, stamped ``dcp_fp32_replay: 1.0`` on the smoke's row 100) was not counted.
  G-DVB   "--map-hires-share-target: declared 0.1 but BUILT None (read from model._map_dose_ctl)" -- ``train()`` attaches the controller
          after its data sources, past ``trainer_capture``'s stop, so the captured model never carried it.

The fixes (``launch_gate.judge_live`` counts ``dcp_fp32_replay`` rows; ``launch_gate.dose_ctl_for_dvb`` attaches the controller with the
TRAINER'S OWN builder for the G-DVB judgement only) are pinned here, each with the measured refusal reproduced as a RED arm and the
must-still-fail cases (an unstamped extra call; a builder that alters or drops the controller). No GPU, no data, no torch model.
"""
from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
from tanitad.train import map_dose_ctl as MDC  # noqa: E402

MEASURED_LIVE = "102 training loss calls observed, expected 100 + 1 fp32 gradient-share replay(s) at [50]"
GS50 = {"step": 50, "gs_fp32_replay": 1.0}
DCP100 = {"step": 100, "dcp_fp32_replay": 1.0, "dc_w_eff": 25.0, "dc_share_read": 0.01}


# ================================================================================================ G-LIVE ==
def _count(n_calls: int, rows: list[dict], expect: int = 100):
    steps = [{"keys": ["loss"], "nonfinite": [], "vals": {"loss": 1.0 + 0.01 * i}} for i in range(n_calls)]
    reasons, det = LG.judge_live([], [], steps, expect_steps=expect, summary={"done": True, "step": expect},
                                 train_rows=[dict(r) for r in rows] + [{"step": expect}], grads={}, prior=None, exit_msg=None)
    return [r for r in reasons if "training loss calls observed" in r], det


def test_the_dose_controller_replay_is_counted_like_the_gradient_share_replay():
    bad, det = _count(102, [GS50, DCP100])
    assert bad == [], bad
    assert det["fp32_replay_steps"] == [50] and det["dc_fp32_replay_steps"] == [100]


def test_RED_without_the_dcp_stamp_the_measured_smoke_is_the_measured_refusal():
    """The 22:48:56Z reading, reproduced: the same 102 calls with only the gs stamp counted."""
    bad, _ = _count(102, [GS50])
    assert bad == [MEASURED_LIVE]


def test_an_extra_call_no_row_stamps_still_fails():
    bad, _ = _count(103, [GS50, DCP100])
    assert bad == ["103 training loss calls observed, expected 100 + 1 fp32 gradient-share replay(s) at [50]"
                   " + 1 dose-controller fp32 replay(s) at [100]"]


def test_a_missing_call_still_fails():
    bad, _ = _count(101, [GS50, DCP100])
    assert len(bad) == 1 and bad[0].startswith("101 training loss calls observed")


def test_without_a_controller_the_rule_and_its_message_are_the_tips():
    assert _count(101, [GS50])[0] == []
    bad, det = _count(100, [GS50])
    assert bad == ["100 training loss calls observed, expected 100 + 1 fp32 gradient-share replay(s) at [50]"]
    assert "dc_fp32_replay_steps" not in det


# ================================================================================================= G-DVB ==
def _args(target=0.1):
    return argparse.Namespace(map_hires_share_target=target, map_hires_share_every=250, map_hires_share_group="trunk",
                              map_hires_w_bounds="1,25", w_map_hires=25.0)


def _trainer(mdc=MDC):
    return types.SimpleNamespace(_mdc=mdc)


def test_RED_the_captured_model_without_the_controller_is_the_measured_refusal():
    m = MDC.dvb_check(types.SimpleNamespace(), _args())
    assert len(m) == 1 and m[0].lever == "--map-hires-share-target" and m[0].declared == 0.1 and m[0].built is None
    assert m[0].read_from == "model._map_dose_ctl"


def test_G_DVB_judges_the_trainers_own_builder_and_the_attachment_is_removed_after():
    model, args = types.SimpleNamespace(), _args()
    with LG.dose_ctl_for_dvb(_trainer(), model, args) as det:
        assert det["attached"] is True and det["target"] == 0.1 and det["every"] == 250 and det["bounds"] == [1.0, 25.0]
        assert MDC.dvb_check(model, args) == []
    assert not hasattr(model, "_map_dose_ctl"), "the gate-made attachment must not outlive the G-DVB judgement"


def test_a_builder_that_alters_a_flag_still_fails_G_DVB():
    fake = types.SimpleNamespace(controller_for=lambda a: MDC.MapDoseController(0.2, 250, "trunk", 1.0, 25.0, 25.0))
    model, args = types.SimpleNamespace(), _args()
    with LG.dose_ctl_for_dvb(_trainer(fake), model, args):
        m = MDC.dvb_check(model, args)
    assert [x.lever for x in m] == ["--map-hires-share-target"] and m[0].built == 0.2


def test_a_builder_that_builds_nothing_still_fails_G_DVB():
    fake = types.SimpleNamespace(controller_for=lambda a: None)
    model, args = types.SimpleNamespace(), _args()
    with LG.dose_ctl_for_dvb(_trainer(fake), model, args) as det:
        m = MDC.dvb_check(model, args)
    assert det["attached"] is False and len(m) == 1 and m[0].built is None


def test_OFF_attaches_nothing_and_G_DVB_stays_clean():
    model, args = types.SimpleNamespace(), _args(target=None)
    with LG.dose_ctl_for_dvb(_trainer(), model, args) as det:
        assert det["attached"] is False and not hasattr(model, "_map_dose_ctl")
        assert MDC.dvb_check(model, args) == []


def test_a_controller_already_on_the_model_is_left_alone():
    own = MDC.MapDoseController(0.1, 250, "trunk", 1.0, 25.0, 25.0)
    model = types.SimpleNamespace(_map_dose_ctl=own)
    with LG.dose_ctl_for_dvb(_trainer(), model, _args()) as det:
        assert det["attached"] is False and model._map_dose_ctl is own
    assert model._map_dose_ctl is own


def test_no_model_is_inert():
    with LG.dose_ctl_for_dvb(_trainer(), None, _args()) as det:
        assert det["attached"] is False


# ========================================================================== reachable from the callers ==
def test_job_model_wraps_judge_dvb_in_the_attachment():
    """REACHABILITY (the 'built, tested, unreachable from its caller' class): the gate's G-DVB call is inside the context manager."""
    src = (ROOT / "scripts" / "launch_gate.py").read_text(encoding="utf-8")
    m = re.search(r"with dose_ctl_for_dvb\(T, model, args\) as _dc_att:[^\n]*\n\s+reasons, det = judge_dvb\(D, model, args, parser, argv",
                  src)
    assert m, "job_model's G-DVB call is no longer wrapped by dose_ctl_for_dvb"
    head = src[:m.start()]
    assert head[head.rfind("\ndef "):].startswith("\ndef job_model("), "the wrapped call must live in job_model"


def test_the_trainer_exposes_the_builder_the_gate_calls():
    """The gate calls `T._mdc.controller_for`; the trainer must import the module under that name and attach what it builds."""
    tsrc = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert "from tanitad.train import map_dose_ctl as _mdc" in tsrc
    assert "_dose = _mdc.controller_for(args)" in tsrc and "model._map_dose_ctl = _dose" in tsrc


def test_the_live_rule_reads_the_stamp_the_controller_writes():
    """The `dcp_` prefix the G-LIVE count reads is the controller's own PROBE_PREFIX (one name, two files)."""
    assert MDC.PROBE_PREFIX == "dcp_"
    src = (ROOT / "scripts" / "launch_gate.py").read_text(encoding="utf-8")
    assert 'r.get("dcp_fp32_replay") == 1.0' in src
