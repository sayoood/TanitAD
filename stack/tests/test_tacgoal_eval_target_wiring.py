"""D-TACGOAL-EVAL-1 — the goal-set target must reach the **EVAL** dataset too.

⛔⛔ WHY THIS FILE EXISTS, AND IT IS NOT A DIAGNOSTIC CONCERN. MEASURED
2026-09-11 on Thor, arm ``C_w0p05`` of the ``--w-tac-goal`` weight sweep:

    400 of 400 training steps completed, ``ckpt step 400 -> ckpt.pt`` written,
    then the step-400 held-out eval raised the REFUSE-DO-NOT-SKIP guard
    ``"--w-tac-goal > 0 but the batch carries no tac_goal_y/tac_goal_w"``
    and the process **exited 1** with its eval row never written.

The refusal is CORRECT — a silently-skipped term while ``config.json`` stamps
the weight is the ``w_agent`` defect verbatim. What was wrong is WHERE it fires:
``tac_goal_targets`` was set in exactly ONE place, on the TRAIN dataset, and the
eval dataset ``e_ds`` is a different object that never got it.

⛔ **THE CONSEQUENCE IS A DEAD 40 k RUN, NOT A MISSING NUMBER.** refcv6 arm C is
specified as refcv5-v2's argv plus ``--w-tac-goal``, and that argv carries
``--eval-every 500``. Arm C would have **died at step 500 of 40,284**, with the
compute paid for.

⭐ **THE PRECEDENT WAS ALREADY IN THE SAME FUNCTION.** The ``--bev-aux`` block
sets ``e_ds.bev_spec`` for exactly this reason and its comment spells the
mechanism out — ``SystemExit`` derives from ``BaseException``, so the eval
block's ``except Exception`` **cannot** catch it. The tac-goal term never got
the same line. This file makes the pair symmetric and keeps it that way.

⛔ Every expectation here is a literal.
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import refc_v3 as v3                       # noqa: E402

_TRAINER_PY = os.path.join(_STACK, "scripts", "refc_v3_train.py")

#: the eval-side wiring line, as a literal. A mutation that deletes it must
#: make this file go RED.
_EVAL_WIRE = "                e_ds.tac_goal_targets = True"
#: the weight gate, used on BOTH datasets. Two occurrences is the fix.
_WEIGHT_GATE = 'if float(getattr(args, "w_tac_goal", 0.0) or 0.0) > 0.0:'


def _src() -> str:
    return io.open(_TRAINER_PY, encoding="utf-8").read()


def _trainer():
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_evalwire", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_evalwire"] = mod
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# 1 — the Python fact that turns a diagnostic refusal into a dead run
# ==========================================================================
def test_SystemExit_is_NOT_an_Exception_so_the_eval_guard_cannot_catch_it():
    """⛔ This is the whole reason the arm died instead of logging an
    ``eval_error`` row. It is a literal about Python, derivable without reading
    the code under test, and it composes with the source fact below."""
    assert issubclass(SystemExit, Exception) is False       # literal
    assert issubclass(SystemExit, BaseException) is True    # literal

    s = _src()
    assert s.count("def train(args) -> dict:") == 1          # control
    # the eval block's handler, verbatim — it catches Exception, not BaseException
    assert s.count("except Exception as exc:          # noqa: BLE001 (by design)") == 1
    assert s.count("except BaseException") == 0              # literal


# ==========================================================================
# 2 — THE FIX: the eval dataset is wired, and gated on the same weight
# ==========================================================================
def test_the_EVAL_dataset_is_given_the_goal_set_target():
    """⛔ THE REGRESSION ARM ANCHOR. Deleting this line reproduces the arm that
    died on Thor, and this assertion is what goes RED."""
    s = _src()
    assert s.count("def train(args) -> dict:") == 1          # control
    assert s.count(_EVAL_WIRE) == 1
    assert s.count("e_ds.tac_goal_negatives = str(getattr(") == 1


def test_BOTH_datasets_are_wired_and_BOTH_are_gated_on_the_weight():
    """⭐ The asymmetry WAS the bug: one assignment, two datasets. Pinned as a
    pair of literals so a future refactor that drops either half is visible."""
    s = _src()
    assert s.count("def train(args) -> dict:") == 1          # control
    assert s.count("            ds.tac_goal_targets = True") == 1   # train
    assert s.count(_EVAL_WIRE) == 1                                  # eval
    assert s.count(_WEIGHT_GATE) == 2                                # literal


def test_the_eval_wiring_sits_INSIDE_the_eval_labels_branch():
    """The targets are built from ``e_ds.v7_by_sid``, which only exists under
    ``if args.eval_labels:``. Wiring the flag outside that branch would move the
    crash rather than remove it."""
    s = _src()
    i_labels = s.find("        if args.eval_labels:")
    i_wire = s.find(_EVAL_WIRE)
    i_perm = s.find("        g_ev = torch.Generator().manual_seed(12345)")
    assert i_labels > 0 and i_wire > 0 and i_perm > 0
    assert i_labels < i_wire < i_perm


def test_the_sibling_bev_wiring_is_still_there():
    """⭐ The precedent that should have caught this. Keeping both lines under
    one test means a refactor cannot quietly drop one and leave the other."""
    s = _src()
    assert s.count("                e_ds.bev_spec = _bev_aux.PolarBEVSpec(") == 1
    assert s.count(_EVAL_WIRE) == 1


# ==========================================================================
# 3 — the mechanism, behaviourally: no flag ⇒ no keys ⇒ the guard must fire
# ==========================================================================
_GEOM = frozenset({"FOLLOW_LANE", "SPEED_BAND", "STOP_POINT", "TURN_L",
                   "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"})


def _ds(tr, monkeypatch, *, enabled: bool):
    """A real ``V3Dataset`` with a v7.2 join — the same rig
    ``test_tac_goal_trainer_flag.py`` uses, so the two files agree on what a
    labelled dataset is."""
    from tanitad.data import v7_labels as v7l
    monkeypatch.setattr(v7l, "_MEASURED_GEOMETRY_TOKENS", _GEOM)
    cfg = v3.refc_v3_smoke_config(True)
    eps = tr._synth_episodes(2, cfg.core, seed=0)
    for i, e in enumerate(eps):
        e.episode_id = str(9000 + i)
    ds = tr.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                      channels=cfg.core.encoder.in_channels)
    lab = v7l.V7Label(
        clip_id="clip_9000", tac_lat="LANE_KEEP", tac_lon="CRUISE",
        str_action="HOLD_MAIN_ROAD", str_goal="HOLD_MAIN_ROAD",
        tac_anchor=None, bands={"tactical_s": [0.0, 60.0]}, t0_s=0.3,
        horizon={}, tac_goals=frozenset({"FOLLOW_LANE", "SPEED_BAND"}),
        tac_goal_meta={"FOLLOW_LANE": {"provenance": "geometry"},
                       "SPEED_BAND": {"provenance": "geometry"}})
    ds.v7_by_sid = {9000: lab}
    ds.v7_dt = 0.1
    ds.tac_goal_targets = enabled
    ds.tac_goal_negatives = "measured"
    first = {e_i: i for i, (e_i, _t) in reversed(list(enumerate(ds.index)))}
    return ds, first


def test_an_UNWIRED_dataset_emits_neither_key_which_is_what_kills_the_run(
        monkeypatch):
    """⛔ The precondition of the crash, read directly: this is the state the
    EVAL dataset was in for every weighted arm."""
    tr = _trainer()
    ds, first = _ds(tr, monkeypatch, enabled=False)
    item = ds[first[0]]
    assert "tac_goal_y" not in item
    assert "tac_goal_w" not in item


def test_a_WIRED_dataset_emits_both_keys_at_the_vocabulary_width(monkeypatch):
    """⭐ The discriminating control. Without it, "the key is absent" would be
    satisfied by a dataset that never emits anything."""
    tr = _trainer()
    ds, first = _ds(tr, monkeypatch, enabled=True)
    item = ds[first[0]]
    assert "tac_goal_y" in item
    assert "tac_goal_w" in item
    assert tuple(item["tac_goal_y"].shape) == (22,)          # literal
    assert tuple(item["tac_goal_w"].shape) == (22,)          # literal


def test_a_clip_with_NO_record_still_gets_both_keys_all_ignored(monkeypatch):
    """⛔ Episode 9001 carries no v7.2 record on purpose. It must still emit
    both keys, explicitly all-ignored — a batch that sometimes carries the key
    and sometimes does not would make the loss-time refusal fire at RANDOM
    instead of at launch, which is strictly worse than failing every time."""
    from tanitad.data import v7_labels as v7l
    tr = _trainer()
    ds, first = _ds(tr, monkeypatch, enabled=True)
    item = ds[first[1]]                                      # the unlabelled ep
    assert "tac_goal_y" in item and "tac_goal_w" in item
    assert float(item["tac_goal_y"].sum()) == 0.0            # literal
    assert float(item["tac_goal_w"].max()) == float(v7l.IGNORE_W)
