"""E15 / GP-2 — the four ``--goal-point-*`` flags on ``refc_v3_train.py``.

⛔ WHAT THIS FILE IS FOR. A flag that PARSES, STAMPS, and DOES NOTHING is the
programme's most expensive failure mode: `--agents head --w-agent 0` builds a
detector supervised by nothing and reports "detection does not help"; `sampler
= "ddim"` reached `config.json` on a model with no denoiser at all. So every
test here asks one of exactly two questions —

* does the flag REACH the config (and the MODEL, and the RECORD)? and
* does a mis-specified combination REFUSE, rather than train and mean nothing?

⭐ And it pins the one thing a reader would otherwise have to know a trainer
line for: ``--goal-point-inject`` turns E13's categorical nav OFF. That is the
pre-registration (PREREG §2 — the ONE variable is the TYPE of the route
conditioning signal), not a side effect, and it is stamped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                      # noqa: E402
from tanitad.refs import goal_point as gp                      # noqa: E402
from tanitad.refs import refc_v3 as v3                         # noqa: E402

FLAGS = ("--goal-point-inject", "--goal-point-geo-prior",
         "--goal-point-t", "--goal-point-w")


def _args(*extra):
    return T.build_parser().parse_args(["--out", "x", "--arm", "hier", *extra])


def _pin(*extra):
    a = _args(*extra)
    return T._pin_trainer_cfg(v3.refc_v3_smoke_config(hier=True), a), a


# ---------------------------------------------------------------------------
# 1. the flags exist, and default to the INERT value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", FLAGS)
def test_the_flag_EXISTS(flag):
    opts = {o for a in T.build_parser()._actions for o in a.option_strings}
    assert flag in opts


def test_the_defaults_leave_the_live_build_UNMOVED():
    """A live 40 k run resumes through this file. Adding the seam must not
    change a run that does not ask for it."""
    cfg, a = _pin()
    assert a.goal_point_inject is False
    assert a.goal_point_geo_prior is False
    assert a.goal_point_w == T.GOAL_POINT_WEIGHT_DEFAULT == 0.0
    assert cfg.goal_point_inject is False
    assert cfg.core.graft_gp_point is False
    assert cfg.nav_inject is True                       # E13 untouched
    T._check_goal_point_args(a)                          # no-op, must not raise


# ---------------------------------------------------------------------------
# 2. the flags REACH the config, the model and the record
# ---------------------------------------------------------------------------

def test_inject_builds_the_head_and_REPLACES_the_categorical_nav():
    cfg, _ = _pin("--goal-point-inject", "--goal-point-w", "1.0")
    assert cfg.goal_point_inject is True
    assert cfg.nav_inject is False, (
        "PREREG §2: the goal point REPLACES E13's categorical nav. Running "
        "both makes the arm a two-variable experiment.")
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg)
    assert m.gp_head is not None and m.gp_cond is not None
    assert m.core.decoder.gp_point_gate is None          # geo prior NOT asked


def test_geo_prior_builds_the_RANKING_gate():
    cfg, _ = _pin("--goal-point-inject", "--goal-point-geo-prior",
                  "--goal-point-w", "1.0")
    assert cfg.core.graft_gp_point is True
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg)
    assert m.core.decoder.gp_point_gate is not None
    assert float(m.core.decoder.gp_point_gate.detach()) == 0.0   # zero-init


def test_goal_point_t_reaches_the_SLOT_and_is_not_a_hardcode():
    for t_s, slot in ((3.0, 4), (4.0, 5), (5.0, 6), (6.0, 7)):
        cfg, _ = _pin("--goal-point-inject", "--goal-point-w", "1.0",
                      "--goal-point-t", str(t_s))
        assert cfg.goal_point_cfg.t_goal_s == t_s
        torch.manual_seed(0)
        m = v3.RefCV3Model(cfg)
        assert m.gp_slot == slot == cfg.core.gp_slot, (t_s, m.gp_slot, slot)


def test_goal_point_w_reaches_the_MODEL_carrier():
    """The weight travels ON the model because `compute_losses_v3` has no
    `args` — the `_w_u0` / `_w_agent` discipline. A weight that is stamped but
    never carried is the `--agent-w-project` silent no-op, verbatim."""
    a = _args("--goal-point-inject", "--goal-point-geo-prior",
              "--goal-point-w", "0.75")
    assert a.goal_point_w == 0.75
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "model._w_goal_point" in src
    assert '_w_goal_point' in src and 'getattr(model, "_w_goal_point"' in src


# ---------------------------------------------------------------------------
# 3. the REFUSALS — each names the arm it would otherwise manufacture
# ---------------------------------------------------------------------------

def test_geo_prior_WITHOUT_inject_is_REFUSED():
    with pytest.raises(SystemExit, match="needs --goal-point-inject"):
        T._check_goal_point_args(_args("--goal-point-geo-prior"))


def test_inject_with_ZERO_weight_is_REFUSED():
    """It would build a head, stamp the edge, and supervise it with nothing —
    and the PREREG §5 head gate would then fail for a reason that is not about
    the goal FORM."""
    with pytest.raises(SystemExit, match="SUPERVISES IT WITH NOTHING"):
        T._check_goal_point_args(_args("--goal-point-inject"))


def test_a_weight_WITHOUT_inject_is_REFUSED():
    with pytest.raises(SystemExit, match="w_agent. defect"):
        T._check_goal_point_args(_args("--goal-point-w", "1.0"))


@pytest.mark.parametrize("t_s", ["2.0", "1.5", "0.5"])
def test_a_goal_INSIDE_the_scored_horizon_is_REFUSED_by_the_FLAG_too(t_s):
    """⛔ The refusal itself lives in `GoalPointConfig.__post_init__`. This one
    exists so the message names the flag instead of surfacing as a dataclass
    traceback 400 lines later."""
    with pytest.raises(SystemExit, match="at or inside the scored horizon"):
        T._check_goal_point_args(_args("--goal-point-inject",
                                       "--goal-point-w", "1.0",
                                       "--goal-point-t", t_s))


def test_a_goal_time_that_is_not_a_bank_slot_is_REFUSED_by_the_FLAG():
    with pytest.raises(SystemExit, match="not a model horizon"):
        T._check_goal_point_args(_args("--goal-point-inject",
                                       "--goal-point-w", "1.0",
                                       "--goal-point-t", "3.7"))


# ---------------------------------------------------------------------------
# 4. the RECORD — provenance, and the model/record agreement
# ---------------------------------------------------------------------------

def test_the_stamp_carries_the_ADMISSIBILITY_DECLARATION():
    """The PI's ruling asks "could this goal have been computed from the
    situation classifier's output?". A reader must be able to answer that from
    the ARTIFACT, months later, without the source tree."""
    cfg, a = _pin("--goal-point-inject", "--goal-point-geo-prior",
                  "--goal-point-w", "1.0")
    st = T._seam_stamp(cfg, a)["goal_point"]
    assert st["goal_point_inject"] is True
    assert st["graft_gp_point"] is True
    assert st["t_goal_s"] == 4.0 and st["t_pred_s"] == 2.0
    assert st["gp_slot"] == 5 and st["gp_scale_m"] == 40.0
    assert st["w_goal_point"] == 1.0
    assert st["replaces_nav_inject"] is True and st["nav_inject"] is False
    pr = st["provenance"]
    assert pr["contains_situation_classifier_output"] is False
    assert pr["situation_classifier_in_graph"] is False
    assert pr["reads_tactical_state_or_logits"] is False
    assert pr["supplied_or_predicted"] == "predicted"
    assert pr["inference_inputs"] == ["the strategic context token (vision)"]
    assert "REFUSES" in pr["leak_guard_time_mode"]
    assert pr == gp.goal_point_provenance(cfg.goal_point_cfg)


def test_the_stamp_is_ABSENT_when_the_edge_is_off():
    cfg, a = _pin()
    assert T._seam_stamp(cfg, a)["goal_point"] is None


def test_the_RECORD_must_match_the_MODEL_both_directions():
    """⛔ `assert_seams_are_built`: a stamped seam the weights do not have is
    FALSE PROVENANCE; a built seam absent from the record is `SEAM_STATE.md`."""
    cfg, a = _pin("--goal-point-inject", "--goal-point-geo-prior",
                  "--goal-point-w", "1.0")
    torch.manual_seed(0)
    m = v3.RefCV3Model(cfg)
    stamp = T._seam_stamp(cfg, a)
    T.assert_seams_are_built(m, stamp)                   # the honest pair

    # (a) the record CLAIMS a head the weights do not have
    lying = dict(stamp)
    torch.manual_seed(0)
    bare = v3.RefCV3Model(v3.refc_v3_smoke_config(hier=True))
    with pytest.raises(SystemExit, match="model.gp_head is None"):
        T.assert_seams_are_built(bare, lying)

    # (b) the model HAS the seam and the record does not say so
    silent = dict(stamp)
    silent["goal_point"] = None
    with pytest.raises(SystemExit, match="absent from the run record"):
        T.assert_seams_are_built(m, silent)

    # (c) the record claims a RANKING seam that ranks nothing
    cfg2, a2 = _pin("--goal-point-inject", "--goal-point-w", "1.0")
    torch.manual_seed(0)
    m2 = v3.RefCV3Model(cfg2)
    st2 = T._seam_stamp(cfg2, a2)
    T.assert_seams_are_built(m2, st2)
    st2["goal_point"] = dict(st2["goal_point"], graft_gp_point=True)
    with pytest.raises(SystemExit, match="gp_point_gate is\\s+None"):
        T.assert_seams_are_built(m2, st2)


def test_the_LABEL_INDEX_is_stated_in_the_record():
    """The goal label is `traj_tgt[:, gp_slot]`, i.e. the trainer's OWN target,
    not a re-derivation beside it. The record says so, because a guard or a
    label re-implemented next to its consumer drifts."""
    cfg, a = _pin("--goal-point-inject", "--goal-point-w", "1.0")
    note = T._seam_stamp(cfg, a)["goal_point"]["label_index_note"]
    assert "waypoint_targets" in note and "TRAIN ONLY" in note
    assert "round(t/dt) - 1" in note


# ---------------------------------------------------------------------------
# 5. the LOSS is wired to the trainer's own target, at the right slot
# ---------------------------------------------------------------------------

def test_the_loss_reads_the_trainer_target_at_the_GOAL_slot_not_the_SCORED_one():
    """⛔ The scored horizon is slot 3 (2.0 s); the goal is slot 5 (4.0 s). If
    the loss ever read the scored slot the label would BE the answer."""
    src = Path(T.__file__).read_text(encoding="utf-8")
    i = src.find("E15 (GP-2) — THE METRIC GOAL POINT'S SUPERVISION")
    assert i > 0, "the goal-point loss block is missing from the trainer"
    blk = src[i:i + 6000]
    assert "gslot = int(model.gp_slot)" in blk
    assert "traj_tgt[:, gslot, 0]" in blk and "traj_tgt[:, gslot, 1]" in blk
    assert "slot_valid[:, gslot]" in blk
    assert "gpm.goal_point_loss" in blk
    # the head gate readouts, in METRES, every eval
    assert "gp_lat_rmse_m" in blk and "gp_range_rmse_m" in blk
    assert "gp_label_rows" in blk


def test_the_head_gate_readouts_are_in_METRES_and_carry_their_n():
    """PREREG §5 bars: <= 1.0 m lateral, <= 2.0 m range. A readout in
    normalised units would silently be 40x optimistic."""
    pred = torch.tensor([[0.50, 0.10], [0.50, 0.10]])
    tgt = torch.tensor([[0.50, 0.05], [0.50, 0.05]])
    valid = torch.ones(2)
    lat = gp.lateral_rmse_m(pred, tgt, valid, 40.0)
    rng = gp.range_rmse_m(pred, tgt, valid, 40.0)
    assert float(lat) == pytest.approx(0.05 * 40.0)       # 2.0 m
    assert float(rng) == pytest.approx(0.0)
