"""Tests for the in-training val-side eval (PI Q3).

⭐ The load-bearing test is `test_an_action_ECHO_scores_perfect_ADE_and_is_caught`:
it builds a model that IGNORES its action input, shows that ADE cannot see the
defect, and shows that action_sensitivity can. That is the whole reason this
module is not an ADE watcher.
"""
import glob
import json

import pytest
import torch

from tanitad.train.intrain_eval import (EvalSplitError, ValSplit,
                                        action_sensitivity, hold_action_baseline,
                                        o5_horizon_ratio, run_intrain_eval,
                                        should_run_val)


def _batch(b=4, s=6, v=10.0, curved=True):
    """⚠️ CURVED BY DEFAULT. A straight constant-velocity ground truth makes the
    hold-action control PERFECT (ADE 0.0), so `beats_hold_action` is vacuous and
    the comparison tests nothing. Real driving curves; the fixture must too."""
    gt = torch.zeros(b, s, 2)
    gt[..., 0] = torch.arange(1, s + 1) * v * 0.1
    if curved:
        gt[..., 1] = (torch.arange(1, s + 1) * 0.1) ** 2      # lateral drift
    return {"gt_wp": gt, "v0": torch.full((b,), v),
            "future_actions2": torch.zeros(b, s, 2)}


# ------------------------------------------------------- action sensitivity
def test_identical_predictions_read_zero_sensitivity():
    p = torch.randn(2, 5, 2)
    r = action_sensitivity(p, p.clone())
    assert r["delta_m"] == 0.0 and r["sensitivity"] == 0.0


def test_sensitivity_is_a_ratio_of_two_reported_absolutes():
    """The ratio alone is unreadable at small magnitudes, so both parts ship."""
    a = torch.tensor([[[3.0, 4.0]]])          # ||a|| = 5
    b = torch.tensor([[[0.0, 0.0]]])
    r = action_sensitivity(a, b)
    assert r["delta_m"] == 5.0 and r["scale_m"] == 5.0 and r["sensitivity"] == 1.0


def test_zero_scale_does_not_divide_by_zero():
    z = torch.zeros(1, 3, 2)
    assert action_sensitivity(z, z)["sensitivity"] == 0.0


# ------------------------------------------------- the defect ADE cannot see
def test_an_action_ECHO_scores_perfect_ADE_and_is_caught():
    """⛔ THE POINT OF THE WHOLE MODULE.

    A model that IGNORES its action input can be perfect on ADE. MEASURED
    precedent: v1.x reproduced an S-curve 97.9 % open-loop and 0.0 % with the
    action held. ADE says "excellent"; sensitivity says "it never listened".
    """
    b = _batch()

    def echo_forward(batch):          # ignores future_actions2 entirely
        return batch["gt_wp"].clone()

    def alt(batch):
        out = dict(batch)
        out["future_actions2"] = torch.randn_like(batch["future_actions2"]) * 5.0
        return out

    rep = run_intrain_eval(echo_forward, [b], alt_action_fn=alt, max_batches=1)
    assert rep["val_ade_m"] == 0.0, "ADE is PERFECT — and it is blind"
    assert rep["action_sensitivity"] == 0.0, (
        "sensitivity must expose that the output ignores the action")


def test_an_action_dependent_model_reads_nonzero_sensitivity():
    b = _batch()

    def listening_forward(batch):
        return batch["gt_wp"] + batch["future_actions2"].sum(-1, keepdim=True)

    def alt(batch):
        out = dict(batch)
        out["future_actions2"] = batch["future_actions2"] + 3.0
        return out

    rep = run_intrain_eval(listening_forward, [b], alt_action_fn=alt, max_batches=1)
    assert rep["action_sensitivity"] > 0.0


def test_missing_alt_fn_reports_ABSENT_rather_than_a_number():
    """⛔ The primary quantity must never be silently omitted."""
    rep = run_intrain_eval(lambda x: x["gt_wp"], [_batch()], max_batches=1)
    assert rep["action_sensitivity"] is None
    assert "ABSENT" in rep["action_sensitivity_status"]


# --------------------------------------------------- the hold-action control
def test_hold_action_baseline_is_constant_velocity_straight():
    base = hold_action_baseline(torch.tensor([10.0]), dt=0.1, n_steps=3)
    assert base.shape == (1, 3, 2)
    assert torch.allclose(base[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(base[..., 1] == 0.0), "no lateral motion when holding"


def test_the_control_is_reported_and_compared():
    """H-CTRL-1: the do-nothing control beat a released arm on every family, so
    an eval that cannot see it can watch a model lose to nothing."""
    rep = run_intrain_eval(lambda x: x["gt_wp"], [_batch()],
                           alt_action_fn=lambda b: b, max_batches=1)
    assert "hold_action_ade_m" in rep
    assert rep["hold_action_ade_m"] > 0.0, "a curved GT must beat the CV control"
    assert rep["beats_hold_action"] is True   # perfect pred beats CV on a curve


def test_a_PERFECT_baseline_is_reported_not_treated_as_absent():
    """⛔ REGRESSION. The first version gated on `if base_sum:`, so a control
    that scored exactly 0.0 m read as 'no baseline computed'. Absence and a real
    zero must never share a branch."""
    rep = run_intrain_eval(lambda x: x["gt_wp"], [_batch(curved=False)],
                           max_batches=1)
    assert rep["hold_action_ade_m"] == 0.0, "a perfect control must be REPORTED"
    assert rep["beats_hold_action"] is False, "0.0 < 0.0 is False — a tie is not a win"


# ---------------------------------------------------------- hygiene / TRAIN-C5
def test_eval_mode_is_set_and_restored():
    """⛔ TRAIN-C5: a readout that forgot .eval() overstated ADE by +175.7 %.
    And a monitor must not leave the model in eval for the next training step."""
    m = torch.nn.Linear(2, 2)
    m.train()
    seen = {}

    def fwd(batch):
        seen["training"] = m.training
        return batch["gt_wp"]

    rep = run_intrain_eval(fwd, [_batch()], model=m, max_batches=1)
    assert seen["training"] is False, "must run under eval()"
    assert m.training is True, "must RESTORE training mode on exit"
    assert rep["eval_mode"] is True


def test_empty_val_set_reports_ABSENT_not_zero():
    """An empty val set that averages to 0.0 is the failure family this module
    exists inside — absence is reported, never averaged."""
    rep = run_intrain_eval(lambda x: x["gt_wp"], [], max_batches=4)
    assert rep["status"] == "ABSENT" and rep["n_batches"] == 0
    assert "val_ade_m" not in rep


def test_max_batches_bounds_the_cost():
    calls = {"n": 0}

    def fwd(batch):
        calls["n"] += 1
        return batch["gt_wp"]

    run_intrain_eval(fwd, [_batch()] * 50, max_batches=3)
    assert calls["n"] == 3, "must stop at max_batches, not run the whole val set"


def test_every_report_carries_its_tier():
    """A trainer number quoted as a capability claim is how 'v1.6 is
    best-in-program' happened."""
    rep = run_intrain_eval(lambda x: x["gt_wp"], [_batch()], max_batches=1)
    assert "T0" in rep["_tier"] and "t1_eval" in rep["_tier"]


# ---------------------------------------------------------------- the val round
_V3 = glob.glob(r"C:/Users/Admin/**/eval_split_v3.json", recursive=True)
V3_SHA = "ea8670e041c14ccb"
needs_v3 = pytest.mark.skipif(not _V3, reason="v3 split not on this box")


def _fake_split(tmp_path, eval_ids, train_ids, name="s.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"version": "t", "eval_clip_ids": list(eval_ids),
                             "train_clip_ids": list(train_ids),
                             "interpretation": {"inadmissible_use": "per-competence"},
                             "based_on_labels_md5": "deadbeef"}), encoding="utf-8")
    return p


# ---- the leak guard: the reason this whole thing can be trusted -------------
def test_a_LEAKING_split_is_REFUSED_at_load(tmp_path):
    """⛔ A val curve computed 300 times per run and leaking every time is worse
    than no curve, because a smooth line reads as evidence."""
    p = _fake_split(tmp_path, ["a", "b"], ["b", "c"])
    with pytest.raises(EvalSplitError) as e:
        ValSplit(p)
    assert "BOTH eval and train" in str(e.value)


def test_scoring_a_TRAIN_episode_is_REFUSED(tmp_path):
    """⭐ THE GUARD THE VAL ROUND LIVES OR DIES ON. A leak would show up as
    unexpectedly GOOD numbers — the most believable possible failure, and one we
    would celebrate rather than investigate."""
    sp = ValSplit(_fake_split(tmp_path, ["a", "b"], ["c", "d"]))
    sp.assert_held_out(["a", "b"])                      # clean
    with pytest.raises(EvalSplitError) as e:
        sp.assert_held_out(["a", "c"])                  # 'c' is a TRAIN clip
    assert "TRAIN episodes" in str(e.value)


def test_the_split_is_LOADED_never_derived(tmp_path):
    """A trainer-derived split is not reproducible across arms — two seeds would
    score different episodes and the curves would not be comparable."""
    with pytest.raises(Exception):
        ValSplit(tmp_path / "does_not_exist.json")


def test_a_wrong_but_VALID_split_file_is_refused_by_sha(tmp_path):
    """⚠️ Three splits are live and v1/v2 are superseded-but-RESOLVABLE, so the
    wrong file loads perfectly well. Only the sha catches it."""
    p = _fake_split(tmp_path, ["a"], ["b"])
    with pytest.raises(EvalSplitError) as e:
        ValSplit(p, require_sha="ea8670e041c14ccb")
    assert "sha mismatch" in str(e.value)


# ---- the stamp -------------------------------------------------------------
def test_every_val_record_can_name_its_split_and_labels(tmp_path):
    st = ValSplit(_fake_split(tmp_path, ["a"], ["b"])).stamp()
    assert st["split_sha"] and st["based_on_labels_md5"] == "deadbeef"
    assert st["inadmissible_use"], "the split's own rule must travel with it"


# ---- one artifact, two cadences --------------------------------------------
def test_two_cadences_off_ONE_split():
    """⭐ Monitoring and selection read the SAME artifact, so the two curves are
    one curve sampled at different intervals — no independence problem, and one
    val budget rather than two."""
    assert should_run_val(100)["monitor"] and not should_run_val(100)["select"]
    assert should_run_val(1000)["monitor"] and should_run_val(1000)["select"]
    assert not should_run_val(0)["monitor"], "step 0 is not a val round"


# ---- the free signal --------------------------------------------------------
def test_o5_ratio_separates_the_banked_arms_by_SCALE():
    """MEASURED across three different recipes: 2k arms sit at 0.31-0.36 and 30k
    arms at ~1.0, so the separation is by scale, not recipe."""
    assert o5_horizon_ratio(0.0654, 0.2125)["regime"] == "shortcut-active"
    assert o5_horizon_ratio(0.0804, 0.2225)["regime"] == "shortcut-active"
    for a, b in ((0.2877, 0.2794), (0.2887, 0.2892), (0.3005, 0.2705)):
        assert o5_horizon_ratio(a, b)["regime"] == "closed"


def test_o5_ratio_reports_ABSENT_rather_than_1_0_when_undefined():
    """⛔ A missing denominator must not become a healthy-looking 1.0."""
    r = o5_horizon_ratio(0.5, 0.0)
    assert r["ratio"] is None and r["status"] == "ABSENT"


def test_the_o5_caveat_travels_with_the_number():
    assert "NOT that no copying occurs" in o5_horizon_ratio(0.3, 0.3)["_read"]


# ---- against the real artifact ---------------------------------------------
@needs_v3
def test_the_real_v3_split_loads_and_is_disjoint():
    sp = ValSplit(_V3[0], require_sha=V3_SHA)
    assert len(sp.eval_ids) == 141
    assert not (sp.eval_ids & sp.train_ids)
    assert sp.stamp()["split_sha"].startswith(V3_SHA)
