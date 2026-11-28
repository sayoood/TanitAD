"""Pins for refcv5 P4 / P13 / P14 -- the three unowned missing pieces.

⛔ EVERY GUARD HERE IS PROVEN REACHABLE BY TRIPPING IT, never by inspecting
source. An AST census once read 0 suspects on BOTH the fixed and the broken
trainer, so a test that only asserts a string is present in a file proves
nothing about whether the code path can run.

What each pin protects, and the measurement that earned it:

* **P13** -- `--sampler-train-t-max` was a DEAD FLAG. MEASURED 2026-09-06 with
  a live spy on `AnchoredDiffusionDecoder._sample`: in TRAINING mode the
  realised timesteps are the INFERENCE ladder `{0, 10}`, and they are
  **identical under `t_max = 50` and `t_max = 1`**. `draw_train_timesteps` is
  the consumer that flag never had.
* **P14** -- `refc_v3_train.py` had NO `--sel-refined` / `--sel-score-emitted`
  flag, so refcv5 could not arm emitted-fan ranking at all and shipped
  `sampler_ranks_the_fan: False`. MEASURED ceiling on the banked fan
  (n = 881 windows / 40 episodes, paired episode-cluster bootstrap):
  ADE 0.4728 -> 0.1914 m, +0.2813 [+0.2127, +0.3543] separated; a >2x-better
  trajectory sits in the fan on 41.09 % of windows (45.40 % on XL).
* **P4** -- the 15 strategic tokens are minted on 4,572/4,572 clips and were
  trained by nothing. The label side existed (`v7_labels.HEADS`); the MODEL
  side did not.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tanitad.data import v7_labels as v7l
from tanitad.refs import refc_sampler as rs
from tanitad.refs import refc_strategic as st
from tanitad.refs.refc import (AnchoredDiffusionDecoder, DecoderConfig,
                               SelectionConfig)

TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "refc_v3_train.py"


# ===================================================================== P13 ==
def test_p13_draw_is_uniform_and_exclusive():
    """`t ~ U[0, t_max)` -- EXCLUSIVE bound, one draw per sample."""
    g = torch.Generator().manual_seed(0)
    t = rs.draw_train_timesteps(20000, 50, torch.device("cpu"), generator=g)
    assert t.shape == (20000,) and t.dtype == torch.long
    assert int(t.min()) == 0 and int(t.max()) == 49, "bound must be EXCLUSIVE"
    assert abs(float(t.double().mean()) - 24.5) < 0.5


def test_p13_regression_arm_is_the_constant_zero_draw():
    """⛔ The pre-registered DELIBERATE REGRESSION: `t_max = 1` draws only
    t = 0 -- and must be ACCEPTED, because a gate that cannot run its own
    broken arm is blind."""
    t = rs.draw_train_timesteps(500, 1, torch.device("cpu"))
    assert int(t.max()) == 0 and float(t.double().var()) == 0.0
    with pytest.raises(ValueError):
        rs.draw_train_timesteps(4, 0, torch.device("cpu"))


def test_p13_sigma_controls_read_their_known_values():
    """Controls that MUST read a known value, or the schedule is not DD's."""
    sch = rs.DDIMSchedule()
    # ⛔ t = 0 is NOT zero noise: steps_offset = 1 means abar_0 < 1.
    assert float(sch.sqrt_one_minus_abar(0)) > 0.0
    # the published truncation sigma
    assert abs(float(sch.sqrt_one_minus_abar(8)) - 0.0316) < 5e-5


def test_p13_train_noise_refuses_a_broadcastable_timestep():
    """A `[B,1,1,1]` t would broadcast SILENTLY and noise every candidate at a
    different level -- the classic bug in this family."""
    sch = rs.DDIMSchedule()
    x0 = torch.randn(4, 3, 8, 2)
    rs.train_noise(sch, x0, torch.zeros(4, dtype=torch.long))       # ok
    with pytest.raises(ValueError):
        rs.train_noise(sch, x0, torch.zeros(4, 1, 1, 1, dtype=torch.long))


def test_p13_noising_actually_moves_the_state_and_scales_with_t():
    sch = rs.DDIMSchedule()
    x0 = torch.randn(64, 4, 8, 2)
    g = torch.Generator().manual_seed(1)
    lo, _ = rs.train_noise(sch, x0, torch.zeros(64, dtype=torch.long), g)
    g = torch.Generator().manual_seed(1)
    hi, _ = rs.train_noise(sch, x0, torch.full((64,), 49, dtype=torch.long), g)
    d_lo = float((lo - x0).abs().mean())
    d_hi = float((hi - x0).abs().mean())
    assert d_lo > 0.0, "t = 0 must still noise (abar_0 < 1)"
    assert d_hi > d_lo * 2.0, "more noise at t = 49 than at t = 0"


# ===================================================================== P14 ==
def _decoder(refined: bool, emitted: bool, emitted_t: int = 0):
    cfg = DecoderConfig()
    cfg.sampler, cfg.sampler_space = "ddim", "control"
    cfg.d, cfg.layers, cfg.n_heads, cfg.aux_hidden = 32, 1, 2, 32
    torch.manual_seed(0)
    dec = AnchoredDiffusionDecoder(
        feat_dim=16, n_steps=4, d_meas=4, d_ctx=8, tac_latent_dim=8,
        anchors=torch.randn(8, 4, 2) * 3.0, cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False,
        sel=SelectionConfig(refined=refined, score_emitted=emitted,
                            score_emitted_t=emitted_t),
        horizons=(5, 10, 15, 20), v0_conditioned=True, control_units="alat")
    with torch.no_grad():
        dec.anchor_controls.copy_(torch.randn(8, 2) * 0.5)
        for p in dec.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05)
    return dec.eval()


def _run(dec, seed=3):
    torch.manual_seed(seed)
    with torch.no_grad():
        return dec(torch.randn(6, 16, 4, 4), torch.randn(6, 4),
                   v_ms=torch.full((6,), 10.0), steps=2)


def test_p14_flag_pair_arms_the_lever_and_the_default_does_not():
    """⛔ THE DEFECT, PINNED IN BOTH DIRECTIONS. Default OFF must stamp False
    (that is the shipped refcv5 state); the pair must stamp True."""
    assert _run(_decoder(False, False))["sel_tele"][
        "sampler_ranks_the_fan"] is False
    assert _run(_decoder(True, True))["sel_tele"][
        "sampler_ranks_the_fan"] is True


def test_p14_arming_moves_the_ranked_score_but_never_the_fan():
    """⭐ THE INVARIANT THAT MAKES P14 SAFE TO ADOPT: the extra pass keeps its
    confidence and DISCARDS its offset, so `anchor_traj` -- and therefore every
    banked oracle-in-fan contrast this lever is paired against -- is
    bit-unchanged. If this ever fails, no prior number is comparable."""
    off, on = _run(_decoder(False, False)), _run(_decoder(True, True))
    assert torch.equal(off["anchor_traj"], on["anchor_traj"])
    assert float((off["sel_score"] - on["sel_score"]).abs().max()) > 1e-8, \
        "the flag would be INERT -- an extra decoder pass bought for nothing"


@pytest.mark.parametrize("flags", (["--sel-refined"], ["--sel-score-emitted"]))
def test_p14_trainer_refuses_the_split_pair(flags, tmp_path):
    """⛔ MUTATION, NOT INSPECTION: actually invoke the trainer with half the
    pair and require a NON-ZERO exit naming the reason.

    `--sel-refined` alone is the MEASURED-HARMFUL lever (0.0259 m separated
    WORSE, 29.82 % of picks flipped); `--sel-score-emitted` alone is INERT
    because the ranked score is `refined if sel.refined else conf`.
    """
    p = subprocess.run(
        [sys.executable, str(TRAINER), *flags, "--steps", "1",
         "--out", str(tmp_path)],
        capture_output=True, timeout=600)
    err = (p.stdout + p.stderr).decode("utf-8", errors="replace")
    assert p.returncode != 0, "the trainer ACCEPTED half the pair"
    assert "sel-refined" in err or "sel-score-emitted" in err, \
        f"refused, but not for this reason: {err[-400:]}"


def test_p14_trainer_help_lists_the_flags_and_mine_are_ascii():
    """The flags must be REACHABLE from `--help`, and MY help strings ASCII.

    ⚠️ **A CORRECTION, LOGGED RATHER THAN QUIETLY DROPPED.** This test was
    first written asserting the WHOLE help is cp1252-encodable, on the belief
    that a non-ASCII marker makes `--help` raise on this dev box. **That is
    false, and MEASURED false:** `--help` exits **0** under
    `PYTHONIOENCODING=cp1252:strict`, because Python does not encode-strict a
    redirected stdout the way a `print()` to a cp1252 console does. The
    UnicodeDecodeError that prompted the test came from the PARENT process
    decoding the child's UTF-8 output as cp1252 -- a harness bug of mine, not a
    trainer defect.

    ⛔ The over-broad version would also have FAILED ON 18 PRE-EXISTING FLAGS
    (`--size`, `--arm`, `--v2-cache`, `--goal-str`, ...) owned by other
    streams, i.e. a test that fails for reasons its owner cannot fix. Scoped to
    what this stream actually controls.
    """
    p = subprocess.run([sys.executable, str(TRAINER), "--help"],
                       capture_output=True, timeout=600)
    assert p.returncode == 0
    txt = p.stdout.decode("utf-8", errors="replace")
    assert "--sel-score-emitted" in txt and "--sel-refined" in txt
    src = TRAINER.read_text(encoding="utf-8")
    for flag in ("--sel-refined", "--sel-score-emitted", "--sel-score-emitted-t"):
        i = src.index(f'g5.add_argument("{flag}"')
        block = src[i:src.index("\n    g5.add_argument", i + 10)
                    if "\n    g5.add_argument" in src[i + 10:] else i + 1200]
        block = block[:block.index(')\n')] if ')\n' in block else block
        block.encode("ascii")     # raises if MY help strings are non-ASCII


# ====================================================================== P4 ==
def test_p4_head_widths_match_the_frozen_vocabulary():
    assert len(v7l.HEADS["str_goal"]) == 8
    assert len(v7l.HEADS["str_action"]) == 7
    out = st.StrategicTokenHead(32)(torch.randn(5, 32))
    assert out["str_goal"].shape == (5, 8)
    assert out["str_action"].shape == (5, 7)


def test_p4_off_vocabulary_token_raises_and_is_not_encoded_as_absent():
    """⛔⛔ THE 11.24 % HOLE. MEASURED on the banked v7 sample: 90 of 801
    `a_str` records carry `REDUCE_TO_FOLLOW_ROUTE`, which is NOT in
    `STRATEGIC_ACTION_TOKENS_V7` (the absence is pinned by
    test_vocab_v7_frozen). Encoding it as -1 would let `ignore_index` swallow
    it and delete that supervision silently."""
    with pytest.raises(st.OffVocabularyToken):
        st.encode_targets(["REDUCE_TO_FOLLOW_ROUTE"], "str_action")
    # the dangerous branch exists, is opt-in, and is named
    assert int(st.encode_targets(["REDUCE_TO_FOLLOW_ROUTE"], "str_action",
                                 strict=False)[0]) == -1
    # ...and an ABSENT label is a DIFFERENT thing, encoded the same way only
    # because it means "no supervision here", which is the band, not a defect.
    assert int(st.encode_targets([None], "str_goal")[0]) == -1


def test_p4_head_width_mismatch_is_refused():
    with pytest.raises(st.StrategicVocabMismatch):
        st.assert_head_matches_vocabulary(torch.zeros(2, 3), "str_goal")


def test_p4_loss_never_guards_itself_out_of_existence():
    """⛔ At weight 0.0 the parameters must still receive a ZEROS gradient, so
    `p.grad is None` keeps meaning NOT WIRED. MEASURED programme-wide: 42 of
    138 optimizer tensors read no gradient because terms were guarded behind
    `if w > 0`, which is indistinguishable from a head that was never built."""
    head = st.StrategicTokenHead(32)
    loss, _ = st.strategic_loss(
        head(torch.randn(8, 32)),
        {"str_goal": torch.randint(0, 8, (8,)),
         "str_action": torch.randint(0, 7, (8,))}, w=0.0)
    loss.backward()
    assert all(p.grad is not None for p in head.parameters())
    assert all(float(p.grad.abs().sum()) == 0.0 for p in head.parameters())


def test_p4_empty_band_is_finite_and_reports_why():
    """⚠️ 88.57 % of frames carry no strategic GT. An all-ignored batch must
    give a FINITE loss (CE returns NaN there) and report `n_supervised = 0`, so
    a zero loss reads as 'nothing was in band', not 'the head is broken' -- the
    mask trap that made a route head look dead until `route_valid=True` moved
    it 0.0 -> 0.687."""
    head = st.StrategicTokenHead(32)
    loss, tele = st.strategic_loss(
        head(torch.randn(8, 32)),
        {"str_goal": torch.full((8,), -1), "str_action": torch.full((8,), -1)},
        w=1.0)
    assert torch.isfinite(loss) and float(loss) == 0.0
    assert tele["str_goal_n_supervised"] == 0
    assert tele["str_action_n_supervised"] == 0


def test_p4_majority_control_reads_exactly_one_over_k():
    """⛔ THE CONTROL THAT MUST READ A KNOWN VALUE. The measured corpus skew is
    FOLLOW_ROUTE 549 / TURN_L 120 / TURN_R 132 of 801. A constant predictor
    scores 0.6854 POOLED -- which would read as a working head -- and exactly
    1/K macro, which is the no-information value."""
    y = torch.cat([torch.zeros(549, dtype=torch.long),
                   torch.ones(120, dtype=torch.long),
                   torch.full((132,), 2, dtype=torch.long)])
    mc = st.majority_control_recall(y, "str_goal")
    assert mc["macro_recall_full_vocab"] == pytest.approx(1 / 8)
    assert mc["macro_recall_over_supported"] == pytest.approx(1 / 3)
    assert mc["majority_share"] == pytest.approx(549 / 801)
    logits = torch.zeros(801, 8)
    logits[:, 0] = 10.0
    pc = st.per_class_recall(logits, y, "str_goal")
    assert pc["pooled_accuracy_DO_NOT_QUOTE"] == pytest.approx(549 / 801)
    assert pc["macro_recall_over_supported"] == pytest.approx(1 / 3)
    # ⛔ a class with NO support reports recall None, never 0.0 -- a 0.0 would
    # be averaged in and silently depress the macro score of a head that was
    # never given a chance to learn that class.
    assert pc["per_class"]["STOP_AT_FOLLOW_ROUTE"]["recall"] is None


def test_p4_head_has_no_nav_input_path():
    """⛔ MEASURED: 53.1 % of the strategic goal label is already in the nav
    command (determinism 0.8302 over 801 clips). Nav is a legitimate inference
    input in general, which is exactly why this head must not take one -- it
    would report half its own input back as skill, the flagship route head's
    369/369 echo one layer up."""
    import inspect
    sig = inspect.signature(st.StrategicTokenHead.forward)
    assert list(sig.parameters) == ["self", "ctx"], (
        "StrategicTokenHead.forward grew a parameter; if it is a nav path, "
        "the head owes a nav-ablated control before any number is quotable")
