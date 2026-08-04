"""S1's CLIMB-OUT — the two ZERO-PARAMETER distribution matches. Rationale:
``Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md`` §6.3 (fifth branch) and
``…/incoming/2026-08-03-s1-climbout/PREREG_S1_CLIMBOUT.md`` §1.

WHY THESE TWO FLAGS EXIST
============================================================================
E-SEL-0 MEASURED that REF-C's discarded refined confidence ranks **0.8372 m**
(base) / **0.9187 m** (XL) WORSE than the shipped t=0 classifier score —
separated, both arms — while still scoring **8.7x / 16.6x chance**. So the
readout is OFF-DISTRIBUTION, not uninformative, and S1 must CLIMB OUT
(supervise it) rather than HARVEST it. The two flags here remove the two places
where the object that is SCORED, the object that is SUPERVISED and the object
that is EMITTED are still three different things:

  S1b ``sel_score_emitted``  MEASURED FROM SOURCE: ``_decode(kv, cond, x_in, t)``
      returns the confidence **of ``x_in``** alongside the offset that improves
      it, and the loop emits ``x = x_in + off``. So ``refined`` scores the
      estimate the LAST pass CONSUMED and the trajectories that leave the
      decoder are scored by NO head. The shipped ranker is 2 passes stale; S1's
      is 1 pass stale.
  S1c ``sel_ce_reach``  the ranked-score CE is a FULL-FAN softmax while the
      argmax solves a 26-28 %-sized problem: 73.76 % (base) / 72.08 % (XL) of
      the fan is unreachable, never selected, and deleting it moves ADE by
      EXACTLY 0.0.

WHAT THIS FILE PINS
============================================================================
(a) CAPACITY: both flags cost EXACTLY 0 parameters. The check that caught a
    +272,001-parameter head where +897 sufficed.
(b) BOUNDEDNESS: S1b must change the READOUT and leave the EMITTED FAN
    bit-unchanged. Every D-SEL contrast is paired against the published
    oracle-in-fan (0.1914 base / 0.1640 XL), which is DEFINED ON THAT FAN — a
    changed fan would silently re-baseline the whole comparison.
(c) THE CONTROL TRAVELS IN THE SAME FORWARD: ``prefinal_logits`` is exactly the
    readout S1 ships today, so the S1b contrast can never be confounded by float
    non-determinism between two runs.
(d) S1c's masked CE is FINITE in loss AND gradient, and its target is ALWAYS a
    survivor — a -inf-masked cross-entropy whose target is masked is the
    standard way to produce NaN.
(e) S1c is a LOSS-ONLY flag: it must not perturb the forward at all.
(f) THE FAIL-LOUD: ``--sel-ce-reach`` without ``--sel-reach-clamp`` is refused at
    PARSE time. A silently-inert flag that config.json records as ON is the
    D-TAC1 ``tactical_speed_input`` failure ("a conservative guard that makes an
    effect unattributable is not conservative").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_train                                                   # noqa: E402
from tanitad.refs.refc import (RefCModel, SelectionConfig,          # noqa: E402
                               param_breakdown, refc_config,
                               refc_smoke_config)
from tests.test_refc import _make_cached_root                       # noqa: E402


def _build(seed: int = 0, **flags) -> RefCModel:
    cfg = refc_smoke_config()
    for k, v in flags.items():
        setattr(cfg, k, v)
    torch.manual_seed(seed)
    return RefCModel(cfg)


FRAMES = torch.randn(6, 4, 1, 64, 64)
V0 = torch.tensor([3.0, 10.0, 20.0, 0.5, 7.0, 14.0])


# ---------- (a) capacity: EXACTLY zero ---------------------------------------

def test_the_climb_out_costs_exactly_zero_parameters():
    """S1b buys one extra decoder PASS, not one extra weight; S1c is a change to
    a loss's normalising set. Neither may cost a parameter, and "zero" is a claim
    that has to be measured — the +272,001 head passed every review that did not
    count."""
    with torch.device("meta"):
        n0 = param_breakdown(RefCModel(refc_config()))["total"]
        for flags in ({"sel_score_emitted": True},
                      {"sel_ce_reach": True},
                      {"sel_score_emitted": True, "sel_ce_reach": True},
                      {"sel_refined": True, "sel_score_emitted": True,
                       "sel_reach_clamp": True, "sel_ce_reach": True}):
            cfg = refc_config()
            for k, v in flags.items():
                setattr(cfg, k, v)
            got = param_breakdown(RefCModel(cfg))["total"] - n0
            assert got == 0, f"{flags}: +{got} parameters, expected +0"
    assert n0 == 104_191_577                       # the registry's number


def test_defaults_are_off_and_the_decoder_policy_carries_only_what_it_reads():
    """``score_emitted`` is a DECODER policy (it changes a forward); ``ce_reach``
    is a TRAINER policy (it changes a loss). Putting the latter on
    ``SelectionConfig`` would advertise a field the decoder never reads."""
    s = SelectionConfig()
    assert s.refined is False and s.score_emitted is False
    assert s.score_emitted_t == -1          # -1 = continue the loop's schedule
    assert not hasattr(s, "ce_reach")
    c = refc_config()
    assert c.sel_score_emitted is False and c.sel_ce_reach is False
    assert c.selection().score_emitted is False
    c.sel_score_emitted = True
    assert c.selection().score_emitted is True and c.selection().any_on


# ---------- (b)+(c) S1b is real, bounded, and carries its own control ---------

def test_s1b_scores_the_emitted_fan_and_leaves_that_fan_bit_unchanged():
    """⛔ THE BOUND THAT MATTERS. The extra pass keeps its CONFIDENCE and
    DISCARDS its OFFSET, so ``anchor_traj`` — the object the published
    oracle-in-fan is defined on and every D-SEL contrast is paired against — is
    bit-identical. If this ever fails, every paired number in the D-SEL family
    has been silently re-baselined."""
    off = _build(sel_refined=True).eval()
    on = _build(sel_refined=True, sel_score_emitted=True).eval()
    on.load_state_dict(off.state_dict(), strict=True)
    with torch.no_grad():
        a = off(FRAMES, v0=V0, steps=2)
        b = on(FRAMES, v0=V0, steps=2)
    for k in ("anchor_traj", "offset", "anchor_logits"):
        assert torch.equal(a[k], b[k]), f"{k} moved — S1b is not bounded"
    assert not torch.equal(a["refined_logits"], b["refined_logits"])
    assert torch.equal(b["sel_score"], b["refined_logits"])


def test_s1b_carries_its_own_control_in_the_same_forward():
    """``prefinal_logits`` IS the readout S1 ships today, bit-for-bit. A
    cross-forward comparison would confound the change with float
    non-determinism between hosts/builds; this one cannot."""
    off = _build(sel_refined=True).eval()
    on = _build(sel_refined=True, sel_score_emitted=True).eval()
    on.load_state_dict(off.state_dict(), strict=True)
    with torch.no_grad():
        a, b = off(FRAMES, v0=V0, steps=2), on(FRAMES, v0=V0, steps=2)
    assert torch.equal(b["prefinal_logits"], a["refined_logits"])
    assert not torch.equal(b["prefinal_logits"], b["refined_logits"])


def test_s1b_is_inert_at_zero_denoise_steps_by_construction():
    """``--mode classifier`` runs 0 denoise passes, so there is no emitted fan
    distinct from the anchors and no extra pass is taken. Pinned so a
    ``--mode classifier --sel-score-emitted`` run cannot read as a live arm."""
    m = _build(sel_refined=True, sel_score_emitted=True).eval()
    with torch.no_grad():
        out = m(FRAMES[:2], v0=V0[:2], steps=0)
    assert out["refined_logits"] is out["anchor_logits"]
    assert "prefinal_logits" not in out


# ---------- (d)+(e) S1c: finite, targeted at a survivor, forward-neutral ------

def test_s1c_is_a_loss_only_flag_and_never_perturbs_the_forward():
    """S1c changes what the CROSS-ENTROPY normalises over. If it also moved a
    forward output, an arm carrying it would not be attributable to the loss."""
    a = _build(sel_refined=True, sel_reach_clamp=True).eval()
    b = _build(sel_refined=True, sel_reach_clamp=True, sel_ce_reach=True).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    with torch.no_grad():
        oa, ob = a(FRAMES, v0=V0, steps=2), b(FRAMES, v0=V0, steps=2)
    for k in ("anchor_logits", "refined_logits", "anchor_traj", "sel_score",
              "sel_idx", "traj", "reach_keep"):
        assert torch.equal(oa[k], ob[k]), k


def test_reach_keep_is_the_mask_the_argmax_actually_used():
    """ONE implementation. The exported mask must be the same object the argmax
    ranked over — re-deriving it in the trainer is how two definitions of the
    same survivor set drift apart, which is why ``reachability_mask`` is a
    re-export and not a copy."""
    m = _build(sel_refined=True, sel_reach_clamp=True).eval()
    with torch.no_grad():
        out = m(FRAMES, v0=V0, steps=2)
    keep = out["reach_keep"]
    assert keep.dtype == torch.bool and keep.shape == out["sel_score"].shape
    assert bool(keep.any(1).all())            # empty-set fallback, per window
    ranked = out["sel_score"].masked_fill(~keep, float("-inf")).argmax(1)
    assert torch.equal(ranked, out["sel_idx"])


def test_s1c_masked_cross_entropy_is_finite_and_targets_a_survivor():
    """A -inf-masked CE whose target is itself masked returns NaN. The target
    therefore MOVES WITH THE SUPPORT: it is the best candidate IN the survivor
    set, not the best in the fan. Both the loss and every gradient must be
    finite, and that is checked rather than argued."""
    m = _build(sel_refined=True, sel_reach_clamp=True, sel_ce_reach=True).train()
    out = m(FRAMES, v0=V0, steps=2)
    keep = out["reach_keep"]
    fan_err = out["anchor_traj"].norm(dim=-1).mean(-1)          # stand-in target
    tgt = fan_err.masked_fill(~keep, float("inf")).argmin(1)
    assert bool(keep.gather(1, tgt[:, None]).all()), "target is not a survivor"
    score = out["sel_score"].masked_fill(
        ~keep, torch.finfo(out["sel_score"].dtype).min / 4)
    loss = F.cross_entropy(score, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # ...and the support is REALLY smaller than the fan, else the flag is inert.
    assert float(keep.float().mean()) < 1.0


# ---------- (f) the fail-loud ------------------------------------------------

def test_s1c_is_refused_without_the_reachability_clamp(tmp_path):
    """⛔ Without S2 there is no survivor mask, so S1c would train the FULL-FAN
    CE while config.json recorded the restricted one — an arm that reads as a
    treatment and behaves as a control. Refused at PARSE time, not after a
    GPU-day."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="requires --sel-reach-clamp"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "x"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-refined", "--sel-ce-reach"])


def test_s1b_is_refused_without_the_supervision_it_depends_on(tmp_path):
    """⛔ THE SAME SILENT-INERT CLASS as S1c-without-S2, plus a MEASURED reason.

    With ``sel_refined`` off the ranked score is the t=0 ``conf``, so moving the
    REFINED readout cannot reach the argmax at all. And E-S1-0 MEASURED that at
    frozen 30 k weights the emitted readout selects **0.9924 m WORSE** than its
    predecessor (2.3024 vs 1.3100 on 881 windows, separated) — because scoring
    the emitted fan moves the head FURTHER from the only distribution
    ``loss_cls`` ever supervises. So S1b is admissible ONLY inside an arm that
    also supervises the readout it moves."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="requires --sel-refined"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "y"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v21",
            "--mode", "diffusion", "--sel-score-emitted"])


def test_s1b_timestep_token_is_selectable_and_still_costs_nothing():
    """The conf head is supervised at t=0 ONLY (``loss_cls`` runs on the t=0
    pass), so WHICH token the emitted readout is evaluated under is a real axis —
    and it is free, because ``time_embed`` already carries every index. Pinned so
    the axis cannot silently acquire a parameter."""
    with torch.device("meta"):
        n0 = param_breakdown(RefCModel(refc_config()))["total"]
        cfg = refc_config()
        cfg.sel_refined = cfg.sel_score_emitted = True
        cfg.sel_score_emitted_t = 0
        assert param_breakdown(RefCModel(cfg))["total"] - n0 == 0
    a = _build(sel_refined=True, sel_score_emitted=True).eval()
    b = _build(sel_refined=True, sel_score_emitted=True,
               sel_score_emitted_t=0).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    with torch.no_grad():
        oa, ob = a(FRAMES, v0=V0, steps=2), b(FRAMES, v0=V0, steps=2)
    assert torch.equal(oa["anchor_traj"], ob["anchor_traj"])   # fan untouched
    assert not torch.equal(oa["refined_logits"], ob["refined_logits"])


def test_trainer_runs_the_climb_out_and_reports_its_ce_support(tmp_path):
    """End to end: the arm trains, the banner records BOTH new axes, and the CE
    support fraction is logged per step — so a silent collapse to the whole fan
    (or to one candidate) is visible in ``train_log.jsonl`` rather than inferred
    afterwards."""
    root = _make_cached_root(tmp_path)
    out = tmp_path / "s1climb"
    refc_train.main([
        "--data-root", str(root), "--out", str(out), "--steps", "2",
        "--batch", "4", "--smoke", "--log-every", "1", "--workers", "0",
        "--labels", "v21", "--mode", "diffusion",
        "--sel-refined", "--sel-score-emitted", "--sel-reach-clamp",
        "--sel-ce-reach"])
    conf = json.loads((out / "config.json").read_text())
    for k in ("sel_refined", "sel_score_emitted", "sel_reach_clamp",
              "sel_ce_reach"):
        assert conf["cfg"][k] is True, k
    # ⭐ ZERO parameters, end to end — the arm is a pure objective/readout change.
    assert conf["param_breakdown"]["selection"] == 0
    met = json.loads((out / "metrics.json").read_text())
    for k in ("cls_refined", "sel_gap", "rank_acc", "frac_sel_2x_worse",
              "ce_support_frac"):
        assert k in met["final"], f"{k} missing from the step log"
    assert 0.0 < met["final"]["ce_support_frac"] <= 1.0
    assert met["final"]["sel_gap"] >= -1e-6      # selected is never < oracle
