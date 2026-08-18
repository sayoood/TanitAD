"""F-11 / catalog S1 — MULTI-TICK STRATEGIC ROLLOUT: both directions.

THE SPEC, quoted (two independent locations, established BEFORE a line of the
implementation was written):

  * ``…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:79`` —
    *"S1 | long-horizon latent prediction (own predictor, Δt ≈ 1 s ticks) on
    the T-layer's latent sequence | strategic dynamics = evolution of manoeuvre
    context, not pixels | ADE(8-30 s) vs CV/corridor baselines at T1"*
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:70`` — *"training
    target is ONE strategic tick ahead (stride_str = 20 steps = 2.0 s at the
    0.5 Hz clock) — a 1-tick loss… Multi-tick strategic rollout training is not
    built. Fix F-11"*, and ``:216`` — *"F-11 | P3 | S1 multi-tick strategic
    rollout (8-30 s = 4-15 strategic ticks)…"*

⛔ THE TWO CENTRAL FACTS THIS FILE PINS.

1. **The cell IS expressible, and the C115 question is why that needed
   checking.** C115 established that ``z_tac`` — hence ``z_str`` — is a function
   of the LAST FRAME ALONE. A cell assuming the strategic LATENT integrates a
   window would be a no-op. This one does not: the temporal structure it uses
   lives in ``predictor_str``, a genuine z(t) -> z(t+stride) map, and a
   multi-tick roll is that map composed with itself.

2. ⛔ **The catalog's 8-30 s HORIZON IS NOT REACHABLE ON THIS CORPUS, and that
   is arithmetic, not opinion.** ``t_max = frames - window - max_horizon``
   (``tanitad/data/_contract.py:120``) and a K-tick roll needs
   ``max_horizon = K*stride_str``. On the 120-frame cache windows/episode is
   ``114 - 20K``: K=4 (8 s) costs 64 % of the windows and K>=6 (12 s) yields
   ZERO. 30 s is longer than a 12 s episode. Pinned below so a later launch
   cannot quietly truncate the ladder instead of amending the spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from test_v6_gstr_port import _small  # noqa: E402
from train_v6_staged import (  # noqa: E402
    S1_CONTROL_MIN_N, STAGE_MAY_INTRODUCE, V6LossWeights,
    reachable_strategic_ticks, s1_persistence_control, s1_rollout_loss,
    synthetic_train_batch, v6_loss_step)


def _stack(seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(_small())


def _targets(n: int, k: int, d: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, d, generator=g)


# ===========================================================================
# 1. ⛔ THE CORPUS LIMIT — the spec-amendment finding, pinned as arithmetic
# ===========================================================================

def test_the_reachable_tick_table_on_the_live_geometry():
    """window 6, stride_str 20, 120-frame cache. Independently corroborated:
    ``PI_DECISIONS_2026-08-12.md`` §D4 MEASURED 94 windows/episode at
    ``max_horizon=20``, which is exactly the K=1 row here."""
    r = reachable_strategic_ticks(120, window=6, stride_str=20)
    assert r["windows_per_episode"][1] == 94      # the 1-tick baseline
    assert r["windows_per_episode"][2] == 74
    assert r["windows_per_episode"][3] == 54
    assert r["windows_per_episode"][4] == 34      # 8 s — the catalog's floor
    assert r["windows_per_episode"][5] == 14      # 10 s — the ceiling
    assert r["windows_per_episode"][6] == 0       # 12 s — corpus exhausted
    assert r["max_k"] == 5
    assert r["horizon_s_at_max_k"] == pytest.approx(10.0)


def test_the_catalog_band_8_to_30s_is_NOT_reachable():
    """⛔ The catalog asks for 4-15 ticks. Only K=4 is reachable at all, and it
    costs 64 % of the windows. This is a SPEC-AMENDMENT finding: it needs a
    longer re-extraction of the SAME episode list (PI decision D4), never a
    re-pick of which episodes enter."""
    r = reachable_strategic_ticks(120, window=6, stride_str=20)
    catalog = list(range(4, 16))                  # 8 s .. 30 s
    reachable = [k for k in catalog if r["windows_per_episode"].get(k, 0) > 0]
    assert reachable == [4, 5]
    assert r["windows_per_episode"][4] / r["windows_per_episode"][1] < 0.4


def test_a_longer_episode_moves_the_table_instead_of_invalidating_the_guard():
    """The 120 figure is INHERITED, so the guard is parameterised on the corpus
    it actually loaded. 30 s (K=15) needs >= 306 frames."""
    assert reachable_strategic_ticks(306, window=6, stride_str=20)["max_k"] == 14
    assert reachable_strategic_ticks(326, window=6, stride_str=20)["max_k"] == 15


def test_reachability_refuses_nonsense_geometry():
    for kw in ({"episode_frames": 0}, {"window": 0}, {"stride_str": 0}):
        args = {"episode_frames": 120, "window": 6, "stride_str": 20} | kw
        with pytest.raises(ValueError):
            reachable_strategic_ticks(args.pop("episode_frames"), **args)


# ===========================================================================
# 2. the loss does what it claims
# ===========================================================================

def test_k1_is_exactly_s1_latent():
    """⭐ The continuity pin: the first tick of the multi-roll IS the existing
    1-tick loss, bit-for-bit. If it were not, F-11 would be a different
    objective wearing S1's name and the K=1-vs-K>1 arms would not be a
    controlled comparison."""
    s = _stack()
    z = torch.randn(6, s.cfg.d_str)
    tgt = _targets(6, 3, s.cfg.d_str)
    _, log = s1_rollout_loss(s, z, tgt)
    cut = s.cfg.isolate_planner_from_encoder
    a = s.act_head_str(s._cut(z, cut))
    zh = s.predictor_str(z, s.vocab_a_str.encode(a["probs"], a["args"]))
    s1_latent = float((zh.float() - tgt[:, 0].float()).abs().mean())
    assert log["s1_multi_k1"] == pytest.approx(s1_latent, abs=1e-12)


def test_the_roll_is_CLOSED_not_teacher_forced():
    """Tick k+1 must be reached from tick k's OWN prediction. Perturbing the
    TARGET at tick 1 must not move the tick-2 prediction; perturbing the
    starting latent must move every tick."""
    s = _stack()
    z = torch.randn(6, s.cfg.d_str)
    t1 = _targets(6, 3, s.cfg.d_str)
    t2 = t1.clone()
    t2[:, 0] += 5.0                       # a wildly different tick-1 target
    _, la = s1_rollout_loss(s, z, t1)
    _, lb = s1_rollout_loss(s, z, t2)
    assert la["s1_multi_k1"] != pytest.approx(lb["s1_multi_k1"])
    assert la["s1_multi_k2"] == pytest.approx(lb["s1_multi_k2"], abs=1e-12)
    assert la["s1_multi_k3"] == pytest.approx(lb["s1_multi_k3"], abs=1e-12)


def test_every_tick_is_logged_so_the_degradation_curve_is_visible():
    """Uniform weighting is a DECLARED choice; the per-tick rows are what make
    it auditable instead of hidden inside one pooled number."""
    s = _stack()
    loss, log = s1_rollout_loss(s, torch.randn(4, s.cfg.d_str),
                                _targets(4, 4, s.cfg.d_str))
    assert log["s1_multi_k"] == 4
    per_k = [log[f"s1_multi_k{j}"] for j in range(1, 5)]
    assert log["s1_multi"] == pytest.approx(sum(per_k) / 4, rel=1e-6)


def test_the_loss_is_differentiable_into_layer_str_and_nothing_below():
    """The roll must train ``predictor_str``/``act_head_str`` — and, because the
    strategic TARGET is built under no_grad by the caller, must reach no
    encoder."""
    s = _stack()
    z = torch.randn(5, s.cfg.d_str, requires_grad=True)
    loss, _ = s1_rollout_loss(s, z, _targets(5, 3, s.cfg.d_str))
    loss.backward()
    assert s.predictor_str.in_proj.weight.grad is not None
    assert torch.linalg.norm(s.predictor_str.in_proj.weight.grad) > 0
    assert all(p.grad is None for p in s.encoder.parameters())


def test_a_perfect_rollout_target_scores_exactly_zero_POSITIVE_CONTROL():
    """Hand the loss its own rollout as the target: the floor is EXACTLY 0.
    (C79: D1 was withdrawn because a probe failed its positive control.)"""
    s = _stack()
    z = torch.randn(4, s.cfg.d_str)
    cut = s.cfg.isolate_planner_from_encoder
    with torch.no_grad():
        cur, rolled = z, []
        for _ in range(3):
            a = s.act_head_str(s._cut(cur, cut))
            cur = s.predictor_str(cur, s.vocab_a_str.encode(a["probs"],
                                                            a["args"]))
            rolled.append(cur)
        tgt = torch.stack(rolled, dim=1)
    loss, _ = s1_rollout_loss(s, z, tgt)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_the_loss_falls_when_the_predictor_is_optimised_POSITIVE_CONTROL():
    s = _stack()
    z = torch.randn(8, s.cfg.d_str)
    tgt = _targets(8, 3, s.cfg.d_str)
    opt = torch.optim.Adam(
        [p for n, p in s.named_parameters()
         if n.startswith(("predictor_str.", "act_head_str."))], lr=1e-3)
    first = float(s1_rollout_loss(s, z, tgt)[0])
    for _ in range(25):
        opt.zero_grad()
        loss, _ = s1_rollout_loss(s, z, tgt)
        loss.backward()
        opt.step()
    assert float(s1_rollout_loss(s, z, tgt)[0]) < first


# ===========================================================================
# 3. ⚠️ THE DEGENERACY — the identity map, and why the control is mandatory
# ===========================================================================

def test_the_HOLD_rollout_beats_an_untrained_one_on_slow_drift_targets():
    """⛔ THE DEGENERATE-AT-INIT CHECK, MEASURED.

    Strategic ticks are 2 s apart on a PER-FRAME encode, so consecutive
    strategic latents are similar. Under that (realistic) regime the identity
    map — emit ``z_str`` and never move — is a strong solution, and at init it
    beats the predictor by a wide margin. F-11 therefore does NOT start at its
    own global minimum the way F-8 does, but its easiest descent direction is
    towards doing nothing. That is exactly what
    :func:`s1_persistence_control` exists to detect, and why no S1 claim is
    admissible without it.
    """
    s = _stack()
    g = torch.Generator().manual_seed(11)
    z = torch.randn(64, s.cfg.d_str, generator=g)
    drift = 0.05 * torch.randn(64, 3, s.cfg.d_str, generator=g)
    tgt = z.unsqueeze(1) + drift                      # slow-drift targets
    out = s1_persistence_control(s, z, tgt)
    assert out["verdict"] == "NO_BETTER_THAN_HOLD"
    assert out["ratio"] > 1.0
    assert out["loss_hold"] < out["loss_model"]


def test_the_control_says_OK_once_the_predictor_actually_predicts():
    """The other direction (C95/C97). ⚠️ The targets here carry a LEARNABLE
    drift (``z + k*d`` for a fixed d) rather than pure noise, because that is
    the only regime in which beating HOLD is possible at all: against
    ``z + noise`` the identity is the Bayes-optimal answer and no predictor can
    win. That asymmetry is itself the finding — the control is informative
    exactly when the strategic latent has learnable dynamics."""
    s = _stack()
    g = torch.Generator().manual_seed(5)
    z = torch.randn(48, s.cfg.d_str, generator=g)
    d = 0.5 * torch.randn(1, s.cfg.d_str, generator=g)
    tgt = torch.stack([z + (j + 1) * d for j in range(2)], dim=1)
    opt = torch.optim.Adam(
        [q for n, q in s.named_parameters()
         if n.startswith(("predictor_str.", "act_head_str."))], lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = s1_rollout_loss(s, z, tgt)
        loss.backward()
        opt.step()
    out = s1_persistence_control(s, z, tgt)
    assert out["verdict"] == "OK", out
    assert out["ratio"] < 1.0


def test_the_control_refuses_a_verdict_below_its_sample_floor():
    """⛔ A hold/model ratio from a handful of windows is noise. MEASURED for
    the sibling T2 control at the null: n=4 spanned 0.397-3.361."""
    assert S1_CONTROL_MIN_N == 32
    s = _stack()
    out = s1_persistence_control(s, torch.randn(8, s.cfg.d_str),
                                 _targets(8, 3, s.cfg.d_str))
    assert out["verdict"] == "REFUSED_TOO_FEW"
    assert "ratio" not in out              # no number to quote out of context


def test_the_control_names_the_case_where_hold_is_exactly_zero():
    """If the strategic targets ARE the current latent there is no dynamics to
    learn, and a ratio against a zero denominator must not be reported as a
    number."""
    s = _stack()
    z = torch.randn(40, s.cfg.d_str)
    out = s1_persistence_control(s, z, z.unsqueeze(1).repeat(1, 3, 1))
    assert out["verdict"] == "DEGENERATE_TARGETS_EQUAL_Z"


# ===========================================================================
# 4. the loss's own guards — each paired with its opposite
# ===========================================================================

def test_the_loss_refuses_K_less_than_2():
    s = _stack()
    with pytest.raises(ValueError, match="K >= 2"):
        s1_rollout_loss(s, torch.randn(4, s.cfg.d_str),
                        _targets(4, 1, s.cfg.d_str))


def test_the_loss_refuses_a_shape_mismatch():
    s = _stack()
    with pytest.raises(ValueError, match=r"\[B, K, d_str\]"):
        s1_rollout_loss(s, torch.randn(4, s.cfg.d_str),
                        torch.randn(4, s.cfg.d_str))
    with pytest.raises(ValueError, match="does not match"):
        s1_rollout_loss(s, torch.randn(4, s.cfg.d_str),
                        _targets(4, 3, s.cfg.d_str + 1))


def test_a_legal_call_raises_NOTHING():
    s = _stack()
    loss, log = s1_rollout_loss(s, torch.randn(4, s.cfg.d_str),
                                _targets(4, 2, s.cfg.d_str))
    assert torch.isfinite(loss) and log["s1_multi_k"] == 2


def test_v6_loss_step_refuses_the_term_without_its_target():
    """A term that cannot fire is worse than an absent one, because the launch
    line advertises it."""
    s = _stack()
    b = synthetic_train_batch(s, batch=2, k=4, seed=3)     # no multi target
    with pytest.raises(ValueError, match="z_str_multi_target"):
        v6_loss_step(s, b, stage="S-S",
                     weights=V6LossWeights(w_s1_multi=1.0))


def test_v6_loss_step_runs_the_term_when_the_target_is_present():
    s = _stack()
    b = synthetic_train_batch(s, batch=2, k=4, seed=3, s1_multi_k=3)
    r = v6_loss_step(s, b, stage="S-S",
                     weights=V6LossWeights(w_s1_multi=1.0))
    assert r["log"]["s1_multi_k"] == 3
    assert torch.isfinite(r["loss"])


def test_the_term_is_ABSENT_from_the_log_when_its_weight_is_zero():
    """Inertness at the loss-assembly level, not just at the build level."""
    s = _stack()
    b = synthetic_train_batch(s, batch=2, k=4, seed=3, s1_multi_k=3)
    r = v6_loss_step(s, b, stage="S-S", weights=V6LossWeights())
    assert not any(k.startswith("s1_multi") for k in r["log"])


# ===========================================================================
# 5. the stage contract
# ===========================================================================

def _pf(*extra: str, stage: str = "S-S") -> list[str]:
    """Refusals from the REAL parser's namespace — a hand-built one silently
    diverges from the flags `main` actually registers (the `shared_encoder`
    bug F-7 hit was exactly that)."""
    import train_v6_staged as T
    a = T.build_parser().parse_args(
        ["--stage", stage, "--out", "unused", *extra])
    return T.preflight(a)


def test_for_stage_zeroes_the_term_where_layer_str_is_frozen():
    """S-W and S-T freeze ``layer_str``; a term in force there would be
    advertised in the launch line and train nothing."""
    w = V6LossWeights(w_s1_multi=1.0)
    assert w.for_stage("S-W").w_s1_multi == 0.0
    assert w.for_stage("S-T").w_s1_multi == 0.0
    assert w.for_stage("S-S").w_s1_multi == 1.0      # its stage
    assert w.for_stage("S-J").w_s1_multi == 1.0      # joint polish


def test_preflight_refuses_K1_and_the_wrong_stages():
    assert any("K=1" in x for x in
               _pf("--w-s1-multi", "1.0", "--s1-multi-k", "1"))
    for st in ("S-W", "S-T"):
        assert any("F-11 is an S-S" in x for x in
                   _pf("--w-s1-multi", "1.0", stage=st))


def test_preflight_refuses_an_unreachable_K_before_the_corpus_mounts():
    p = _pf("--w-s1-multi", "1.0", "--s1-multi-k", "6")
    assert any("ZERO windows" in x and "8-30 s band" in x for x in p)


def test_preflight_passes_a_legal_F11_launch():
    """⭐ C95/C97 — the passes-everything twin of the refusals above."""
    p = _pf("--w-s1-multi", "1.0", "--s1-multi-k", "4")
    assert not [x for x in p if "s1-multi" in x or "F-11" in x], p


def test_F11_needs_no_stage_may_introduce_entry():
    """⭐ THE INSERTION-POINT ANSWER, pinned. ``predictor_str`` and
    ``act_head_str`` are ALREADY ``layer_str``, so F-11 adds no key and there
    is nothing for the allowlist to adjudicate — it may be enabled over an
    existing checkpoint loaded tensor-strict, like F-8."""
    for stage, allowed in STAGE_MAY_INTRODUCE.items():
        assert not any("s1_multi" in p for p in allowed), (stage, allowed)


def test_the_06b8782_class_does_not_apply_to_F11():
    from tanitad.models.v6 import LADDER_UNTRAINED_GROUPS, MODULE_GROUPS
    assert MODULE_GROUPS == ("encoder", "readout", "predictor_op", "layer_tac",
                             "layer_str", "planner", "aux", "interp")
    assert LADDER_UNTRAINED_GROUPS == frozenset({"interp"})


# ===========================================================================
# 6. ⛔ INERT AT DEFAULT
# ===========================================================================

def test_default_build_is_untouched_at_the_production_geometry():
    """⛔ 87,893,449 params / 405 keys — the live tensor-strict v6F S-W resume
    depends on this. F-11 adds ZERO parameters."""
    m = V6Stack(V6Config())
    assert sum(p.numel() for p in m.parameters()) == 87_893_449
    assert len(m.state_dict()) == 405


def test_the_default_weight_is_zero_everywhere():
    assert V6LossWeights().w_s1_multi == 0.0
    for st in ("S-W", "S-T", "S-S", "S-J"):
        assert V6LossWeights().for_stage(st).w_s1_multi == 0.0


def test_the_batched_per_tick_target_build_does_not_transpose():
    """⛔ THE RESHAPE THE TRAINER ACTUALLY DOES, pinned.

    The per-tick targets are built in ONE encoder pass over ``[B, K]`` future
    frames flattened into the batch axis and reshaped back — the same batching
    discipline the ``need_k`` block uses, because K separate encodes would
    waste the dimension the GPU exists for. **A flatten/reshape pair is exactly
    where a silent transpose hides**, and a transposed target would train the
    roll against the wrong window's future while every loss curve looked
    plausible. This asserts row (i, j) of the batched build equals the
    single-frame target for frame (i, j).

    (The trainer's own copy of this needs a corpus, so it is the one part of
    F-11 that `--dry-run` cannot exercise; this is the seam-level proof.)
    """
    s = _stack()
    b, k = 3, 2
    c, (h, w) = s.cfg.encoder.in_channels, s.cfg.encoder.image_hw()
    g = torch.Generator().manual_seed(2)
    ff = torch.randn(b, k, c, h, w, generator=g)
    with torch.no_grad():
        flat = ff.reshape(b * k, c, h, w)
        batched = s.layer_targets(s.readout(s.encoder(flat)), None,
                                  None)["z_str"].reshape(b, k, -1)
        for i in range(b):
            for j in range(k):
                one = s.layer_targets(
                    s.readout(s.encoder(ff[i, j][None])), None, None)["z_str"]
                assert torch.allclose(batched[i, j], one[0], atol=1e-5), (i, j)
