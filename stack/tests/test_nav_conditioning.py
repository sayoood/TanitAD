"""Gates for nav conditioning (SPEC_NAV_CONDITIONING_ALL_LAYERS.md §4).

The four acceptance criteria, each as a test:
  1. the embedding is proven SHARED — change it once, all three layers move;
  2. a MISSING nav token RAISES rather than defaulting;
  3. the controls are implemented AND SHOWN TO FIRE — in particular a
     deliberate-regression arm whose nav is shuffled must be DETECTED;
  4. provenance is stamped.
"""
import pytest
import torch

from tanitad.models.nav_conditioning import (NAV_CONTROLS, NavArgStats,
                                             NavConditioner, NavTokenMissing,
                                             apply_nav_control,
                                             nav_provenance_stamp)
from tanitad.models.vocab_v7 import NAV_COMMAND_TOKENS


def _cond(d_model=16, **kw):
    """⚠️ zero_init=False by DEFAULT here. The production default IS zero-init
    (so introducing nav mid-ladder is loss-continuous), but a zero output
    projection makes every output 0.0 regardless of its input — which would make
    the sharedness and control tests pass VACUOUSLY. Zero-init gets its own
    dedicated test instead of silently disabling every other one."""
    torch.manual_seed(0)
    kw.setdefault("zero_init", False)
    return NavConditioner(d_model=d_model, **kw)


# ---------------------------------------------------------------- gate 1
def test_the_embedding_is_shared_across_layers():
    """⭐ SHARED BY CONSTRUCTION. Three per-layer copies would drift under
    training and the layers would stop meaning the same thing by the same token.
    Mutate the ONE embedding and every layer must move."""
    c = _cond()
    ids = torch.tensor([0, 1])
    args = torch.zeros(2, 2)
    before = {ln: c(ids, args, ln).clone() for ln in c.layers}
    with torch.no_grad():
        c.embed.weight += 1.0                      # ONE mutation
    after = {ln: c(ids, args, ln) for ln in c.layers}
    for ln in c.layers:
        assert not torch.allclose(before[ln], after[ln]), (
            f"{ln} did not move ⇒ it is not reading the shared embedding")
    assert set(c.layers) == {"operative", "tactical", "strategic"}


def test_all_three_layers_including_OPERATIVE_are_conditioned():
    """The directive names the operative layer explicitly."""
    c = _cond()
    for ln in ("operative", "tactical", "strategic"):
        out = c(torch.tensor([0]), torch.zeros(1, 2), ln)
        assert out.shape == (1, 16)


def test_unknown_layer_is_refused():
    with pytest.raises(KeyError):
        _cond()(torch.tensor([0]), torch.zeros(1, 2), "planner")


# ---------------------------------------------------------------- gate 2
def test_missing_nav_token_RAISES_and_never_defaults():
    """⛔ A silent default is the `_ensure_ego` size-threshold trap in a new
    costume: it would train an arm without the channel while its config claims
    otherwise, and make two arms silently incomparable."""
    with pytest.raises(NavTokenMissing) as e:
        _cond().ids_from_batch({"frames": torch.zeros(1)})
    assert "MANDATORY" in str(e.value)


def test_unknown_nav_token_RAISES():
    with pytest.raises(NavTokenMissing) as e:
        _cond().ids_from_batch({"nav_command": {"token": ["NAV_TELEPORT"]}})
    assert "unknown nav token" in str(e.value)


def test_known_tokens_map_to_ids():
    ids = _cond().ids_from_batch({"nav_command": {"token": list(NAV_COMMAND_TOKENS)}})
    assert ids.tolist() == list(range(len(NAV_COMMAND_TOKENS)))


# ---------------------------------------------------------------- gate 3
def test_all_controls_exist():
    assert set(NAV_CONTROLS) == {"real", "hold", "shuffled", "none"}


def test_shuffled_gives_every_item_ANOTHER_clips_nav():
    """⚠️ A random permutation leaves ~1/B items in place, which silently
    weakens the control on small batches — it would be partly measuring the real
    channel. The roll guarantees nothing keeps its own nav."""
    ids = torch.tensor([0, 1, 2, 0])
    args = torch.randn(4, 2)
    sid, sargs, active = apply_nav_control(ids, args, "shuffled")
    assert active is True
    assert not torch.equal(sid, ids)
    # no position may retain its own ARGS row (ids can collide by vocab reuse)
    assert not any(torch.equal(sargs[i], args[i]) for i in range(4))


def test_shuffled_REFUSES_a_batch_of_one():
    """With one item there is no other clip's nav; returning the same nav would
    make the control silently inert."""
    with pytest.raises(ValueError) as e:
        apply_nav_control(torch.tensor([0]), torch.zeros(1, 2), "shuffled")
    assert "batch >= 2" in str(e.value)


def test_none_switches_the_channel_off_and_reports_it():
    ids, args, active = apply_nav_control(torch.tensor([1, 2]),
                                          torch.ones(2, 2), "none")
    assert active is False, "the ablation must REPORT that the channel is off"
    assert torch.all(ids == 0) and torch.all(args == 0)


def test_THE_CONTROL_FIRES_on_a_deliberate_regression_arm():
    """⛔ GATE 3, THE LOAD-BEARING ONE (spec §4.3).

    A control that cannot detect a shuffled nav proves nothing about a real one —
    TRAIN-C8: presence of a control is not validity of a control. Build an arm
    that GENUINELY depends on nav, shuffle its nav, and require the output to
    change materially.
    """
    c = _cond()
    ids = torch.tensor([0, 1, 2, 0])
    args = torch.randn(4, 2)

    def nav_dependent_arm(i, a):                 # a model that really uses nav
        return c(i, a, "operative")

    real = nav_dependent_arm(ids, args)
    sid, sargs, _ = apply_nav_control(ids, args, "shuffled")
    shuffled = nav_dependent_arm(sid, sargs)
    delta = (real - shuffled).norm() / real.norm().clamp_min(1e-8)
    assert delta > 0.05, (
        f"shuffled-nav moved the output only {delta:.4f} on an arm BUILT to "
        f"depend on nav ⇒ the control cannot detect what it exists to detect")


def test_the_control_reads_ZERO_on_an_arm_that_ignores_nav():
    """⭐ The other direction, and it is what makes the control INFORMATIVE: on
    an arm that ignores nav, shuffling must change nothing. That is the INERT
    verdict the prereg's outcome 2 turns on."""
    def nav_blind_arm(i, a):
        return torch.ones(i.shape[0], 8)         # ignores both inputs
    ids, args = torch.tensor([0, 1, 2, 0]), torch.randn(4, 2)
    sid, sargs, _ = apply_nav_control(ids, args, "shuffled")
    assert torch.equal(nav_blind_arm(ids, args), nav_blind_arm(sid, sargs))


def test_zero_init_makes_INTRODUCTION_loss_continuous():
    """⛔ The production default: at init the nav term is EXACTLY zero, so
    adding the channel to a running ladder does not discontinuously change the
    loss. Same discipline as FiLM's zero-init (predictor.py:41-42)."""
    c = NavConditioner(d_model=16)                       # production defaults
    out = c(torch.tensor([0, 1]), torch.randn(2, 2), "operative")
    assert torch.all(out == 0.0), "an introduced channel must start inert"


def test_the_nav_term_is_GATED_so_it_cannot_dilute_the_action_channel():
    """⛔ H26, MEASURED (predictor.py:126): the UNGATED intent_proj term reached
    norm ~31.4 against act_emb ~28.3, DILUTING the action conditioning, and
    engaging intent was net-HARMFUL to the operative. Nav enters the same
    additive `cond` pathway as a THIRD term, so an ungated nav term would
    reproduce H26 with one more competitor. The gate starts small and grows only
    if training earns it."""
    c = _cond(gate_init=0.1)
    assert set(c.gate) == {"operative", "tactical", "strategic"}
    for ln in c.layers:
        assert float(c.gate[ln]) == pytest.approx(0.1)
        assert c.gate[ln].requires_grad, "the gate must be LEARNABLE, not a constant"
    ungated = _cond(gate_init=1.0)
    ids, args = torch.tensor([0, 1]), torch.randn(2, 2)
    small = c(ids, args, "operative").norm()
    big = ungated(ids, args, "operative").norm()
    assert small < big, "a smaller gate must produce a smaller term"


# ---------------------------------------------------------------- gate 4
def test_provenance_is_stamped_and_oracle_is_flagged():
    s = nav_provenance_stamp("ego-future", control="real")
    assert s["nav_is_oracle"] is True
    assert "shuffled-nav" in s["_read"], "the stamp must name the required control"
    assert nav_provenance_stamp("nav-system")["nav_is_oracle"] is False


def test_unknown_provenance_is_refused():
    with pytest.raises(ValueError):
        nav_provenance_stamp("guessed")


def test_a_nav_system_arm_is_marked_NOT_POOLABLE_with_an_oracle_arm():
    assert "NOT poolable" in nav_provenance_stamp("nav-system")["_read"]


# ---------------------------------------------------------------- arg stats
def test_arg_stats_come_from_the_FIT_SPLIT_and_refuse_emptiness():
    st = NavArgStats.from_fit_split([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    assert st.n_fit == 3 and st.distance_mean == pytest.approx(20.0)
    assert "FIT-SPLIT ONLY" in st.to_dict()["_read"]
    with pytest.raises(ValueError):
        NavArgStats.from_fit_split([], [])


def test_normalise_uses_the_stored_stats_not_the_batch():
    """⛔ Per-batch normalisation is the ridge-probe failure in a new costume: a
    statistic fitted on the data it will score inflates apparent information."""
    st = NavArgStats.from_fit_split([0.0, 100.0], [0.0, 10.0])
    a = st.normalise(torch.tensor([50.0]), torch.tensor([5.0]))
    b = st.normalise(torch.tensor([50.0, 999.0]), torch.tensor([5.0, 99.0]))
    assert torch.allclose(a[0], b[0]), "an added outlier must not move the first row"


# ------------------------------------------- the operative port (predictor.py)
def test_operative_port_is_None_safe_and_byte_identical():
    """⛔ nav_cond=None must reproduce the previous behaviour EXACTLY, so every
    existing checkpoint stays a strict subset and no arm changes by accident."""
    from tanitad.models.predictor import OperativePredictor, PredictorConfig
    torch.manual_seed(0)
    cfg = PredictorConfig()
    m = OperativePredictor(cfg, 8)
    st = torch.randn(2, cfg.window, 8)
    ac = torch.randn(2, cfg.window, cfg.action_dim)
    a, b = m(st, ac), m(st, ac, nav_cond=None)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_real_nav_term_REACHES_the_operative_output():
    """⚠️ FiLM's to_scale_shift is ZERO-INIT (predictor.py:41-42), so at init NO
    conditioning term moves the output — actions included. The control below
    proves the inertness is GLOBAL rather than a nav wiring bug, then FiLM is
    made live and nav must reach the output."""
    from tanitad.models.predictor import OperativePredictor, PredictorConfig
    torch.manual_seed(0)
    cfg = PredictorConfig()
    m = OperativePredictor(cfg, 8)
    st = torch.randn(2, cfg.window, 8)
    ac = torch.randn(2, cfg.window, cfg.action_dim)
    d = m.act_emb[-1].out_features

    base = m(st, ac)
    # CONTROL: actions are equally inert at init ⇒ global zero-init, not a nav bug
    assert all(torch.equal(base[k], m(st, torch.randn_like(ac))[k]) for k in base)
    assert all(torch.equal(base[k], m(st, ac, nav_cond=torch.randn(2, d))[k])
               for k in base)

    with torch.no_grad():
        for blk in m.blocks:
            torch.nn.init.normal_(blk.film.to_scale_shift.weight, std=0.02)
    live = m(st, ac)
    assert any(not torch.equal(live[k], m(st, ac, nav_cond=torch.randn(2, d))[k])
               for k in live), "with FiLM live, nav must reach the output"


# ------------------------------------------------- per-layer widths (the sites)
def test_each_layer_gets_ITS_OWN_width():
    """⛔ THE THREE CONDITIONING SITES HAVE DIFFERENT SHAPES.

    MEASURED from source: the operative site is an ADDITIVE FiLM pathway at the
    predictor's hidden width (predictor.py:199), while FTac CONCATENATES
    (tactical.py:267-272) and nav must instead be projected into the existing
    g_flat width and ADDED — the `cond_tac_dyn` precedent (v6.py:4904-4907).
    A single d_model would give two of three layers the WRONG width and die at
    the first forward.
    """
    c = NavConditioner(widths={"operative": 256, "tactical": 128,
                               "strategic": 64}, zero_init=False)
    ids, args = torch.tensor([0, 1]), torch.randn(2, 2)
    for ln, w in c.widths.items():
        assert c(ids, args, ln).shape == (2, w), ln


def test_constructing_without_widths_or_d_model_RAISES():
    """A silent default width is the same class of defect as a silent default
    nav token — it would produce a shape error far from its cause."""
    with pytest.raises(ValueError) as e:
        NavConditioner()
    assert "widths=" in str(e.value)


def test_d_model_still_serves_the_uniform_test_case():
    c = NavConditioner(d_model=16, zero_init=False)
    assert set(c.widths.values()) == {16}
