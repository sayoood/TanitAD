"""v7F's fix set R1..R6 -- is each one carried by refcv8? Reachability from the ACTUAL caller, by probe.

PI, 2026-10-18: "assure that refcv8 includes the fixes". v7F (train_v6_staged.py / V6Stack) was never landed and is
superseded; refcv8 is a different line (refc_v3_train.py / RefCV3Model). The five fix classes are therefore not
ported by diff -- each is decided by EVIDENCE from refcv8's own code and its live argv
(`stack/ops/runs.d/refcv8-r101-s0.argv.json`):

  R2  start defects of nav conditioning ... three defects that CANNOT exist here (proven below) + nav REACHES
  R3  EVERY tactical label family reaches the tactical loss, traffic-light colours included
  R4  the tactical behaviours reach the operative plan (swap them -> the plan moves)
  R6  the strategic layer is OFF (`--no-strategic`) -- bypassed, never merely untrained
  R1  see `test_refcv8_plan_vmax_clamp.py`

EVERY probe carries a CONTROL that must read EXACTLY 0.0 and a DELIBERATE-REGRESSION arm that must go RED. A probe
whose control is not exactly zero is a broken probe, not a result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
for _p in (str(_HERE), str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402

LIVE_ARGV = _STACK / "ops" / "runs.d" / "refcv8-r101-s0.argv.json"


@pytest.fixture(scope="module")
def T():
    return R.trainer()


def _live_args(T):
    if not LIVE_ARGV.exists():
        pytest.skip("the live refcv8 argv file is not in this tree")
    return T.build_parser().parse_args(json.loads(LIVE_ARGV.read_text(encoding="utf-8"))["argv"])


# =========================================================================== #
# 0. the live argv carries the levers every class below hangs on               #
# =========================================================================== #
def test_the_live_refcv8_argv_names_the_levers_the_audit_relies_on(T):
    a = _live_args(T)
    assert a.no_strategic is True                                           # R6
    assert a.tac_decoder_v6 is True and a.w_tac_v6 == 1.0                   # R3: the 22-goal + 8 + 8 head, live weight
    assert a.max_speed_input_v6 is True and a.speed_ceiling_filter is True  # R1: input + argmax filter
    assert a.graft_tac8_prior is True and a.graft_behaviour_sel is True     # R4: tactical posterior -> the plan
    assert a.refcv8 is True and a.r8_n_alloc == 32 and a.r8_alloc_emit is True
    assert a.nav_from_v7 is True and a.r8_nav_from_v9 is True               # R2: nav sources
    # the DEFAULTS the audit leans on, as literals: the generic goal head and its weight are NOT what trains R3
    assert a.w_tac_goal == 0.0 and not a.tac_goal_tok_head                  # `--tac-decoder-v6` carries the goal set


def test_R6_the_flag_reaches_the_core_config_through_the_real_pin_and_the_default_is_off(T):
    """argv `--no-strategic` -> argparse (`refc_v3_train.py:12682`) -> `_pin_trainer_cfg` (`:577`) ->
    `cfg.core.no_strategic`, the ONE field `refc.py` and `refc_v3.py` read."""
    def pin(extra):
        args = T.build_parser().parse_args(["--arm", "hier", "--size", "tiny", "--out", "X"] + extra)
        cfg = T.v3.RefCV3Config(hier=True)
        T._pin_trainer_cfg(cfg, args)
        return cfg.core.no_strategic
    assert pin(["--no-strategic"]) is True
    assert pin([]) is False                                                   # the default, as a literal


# =========================================================================== #
# 1. R6 -- the strategic layer is BYPASSED, not merely untrained               #
# =========================================================================== #
def _strategic_modules(m):
    from tanitad.refs.refc_v3 import BYPASS_DECLARATIONS, _module_at
    mods = []
    for path, flag, _why in BYPASS_DECLARATIONS:
        if flag == "no_strategic":
            mod = _module_at(m, path)
            if mod is not None:
                mods.append((path, mod))
    return mods


def _plan_keys(o):
    return {k: o[k] for k in ("traj", "anchor_traj", "sel_score_v3", "g_tac", "z_tac") if k in o}


def test_R6_every_strategic_parameter_is_OFF_the_plan_path(T):
    cfg, m = R.build(T, False)
    bt = R.batch(cfg)
    mods = _strategic_modules(m)
    assert len(mods) >= 4, [p for p, _ in mods]                              # NOT VACUOUS: the layer exists here

    def perturbed(bypass: bool):
        cfg.core.no_strategic = bypass
        base = R.forward(cfg, m, bt)
        saved = {n: p.detach().clone() for _p, mod in mods for n, p in mod.named_parameters(prefix=_p)}
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for _p, mod in mods:
                for p in mod.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.5)
        try:
            pert = R.forward(cfg, m, bt)
        finally:
            with torch.no_grad():
                for _p, mod in mods:
                    for n, p in mod.named_parameters():
                        p.copy_(saved[f"{_p}.{n}"])
        return base, pert

    base, pert = perturbed(True)                                             # --no-strategic
    for k, v in _plan_keys(base).items():
        assert torch.equal(v, _plan_keys(pert)[k]), f"{k} moved under --no-strategic"      # exactly 0.0
    assert not torch.equal(base["ctx"], pert["ctx"])        # CONTROL: the perturbation is real (ctx is diagnostic)
    base0, pert0 = perturbed(False)                                          # strategic ON (the control arm)
    moved = [k for k, v in _plan_keys(base0).items() if not torch.equal(v, _plan_keys(pert0)[k])]
    assert moved, "with the strategic layer ON the same perturbation moved nothing -- the probe cannot fire"
    cfg.core.no_strategic = False


def test_R6_REGRESSION_ARM_the_bypass_removed_goes_RED(T, monkeypatch):
    """With `ctx_dec = None if no_strategic else ctx` neutered (the decoder is handed the strategic token again) the
    R6 assertion above must FAIL: emulate by reading the strategic ON arm under the bypass flag's name."""
    cfg, m = R.build(T, False)
    bt = R.batch(cfg)
    mods = _strategic_modules(m)
    cfg.core.no_strategic = False                                            # the bypass "removed"
    base = R.forward(cfg, m, bt)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _p, mod in mods:
            for p in mod.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 0.5)
    pert = R.forward(cfg, m, bt)
    assert any(not torch.equal(v, _plan_keys(pert)[k]) for k, v in _plan_keys(base).items())


# =========================================================================== #
# 2. R2 -- the three v7F start defects cannot exist here, and nav REACHES      #
# =========================================================================== #
def test_R2_defect_1_the_batch_always_carries_nav_and_a_missing_key_is_LOUD(T):
    cfg, m = R.build(T, False)
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    it = ds[0]
    assert "nav_cmd" in it and "nav_valid" in it                              # every item, from the dataset
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    del batch["nav_cmd"]
    with pytest.raises(KeyError):                                             # not a silent skip (v7F: NavTokenMissing)
        T.compute_losses_v3(m, batch, "cpu", mode="diffusion")


def test_R2_defect_2_the_nav_token_reaches_the_forward_unmodified_on_BOTH_wrappers(T):
    """v7F: the batch literal splatted the emitter output and then wrote `nav_token: b.get(...)` -- the later key
    won and every emitted token became None. Here: ONE hop, `batch['nav_cmd'] -> model(nav_cmd=...)`; the refcv8
    wrapper merges `{**kwargs, **r8_fwd}` and r8_fwd's keys are listed -- `nav_cmd` is not among them."""
    from tanitad.train import refcv8_train as RT
    cfg, m = R.build(T, True)
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    seen = {}
    real = m.forward

    def spy(*a, **k):
        seen["nav_cmd"] = k.get("nav_cmd")
        return real(*a, **k)
    m.forward = spy
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["tac_goal_y"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    batch["tac_goal_w"] = torch.zeros(2, len(v7l.TAC_GOAL_TOKENS))
    m._w_tac_v6 = 1.0
    T.compute_losses_v3(m, batch, "cpu", mode="diffusion")
    m.forward = real
    assert seen["nav_cmd"] is not None and seen["nav_cmd"].tolist() == [1, 2]
    # the keys the refcv8 wrapper adds to the forward are exactly these -- none is the nav token
    import inspect
    src = inspect.getsource(RT.r8_before_forward)
    for k in ("r8_nav", "r8_rc", "v_max_ms", "v_max_valid"):
        assert f'fwd["{k}"]' in src or f"fwd['{k}']" in src or k in src
    assert 'fwd["nav_cmd"]' not in src and "fwd['nav_cmd']" not in src


def test_R2_defect_3_every_trainable_parameter_is_in_exactly_one_optimiser_group_nav_included(T):
    cfg, m = R.build(T, True)
    args = SimpleNamespace(opt="dd", lr=1e-4, encoder_lr_mult=0.5, weight_decay=1e-4)
    opt = T.build_optimizer(m, args)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    n_total = sum(1 for p in m.parameters() if p.requires_grad)
    assert len(in_opt) == n_total and sum(len(g["params"]) for g in opt.param_groups) == n_total
    nav = [(n, p) for n, p in m.named_parameters() if "nav" in n.lower() and p.requires_grad]
    assert len(nav) >= 4, [n for n, _ in nav]                                 # NOT VACUOUS: the conditioner exists
    assert all(id(p) in in_opt for _n, p in nav)
    # CONTROL: a parameter built OUTSIDE the model is in no group
    stray = torch.nn.Parameter(torch.zeros(1))
    assert id(stray) not in in_opt


def test_R2_nav_REACHES_the_tactical_state_and_the_operative_selection(T):
    cfg, m = R.build(T, False)
    bt = R.batch(cfg)
    swapped = dict(bt, nav_cmd=torch.tensor([2, 0, 1]))
    a = R.forward(cfg, m, bt)
    same = R.forward(cfg, m, bt)
    for k in ("traj", "sel_score_v3", "anchor_logits", "z_tac"):
        assert torch.equal(a[k], same[k]), k                                  # CONTROL: identical nav -> exactly 0.0
    b = R.forward(cfg, m, swapped)
    # OPERATIVE: nav rides the measurement vector, NOT a zero-init port -> the plan's selection surface moves at init
    assert float((a["anchor_logits"] - b["anchor_logits"]).abs().max()) > 0.0
    assert float((a["sel_score_v3"] - b["sel_score_v3"]).abs().max()) > 0.0
    # TACTICAL: `nav_to_tac` is the zero-init OUTPUT projection (E13) -> z_tac is exactly unmoved at step 0 ...
    assert torch.equal(a["z_tac"], b["z_tac"])
    # ... but the port RECEIVES gradient at step 0 (so training can open it) -- the F7 question, answered for refcv8
    m.train()
    out = R.forward(cfg, m, bt, train=True)
    out["z_tac"].sum().backward()
    assert m.nav_to_tac.weight.grad is not None and float(m.nav_to_tac.weight.grad.abs().sum()) > 0.0
    m.zero_grad(set_to_none=True)
    # once opened, nav moves the TACTICAL state too
    with torch.no_grad():
        m.nav_to_tac.weight.normal_(0.0, 0.5)
    c, d = R.forward(cfg, m, bt), R.forward(cfg, m, swapped)
    assert not torch.equal(c["z_tac"], d["z_tac"])


# =========================================================================== #
# 3. R3 -- EVERY tactical label family reaches the tactical loss (end to end)  #
# =========================================================================== #
def _loss_batch(T, cfg, seed=0):
    eps = T._synth_episodes(2, cfg.core, seed=seed)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True                                                     # emit `pose_hist` (the live argv has --ego-history)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    batch["lat_v7"] = torch.tensor([0, 6], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([0, 3], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    n = len(v7l.TAC_GOAL_TOKENS)
    batch["tac_goal_y"] = torch.zeros(2, n)
    batch["tac_goal_w"] = torch.ones(2, n)
    return batch


def _tac_model(T):
    cfg, m = R.build(T, False)
    v7l = T.v7l
    m._w_tac_v6 = 1.0
    n = len(v7l.TAC_GOAL_TOKENS)
    m._tac_goal_pos_weight = torch.ones(n)
    mask = torch.ones(n)
    mask[list(v7l.TAC_GOAL_TOKENS).index("SPEED_BAND")] = 0.0                  # the one class the v9 census masks
    m._tac_goal_class_mask = mask
    return cfg, m


def _loss(T, m, batch, seed=5):
    """EVAL mode: the training forward draws sampler / dropout noise, so a same-seed A/B is not exactly repeatable
    there; the loss graph is identical in eval and the CONTROL below must read exactly 0.0 to prove it."""
    torch.manual_seed(seed)
    m.eval()
    return T.compute_losses_v3(m, batch, "cpu", mode="diffusion")["loss"]


def _with(batch, **kw):
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for k, fn in kw.items():
        fn(b)
    return b


def _delta(T, m, b0, b1):
    return abs(float(_loss(T, m, b1)) - float(_loss(T, m, b0)))


def test_R3_the_traffic_light_COLOUR_changes_the_loss_and_its_controls_read_exactly_zero(T):
    cfg, m = _tac_model(T)
    v7l = T.v7l
    tok = list(v7l.TAC_GOAL_TOKENS)
    red, green, band = tok.index("TRAFFIC_LIGHT_REACT_RED"), tok.index("TRAFFIC_LIGHT_REACT_GREEN"), tok.index("SPEED_BAND")
    b0 = _loss_batch(T, cfg)
    assert _delta(T, m, b0, _with(b0)) == 0.0                                 # CONTROL 1: same batch, same seed -> 0.0
    d_red = _delta(T, m, b0, _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), red), 1.0)))
    d_green = _delta(T, m, b0, _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), green), 1.0)))
    assert d_red > 0.0 and d_green > 0.0                                     # each colour reaches the loss
    r = _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), red), 1.0))
    g = _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), green), 1.0))
    assert _delta(T, m, r, g) > 0.0                                          # RED vs GREEN are told apart (the D-TLIGHT-1 mutation)
    # CONTROL 2: a MASKED class and a w=0 cell are invisible to the loss -- exactly 0.0
    assert _delta(T, m, b0, _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), band), 1.0))) == 0.0
    w0 = _with(b0, a=lambda b: b["tac_goal_w"].__setitem__((slice(None), red), 0.0))
    w0f = _with(w0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), red), 1.0))
    assert _delta(T, m, w0, w0f) == 0.0


def test_R3_lat_lon_action_labels_reach_the_loss_and_the_ignore_id_does_not(T):
    cfg, m = _tac_model(T)
    v7l = T.v7l
    b0 = _loss_batch(T, cfg)
    assert _delta(T, m, b0, _with(b0, a=lambda b: b.__setitem__("lat_v7", torch.tensor([6, 6])))) > 0.0
    assert _delta(T, m, b0, _with(b0, a=lambda b: b.__setitem__("lon_v7", torch.tensor([7, 7])))) > 0.0
    ig = torch.full((2,), v7l.IGNORE_ID, dtype=torch.long)
    i0 = _with(b0, a=lambda b: b.__setitem__("lat_v7", ig.clone()))
    i1 = _with(b0, a=lambda b: b.__setitem__("lat_v7", ig.clone()), c=lambda b: b.__setitem__("lon_v7", ig.clone()))
    i2 = _with(i0, a=lambda b: b.__setitem__("lon_v7", ig.clone()))
    assert _delta(T, m, i1, i2) == 0.0                                       # CONTROL: all-ignored rows move nothing


def test_R3_the_gradient_of_the_traffic_light_term_lands_on_the_tactical_goal_head_and_nowhere_strategic(T):
    cfg, m = _tac_model(T)
    tok = list(T.v7l.TAC_GOAL_TOKENS)
    red = tok.index("TRAFFIC_LIGHT_REACT_RED")
    b0 = _loss_batch(T, cfg)
    b1 = _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), red), 1.0))
    la, lb = _loss(T, m, b1), _loss(T, m, b0)
    params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(la - lb, [p for _n, p in params], allow_unused=True)
    reach = {n: (0.0 if g is None else float(g.abs().sum())) for (n, _p), g in zip(params, grads)}
    tac = sum(v for n, v in reach.items() if n.startswith("tac_decoder_v6."))
    assert tac > 0.0, "the traffic-light cell put no gradient on the tactical decoder"
    strat = [n for n, v in reach.items() if v != 0.0 and any(n.startswith(p) for p in
             ("core.strategic.", "core.route_head", "str_goal_head", "nav_to_str", "gstr_"))]
    assert strat == [], strat                                                # CONTROL: the strategic layer: exactly 0.0
    enc = sum(v for n, v in reach.items() if n.startswith("core.encoder."))
    # the label is a TARGET of a head that reads the scene: it MAY reach the trunk; what it must not do is nothing
    assert tac > 0.0 and enc >= 0.0


def test_R3_REGRESSION_ARM_the_goal_head_unwired_goes_RED(T, monkeypatch):
    """`w_tac_v6 = 0` (the tac_goal_tok_head defect: built, forward-run, weight 0 -> the term is absent): the SAME
    colour flip that moved the loss above now reads exactly 0.0 -- the assertion `d_red > 0` FAILS."""
    cfg, m = _tac_model(T)
    m._w_tac_v6 = 0.0
    red = list(T.v7l.TAC_GOAL_TOKENS).index("TRAFFIC_LIGHT_REACT_RED")
    b0 = _loss_batch(T, cfg)
    d = _delta(T, m, b0, _with(b0, a=lambda b: b["tac_goal_y"].__setitem__((slice(None), red), 1.0)))
    assert d == 0.0


# =========================================================================== #
# 4. R4 -- the tactical behaviours reach the OPERATIVE plan                    #
# =========================================================================== #
def _swap_hook(m, lat_cls=None, lon_cls=None):
    """Replace the tactical decoder's lat / lon posteriors at the SOURCE with a one-hot on the chosen class: the
    'swap the behaviours' experiment, applied to every route at once (tac8 prior, behaviour gate, r8 feeds)."""
    def hook(_mod, _inp, out):
        out = dict(out)
        if lat_cls is not None:
            z = torch.full_like(out["lat_logits"], -8.0)
            z[..., lat_cls] = 8.0
            out["lat_logits"] = z
        if lon_cls is not None:
            z = torch.full_like(out["lon_logits"], -8.0)
            z[..., lon_cls] = 8.0
            out["lon_logits"] = z
        return out
    return m.tac_decoder_v6.register_forward_hook(hook)


def _wake(m):
    """Training moves the zero-init ports off zero; emulate the smallest such move (every port, deterministic)."""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for mod in (m.core.decoder.tac8_lat_to_anchor, m.core.decoder.tac8_lon_to_anchor):
            mod.weight.normal_(0.0, 0.5, generator=g)
        m.core.decoder.r8_mod.proj[-1].weight.normal_(0.0, 0.2, generator=g)
        for p in (m.core.decoder.r8_sel.beta_lat, m.core.decoder.r8_sel.beta_lon):
            p.fill_(1.5)


def _plan(T, m, cfg, bt, lat=None, lon=None, train=False):
    h = _swap_hook(m, lat, lon)
    try:
        return R.forward(cfg, m, bt, train=train)
    finally:
        h.remove()


def _r4_model(T, monkeypatch, alloc_emit=True, **kw):
    """The live argv has `--graft-tac8-prior`: `_pin_trainer_cfg` sets `cfg.core.graft_tac8_prior`. `_refcv8_rig.build`
    does not pass it, so wrap the pin for this build only."""
    real = T._pin_trainer_cfg

    def pin(cfg, args):
        out = real(cfg, args)
        out.core.graft_tac8_prior = True
        return out
    monkeypatch.setattr(T, "_pin_trainer_cfg", pin)
    return R.build(T, True, n_alloc=8, alloc_emit=alloc_emit, **kw)


def test_R4_before_emission_the_swap_moves_the_emitted_plan_by_exactly_zero_and_every_port_gets_gradient(T, monkeypatch):
    """Steps < `--r8-alloc-emit-start` (1000 in the live argv): the extra candidates are generated and trained but
    never EMITTED, and every port is zero-init -> the warm-started plan is refcv7's, so a swap moves it by exactly
    0.0 (the same number v7F MEASURED at the tip -- but there NO PORT existed). What separates the two cases: here
    the plan-side loss lands a NON-ZERO gradient on every port, so training can open them."""
    cfg, m = _r4_model(T, monkeypatch, alloc_emit=False)
    bt = R.batch(cfg)
    a = _plan(T, m, cfg, bt, lat=6, lon=3)
    b = _plan(T, m, cfg, bt, lat=7, lon=3)
    c = _plan(T, m, cfg, bt, lat=6, lon=7)
    assert torch.equal(a["traj"], b["traj"]) and torch.equal(a["traj"], c["traj"])
    out = _plan(T, m, cfg, bt, lat=6, lon=3, train=True)
    loss = torch.nn.functional.cross_entropy(out["sel_score_v3"], torch.zeros(out["sel_score_v3"].shape[0], dtype=torch.long))
    loss = loss + out["traj"].abs().mean()
    loss.backward()
    dec = m.core.decoder
    for name, p in (("tac8_lat", dec.tac8_lat_to_anchor.weight), ("tac8_lon", dec.tac8_lon_to_anchor.weight),
                    ("beta_lat", dec.r8_sel.beta_lat), ("beta_lon", dec.r8_sel.beta_lon),
                    ("mod_last", dec.r8_mod.proj[-1].weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name


def test_R4_after_emission_starts_the_swap_moves_the_EMITTED_plan_with_every_port_still_zero_init(T, monkeypatch):
    """From the emission step on, the tactical posterior decides WHICH hypotheses the allocated candidates carry
    (parameter-free, proportional to p(h)) -- so a swap moves the fan and the emitted plan BEFORE any port opens."""
    cfg, m = _r4_model(T, monkeypatch, alloc_emit=True)
    m.eval()
    bt = R.batch(cfg)
    assert float(m.core.decoder.r8_sel.beta_lat) == 0.0 and float(m.core.decoder.tac8_lat_to_anchor.weight.abs().sum()) == 0.0
    base = _plan(T, m, cfg, bt, lat=6, lon=3)
    again = _plan(T, m, cfg, bt, lat=6, lon=3)
    assert torch.equal(base["traj"], again["traj"])                           # CONTROL: the same swap twice -> exactly 0.0
    lat_swap = _plan(T, m, cfg, bt, lat=7, lon=3)                             # TURN_L -> TURN_R
    lon_swap = _plan(T, m, cfg, bt, lat=6, lon=7)                             # BRAKE_TO -> ACCELERATE
    both = _plan(T, m, cfg, bt, lat=7, lon=7)
    # the CANDIDATE SET moves on each axis (the swap reaches the planner's fan) ...
    assert float((base["anchor_traj"] - lat_swap["anchor_traj"]).abs().max()) > 0.0
    assert float((base["anchor_traj"] - lon_swap["anchor_traj"]).abs().max()) > 0.0
    # ... and the EMITTED plan moves once an allocated candidate wins (zero-init selection => base scores decide
    # most windows; MEASURED on this rig: lat swap 0.31 m, both axes 0.94 m, lon alone 0.0 -- stated, not hidden)
    assert float((base["traj"] - both["traj"]).abs().max()) > 0.0


def test_R4_once_the_ports_move_a_swap_of_lat_or_lon_moves_the_EMITTED_plan_and_the_control_reads_zero(T, monkeypatch):
    cfg, m = _r4_model(T, monkeypatch)
    if m.core.decoder.tac8_lat_to_anchor is None:
        pytest.skip("this rig build has no tac8 port")
    _wake(m)
    m.eval()
    bt = R.batch(cfg)
    base = _plan(T, m, cfg, bt, lat=6, lon=3)                                 # TURN_L + BRAKE_TO
    again = _plan(T, m, cfg, bt, lat=6, lon=3)
    assert torch.equal(base["traj"], again["traj"])                           # CONTROL: the same swap twice -> exactly 0.0
    lat_swap = _plan(T, m, cfg, bt, lat=7, lon=3)                             # TURN_R + BRAKE_TO
    lon_swap = _plan(T, m, cfg, bt, lat=6, lon=7)                             # TURN_L + ACCELERATE
    d_lat = float((base["sel_score_v3"] - lat_swap["sel_score_v3"]).abs().max())
    d_lon = float((base["sel_score_v3"] - lon_swap["sel_score_v3"]).abs().max())
    assert d_lat > 0.0 and d_lon > 0.0                                        # the SELECTION surface moved on both axes
    moved = (not torch.equal(base["traj"], lat_swap["traj"])) or (not torch.equal(base["traj"], lon_swap["traj"]))
    moved = moved or (not torch.equal(base["anchor_traj"], lat_swap["anchor_traj"])) \
        or (not torch.equal(base["anchor_traj"], lon_swap["anchor_traj"]))
    assert moved, "the swap moved the scores but neither the fan nor the emitted plan"


def test_R4_REGRESSION_ARM_a_severed_port_goes_RED(T, monkeypatch):
    cfg, m = _r4_model(T, monkeypatch)
    if m.core.decoder.tac8_lat_to_anchor is None:
        pytest.skip("this rig build has no tac8 port")
    _wake(m)
    m.eval()
    bt = R.batch(cfg)
    # sever the whole tactical -> planner hand-off: the planner sees a CONSTANT posterior whatever the decoder says
    monkeypatch.setattr(C, "hypothesis_posteriors",
                        lambda lat, lon: (torch.zeros(lat.shape[0], C.N_LAT3), torch.zeros(lon.shape[0], C.N_LON6)))
    monkeypatch.setattr(v6tac, "planner_feeds", lambda out, valid_threshold=0.5: {
        "lat_logprob": torch.zeros_like(out["lat_logits"]), "lon_logprob": torch.zeros_like(out["lon_logits"]),
        "valid_behaviour": torch.ones(out["lat_logits"].shape[0], 22),
        "valid_mask": torch.ones(out["lat_logits"].shape[0], 22, dtype=torch.bool)})
    a = _plan(T, m, cfg, bt, lat=6, lon=3)
    b = _plan(T, m, cfg, bt, lat=7, lon=7)
    assert torch.equal(a["sel_score_v3"], b["sel_score_v3"]) and torch.equal(a["traj"], b["traj"])   # 0.0: R4 would FAIL
