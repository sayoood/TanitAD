

# =========================================================================== #
# SPEC_RL_A1 (2026-10-11): KL anchor, kill-switch, absolute warmup            #
# Expectations are LITERALS (hand-derived, SPEC_RL_A1 sec. 3 / sec. 5). Every   #
# mutation arm reintroduces a plausible port error and must MISS the literal.  #
# =========================================================================== #
#: table abar = [0.81, 0.36], t = 1 -> t_prev = 0, eta 1, clip_sample OFF, Delta x0 = +0.1 on [1,1,4,2]:
#: c = d mu / d x0 = 0.9 - 0.75 * sqrt(0.19 - 95/576) = 25/32,  sigma^2 = 95/576 (> floor 0.1^2),
#: KL = c^2 * sum(Delta^2) / (2 sigma^2) = (625/1024) * 0.08 * 576/190,  dKL/dx0_theta = c^2 * 0.1 / sigma^2.
KL_LIT = 0.14802631578947367
KL_GRAD_LIT = 0.3700657894736842


def _kl_case():
    table = torch.tensor([0.81, 0.36], dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    x0_ref = torch.rand(1, 1, 4, 2, generator=g, dtype=torch.float64) * 1.6 - 0.8
    x_t = torch.rand(1, 1, 4, 2, generator=g, dtype=torch.float64) * 1.6 - 0.8
    x0_th = (x0_ref + 0.1).requires_grad_(True)
    return table, x0_ref, x_t, x0_th


def _kl_consts():
    return D.DDV2Constants(clip_sample=False)


def test_kl_analytic_literal_and_gradient():
    table, x0_ref, x_t, x0_th = _kl_case()
    kl = R.step_kl_to_reference(x0_th, x0_ref, x_t, 1, table, eta=1.0, consts=_kl_consts())
    assert tuple(kl.shape) == (1, 1)
    assert abs(float(kl) - KL_LIT) < 1e-12
    kl.sum().backward()
    assert torch.allclose(x0_th.grad, torch.full_like(x0_th, KL_GRAD_LIT), rtol=0, atol=1e-12)


def test_kl_is_exactly_zero_at_identity_and_draws_no_random_number():
    table, x0_ref, x_t, _ = _kl_case()
    state = torch.get_rng_state()
    kl = R.step_kl_to_reference(x0_ref.clone(), x0_ref, x_t, 1, table, eta=1.0, consts=_kl_consts())
    assert torch.equal(kl, torch.zeros(1, 1, dtype=torch.float64))
    assert torch.equal(torch.get_rng_state(), state)


def test_kl_is_nonnegative_on_the_release_table_at_every_label():
    table = D.diffusers_alphas_cumprod()
    g = torch.Generator().manual_seed(17)
    for t in (18, 16, 10, 2, 0):
        x0_ref = torch.randn(3, 5, 4, 2, generator=g)
        x0_th = x0_ref + 0.3 * torch.randn(3, 5, 4, 2, generator=g)
        x_t = torch.randn(3, 5, 4, 2, generator=g)
        kl = R.step_kl_to_reference(x0_th, x0_ref, x_t, t, table)
        assert tuple(kl.shape) == (3, 5)
        assert float(kl.min()) >= -1e-5
        assert float(kl.sum()) > 0.0


def test_kl_gradient_reaches_theta_only():
    table, x0_ref, x_t, x0_th = _kl_case()
    x0_ref = x0_ref.clone().requires_grad_(True)
    R.step_kl_to_reference(x0_th, x0_ref, x_t, 1, table, eta=1.0, consts=_kl_consts()).sum().backward()
    assert x0_ref.grad is None
    assert x0_th.grad is not None


def _fake_ddim(mean_rule):
    """A ddim_logprob_step with a MUTATED mean: the release's sigma and floor, a wrong mu."""
    def fake(x0, sample, t, t_prev, table, *, prev_sample, likelihood_std_floor, **_):
        a_p = float(table[t_prev]) if t_prev >= 0 else 1.0
        a_t = float(table[t])
        var = (1 - a_p) / (1 - a_t) * (1 - a_t / a_p)
        s = max(math.sqrt(var), likelihood_std_floor)
        mean = mean_rule(x0, a_p)
        lp = (-(prev_sample - mean) ** 2 / (2 * s * s) - math.log(s)
              - 0.5 * math.log(2 * math.pi)).sum(dim=(-2, -1))
        return prev_sample, lp, mean
    return fake


@pytest.mark.parametrize("mutation, expect", [
    ("floor_1", 0.0244140625),               # likelihood floor 1.0 instead of 0.1
    ("t_prev_eq_t", 0.0),                    # the step evaluated t -> t (mean = x_t, no x0 dependence)
    ("sign_flip", -0.14802631578947367),     # lp_theta - lp_ref
    ("no_eps_direction", 0.19644631578947368),  # mu = sqrt(abar_prev) x0 only
    ("x0_space", 0.24252631578947368),       # mu = x0 (KL in x0 space, same sigma)
])
def test_kl_mutations_miss_the_literal(monkeypatch, mutation, expect):
    real = D.ddim_logprob_step
    if mutation == "floor_1":
        def fake(*a, **k):
            return real(*a, **{**k, "likelihood_std_floor": 1.0})
    elif mutation == "t_prev_eq_t":
        def fake(x0, sample, t, t_prev, table, **k):
            return real(x0, sample, t, t, table, **k)
    elif mutation == "sign_flip":
        def fake(*a, **k):
            prev, lp, mean = real(*a, **k)
            return prev, -lp, mean
    elif mutation == "no_eps_direction":
        fake = _fake_ddim(lambda x0, a_p: math.sqrt(a_p) * x0)
    else:
        fake = _fake_ddim(lambda x0, a_p: x0)
    monkeypatch.setattr(D, "ddim_logprob_step", fake)
    table, x0_ref, x_t, x0_th = _kl_case()
    got = float(R.step_kl_to_reference(x0_th, x0_ref, x_t, 1, table, eta=1.0, consts=_kl_consts()))
    assert abs(got - expect) < 1e-9
    assert abs(got - KL_LIT) > 1e-3


def test_swapped_params_restores_bitwise_also_on_an_exception():
    dec = _dec()
    named = R.trainable_named(dec)
    before = {n: p.detach().clone() for n, p in named}
    ref = {n: p.detach() + 1.0 for n, p in named}
    with R.swapped_params(named, ref):
        assert all(torch.equal(p, ref[n]) for n, p in named)
    assert all(torch.equal(p, before[n]) for n, p in named)
    with pytest.raises(RuntimeError, match="boom"):
        with R.swapped_params(named, ref):
            raise RuntimeError("boom")
    assert all(torch.equal(p, before[n]) for n, p in named)
    assert all(p.grad is None for _, p in named)


def test_decoder_kl_zero_at_identity_then_positive_with_grad_after_a_perturbation():
    dec = _dec()
    inp, _ = _captured(dec)
    named = R.trainable_named(dec)
    base = {n: p.detach().clone() for n, p in named}
    table = D.diffusers_alphas_cumprod()
    x = R.anchor_state(dec, 2)
    with R.swapped_params(named, base), torch.no_grad():
        x0_ref = R.make_x0_fn(dec, inp)(x, 8).detach()
    kl0 = R.step_kl_to_reference(R.make_x0_fn(dec, inp)(x, 8), x0_ref, x, 8, table, consts=_kl_consts())
    assert torch.equal(kl0, torch.zeros_like(kl0))
    kl0.mean().backward()
    grads = [p.grad for _, p in named if p.grad is not None]
    assert grads and all(int(torch.count_nonzero(g_)) == 0 for g_ in grads)
    g = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for _, p in named:
            p.add_(0.05 * torch.randn(p.shape, generator=g))
            p.grad = None
    x0_th = R.make_x0_fn(dec, inp)(x, 8)
    assert not torch.equal(x0_th.detach(), x0_ref)
    kl = R.step_kl_to_reference(x0_th, x0_ref, x, 8, table, consts=_kl_consts())
    assert float(kl.min()) >= 0.0 and float(kl.sum()) > 0.0
    kl.mean().backward()
    assert any(p.grad is not None and int(torch.count_nonzero(p.grad)) > 0 for _, p in named)
    with R.swapped_params(named, base), torch.no_grad():
        assert torch.equal(R.make_x0_fn(dec, inp)(x, 8), x0_ref)


def test_reference_swap_after_a_theta_forward_invalidates_its_graph():
    """Why rl_step computes every reference x0 BEFORE the first theta forward of a micro-batch."""
    dec = _dec()
    inp, _ = _captured(dec)
    named = R.trainable_named(dec)
    base = {n: p.detach().clone() for n, p in named}
    x0_th = R.make_x0_fn(dec, inp)(R.anchor_state(dec, 2), 8)
    with R.swapped_params(named, base):
        pass
    with pytest.raises(RuntimeError, match="inplace"):
        x0_th.sum().backward()


def _hrow(gn=1.0, oob=0.2, pdn=0.5):
    return {"grad_norm": gn, "x0_out_of_box_frac_final_a_lon": oob, "param_delta_norm": pdn}


def test_halt_keys_are_the_trainer_stats():
    assert R.HALT_KEYS == ("grad_norm", "x0_out_of_box_frac_final_a_lon", "param_delta_norm")


def test_halt_one_spike_does_not_trip_the_median_rules():
    hist = [_hrow() for _ in range(19)] + [_hrow(gn=500.0, oob=1.0)]
    assert R.halt_check(hist, gn_median=5.0, oob_median=0.4, pdn=3.0, window=20) is None


def test_halt_sustained_rise_trips_k1_and_k2_with_literal_reasons():
    hist = [_hrow() for _ in range(9)] + [_hrow(gn=6.0, oob=0.5) for _ in range(11)]
    assert R.halt_check(hist, gn_median=5.0, oob_median=0.4, window=20) == (
        "K1 median20 grad_norm 6.0000 > 5; K2 median20 x0_out_of_box_frac_final_a_lon 0.5000 > 0.4")
    half = [_hrow() for _ in range(10)] + [_hrow(gn=6.0, oob=0.5) for _ in range(10)]
    assert R.halt_check(half, gn_median=5.0, oob_median=0.4, window=20) is None


def test_halt_k3_is_instantaneous_and_k1_k2_wait_for_the_window():
    assert R.halt_check([_hrow(pdn=3.5)], pdn=3.0) == "K3 param_delta_norm 3.5000 > 3"
    early = [_hrow(gn=100.0, oob=1.0) for _ in range(19)]
    assert R.halt_check(early, gn_median=5.0, oob_median=0.4, pdn=3.0, window=20) is None


def test_halt_zero_disables_and_bad_window_raises():
    hot = [_hrow(gn=100.0, oob=1.0, pdn=100.0) for _ in range(25)]
    assert R.halt_check(hot) is None
    assert R.halt_check([]) is None
    with pytest.raises(R.Ddv2V7Error):
        R.halt_check(hot, gn_median=5.0, window=0)


def _trainer_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "ddv2_rl_refcv7.py"
    spec = importlib.util.spec_from_file_location("ddv2_rl_refcv7_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lr_at_literals_fraction_and_absolute_warmup():
    lr_at = _trainer_module().lr_at
    assert abs(lr_at(0, 2000, 2e-4) - 1e-6) < 1e-15
    assert abs(lr_at(158, 2000, 2e-4) - 1.59e-4) < 1e-15
    assert abs(lr_at(0, 2000, 5e-5, warmup_steps=100) - 5e-7) < 1e-15
    assert abs(lr_at(99, 2000, 5e-5, warmup_steps=100) - 5e-5) < 1e-15
    assert abs(lr_at(100, 2000, 5e-5, warmup_steps=100) - 5e-5) < 1e-15
    assert abs(lr_at(1999, 2000, 5e-5, warmup_steps=100) - 1.0000338e-6) < 1e-12


def test_train_parser_carries_the_a1_flags(capsys):
    mod = _trainer_module()
    with pytest.raises(SystemExit) as e:
        mod.main(["train", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--warmup-steps", "--kl-beta", "--halt-gn-median", "--halt-oob-median", "--halt-pdn",
                 "--halt-window"):
        assert flag in out
