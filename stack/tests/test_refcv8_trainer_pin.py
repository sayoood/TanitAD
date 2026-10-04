"""refcv8 WP-B -- the trainer's argv pin ``_pin_refcv8`` and the refcv8 rows of the EFFECTIVE-WEIGHT audit.

The binding MM rulings it carries (2026-10-04): route-checkpoint dropout >= 0.3; nav-argument dropout in (0, 1) (the
token is kept, the 'unknown distance/time' rows are trained); and M18 -- a flag that would be stamped and read by
nothing is refused (the dead-flag family); the v9 wiring refuses an eval release without a train one, a
missing eval release, RC noise below the E2-certified sigmas, and an agent join without the ego-box mask.

The five ``--w-r8-*`` weights sit in ``REFC_WEIGHT_GATES`` (the exhaustiveness contract,
``test_v6_effective_weights.py``) and in ``launch_gate.LIVE_WEIGHT_RULES`` (G-LIVE's), and all five DEFAULT TO 0.0:
``effective_weights.classify`` refuses ``effective > 0 with the gate shut`` WITHOUT consulting explicitness, so a
non-zero default would refuse every run that does not pass ``--refcv8``. A refcv8 launch therefore states its weights,
and ``_pin_refcv8`` refuses the converse (seams built, weight 0).
"""
from __future__ import annotations

import types

import pytest

from _refcv8_rig import trainer
from tanitad.refs import refcv8_conditioning as r8c

T = trainer()
BASE = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X"]
ON = BASE + ["--refcv8", "--w-r8-cons", "0.05"]
R8_WEIGHTS = ("w_r8_cons", "w_r8_alloc_l1", "w_r8_sat", "w_r8_listwise", "w_r8_subscore")


def _pin(argv):
    args = T.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, args)
    return cfg.refcv8


def test_GREEN_the_default_pin_enables_the_seams_with_the_binding_defaults():
    r = _pin(ON)
    assert r.enable is True and r.n_alloc == 0 and r.base_constraints is False and r.modulate_base is True
    assert r.w_cons == 0.05
    args = T.build_parser().parse_args(ON)
    assert float(args.r8_rc_dropout) >= 0.3                                 # MM binding: RC dropout >= 0.3
    assert 0.0 < float(args.r8_nav_args_dropout) < 1.0                      # MM binding: nav-args dropout on
    assert _pin(BASE).enable is False                                       # no --refcv8: nothing is pinned


def test_GREEN_the_v9_wiring_pins_through_with_its_binding_defaults():
    args = T.build_parser().parse_args(ON)
    assert (args.r8_rc_variant, args.r8_v9_lat_variant) == ("A50", "a")              # MM: RC-A50; D-WPA-1 variant a
    assert (args.r8_rc_noise_along_m, args.r8_rc_noise_lat_m) == (2.0, 0.75)          # WP-A E2' certified sigmas
    assert (args.r8_v9_md5, args.r8_v9_eval_md5) == ("f63ece410b725febb8a5242cf2b01d3c",
                                                     "6b5c7f207cffc3b7eb3cd527fd433599")
    assert args.r8_no_rc is False
    _pin(ON + ["--r8-v9-labels", "v9.npz", "--r8-v9-labels-eval", "v9e.npz", "--eval-cache", "E",
               "--r8-nav-from-v9"])
    _pin(ON + ["--r8-v9-labels", "v9.npz", "--r8-no-rc"])                            # the RC switch-off
    _pin(ON + ["--agent-join", "J", "--join-defect-masks", "M"])                       # the mask is named -> admitted


T2 = ["--r8-n-alloc", "32", "--r8-alloc-emit", "--w-r8-alloc-l1", "1.0", "--w-r8-sat", "0.1", "--w-r8-listwise", "1.0"]


def test_GREEN_the_T2_recipe_pins_through():
    r = _pin(ON + T2)
    assert (r.n_alloc, r.alloc_emit, r.w_alloc_l1, r.w_sat, r.w_listwise) == (32, True, 1.0, 0.1, 1.0)


@pytest.mark.parametrize("argv,needle", [
    (BASE + ["--r8-n-alloc", "8"], "without --refcv8"),
    (BASE + ["--w-r8-listwise", "1.0"], "without --refcv8"),
    (BASE + ["--w-r8-cons", "0.05"], "without --refcv8"),
    (BASE + ["--w-r8-alloc-l1", "1.0"], "without --refcv8"),
    (BASE + ["--r8-v9-labels", "v9.jsonl.gz"], "without --refcv8"),
    (BASE + ["--r8-nav-from-v9"], "without --refcv8"),
    (["--arm", "hier", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05"], "--tac-decoder-v6"),
    (["--arm", "hier", "--tac-decoder-v6", "--out", "X", "--refcv8", "--w-r8-cons", "0.05"], "pass --sampler ddim"),
    (ON + ["--r8-rc-dropout", "0.2"], "< 0.3"),
    (ON + ["--r8-nav-args-dropout", "0"], "(0, 1)"),
    (ON + ["--r8-nav-args-dropout", "1"], "(0, 1)"),
    (BASE + ["--r8-no-rc"], "without --refcv8"),
    (ON + ["--r8-v9-labels-eval", "v9e.npz"], "without --r8-v9-labels"),
    (ON + ["--r8-nav-from-v9"], "without --r8-v9-labels"),
    (ON + ["--r8-v9-labels", "v9.npz", "--eval-cache", "E"], "no --r8-v9-labels-eval"),
    (ON + ["--r8-rc-noise-along-m", "1.0"], "E2'-CERTIFIED"),
    (ON + ["--r8-rc-noise-lat-m", "0.5"], "E2'-CERTIFIED"),
    (ON + ["--agent-join", "J"], "--join-defect-masks"),
    (ON + ["--w-r8-sat", "0.1"], "no allocated candidates"),
    (BASE + ["--refcv8"], "never supervised"),                          # the default 0.0 is not a refcv8 weight
    (ON + ["--w-r8-cons", "0"], "never supervised"),
    (ON + ["--r8-n-alloc", "8"], "never matched to GT"),                # allocation with the default alloc_l1 0.0
])
def test_every_dead_or_unsafe_refcv8_argv_refuses(argv, needle):
    with pytest.raises(SystemExit) as e:
        _pin(argv)
    assert needle in str(e.value), str(e.value)[:300]


def _calls_pin_refcv8(src: str) -> bool:
    a = src.index("\ndef _pin_trainer_cfg(")
    body = src[a:src.index("\ndef ", a + 1)]
    return "    _pin_refcv8(cfg, args)\n" in body


def test_the_refusals_are_reached_from_the_full_trainer_pin():
    """``_pin_refcv8`` is called from ``_pin_trainer_cfg`` -- the path train() AND the eval loader both take (a SOURCE
    pin: a minimal argv cannot pass the refcv6/7 pins that run in the same function). Deliberate regression: drop
    the call -> RED."""
    import inspect
    src = inspect.getsource(T).replace("\r\n", "\n")
    assert _calls_pin_refcv8(src)
    assert not _calls_pin_refcv8(src.replace("    _pin_refcv8(cfg, args)\n", "    pass\n", 1))


# =========================================================================== #
# the effective-weight audit (REFC_WEIGHT_GATES) and G-LIVE (LIVE_WEIGHT_RULES)  #
# =========================================================================== #
def _ew_args(argv):
    ap = T.build_parser()
    a = ap.parse_args(argv)
    a._ew_parser, a._ew_argv = ap, list(argv)
    return a


def _ew_refusal(argv):
    try:
        T.check_effective_weights(_ew_args(argv))
        return None
    except SystemExit as e:
        return str(e)


def test_the_five_weights_are_gated_and_default_to_zero():
    """The zero-default invariant, as LITERALS: a non-zero default here refuses every run without --refcv8."""
    ap = T.build_parser()
    defaults = {a.dest: a.default for a in ap._actions}
    assert [defaults[d] for d in R8_WEIGHTS] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert set(R8_WEIGHTS) <= set(T.REFC_WEIGHT_GATES)


def test_GREEN_a_run_without_refcv8_passes_the_audit_with_the_five_rows_OFF_BY_DEFAULT():
    from tanitad import effective_weights as ew
    argv = ["--arm", "hier", "--out", "X"]
    assert _ew_refusal(argv) is None
    rows, _src = T.effective_weight_rows_v3(_ew_args(argv))
    st = {r.flag: r.status for r in rows}
    assert all(st[f"--{d.replace('_', '-')}"] == ew.OFF_BY_DEFAULT for d in R8_WEIGHTS)


def test_GREEN_the_T2_recipe_TRAINS_every_term_it_names():
    from tanitad import effective_weights as ew
    argv = ON + T2 + ["--w-r8-subscore", "0.5"]
    assert _ew_refusal(argv) is None
    rows, _src = T.effective_weight_rows_v3(_ew_args(argv))
    st = {r.flag: r.status for r in rows}
    assert all(st[f"--{d.replace('_', '-')}"] == ew.TRAINS for d in R8_WEIGHTS)


@pytest.mark.parametrize("argv,flag", [
    (BASE + ["--w-r8-cons", "0.05"], "--w-r8-cons"),
    (BASE + ["--w-r8-listwise", "1.0"], "--w-r8-listwise"),
    (BASE + ["--w-r8-subscore", "0.5"], "--w-r8-subscore"),
    (ON + ["--w-r8-alloc-l1", "1.0"], "--w-r8-alloc-l1"),            # --refcv8 but no allocation
    (ON + ["--w-r8-sat", "0.1"], "--w-r8-sat"),
])
def test_RED_a_weight_whose_gate_is_shut_is_refused_by_the_audit(argv, flag):
    got = _ew_refusal(argv)
    assert got is not None and flag in got, got


def test_G_LIVE_has_a_rule_for_every_registered_weight_and_reads_the_r8_keys():
    """The launch gate's G-LIVE table is EXHAUSTIVE over the trainer's registry (a missing rule is a launch-time
    problem for EVERY arm) -- checked here with the REAL trainer, not a fake registry."""
    import launch_gate as LG
    _terms, problems = LG.declared_terms(T, _ew_args(BASE))
    assert [p for p in problems if "G-LIVE has no rule" in p] == []
    terms, _p = LG.declared_terms(T, _ew_args(ON + T2 + ["--w-r8-subscore", "0.5"]))
    ids = {t["id"]: t["keys"] for t in terms}
    assert {d: ids.get(d) for d in R8_WEIGHTS} == {
        "w_r8_cons": ["r8_cons"], "w_r8_alloc_l1": ["r8_alloc_l1"], "w_r8_sat": ["r8_sat"],
        "w_r8_listwise": ["r8_listwise"], "w_r8_subscore": ["r8_subscore"]}
    terms0, _p = LG.declared_terms(T, _ew_args(BASE))
    assert not {t["id"] for t in terms0} & set(R8_WEIGHTS)                 # a refcv7 arm declares none of them
