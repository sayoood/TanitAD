"""PREREG_V8F_ACTSENS: the IDM-weight lever ``--wm-idm-gamma G1 G2 G3`` (REACHABILITY_ACTSENS section 1).

Controls (each has a deliberate-regression mutation run by ``mutate_actsens.py`` on the working tree, restoring it afterwards; the named test must go RED):
1. Dose reach: the level-1 IDM-head gradient scales with the level-1 gamma (10.0x at 1000 vs 100), levels 2 / 3 stay bit-equal, the predictor gradient is bit-equal, ``--wm a`` reads EXACTLY 0.0 (``test_dose_*``).
2. C3 identity: the level/L1 difference between gamma 1000 and 100 equals 900 x idm/L1, the unweighted terms are equal (``test_c3_*``).
3. C1 default: a config with the field absent builds a state_dict bit-equal to ``wm_gamma_levels=(100, 100, 100)``, and the flag-absent argv declares () (``test_default_*``).
4. G-DVB: the read-back reads the BUILT LevelConfigs, never ``WMAux.gamma`` (``test_gate_*``).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
DOSE = (1000.0, 100.0, 100.0)
BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]


def cfg_of(**kw):
    return M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm=kw.pop("wm", "b"), hmax=28, kin_mode="yaw", **kw)


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return M.WMAux(cfg)


def batch(cfg):
    g = torch.Generator().manual_seed(7)
    return torch.randn(2, 28, cfg.wm_dim, generator=g), torch.randn(2, 27, 2, generator=g)


def run(wm, cfg):
    z, a = batch(cfg)
    wm.zero_grad(set_to_none=True)
    out = wm.stack.loss(z, a, generator=torch.Generator().manual_seed(11))
    out["total"].backward()
    gi = [float(sum(p.grad.abs().sum() for p in h.parameters() if p.grad is not None)) for h in wm.stack.idm_heads]
    gi_raw = [torch.cat([p.grad.flatten() for p in h.parameters() if p.grad is not None]).clone() for h in wm.stack.idm_heads]
    gp = [torch.cat([p.grad.flatten() for p in pr.parameters() if p.grad is not None]).clone() for pr in wm.stack.predictors]
    return gi, gi_raw, gp, {k: float(v.detach()) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ control 1: dose reach (gradient)
def test_dose_level1_idm_gradient_is_ten_times_and_other_levels_are_bit_equal():
    c0, c1 = cfg_of(), cfg_of(wm_gamma_levels=DOSE)
    gi0, raw0, gp0, _ = run(build(c0), c0)
    gi1, raw1, gp1, _ = run(build(c1), c1)
    assert gi0[0] > 0.0 and abs(gi1[0] / gi0[0] - 10.0) < 1e-4, (gi0, gi1)
    assert torch.equal(raw1[1], raw0[1]) and torch.equal(raw1[2], raw0[2])              # levels 2 / 3: the same weight, the same gradient, bit for bit
    assert all(torch.equal(x, y) for x, y in zip(gp0, gp1))                                # the predictor's gradient never sees the IDM weight directly


@pytest.mark.parametrize("lv", [0, 1, 2])
def test_dose_each_level_scales_only_its_own_head(lv):
    g = [100.0, 100.0, 100.0]
    g[lv] = 300.0
    c0, c1 = cfg_of(), cfg_of(wm_gamma_levels=tuple(g))
    gi0, raw0, _, _ = run(build(c0), c0)
    gi1, raw1, _, _ = run(build(c1), c1)
    for i in range(3):
        if i == lv:
            assert abs(gi1[i] / gi0[i] - 3.0) < 1e-4
        else:
            assert torch.equal(raw1[i], raw0[i]), f"level {i + 1} moved when only level {lv + 1} was dosed"


def test_dose_wm_a_reads_exactly_zero_control():
    c = cfg_of(wm="a")
    gi, _, _, out = run(build(c), c)
    assert gi == [0.0, 0.0, 0.0], gi


# ------------------------------------------------------------------------------------------------ control 2: C3 identity
def test_c3_level_difference_is_900_times_idm_and_unweighted_terms_are_equal():
    c0, c1 = cfg_of(), cfg_of(wm_gamma_levels=DOSE)
    _, _, _, o0 = run(build(c0), c0)
    _, _, _, o1 = run(build(c1), c1)
    for k in ("pred/L1", "sigreg/L1", "idm/L1", "pred/L2", "sigreg/L2", "idm/L2", "pred/L3", "sigreg/L3", "idm/L3"):
        assert o0[k] == o1[k], k
    d = o1["level/L1"] - o0["level/L1"]
    assert abs(d - 900.0 * o0["idm/L1"]) <= 1e-5 * abs(900.0 * o0["idm/L1"]), (d, 900.0 * o0["idm/L1"])
    assert o1["level/L2"] == o0["level/L2"] and o1["level/L3"] == o0["level/L3"]


# ------------------------------------------------------------------------------------------------ control 3: C1 default
def test_default_absent_field_equals_the_explicit_record_and_the_flag_absent_argv_declares_nothing():
    wa, wb = build(cfg_of()), build(cfg_of(wm_gamma_levels=(100.0, 100.0, 100.0)))
    sa, sb = wa.state_dict(), wb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert [r["gamma"] for r in wa.stack.built_levers().values()] == [100.0] * 3
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.wm_idm_gamma is None and TR.declared_wm_idm_gamma(a) == () and M.V8FMiniConfig.__dataclass_fields__["wm_gamma_levels"].default == ()
    assert TR.FLAG_CHECKS["wm_idm_gamma"] == "lever:wm_idm_gamma"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)


def test_default_the_dose_changes_no_parameter_only_the_weight():
    wa, wb = build(cfg_of()), build(cfg_of(wm_gamma_levels=DOSE))
    sa, sb = wa.state_dict(), wb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)               # same seed, same init: the lever moves the loss weight only
    assert [r["gamma"] for r in wb.stack.built_levers().values()] == list(DOSE)


def test_config_refuses_a_malformed_or_declared_and_unbuilt_dose():
    for bad in ((1000.0, 100.0), (1000.0, 0.0, 100.0), (1000.0, -1.0, 100.0), (float("nan"), 1.0, 1.0)):
        with pytest.raises(ValueError):
            cfg_of(wm_gamma_levels=bad)
    with pytest.raises(ValueError):
        cfg_of(wm="a", wm_gamma_levels=DOSE)                                                  # M6: named with --wm a
    with pytest.raises(ValueError):
        cfg_of(wm="off", wm_gamma_levels=DOSE)


# ------------------------------------------------------------------------------------------------ control 4: the gate reads the BUILT levels
@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("actsens")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    return {"root": root, "ck": root / "m2b" / "ckpt_last.pt"}


WMB = ["--hmax", "28", "--wm", "b"]


def gate(init, name, extra=(), args=WMB):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *BASE, *args, *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


def test_gate_passes_the_dose_and_reads_the_built_levels_back(init):
    assert gate(init, "g_ok", ["--wm-idm-gamma", "1000", "100", "100"]) == 0
    rb = sc_of(init, "g_ok")["levers"]["wm_idm_gamma_readback"]
    assert rb["built"] == [1000.0, 100.0, 100.0] and rb["declared"] == [1000.0, 100.0, 100.0] and rb["wm_aux_gamma"] == 100.0


def test_gate_flag_absent_keeps_the_key_set_and_built_is_the_record(init):
    assert gate(init, "g_off") == 0
    assert "wm_idm_gamma_readback" not in sc_of(init, "g_off")["levers"]
    a = TR.build_parser().parse_args(["--out", "x", *BASE[1:], *WMB])
    cfg = TR.make_config(a)
    assert cfg.wm_gamma_levels == ()


def test_gate_refuses_a_builder_that_ignores_the_config_field_while_the_old_check_stays_green(init, monkeypatch):
    """M2: the config says 1000, the LevelConfigs the loss multiplies by keep 100, WMAux.gamma stays 100 (the old trainer check compares exactly that)."""
    orig = M.WMAux.__init__

    def ignoring(self, cfg):
        orig(self, dataclasses.replace(cfg, wm_gamma_levels=()))
    monkeypatch.setattr(M.WMAux, "__init__", ignoring)
    rc = gate(init, "r_m2", ["--wm-idm-gamma", "1000", "100", "100"])
    assert isinstance(rc, str) and "lever wm_idm_gamma" in rc and "BUILT LevelConfigs" in rc, rc
    assert "wm b declared but built gamma" not in rc                                       # the older WMAux.gamma check did NOT see it


def test_gate_refuses_when_the_declared_dose_is_not_in_the_built_config(init, monkeypatch):
    """M1: the make_config line deleted."""
    orig = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a: dataclasses.replace(orig(a), wm_gamma_levels=()))
    rc = gate(init, "r_m1", ["--wm-idm-gamma", "1000", "100", "100"])
    assert isinstance(rc, str) and "lever wm_idm_gamma" in rc, rc


def test_gate_refuses_the_flag_with_wm_a_or_off_and_a_non_positive_gamma(init):
    for args, extra in ((["--hmax", "28", "--wm", "a"], ["--wm-idm-gamma", "1000", "100", "100"]), (["--hmax", "28", "--wm", "off"], ["--wm-idm-gamma", "1000", "100", "100"]),
                        (WMB, ["--wm-idm-gamma", "1000", "0", "100"])):
        try:
            rc = gate(init, "r_bad", extra, args=args)
        except ValueError as e:                                                           # the config refuses at build time (wm a / off); the gate's own clause refuses the rest
            rc = f"ValueError: {e}"
        assert rc != 0 and isinstance(rc, str), (args, extra, rc)


def test_gate_g_live_every_wm_group_still_gets_a_gradient_under_the_dose(init):
    assert gate(init, "g_live", ["--wm-idm-gamma", "1000", "100", "100"]) == 0                # selfcheck-only runs G-LIVE inside the trainer
    sc = sc_of(init, "g_live")
    assert sc.get("live") is not None


# ------------------------------------------------------------------------------------------------ both levers on the LADDER path (--wm-only --wm-windows wmonly), the path every ACTSENS arm takes
LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]


def lgate(init, name, extra=()):
    try:
        rc = TR.main(["--out", str(init["root"] / name), *BASE, "--init-ckpt", str(init["ck"]), *LADDER, *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def test_ladder_path_gate_passes_both_levers_and_reads_both_back(init):
    assert lgate(init, "l_both", ["--wm-idm-gamma", "1000", "100", "100", "--wm-sampler", "divrs"]) == 0
    sc = sc_of(init, "l_both")
    assert sc["levers"]["wm_idm_gamma_readback"]["built"] == [1000.0, 100.0, 100.0]
    smp = sc["levers"]["wm_sampler"]
    assert smp["kind"] == "divrs" and sc["wm_sampler_probe"]["pass"] is True
    assert smp["n_windows"] == sc["info"]["n_windows"] == sc["info"]["wm_windows"]["n_windows"]                   # REACHABILITY (b): the key the ladder path updates
    assert sc["declared_vs_built"] == "PASS"


def test_ladder_path_flag_absent_arms_keep_their_key_set_and_record_a_uniform_sampler(init):
    assert lgate(init, "l_none") == 0
    sc = sc_of(init, "l_none")
    assert "wm_idm_gamma_readback" not in sc["levers"] and "wm_sampler" not in sc["levers"] and "wm_sampler_probe" not in sc
    assert sc["info"]["wm_sampler"]["kind"] == "uniform"


def test_ladder_path_uniform_default_reproduces_the_chunked_epoch_order_bit_for_bit(init):
    """C1 on the sampler: the flag-absent arm's sampler IS D.ChunkedSampler called with the baseline arguments (same epoch order), not a DiversitySampler."""
    from tanitad.data import v8f_mini_data as D
    a = TR.build_parser().parse_args(["--out", "x", *BASE[1:]])
    wins = [(i % 5, 40 + (i % 7), 0, 0) for i in range(400)]
    cfg = cfg_of()
    got = TR.make_sampler(a, cfg, wins, None, None, {"lru": 10})
    ref = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, 10 - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    assert type(got) is D.ChunkedSampler and [list(b) for b in got.epoch(0)] == [list(b) for b in ref.epoch(0)]


# ------------------------------------------------------------------------------------------------ AMEND_ACTSENS_1: C3's instrument is G-LIVE's FIRST step
UNW = ["wm", "total"] + [f"wm/{t}/L{l}" for t in ("pred", "sigreg", "idm") for l in (1, 2, 3)] + [f"wm/act_sigreg/L{l}" for l in (2, 3)]


def test_amend1_live_first_identity_dose_against_default_same_seed(init):
    assert lgate(init, "a1_ref", ["--wm-idm-gamma", "100", "100", "100"]) == 0
    assert lgate(init, "a1_dose", ["--wm-idm-gamma", "1000", "100", "100"]) == 0
    r, d = sc_of(init, "a1_ref")["live_first"], sc_of(init, "a1_dose")["live_first"]
    assert set(r) == set(d) and len(r) > 10
    unw = [k for k in r if k.startswith("wm/") and not k.startswith("wm/level/")]
    assert unw and "wm/idm/L1" in unw
    for k in unw:
        assert r[k] == d[k], k
    assert r["wm/level/L2"] == d["wm/level/L2"] and r["wm/level/L3"] == d["wm/level/L3"]
    diff = d["wm/level/L1"] - r["wm/level/L1"]
    want = 900.0 * r["wm/idm/L1"]
    assert abs(diff - want) <= 1e-5 * abs(want), (diff, want)
    # the last step is NOT the first step: the instrument has to be the first (live.terms moved under the update)
    assert sc_of(init, "a1_dose")["live"]["terms"] != d


def test_amend1_flag_absent_has_no_live_first_and_its_live_terms_equal_the_default_named(init):
    assert lgate(init, "a1_off") == 0
    off, ref = sc_of(init, "a1_off"), sc_of(init, "a1_ref")
    assert "live_first" not in off
    assert "terms_first" not in off["live"] and set(off["live"]) == set(ref["live"])
    assert off["live"]["terms"] == ref["live"]["terms"]
