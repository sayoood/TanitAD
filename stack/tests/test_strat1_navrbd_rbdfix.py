"""RBDFIX: the G-LIVE reading of the NAVRB group for a non-zero channel is judged against the channel rows the real self-check batch READS (``navrb_probe.batch_n_on`` / ``batch_group_n_on``), not against a
demand of > 0 whatever the batch carries (the N1Dx_s0 false refusal at 0c3b374f: a shuffled channel whose real batch reads no row gives gradient exactly 0.0, which is correct). CPU only.

Asserted: (checker) no row read + live 0.0 PASSES; rows read + live 0 REFUSED; no row read + live > 0 REFUSED; N1D real unchanged; a D kind without the field REFUSED; the per-group dead rule uses the read counts;
N1Z unchanged. (probe) the field is the count of rows e_nav_rb is called with (hook), not the whole tensor. (training path) only ``navrb_probe`` differs from the 0c3b374f trainer (AST), the probe consumes no RNG
and leaves the model untouched, and a 3-step train() through main() is bit-identical to the 0c3b374f trainer. Each new branch has a MUTATION that must turn a named test RED."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import strat1_check as REAL  # noqa: E402
import test_v8f_navrbd as NB  # noqa: E402
from test_v8f_navrbd import world, make_batch, _argv, N1, RBD, TRAIN_P, GROUPS, M, TR, D  # noqa: E402,F401
from test_strat1_navrbd_check import _sc as _sc_d, _toks, _mutant, GROUPS as CG  # noqa: E402
from test_strat1_token import COMMIT, rig  # noqa: E402,F401

assert Path(REAL.__file__).resolve().is_relative_to(HERE.parent.resolve())
SNAP = HERE / "v8f_mini_train_tip_snapshot_rbdfix.txt"
SNAP_SHA = "0bc40a2fccbeeb0f"                      # first 16 hex of sha256 of the 0c3b374f stack/scripts/v8f_mini_train.py


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _no_rows(sc):
    """The N1Dx_s0 selfcheck: the table has every group, the real batch reads none; live and group gradients exactly 0.0, planted > 0."""
    pr = sc["navrb_probe"]
    pr.update({"batch_n_on": 0, "batch_group_n_on": {g: 0 for g in CG}, "group_grads": {g: 0.0 for g in CG}})
    sc["live"]["groups_grad"]["NAVRB"] = 0.0
    return sc


def _rows(sc):
    sc["navrb_probe"].update({"batch_n_on": 6, "batch_group_n_on": {g: 2 for g in CG}})
    return sc


# ---- checker, written against a module so a mutated copy runs the same code
def check_checker(SC, tmp_path):
    for dk in ("N1D", "N1Dx"):
        t = _toks(SC, dk, tmp_path)
        assert SC.gate_problems(dk, t, _rows(_sc_d(SC, dk))) == [], dk                           # N1D real unchanged
    t = _toks(SC, "N1Dx", tmp_path)
    assert SC.gate_problems("N1Dx", t, _no_rows(_sc_d(SC, "N1Dx"))) == []                          # (1) no row read, live 0.0: PASS
    sc = _rows(_sc_d(SC, "N1Dx")); sc["live"]["groups_grad"]["NAVRB"] = 0.0                       # (2) rows read, live grad 0: REFUSED
    assert any("NAVRB group has no gradient" in p for p in SC.gate_problems("N1Dx", t, sc))
    sc = _no_rows(_sc_d(SC, "N1Dx")); sc["live"]["groups_grad"]["NAVRB"] = 0.3                    # (3) no row read, live > 0: inconsistent, REFUSED
    assert any("reads no channel row" in p for p in SC.gate_problems("N1Dx", t, sc))
    sc = _no_rows(_sc_d(SC, "N1Dx")); sc["navrb_probe"]["e_nav_rb_grad"] = 0.0                    # the planted control is still required
    assert SC.gate_problems("N1Dx", t, sc)
    sc = _sc_d(SC, "N1Dx"); sc["navrb_probe"].pop("batch_n_on"); sc["navrb_probe"].pop("batch_group_n_on", None)   # a D kind without the field: REFUSED
    assert any("batch_n_on missing" in p for p in SC.gate_problems("N1Dx", t, sc))
    sc = _rows(_sc_d(SC, "N1Dx")); sc["navrb_probe"]["batch_group_n_on"]["c13"] = 3; sc["navrb_probe"]["group_grads"]["c13"] = 0.0   # a group read non-zero with no gradient
    assert any("c13" in p and "no gradient" in p for p in SC.gate_problems("N1Dx", t, sc))
    sc = _no_rows(_sc_d(SC, "N1Dx")); sc["navrb_probe"]["group_grads"]["c14"] = 0.2                # a group read as zero with a gradient
    assert any("c14" in p and "inconsistent" in p for p in SC.gate_problems("N1Dx", t, sc))
    sc = _rows(_sc_d(SC, "N1Dx")); sc["navrb_probe"]["batch_n_on"] = -1                            # malformed
    assert SC.gate_problems("N1Dx", t, sc)


def check_other_kinds_unchanged(SC, tmp_path):
    tz = _toks(SC, "N1Z", tmp_path)                                                                # N1Z: no field, live exactly 0.0, still passes
    from test_strat1_navrb2_check import _sc as _sc_r, _toks as _tr
    assert SC.gate_problems("N1Z", tz, _sc_r(SC, "N1Z")) == []
    tr = _tr(SC, "N1R", tmp_path)
    assert SC.gate_problems("N1R", tr, _sc_r(SC, "N1R")) == []                                     # an R kind without the field: the > 0 rule as before
    sc = _sc_r(SC, "N1R"); sc["live"]["groups_grad"]["NAVRB"] = 0.0
    assert SC.gate_problems("N1R", tr, sc)


CHECKS = {"checker": check_checker, "other": check_other_kinds_unchanged}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_rbdfix_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


CMUT = {
    "ignore_batch_n_on_zero_branch": ('        elif bno == 0:\n            if ng != 0.0:', '        elif False:\n            if ng != 0.0:'),
    "drop_inconsistent_clause": ('            if ng != 0.0:                                                       # no channel row', '            if False:                                                           # no channel row'),
    "drop_missing_field_clause": ('        if bno is None and dk:', '        if False:'),
    "drop_group_read_dead_clause": ('            if dead:\n                bad.append(f"navrb_probe: groups {dead} are read non-zero', '            if False:\n                bad.append(f"navrb_probe: groups {dead} are read non-zero'),
    "drop_group_read_zero_clause": ('            if alive:', '            if False:'),
    "no_row_still_demands_positive": ('        elif bno == 0:\n            if ng != 0.0:', '        elif bno == 0 and False:\n            if ng != 0.0:'),
}


@pytest.mark.parametrize("mname", sorted(CMUT))
def test_checker_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, "rbdfix_" + mname, *CMUT[mname])
    with pytest.raises(Exception):
        check_checker(m, tmp_path)


# ---- the probe
def _probe(mod, world, batch_fn=None, seed=0):
    b, J = make_batch(world, "exitdist")
    if batch_fn:
        b = batch_fn(b)
    cfg = M.V8FMiniConfig(**{**NB.TINY, "nav_rb": "on", "nav_rb_feat": "exitdist"})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    m = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)
    with torch.no_grad():
        m.head_o[2].weight.copy_(torch.randn(m.head_o[2].weight.shape, generator=torch.Generator().manual_seed(11)) * 0.05)
    return mod.navrb_probe(m, b, cfg), m


def _zero(b):
    return {**b, "nav_rb": torch.zeros_like(b["nav_rb"])}


def check_probe(mod, world):
    res, _ = _probe(mod, world)
    assert res["pass"] and res["batch_n_on"] > 0 and set(res["batch_group_n_on"]) == set(GROUPS)
    assert all((res["batch_group_n_on"][g] > 0) == (res["group_grads"][g] > 0.0) for g in GROUPS)   # read rows <=> gradient
    res0, _ = _probe(mod, world, _zero)                                                            # a batch with no channel row
    assert res0["pass"] and res0["planted"] and res0["batch_n_on"] == 0 and all(v == 0 for v in res0["batch_group_n_on"].values())
    assert all(v == 0.0 for v in res0["group_grads"].values()) and res0["e_nav_rb_grad"] > 0.0
    b, _ = make_batch(world, "exitdist")                                                           # the rows the tokens read are a part of the tensor: the whole-tensor count is larger
    whole = int((b["nav_rb"].reshape(-1, 15) != 0).any(dim=1).sum())
    assert 0 < res["batch_n_on"] < whole, (res["batch_n_on"], whole)
    return True


def test_probe_reports_the_rows_read(world):
    assert check_probe(TR, world)


PMUT = {
    "drop_the_field": ('        res["batch_n_on"] = int(', '        _dropped = int('),
    "count_the_whole_tensor": ('        rr = torch.cat(read_rows, 0).numpy()', '        rr = b_real["nav_rb"].reshape(-1, res["width"]).numpy()'),
}


@pytest.mark.parametrize("mname", sorted(PMUT))
def test_probe_mutation_turns_a_check_red(mname, world):
    mod = NB.trainer_module(*PMUT[mname], "rbdfix_" + mname)
    with pytest.raises(Exception):
        check_probe(mod, world)


# ---- the training path is unchanged
def _tip_module(tag):
    p = TRAIN_P.with_name(f"_tip_rbdfix_{tag}.py")
    p.write_text(SNAP.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(f"_tip_rbdfix_{tag}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        p.unlink()


def test_the_snapshot_is_the_0c3b374f_trainer():
    assert sha16(SNAP) == SNAP_SHA


def test_only_navrb_probe_differs_from_the_0c3b374f_trainer():
    a, b = ast.parse(SNAP.read_text(encoding="utf-8")), ast.parse(TRAIN_P.read_text(encoding="utf-8"))
    assert len(a.body) == len(b.body)
    diff = [getattr(x, "name", type(x).__name__) for x, y in zip(a.body, b.body) if ast.dump(x) != ast.dump(y)]
    assert diff == ["navrb_probe"], diff


def test_the_probe_keeps_every_old_value_adds_two_keys_and_touches_neither_rng_nor_model(world):
    base = _tip_module("probe")
    torch.manual_seed(123)
    r0, m0 = _probe(base, world)
    s0 = torch.get_rng_state()
    torch.manual_seed(123)
    r1, m1 = _probe(TR, world)
    s1 = torch.get_rng_state()
    assert torch.equal(s0, s1)                                                                     # no extra RNG draw
    assert set(r1) - set(r0) == {"batch_n_on", "batch_group_n_on"} and not set(r0) - set(r1)
    assert all(r0[k] == r1[k] for k in r0)                                                         # every old value identical, floats included
    assert all(torch.equal(p, dict(m1.named_parameters())[n]) for n, p in m0.named_parameters())


def test_train_through_main_is_bit_identical_to_the_0c3b374f_trainer(world):
    base = _tip_module("main")
    outs = {}
    for tag, mod in (("rbdfix_base", base), ("rbdfix_new", TR)):
        assert mod.main(_argv(world, tag, ["--steps", "3", *N1(world), *RBD(world)])) == 0
        outs[tag] = Path(world["root"]) / tag
    a = torch.load(outs["rbdfix_base"] / "ckpt_last.pt", weights_only=False)
    b = torch.load(outs["rbdfix_new"] / "ckpt_last.pt", weights_only=False)
    assert list(a["model"]) == list(b["model"]) and all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])     # every parameter, bit for bit
    sa, sb = a["opt"]["state"], b["opt"]["state"]
    assert set(sa) == set(sb) and all(torch.equal(sa[i][k], sb[i][k]) for i in sa for k in sa[i] if torch.is_tensor(sa[i][k]))
    ja = json.loads((outs["rbdfix_base"] / "selfcheck.json").read_text(encoding="utf-8"))
    jb = json.loads((outs["rbdfix_new"] / "selfcheck.json").read_text(encoding="utf-8"))
    pa, pb = ja["navrb_probe"], jb["navrb_probe"]
    assert set(pb) - set(pa) == {"batch_n_on", "batch_group_n_on"} and all(pa[k] == pb[k] for k in pa)
    assert ja["live"] == jb["live"] and ja["levers"] == jb["levers"]
