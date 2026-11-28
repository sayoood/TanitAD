"""The post-training zero-gradient gate (stack/scripts/zerograd_gate.py): every case, with teeth.

A tiny torch model is trained with REAL Adam steps and its checkpoint is written in the layout the
trainers use (``{"model": state_dict, "opt": opt.state_dict(), "step": n}``), so the gate is exercised on
the optimizer-state structure torch actually produces, not on a hand-built mock.

Cases (the CLAUDE.md "built, tested, and unreachable from its caller" class):
  control   every parameter receives a gradient                      -> PASS
  case (a)  a head whose output is detached: NO optimizer state        -> FAIL naming it
  case (b)  a head multiplied by a loss weight of 0.0: state EXISTS,
            all exactly zero (``0.0 * term`` yields a zeros grad,
            not None -- the case a count-only check misses)           -> FAIL naming it
  --allow   suppresses exactly the named parameters (re.fullmatch)
  MUTATION  disable either check and the matching broken checkpoint PASSES -- the tests have teeth.
CPU only, no data, a few seconds.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import zerograd_gate as G  # noqa: E402

GATE_PY = ROOT / "scripts" / "zerograd_gate.py"


class Tiny(nn.Module):
    """enc (with BatchNorm => float buffers in the state_dict) + three heads + a scalar parameter + a float buffer."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU())
        self.head_ok = nn.Linear(16, 4)
        self.head_det = nn.Linear(16, 3)
        self.head_z = nn.Linear(16, 5)
        self.temp = nn.Parameter(torch.zeros(()))
        self.register_buffer("anchors", torch.zeros(4, 4))


def _train(path, *, detach=False, zero=False, groups=None, steps=6, save_opt=True, optimizer="adam",
           extra=None):
    torch.manual_seed(0)
    m = Tiny()
    if groups is None:
        opt = (torch.optim.Adam(m.parameters(), lr=1e-2) if optimizer == "adam"
               else torch.optim.SGD(m.parameters(), lr=1e-2))
    else:
        opt = torch.optim.Adam(groups(m), lr=1e-2)
    for _ in range(steps):
        x = torch.randn(32, 8)
        h = m.enc(x)
        loss = m.head_ok(h).pow(2).mean() + (m.temp - 1.0).pow(2)
        d = m.head_det(h).pow(2).mean()
        loss = loss + (d.detach() if detach else d)                  # case (a): output detached, params get no grad
        z = m.head_z(h).pow(2).mean()
        loss = loss + (0.0 * z if zero else z)                       # case (b): zeros grad tensor, NOT None
        opt.zero_grad()
        loss.backward()
        opt.step()
    ck = {"model": m.state_dict(), "step": steps}
    if save_opt:
        ck["opt"] = opt.state_dict()
    ck.update(extra or {})
    torch.save(ck, path)
    return m, opt


def _run(path, *args):
    return G.main(["--ckpt", str(path), *args])


def _names(rep, key="offenders"):
    return sorted(n for r in rep[key] for n in r["names"])


def _rep(path, **kw):
    ck, _ = G.load_ckpt(str(path))
    return G.analyze(ck, **kw)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def ck_ok(tmp_path_factory):
    p = tmp_path_factory.mktemp("zg") / "ok.pt"
    _train(p)
    return p


@pytest.fixture(scope="module")
def ck_detached(tmp_path_factory):
    p = tmp_path_factory.mktemp("zg") / "det.pt"
    m, opt = _train(p, detach=True)
    # PRECONDITION (case a): the head really has no optimizer state, so the test is about what it says it is
    assert all(prm not in opt.state for prm in m.head_det.parameters())
    return p


@pytest.fixture(scope="module")
def ck_zero(tmp_path_factory):
    p = tmp_path_factory.mktemp("zg") / "zero.pt"
    m, opt = _train(p, zero=True)
    # PRECONDITION (case b): state EXISTS and is exactly zero -- a count of "registered vs with state" reads clean
    for prm in m.head_z.parameters():
        assert prm in opt.state
        assert all(int(torch.count_nonzero(v)) == 0 for k, v in opt.state[prm].items() if k != "step")
    return p


@pytest.fixture(scope="module")
def ck_both(tmp_path_factory):
    p = tmp_path_factory.mktemp("zg") / "both.pt"
    _train(p, detach=True, zero=True)
    return p


# ------------------------------------------------------------------------------------------------ #
def test_all_trained_passes(ck_ok, capsys):                      # the CONTROL: the gate must be able to say PASS
    assert _run(ck_ok, "--control", r"head_ok\..*", "--control", r"enc\.0\..*") == G.EXIT_PASS
    rep = _rep(ck_ok)
    assert rep["verdict"] == "PASS" and rep["offenders"] == []
    assert rep["optimizer"]["n_registered"] == rep["optimizer"]["n_trained"] == 11
    assert rep["alignment"]["mode"] == "global-shape-dp" and rep["alignment"]["n_ids_unnamed"] == 0
    # BatchNorm running stats are excluded from the name candidates, so every name is unique; without that prior
    # the same checkpoint has ambiguous BN names (and still the same verdict -- naming never decides it)
    assert rep["alignment"]["unique"] and rep["alignment"]["n_ids_name_ambiguous"] == 0
    raw = _rep(ck_ok, exclude_entries=())
    assert raw["alignment"]["n_ids_name_ambiguous"] > 0 and raw["verdict"] == "PASS"
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_detached_head_fails_naming_it_case_a(ck_detached, capsys):
    assert _run(ck_detached) == G.EXIT_FAIL
    rep = _rep(ck_detached)
    assert _names(rep) == ["head_det.bias", "head_det.weight"]
    assert {r["case"] for r in rep["offenders"]} == {"a_no_state"}
    assert not any(r["ambiguous"] for r in rep["offenders"])
    assert rep["untrained_numel"]["offending_lo"] == rep["untrained_numel"]["offending_hi"] == 3 * 16 + 3
    out = capsys.readouterr().out
    assert "head_det.weight" in out and "VERDICT: FAIL" in out


def test_zero_weight_head_fails_naming_it_case_b(ck_zero):
    assert _run(ck_zero) == G.EXIT_FAIL
    rep = _rep(ck_zero)
    assert _names(rep) == ["head_z.bias", "head_z.weight"]
    assert {r["case"] for r in rep["offenders"]} == {"b_zero_state"}
    assert rep["optimizer"]["n_case_a_no_state"] == 0 and rep["optimizer"]["n_case_b_zero_state"] == 2
    assert rep["optimizer"]["n_with_state_entry"] == rep["optimizer"]["n_registered"]      # case (a) alone reads clean
    assert rep["untrained_numel"]["offending_lo"] == 5 * 16 + 5


def test_allow_suppresses_exactly_the_named(ck_both, ck_detached):
    assert _names(_rep(ck_both)) == ["head_det.bias", "head_det.weight", "head_z.bias", "head_z.weight"]
    # allowing one head leaves the other failing
    rep = _rep(ck_both, allow=[r"head_det\..*"])
    assert _names(rep) == ["head_z.bias", "head_z.weight"] and _names(rep, "allowed") == [
        "head_det.bias", "head_det.weight"]
    assert _run(ck_both, "--allow", r"head_det\..*") == G.EXIT_FAIL
    # allowing both passes, and is visible in the report
    assert _run(ck_both, "--allow", r"head_det\..*", "--allow", r"head_z\..*") == G.EXIT_PASS
    # a pattern naming ONE tensor leaves its sibling failing
    rep = _rep(ck_detached, allow=[r"head_det\.weight"])
    assert _names(rep) == ["head_det.bias"] and _names(rep, "allowed") == ["head_det.weight"]
    # re.fullmatch: a prefix does not widen into a suppression, and the dead pattern is reported as unused
    rep = _rep(ck_detached, allow=["head_de", r"head_det"])
    assert _names(rep) == ["head_det.bias", "head_det.weight"] and rep["unused_allow"] == ["head_de", "head_det"]
    assert _run(ck_detached, "--allow", "head_de") == G.EXIT_FAIL


def test_MUTATION_disabling_case_b_lets_the_zero_weight_head_pass(ck_zero, ck_detached, monkeypatch):
    assert _rep(ck_zero)["verdict"] == "FAIL"                       # baseline: red
    real = G.classify_state
    monkeypatch.setattr(G, "classify_state", lambda e: G.TRAINED if real(e) == G.ZERO_STATE else real(e))
    assert _rep(ck_zero)["verdict"] == "PASS"                       # case (b) check removed -> the defect ships
    assert _run(ck_zero) == G.EXIT_PASS
    assert _rep(ck_detached)["verdict"] == "FAIL"                   # ...and only case (b) was removed


def test_MUTATION_disabling_case_a_lets_the_detached_head_pass(ck_detached, ck_zero, monkeypatch):
    assert _rep(ck_detached)["verdict"] == "FAIL"
    real = G.classify_state
    monkeypatch.setattr(G, "classify_state", lambda e: G.TRAINED if real(e) == G.NO_STATE else real(e))
    assert _rep(ck_detached)["verdict"] == "PASS"
    assert _rep(ck_zero)["verdict"] == "FAIL"


def test_a_tiny_but_nonzero_gradient_is_not_a_zero_gradient_and_nan_is_not_zero():
    # g*g underflows (|g| < ~3e-23) while exp_avg does not: judging by exp_avg_sq alone would call this dead.
    tiny = {"step": torch.tensor(5.0), "exp_avg": torch.tensor([1e-25]), "exp_avg_sq": torch.tensor([0.0])}
    assert float(tiny["exp_avg_sq"].sum()) == 0.0
    assert G.classify_state(tiny) == G.TRAINED
    nan = {"step": torch.tensor(5.0), "exp_avg": torch.tensor([float("nan")]), "exp_avg_sq": torch.tensor([0.0])}
    assert G.classify_state(nan) == G.TRAINED
    zeros = {"step": torch.tensor(5.0), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.zeros(3)}
    assert G.classify_state(zeros) == G.ZERO_STATE
    assert G.classify_state(None) == G.NO_STATE and G.classify_state({}) == G.NO_STATE


# ------------------------------------------------------------------------------------------------ #
# naming: a same-shaped buffer next to a no-state parameter is AMBIGUOUS -- listed, never guessed
# ------------------------------------------------------------------------------------------------ #
class Amb(nn.Module):
    def __init__(self):
        super().__init__()
        self.dead = nn.Parameter(torch.zeros(4))                    # state_dict order: dead, cache, lin.*
        self.register_buffer("cache", torch.zeros(4))
        self.lin = nn.Linear(4, 4)


def _amb(path, **extra):
    m = Amb()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        m.lin(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": 3, **extra}, path)


def test_ambiguous_name_is_listed_and_allow_fails_closed(tmp_path):
    p = tmp_path / "amb.pt"
    _amb(p)
    rep = _rep(p)
    (row,) = rep["offenders"]
    assert row["ambiguous"] and sorted(row["names"]) == ["cache", "dead"]
    assert rep["alignment"]["n_ids_name_ambiguous"] == 1 and rep["verdict"] == "FAIL"
    # the allow names ONE candidate: it must NOT suppress a row that might be the other one
    assert _run(p, "--allow", "dead") == G.EXIT_FAIL
    assert _run(p, "--allow", "dead", "--allow", "cache") == G.EXIT_PASS
    # declaring the buffer resolves the name: the ambiguity was real, and --exclude-entry is how it is removed
    assert _rep(p, exclude_entries=["cache"])["offenders"][0]["names"] == ["dead"]
    assert _run(p, "--exclude-entry", "cache", "--allow", "dead") == G.EXIT_PASS


def test_exact_naming_from_param_names_resolves_the_ambiguity(tmp_path):
    names = ["dead", "lin.weight", "lin.bias"]
    p = tmp_path / "amb_names.pt"
    _amb(p, param_names=names)                                      # the trainer-provided key
    rep = _rep(p)
    (row,) = rep["offenders"]
    assert rep["alignment"]["mode"] == "param_names" and row["names"] == ["dead"] and not row["ambiguous"]
    assert _run(p, "--allow", "dead") == G.EXIT_PASS
    # the same via the CLI file, and an inconsistent name list is refused rather than trusted
    pn = tmp_path / "names.json"
    pn.write_text(json.dumps(names))
    p2 = tmp_path / "amb_plain.pt"
    _amb(p2)
    assert _run(p2, "--param-names", str(pn), "--allow", "dead") == G.EXIT_PASS
    pn.write_text(json.dumps(["dead", "lin.bias", "lin.weight"]))   # swapped: shapes disagree with the state
    assert _run(p2, "--param-names", str(pn)) == G.EXIT_ERROR


def test_group_order_that_differs_from_state_dict_order_is_named_per_group(tmp_path):
    # the --opt dd layout: an encoder group and a head group, listed in an order that is not state_dict order
    def groups(m):
        enc = list(m.enc.parameters())
        heads = [p for n, p in m.named_parameters() if not n.startswith("enc.")]
        return [{"params": heads}, {"params": enc, "lr": 5e-3}]

    p = tmp_path / "dd.pt"
    _train(p, detach=True, groups=groups)
    rep = _rep(p)
    assert rep["alignment"]["mode"] == "per-group-shape-dp"
    assert _names(rep) == ["head_det.bias", "head_det.weight"]
    assert not any(r["ambiguous"] for r in rep["offenders"])
    assert rep["verdict"] == "FAIL"


def test_alignment_dp_matches_brute_force_enumeration():
    import random
    rnd = random.Random(7)
    pool = [(2,), (3,), (2, 3)]
    for _ in range(60):
        n, m = rnd.randint(0, 4), rnd.randint(0, 6)
        ents = [rnd.choice(pool) for _ in range(m)]
        ids = [rnd.choice(pool + [None]) for _ in range(n)]
        total, cands = G.align_candidates(ids, ents)
        count, seen = 0, [set() for _ in range(n)]
        for combo in itertools.combinations(range(m), n):
            if all(s is None or tuple(s) == ents[j] for s, j in zip(ids, combo)):
                count += 1
                for i, j in enumerate(combo):
                    seen[i].add(j)
        assert total == count
        assert [set(c) for c in cands] == seen


# ------------------------------------------------------------------------------------------------ #
# controls, un-evaluable checkpoints, and the real CLI contract
# ------------------------------------------------------------------------------------------------ #
def test_a_control_that_matches_no_trained_parameter_fails_the_gate(ck_ok, ck_detached):
    assert _run(ck_ok, "--control", r"nonexistent\..*") == G.EXIT_FAIL       # a probe that cannot see its control
    assert _run(ck_detached, "--control", r"head_det\..*") == G.EXIT_FAIL    # the dead head is not a trained control
    rep = _rep(ck_ok, controls=[r"head_ok\..*", r"zzz"])
    assert [c["ok"] for c in rep["controls"]] == [True, False]


def test_unevaluable_checkpoints_are_an_error_never_a_pass(tmp_path, capsys):
    p = tmp_path / "weights_only.pt"
    _train(p, save_opt=False)
    assert _run(p) == G.EXIT_ERROR
    assert "no optimizer state" in capsys.readouterr().err
    sgd = tmp_path / "sgd.pt"
    _train(sgd, optimizer="sgd")                                    # no momentum => no per-parameter state at all
    assert _run(sgd) == G.EXIT_ERROR
    assert _run(tmp_path / "missing.pt") == G.EXIT_ERROR


def test_cli_subprocess_contract_exit_code_json_and_sha256(ck_detached, tmp_path):
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, str(GATE_PY), "--ckpt", str(ck_detached), "--json-out", str(out),
                        "--sha256"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["verdict"] == "FAIL"
    assert sorted(n for o in rep["offenders"] for n in o["names"]) == ["head_det.bias", "head_det.weight"]
    assert rep["ckpt_sha256"] == hashlib.sha256(Path(ck_detached).read_bytes()).hexdigest()
    assert rep["load"]["weights_only"] is True
    assert "VERDICT: FAIL" in r.stdout
