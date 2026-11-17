"""PREREG_V8F_STRAT1 SEEDS read COSTS-HEADS: the fourth arm kind S1d (= S1 with ``--ctx-detach on``) in ``stack/scripts/strat1_check.py``. No model run, no GPU.

Asserted: S1d renders ``--ctx-detach on`` and nothing else differs from S1; the argv sha256 of C0 / N1 / S1 is the a4357e79 value (S1 still renders ``off``); the S1d gate read accepts a detach-on selfcheck (adapter ctx gradient exactly 0.0) and
refuses the S1 shape under the S1d kind (and the reverse); the S1d post check and binding checks run. Every new branch has a MUTATION: a copy of the checker with that branch dropped, which one of the checks below must turn RED.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
import strat1_check as REAL  # noqa: E402
from test_strat1_token import COMMIT, rig  # noqa: E402,F401  (the fake-tree rig)

BOX2 = "D:" + chr(92) + "tanitad"
BASE = {"OBD-CTL_s0": {"argv": ["--init-ckpt", BOX2 + chr(92) + "M2b_s0", "--seed", "0", "--dagger-seed", "0", "--out", "x"]},
        "OBD-CTL_s1": {"argv": ["--init-ckpt", BOX2 + chr(92) + "M2b_s1", "--seed", "1", "--dagger-seed", "1", "--out", "x"]}}
A4357E79_SHA = {  # MEASURED: render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=BASE) on strat1_check.py exactly as at a4357e79
    ("C0", "s0"): "92065ff4b198a40a390161736c1cebe5358f350cbf126830f4370a55b3980975", ("C0", "s1"): "7919d4f91ee249163d5d4f72fa57ba5d7833ec8e2aa11add8a62290533a353bb",
    ("N1", "s0"): "d6e0f8a34d3cb50d588c4454d601c04b2c13d0d575f0cf59f93cb153720b2ddd", ("N1", "s1"): "dca7047008458780ba90e51dc336058fc49ffcbc1a387d19c189c4a44493c8db",
    ("S1", "s0"): "501afa6b4f7b33794626c3b4366f31c8e708a4d56dfb26de327c7fcad50e2c53", ("S1", "s1"): "b98a3f16e3626cfe5145f2f9460d716016e48f62fe58426c61706f47d68a65a1"}


def _base(tmp_path):
    p = tmp_path / "argvs.json"
    p.write_text(json.dumps(BASE), encoding="utf-8")
    return str(p)


def _render(SC, kind, seed, tmp_path):
    return SC.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path))


def _sc(SC, kind):
    """A minimal selfcheck.json that the gate read accepts for ``kind`` (S1d: detach on, adapter ctx gradient exactly 0.0)."""
    det = "on" if kind == "S1d" else "off"
    return {"declared_vs_built": "PASS", "steps_total": SC.RUN_STEPS, "live": {"pass": True, "groups_grad": {"CTX": 1.0}},
            "levers": {"ctx_heads": "on", "w_ctx": 0.5, "ctx_dim": 41, "ctx_vision_only_source": True, "ctx_detach": det},
            "nav_source_probe": {"pass": True, "source": "text", "n_text_ne_v9": 5},
            "ctx_head_probe": {"pass": True, "control_head_grad_w0": 0.0, "control_allmasked_head_grad": 0.0, "control_adapter_grad_detach_on": 0.0,
                               "adapter_grad_ctx": 0.0 if kind == "S1d" else 0.25},
            "ctx_labels": {"sha256": SC.STRAT_LABELS_SHA256}, "acc_probe": {"ok": 1}, "dagger_probe": {"ok": 1}, "otband_probe": {"detach_upper": True}}


def _toks(SC, kind, tmp_path):
    return _render(SC, kind, "s0", tmp_path) + ["--detach-upper", "--steps", str(SC.RUN_STEPS)]


# ---- the checks, written against a module so a mutated copy can be run through the same code ----
def check_render(SC, tmp_path):
    s1, s1d = _render(SC, "S1", "s0", tmp_path), _render(SC, "S1d", "s0", tmp_path)
    assert SC.flag_value(s1d, "--ctx-detach") == "on" and SC.flag_value(s1, "--ctx-detach") == "off"
    diff = [i for i, (x, y) in enumerate(zip(s1, s1d)) if x != y]
    assert len(s1) == len(s1d) and [s1d[i - 1] for i in diff] == ["--ctx-detach"] and [s1d[i] for i in diff] == ["on"]
    assert SC.flag_value(s1d, "--ctx-heads") == "on" and SC.flag_value(s1d, "--nav-source") == "text" and SC.flag_value(s1d, "--ctx-lr-mult") == "10"
    for seed in ("s0", "s1", "s10", "s11"):
        a = _render(SC, "S1d", seed, tmp_path)
        assert SC.flag_value(a, "--ctx-detach") == "on" and SC.flag_value(a, "--seed") == ({"s0": "0", "s1": "1"}.get(seed, seed[1:]))


def check_gate(SC, tmp_path):
    t, sc = _toks(SC, "S1d", tmp_path), _sc(SC, "S1d")
    assert SC.gate_problems("S1d", t, sc) == []
    s1_t, s1_sc = _toks(SC, "S1", tmp_path), _sc(SC, "S1")
    assert SC.gate_problems("S1", s1_t, s1_sc) == []
    assert SC.gate_problems("S1d", s1_t, sc)                                   # argv detach off under S1d: refused
    bl = json.loads(json.dumps(sc))
    bl["levers"]["ctx_detach"] = "off"
    assert SC.gate_problems("S1d", t, bl)                                      # built detach off under S1d: refused
    assert SC.gate_problems("S1", t, s1_sc)                                    # argv detach on under S1: refused
    leak = json.loads(json.dumps(sc))
    leak["ctx_head_probe"]["adapter_grad_ctx"] = 0.25
    assert SC.gate_problems("S1d", t, leak)                                    # S1d needs exactly 0.0 on the adapter
    nograd = json.loads(json.dumps(s1_sc))
    nograd["ctx_head_probe"]["adapter_grad_ctx"] = 0.0
    assert SC.gate_problems("S1", s1_t, nograd)                                # S1 needs > 0
    off = json.loads(json.dumps(sc))
    off["levers"]["ctx_heads"] = "off"
    assert SC.gate_problems("S1d", t, off)                                     # S1d is a ctx-on arm


def check_post(SC, tmp_path):
    import torch
    from tanitad.models import v8f_mini as M
    d = 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(M.CTX_INIT_SEED)
        fresh = M._mlp(d, d, M.CTX_DIM).state_dict()
    init = {"adapter.w": torch.zeros(2)}
    fin_moved = {"adapter.w": torch.ones(2), **{"ctx_head." + k: v + 1.0 for k, v in fresh.items()}}
    fin_fresh = {"adapter.w": torch.ones(2), **{"ctx_head." + k: v.clone() for k, v in fresh.items()}}
    assert SC.moved_identity(init, fin_moved, "S1d") == []
    assert SC.moved_identity(init, {"adapter.w": torch.ones(2)}, "S1d")        # no ctx_head keys: refused
    assert SC.moved_identity(init, fin_fresh, "S1d")                           # ctx_head did not move: refused
    assert SC.moved_identity(init, fin_moved, "C0")                            # ctx keys in a ctx-off arm: refused
    assert SC.moved_identity(init, fin_moved, "S1") == []


def check_nav(SC, tmp_path):
    missing = str(tmp_path / "no_such_report.json")
    assert SC.nav_report_problems("S1d", "/n.npz", missing)                    # S1d reads the text nav: the report is required
    assert SC.nav_report_problems("C0", "/n.npz", missing) == []


CHECKS = {"render": check_render, "gate": check_gate, "post": check_post, "nav": check_nav}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_s1d_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("key", sorted(A4357E79_SHA))
def test_existing_arm_argv_sha_is_unchanged(key, tmp_path):
    kind, seed = key
    assert REAL.argv_sha(_render(REAL, kind, seed, tmp_path)) == A4357E79_SHA[key]


def test_kinds_and_seeds_include_s1d():
    assert REAL.KINDS[:4] == ("C0", "N1", "S1", "S1d") and set(REAL.SEEDS) >= {"s0", "s1", "s10", "s11"}


def test_s1d_binding_and_token_run(rig, tmp_path):
    """The binding and the runner accept S1d as they accept S1; a token minted for S1 is refused under the S1d kind (the kind is bound)."""
    b, bad = REAL.binding("S1d", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"])
    assert bad == []
    assert REAL.binding("S1d", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=str(tmp_path / "none.json"))[1]
    rd = os.path.join(rig["a"].runs, "S1d_s0")
    os.makedirs(os.path.join(rd, "gate"))
    json.dump(rig["toks"], open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    sc = os.path.join(rd, "gate", "selfcheck.json")
    open(sc, "w", encoding="utf-8").write('{"declared_vs_built": "PASS"}')
    REAL.write_token(os.path.join(rd, "gate", "gate_token.json"), b, {c: {"pass": True} for c in REAL.REQUIRED_CHECKS}, REAL.sha_file(sc))
    assert REAL.token_problems(os.path.join(rd, "gate", "gate_token.json"), "S1d", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"]) == []
    assert REAL.run_arm("S1d_s0", "S1d", rig["a"]) == 0
    assert REAL.token_problems(os.path.join(rd, "gate", "gate_token.json"), "S1", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"])


def _mutant(tmp_path, name, old, new):
    src = open(os.path.join(SCRIPTS, "strat1_check.py"), "rb").read().decode("utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    d = tmp_path / "mut" / name
    d.mkdir(parents=True)
    (d / "strat1_check.py").write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("strat1_mut_" + name, str(d / "strat1_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MUTATIONS = {
    "kinds_drop_s1d": ('KINDS = ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z")', 'KINDS = ("C0", "N1", "S1", "N1R", "S1R", "S1dR", "N1Z")'),
    "render_detach_off": ('"--ctx-detach", "on" if bk == "S1d" else "off"]', '"--ctx-detach", "off"]'),
    "ctx_kinds_drop_s1d": ('CTX_KINDS = ("S1", "S1d", "S1R", "S1dR")', 'CTX_KINDS = ("S1", "S1R", "S1dR")'),
    "nav_kinds_drop_s1d": ('NAV_KINDS = ("N1", "S1", "S1d") + RB_KINDS', 'NAV_KINDS = ("N1", "S1") + RB_KINDS'),
    "gate_drop_detach_check": ('if flag_value(toks, "--ctx-detach") != want_det or lv.get("ctx_detach", "off") != want_det:', 'if False:'),
    "gate_drop_adapter_grad_check": ('if not isinstance(ag, (int, float)) or (ag != 0.0 if bk == "S1d" else not ag > 0.0):', 'if False:'),
    "post_drop_s1d": ('    if base_kind(kind) in CTX_KINDS:\n        if not ck:', '    if base_kind(kind) == "S1":\n        if not ck:'),
}
EXPECT_RED = {"kinds_drop_s1d": "render", "render_detach_off": "render", "ctx_kinds_drop_s1d": "render", "nav_kinds_drop_s1d": "nav",
              "gate_drop_detach_check": "gate", "gate_drop_adapter_grad_check": "gate", "post_drop_s1d": "post"}


@pytest.mark.parametrize("mname", sorted(MUTATIONS))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(Exception):
        CHECKS[EXPECT_RED[mname]](m, tmp_path)
