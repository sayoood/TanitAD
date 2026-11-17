"""PREREG_V8F_NAVRB2: the arm kinds N1R / S1R / S1dR / N1Z in ``stack/scripts/strat1_check.py`` (HANDOFF step 1). No model run, no GPU (the post check builds tiny tensors only).

Asserted: an R kind renders its base kind's argv byte for byte and APPENDS exactly the five --nav-rb flags (N1Z: variant zero); the argv sha256 of C0 / N1 / S1 at s0 / s1 is the a4357e79 value (no existing kind moves); the gate read
accepts an R selfcheck and refuses each missing or false control (zero-channel control, step-0 identity, gradient, sha, variant, lever); N1Z reads gradient exactly 0.0 and the weight declared dead; a STRAT1 kind refuses any nav_rb
trace; the post check needs ``e_nav_rb.weight`` moved off 0.0 (N1Z: exactly 0.0, non-R: absent); the binding reads the nav_rb builder report (G-CLOCK-RB). Every new branch has a MUTATION: a copy of the checker with that branch dropped,
which one of the checks must turn RED.
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
from test_strat1_s1d import A4357E79_SHA, _base, _sc as _s1_sc  # noqa: E402
from test_strat1_token import COMMIT, rig  # noqa: E402,F401  (the fake-tree rig)

RBSHA = "ab" * 32


def _render(SC, kind, seed, tmp_path, rb="/rb.npz"):
    return SC.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)


def _toks(SC, kind, tmp_path):
    t = _render(SC, kind, "s0", tmp_path) + ["--detach-upper"]
    t += ["--steps", str(SC.RUN_STEPS)]
    t[t.index("--nav-rb-sha256") + 1] = RBSHA if "--nav-rb-sha256" in t else None
    return t


def _sc(SC, kind):
    """A minimal selfcheck.json the gate read accepts for ``kind``."""
    bk = SC.RB_BASE.get(kind, kind)
    sc = _s1_sc(SC, "S1d" if bk == "S1d" else "S1") if bk in ("S1", "S1d") else _s1_sc(SC, "S1")
    if bk not in ("S1", "S1d"):                                           # a ctx-off arm
        sc["levers"] = {"ctx_heads": "off", "ctx_detach": "off"}
        sc["live"] = {"pass": True, "groups_grad": {}}
        sc.pop("ctx_head_probe"), sc.pop("ctx_labels")
    zero = kind == "N1Z"
    sc["levers"]["nav_rb"] = "on"
    sc["info"] = {"nav_rb": {"sha256": RBSHA, "variant": "zero" if zero else "real", "n_on": 0 if zero else 7}}
    sc["navrb_probe"] = {"pass": True, "step0_identity": True, "keys_read_has_nav_rb": True, "e_nav_rb_grad": 0.5, "control_e_nav_rb_grad_zero_channel": 0.0}
    sc["live"]["groups_grad"] = {**sc["live"].get("groups_grad", {}), "NAVRB": 0.0 if zero else 0.25}
    if zero:
        sc["live"]["declared_dead"] = ["e_nav_rb.weight"]
    return sc


def _edit(sc, path, val):
    d = json.loads(json.dumps(sc))
    o = d
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = val
    return d


# ---- the checks, written against a module so a mutated copy can be run through the same code ----
def check_render(SC, tmp_path):
    for rk in ("N1R", "S1R", "S1dR", "N1Z"):
        bk = SC.RB_BASE[rk]
        for seed in ("s0", "s1", "s10", "s11"):
            b, r = _render(SC, bk, seed, tmp_path), _render(SC, rk, seed, tmp_path)
            assert r[:len(b)] == b, (rk, seed)                               # the base argv byte for byte
            tail = r[len(b):]
            assert tail[0::2] == list(SC.RB_FLAGS) and tail[1] == "on" and tail[3] == "/rb.npz" and tail[7] == ("zero" if rk == "N1Z" else "real") and tail[9] == "10"
            assert tail[5] == "<sha of navtext_nav_rb.npz>"
    assert SC.flag_value(_render(SC, "S1dR", "s0", tmp_path), "--ctx-detach") == "on" and SC.flag_value(_render(SC, "S1R", "s0", tmp_path), "--ctx-detach") == "off"
    assert SC.flag_value(_render(SC, "N1R", "s0", tmp_path), "--ctx-heads") == "off" and SC.flag_value(_render(SC, "S1R", "s0", tmp_path), "--ctx-heads") == "on"
    assert not any(f in _render(SC, "S1", "s0", tmp_path) for f in SC.RB_FLAGS)


def check_gate(SC, tmp_path):
    for rk in ("N1R", "S1R", "S1dR", "N1Z"):
        t, sc = _toks(SC, rk, tmp_path), _sc(SC, rk)
        assert SC.gate_problems(rk, t, sc) == [], rk
    t, sc = _toks(SC, "N1R", tmp_path), _sc(SC, "N1R")
    for path, val in ((("navrb_probe", "control_e_nav_rb_grad_zero_channel"), 1e-9), (("navrb_probe", "step0_identity"), False), (("navrb_probe", "e_nav_rb_grad"), 0.0),
                      (("navrb_probe", "pass"), False), (("navrb_probe", "keys_read_has_nav_rb"), False), (("levers", "nav_rb"), "off"), (("info", "nav_rb", "sha256"), "cd" * 32),
                      (("info", "nav_rb", "variant"), "zero"), (("info", "nav_rb", "n_on"), 0), (("live", "groups_grad", "NAVRB"), 0.0)):
        assert SC.gate_problems("N1R", t, _edit(sc, path, val)), path
    no_probe = json.loads(json.dumps(sc))
    del no_probe["navrb_probe"]
    assert SC.gate_problems("N1R", t, no_probe)
    assert SC.gate_problems("N1R", [x for x in t if x != "--nav-rb-lr-mult"], sc)              # an unnamed flag is refused
    assert SC.gate_problems("N1R", _toks(SC, "N1Z", tmp_path), sc)                             # the zero-variant argv under N1R
    assert SC.gate_problems("N1R", t[:t.index("--nav-rb-lr-mult") + 1] + ["30"] + t[t.index("--nav-rb-lr-mult") + 2:], sc)   # the dose is the card's 10
    # N1Z: gradient exactly 0.0 and the weight declared dead
    tz, sz = _toks(SC, "N1Z", tmp_path), _sc(SC, "N1Z")
    assert SC.gate_problems("N1Z", tz, _edit(sz, ("live", "groups_grad", "NAVRB"), 0.25))
    assert SC.gate_problems("N1Z", tz, _edit(sz, ("live", "declared_dead"), []))
    assert SC.gate_problems("N1Z", t, sz)                                                      # the real-variant argv under N1Z
    # S1R keeps S1's ctx rules; S1dR keeps S1d's (adapter gradient exactly 0.0)
    ts, ss = _toks(SC, "S1R", tmp_path), _sc(SC, "S1R")
    assert SC.gate_problems("S1R", ts, _edit(ss, ("ctx_head_probe", "adapter_grad_ctx"), 0.0))
    td, sd = _toks(SC, "S1dR", tmp_path), _sc(SC, "S1dR")
    assert SC.gate_problems("S1dR", td, _edit(sd, ("ctx_head_probe", "adapter_grad_ctx"), 0.25))
    assert SC.gate_problems("S1dR", ts, sd)                                                    # detach off in the argv under S1dR
    # a STRAT1 kind refuses any nav_rb trace; a ctx-on R selfcheck under a ctx-off R kind is refused
    base = _toks(SC, "N1R", tmp_path)
    plain_t = [x for i, x in enumerate(base) if x not in SC.RB_FLAGS and not (i and base[i - 1] in SC.RB_FLAGS)]
    plain_sc = json.loads(json.dumps(sc))
    plain_sc["levers"]["nav_rb"] = "off"
    plain_sc.pop("navrb_probe")
    plain_sc["info"] = {}
    assert SC.gate_problems("N1", plain_t, plain_sc) == []
    assert SC.gate_problems("N1", base, plain_sc)                                              # an N1 argv that names the flags
    assert SC.gate_problems("N1", plain_t, sc)                                                 # an N1 selfcheck that built the channel
    assert SC.gate_problems("N1R", plain_t, plain_sc)                                          # an R kind with no channel


def check_post(SC, tmp_path):
    import torch
    init = {"adapter.w": torch.zeros(2)}
    ok = {"adapter.w": torch.ones(2), "e_nav_rb.weight": torch.full((4, 1), 0.01)}
    assert SC.moved_identity(init, ok, "N1R") == []
    assert SC.moved_identity(init, {"adapter.w": torch.ones(2)}, "N1R")                         # no channel weight
    assert SC.moved_identity(init, {**ok, "e_nav_rb.weight": torch.zeros(4, 1)}, "N1R")        # a weight that never left 0.0
    assert SC.moved_identity(init, {**ok, "e_nav_rb.weight": torch.zeros(4, 1)}, "N1Z") == []  # N1Z: exactly 0.0
    assert SC.moved_identity(init, ok, "N1Z")                                                   # N1Z with a moved weight
    assert SC.moved_identity(init, ok, "N1")                                                    # a channel key in a non-R checkpoint
    assert SC.moved_identity(init, {"adapter.w": torch.ones(2)}, "N1") == []


def check_report(SC, tmp_path):
    rb = tmp_path / "rb.npz"
    rb.write_bytes(b"rbbytes")
    good = {sp: {"G-CLOCK-RB_pass": True, "G-CLOCK-RB_regression_arm_shift_5": {"pass": False}} for sp in ("train", "eval")}
    good["navtext_nav_rb_sha256"] = SC.sha_file(str(rb))
    rp = tmp_path / "rep.json"
    rp.write_text(json.dumps(good), encoding="utf-8")
    assert SC.rb_report_problems("N1R", str(rb), str(rp)) == []
    assert SC.rb_report_problems("N1", str(rb), str(tmp_path / "none.json")) == []             # only R kinds read it
    assert SC.rb_report_problems("N1R", str(rb), str(tmp_path / "none.json"))
    for mut in (lambda r: r["eval"].update({"G-CLOCK-RB_pass": False}), lambda r: r["train"]["G-CLOCK-RB_regression_arm_shift_5"].update({"pass": True}),
                lambda r: r.update({"navtext_nav_rb_sha256": "00" * 32})):
        bad = json.loads(json.dumps(good))
        mut(bad)
        rp.write_text(json.dumps(bad), encoding="utf-8")
        assert SC.rb_report_problems("N1R", str(rb), str(rp))


CHECKS = {"render": check_render, "gate": check_gate, "post": check_post, "report": check_report}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_navrb2_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("key", sorted(A4357E79_SHA))
def test_existing_arm_argv_sha_is_unchanged(key, tmp_path):
    kind, seed = key
    assert REAL.argv_sha(REAL.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path))) == A4357E79_SHA[key]


def test_kinds_and_bases():
    assert REAL.KINDS == ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z")
    assert REAL.RB_BASE == {"N1R": "N1", "S1R": "S1", "S1dR": "S1d", "N1Z": "N1"} and set(REAL.RB_KINDS) <= set(REAL.NAV_KINDS)
    assert set(REAL.CTX_KINDS) == {"S1", "S1d", "S1R", "S1dR"}
    assert "--nav-rb-npz" in REAL.DATA_FLAGS_RB and "--nav-rb-npz" not in REAL.DATA_FLAGS


def test_binding_and_token_run_for_an_r_kind(rig, tmp_path):
    """The binding reads the nav_rb data file and its builder report; a token minted for N1R is refused under another kind; a changed channel table is refused."""
    inp = os.path.dirname(rig["nav_report"])
    rb = os.path.join(inp, "navtext_nav_rb.npz")
    open(rb, "wb").write(b"rbbytes")
    rep = {sp: {"G-CLOCK-RB_pass": True, "G-CLOCK-RB_regression_arm_shift_5": {"pass": False}} for sp in ("train", "eval")}
    rep["navtext_nav_rb_sha256"] = REAL.sha_file(rb)
    rbr = os.path.join(inp, "navtext_nav_rb_report.json")
    json.dump(rep, open(rbr, "w", encoding="utf-8"))
    toks = list(rig["toks"]) + ["--nav-rb", "on", "--nav-rb-npz", rb, "--nav-rb-sha256", REAL.sha_file(rb), "--nav-rb-variant", "real", "--nav-rb-lr-mult", "10"]
    b, bad = REAL.binding("N1R", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)
    assert bad == [] and b["nav_rb_report_sha256"] == REAL.sha_file(rbr) and "--nav-rb-npz" in b["data"]
    assert REAL.binding("N1R", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"])[1]            # no report named for an R kind
    assert "nav_rb_report_sha256" not in REAL.binding("N1", rig["toks"], rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"])[0]
    rd = os.path.join(rig["a"].runs, "N1R_s0")
    os.makedirs(os.path.join(rd, "gate"))
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    sc = os.path.join(rd, "gate", "selfcheck.json")
    open(sc, "w", encoding="utf-8").write('{"declared_vs_built": "PASS"}')
    tp = os.path.join(rd, "gate", "gate_token.json")
    REAL.write_token(tp, b, {c: {"pass": True} for c in REAL.REQUIRED_CHECKS}, REAL.sha_file(sc))
    assert REAL.token_problems(tp, "N1R", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr) == []
    assert REAL.token_problems(tp, "N1Z", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)      # the kind is bound
    open(rb, "wb").write(b"rbbytes2")
    assert REAL.token_problems(tp, "N1R", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)      # a changed channel table is refused


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
    "drop_zero_control_clause": ('    if pr.get("control_e_nav_rb_grad_zero_channel") != 0.0:', '    if False:'),
    "drop_step0_identity": ('    if pr.get("step0_identity") is not True:', '    if False:'),
    "drop_grad_clause": ('    if not isinstance(g, (int, float)) or not g > 0.0:', '    if False:'),
    "drop_sha_clause": ('    if rec.get("sha256") != flag_value(toks, "--nav-rb-sha256") or not rec.get("sha256"):', '    if False:'),
    "drop_n1z_dead_clause": ('        if ng != 0.0 or "e_nav_rb.weight" not in (live.get("declared_dead") or []):', '        if False:'),
    "render_variant_real": ('"zero" if kind == "N1Z" else "real", "--nav-rb-lr-mult"', '"real", "--nav-rb-lr-mult"'),
    "render_drop_rb": ('rb_npz=""):\n    """The frozen argv', 'rb_npz=""):\n    rb_npz = rb_npz\n    """The frozen argv') if False else ('    return t + nav + ctx + rb\n', '    return t + nav + ctx\n'),
    "base_s1dr_is_s1": ('"S1dR": "S1d"', '"S1dR": "S1"'),
    "post_drop_weight_moved": ('        elif not bool((w != 0).any()):', '        elif False:'),
    "post_drop_nonr_refusal": ('    elif rb:\n        bad.append(f"e_nav_rb keys', '    elif False:\n        bad.append(f"e_nav_rb keys'),
    "report_drop_regression": ('        if r.get(sp, {}).get("G-CLOCK-RB_regression_arm_shift_5", {}).get("pass") is not False:', '        if False:'),
    "data_flag_drop": ('DATA_FLAGS_RB = ("--nav-rb-npz",)', 'DATA_FLAGS_RB = ()'),
}
EXPECT_RED = {"drop_zero_control_clause": "gate", "drop_step0_identity": "gate", "drop_grad_clause": "gate", "drop_sha_clause": "gate", "drop_n1z_dead_clause": "gate", "render_variant_real": "render",
              "render_drop_rb": "render", "base_s1dr_is_s1": "gate", "post_drop_weight_moved": "post", "post_drop_nonr_refusal": "post", "report_drop_regression": "report",
              "data_flag_drop": "kinds"}


def check_kinds(SC, tmp_path):
    assert "--nav-rb-npz" in SC.DATA_FLAGS_RB


CHECKS["kinds"] = check_kinds


@pytest.mark.parametrize("mname", sorted(MUTATIONS))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(Exception):
        CHECKS[EXPECT_RED[mname]](m, tmp_path)
