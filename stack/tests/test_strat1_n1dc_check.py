"""PREREG_V8F_N1DC: the arm kind N1DC (base N1, the 15-column channel) in ``stack/scripts/strat1_check.py``. No model run for the gate reads.

Asserted: the C kind renders N1D's argv byte for byte and APPENDS exactly ``--o-code-skip on``; every other kind's argv (C0, N1, S1, S1d, N1R, S1R, S1dR, N1Z, N1D, N1Dx, N1DW4, N1DW2) is the 134f5da1 checker's, byte for byte,
at every seed (no existing kind moves); the gate read accepts a C selfcheck and refuses each missing or false control (flag named once as on, built lever, the probe's gradients / identity / exact-0.0 controls, the OCODE group);
a non-C kind refuses the flag, the lever and the probe; the pairing rule (N1DC argv == the banked N1D argv + the flag + its own --out, nothing else); the post check wants a moved [n_q * d, 60] o_code.weight; hygiene refuses a
non-bool flag. Each new branch has a MUTATION: a copy of the checker with that branch dropped, which one of the checks must turn RED (K-MUT e: the gate / checker refuses when the flag's entry is removed)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
import strat1_check as REAL  # noqa: E402
from test_strat1_s1d import _base  # noqa: E402
from test_strat1_navrb2_check import _edit  # noqa: E402
from test_strat1_navrbd_check import _render, _toks as _toks_d, _sc as _sc_d  # noqa: E402

assert os.path.realpath(REAL.__file__).startswith(os.path.realpath(os.path.join(HERE, ".."))), REAL.__file__       # the study tree's checker
SNAP = os.path.join(HERE, "strat1_check_tip_snapshot_n1dc.txt")
SNAP_GIT_BLOB = "884e9c3eef5b59107a7751cb858205761180abee"            # git blob of stack/scripts/strat1_check.py at 134f5da1
OLD_KINDS = ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z", "N1D", "N1Dx", "N1DW4", "N1DW2")
SEEDS = ("s0", "s1", "s10", "s11")


def _tip():
    b = open(SNAP, "rb").read()
    assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == SNAP_GIT_BLOB
    mod = types.ModuleType("strat1_check_tip_n1dc")
    mod.__file__ = os.path.join(SCRIPTS, "strat1_check.py")
    exec(compile(b.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def _ctoks(SC, tmp_path, kind="N1DC"):
    """The C kind's argv as the gate sees it: the render (N1D's + the flag) + the OBD recipe's --detach-upper and --steps."""
    t = _render(SC, kind, "s0", tmp_path, "/rbd.npz") + ["--detach-upper", "--steps", str(SC.RUN_STEPS)]
    t[t.index("--nav-rb-sha256") + 1] = "cd" * 32
    return t


def _csc(SC):
    """A minimal selfcheck.json the gate read accepts for the C kind: N1D's, plus the built lever, the OCODE group and the probe's record."""
    sc = _sc_d(SC, "N1D")
    sc["levers"]["o_code_skip"] = "on"
    sc.setdefault("live", {}).setdefault("groups_grad", {})["OCODE"] = 0.3
    sc["o_code_probe"] = {"pass": True, "grad_total": 0.4, "grad_ade": 0.3, "plan_move_when_weight_set": 0.01, "control_tside_grad": 0.0, "control_repeat_grad_diff": 0.0, "control_prior_grad": 0.0,
                          "control_ade_grad_on_upper": 0.0, "identity_terms_differing": [], "identity_outputs_differing": [], "control_real_init_grad": 0.0}
    return sc


def check_render(SC, tmp_path):
    for seed in SEEDS:
        d = _render(SC, "N1D", seed, tmp_path, "/rbd.npz")
        c = _render(SC, "N1DC", seed, tmp_path, "/rbd.npz")
        assert c == d + ["--o-code-skip", "on"], seed
        assert SC.flag_value(c, "--nav-rb-feat") == "exitdist" and SC.flag_value(c, "--nav-rb-npz") == "/rbd.npz" and SC.flag_value(c, "--nav-rb-variant") == "real"
        for k in OLD_KINDS:
            assert "--o-code-skip" not in _render(SC, k, seed, tmp_path, "/rbd.npz" if k in SC.D_KINDS + SC.W_KINDS else "/rb.npz"), (k, seed)
    assert SC.rb_names("N1DC") == ("navtext_nav_rbd.npz", "navtext_nav_rbd_report.json")


def check_gate(SC, tmp_path):
    t, sc = _ctoks(SC, tmp_path), _csc(SC)
    assert SC.gate_problems("N1DC", t, sc) == []
    assert SC.gate_problems("N1DC", t[:t.index("--o-code-skip")] + t[t.index("--o-code-skip") + 2:], sc), "the flag not named"
    assert SC.gate_problems("N1DC", t + ["--o-code-skip", "on"], sc), "named twice"
    e = list(t); e[e.index("--o-code-skip") + 1] = "off"
    assert SC.gate_problems("N1DC", e, sc), "named as off"
    for path, val in ((("levers", "o_code_skip"), "off"), (("o_code_probe", "pass"), False), (("o_code_probe", "grad_total"), 0.0), (("o_code_probe", "grad_ade"), 0.0),
                      (("o_code_probe", "plan_move_when_weight_set"), 0.0), (("o_code_probe", "control_tside_grad"), 1e-9), (("o_code_probe", "control_repeat_grad_diff"), 1e-9),
                      (("o_code_probe", "control_prior_grad"), 1e-9), (("o_code_probe", "control_ade_grad_on_upper"), 1e-9), (("o_code_probe", "control_ade_grad_on_upper"), None),
                      (("o_code_probe", "identity_terms_differing"), ["ade"]), (("o_code_probe", "identity_outputs_differing"), ["traj30"]), (("live", "groups_grad", "OCODE"), 0.0)):
        assert SC.gate_problems("N1DC", t, _edit(sc, path, val)), path
    for key in ("levers", "o_code_probe"):
        d = json.loads(json.dumps(sc))
        if key == "levers":
            del d["levers"]["o_code_skip"]
        else:
            del d["o_code_probe"]
        assert SC.gate_problems("N1DC", t, d), key
    d = json.loads(json.dumps(sc)); del d["live"]["groups_grad"]["OCODE"]
    assert SC.gate_problems("N1DC", t, d), "no OCODE group"
    for k in ("N1D", "N1R", "N1DW4"):
        assert SC.gate_problems(k, t, sc), f"a {k} arm refuses the flag, the lever and the probe"
    td, sd = _toks_d(SC, "N1D", tmp_path), _sc_d(SC, "N1D")
    assert SC.gate_problems("N1D", td, sd) == [], "N1D itself is unchanged"
    assert SC.gate_problems("N1D", td + ["--o-code-skip", "on"], sd), "an N1D argv naming the flag"
    assert SC.gate_problems("N1D", td, _edit(sd, ("levers", "o_code_skip"), "on")), "an N1D selfcheck that built the skip"
    assert SC.gate_problems("N1DC", td, _csc(SC)), "the skip-less argv under the C kind"


def check_pair(SC, tmp_path):
    ref = _render(SC, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(SC, "N1DC", "s0", tmp_path, "/rbd.npz")
    w[w.index("--out") + 1] = "/other/out"
    assert SC.c_pairing_problems("N1DC", w, ref) == []
    assert SC.c_pairing_problems("N1DC", ref, ref), "no flag"
    assert SC.c_pairing_problems("N1DC", w + ["--o-code-skip", "on"], ref), "named twice"
    e = list(w); e[e.index("--o-code-skip") + 1] = "off"
    assert SC.c_pairing_problems("N1DC", e, ref), "named as off"
    for flag, val in (("--seed", "9"), ("--nav-rb-lr-mult", "30"), ("--nav-text-npz", "/another.npz"), ("--nav-rb-sha256", "ef" * 32), ("--nav-rb-npz", "/rb.npz"), ("--steps", "4000"), ("--lr", "2e-4"),
                      ("--nav-rb-variant", "shuffle"), ("--dagger-p", "0.2")):
        e = list(w)
        if flag in e:
            e[e.index(flag) + 1] = val
            assert SC.c_pairing_problems("N1DC", e, ref), flag
    assert SC.c_pairing_problems("N1DC", w + ["--extra", "1"], ref) and SC.c_pairing_problems("N1DC", w, ref + ["--extra", "1"])
    assert SC.c_pairing_problems("N1DC", w + ["--rb-window-w", "4"], ref), "an argv that also names the weight"
    assert SC.c_pairing_problems("N1DC", w, ref + ["--rb-window-w", "4"]), "a reference that names the weight"
    assert SC.c_pairing_problems("N1DC", w, [x for x in ref if x not in ("--nav-rb-feat", "exitdist")]), "the reference is not an N1D argv"
    assert SC.c_pairing_problems("N1DC", w, w), "the reference names the flag"
    assert SC.c_pairing_problems("N1D", ref, ref), "not a C kind"


def check_post(SC, tmp_path):
    import torch
    init = {"read_s.w": torch.zeros(2), "head_s.w": torch.zeros(2), "adapter.w": torch.zeros(2), "read_t.w": torch.zeros(2), "read_o.w": torch.zeros(2), "head_o.w": torch.zeros(2)}
    fin = {k: v + 1.0 for k, v in init.items()}
    fin["e_nav_rb.weight"] = torch.ones(4, 15)
    fin["head_o.0.weight"] = torch.ones(8, 64)
    assert SC.moved_identity(init, {**fin, "o_code.weight": torch.ones(64, 60)}, "N1DC") == []
    assert SC.moved_identity(init, fin, "N1DC"), "a C checkpoint without o_code.weight"
    assert SC.moved_identity(init, {**fin, "o_code.weight": torch.zeros(64, 60)}, "N1DC"), "a weight that did not move"
    assert SC.moved_identity(init, {**fin, "o_code.weight": torch.ones(64, 59)}, "N1DC"), "a wrong input width"
    assert SC.moved_identity(init, {**fin, "o_code.weight": torch.ones(63, 60)}, "N1DC"), "a width that is not head_o's"
    assert SC.moved_identity(init, {**fin, "o_code.weight": torch.ones(64, 60)}, "N1D"), "o_code keys in an N1D checkpoint"
    assert SC.moved_identity(init, fin, "N1D") == [], "N1D without o_code"


def check_hyg(SC, tmp_path):
    assert SC.hyg_c() == [] and SC.hyg_w() == [] and SC.hyg() == [] and SC.hyg_rb() == []


CHECKS = {"render": check_render, "gate": check_gate, "pair": check_pair, "post": check_post, "hyg": check_hyg}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_n1dc_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", OLD_KINDS)
def test_every_existing_kind_renders_the_134f5da1_argv_byte_for_byte(kind, seed, tmp_path):
    tip = _tip()
    rb = "/rbd.npz" if kind in tip.D_KINDS + tip.W_KINDS else "/rb.npz"
    old = tip.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    new = REAL.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    assert new == old and REAL.argv_sha(new) == tip.argv_sha(old)


def test_kinds_bases_and_the_snapshot_tip():
    assert REAL.KINDS == OLD_KINDS + ("N1DC", "N1DH") and REAL.C_KINDS == ("N1DC",) and REAL.C_FLAG == "--o-code-skip"
    assert REAL.RB_BASE["N1DC"] == "N1" and "N1DC" in REAL.RB_KINDS and "N1DC" in REAL.NAV_KINDS and "N1DC" not in REAL.CTX_KINDS and "N1DC" not in REAL.W_KINDS and "N1DC" not in REAL.D_KINDS
    assert REAL.base_kind("N1DC") == "N1" and REAL.DCHAN_KINDS == REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS + REAL.H_KINDS and "--o-code-skip" not in REAL.RB_FLAGS
    assert _tip().KINDS == OLD_KINDS and REAL.arm_tag("N1DC", "s0", REAL.RUN_STEPS) == "N1DC_s0"


def test_the_cli_pair_subcommand_takes_the_c_kind(tmp_path):
    ref = _render(REAL, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(REAL, "N1DC", "s0", tmp_path, "/rbd.npz")
    a, b = tmp_path / "w.json", tmp_path / "ref.json"
    a.write_text(json.dumps(w), encoding="utf-8"); b.write_text(json.dumps(ref), encoding="utf-8")
    assert REAL.main(["pair", "N1DC", "--argv-json", str(a), "--ref-argv-json", str(b)]) == 0
    assert REAL.main(["pair", "N1DC", "--argv-json", str(b), "--ref-argv-json", str(b)]) == 1


def _mutant(tmp_path, name, old, new):
    src = open(os.path.join(SCRIPTS, "strat1_check.py"), "rb").read().decode("utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    d = tmp_path / "mut" / name
    d.mkdir(parents=True)
    (d / "strat1_check.py").write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("strat1_mutc_" + name, str(d / "strat1_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MUTATIONS = {
    "render_without_the_flag": ('        rb += [C_FLAG, "on"]', '        rb += []'),
    "render_flag_off": ('        rb += [C_FLAG, "on"]', '        rb += [C_FLAG, "off"]'),
    "gate_does_not_call_the_c_read": ('    bad += c_gate_problems(kind, toks, sc) ', '    bad += [] '),
    "gate_drop_built_lever_clause": ('    if lv.get("o_code_skip") != "on":', '    if False:'),
    "gate_drop_named_once_clause": ('    if toks.count(C_FLAG) != 1 or flag_value(toks, C_FLAG) != "on":\n        bad.append(f"{C_FLAG}: arm', '    if False:\n        bad.append(f"{C_FLAG}: arm'),
    "gate_drop_exact_zero_controls": ('    for key in ("control_tside_grad", "control_repeat_grad_diff", "control_prior_grad", "control_ade_grad_on_upper"):', '    for key in ():'),
    "gate_drop_identity_clause": ('    for key in ("identity_terms_differing", "identity_outputs_differing"):', '    for key in ():'),
    "gate_drop_ocode_group_clause": ('    if not (live.get("groups_grad") or {}).get("OCODE", 0.0) > 0.0:', '    if False:'),
    "gate_non_c_kind_may_name_the_flag": ('        if C_FLAG in toks:\n            bad.append(f"a {kind} argv names {C_FLAG}")', '        if False:\n            bad.append(f"a {kind} argv names {C_FLAG}")'),
    "pair_drop_diff_clause": ('    if diff:\n        bad.append(f"the {kind} argv departs from N1D\'s beyond the skip flag', '    if False:\n        bad.append(f"the {kind} argv departs from N1D\'s beyond the skip flag'),
    "pair_drop_value_clause": ('    if toks.count(C_FLAG) != 1 or flag_value(toks, C_FLAG) != "on":\n        return bad', '    if toks.count(C_FLAG) != 1:\n        return bad'),
    "post_drop_moved_clause": ('            if not bool((w != 0).any()):\n                bad.append(f"{kind}: o_code.weight did not move', '            if False:\n                bad.append(f"{kind}: o_code.weight did not move'),
    "names_c_use_r_files": ('    return (RBD_NPZ, RBD_REPORT) if kind in DCHAN_KINDS else', '    return (RBD_NPZ, RBD_REPORT) if kind in D_KINDS + W_KINDS else'),
}
EXPECT_RED = {"render_without_the_flag": "render", "render_flag_off": "render", "gate_does_not_call_the_c_read": "gate", "gate_drop_built_lever_clause": "gate", "gate_drop_named_once_clause": "gate",
              "gate_drop_exact_zero_controls": "gate", "gate_drop_identity_clause": "gate", "gate_drop_ocode_group_clause": "gate", "gate_non_c_kind_may_name_the_flag": "gate",
              "pair_drop_diff_clause": "pair", "pair_drop_value_clause": "pair", "post_drop_moved_clause": "post", "names_c_use_r_files": "render"}


@pytest.mark.parametrize("mname", sorted(EXPECT_RED))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(AssertionError):                                                  # an assertion of the NAMED check, not a crash of the mutant
        CHECKS[EXPECT_RED[mname]](m, tmp_path)


def test_the_config_refuses_a_non_bool_skip_and_unknown_keys():
    from tanitad.models import v8f_mini as M
    for bad in ("on", 1, None):
        with pytest.raises(ValueError):
            M.V8FMiniConfig(o_code_skip=bad)
    with pytest.raises(TypeError):
        M.V8FMiniConfig(o_code_skipz=True)
