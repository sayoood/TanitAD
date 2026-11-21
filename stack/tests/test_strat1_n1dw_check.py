"""PREREG_V8F_N1DW: the arm kinds N1DW4 / N1DW2 (base N1, the 15-column channel) in ``stack/scripts/strat1_check.py``. No model run for the gate reads.

Asserted: a W kind renders N1D's argv byte for byte and APPENDS exactly ``--rb-window-w <w>`` (4 / 2); every other kind's argv (C0, N1, S1, S1d, N1R, S1R, S1dR, N1Z, N1D, N1Dx) is the 0c3b374f checker's, byte for
byte, at every seed (no existing kind moves); the gate read accepts a W selfcheck and refuses each missing or false control (flag named once with the kind's value, built lever, the probe's term / head gradient / exact-0.0
controls); a non-W kind refuses the flag and the probe; the pairing rule (N1DW argv == the banked N1D argv + the flag + its own --out, nothing else); the post check wants a [d, 15] weight; hygiene refuses a bad weight.
Each new branch has a MUTATION: a copy of the checker with that branch dropped, which one of the checks must turn RED."""
from __future__ import annotations

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
SNAP = os.path.join(HERE, "strat1_check_tip_snapshot_n1dw.txt")
SNAP_GIT_BLOB = "4cb59d02cd5e79850a359c693663450c7ebc22ad"            # git blob of stack/scripts/strat1_check.py at 0c3b374f
OLD_KINDS = ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z", "N1D", "N1Dx")
SEEDS = ("s0", "s1", "s10", "s11")


def _tip():
    import hashlib
    b = open(SNAP, "rb").read()
    assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == SNAP_GIT_BLOB
    mod = types.ModuleType("strat1_check_tip_n1dw")
    mod.__file__ = os.path.join(SCRIPTS, "strat1_check.py")
    exec(compile(b.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def _wtoks(SC, kind, tmp_path):
    """The W kind's argv as the gate sees it: the render (N1D's + the flag) + the OBD recipe's --detach-upper and --steps."""
    t = _render(SC, kind, "s0", tmp_path, "/rbd.npz") + ["--detach-upper", "--steps", str(SC.RUN_STEPS)]
    t[t.index("--nav-rb-sha256") + 1] = "cd" * 32
    return t


def _wsc(SC, kind):
    """A minimal selfcheck.json the gate read accepts for a W kind: N1D's, plus the built lever and the probe's record."""
    sc = _sc_d(SC, "N1D")
    w = float(SC.W_VALUE[kind])
    sc["levers"]["rb_window_w"] = w
    sc["rb_window_probe"] = {"pass": True, "rb_window_w_declared": w, "term_forced": 0.7, "term_at_unit_weight": None, "head_o_grad_diff": 0.3, "rb_n": 3, "rb_eligible": 12,
                             "control_allfalse_term": 0.0, "control_allfalse_total_diff": 0.0, "control_allfalse_grad_diff": 0.0, "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0,
                             "control_prior_grad_diff": 0.0}
    return sc


def check_render(SC, tmp_path):
    for seed in SEEDS:
        d = _render(SC, "N1D", seed, tmp_path, "/rbd.npz")
        for wk, val in (("N1DW4", "4"), ("N1DW2", "2")):
            w = _render(SC, wk, seed, tmp_path, "/rbd.npz")
            assert w == d + ["--rb-window-w", val], (wk, seed)
            assert SC.flag_value(w, "--nav-rb-feat") == "exitdist" and SC.flag_value(w, "--nav-rb-npz") == "/rbd.npz" and SC.flag_value(w, "--nav-rb-variant") == "real"
        for k in OLD_KINDS:
            assert "--rb-window-w" not in _render(SC, k, seed, tmp_path, "/rbd.npz" if k in SC.D_KINDS else "/rb.npz"), (k, seed)
    assert SC.rb_names("N1DW4") == ("navtext_nav_rbd.npz", "navtext_nav_rbd_report.json") == SC.rb_names("N1DW2")


def check_gate(SC, tmp_path):
    for wk in ("N1DW4", "N1DW2"):
        t, sc = _wtoks(SC, wk, tmp_path), _wsc(SC, wk)
        assert SC.gate_problems(wk, t, sc) == [], wk
        other = "N1DW2" if wk == "N1DW4" else "N1DW4"
        assert SC.gate_problems(wk, _wtoks(SC, other, tmp_path), sc), "the other weight's argv"
        assert SC.gate_problems(wk, t, _wsc(SC, other)), "the other weight's built lever and probe"
        assert SC.gate_problems(wk, t[:t.index("--rb-window-w")] + t[t.index("--rb-window-w") + 2:], sc), "the flag not named"
        assert SC.gate_problems(wk, t + ["--rb-window-w", SC.W_VALUE[wk]], sc), "named twice"
        for path, val in ((("levers", "rb_window_w"), 1.0), (("rb_window_probe", "pass"), False), (("rb_window_probe", "rb_window_w_declared"), 1.0), (("rb_window_probe", "term_forced"), 0.0),
                          (("rb_window_probe", "head_o_grad_diff"), 0.0), (("rb_window_probe", "control_allfalse_term"), 1e-9), (("rb_window_probe", "control_allfalse_total_diff"), 1e-9),
                          (("rb_window_probe", "control_allfalse_grad_diff"), 1e-9), (("rb_window_probe", "control_repeat_grad_diff"), 1e-9), (("rb_window_probe", "control_prior_grad_diff"), 1e-9),
                          (("rb_window_probe", "term_at_unit_weight"), 0.1)):
            assert SC.gate_problems(wk, t, _edit(sc, path, val)), path
        for key in ("levers", "rb_window_probe"):
            d = json.loads(json.dumps(sc))
            if key == "levers":
                del d["levers"]["rb_window_w"]
            else:
                del d["rb_window_probe"]
            assert SC.gate_problems(wk, t, d), key
        assert SC.gate_problems("N1D", t, sc), "an N1D arm refuses the flag, the lever and the probe"
        assert SC.gate_problems("N1R", t, sc), "an R arm too"
    td, sd = _toks_d(SC, "N1D", tmp_path), _sc_d(SC, "N1D")
    assert SC.gate_problems("N1D", td, sd) == [], "N1D itself is unchanged"
    assert SC.gate_problems("N1D", td + ["--rb-window-w", "4"], sd), "an N1D argv naming the flag"
    assert SC.gate_problems("N1D", td, _edit(sd, ("levers", "rb_window_w"), 4.0)), "an N1D selfcheck that built a weight"
    assert SC.gate_problems("N1DW4", td, _wsc(SC, "N1DW4")), "the unweighted argv under a W kind"


def check_pair(SC, tmp_path):
    ref = _render(SC, "N1D", "s0", tmp_path, "/rbd.npz")
    for wk in ("N1DW4", "N1DW2"):
        w = _render(SC, wk, "s0", tmp_path, "/rbd.npz")
        w[w.index("--out") + 1] = "/other/out"
        assert SC.w_pairing_problems(wk, w, ref) == [], wk
        assert SC.w_pairing_problems(wk, ref, ref), "no flag"
        assert SC.w_pairing_problems(wk, w + ["--rb-window-w", SC.W_VALUE[wk]], ref), "named twice"
        e = list(w); e[e.index("--rb-window-w") + 1] = "2" if wk == "N1DW4" else "4"
        assert SC.w_pairing_problems(wk, e, ref), "the other weight"
        e = list(w); e[e.index("--rb-window-w") + 1] = "3"
        assert SC.w_pairing_problems(wk, e, ref), "a weight outside the card"
        for flag, val in (("--seed", "9"), ("--nav-rb-lr-mult", "30"), ("--nav-text-npz", "/another.npz"), ("--nav-rb-sha256", "ef" * 32), ("--nav-rb-npz", "/rb.npz"), ("--steps", "4000"), ("--lr", "2e-4"),
                          ("--nav-rb-variant", "shuffle"), ("--dagger-p", "0.2")):
            e = list(w)
            if flag in e:
                e[e.index(flag) + 1] = val
                assert SC.w_pairing_problems(wk, e, ref), flag
        assert SC.w_pairing_problems(wk, w + ["--extra", "1"], ref) and SC.w_pairing_problems(wk, w, ref + ["--extra", "1"])
        assert SC.w_pairing_problems(wk, w, [x for x in ref if x not in ("--nav-rb-feat", "exitdist")]), "the reference is not an N1D argv"
    assert SC.w_pairing_problems("N1D", ref, ref), "not a W kind"
    wrong = _render(SC, "N1DW4", "s0", tmp_path, "/rbd.npz")
    assert SC.w_pairing_problems("N1DW2", wrong, ref), "an N1DW4 argv under the N1DW2 kind"
    assert SC.w_pairing_problems("N1DW4", _render(SC, "N1DW4", "s0", tmp_path, "/rbd.npz"), _render(SC, "N1DW2", "s0", tmp_path, "/rbd.npz")), "the reference names the flag"


def check_post(SC, tmp_path):
    import torch
    init = {"read_s.w": torch.zeros(2), "head_s.w": torch.zeros(2), "adapter.w": torch.zeros(2), "read_t.w": torch.zeros(2), "read_o.w": torch.zeros(2), "head_o.w": torch.zeros(2)}
    fin = {k: v + 1.0 for k, v in init.items()}
    for wk in ("N1DW4", "N1DW2"):
        assert SC.moved_identity(init, {**fin, "e_nav_rb.weight": torch.ones(4, 15)}, wk) == [], wk
        assert SC.moved_identity(init, {**fin, "e_nav_rb.weight": torch.ones(4, 1)}, wk), "a 1-column weight under a W kind"
        assert SC.moved_identity(init, {**fin, "e_nav_rb.weight": torch.zeros(4, 15)}, wk), "a weight that did not move"


def check_hyg(SC, tmp_path):
    assert SC.hyg_w() == [] and SC.hyg() == [] and SC.hyg_rb() == []


CHECKS = {"render": check_render, "gate": check_gate, "pair": check_pair, "post": check_post, "hyg": check_hyg}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_n1dw_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", OLD_KINDS)
def test_every_existing_kind_renders_the_0c3b374f_argv_byte_for_byte(kind, seed, tmp_path):
    tip = _tip()
    rb = "/rbd.npz" if kind in tip.D_KINDS else "/rb.npz"
    old = tip.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    new = REAL.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    assert new == old and REAL.argv_sha(new) == tip.argv_sha(old)


def test_kinds_bases_and_the_hygiene_of_the_snapshot_tip():
    assert REAL.KINDS == OLD_KINDS + ("N1DW4", "N1DW2", "N1DC", "N1DH", "N1DM") and REAL.W_KINDS == ("N1DW4", "N1DW2") and REAL.W_VALUE == {"N1DW4": "4", "N1DW2": "2"}
    assert REAL.RB_BASE["N1DW4"] == REAL.RB_BASE["N1DW2"] == "N1" and set(REAL.W_KINDS) <= set(REAL.RB_KINDS) <= set(REAL.NAV_KINDS) and not set(REAL.W_KINDS) & set(REAL.CTX_KINDS)
    assert REAL.base_kind("N1DW4") == "N1" and REAL.DCHAN_KINDS == REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and "--rb-window-w" not in REAL.RB_FLAGS and REAL.RB_W_FLAG == "--rb-window-w"
    assert _tip().KINDS == OLD_KINDS and REAL.arm_tag("N1DW4", "s0", REAL.RUN_STEPS) == "N1DW4_s0"


def test_the_cli_pair_subcommand_takes_a_w_kind(tmp_path):
    ref = _render(REAL, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(REAL, "N1DW4", "s0", tmp_path, "/rbd.npz")
    a, b = tmp_path / "w.json", tmp_path / "ref.json"
    a.write_text(json.dumps(w), encoding="utf-8"); b.write_text(json.dumps(ref), encoding="utf-8")
    assert REAL.main(["pair", "N1DW4", "--argv-json", str(a), "--ref-argv-json", str(b)]) == 0
    assert REAL.main(["pair", "N1DW2", "--argv-json", str(a), "--ref-argv-json", str(b)]) == 1


def _mutant(tmp_path, name, old, new):
    src = open(os.path.join(SCRIPTS, "strat1_check.py"), "rb").read().decode("utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    d = tmp_path / "mut" / name
    d.mkdir(parents=True)
    (d / "strat1_check.py").write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("strat1_mutw_" + name, str(d / "strat1_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MUTATIONS = {
    "render_without_the_weight": ('        rb += [RB_W_FLAG, W_VALUE[kind]]', '        rb += []'),
    "render_wrong_weight": ('        rb += [RB_W_FLAG, W_VALUE[kind]]', '        rb += [RB_W_FLAG, "4"]'),
    "gate_does_not_call_the_w_read": ('    bad += w_gate_problems(kind, toks, sc) ', '    bad += [] '),
    "gate_drop_built_lever_clause": ('    if lv.get("rb_window_w") != want:', '    if False:'),
    "gate_drop_named_once_clause": ('    if toks.count(RB_W_FLAG) != 1 or flag_value(toks, RB_W_FLAG) != W_VALUE[kind]:\n        bad.append(f"{RB_W_FLAG}: arm', '    if False:\n        bad.append(f"{RB_W_FLAG}: arm'),
    "gate_drop_allfalse_controls": ('    for key in ("control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_grad_diff", "control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff"):',
                                    '    for key in ():'),
    "gate_non_w_kind_may_name_the_flag": ('        if RB_W_FLAG in toks:\n            bad.append(f"a {kind} argv names {RB_W_FLAG}")', '        if False:\n            bad.append(f"a {kind} argv names {RB_W_FLAG}")'),
    "pair_drop_diff_clause": ('    if diff:\n        bad.append(f"the {kind} argv differs from N1D', '    if False:\n        bad.append(f"the {kind} argv differs from N1D'),
    "pair_drop_value_clause": ('    if toks.count(RB_W_FLAG) != 1 or flag_value(toks, RB_W_FLAG) != W_VALUE[kind]:\n        return bad', '    if toks.count(RB_W_FLAG) != 1:\n        return bad'),
    "post_drop_width_clause": ('        if w is not None and kind in DCHAN_KINDS and tuple(w.shape)[1:] != (15,):', '        if False:'),
    "names_w_use_r_files": ('    return (RBD_NPZ, RBD_REPORT) if kind in DCHAN_KINDS else', '    return (RBD_NPZ, RBD_REPORT) if kind in D_KINDS else'),
}
EXPECT_RED = {"render_without_the_weight": "render", "render_wrong_weight": "render", "gate_does_not_call_the_w_read": "gate", "gate_drop_built_lever_clause": "gate", "gate_drop_named_once_clause": "gate",
              "gate_drop_allfalse_controls": "gate", "gate_non_w_kind_may_name_the_flag": "gate", "pair_drop_diff_clause": "pair", "pair_drop_value_clause": "pair", "post_drop_width_clause": "post",
              "names_w_use_r_files": "render"}


@pytest.mark.parametrize("mname", sorted(EXPECT_RED))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(AssertionError):                                                  # an assertion of the NAMED check, not a crash of the mutant
        CHECKS[EXPECT_RED[mname]](m, tmp_path)


def test_the_config_refuses_bad_weights_and_unknown_keys():
    from tanitad.models import v8f_mini as M
    for kw in (dict(rb_window_w=0.5), dict(rb_window_w=float("nan")), dict(rb_window_w=float("inf"))):
        with pytest.raises(ValueError):
            M.V8FMiniConfig(**kw)
    with pytest.raises(TypeError):
        M.V8FMiniConfig(rb_window_wz=2.0)
