"""PREREG_V8F_N1DH (sha256 f1df05bc): the arm kind N1DH (base N1, the 15-column channel) in ``stack/scripts/strat1_check.py``. No model run for the gate reads.

Asserted: the H kind renders N1D's argv byte for byte and APPENDS exactly ``--aux-route cond --w-aux-route 0.5``; every other kind's argv (C0, N1, S1, S1d, N1R, S1R, S1dR, N1Z, N1D, N1Dx, N1DW4, N1DW2, N1DC) is the bcf3ba3f
argv at every seed (no existing kind moves); the gate read accepts an H selfcheck and refuses each missing or false control (both flags named once, the card's weight, the built levers, the probe's gradients / identity / exact-0.0
controls, the ROUTE group, the route term); a non-H kind refuses the flags, the levers and the probe; the pairing rule (N1DH argv == the banked N1D argv + the two flags + its own --out, nothing else, neither names the skip or
the weight); the post check wants moved [3, n_q * d] / [3] / [n_q * d, 3] route modules; the config hygiene (ce, weights, code mix, interface, unknown key). Each new branch has a MUTATION: a copy of the checker with that branch
dropped, which one of the checks must turn RED."""
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
SNAP = os.path.join(HERE, "strat1_check_tip_snapshot_n1dh.txt")
SNAP_GIT_BLOB = "3dda964d34db1b865194bd58a584222708f29ee4"            # git blob of stack/scripts/strat1_check.py at bcf3ba3f
OLD_KINDS = ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z", "N1D", "N1Dx", "N1DW4", "N1DW2", "N1DC")
SEEDS = ("s0", "s1", "s10", "s11")
FL = ["--aux-route", "cond", "--w-aux-route", "0.5"]


def _tip():
    b = open(SNAP, "rb").read()
    assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == SNAP_GIT_BLOB
    mod = types.ModuleType("strat1_check_tip_n1dh")
    mod.__file__ = os.path.join(SCRIPTS, "strat1_check.py")
    exec(compile(b.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def _htoks(SC, tmp_path, kind="N1DH"):
    """The H kind's argv as the gate sees it: the render (N1D's + the two flags) + the OBD recipe's --detach-upper and --steps."""
    t = _render(SC, kind, "s0", tmp_path, "/rbd.npz") + ["--detach-upper", "--steps", str(SC.RUN_STEPS)]
    t[t.index("--nav-rb-sha256") + 1] = "cd" * 32
    return t


def _hsc(SC):
    """A minimal selfcheck.json the gate read accepts for the H kind: N1D's, plus the built levers, the ROUTE group, the route term and the probe's record."""
    sc = _sc_d(SC, "N1D")
    sc["levers"]["aux_route"], sc["levers"]["w_aux_route"] = "cond", 0.5
    sc.setdefault("live", {}).setdefault("groups_grad", {})["ROUTE"] = 0.3
    sc["live"].setdefault("terms", {})["route"] = 1.1
    sc["aux_route_probe"] = {"pass": True, "head_grad_total": 0.4, "head_grad_route": 0.4, "emb_grad_total": 0.2, "emb_grad_ade": 0.2, "plan_move_when_emb_set": 0.01, "own_rows_probe": 9,
                             "control_decoder_grad_on_head": 0.0, "control_tside_grad_on_head": 0.0, "control_tside_grad_on_emb": 0.0, "control_repeat_grad_diff": 0.0, "control_prior_grad": 0.0,
                             "control_no_own_row_route_term": 0.0, "control_no_own_row_head_grad": 0.0, "identity_terms_differing": [], "identity_outputs_differing": []}
    return sc


def check_render(SC, tmp_path):
    for seed in SEEDS:
        d = _render(SC, "N1D", seed, tmp_path, "/rbd.npz")
        h = _render(SC, "N1DH", seed, tmp_path, "/rbd.npz")
        assert h == d + FL, seed
        assert SC.flag_value(h, "--nav-rb-feat") == "exitdist" and SC.flag_value(h, "--nav-rb-npz") == "/rbd.npz" and SC.flag_value(h, "--nav-rb-variant") == "real"
        assert "--o-code-skip" not in h and "--rb-window-w" not in h
        for k in OLD_KINDS:
            r = _render(SC, k, seed, tmp_path, "/rbd.npz" if k in SC.D_KINDS + SC.W_KINDS + SC.C_KINDS else "/rb.npz")
            assert "--aux-route" not in r and "--w-aux-route" not in r, (k, seed)
    assert SC.rb_names("N1DH") == ("navtext_nav_rbd.npz", "navtext_nav_rbd_report.json")


def check_gate(SC, tmp_path):
    t, sc = _htoks(SC, tmp_path), _hsc(SC)
    assert SC.gate_problems("N1DH", t, sc) == []
    for fl in ("--aux-route", "--w-aux-route"):
        i = t.index(fl)
        assert SC.gate_problems("N1DH", t[:i] + t[i + 2:], sc), f"{fl} not named"
        assert SC.gate_problems("N1DH", t + [fl, t[i + 1]], sc), f"{fl} named twice"
    e = list(t); e[e.index("--aux-route") + 1] = "off"
    assert SC.gate_problems("N1DH", e, sc), "named as off"
    e = list(t); e[e.index("--w-aux-route") + 1] = "0.25"
    assert SC.gate_problems("N1DH", e, sc), "a weight other than the card's"
    for path, val in ((("levers", "aux_route"), "off"), (("levers", "w_aux_route"), 0.25), (("aux_route_probe", "pass"), False), (("aux_route_probe", "head_grad_total"), 0.0), (("aux_route_probe", "head_grad_route"), 0.0),
                      (("aux_route_probe", "emb_grad_total"), 0.0), (("aux_route_probe", "emb_grad_ade"), 0.0), (("aux_route_probe", "plan_move_when_emb_set"), 0.0), (("aux_route_probe", "own_rows_probe"), 0),
                      (("aux_route_probe", "control_decoder_grad_on_head"), 1e-9), (("aux_route_probe", "control_decoder_grad_on_head"), None), (("aux_route_probe", "control_tside_grad_on_head"), 1e-9),
                      (("aux_route_probe", "control_tside_grad_on_emb"), 1e-9), (("aux_route_probe", "control_repeat_grad_diff"), 1e-9), (("aux_route_probe", "control_prior_grad"), 1e-9),
                      (("aux_route_probe", "control_no_own_row_route_term"), 1e-9), (("aux_route_probe", "control_no_own_row_head_grad"), 1e-9),
                      (("aux_route_probe", "identity_terms_differing"), ["ade"]), (("aux_route_probe", "identity_outputs_differing"), ["traj30"]), (("live", "groups_grad", "ROUTE"), 0.0)):
        assert SC.gate_problems("N1DH", t, _edit(sc, path, val)), path
    for key in ("aux_route", "w_aux_route"):
        d = json.loads(json.dumps(sc)); del d["levers"][key]
        assert SC.gate_problems("N1DH", t, d), key
    d = json.loads(json.dumps(sc)); del d["aux_route_probe"]
    assert SC.gate_problems("N1DH", t, d), "no probe"
    d = json.loads(json.dumps(sc)); del d["live"]["groups_grad"]["ROUTE"]
    assert SC.gate_problems("N1DH", t, d), "no ROUTE group"
    d = json.loads(json.dumps(sc)); del d["live"]["terms"]["route"]
    assert SC.gate_problems("N1DH", t, d), "no route term"
    for k in ("N1D", "N1R", "N1DW4", "N1DC"):
        assert SC.gate_problems(k, t, sc), f"a {k} arm refuses the flags, the levers and the probe"
    td, sd = _toks_d(SC, "N1D", tmp_path), _sc_d(SC, "N1D")
    assert SC.gate_problems("N1D", td, sd) == [], "N1D itself is unchanged"
    assert SC.gate_problems("N1D", td + FL, sd), "an N1D argv naming the flags"
    assert SC.gate_problems("N1D", td, _edit(sd, ("levers", "aux_route"), "cond")), "an N1D selfcheck that built the head"
    assert SC.gate_problems("N1DH", td, _hsc(SC)), "the flag-less argv under the H kind"
    assert SC.gate_problems("N1DH", t + ["--o-code-skip", "on"], sc), "an H argv that also names the skip"


def check_pair(SC, tmp_path):
    ref = _render(SC, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(SC, "N1DH", "s0", tmp_path, "/rbd.npz")
    w[w.index("--out") + 1] = "/other/out"
    assert SC.h_pairing_problems("N1DH", w, ref) == []
    assert SC.h_pairing_problems("N1DH", ref, ref), "no flags"
    for fl in ("--aux-route", "--w-aux-route"):
        i = w.index(fl)
        assert SC.h_pairing_problems("N1DH", w[:i] + w[i + 2:], ref), f"{fl} missing"
        assert SC.h_pairing_problems("N1DH", w + [fl, w[i + 1]], ref), f"{fl} twice"
    e = list(w); e[e.index("--aux-route") + 1] = "off"
    assert SC.h_pairing_problems("N1DH", e, ref), "named as off"
    e = list(w); e[e.index("--w-aux-route") + 1] = "1.0"
    assert SC.h_pairing_problems("N1DH", e, ref), "another weight"
    for flag, val in (("--seed", "9"), ("--nav-rb-lr-mult", "30"), ("--nav-text-npz", "/another.npz"), ("--nav-rb-sha256", "ef" * 32), ("--nav-rb-npz", "/rb.npz"), ("--steps", "4000"), ("--lr", "2e-4"),
                      ("--nav-rb-variant", "shuffle"), ("--dagger-p", "0.2")):
        e = list(w)
        if flag in e:
            e[e.index(flag) + 1] = val
            assert SC.h_pairing_problems("N1DH", e, ref), flag
    assert SC.h_pairing_problems("N1DH", w + ["--extra", "1"], ref) and SC.h_pairing_problems("N1DH", w, ref + ["--extra", "1"])
    assert SC.h_pairing_problems("N1DH", w + ["--rb-window-w", "4"], ref), "an argv that also names the weight"
    assert SC.h_pairing_problems("N1DH", w + ["--o-code-skip", "on"], ref), "an argv that also names the skip"
    assert SC.h_pairing_problems("N1DH", w, ref + ["--rb-window-w", "4"]), "a reference that names the weight"
    assert SC.h_pairing_problems("N1DH", w, ref + ["--o-code-skip", "on"]), "a reference that names the skip"
    assert SC.h_pairing_problems("N1DH", w, [x for x in ref if x not in ("--nav-rb-feat", "exitdist")]), "the reference is not an N1D argv"
    assert SC.h_pairing_problems("N1DH", w, w), "the reference names the flags"
    assert SC.h_pairing_problems("N1D", ref, ref), "not an H kind"


def check_post(SC, tmp_path):
    import torch
    init = {"read_s.w": torch.zeros(2), "head_s.w": torch.zeros(2), "adapter.w": torch.zeros(2), "read_t.w": torch.zeros(2), "read_o.w": torch.zeros(2), "head_o.w": torch.zeros(2)}
    fin = {k: v + 1.0 for k, v in init.items()}
    fin["e_nav_rb.weight"] = torch.ones(4, 15)
    fin["head_o.0.weight"] = torch.ones(8, 64)
    ok = {"route_head.weight": torch.ones(3, 64), "route_head.bias": torch.ones(3), "route_emb.weight": torch.ones(64, 3)}
    assert SC.moved_identity(init, {**fin, **ok}, "N1DH") == []
    assert SC.moved_identity(init, fin, "N1DH"), "an H checkpoint without the route keys"
    assert SC.moved_identity(init, {**fin, **{k: v for k, v in ok.items() if k != "route_head.bias"}}, "N1DH"), "no bias"
    assert SC.moved_identity(init, {**fin, **ok, "route_emb.weight": torch.zeros(64, 3)}, "N1DH"), "an embedding that did not move"
    assert SC.moved_identity(init, {**fin, **ok, "route_head.weight": torch.ones(4, 64)}, "N1DH"), "a wrong class count"
    assert SC.moved_identity(init, {**fin, **ok, "route_emb.weight": torch.ones(63, 3)}, "N1DH"), "a width that is not head_o's"
    assert SC.moved_identity(init, {**fin, **ok}, "N1D"), "route keys in an N1D checkpoint"
    assert SC.moved_identity(init, fin, "N1D") == [], "N1D without them"


def check_hyg(SC, tmp_path):
    assert SC.hyg_h() == [] and SC.hyg_c() == [] and SC.hyg_w() == [] and SC.hyg() == [] and SC.hyg_rb() == []


CHECKS = {"render": check_render, "gate": check_gate, "pair": check_pair, "post": check_post, "hyg": check_hyg}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_n1dh_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", OLD_KINDS)
def test_every_existing_kind_renders_the_bcf3ba3f_argv_byte_for_byte(kind, seed, tmp_path):
    tip = _tip()
    rb = "/rbd.npz" if kind in tip.D_KINDS + tip.W_KINDS + tip.C_KINDS else "/rb.npz"
    old = tip.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    new = REAL.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    assert new == old and REAL.argv_sha(new) == tip.argv_sha(old)


def test_kinds_bases_and_the_snapshot_tip():
    assert REAL.KINDS == OLD_KINDS + ("N1DH", "N1DM") and REAL.H_KINDS == ("N1DH",) and REAL.H_FLAGS == ("--aux-route", "--w-aux-route") and REAL.H_WEIGHT == "0.5"
    assert REAL.RB_BASE["N1DH"] == "N1" and "N1DH" in REAL.RB_KINDS and "N1DH" in REAL.NAV_KINDS and "N1DH" not in REAL.CTX_KINDS and "N1DH" not in REAL.W_KINDS and "N1DH" not in REAL.D_KINDS and "N1DH" not in REAL.C_KINDS
    assert REAL.base_kind("N1DH") == "N1" and REAL.DCHAN_KINDS == REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and not set(REAL.H_FLAGS) & set(REAL.RB_FLAGS)
    assert _tip().KINDS == OLD_KINDS and REAL.arm_tag("N1DH", "s0", REAL.RUN_STEPS) == "N1DH_s0"


def test_the_cli_pair_subcommand_takes_the_h_kind(tmp_path):
    ref = _render(REAL, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(REAL, "N1DH", "s0", tmp_path, "/rbd.npz")
    a, b = tmp_path / "w.json", tmp_path / "ref.json"
    a.write_text(json.dumps(w), encoding="utf-8"); b.write_text(json.dumps(ref), encoding="utf-8")
    assert REAL.main(["pair", "N1DH", "--argv-json", str(a), "--ref-argv-json", str(b)]) == 0
    assert REAL.main(["pair", "N1DH", "--argv-json", str(b), "--ref-argv-json", str(b)]) == 1


def _mutant(tmp_path, name, old, new):
    src = open(os.path.join(SCRIPTS, "strat1_check.py"), "rb").read().decode("utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    d = tmp_path / "mut" / name
    d.mkdir(parents=True)
    (d / "strat1_check.py").write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("strat1_muth_" + name, str(d / "strat1_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MUTATIONS = {
    "render_without_the_flags": ('        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]', '        rb += []'),
    "render_cond_off": ('        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]', '        rb += [H_FLAGS[0], "off", H_FLAGS[1], H_WEIGHT]'),
    "gate_does_not_call_the_h_read": ('    bad += h_gate_problems(kind, toks, sc) ', '    bad += [] '),
    "gate_drop_built_lever_clause": ('    if lv.get("aux_route") != "cond":', '    if False:'),
    "gate_drop_weight_lever_clause": ('    if lv.get("w_aux_route") != float(H_WEIGHT):', '    if False:'),
    "gate_drop_named_once_clause": ('    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != "cond":\n        bad.append', '    if False:\n        bad.append'),
    "gate_drop_weight_named_clause": ('    if toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:\n        bad.append', '    if False:\n        bad.append'),
    "gate_drop_exact_zero_controls": ('    for key in ("control_decoder_grad_on_head", "control_tside_grad_on_head",', '    for key in () and ("control_decoder_grad_on_head", "control_tside_grad_on_head",'),
    "gate_drop_identity_clause": ('    for key in ("identity_outputs_differing", "identity_terms_differing"):\n        if pr.get(key) != []:\n            bad.append(f"aux_route_probe', '    for key in ():\n        if pr.get(key) != []:\n            bad.append(f"aux_route_probe'),
    "gate_drop_route_group_clause": ('    if not (live.get("groups_grad") or {}).get("ROUTE", 0.0) > 0.0:', '    if False:'),
    "gate_drop_route_term_clause": ('    if "route" not in (live.get("terms") or {}):', '    if False:'),
    "gate_non_h_kind_may_name_the_flags": ('        for fl in H_FLAGS:\n            if fl in toks:\n                bad.append(f"a {kind} argv names {fl}")', '        for fl in ():\n            if fl in toks:\n                bad.append(f"a {kind} argv names {fl}")'),
    "pair_drop_diff_clause": ('    if diff:\n        bad.append(f"the {kind} argv departs from N1D\'s beyond the two route flags', '    if False:\n        bad.append(f"the {kind} argv departs from N1D\'s beyond the two route flags'),
    "pair_drop_value_clause": (' or flag_value(toks, H_FLAGS[0]) != "cond" or toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:\n        return bad', ' or toks.count(H_FLAGS[1]) != 1:\n        return bad'),
    "post_drop_moved_clause": ('            if not bool((we != 0).any()):\n                bad.append(f"{kind}: route_emb.weight did not move', '            if False:\n                bad.append(f"{kind}: route_emb.weight did not move'),
    "post_drop_foreign_key_clause": ('    elif rr:\n        bad.append(f"route keys', '    elif False:\n        bad.append(f"route keys'),
    "names_h_use_r_files": ('    return (RBD_NPZ, RBD_REPORT) if kind in DCHAN_KINDS else', '    return (RBD_NPZ, RBD_REPORT) if kind in D_KINDS + W_KINDS + C_KINDS else'),
}
EXPECT_RED = {"render_without_the_flags": "render", "render_cond_off": "render", "gate_does_not_call_the_h_read": "gate", "gate_drop_built_lever_clause": "gate", "gate_drop_weight_lever_clause": "gate",
              "gate_drop_named_once_clause": "gate", "gate_drop_weight_named_clause": "gate", "gate_drop_exact_zero_controls": "gate", "gate_drop_identity_clause": "gate", "gate_drop_route_group_clause": "gate",
              "gate_drop_route_term_clause": "gate", "gate_non_h_kind_may_name_the_flags": "gate", "pair_drop_diff_clause": "pair", "pair_drop_value_clause": "pair", "post_drop_moved_clause": "post",
              "post_drop_foreign_key_clause": "post", "names_h_use_r_files": "render"}


@pytest.mark.parametrize("mname", sorted(EXPECT_RED))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(AssertionError):                                                  # an assertion of the NAMED check, not a crash of the mutant
        CHECKS[EXPECT_RED[mname]](m, tmp_path)


def test_the_config_refuses_the_ce_mode_bad_weights_and_unknown_keys():
    from tanitad.models import v8f_mini as M
    for kw in (dict(aux_route="ce"), dict(aux_route="on"), dict(aux_route=None), dict(aux_route="cond", w_aux_route=-0.1), dict(aux_route="cond", w_aux_route=float("inf"))):
        with pytest.raises(ValueError):
            M.V8FMiniConfig(**kw)
    with pytest.raises(TypeError):
        M.V8FMiniConfig(aux_routez="cond")
