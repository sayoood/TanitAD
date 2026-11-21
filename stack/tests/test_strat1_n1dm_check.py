"""PREREG_V8F_N1DM (draft): the arm kind N1DM (base N1, the 15-column channel) in ``stack/scripts/strat1_check.py``. No model run for the gate reads.

Asserted: the M kind renders N1D's argv byte for byte and APPENDS exactly ``--aux-route modes --w-aux-route 0.5`` (N1DH still renders ``cond``; every other kind's argv is the 1f217bf0 argv at every seed: no existing kind moves);
the gate read accepts an M selfcheck and refuses each missing or false control (both flags named once, the weight, the built levers, the probe's gradients on route_head AND on each of the 3 branches, the permutation, the identity, the
exact-0.0 controls, the ROUTE group, the route term); an H or non-M kind is not read as M; the pairing rule (N1DM argv == the banked N1D argv + the two flags + its own --out); the post check wants moved route_head + 3 zero-start
branches and NO route_emb; the hygiene. Each new branch has a MUTATION: a copy of the checker with that branch dropped, which one of the checks must turn RED."""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
import strat1_check as REAL  # noqa: E402
from test_strat1_navrb2_check import _edit  # noqa: E402
from test_strat1_navrbd_check import _render, _toks as _toks_d, _sc as _sc_d  # noqa: E402
from test_strat1_n1dh_check import SCRIPTS, SEEDS, _hsc, _htoks, _mutant, _tip as _tip_h, OLD_KINDS as OLD_H_KINDS  # noqa: E402

assert os.path.realpath(REAL.__file__).startswith(os.path.realpath(os.path.join(HERE, ".."))), REAL.__file__
FL = ["--aux-route", "modes", "--w-aux-route", "0.5"]
FLH = ["--aux-route", "cond", "--w-aux-route", "0.5"]
KINDS_1F217BF0 = OLD_H_KINDS + ("N1DH",)


def _mtoks(SC, tmp_path):
    return _htoks(SC, tmp_path, kind="N1DM")


def _msc(SC):
    sc = _hsc(SC)
    sc["levers"]["aux_route"] = "modes"
    pr = sc["aux_route_probe"]
    pr.update({"real_init_emb_each": [0.1, 0.2, 0.3], "emb_each": [0.1, 0.2, 0.3], "plan_move_when_p_permuted": 0.02})
    return sc


def check_render(SC, tmp_path):
    for seed in SEEDS:
        d = _render(SC, "N1D", seed, tmp_path, "/rbd.npz")
        m = _render(SC, "N1DM", seed, tmp_path, "/rbd.npz")
        h = _render(SC, "N1DH", seed, tmp_path, "/rbd.npz")
        assert m == d + FL and h == d + FLH, seed
        assert SC.flag_value(m, "--nav-rb-feat") == "exitdist" and SC.flag_value(m, "--nav-rb-npz") == "/rbd.npz" and SC.flag_value(m, "--nav-rb-variant") == "real"
        assert "--o-code-skip" not in m and "--rb-window-w" not in m
    assert SC.rb_names("N1DM") == ("navtext_nav_rbd.npz", "navtext_nav_rbd_report.json")


def check_gate(SC, tmp_path):
    t, sc = _mtoks(SC, tmp_path), _msc(SC)
    assert SC.gate_problems("N1DM", t, sc) == []
    for fl in ("--aux-route", "--w-aux-route"):
        i = t.index(fl)
        assert SC.gate_problems("N1DM", t[:i] + t[i + 2:], sc), f"{fl} not named"
        assert SC.gate_problems("N1DM", t + [fl, t[i + 1]], sc), f"{fl} named twice"
    for val in ("off", "cond"):
        e = list(t); e[e.index("--aux-route") + 1] = val
        assert SC.gate_problems("N1DM", e, sc), f"named as {val}"
    e = list(t); e[e.index("--w-aux-route") + 1] = "0.25"
    assert SC.gate_problems("N1DM", e, sc), "a weight other than the card's"
    P = "aux_route_probe"
    for path, val in ((("levers", "aux_route"), "cond"), (("levers", "aux_route"), "off"), (("levers", "w_aux_route"), 0.25), ((P, "pass"), False), ((P, "head_grad_total"), 0.0), ((P, "head_grad_route"), 0.0),
                      ((P, "emb_grad_total"), 0.0), ((P, "emb_grad_ade"), 0.0), ((P, "plan_move_when_emb_set"), 0.0), ((P, "plan_move_when_p_permuted"), 0.0), ((P, "plan_move_when_p_permuted"), None),
                      ((P, "own_rows_probe"), 0), ((P, "real_init_emb_each"), [0.1, 0.0, 0.3]), ((P, "real_init_emb_each"), [0.1, 0.2]), ((P, "emb_each"), [0.0, 0.2, 0.3]), ((P, "emb_each"), None),
                      ((P, "control_decoder_grad_on_head"), 1e-9), ((P, "control_decoder_grad_on_head"), None), ((P, "control_tside_grad_on_head"), 1e-9), ((P, "control_tside_grad_on_emb"), 1e-9),
                      ((P, "control_repeat_grad_diff"), 1e-9), ((P, "control_prior_grad"), 1e-9), ((P, "control_no_own_row_route_term"), 1e-9), ((P, "control_no_own_row_head_grad"), 1e-9),
                      ((P, "identity_terms_differing"), ["ade"]), ((P, "identity_outputs_differing"), ["traj30"]), (("live", "groups_grad", "ROUTE"), 0.0)):
        assert SC.gate_problems("N1DM", t, _edit(sc, path, val)), path
    for key in ("aux_route", "w_aux_route"):
        d = json.loads(json.dumps(sc)); del d["levers"][key]
        assert SC.gate_problems("N1DM", t, d), key
    d = json.loads(json.dumps(sc)); del d[P]
    assert SC.gate_problems("N1DM", t, d), "no probe"
    d = json.loads(json.dumps(sc)); del d["live"]["groups_grad"]["ROUTE"]
    assert SC.gate_problems("N1DM", t, d), "no ROUTE group"
    d = json.loads(json.dumps(sc)); del d["live"]["terms"]["route"]
    assert SC.gate_problems("N1DM", t, d), "no route term"
    # the other kinds: an H kind is not an M kind and an M kind is not an H kind, and neither reads the other's selfcheck
    th, sh = _htoks(SC, tmp_path), _hsc(SC)
    assert SC.gate_problems("N1DH", th, sh) == [], "N1DH itself is unchanged"
    assert SC.gate_problems("N1DM", th, sc), "an M arm with the cond argv"
    assert SC.gate_problems("N1DH", t, sh), "an H arm with the modes argv"
    assert SC.gate_problems("N1DH", th, sc), "an H arm whose selfcheck built modes"
    assert SC.gate_problems("N1DM", t, sh), "an M arm whose selfcheck built cond"
    for k in ("N1D", "N1R", "N1DW4", "N1DC"):
        assert SC.gate_problems(k, t, sc), f"a {k} arm refuses the modes flags, the levers and the probe"
    td, sd = _toks_d(SC, "N1D", tmp_path), _sc_d(SC, "N1D")
    assert SC.gate_problems("N1D", td, sd) == [], "N1D itself is unchanged"
    assert SC.gate_problems("N1D", td + FL, sd), "an N1D argv naming the modes flags"
    assert SC.gate_problems("N1D", td, _edit(sd, ("levers", "aux_route"), "modes")), "an N1D selfcheck that built the branches"
    assert SC.gate_problems("N1DM", td, _msc(SC)), "the flag-less argv under the M kind"
    assert SC.gate_problems("N1DM", t + ["--o-code-skip", "on"], sc), "an M argv that also names the skip"


def check_pair(SC, tmp_path):
    ref = _render(SC, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(SC, "N1DM", "s0", tmp_path, "/rbd.npz")
    w[w.index("--out") + 1] = "/other/out"
    assert SC.m_pairing_problems("N1DM", w, ref) == []
    assert SC.m_pairing_problems("N1DM", ref, ref), "no flags"
    for fl in ("--aux-route", "--w-aux-route"):
        i = w.index(fl)
        assert SC.m_pairing_problems("N1DM", w[:i] + w[i + 2:], ref), f"{fl} missing"
        assert SC.m_pairing_problems("N1DM", w + [fl, w[i + 1]], ref), f"{fl} twice"
    for val in ("off", "cond"):
        e = list(w); e[e.index("--aux-route") + 1] = val
        assert SC.m_pairing_problems("N1DM", e, ref), f"named as {val}"
    e = list(w); e[e.index("--w-aux-route") + 1] = "1.0"
    assert SC.m_pairing_problems("N1DM", e, ref), "another weight"
    for flag, val in (("--seed", "9"), ("--nav-rb-lr-mult", "30"), ("--nav-text-npz", "/another.npz"), ("--nav-rb-sha256", "ef" * 32), ("--nav-rb-npz", "/rb.npz"), ("--steps", "4000"), ("--lr", "2e-4"),
                      ("--nav-rb-variant", "shuffle"), ("--dagger-p", "0.2")):
        e = list(w)
        if flag in e:
            e[e.index(flag) + 1] = val
            assert SC.m_pairing_problems("N1DM", e, ref), flag
    assert SC.m_pairing_problems("N1DM", w + ["--extra", "1"], ref) and SC.m_pairing_problems("N1DM", w, ref + ["--extra", "1"])
    assert SC.m_pairing_problems("N1DM", w + ["--rb-window-w", "4"], ref) and SC.m_pairing_problems("N1DM", w + ["--o-code-skip", "on"], ref)
    assert SC.m_pairing_problems("N1DM", w, ref + ["--rb-window-w", "4"]) and SC.m_pairing_problems("N1DM", w, ref + ["--o-code-skip", "on"])
    assert SC.m_pairing_problems("N1DM", w, ref + FLH), "the reference is an N1DH argv, not an N1D argv"
    assert SC.m_pairing_problems("N1DM", w, [x for x in ref if x not in ("--nav-rb-feat", "exitdist")]), "the reference is not an N1D argv"
    assert SC.m_pairing_problems("N1DM", w, w), "the reference names the flags"
    assert SC.m_pairing_problems("N1D", ref, ref), "not an M kind"
    wh = _render(SC, "N1DH", "s0", tmp_path, "/rbd.npz")
    assert SC.m_pairing_problems("N1DM", wh, ref), "an N1DH argv labelled N1DM"
    assert SC.h_pairing_problems("N1DH", w, ref), "an N1DM argv labelled N1DH"


def check_post(SC, tmp_path):
    import torch
    init = {"read_s.w": torch.zeros(2), "head_s.w": torch.zeros(2), "adapter.w": torch.zeros(2), "read_t.w": torch.zeros(2), "read_o.w": torch.zeros(2), "head_o.w": torch.zeros(2)}
    fin = {k: v + 1.0 for k, v in init.items()}
    fin["e_nav_rb.weight"] = torch.ones(4, 15)
    fin["head_o.0.weight"] = torch.ones(8, 64)
    fin["head_o.2.weight"] = torch.ones(12, 8)
    ok = {"route_head.weight": torch.ones(3, 64), "route_head.bias": torch.ones(3), **{f"route_br.{k}.weight": torch.ones(12, 64) for k in range(3)}}
    assert SC.moved_identity(init, {**fin, **ok}, "N1DM") == []
    assert SC.moved_identity(init, fin, "N1DM"), "an M checkpoint without the route keys"
    assert SC.moved_identity(init, {**fin, **{k: v for k, v in ok.items() if k != "route_head.bias"}}, "N1DM"), "no bias"
    assert SC.moved_identity(init, {**fin, **{k: v for k, v in ok.items() if k != "route_br.2.weight"}}, "N1DM"), "a missing branch"
    assert SC.moved_identity(init, {**fin, **ok, "route_br.1.weight": torch.zeros(12, 64)}, "N1DM"), "a branch that did not move"
    assert SC.moved_identity(init, {**fin, **ok, "route_head.weight": torch.ones(4, 64)}, "N1DM"), "a wrong class count"
    assert SC.moved_identity(init, {**fin, **ok, "route_br.0.weight": torch.ones(11, 64)}, "N1DM"), "a branch width that is not head_o's output"
    assert SC.moved_identity(init, {**fin, **ok, "route_br.0.weight": torch.ones(12, 63)}, "N1DM"), "a branch input that is not the route head's"
    assert SC.moved_identity(init, {**fin, **ok, "route_emb.weight": torch.ones(64, 3)}, "N1DM"), "a route_emb in an M checkpoint"
    assert SC.moved_identity(init, {**fin, **ok}, "N1D"), "route keys in an N1D checkpoint"
    okh = {"route_head.weight": torch.ones(3, 64), "route_head.bias": torch.ones(3), "route_emb.weight": torch.ones(64, 3)}
    assert SC.moved_identity(init, {**fin, **okh, "route_br.0.weight": torch.ones(12, 64)}, "N1DH"), "branches in an otherwise valid H checkpoint"
    assert SC.moved_identity(init, {**fin, **okh}, "N1DH") == [], "the H check is unchanged"
    assert SC.moved_identity(init, {**fin, **okh}, "N1DM"), "an H checkpoint labelled N1DM"


def check_hyg(SC, tmp_path):
    assert SC.hyg_m() == [] and SC.hyg_h() == [] and SC.hyg_c() == [] and SC.hyg_w() == [] and SC.hyg() == [] and SC.hyg_rb() == []


CHECKS = {"render": check_render, "gate": check_gate, "pair": check_pair, "post": check_post, "hyg": check_hyg}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_n1dm_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", KINDS_1F217BF0)
def test_every_existing_kind_renders_the_1f217bf0_argv_byte_for_byte(kind, seed, tmp_path):
    import hashlib
    from test_strat1_s1d import _base
    snap = os.path.join(HERE, "strat1_check_tip_snapshot_n1dm.txt")
    b = open(snap, "rb").read()
    assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == "97ce044806aa5fb3c94afafaf0e491aab31dc3d4", "git blob of strat1_check.py at 1f217bf0"
    import types
    tip = types.ModuleType("strat1_check_tip_n1dm")
    tip.__file__ = os.path.join(SCRIPTS, "strat1_check.py")
    exec(compile(b.decode("utf-8"), tip.__file__, "exec"), tip.__dict__)
    rb = "/rbd.npz" if kind in tip.D_KINDS + tip.W_KINDS + tip.C_KINDS + tip.H_KINDS else "/rb.npz"
    old = tip.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    new = REAL.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)
    assert new == old and REAL.argv_sha(new) == tip.argv_sha(old)


def test_kinds_bases_and_the_cli():
    assert REAL.KINDS == KINDS_1F217BF0 + ("N1DM",) and REAL.M_KINDS == ("N1DM",) and REAL.M_MODE == "modes" and REAL.H_FLAGS == ("--aux-route", "--w-aux-route") and REAL.H_WEIGHT == "0.5"
    assert REAL.RB_BASE["N1DM"] == "N1" and "N1DM" in REAL.RB_KINDS and "N1DM" in REAL.NAV_KINDS and not {"N1DM"} & set(REAL.CTX_KINDS + REAL.W_KINDS + REAL.D_KINDS + REAL.C_KINDS + REAL.H_KINDS)
    assert REAL.base_kind("N1DM") == "N1" and REAL.DCHAN_KINDS == REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS + REAL.H_KINDS + REAL.M_KINDS and REAL.arm_tag("N1DM", "s0", REAL.RUN_STEPS) == "N1DM_s0"


def test_the_cli_pair_subcommand_takes_the_m_kind(tmp_path):
    ref = _render(REAL, "N1D", "s0", tmp_path, "/rbd.npz")
    w = _render(REAL, "N1DM", "s0", tmp_path, "/rbd.npz")
    a, b = tmp_path / "w.json", tmp_path / "ref.json"
    a.write_text(json.dumps(w), encoding="utf-8"); b.write_text(json.dumps(ref), encoding="utf-8")
    assert REAL.main(["pair", "N1DM", "--argv-json", str(a), "--ref-argv-json", str(b)]) == 0
    assert REAL.main(["pair", "N1DM", "--argv-json", str(b), "--ref-argv-json", str(b)]) == 1


MUTATIONS = {
    "render_without_the_flags": ('        rb += [H_FLAGS[0], M_MODE, H_FLAGS[1], H_WEIGHT]', '        rb += []'),
    "render_modes_as_cond": ('        rb += [H_FLAGS[0], M_MODE, H_FLAGS[1], H_WEIGHT]', '        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]'),
    "gate_does_not_call_the_m_read": ('    bad += m_gate_problems(kind, toks, sc) ', '    bad += [] '),
    "gate_drop_built_lever_clause": ('    if lv.get("aux_route") != M_MODE:', '    if False:'),
    "gate_drop_weight_lever_clause": ('    if float(lv.get("w_aux_route", -1.0)) != float(H_WEIGHT):', '    if False:'),
    "gate_drop_named_once_clause": ('    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != M_MODE:\n        bad.append', '    if False:\n        bad.append'),
    "gate_drop_weight_named_clause": ('    if flag_value(toks, H_FLAGS[1]) != H_WEIGHT or toks.count(H_FLAGS[1]) != 1:\n        bad.append', '    if False:\n        bad.append'),
    "gate_drop_each_branch_clause": ('    for key in ("real_init_emb_each", "emb_each"):', '    for key in ():'),
    "gate_drop_permutation_clause": ('"plan_move_when_emb_set", "plan_move_when_p_permuted", "own_rows_probe"):', '"plan_move_when_emb_set", "own_rows_probe"):'),
    "gate_drop_exact_zero_controls": ('    for key in zero_keys:', '    for key in ():'),
    "gate_drop_identity_clause": ('    for key in ident_keys:', '    for key in ():'),
    "gate_drop_route_group_clause": ('    if not grp > 0.0:', '    if False:'),
    "gate_drop_route_term_clause": ('    if "route" not in set(live.get("terms") or {}):', '    if False:'),
    "h_gate_reads_the_m_kind": ('    if kind in M_KINDS:\n        return bad ', '    if False:\n        return bad '),
    "pair_drop_diff_clause": ('    if len(diff):\n        bad.append(f"the {kind} argv departs', '    if False:\n        bad.append(f"the {kind} argv departs'),
    "pair_drop_value_clause": (' or flag_value(toks, H_FLAGS[0]) != M_MODE or toks.count(H_FLAGS[1]) != 1', ' or toks.count(H_FLAGS[1]) != 1'),
    "post_drop_moved_clause": ('                if not bool((w != 0).any()):\n                    bad.append(f"{kind}: route_br.{k}.weight did not move', '                if False:\n                    bad.append(f"{kind}: route_br.{k}.weight did not move'),
    "post_drop_route_emb_clause": ('        if any(k.startswith("route_emb.") for k in fin_sd):', '        if False:'),
    "post_drop_h_branch_clause": ('        if any(k.startswith("route_br.") for k in fin_sd):\n            bad.append(f"{kind} checkpoint carries route_br keys (the cond arm', '        if False:\n            bad.append(f"{kind} checkpoint carries route_br keys (the cond arm'),
    "hyg_drop_mode_value": ('dict(aux_route="mode"), ', ''),
}
EXPECT_RED = {"render_without_the_flags": "render", "render_modes_as_cond": "render", "gate_does_not_call_the_m_read": "gate", "gate_drop_built_lever_clause": "gate", "gate_drop_weight_lever_clause": "gate",
              "gate_drop_named_once_clause": "gate", "gate_drop_weight_named_clause": "gate", "gate_drop_each_branch_clause": "gate", "gate_drop_permutation_clause": "gate", "gate_drop_exact_zero_controls": "gate",
              "gate_drop_identity_clause": "gate", "gate_drop_route_group_clause": "gate", "gate_drop_route_term_clause": "gate", "h_gate_reads_the_m_kind": "gate", "pair_drop_diff_clause": "pair", "pair_drop_value_clause": "pair", "post_drop_moved_clause": "post", "post_drop_route_emb_clause": "post",
              "post_drop_h_branch_clause": "post", "hyg_drop_mode_value": "hyg"}


@pytest.mark.parametrize("mname", sorted(EXPECT_RED))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(AssertionError):
        CHECKS[EXPECT_RED[mname]](m, tmp_path)
