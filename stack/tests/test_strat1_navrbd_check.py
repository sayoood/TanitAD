"""PREREG_V8F_NAVRBD: the arm kinds N1D / N1Dx (base N1) in ``stack/scripts/strat1_check.py``. No model run for the gate reads (the geval test builds one tiny model on CPU).

Asserted: a D kind renders N1R's argv byte for byte with a different channel table and APPENDS exactly ``--nav-rb-feat exitdist`` (N1Dx: variant shuffle); every N1R / S1R / S1dR / N1Z argv sha256 is the
e71782a3 checker's (no existing kind moves); the gate read accepts a D selfcheck and refuses each missing or false control (the five column groups: planted gradient > 0, zero control exactly 0.0 per group, the
loaded array's per-group n_on > 0, feat / width / lever); a D kind refuses a missing feat flag and an R kind refuses ``exitdist``; the pairing rule (N1D argv == N1R argv + the feat flag + a new npz / sha); the post
check wants a [d, 15] weight; the G-GEN regression arm (a cfg that drops the feat) is refused; the binding reads the D builder report. Every new branch has a MUTATION: a copy of the checker with that branch
dropped, which one of the checks must turn RED."""
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
from test_strat1_s1d import _base  # noqa: E402
from test_strat1_navrb2_check import _sc as _sc_r, _edit, _toks as _toks_r, RBSHA  # noqa: E402
from test_strat1_token import COMMIT, rig  # noqa: E402,F401  (the fake-tree rig)

assert os.path.realpath(REAL.__file__).startswith(os.path.realpath(os.path.join(HERE, ".."))), REAL.__file__       # the study tree's checker
GROUPS = ("c0", "c1_6", "c7_12", "c13", "c14")
#: MEASURED: render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=BASE, rb_npz="/rb.npz") on strat1_check.py exactly as at e71782a3
E71782A3_RB_SHA = {("N1R", "s0"): "94a6e0eecb17a4edcedd9777697472c1b418e9683318d832d482be5b4cc955e3", ("N1R", "s1"): "43352aaf12a24df68e3e7198f987a4732318e8c4a103dc3f4c77ff5e53f0f463",
                   ("S1R", "s0"): "1ce3df29547eb7d28c9538c2b5fbe8ef7938657e21f66946ce55bd84493773f5", ("S1R", "s1"): "74baecfb4479c4f08223b507b4ef3c091a209e08b084dd1e2079b85514e0aa53",
                   ("S1dR", "s0"): "cac379a4131ab69e546f335a85dac284ad1e986ff562e9aade3746753dba87c5", ("S1dR", "s1"): "1671063465af2fd1e1f034d363fce3dc86ae10c44be6e26443b41451297a19f4",
                   ("N1Z", "s0"): "c0e2c70fcade59a5ff3c4ab66b39f2f5db2241749bda2b9705bbaf861fbf79f4", ("N1Z", "s1"): "ce891942b582e52444f25ccaaa46b0d14a2c6bdabbcd8c425ea8767fc8efded9"}
RBDSHA = "cd" * 32


def _render(SC, kind, seed, tmp_path, rb="/rb.npz"):
    return SC.render(kind, seed, "/r", "/o", nav_npz="/n.npz", ctx_npz="/c.npz", base_argvs=_base(tmp_path), rb_npz=rb)


def _toks(SC, kind, tmp_path):
    t = _render(SC, kind, "s0", tmp_path, "/rbd.npz" if kind in ("N1D", "N1Dx") else "/rb.npz") + ["--detach-upper", "--steps", str(SC.RUN_STEPS)]
    t[t.index("--nav-rb-sha256") + 1] = RBDSHA if kind in ("N1D", "N1Dx") else RBSHA
    return t


def _sc(SC, kind):
    """A minimal selfcheck.json the gate read accepts for ``kind`` (a D kind: the 15-column channel and its per-group probe)."""
    sc = _sc_r(SC, "N1R")
    if kind not in ("N1D", "N1Dx"):
        return sc
    sc["levers"]["nav_rb_feat"] = "exitdist"
    sc["info"]["nav_rb"] = {"sha256": RBDSHA, "variant": "shuffle" if kind == "N1Dx" else "real", "n_on": 7, "feat": "exitdist", "width": 15, "group_n_on": {g: 5 for g in GROUPS}}
    if kind == "N1Dx":
        sc["info"]["nav_rb"]["donor_map_sha256"] = "ef" * 32
    sc["navrb_probe"].update({"feat": "exitdist", "width": 15, "group_n_on": {g: 3 for g in GROUPS}, "group_grads": {g: 0.1 for g in GROUPS}, "group_grads_planted": {g: 0.2 for g in GROUPS},
                              "group_control_zero_channel": {g: 0.0 for g in GROUPS}, "group_unread_in_batch": []})
    return sc


# ---- the checks, written against a module so a mutated copy can be run through the same code ----
def check_render(SC, tmp_path):
    for seed in ("s0", "s1", "s10", "s11"):
        r = _render(SC, "N1R", seed, tmp_path, "/rbd.npz")
        for dk in ("N1D", "N1Dx"):
            d = _render(SC, dk, seed, tmp_path, "/rbd.npz")
            assert d[-2:] == ["--nav-rb-feat", "exitdist"] and len(d) == len(r) + 2, (dk, seed)
            body = d[:-2]
            diff = [i for i, (x, y) in enumerate(zip(body, r)) if x != y]
            assert [body[i - 1] for i in diff] == (["--nav-rb-variant"] if dk == "N1Dx" else []) and [body[i] for i in diff] == (["shuffle"] if dk == "N1Dx" else [])
            assert SC.flag_value(d, "--nav-rb-npz") == "/rbd.npz" and SC.flag_value(d, "--nav-rb-lr-mult") == "10" and SC.flag_value(d, "--nav-rb") == "on"
    assert "--nav-rb-feat" not in _render(SC, "N1R", "s0", tmp_path) and "--nav-rb-feat" not in _render(SC, "N1Z", "s0", tmp_path) and "--nav-rb-feat" not in _render(SC, "N1", "s0", tmp_path)
    assert SC.rb_names("N1D") == ("navtext_nav_rbd.npz", "navtext_nav_rbd_report.json") and SC.rb_names("N1R")[0] == "navtext_nav_rb.npz"


def check_gate(SC, tmp_path):
    for dk in ("N1D", "N1Dx"):
        assert SC.gate_problems(dk, _toks(SC, dk, tmp_path), _sc(SC, dk)) == [], dk
    t, sc = _toks(SC, "N1D", tmp_path), _sc(SC, "N1D")
    for path, val in ((("levers", "nav_rb_feat"), "state"), (("info", "nav_rb", "feat"), "state"), (("info", "nav_rb", "width"), 1), (("info", "nav_rb", "variant"), "zero"),
                      (("navrb_probe", "feat"), "state"), (("navrb_probe", "width"), 1), (("navrb_probe", "pass"), False),
                      (("navrb_probe", "control_e_nav_rb_grad_zero_channel"), 1e-9), (("navrb_probe", "e_nav_rb_grad"), 0.0)):
        assert SC.gate_problems("N1D", t, _edit(sc, path, val)), path
    for g in GROUPS:                                                                                         # each column group has its own clause
        assert SC.gate_problems("N1D", t, _edit(sc, ("navrb_probe", "group_grads_planted", g), 0.0)), ("planted", g)
        assert SC.gate_problems("N1D", t, _edit(sc, ("navrb_probe", "group_control_zero_channel", g), 1e-12)), ("control", g)
        assert SC.gate_problems("N1D", t, _edit(sc, ("info", "nav_rb", "group_n_on", g), 0)), ("n_on", g)
        dead = _edit(_edit(sc, ("navrb_probe", "group_n_on", g), 4), ("navrb_probe", "group_grads", g), 0.0)   # non-zero in the batch, no gradient, NOT named unread
        assert SC.gate_problems("N1D", t, dead), ("dead-in-batch", g)
        named = _edit(dead, ("navrb_probe", "group_unread_in_batch"), [g])
        assert SC.gate_problems("N1D", t, named) == [], ("unread named", g)
    for key in ("group_grads", "group_grads_planted", "group_control_zero_channel", "group_n_on"):             # a missing record
        d = json.loads(json.dumps(sc))
        del d["navrb_probe"][key]
        assert SC.gate_problems("N1D", t, d), key
    d = json.loads(json.dumps(sc))
    del d["info"]["nav_rb"]["group_n_on"]
    assert SC.gate_problems("N1D", t, d)
    d = json.loads(json.dumps(sc))
    del d["levers"]["nav_rb_feat"]
    assert SC.gate_problems("N1D", t, d)
    assert SC.gate_problems("N1D", [x for x in t if x not in ("--nav-rb-feat", "exitdist")], sc)               # a D kind refuses a missing feat flag
    assert SC.gate_problems("N1D", t[:t.index("--nav-rb-feat") + 1] + ["state"] + t[t.index("--nav-rb-feat") + 2:], sc)
    assert SC.gate_problems("N1D", _toks(SC, "N1Dx", tmp_path), sc)                                            # the shuffle argv under N1D
    assert SC.gate_problems("N1Dx", t, _sc(SC, "N1Dx"))                                                        # the real argv under N1Dx
    nodonor = _sc(SC, "N1Dx")
    del nodonor["info"]["nav_rb"]["donor_map_sha256"]
    assert SC.gate_problems("N1Dx", _toks(SC, "N1Dx", tmp_path), nodonor)
    # the R kinds refuse exitdist (argv and built) and keep their own read
    tr, scr = _toks_r(SC, "N1R", tmp_path), _sc_r(SC, "N1R")
    assert SC.gate_problems("N1R", tr, scr) == []
    assert SC.gate_problems("N1R", tr + ["--nav-rb-feat", "exitdist"], scr)
    assert SC.gate_problems("N1R", tr, _edit(scr, ("levers", "nav_rb_feat"), "exitdist"))
    assert SC.gate_problems("N1R", t, sc)                                                                      # the D argv and the D selfcheck under the R kind


def check_pair(SC, tmp_path):
    r = _render(SC, "N1R", "s0", tmp_path, "/rb.npz")
    r[r.index("--nav-rb-sha256") + 1] = RBSHA
    for dk in ("N1D", "N1Dx"):
        d = _render(SC, dk, "s0", tmp_path, "/rbd.npz")
        d[d.index("--nav-rb-sha256") + 1] = RBDSHA
        d[d.index("--out") + 1] = "/other/out"
        assert SC.pairing_problems(dk, d, r) == [], dk
        assert SC.pairing_problems(dk, [x for x in d if x not in ("--nav-rb-feat", "exitdist")], r)           # no feat flag
        assert SC.pairing_problems(dk, d + ["--nav-rb-feat", "exitdist"], r)                                   # named twice
        e = list(d); e[e.index("--seed") + 1] = "9"
        assert SC.pairing_problems(dk, e, r), "a moved seed"
        e = list(d); e[e.index("--nav-rb-lr-mult") + 1] = "30"
        assert SC.pairing_problems(dk, e, r), "a moved dose"
        e = list(d); e[e.index("--nav-text-npz") + 1] = "/another.npz"
        assert SC.pairing_problems(dk, e, r), "a moved nav table"
        e = list(d); e[e.index("--nav-rb-sha256") + 1] = RBSHA
        assert SC.pairing_problems(dk, e, r), "the N1R channel's sha (not a new table)"
        e = list(d); e[e.index("--nav-rb-npz") + 1] = "/rb.npz"
        assert SC.pairing_problems(dk, e, r), "the N1R channel's npz"
        assert SC.pairing_problems(dk, d + ["--extra", "1"], r) and SC.pairing_problems(dk, d, r + ["--extra", "1"])
    d = _render(SC, "N1D", "s0", tmp_path, "/rbd.npz")
    d[d.index("--nav-rb-sha256") + 1] = RBDSHA
    e = list(d); e[e.index("--nav-rb-variant") + 1] = "shuffle"
    assert SC.pairing_problems("N1D", e, r), "N1D with the shuffle variant"
    dx = _render(SC, "N1Dx", "s0", tmp_path, "/rbd.npz")
    dx[dx.index("--nav-rb-sha256") + 1] = RBDSHA
    e = list(dx); e[e.index("--nav-rb-variant") + 1] = "real"
    assert SC.pairing_problems("N1Dx", e, r), "N1Dx with the real variant"
    assert SC.pairing_problems("N1R", d, r)
    assert SC.pairing_problems("N1D", d, d)                                                                    # the reference must be the N1R argv (no feat flag)


def check_post(SC, tmp_path):
    import torch
    init = {"adapter.w": torch.zeros(2)}
    ok15 = {"adapter.w": torch.ones(2), "e_nav_rb.weight": torch.full((4, 15), 0.01)}
    for dk in ("N1D", "N1Dx"):
        assert SC.moved_identity(init, ok15, dk) == [], dk
        assert SC.moved_identity(init, {**ok15, "e_nav_rb.weight": torch.full((4, 1), 0.01)}, dk)             # a 1-column weight under a D kind
        assert SC.moved_identity(init, {**ok15, "e_nav_rb.weight": torch.zeros(4, 15)}, dk)                   # never moved
    assert SC.moved_identity(init, ok15, "N1R")                                                                # a 15-column weight under an R kind
    assert SC.moved_identity(init, {"adapter.w": torch.ones(2), "e_nav_rb.weight": torch.full((4, 1), 0.01)}, "N1R") == []
    w = torch.zeros(4, 15)
    w[:, 13] = 0.3
    n = SC.group_weight_norms(w)
    assert set(n) == set(GROUPS) and n["c13"] > 0.0 and all(n[g] == 0.0 for g in GROUPS if g != "c13")


def check_report(SC, tmp_path):
    rb = tmp_path / "rbd.npz"
    rb.write_bytes(b"rbdbytes")
    good = {sp: {"G-CLOCK-RBD_pass": True, "G-CLOCK-RBD_regression_arm_shift_5": {"pass": False}} for sp in ("train", "eval")}
    good["table_sha256"] = SC.sha_file(str(rb))
    rp = tmp_path / "rep.json"
    rp.write_text(json.dumps(good), encoding="utf-8")
    assert SC.rb_report_problems("N1D", str(rb), str(rp)) == [] and SC.rb_report_problems("N1Dx", str(rb), str(rp)) == []
    assert SC.rb_report_problems("N1D", str(rb), str(tmp_path / "none.json"))
    assert SC.rb_report_problems("N1R", str(rb), str(rp))                                                       # the R kind reads the R builder's keys: the D report is not its report
    for mut in (lambda r: r["eval"].update({"G-CLOCK-RBD_pass": False}), lambda r: r["train"]["G-CLOCK-RBD_regression_arm_shift_5"].update({"pass": True}),
                lambda r: r.update({"table_sha256": "00" * 32})):
        bad = json.loads(json.dumps(good))
        mut(bad)
        rp.write_text(json.dumps(bad), encoding="utf-8")
        assert SC.rb_report_problems("N1D", str(rb), str(rp))


def _tiny_ckpt(path, feat):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    from tanitad.models import v8f_mini as M
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, nav_rb="on", nav_rb_feat=feat)
    m = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    torch.save({"model": m.state_dict(), "cfg": cfg.__dict__}, path)
    return m


def check_geval(SC, tmp_path):
    p = tmp_path / "ck15.pt"
    m = _tiny_ckpt(p, "exitdist")
    bad, rec = SC.geval(str(p), m.built_levers())
    assert bad == [] and rec["regression_arm_nav_rb_refused"] and rec["regression_arm_nav_rb_feat_refused"] and rec["built_levers"]["nav_rb_feat"] == "exitdist"
    bad, _ = SC.geval(str(p), {**m.built_levers(), "nav_rb_feat": "state"})                                    # a selfcheck whose lever differs from the evaluator's build
    assert bad
    p1 = tmp_path / "ck1.pt"
    m1 = _tiny_ckpt(p1, "state")
    bad, rec = SC.geval(str(p1), m1.built_levers())
    assert bad == [] and "regression_arm_nav_rb_feat_refused" not in rec                                       # an N1R checkpoint: no feat arm, nothing changes
    assert SC.hyg_rb() == []


CHECKS = {"render": check_render, "gate": check_gate, "pair": check_pair, "post": check_post, "report": check_report, "geval": check_geval}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_navrbd_checks_pass_on_the_real_checker(name, tmp_path):
    CHECKS[name](REAL, tmp_path)


@pytest.mark.parametrize("key", sorted(E71782A3_RB_SHA))
def test_existing_r_kind_argv_sha_is_the_e71782a3_value(key, tmp_path):
    kind, seed = key
    assert REAL.argv_sha(_render(REAL, kind, seed, tmp_path)) == E71782A3_RB_SHA[key]


def test_kinds_and_bases():
    assert REAL.KINDS == ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z", "N1D", "N1Dx", "N1DW4", "N1DW2", "N1DC") and REAL.D_KINDS == ("N1D", "N1Dx")      # PREREG_V8F_N1DW adds the two W kinds, PREREG_V8F_N1DC the C kind
    assert REAL.RB_BASE["N1D"] == REAL.RB_BASE["N1Dx"] == "N1" and set(REAL.D_KINDS) <= set(REAL.RB_KINDS) <= set(REAL.NAV_KINDS) and not set(REAL.D_KINDS) & set(REAL.CTX_KINDS)
    assert REAL.base_kind("N1D") == "N1" and "--nav-rb-feat" not in REAL.RB_FLAGS and REAL.RB_FEAT_FLAG == "--nav-rb-feat"


def test_binding_and_token_run_for_a_d_kind(rig, tmp_path):
    inp = os.path.dirname(rig["nav_report"])
    rb = os.path.join(inp, "navtext_nav_rbd.npz")
    open(rb, "wb").write(b"rbdbytes")
    rep = {sp: {"G-CLOCK-RBD_pass": True, "G-CLOCK-RBD_regression_arm_shift_5": {"pass": False}} for sp in ("train", "eval")}
    rep["table_sha256"] = REAL.sha_file(rb)
    rbr = os.path.join(inp, "navtext_nav_rbd_report.json")
    json.dump(rep, open(rbr, "w", encoding="utf-8"))
    toks = list(rig["toks"]) + ["--nav-rb", "on", "--nav-rb-npz", rb, "--nav-rb-sha256", REAL.sha_file(rb), "--nav-rb-variant", "real", "--nav-rb-lr-mult", "10", "--nav-rb-feat", "exitdist"]
    b, bad = REAL.binding("N1D", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)
    assert bad == [] and b["nav_rb_report_sha256"] == REAL.sha_file(rbr) and "--nav-rb-npz" in b["data"]
    assert REAL.binding("N1D", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"])[1]            # no report named for a D kind
    rd = os.path.join(rig["a"].runs, "N1D_s0")
    os.makedirs(os.path.join(rd, "gate"))
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    sc = os.path.join(rd, "gate", "selfcheck.json")
    open(sc, "w", encoding="utf-8").write('{"declared_vs_built": "PASS"}')
    tp = os.path.join(rd, "gate", "gate_token.json")
    REAL.write_token(tp, b, {c: {"pass": True} for c in REAL.REQUIRED_CHECKS}, REAL.sha_file(sc))
    assert REAL.token_problems(tp, "N1D", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr) == []
    assert REAL.token_problems(tp, "N1Dx", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)      # the kind is bound
    toks2 = [x for x in toks if x not in ("--nav-rb-feat", "exitdist")]
    assert REAL.token_problems(tp, "N1D", toks2, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)      # a feat dropped from the argv: the argv sha is bound
    open(rb, "wb").write(b"rbdbytes2")
    assert REAL.token_problems(tp, "N1D", toks, rig["root"], COMMIT, rig["suite"], nav_report=rig["nav_report"], rb_report=rbr)       # a changed table is refused


def _mutant(tmp_path, name, old, new):
    src = open(os.path.join(SCRIPTS, "strat1_check.py"), "rb").read().decode("utf-8")
    assert src.count(old) == 1, (name, src.count(old))
    d = tmp_path / "mut" / name
    d.mkdir(parents=True)
    (d / "strat1_check.py").write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("strat1_mutd_" + name, str(d / "strat1_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MUTATIONS = {
    "drop_group_gradient_clause": ('            elif need == "gt":', '            elif False:'),
    "drop_group_zero_control_clause": ('            elif need == "zero":', '            elif False:'),
    "drop_group_n_on_clause": ('        if set(gno) != set(RBD_GROUPS) or any(not (gno.get(g) or 0) > 0 for g in RBD_GROUPS):', '        if False:'),
    "drop_dead_in_batch_clause": ('            if dead:\n                bad.append(f"navrb_probe: groups {dead} have non-zero', '            if False:\n                bad.append(f"navrb_probe: groups {dead} have non-zero'),
    "drop_feat_lever_clause": ('    if dk and lv.get("nav_rb_feat") != "exitdist":', '    if False:'),
    "drop_info_feat_clause": ('        if rec.get("feat") != "exitdist" or rec.get("width") != 15:', '        if False:'),
    "drop_missing_feat_flag_clause": ('    if dk and RB_FEAT_FLAG not in toks:', '    if False:'),
    "drop_r_kind_refusal": ('    if not dk and flag_value(toks, RB_FEAT_FLAG) == "exitdist":', '    if False:'),
    "render_without_the_feat": ('        rb += [RB_FEAT_FLAG, "exitdist"]', '        rb += []'),
    "render_dx_variant_real": ('        rb[rb.index("--nav-rb-variant") + 1] = "shuffle"', '        pass'),
    "pair_drop_diff_clause": ('    if diff:\n        bad.append(f"the {kind} argv differs from N1R', '    if False:\n        bad.append(f"the {kind} argv differs from N1R'),
    "pair_drop_new_sha_clause": ('    if flag_value(toks, "--nav-rb-sha256") in (None, "", flag_value(ref, "--nav-rb-sha256")):', '    if False:'),
    "post_drop_width_clause": ('        if w is not None and kind in DCHAN_KINDS and tuple(w.shape)[1:] != (15,):', '        if False:'),      # PREREG_V8F_N1DW: the 15-column kinds are DCHAN_KINDS (D + W)
    "geval_drop_feat_regression_arm": ('    if ck["cfg"].get("nav_rb") == "on" and ck["cfg"].get("nav_rb_feat", "state") != "state":', '    if False:'),
    "report_use_r_names_for_d": ('            if not r.get(sp, {}).get("G-CLOCK-RBD_pass"):', '            if False:'),
    "names_d_use_r_files": ('    return (RBD_NPZ, RBD_REPORT) if kind in DCHAN_KINDS else', '    return (RBD_NPZ, RBD_REPORT) if False else'),      # PREREG_V8F_N1DW: as above
}
EXPECT_RED = {"drop_group_gradient_clause": "gate", "drop_group_zero_control_clause": "gate", "drop_group_n_on_clause": "gate", "drop_dead_in_batch_clause": "gate", "drop_feat_lever_clause": "gate",
              "drop_info_feat_clause": "gate", "drop_missing_feat_flag_clause": "gate", "drop_r_kind_refusal": "gate", "render_without_the_feat": "render", "render_dx_variant_real": "render",
              "pair_drop_diff_clause": "pair", "pair_drop_new_sha_clause": "pair", "post_drop_width_clause": "post", "geval_drop_feat_regression_arm": "geval", "report_use_r_names_for_d": "report",
              "names_d_use_r_files": "render"}


@pytest.mark.parametrize("mname", sorted(MUTATIONS))
def test_mutation_turns_a_check_red(mname, tmp_path):
    m = _mutant(tmp_path, mname, *MUTATIONS[mname])
    with pytest.raises(Exception):
        CHECKS[EXPECT_RED[mname]](m, tmp_path)


B2_TABLES = "F:/TanitAD-scratch/strat1/navrbd/b2/tables"


@pytest.mark.skipif(not os.path.isfile(B2_TABLES + "/navtext_nav_rbd_report.json"), reason="the B2 builder output is a study-box artifact")
def test_checker_accepts_the_real_builder_report_and_refuses_a_changed_table(tmp_path):
    """The real D builder report (key ``table_sha256``) passes for its own npz; the same report against another file is refused."""
    npz, rep = B2_TABLES + "/navtext_nav_rbd.npz", B2_TABLES + "/navtext_nav_rbd_report.json"
    assert REAL.rb_report_problems("N1D", npz, rep) == [] and REAL.rb_report_problems("N1Dx", npz, rep) == []
    other = tmp_path / "other.npz"
    other.write_bytes(b"not the table")
    assert any("different npz" in p for p in REAL.rb_report_problems("N1D", str(other), rep))
