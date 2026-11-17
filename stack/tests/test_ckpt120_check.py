"""PREREG_V8F_CKPT120 launch gate (part 2b): ``ckpt120_check.py`` (argv, gate, token, runner), ``ckpt120_reach.py`` (K-REACH-e). A fake tree, fake data and a fake trainer: no model training, no GPU.

Every check has a deliberate-regression arm (a MUTATION that must turn the gate RED): each selfcheck-probe key (the value it must read: non-zero where the lever is on, EXACTLY 0.0 for each control), each argv lever, each token binding,
the post identity (the far head and e_nav_s must leave their inits), G-EVAL (a cfg that drops a head is refused) and the K-REACH-e gate (a planted dead head must read HEAD-DEAD).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import ckpt120_check as C  # noqa: E402
import strat1_check as S1  # noqa: E402

COMMIT = "a4357e79" + "0" * 32
FAKE_TRAINER = """import json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
import os
os.makedirs(out, exist_ok=True)
json.dump(a, open(os.path.join(out, "ran_argv.json"), "w"))
"""
ARMS = [(k, s) for k, v in C.ARMS.items() for s in v]


def _w(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _junit(d, results):
    os.makedirs(d, exist_ok=True)
    cases = "".join(f'<testcase classname="t.mod" name="{n}">' + ("" if s == "pass" else "<failure/>") + "</testcase>" for n, s in results.items())
    open(os.path.join(d, "shard0.xml"), "w", encoding="utf-8").write(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")


def good_selfcheck(kind, toks):
    """A selfcheck.json that satisfies every read of the gate for this arm kind (the shape the trainer writes)."""
    perm, nav, var = C.SPEC[kind]
    f = lambda n: S1.flag_value(toks, n)  # noqa: E731
    lv = {"s_far": "on", "w_ckpt_far": float(f("--w-ckpt-far")), "s_far_arc_m": 120.0, "s_far_zero_init": True, "s_far_in": 512, "detach_upper": True}
    if nav == "on":
        lv["nav_strat"] = "on"
    if perm != "off":
        lv["s_far_label_perm"] = perm
    if var != "real":
        lv["nav_strat_variant"] = var
    probe = {"w_ckpt_far": float(f("--w-ckpt-far")), "term_ckpt_far": 2.14, "total_diff": 0.7147, "total_diff_expected": 0.7147, "head_grad": 6.5, "control_head_grad_w0": 0.0, "control_allmasked_term": 0.0,
             "control_allmasked_head_grad": 0.0, "t_iso_far": True, "adapter_grad_s_detach_off": 4.3, "control_adapter_grad_s_detach_on": 0.0, "pass": True}
    if nav == "on":
        probe.update(nav_s_grad=0.23, nav_s_moves_s=True, nav_s_t_o_identical=True, nav_s_in_keys_read=True)
    sc = {"declared_vs_built": "PASS", "steps_total": 3000, "levers": lv, "live": {"pass": True, "groups_grad": {"SFAR": 1.0}},
          "nav_source_probe": {"pass": True, "source": "text", "n_text_ne_v9": 5}, "acc_probe": {}, "dagger_probe": {}, "otband_probe": {"detach_upper": True},
          "ckpt120_probe": probe, "rc120": {"sha256": f("--rc120-sha256"), "perm": perm, "n_ok_table": 10, "n_ok_after_perm": 9 if perm != "off" else 10}}
    if nav == "on":
        sc["nav_strat"] = {"sha256": f("--nav-strat-sha256"), "variant": var, "n_rows_differ_from_real": 7 if var == "donor" else 0}
    return sc


def _build_rig(tmp_path):
    root = str(tmp_path / "tree")
    _w(os.path.join(root, "stack", "scripts", "v8f_mini_train.py"), FAKE_TRAINER.encode())
    _w(os.path.join(root, "stack", "tanitad", "m.py"), b"x = 1\n")
    _w(os.path.join(root, "taniteval", "e.py"), b"y = 2\n")
    inp = str(tmp_path / "inputs").replace(chr(92), "/")                          # render maps box-2 paths under a forward-slash root
    cache = str(tmp_path / "cache")
    _w(os.path.join(cache, "train", "manifest.jsonl"), b'{"clip": "a"}\n')
    _w(os.path.join(cache, "eval", "manifest.jsonl"), b'{"clip": "b"}\n')
    B = "D:" + chr(92) + "tanitad"
    base = ["--arm", "M2b", "--seed", "0", "--steps", "3000", "--out", B + chr(92) + "runs" + chr(92) + "v8f_mini_full" + chr(92) + "x", "--cache-dir", B + chr(92) + "cache_full", "--init-ckpt", B + chr(92) + "init.pt", "--detach-upper",
            "--join-train", B + chr(92) + "j.npz"]
    frozen = _w(os.path.join(inp, "frozen", "argvs_obd.json"), json.dumps({"OBD-CTL_s0": {"argv": base}, "OBD-CTL_s1": {"argv": [("1" if (i and base[i - 1] == "--seed") else x) for i, x in enumerate(base)]}}).encode())
    for n in ("init.pt", "j.npz"):
        _w(os.path.join(inp, n), n.encode())
    nav = _w(os.path.join(inp, "strat1", "navtext_nav.npz"), b"navbytes")
    rc = _w(os.path.join(inp, "ckpt120", "rc120_table.npz"), b"rc120bytes")
    ns = _w(os.path.join(inp, "ckpt120", "navtext_nav_s.npz"), b"navsbytes")
    rep = {sp: {"G-CLOCK-NAV_pass": True, "G-CLOCK-NAV_regression_arm_plus_travelled": {"pass": False}} for sp in ("train", "eval")}
    rep["rule"] = "BUILD_SPEC section 5 with step 5 d_ann = infinity (PREREG_V8F_CKPT120 nav_s: countdown, no announcement cut, UTURN none, bare text d = 0)"
    rep["navtext_nav_sha256"] = S1.sha_file(ns)
    nsr = os.path.join(inp, "ckpt120", C.NAV_S_REPORT)
    json.dump(rep, open(nsr, "w", encoding="utf-8"))
    _junit(str(tmp_path / "jb"), {"a": "pass", "b": "pass", "c": "fail"})
    _junit(str(tmp_path / "jn"), {"a": "pass", "b": "pass", "c": "fail", "d": "pass"})
    suite = str(tmp_path / "gsuite_result.json")
    json.dump(C.suite_compare(str(tmp_path / "jb"), str(tmp_path / "jn"), COMMIT, "a4357e79"), open(suite, "w", encoding="utf-8"))
    return dict(root=root, inp=inp, cache=cache, frozen=frozen, nav=nav, rc=rc, ns=ns, nsr=nsr, suite=suite, runs=str(tmp_path / "runs"), tmp=tmp_path)


@pytest.fixture
def rig(tmp_path):
    return _build_rig(tmp_path)


@pytest.fixture(scope="module")
def ro(tmp_path_factory):
    """The same rig, built ONCE and never modified: the argv / gate tests only render argvs from it (the files are small and hashed, the F: drive is slow)."""
    return _build_rig(tmp_path_factory.mktemp("ro"))


def toks_of(r, kind, seed):
    return C.render(kind, seed, r["inp"], os.path.join(r["runs"], C.arm_tag(kind, seed), "train"), r["nav"], r["rc"], r["ns"], base_argvs=r["frozen"], cache_dir=r["cache"])


def arm(rig, kind, seed, steps=3000):
    tag = C.arm_tag(kind, seed, steps)
    out = os.path.join(rig["runs"], tag, "train")
    toks = C.render(kind, seed, rig["inp"], out, rig["nav"], rig["rc"], rig["ns"], base_argvs=rig["frozen"], cache_dir=rig["cache"])
    rd = os.path.join(rig["runs"], tag)
    os.makedirs(os.path.join(rd, "gate"), exist_ok=True)
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    sc = os.path.join(rd, "gate", "selfcheck.json")
    json.dump(good_selfcheck(kind, toks), open(sc, "w", encoding="utf-8"))
    nsr = rig["nsr"] if C.SPEC[kind][1] == "on" else ""
    b, bad = C.binding(kind, toks, rig["root"], COMMIT, rig["suite"], nav_s_report=nsr)
    assert bad == [], bad
    C.write_token(os.path.join(rd, "gate", "gate_token.json"), b, {c: {"pass": True} for c in C.REQUIRED_CHECKS}, S1.sha_file(sc))
    a = argparse.Namespace(runs=rig["runs"], inputs=rig["inp"], root=rig["root"], commit=COMMIT, suite=rig["suite"], trainer="", py=sys.executable, gpu="")
    return dict(tag=tag, rd=rd, toks=toks, a=a, out=out, nsr=nsr)


# ------------------------------------------------------------------------------------------------ argv
@pytest.mark.parametrize("kind,seed", ARMS)
def test_argv_names_every_ckpt120_flag_and_the_arm_values(ro, kind, seed):
    t = C.render(kind, seed, ro["inp"], "/o", ro["nav"], ro["rc"], ro["ns"], base_argvs=ro["frozen"], cache_dir=ro["cache"])
    perm, nav, var = C.SPEC[kind]
    assert all(f in t for f in C.CKPT_FLAGS) and all(f in t for f in S1.NINE)
    assert [S1.flag_value(t, f) for f in ("--s-far", "--w-ckpt-far", "--s-far-lr-mult", "--s-far-label-perm", "--s-adapter-detach", "--nav-strat", "--nav-strat-variant")] == ["on", "0.3333", "10", perm, "off", nav, var]
    assert S1.flag_value(t, "--rc120-sha256") == S1.sha_file(ro["rc"])
    assert S1.flag_value(t, "--nav-strat-sha256") == (S1.sha_file(ro["ns"]) if nav == "on" else "")
    assert S1.flag_value(t, "--nav-source") == "text" and S1.flag_value(t, "--ctx-heads") == "off"      # N1 is the base of record
    assert S1.flag_value(t, "--seed") == ("2" if seed == "rep" else ("1" if seed == "s1" else "0"))


def test_h1_rep_is_h1_s0_with_only_the_seed_and_out_moved(ro):
    a = C.render("H1", "s0", ro["inp"], "/o", ro["nav"], ro["rc"], ro["ns"], base_argvs=ro["frozen"], cache_dir=ro["cache"])
    b = C.render("H1", "rep", ro["inp"], "/o", ro["nav"], ro["rc"], ro["ns"], base_argvs=ro["frozen"], cache_dir=ro["cache"])
    d = [i for i in range(len(a)) if a[i] != b[i]]
    assert len(a) == len(b) and [a[i - 1] for i in d] == ["--seed"] and b[d[0]] == "2"


def test_arms_outside_the_card_are_refused(ro):
    for kind, seed in (("R1", "s1"), ("R2", "rep"), ("H2", "rep"), ("X", "s0")):
        with pytest.raises(ValueError):
            C.render(kind, seed, ro["inp"], "/o", base_argvs=ro["frozen"])
    assert C.main(["argv", "R1", "s1", "--root", "/r", "--out", "/o"]) == 1


# ------------------------------------------------------------------------------------------------ gate: the good selfcheck passes, every mutation is RED
@pytest.mark.parametrize("kind,seed", ARMS)
def test_good_selfcheck_passes(ro, kind, seed):
    t = toks_of(ro, kind, seed)
    assert C.gate_problems(kind, t, good_selfcheck(kind, t)) == []


def _mut_set(path, val):
    def f(sc, toks):
        d = sc
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        return sc, toks
    return f


def _mut_del(path):
    def f(sc, toks):
        d = sc
        for k in path[:-1]:
            d = d[k]
        d.pop(path[-1])
        return sc, toks
    return f


def _mut_argv_set(flag, val):
    def f(sc, toks):
        toks = list(toks)
        toks[toks.index(flag) + 1] = val
        return sc, toks
    return f


def _mut_argv_drop(flag):
    def f(sc, toks):
        toks = list(toks)
        i = toks.index(flag)
        return sc, toks[:i] + toks[i + 2:]
    return f


P = ["ckpt120_probe"]
MUT_ALL = {
    "declared_vs_built FAIL": _mut_set(["declared_vs_built"], "FAIL"),
    "probe missing": _mut_del(P),
    "probe skipped": _mut_set(P, {"skipped": "x"}),
    "probe not pass": _mut_set(P + ["pass"], False),
    "head_grad 0 (the term does not reach the head)": _mut_set(P + ["head_grad"], 0.0),
    "control w0 reads 1e-9": _mut_set(P + ["control_head_grad_w0"], 1e-9),
    "control all-masked term reads 1e-9": _mut_set(P + ["control_allmasked_term"], 1e-9),
    "control all-masked grad reads 1e-9": _mut_set(P + ["control_allmasked_head_grad"], 1e-9),
    "control all-masked grad key deleted": _mut_del(P + ["control_allmasked_head_grad"]),
    "t_iso_far False": _mut_set(P + ["t_iso_far"], False),
    "adapter grad detach-off 0": _mut_set(P + ["adapter_grad_s_detach_off"], 0.0),
    "adapter control detach-on reads 1e-9": _mut_set(P + ["control_adapter_grad_s_detach_on"], 1e-9),
    "total moved by the wrong amount": _mut_set(P + ["total_diff"], 0.3),
    "probe w differs from the argv": _mut_set(P + ["w_ckpt_far"], 0.5),
    "lever s_far missing": _mut_del(["levers", "s_far"]),
    "lever arc 80": _mut_set(["levers", "s_far_arc_m"], 80.0),
    "lever w differs": _mut_set(["levers", "w_ckpt_far"], 1.0),
    "lever s_adapter_detach on": _mut_set(["levers", "s_adapter_detach"], "on"),
    "rc120 sha differs": _mut_set(["rc120", "sha256"], "0" * 64),
    "rc120 record missing": _mut_del(["rc120"]),
    "rc120 no survivors": _mut_set(["rc120", "n_ok_after_perm"], 0),
    "live not pass": _mut_set(["live", "pass"], False),
    "SFAR group dead": _mut_set(["live", "groups_grad", "SFAR"], 0.0),
    "argv drops --s-adapter-detach": _mut_argv_drop("--s-adapter-detach"),
    "argv drops --s-far-lr-mult": _mut_argv_drop("--s-far-lr-mult"),
    "argv drops --nav-strat-variant": _mut_argv_drop("--nav-strat-variant"),
    "argv --w-ckpt-far 0": _mut_argv_set("--w-ckpt-far", "0"),
    "argv --s-far off": _mut_argv_set("--s-far", "off"),
    "argv --s-adapter-detach on": _mut_argv_set("--s-adapter-detach", "on"),
    "argv --s-far-lr-mult 1": _mut_argv_set("--s-far-lr-mult", "1"),
    "argv --rc120-sha256 empty": _mut_argv_set("--rc120-sha256", ""),
}
MUT_NAV = {
    "nav_s_grad 0 (e_nav_s unreached)": _mut_set(P + ["nav_s_grad"], 0.0),
    "nav_s_t_o_identical False": _mut_set(P + ["nav_s_t_o_identical"], False),
    "nav_s_moves_s False": _mut_set(P + ["nav_s_moves_s"], False),
    "nav_s not in keys read": _mut_set(P + ["nav_s_in_keys_read"], False),
    "lever nav_strat missing": _mut_del(["levers", "nav_strat"]),
    "nav_strat record sha differs": _mut_set(["nav_strat", "sha256"], "1" * 64),
    "nav_strat record missing": _mut_del(["nav_strat"]),
}


@pytest.mark.parametrize("name", sorted(MUT_ALL))
@pytest.mark.parametrize("kind", ["H1", "H2", "R1", "R2"])
def test_mutation_turns_gate_red(ro, kind, name):
    t = toks_of(ro, kind, "s0")
    sc = good_selfcheck(kind, t)
    sc, t2 = MUT_ALL[name](sc, t)
    assert C.gate_problems(kind, t2, sc), f"{name} did not turn the gate RED on {kind}"


@pytest.mark.parametrize("name", sorted(MUT_NAV))
@pytest.mark.parametrize("kind", ["H2", "R2"])
def test_nav_s_mutation_turns_gate_red(ro, kind, name):
    t = toks_of(ro, kind, "s0")
    sc, t2 = MUT_NAV[name](good_selfcheck(kind, t), t)
    assert C.gate_problems(kind, t2, sc)


def test_r1_that_built_no_permutation_and_r2_that_did_not_derange_are_red(ro):
    t = toks_of(ro, "R1", "s0")
    sc = good_selfcheck("R1", t)
    sc["rc120"]["perm"] = "off"
    assert C.gate_problems("R1", t, sc)
    del sc["levers"]["s_far_label_perm"]
    assert C.gate_problems("R1", t, sc)
    t2 = toks_of(ro, "R2", "s0")
    sc2 = good_selfcheck("R2", t2)
    sc2["nav_strat"]["n_rows_differ_from_real"] = 0
    assert C.gate_problems("R2", t2, sc2)


def test_a_nav_strat_off_arm_that_carries_nav_s_is_red(ro):
    t = toks_of(ro, "H1", "s0")
    sc = good_selfcheck("H1", t)
    sc["ckpt120_probe"]["nav_s_grad"] = 1.0
    assert C.gate_problems("H1", t, sc)
    sc = good_selfcheck("H1", t)
    sc["nav_strat"] = {"sha256": "x"}
    assert C.gate_problems("H1", t, sc)
    sc = good_selfcheck("H1", t)
    sc["levers"]["nav_strat"] = "on"
    assert C.gate_problems("H1", t, sc)


def test_wrong_kind_for_the_argv_is_red(ro):
    t = toks_of(ro, "H1", "s0")
    assert C.gate_problems("R1", t, good_selfcheck("R1", t))                 # argv says perm off, the arm kind needs clip
    assert C.gate_problems("H2", t, good_selfcheck("H2", t))


def test_inherited_reads_still_apply(ro):
    t = toks_of(ro, "H1", "s0")
    sc = good_selfcheck("H1", t)
    sc["nav_source_probe"]["source"] = "v9"
    assert C.gate_problems("H1", t, sc)
    sc = good_selfcheck("H1", t)
    sc["otband_probe"]["detach_upper"] = False
    assert C.gate_problems("H1", t, sc)
    assert C.gate_problems("H1", t, {**good_selfcheck("H1", t), "steps_total": 10})


# ------------------------------------------------------------------------------------------------ token and runner
def _tp(rig, k):
    return C.token_problems(os.path.join(k["rd"], "gate", "gate_token.json"), k["tag"].split("_")[0], k["toks"], rig["root"], COMMIT, rig["suite"], nav_s_report=k["nsr"])


def _refused(rig, k, kind):
    assert C.run_arm(k["tag"], kind, k["a"]) == 2
    assert not os.path.exists(os.path.join(k["out"], "ran_argv.json")), "the trainer started although the runner refused"


@pytest.mark.parametrize("kind,seed", ARMS)
def test_untampered_passes_and_trainer_gets_exact_argv(rig, kind, seed):
    k = arm(rig, kind, seed)
    assert _tp(rig, k) == []
    assert C.run_arm(k["tag"], kind, k["a"]) == 0
    assert json.load(open(os.path.join(k["out"], "ran_argv.json"), encoding="utf-8")) == k["toks"]


def test_no_token_refuses(rig):
    k = arm(rig, "H1", "s0")
    os.remove(os.path.join(k["rd"], "gate", "gate_token.json"))
    _refused(rig, k, "H1")


@pytest.mark.parametrize("which", ["rc120", "nav_s", "nav", "join", "init", "manifest", "navsreport"])
def test_changed_data_file_refuses(rig, which):
    k = arm(rig, "H2", "s0")
    p = {"rc120": rig["rc"], "nav_s": rig["ns"], "nav": rig["nav"], "join": os.path.join(rig["inp"], "j.npz"), "init": os.path.join(rig["inp"], "init.pt"), "manifest": os.path.join(rig["cache"], "train", "manifest.jsonl"),
         "navsreport": rig["nsr"]}[which]
    with open(p, "ab") as f:
        f.write(b"\x00")
    _refused(rig, k, "H2")


def test_tampered_argv_refuses(rig):
    for flag, val in (("--seed", "1"), ("--w-ckpt-far", "0.3334"), ("--s-far-label-perm", "clip")):
        k = arm(rig, "H1", "s0")
        t = list(k["toks"])
        t[t.index(flag) + 1] = val
        json.dump(t, open(os.path.join(k["rd"], "argv.json"), "w", encoding="utf-8"))
        _refused(rig, k, "H1")


def test_tampered_argv_byte_refuses(rig):
    k = arm(rig, "H1", "s0")
    p = os.path.join(k["rd"], "argv.json")
    s = open(p, encoding="utf-8").read()
    assert "0.3333" in s
    open(p, "w", encoding="utf-8").write(s.replace("0.3333", "0.3330"))
    _refused(rig, k, "H1")


@pytest.mark.parametrize("target", ["tree", "trainer"])
def test_changed_tree_or_trainer_refuses(rig, target):
    k = arm(rig, "H1", "s0")
    p = os.path.join(rig["root"], "stack", "tanitad", "m.py") if target == "tree" else os.path.join(rig["root"], "stack", "scripts", "v8f_mini_train.py")
    with open(p, "ab") as f:
        f.write(b"# edit\n")
    _refused(rig, k, "H1")


def test_wrong_commit_refuses(rig):
    k = arm(rig, "H1", "s0")
    k["a"].commit = "b" * 40
    _refused(rig, k, "H1")


def test_gsuite_with_a_regression_refuses(rig):
    k = arm(rig, "H1", "s0")
    _junit(str(rig["tmp"] / "jn2"), {"a": "pass", "b": "fail", "c": "fail", "d": "pass"})
    json.dump(C.suite_compare(str(rig["tmp"] / "jb"), str(rig["tmp"] / "jn2"), COMMIT, "a4357e79"), open(rig["suite"], "w", encoding="utf-8"))
    _refused(rig, k, "H1")


def test_a_strat1_gsuite_file_is_not_accepted(rig):
    k = arm(rig, "H1", "s0")
    r = json.load(open(rig["suite"], encoding="utf-8"))
    r["schema"], r["written_by"] = "strat1-gsuite-result-v1", "strat1_check.py suite"
    json.dump(r, open(rig["suite"], "w", encoding="utf-8"))
    _refused(rig, k, "H1")


def test_edited_token_body_and_hand_written_token_refuse(rig):
    k = arm(rig, "H1", "s0")
    p = os.path.join(k["rd"], "gate", "gate_token.json")
    tok = json.load(open(p, encoding="utf-8"))
    tok["commit"] = "c" * 40
    json.dump(tok, open(p, "w", encoding="utf-8"))
    _refused(rig, k, "H1")
    k = arm(rig, "H1", "s1")
    p = os.path.join(k["rd"], "gate", "gate_token.json")
    tok = json.load(open(p, encoding="utf-8"))
    tok["written_by"] = "a human"
    tok["body_sha256"] = C.body_sha(tok)
    json.dump(tok, open(p, "w", encoding="utf-8"))
    _refused(rig, k, "H1")


def test_a_token_of_another_arm_refuses(rig):
    k = arm(rig, "H1", "s0")
    k2 = arm(rig, "R1", "s0")
    import shutil
    shutil.copy2(os.path.join(k["rd"], "gate", "gate_token.json"), os.path.join(k2["rd"], "gate", "gate_token.json"))
    _refused(rig, k2, "R1")


def test_the_same_token_cannot_start_a_different_kind(rig):
    k = arm(rig, "H1", "s0")
    assert C.run_arm(k["tag"], "H2", k["a"]) == 2                           # the token binds kind H1


def test_token_cannot_be_written_with_a_failed_check(rig):
    k = arm(rig, "H1", "s0")
    b, _ = C.binding("H1", k["toks"], rig["root"], COMMIT, rig["suite"])
    for c in C.REQUIRED_CHECKS:
        checks = {x: {"pass": True} for x in C.REQUIRED_CHECKS}
        checks[c] = {"pass": False}
        with pytest.raises(ValueError):
            C.write_token(os.path.join(k["rd"], "gate", "t2.json"), b, checks, "x")


def test_a_nav_s_report_whose_regression_arm_passes_refuses(rig):
    rep = json.load(open(rig["nsr"], encoding="utf-8"))
    rep["train"]["G-CLOCK-NAV_regression_arm_plus_travelled"]["pass"] = True      # the '+ travelled' arm must FAIL
    json.dump(rep, open(rig["nsr"], "w", encoding="utf-8"))
    t = C.render("H2", "s0", rig["inp"], "/o", rig["nav"], rig["rc"], rig["ns"], base_argvs=rig["frozen"], cache_dir=rig["cache"])
    assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"])
    assert C.nav_s_report_problems("H1", rig["ns"], rig["nsr"]) == []        # an arm with no nav_s binds no report
    assert t


REAL_NAV_S_REPORT = os.environ.get("CKPT120_REAL_NAV_S_REPORT", "F:/TanitAD-scratch/strat1/ckpt120build/nav_s/navtext_nav_s_report.json")


@pytest.mark.skipif(not os.path.isfile(REAL_NAV_S_REPORT), reason="the real nav_s builder report is not on this machine")
def test_the_real_builder_report_passes_the_checker_and_three_mutations_turn_it_red(rig):
    real = json.load(open(REAL_NAV_S_REPORT, encoding="utf-8"))
    real["navtext_nav_sha256"] = S1.sha_file(rig["ns"])                     # the fixture npz stands in for the real table; every other key is the builder's own
    json.dump(real, open(rig["nsr"], "w", encoding="utf-8"))
    assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"]) == []
    for mut in (lambda r: r.__setitem__("rule", r["rule"].replace("d_ann = infinity", "d_ann = 30")),                    # not the nav_s variant
                lambda r: r["eval"].__setitem__("G-CLOCK-NAV_pass", False),                                              # clock gate red
                lambda r: r["train"]["G-CLOCK-NAV_regression_arm_plus_travelled"].__setitem__("pass", True)):          # regression arm must fail
        bad = json.loads(json.dumps(real))
        mut(bad)
        json.dump(bad, open(rig["nsr"], "w", encoding="utf-8"))
        assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"])


def test_a_report_without_the_nav_s_rule_or_with_the_old_key_spelling_refuses(rig):
    rep = json.load(open(rig["nsr"], encoding="utf-8"))
    assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"]) == []
    del rep["rule"]
    json.dump(rep, open(rig["nsr"], "w", encoding="utf-8"))
    assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"])
    rep = {sp: {"G-CLOCK-NAV-S_pass": True} for sp in ("train", "eval")}      # the spelling the checker used to read
    rep["rule"], rep["navtext_nav_sha256"] = "d_ann = infinity", S1.sha_file(rig["ns"])
    json.dump(rep, open(rig["nsr"], "w", encoding="utf-8"))
    assert C.nav_s_report_problems("H2", rig["ns"], rig["nsr"])


def test_a_missing_rc120_table_or_wrong_declared_sha_is_a_binding_problem(rig):
    k = arm(rig, "H1", "s0")
    t = list(k["toks"])
    t[t.index("--rc120-sha256") + 1] = "f" * 64
    assert any("rc120" in x for x in C.data_binding(t)[1])
    t = list(k["toks"])
    t[t.index("--rc120-npz") + 1] = "/nonexistent/rc.npz"
    assert any("not a file" in x for x in C.data_binding(t)[1])


# ------------------------------------------------------------------------------------------------ post identity, G-EVAL, G-HYG (real model classes, tiny, CPU)
torch = pytest.importorskip("torch")
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)


def _model(**kw):
    from tanitad.models import v8f_mini as M
    torch.manual_seed(0)
    return M, M.V8FMini(M.V8FMiniConfig(**TINY, **kw), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))


def test_hyg_refuses_every_bad_ckpt120_config():
    assert C.hyg() == []


def test_moved_identity_reads_untrained_heads_as_red():
    M, m_init = _model()
    _, m_h2 = _model(s_far="on", w_ckpt_far=0.3333, nav_strat="on")
    init_sd = {k: v.clone() for k, v in m_init.state_dict().items()}
    fin = {k: v.clone() + (0.1 if k in init_sd else 0.0) for k, v in m_h2.state_dict().items()}
    ok = {k: (v if k in init_sd else v + 0.1) for k, v in fin.items()}
    for k in ok:
        if k in init_sd:
            ok[k] = init_sd[k] + 0.1
    assert C.moved_identity(init_sd, ok, "H2") == []
    assert C.moved_identity(init_sd, ok, "H1") != []                          # e_nav_s keys in an arm that has no nav_s
    zero_far = {k: (torch.zeros_like(v) if k.startswith("s_far_out.") else v) for k, v in ok.items()}
    assert any("zero init" in x for x in C.moved_identity(init_sd, zero_far, "H2"))
    fresh_nav = {k: (m_h2.state_dict()[k].clone() if k.startswith("e_nav_s.") else v) for k, v in ok.items()}
    assert any("seeded init" in x for x in C.moved_identity(init_sd, fresh_nav, "H2"))
    no_far = {k: v for k, v in ok.items() if not k.startswith("s_far_out.")}
    assert C.moved_identity(init_sd, no_far, "H2")


def _ck(tmp_path, **kw):
    M, m = _model(**kw)
    p = str(tmp_path / "ck.pt")
    torch.save({"model": m.state_dict(), "cfg": {k: getattr(m.cfg, k) for k in m.cfg.__dataclass_fields__}, "step": 1}, p)
    return p, m


def test_geval_loads_strictly_and_refuses_a_cfg_that_drops_a_head(tmp_path):
    p, m = _ck(tmp_path, s_far="on", w_ckpt_far=0.3333, nav_strat="on")
    bad, rec = C.geval(p, m.built_levers())
    assert bad == [] and rec["regression_arm_drop_s_far_refused"] and rec["regression_arm_drop_nav_strat_refused"]
    lv = dict(m.built_levers())
    lv["s_far_arc_m"] = 80.0                                                 # the selfcheck built something else
    assert C.geval(p, lv)[0]
    (tmp_path / "off").mkdir()
    p0, m0 = _ck(tmp_path / "off")                                           # a flag-off checkpoint loads strictly and has no regression arm to refuse
    assert C.geval(p0, m0.built_levers()) == ([], {"built_levers": {}})


# ------------------------------------------------------------------------------------------------ K-REACH-e
import ckpt120_reach as R  # noqa: E402


class Tiny(torch.nn.Module):
    """head_s (the control), the far head and a nav_s token, plus a frozen-by-design parameter outside the CKPT120 scope."""

    def __init__(self, nav):
        super().__init__()
        self.read_s = torch.nn.Linear(6, 7)
        self.head_s = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.GELU(), torch.nn.Linear(5, 4))
        self.s_far_out = torch.nn.Linear(5, 3)
        self.e_nav_s = torch.nn.Linear(5, 11) if nav else None
        self.frozen = torch.nn.Linear(2, 13)

    def forward(self, x, use_far=True, use_nav=True):
        z = self.head_s[1](self.head_s[0](self.read_s(x)))             # S's hidden layer, where the far head sits
        y = self.head_s[2](z).sum()
        if use_far:
            y = y + self.s_far_out(z).sum()
        if self.e_nav_s is not None and use_nav:
            y = y + self.e_nav_s(torch.ones(x.shape[0], 5)).sum()
        return y


def _reach_ck(nav=True, use_far=True, use_nav=True, zero_grad_far=False, steps=3):
    torch.manual_seed(1)
    m = Tiny(nav)
    m.frozen.weight.requires_grad_(False)
    m.frozen.bias.requires_grad_(False)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        y = m(torch.randn(4, 6), use_far, use_nav)
        if zero_grad_far:
            y = y + 0.0 * m.s_far_out(torch.ones(4, 5)).sum()           # the "--w-x default 0.0" case: an all-zero gradient tensor, so Adam allocates state
        y.backward()
        opt.step()
    return {"model": m.state_dict(), "opt": opt.state_dict(), "step": steps}


def test_reach_gate_passes_a_healthy_checkpoint_with_both_controls():
    r = R.gate(_reach_ck(nav=True), expect_nav_s=True)
    assert r["verdict"] == "PASS" and r["controls"]["positive"]["ok"] and r["controls"]["negative"]["ok"] and r["controls"]["negative"]["verdict"] == "HEAD-DEAD"
    r1 = R.gate(_reach_ck(nav=False), expect_nav_s=False)
    assert r1["verdict"] == "PASS"


def test_reach_gate_names_a_far_head_the_loss_never_reached():
    r = R.gate(_reach_ck(nav=True, use_far=False), expect_nav_s=True)
    assert r["verdict"] == "HEAD-DEAD" and set(r["dead"]) == {"s_far_out.weight", "s_far_out.bias"}


def test_reach_gate_names_a_nav_s_token_the_loss_never_reached():
    r = R.gate(_reach_ck(nav=True, use_nav=False), expect_nav_s=True)
    assert r["verdict"] == "HEAD-DEAD" and set(r["dead"]) == {"e_nav_s.weight", "e_nav_s.bias"}


def test_reach_gate_catches_the_all_zero_gradient_case_not_only_missing_state():
    r = R.gate(_reach_ck(nav=True, use_far=False, zero_grad_far=True), expect_nav_s=True)
    assert r["verdict"] == "HEAD-DEAD" and any(v[0]["case"].startswith(("b", "(b")) or "zero" in v[0]["case"].lower() for v in r["dead"].values())


def test_reach_gate_ignores_a_deliberately_frozen_parameter_outside_its_scope():
    r = R.gate(_reach_ck(nav=True), expect_nav_s=True)
    assert r["verdict"] == "PASS"                                            # `frozen.*` has no state, and is not this gate's business


def test_reach_gate_expectations_about_nav_s_are_checked_both_ways():
    assert R.gate(_reach_ck(nav=False), expect_nav_s=True)["verdict"] != "PASS"      # the arm lost its token
    assert R.gate(_reach_ck(nav=True), expect_nav_s=False)["verdict"] != "PASS"      # an arm that should have none carries one


def test_reach_gate_without_a_far_head_is_not_a_pass():
    ck = _reach_ck(nav=True)
    ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("s_far_out.")}
    assert R.gate(ck, expect_nav_s=True)["verdict"] != "PASS"


def test_reach_gate_without_optimizer_state_is_inconclusive_never_pass(tmp_path):
    ck = _reach_ck()
    p = str(tmp_path / "w.pt")
    torch.save({"model": ck["model"]}, p)
    out = str(tmp_path / "o.json")
    assert R.main(["--ckpt", p, "--out", out, "--expect-nav-s", "on"]) == 2
    assert json.load(open(out, encoding="utf-8"))["verdict"] == "INCONCLUSIVE"


def test_reach_gate_negative_control_fails_closed_when_the_gate_is_blind(monkeypatch):
    real = R.analyze

    def blind(ck, expect_nav_s=False, control=R.DEFAULT_CONTROL, param_names=None, _mutate=None):
        r = real(ck, expect_nav_s, control, param_names)                      # ignores the mutation: reads the healthy checkpoint
        return r
    monkeypatch.setattr(R, "analyze", blind)
    r = R.gate(_reach_ck(nav=True), expect_nav_s=True)
    assert r["verdict"] == "INCONCLUSIVE" and not r["controls"]["negative"]["ok"]


def test_reach_gate_positive_control_fails_closed_when_it_matches_nothing():
    r = R.gate(_reach_ck(nav=True), expect_nav_s=True, control=r"no_such_head\..*")
    assert r["verdict"] == "INCONCLUSIVE" and not r["controls"]["positive"]["ok"]


def test_reach_cli_writes_the_verdict_file_and_exit_codes(tmp_path):
    ck = _reach_ck(nav=True)
    p, bad_p = str(tmp_path / "ok.pt"), str(tmp_path / "dead.pt")
    torch.save(ck, p)
    torch.save(_reach_ck(nav=True, use_far=False), bad_p)
    o1, o2 = str(tmp_path / "ok.json"), str(tmp_path / "dead.json")
    assert R.main(["--ckpt", p, "--out", o1, "--expect-nav-s", "on"]) == 0
    assert R.main(["--ckpt", bad_p, "--out", o2, "--expect-nav-s", "on"]) == 1
    assert json.load(open(o1, encoding="utf-8"))["verdict"] == "PASS" and json.load(open(o2, encoding="utf-8"))["verdict"] == "HEAD-DEAD"
    assert R.main(["--ckpt", str(tmp_path / "missing.pt"), "--out", str(tmp_path / "m.json")]) == 2
    assert os.path.exists(tmp_path / "m.json") and not os.path.exists(str(tmp_path / "m.json") + ".part")
