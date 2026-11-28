"""test_reps2.py -- tests of the REPS2 kit (dev box, CPU, whole file). Run:
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q F:/TanitAD-scratch/dscale10k/reps2/test_reps2.py --basetemp=F:/TanitAD-scratch/dscale10k/reps2/tmp/pytest
Covers: (1) the patched checker admits UE_s2 / UE_s3 and nothing else; every mutant is refused; the reps1 checker's behaviour on every other arm is IDENTICAL (differential test over all arms x argvs);
(2) the token binds the reference argv; (3) gen_argv_reps2.py (pairs, 40+ deliberate regressions refused, cross-checked against the checker's own rep_ref_problems); (4) init_reps2.compare;
(5) static: every python file parses, every shell script passes bash -n, the wrapper pins the same tree and commit as the reps1 wrapper for UE_s1."""
import ast
import difflib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(os.path.dirname(HERE), "reps1")
REF_ARGV = os.path.join(HERE, "pod_ref", "argv")
SRC1 = os.path.join(R1, "pod_copy", "dscale10k_check_reps1.py")
SRC2 = os.environ.get("REPS2_CHECKER", os.path.join(HERE, "pod_copy", "dscale10k_check_reps2.py"))
MERGE = os.path.join(HERE, "pod_ref", "dscale10k_merge.py")
ROOT = "/root/dscale10k"
STEPS = 61568


def _load(path, name, tmp):
    d = os.path.join(str(tmp), name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(path, os.path.join(d, "dscale10k_check.py"))
    shutil.copy(MERGE, os.path.join(d, "dscale10k_merge.py"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, "dscale10k_check.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods(tmp_path_factory):
    t = tmp_path_factory.mktemp("mods")
    return _load(SRC1, "chk_reps1", t), _load(SRC2, "chk_reps2", t)


def argv(name):
    return json.load(open(os.path.join(REF_ARGV, name + ".A.json")))["argv"]


def setf(t, f, v):
    t = list(t)
    t[t.index(f) + 1] = v
    return t


def derive(arm, seed):
    return setf(setf(argv("UE_s0"), "--seed", str(seed)), "--out", f"{ROOT}/runs/{arm}/A")


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("genroot")).replace("\\", "/")
    os.makedirs(os.path.join(root, "argv"))
    ref = setf(argv("UE_s0"), "--out", f"{root}/runs/UE_s0/A")            # a reference whose --out lives under the test root
    json.dump({"argv": ref}, open(os.path.join(root, "argv", "UE_s0.A.json"), "w"))
    json.dump({"argv": setf(setf(ref, "--seed", "1"), "--out", f"{root}/runs/UE_s1/A")}, open(os.path.join(root, "argv", "UE_s1.A.json"), "w"))
    os.environ["DS_ROOT"] = root
    os.environ.pop("DS_REF", None)
    spec = importlib.util.spec_from_file_location("gen_reps2", os.path.join(HERE, "gen_argv_reps2.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


# ---------------------------------------------------------------------------------------------------------- 1 the checker
def test_patch_adds_only_the_intended_lines(mods):
    a = open(SRC1, encoding="utf-8").read().split("\n")
    b = open(SRC2, encoding="utf-8").read().split("\n")
    removed = [l[2:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("- ") and not l.startswith("---")]
    # the only lines of the reps1 checker that disappear are the ones the patch REPLACES (signature, return, the three ue_argv/b_dry spots, the diff-parser line, the C-ARGV call)
    assert len(removed) <= 12, removed
    for l in removed:
        assert any(k in l for k in ("def argv_diff_problems", "return bad", "b_dry", "b-dry", "--base-argv", "a.ue_steps", "argv_diff_problems(", 'put("C-ARGV"', "a union arm that names the corpus")), l


def test_arms_table_adds_exactly_two_arms(mods):
    c1, c2 = mods
    assert set(c2.ARMS) - set(c1.ARMS) == {"UE_s2", "UE_s3"}
    assert all(c2.ARMS[k] == c1.ARMS[k] for k in c1.ARMS)
    assert c2.ARMS["UE_s2"] == dict(set="U", stage_b=False, seed=2, dagger_seed=None, steps="ue")
    assert c2.ARMS["UE_s3"] == dict(set="U", stage_b=False, seed=3, dagger_seed=None, steps="ue")
    assert c2.MIX_ARMS == c1.MIX_ARMS and c2.REP_ARMS == ("UE_s2", "UE_s3")
    for a in ("UE_s2", "UE_s3"):
        assert c2.required_checks(a, "A") == c2.required_checks("UE_s1", "A") == c1.required_checks("UE_s1", "A")


def test_reps1_refuses_the_new_arms(mods):
    c1, _ = mods
    for a in ("UE_s2", "UE_s3", "UM_s2"):
        p = c1.argv_diff_problems(argv("B_s0"), derive("UE_s2", 2), a, "A", STEPS)
        assert p and "unknown arm" in p[0]


@pytest.mark.parametrize("arm,seed", [("UE_s2", 2), ("UE_s3", 3)])
def test_admits_the_two_new_arms(mods, arm, seed):
    _, c2 = mods
    assert c2.argv_diff_problems(argv("B_s0"), derive(arm, seed), arm, "A", STEPS, argv("UE_s0")) == []


@pytest.mark.parametrize("arm,seed", [("UE_s2", 2), ("UE_s3", 3)])
def test_mutants_are_refused(mods, arm, seed):
    _, c2 = mods
    good = derive(arm, seed)
    other = 5 - seed
    mut = {
        "second_flag_batch": setf(good, "--batch", "32"), "second_flag_steps": setf(good, "--steps", "61569"), "second_flag_kin_mode": setf(good, "--kin-mode", "raw"),
        "second_flag_grid": setf(good, "--grid", "pool2x5"), "arm_flag": setf(good, "--arm", "M2a"), "flag_added": good + ["--lr", "1e-3"], "flag_removed": [x for x in good if x != "--fit-join"],
        "wrong_seed_0": setf(good, "--seed", "0"), "wrong_seed_1": setf(good, "--seed", "1"), "wrong_seed_other_new": setf(good, "--seed", str(other)),
        "out_of_UE_s0": setf(good, "--out", f"{ROOT}/runs/UE_s0/A"), "out_of_UE_s1": setf(good, "--out", f"{ROOT}/runs/UE_s1/A"), "out_of_other_new": setf(good, "--out", f"{ROOT}/runs/UE_s{other}/A"),
        "out_outside_runs": setf(good, "--out", f"{ROOT}/tmp/x/A"),
    }
    for fl in ("--v9-train", "--lead-train", "--speed-train", "--tracks", "--join-train", "--cache-dir", "--p10"):
        mut["data_manifest" + fl] = setf(good, fl, good[good.index(fl) + 1] + ".other")
    for name, t in mut.items():
        assert c2.argv_diff_problems(argv("B_s0"), t, arm, "A", STEPS, argv("UE_s0")), name


def test_data_manifest_is_refused_by_the_seed_only_rule_not_by_the_old_rule(mods):
    """The B_s0-relative rule ALLOWS the union data flags to differ (DATA_FLAGS), so it alone would not stop 'another data manifest'; the new rule does."""
    _, c2 = mods
    t = setf(derive("UE_s2", 2), "--lead-train", "/root/dscale10k/merged/lead_other.npz")
    assert any(x.startswith("UE_s2: no reference argv") for x in c2.argv_diff_problems(argv("B_s0"), t, "UE_s2", "A", STEPS, None))
    p = c2.argv_diff_problems(argv("B_s0"), t, "UE_s2", "A", STEPS, argv("UE_s0"))
    assert any("lead_other" in x or "--lead-train" in x for x in p)
    assert not [x for x in c2.argv_diff_problems(argv("B_s0"), t, "UE_s2", "A", STEPS, None) if "--lead-train" in x and "reference" not in x]


def test_no_reference_or_a_wrong_reference_is_refused(mods):
    _, c2 = mods
    good = derive("UE_s2", 2)
    assert c2.argv_diff_problems(argv("B_s0"), good, "UE_s2", "A", STEPS, None)
    assert any("not ['0']" in x for x in c2.argv_diff_problems(argv("B_s0"), good, "UE_s2", "A", STEPS, argv("UE_s1")))
    assert c2.argv_diff_problems(argv("B_s0"), good, "UE_s2", "A", STEPS, argv("UM_s0"))


def test_stage_b_is_refused(mods):
    _, c2 = mods
    assert any("stage-A-only" in x for x in c2.argv_diff_problems(argv("B_s0"), derive("UE_s2", 2), "UE_s2", "B", STEPS, argv("UE_s0")))


def test_um_and_other_arm_names_are_refused(mods):
    _, c2 = mods
    for name in ("UM_s2", "UM_s3", "UE_s4", "UE_s5", "U_s2", "B_s2", "BL_s2", "UE_s2 ", "ue_s2", ""):
        p = c2.argv_diff_problems(argv("B_s0"), derive("UE_s2", 2), name, "A", STEPS, argv("UE_s0"))
        assert p and "unknown arm" in p[0], name
    for name in ("UM_s2", "UE_s4"):
        with pytest.raises(SystemExit):                                   # the CLI itself refuses the name (argparse choices)
            c2.main(["diff", "--arm", name, "--argv", "x", "--base-argv", "y"])


def test_wrong_seed_under_a_name_through_the_cli(mods, tmp_path, capsys):
    _, c2 = mods
    base = tmp_path / "base.json"
    json.dump({"argv": argv("B_s0")}, open(base, "w"))
    ref = tmp_path / "ref.json"
    json.dump({"argv": argv("UE_s0")}, open(ref, "w"))
    ok = tmp_path / "ok.json"
    json.dump({"argv": derive("UE_s2", 2)}, open(ok, "w"))
    bad = tmp_path / "bad.json"
    json.dump({"argv": derive("UE_s2", 3)}, open(bad, "w"))
    args = ["diff", "--arm", "UE_s2", "--base-argv", str(base), "--ue-steps", str(STEPS), "--rep-ref-argv", str(ref), "--argv"]
    assert c2.main(args + [str(ok)]) == 0 and "PASS" in capsys.readouterr().out
    assert c2.main(args + [str(bad)]) == 1 and "REFUSED" in capsys.readouterr().out
    assert c2.main(["diff", "--arm", "UE_s2", "--base-argv", str(base), "--ue-steps", str(STEPS), "--argv", str(ok)]) == 1      # no reference: refused


def test_every_other_arm_behaves_identically(mods):
    """Differential: the reps1 and the reps2 checker return the SAME problems for every old arm x every argv x both stages."""
    c1, c2 = mods
    argvs = {n: argv(n) for n in ("B_s0", "UE_s0", "UE_s1", "UM_s0", "UM_s1", "U_s0")}
    argvs["UE_s2"] = derive("UE_s2", 2)
    n = 0
    for arm in c1.ARMS:
        for an, t in argvs.items():
            for stage in ("A", "B"):
                for steps in (STEPS, None):
                    assert c1.argv_diff_problems(argv("B_s0"), t, arm, stage, steps) == c2.argv_diff_problems(argv("B_s0"), t, arm, stage, steps), (arm, an, stage, steps)
                    n += 1
    assert n == len(c1.ARMS) * 7 * 4


def test_old_arms_do_not_need_or_use_a_reference(mods):
    c1, c2 = mods
    assert c2.argv_diff_problems(argv("B_s0"), argv("UE_s1"), "UE_s1", "A", STEPS) == c2.argv_diff_problems(argv("B_s0"), argv("UE_s1"), "UE_s1", "A", STEPS, argv("UM_s0"))


# ---------------------------------------------------------------------------------------------------------- 2 the token binds the reference
def test_token_binds_the_reference_argv(mods, tmp_path):
    _, c2 = mods
    tree = tmp_path / "tree"
    for d in (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval",)):
        os.makedirs(tree.joinpath(*d))
    tree.joinpath("stack", "tanitad", "m.py").write_text("x = 1\n")
    tree.joinpath("stack", "scripts", "v8f_mini_train.py").write_text("y = 2\n")
    ref = tmp_path / "UE_s0.A.json"
    json.dump({"argv": argv("UE_s0")}, open(ref, "w"))
    toks = derive("UE_s2", 2)
    for fl in c2.DATA_BIND_FLAGS:                                         # the data files of the argv, as local dummy files, so that the binding has real files to hash
        if fl in toks:
            f = tmp_path / ("data_" + fl[2:])
            f.write_text(fl)
            toks = setf(toks, fl, str(f))
    cd = tmp_path / "cache"
    os.makedirs(cd / "train")
    (cd / "train" / "manifest.jsonl").write_text("{}\n")
    toks = setf(toks, "--cache-dir", str(cd))
    extras = {"rep_ref_argv": str(ref)}
    b, bad = c2.binding("UE_s2", "A", toks, str(tree), "cdbff971", extras)
    assert not [x for x in bad if "rep_ref" in x] and "rep_ref_argv" in b["data"]
    checks = {c: {"pass": True} for c in c2.required_checks("UE_s2", "A")}
    tp = str(tmp_path / "gate_token.json")
    c2.write_token(tp, b, checks, "UE_s2")
    assert c2.token_problems(tp, "UE_s2", "A", toks, str(tree), "cdbff971", extras) == []
    ref.write_text(json.dumps({"argv": argv("UE_s1")}))                   # the reference file changed after the mint
    assert any("rep_ref_argv" in x for x in c2.token_problems(tp, "UE_s2", "A", toks, str(tree), "cdbff971", extras))
    ref.write_text(json.dumps({"argv": argv("UE_s0")}))
    assert c2.token_problems(tp, "UE_s2", "A", toks, str(tree), "cdbff971", extras) == []
    assert any("rep_ref_argv" in x for x in c2.token_problems(tp, "UE_s2", "A", toks, str(tree), "cdbff971", {}))                 # a runner without the reference
    assert c2.token_problems(tp, "UE_s2", "A", setf(toks, "--seed", "3"), str(tree), "cdbff971", extras)                            # the argv of the other seed
    assert c2.token_problems(tp, "UE_s2", "A", toks, str(tree), "deadbeef", extras)                                                 # another commit
    assert c2.token_problems(tp, "UE_s3", "A", toks, str(tree), "cdbff971", extras)                                                 # the token is bound to its arm name


def _env(tmp_path, c2):
    tree = tmp_path / "tree"
    for d in (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval")):
        os.makedirs(tree.joinpath(*(d if isinstance(d, tuple) else (d,))))
    tree.joinpath("stack", "tanitad", "m.py").write_text("x = 1\n")
    tree.joinpath("stack", "scripts", "v8f_mini_train.py").write_text("open('started.txt', 'a').write('x')\n")      # a stub trainer: it only records that it was started
    ref = tmp_path / "UE_s0.A.json"
    toks = derive("UE_s2", 2)
    for fl in c2.DATA_BIND_FLAGS:
        if fl in toks:
            f = tmp_path / ("data_" + fl[2:])
            f.write_text(fl)
            toks = setf(toks, fl, str(f))
    cd = tmp_path / "cache"
    os.makedirs(cd / "train")
    (cd / "train" / "manifest.jsonl").write_text("{}\n")
    toks = setf(toks, "--cache-dir", str(cd))
    json.dump({"argv": setf(setf(toks, "--seed", "0"), "--out", f"{ROOT}/runs/UE_s0/A")}, open(ref, "w"))      # the reference: the same local data files, seed 0, its own --out
    ap = tmp_path / "UE_s2.A.json"
    json.dump({"argv": toks}, open(ap, "w"))
    return tree, ref, toks, ap


def test_mint_and_run_both_bind_the_reference(mods, tmp_path, monkeypatch):
    """The real mint() and the real run_arm() (with a stub trainer): mint binds the reference file into the token; the runner starts the trainer only while the reference is unchanged."""
    import argparse
    _, c2 = mods
    tree, ref, toks, ap = _env(tmp_path, c2)
    monkeypatch.setattr(c2, "required_checks", lambda arm, stage="A": ())      # the evidence of the other checks is pod-side; this test is about the binding
    gd = tmp_path / "gate"
    base = tmp_path / "B_s0.A.json"
    json.dump({"argv": setf(argv("B_s0"), "--p10", toks[toks.index("--p10") + 1])}, open(base, "w"))
    ev = tmp_path / "evidence.json"
    json.dump({"ue_steps": STEPS}, open(ev, "w"))
    ns = argparse.Namespace(arm="UE_s2", stage="A", argv=str(ap), base_argv=str(base), evidence=str(ev), dry="", ue_argv="", tacsrc_list="", b_dry="", merge_record="", r_release="", union_release="", tenk_v9="",
                            ue_dry="", corpus_join="", union_join="", eval_ids="", holdout_manifest="", holdout=(0, 0), digest_json=None, selfcheck="", root=str(tree), commit="cdbff971", gate_dir=str(gd),
                            rep_ref_argv=str(ref), py=sys.executable, gpu="")
    assert c2.mint(ns) == []
    tok = json.load(open(gd / "gate_token.json"))
    assert "rep_ref_argv" in tok["data"] and tok["data"]["rep_ref_argv"]["path"] == str(ref)
    assert tok["checks"]["C-ARGV"]["pass"] is True                               # C-ARGV, with the reference, reads PASS on the seed-2 argv
    rn = argparse.Namespace(arm="UE_s2", stage="A", argv=str(ap), root=str(tree), commit="cdbff971", gate_dir=str(gd), py=sys.executable, gpu="", evidence=str(ev), dry="", merge_record="", union_release="", tenk_v9="",
                            r_release="", selfcheck="", tacsrc_list="", ue_argv="", b_dry="", rep_ref_argv=str(ref))
    started = tree / "started.txt"
    assert c2.run_arm(rn) == 0 and started.exists()                              # the token binds argv, commit, tree, data and the reference: the stub trainer starts
    started.unlink()
    ref.write_text(json.dumps({"argv": argv("UE_s1")}))                          # the reference changed after the mint
    assert c2.run_arm(rn) == 2 and not started.exists()                          # REFUSED, the trainer never starts
    ref.write_text(json.dumps({"argv": argv("UE_s0")}))
    rn2 = argparse.Namespace(**{**vars(rn), "rep_ref_argv": ""})                  # a runner that does not name the reference
    assert c2.run_arm(rn2) == 2 and not started.exists()
    json.dump({"argv": setf(toks, "--seed", "3")}, open(ap, "w"))                # the argv of the other seed under this token
    assert c2.run_arm(rn) == 2 and not started.exists()


def test_mint_refuses_a_reference_that_is_not_seed_zero(mods, tmp_path, monkeypatch):
    import argparse
    _, c2 = mods
    tree, ref, toks, ap = _env(tmp_path, c2)
    monkeypatch.setattr(c2, "required_checks", lambda arm, stage="A": ())
    base = tmp_path / "B_s0.A.json"
    json.dump({"argv": setf(argv("B_s0"), "--p10", toks[toks.index("--p10") + 1])}, open(base, "w"))
    json.dump({"argv": setf(setf(toks, "--seed", "1"), "--out", f"{ROOT}/runs/UE_s1/A")}, open(ref, "w"))
    ev = tmp_path / "evidence.json"
    json.dump({"ue_steps": STEPS}, open(ev, "w"))
    ns = argparse.Namespace(arm="UE_s2", stage="A", argv=str(ap), base_argv=str(base), evidence=str(ev), dry="", ue_argv="", tacsrc_list="", b_dry="", merge_record="", r_release="", union_release="", tenk_v9="",
                            ue_dry="", corpus_join="", union_join="", eval_ids="", holdout_manifest="", holdout=(0, 0), digest_json=None, selfcheck="", root=str(tree), commit="cdbff971", gate_dir=str(tmp_path / "g"),
                            rep_ref_argv=str(ref))
    c2.mint(ns)
    assert json.load(open(tmp_path / "g" / "checks.json"))["C-ARGV"]["pass"] is False


# ---------------------------------------------------------------------------------------------------------- 3 gen_argv_reps2
def test_gen_argv_writes_the_two_argv_and_a_clean_proof(gen):
    assert gen.main() == 0
    proof = json.load(open(gen.OUTD + "/PAIRING_PROOF.json"))
    assert proof["all_pairs_pass"] and proof["all_regressions_refused"] and proof["n_regression"] >= 40
    assert {"UE_s2_vs_UE_s0", "UE_s3_vs_UE_s0"} <= set(proof["pairs"])
    assert any("UE_s1_vs_UE_s0" in k for k in proof["pairs"])
    for a, s in (("UE_s2", 2), ("UE_s3", 3)):
        t = json.load(open(f"{gen.D}/argv/{a}.A.json"))["argv"]
        ref = json.load(open(gen.REF))["argv"]
        diff = [(x, y) for x, y in zip(ref, t) if x != y]
        assert len(ref) == len(t) and len(diff) == 2 and t[t.index("--seed") + 1] == str(s) and t[t.index("--out") + 1] == f"{gen.D}/runs/{a}/A"


def test_gen_argv_is_idempotent_and_refuses_a_changed_file(gen):
    assert gen.main() == 0
    p = f"{gen.D}/argv/UE_s2.A.json"
    t = json.load(open(p))
    t["argv"][t["argv"].index("--batch") + 1] = "32"
    json.dump(t, open(p, "w"))
    with pytest.raises(AssertionError, match="exists and differs"):
        gen.main()
    t["argv"][t["argv"].index("--batch") + 1] = "64"
    json.dump(t, open(p, "w"))
    assert gen.main() == 0


def test_gen_argv_refuses_a_reference_that_is_not_seed_0(gen):
    ref = json.load(open(gen.REF))["argv"]
    with pytest.raises(AssertionError, match="not 0"):
        gen.derive(setf(ref, "--seed", "1"), "UE_s2")
    with pytest.raises(AssertionError):
        gen.derive([x for x in ref if x != "--seed"], "UE_s2")


def test_every_gen_mutant_is_refused_by_both_implementations(gen, mods):
    """The generator's pair check AND the checker's rep_ref_problems (two separate implementations) must each refuse every mutant, and accept the unmutated argv."""
    _, c2 = mods
    ref = json.load(open(gen.REF))["argv"]
    new = {a: gen.derive(ref, a) for a in gen.SEEDS}
    mut = gen.mutants(new)
    assert len(mut) >= 40
    pin = lambda a: {"--seed": [str(gen.SEEDS[a])], "--out": [f"{gen.D}/runs/{a}/A"]}
    for a in gen.SEEDS:
        assert gen.pair_problems(ref, new[a], {"--seed", "--out"}, pin(a)) == []
        assert c2.rep_ref_problems(new[a], ref, a) == []
    for name, a, t in mut:
        assert gen.pair_problems(ref, t, {"--seed", "--out"}, pin(a)), name
        assert c2.rep_ref_problems(t, ref, a), name


def test_pair_problems_parse_errors_are_refusals(gen):
    assert gen.pair_problems(["--a", "1"], ["--a", "1", "--a", "2"], set(), {})
    assert gen.pair_problems(["--a", "1"], ["x", "--a", "1"], set(), {})


# ---------------------------------------------------------------------------------------------------------- 4 init_reps2.compare
def _rec(state="s", cfg=None, buf="b"):
    return {"cfg": cfg or {"a": 1}, "state_sha256": state, "buffers_sha256": buf, "params_only_sha256": "p", "n_trainable": 7214495}


def _init_mod():
    spec = importlib.util.spec_from_file_location("init_reps2", os.path.join(HERE, "init_reps2.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_init_compare():
    I = _init_mod()
    ue0, ue1 = _rec("s0"), _rec("s1")
    assert I.compare("UE_s2", _rec("s2"), ue0, ue1, None, []) == []
    assert I.compare("UE_s3", _rec("s3"), ue0, ue1, _rec("s2"), []) == []
    assert any("cfg differs from UE_s0" in x for x in I.compare("UE_s2", _rec("s2", {"a": 2}), ue0, ue1, None, []))
    assert any("equals UE_s0" in x for x in I.compare("UE_s2", _rec("s0"), ue0, ue1, None, []))             # the seed did not change the init
    assert any("equals UE_s1" in x for x in I.compare("UE_s2", _rec("s1"), ue0, ue1, None, []))
    assert any("other new arm" in x for x in I.compare("UE_s3", _rec("s2"), ue0, ue1, _rec("s2"), []))
    assert any("buffers differ" in x for x in I.compare("UE_s2", _rec("s2", buf="x"), ue0, ue1, None, []))
    assert I.compare("UE_s2", _rec("s2"), ue0, ue1, None, ["hyg problem"]) == ["hyg problem"]


# ---------------------------------------------------------------------------------------------------------- 5 static
PYFILES = ["gen_argv_reps2.py", "init_reps2.py", "assemble_evidence_reps2_ue.py", "verify_token_reps2.py", "patch_checker_reps2.py", os.path.join("pod_copy", "dscale10k_check_reps2.py")]
SHFILES = ["blockreps2.sh", "blockreps2_prep.sh", "chk_arm_reps2.sh", "reps2_marker_watch.sh", "fake_leaf_reps2.sh", "dryrun_reps2.sh", "deliver_reps2.sh", "install_reps2.sh"]


@pytest.mark.parametrize("f", PYFILES)
def test_python_files_parse_and_are_lf(f):
    raw = open(os.path.join(HERE, f), "rb").read()
    assert b"\r" not in raw
    ast.parse(raw.decode("utf-8"))


@pytest.mark.parametrize("f", SHFILES)
def test_shell_files_pass_bash_n_and_are_lf(f):
    p = os.path.join(HERE, f)
    assert b"\r" not in open(p, "rb").read()
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    r = subprocess.run([bash, "-n", p], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr


def test_wrapper_pins_the_reps1_tree_and_commit():
    w2 = open(os.path.join(HERE, "chk_arm_reps2.sh"), encoding="utf-8").read()
    w1 = open(os.path.join(R1, "chk_arm_reps1.sh"), encoding="utf-8").read()
    c1 = [l for l in w1.split("\n") if l.strip().startswith("UE_s1)")][0]
    commit = c1.split("COMMIT=")[1].split(";")[0]
    assert commit == "cdbff971fa6dad4d20b937fe8abda6cb9f622adb" and f"COMMIT={commit}" in w2
    assert "--rep-ref-argv $REFARGV" in w2 and w2.count("$FILES") == 2          # mint and run bind the same files, the reference included
    blk = w2.split("case $ARM in")[1].split("esac")[0]
    assert "serves UE_s2 and UE_s3" in blk and "UM_s" not in blk


def test_chain_never_names_a_forbidden_target():
    import re
    code = "\n".join(l for l in open(os.path.join(HERE, "blockreps2.sh"), encoding="utf-8").read().split("\n") if not l.lstrip().startswith("#"))
    for bad in ("kill", "pkill", "taskkill", "rm -rf", "/root/ruled", "/root/rollhist", "tanitad-push"):
        assert bad not in code, bad
    assert "rm -f $MARKER $MINE" in code                                      # the only deletion: its own marker, and only if the watcher set it
    assert len(re.findall(r"\brm\b", code)) == 1
    assert all("--query-gpu" in l for l in code.split("\n") if "nvidia-smi" in l)


def test_assembler_refuses_evidence_of_another_arm():
    """The checker binds evidence files but not their arm (a dry mint with UE_s1's evidence read PASS for UE_s2): the REPS2 assembler refuses a probe or init record that is not the arm's own."""
    a = open(os.path.join(HERE, "assemble_evidence_reps2_ue.py"), encoding="utf-8").read()
    assert a.count("EVIDENCE-REFUSED: the reach probe") == 1 and a.count("EVIDENCE-REFUSED: the init record of") == 1
    assert 'os.path.basename(pj.get("argv", "")) != f"{arm}.A.json"' in a and 'int(arm[-1])' in a
    assert 'item("init_" + arm.lower())' in a and 'assert arm in ("UE_s2", "UE_s3")' in a
    assert "UE_s1" not in "\n".join(l for l in a.split("\n")[2:] if "REPS2 guard" not in l and "dry mint" not in l and not l.lstrip().startswith("#") and "UE_s0..UE_s3" not in l and "lineage" not in l)
