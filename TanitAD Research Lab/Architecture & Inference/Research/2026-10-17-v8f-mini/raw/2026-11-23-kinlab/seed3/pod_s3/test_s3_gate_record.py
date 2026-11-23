"""test_s3_gate_record.py -- the whole file: rh_gate_record_s3.py (B0_s2, RH_s2) and ki_gate_record_s3.py (KI_s2) mint on good inputs and REFUSE every bad one; verify re-hashes what the record binds.
Fixtures: the REAL s0 selfchecks of the pod (fx/*_selfcheck.json: B0_s0, RH_s0, KI_s0), whose probe values satisfy the gate; scratch RH_ROOT. CPU, stdlib. Run: python test_s3_gate_record.py (rc 0 = all pass).
S3_TEST_RHGR / S3_TEST_KIGR name the tool under test (the mutant runner sets them)."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fx")
RHGR = os.environ.get("S3_TEST_RHGR", os.path.join(HERE, "rh_gate_record_s3.py"))
KIGR = os.environ.get("S3_TEST_KIGR", os.path.join(HERE, "ki_gate_record_s3.py"))
ARGV = os.path.join(HERE, "s3_argv.py")
TMPBASE = os.environ.get("S3_TMP") or None
PY = sys.executable
UE2 = "6d4345258e07dee15b1c5171268182967189df1cdc13b89b81d26aae71c3f327"
RHC, KIC = "97fe0a56767320dbc5f4b823d67fb07a3209a745", "4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb"


def run(tool, root, *args):
    env = dict(os.environ, RH_ROOT=root)
    r = subprocess.run([PY, tool, *args], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def wj(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w"), indent=1)


def setkey(o, key, val):
    if isinstance(o, dict):
        for k in list(o):
            if k == key:
                o[k] = val
            else:
                setkey(o[k], key, val)
    elif isinstance(o, list):
        for v in o:
            setkey(v, key, val)


def make_root():
    root = tempfile.mkdtemp(dir=TMPBASE)
    for arm in ("B0_s0", "RH_s0"):
        os.makedirs(f"{root}/runs/{arm}")
        shutil.copy(f"{FX}/{arm}_argv.json", f"{root}/runs/{arm}/argv.json")
    os.makedirs(f"{root}/bank/s2")
    open(f"{root}/bank/s2/bank.json", "w").write('{"common": {}}')
    env = dict(os.environ, RH_ROOT=root)
    for arms in (["B0_s2"], ["RH_s2", "KI_s2"]):
        r = subprocess.run([PY, ARGV, "render", "--arms", *arms], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
    wj(f"{root}/gsuite_result_rh.json", {"new_commit": RHC, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0})
    wj(f"{root}/gsuite_result_ki.json", {"new_commit": KIC, "n_regressions": 0, "new_test_failures": 0, "missing_in_new": 0})
    wj(f"{root}/pilot/PILOT_RH.json", {"pass": True, "readout": "causal_g1x"})
    wj(f"{root}/bank/s2/BANK_CHECK.json", {"pass": True, "readout": "causal_g1x"})
    wj(f"{root}/bank/s2/S3_BANK_CHECK.json", {"pass": True, "generator_ckpt_sha256": UE2, "coverage_fraction": 0.79, "partial": False})
    for t, sc in (("B0_s2", "B0_s0"), ("RH_s2", "RH_s0"), ("KI_s2", "KI_s0")):
        os.makedirs(f"{root}/runs/{t}/selfcheck", exist_ok=True)
        shutil.copy(f"{FX}/{sc}_selfcheck.json", f"{root}/runs/{t}/selfcheck/selfcheck.json")
        open(f"{root}/runs/{t}/pair_{t}.log", "w").write("ZZ ROLLHIST-PAIR PASS\n")
    os.makedirs(f"{root}/done_s3")
    open(f"{root}/done_s3/tree_rh.done", "w").write("x")
    open(f"{root}/done_s3/tree_ki.done", "w").write("x")
    return root


TOOLS = {"B0_s2": RHGR, "RH_s2": RHGR, "KI_s2": KIGR}


def mint(root, tag, ro="causal_g1x"):
    return run(TOOLS[tag], root, "mint", "--tag", tag, "--readout", ro)


def case(name, fn, fails):
    try:
        fn()
        print("PASS", name)
    except AssertionError as e:
        fails.append(name)
        print("FAIL", name, str(e)[:240])


def main():
    fails = []

    def t_good():
        root = make_root()
        for tag in ("B0_s2", "RH_s2", "KI_s2"):
            rc, out = mint(root, tag)
            assert rc == 0 and "MINTED" in out, f"{tag}: {out}"
            rc, out = run(TOOLS[tag], root, "verify", "--tag", tag)
            assert rc == 0 and "VERIFY OK" in out, out
        shutil.rmtree(root, ignore_errors=True)
    case("B0_s2 / RH_s2 / KI_s2 mint and verify on good inputs", t_good, fails)

    def refuse(tag, mutate, why):
        def f():
            root = make_root()
            mutate(root)
            rc, out = mint(root, tag)
            assert rc != 0 and "REFUSED" in out and "MINTED" not in out, f"{why}: rc={rc} {out[-300:]}"
            assert not os.path.exists(f"{root}/runs/{tag}/gate/RH_GATE_RECORD.json"), f"{why}: a record was written"
            shutil.rmtree(root, ignore_errors=True)
        return f

    def edit_json(rel, fn):
        def m(root):
            p = f"{root}/{rel}"
            o = json.load(open(p))
            r = fn(o)
            json.dump(o if r is None else r, open(p, "w"))
        return m

    def rm(rel):
        return lambda root: os.remove(f"{root}/{rel}")

    def edit_argv(tag, fn):
        def m(root):
            p = f"{root}/runs/{tag}/argv.json"
            t = json.load(open(p))["argv"]
            json.dump({"argv": fn(t)}, open(p, "w"))
        return m
    S = {"B0_s2": "runs/B0_s2/selfcheck/selfcheck.json", "RH_s2": "runs/RH_s2/selfcheck/selfcheck.json", "KI_s2": "runs/KI_s2/selfcheck/selfcheck.json"}
    for tag in ("B0_s2", "RH_s2", "KI_s2"):
        gs = "gsuite_result_ki.json" if tag == "KI_s2" else "gsuite_result_rh.json"
        case(f"{tag}: refuse a G-SUITE at another commit", refuse(tag, edit_json(gs, lambda o: o.update(new_commit="cdbff971fa6dad4d20b937fe8abda6cb9f622adb")), "commit"), fails)
        case(f"{tag}: refuse a G-SUITE with regressions", refuse(tag, edit_json(gs, lambda o: o.update(n_regressions=1)), "regress"), fails)
        case(f"{tag}: refuse a G-SUITE with missing tests", refuse(tag, edit_json(gs, lambda o: o.update(missing_in_new=2)), "missing"), fails)
        case(f"{tag}: refuse a pilot of another readout", refuse(tag, edit_json("pilot/PILOT_RH.json", lambda o: o.update(readout="causal_g1")), "pilot"), fails)
        case(f"{tag}: refuse a missing tree marker", refuse(tag, rm("done_s3/tree_ki.done" if tag == "KI_s2" else "done_s3/tree_rh.done"), "tree marker"), fails)
        case(f"{tag}: refuse a pair log without PASS", refuse(tag, lambda r, t=tag: open(f"{r}/runs/{t}/pair_{t}.log", "w").write("PROBLEM\n"), "pair"), fails)
        case(f"{tag}: refuse a selfcheck declared_vs_built FAIL", refuse(tag, edit_json(S[tag], lambda o: o.update(declared_vs_built="FAIL")), "dvb"), fails)
        case(f"{tag}: refuse a selfcheck without the acc_probe pass", refuse(tag, edit_json(S[tag], lambda o: o["acc_probe"].update({"pass": False})), "acc"), fails)
    for tag in ("RH_s2", "KI_s2"):
        case(f"{tag}: refuse a bank check that did not pass", refuse(tag, edit_json("bank/s2/BANK_CHECK.json", lambda o: o.update({"pass": False})), "bank"), fails)
        case(f"{tag}: refuse the stock bank check of another readout", refuse(tag, edit_json("bank/s2/BANK_CHECK.json", lambda o: o.update({"readout": "causal_g1"})), "bank ro"), fails)
        case(f"{tag}: refuse a missing S3_BANK_CHECK.json", refuse(tag, rm("bank/s2/S3_BANK_CHECK.json"), "s3 bank"), fails)
        case(f"{tag}: refuse a bank generated by another checkpoint", refuse(tag, edit_json("bank/s2/S3_BANK_CHECK.json", lambda o: o.update(generator_ckpt_sha256="2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914")), "ckpt"), fails)
        case(f"{tag}: refuse coverage 0.4", refuse(tag, edit_json("bank/s2/S3_BANK_CHECK.json", lambda o: o.update(coverage_fraction=0.4)), "cov"), fails)
        case(f"{tag}: refuse a partial bank", refuse(tag, edit_json("bank/s2/S3_BANK_CHECK.json", lambda o: o.update(partial=True)), "partial"), fails)
        case(f"{tag}: refuse an S3 bank check that is not a pass", refuse(tag, edit_json("bank/s2/S3_BANK_CHECK.json", lambda o: o.update({"pass": False})), "s3 pass"), fails)
        case(f"{tag}: refuse an argv whose bank sha is not bank.json's", refuse(tag, lambda r, t=tag: open(f"{r}/bank/s2/bank.json", "w").write('{"common": {"x": 1}}'), "bank sha"), fails)
        case(f"{tag}: refuse a selfcheck whose dagger_probe did not pass", refuse(tag, edit_json(S[tag], lambda o: o["dagger_probe"].update({"pass": False})), "probe"), fails)
        case(f"{tag}: refuse dagger-p 0", refuse(tag, edit_argv(tag, lambda t: [("0" if (i and t[i - 1] == "--dagger-p") else x) for i, x in enumerate(t)]), "p0"), fails)
    # KINPUT items
    case("KI_s2: refuse base_source own", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: setkey(o, "base_source", "own")), "own"), fails)
    case("KI_s2: refuse rows_changed.base != 0", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: o["dagger_probe"]["rows_changed"].update({"base": 3})), "rows_changed"), fails)
    case("KI_s2: refuse n_selected_at_p1 = 0", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: setkey(o, "n_selected_at_p1", 0)), "n_sel"), fails)
    case("KI_s2: refuse dagger_label not human", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: setkey(o["levers"], "dagger_label", "kin_v1")), "label"), fails)
    case("KI_s2: refuse built dagger_base own", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: setkey(o["levers"], "dagger_base", "own")), "built base"), fails)
    case("KI_s2: refuse control_p0_grad_diff != 0", refuse("KI_s2", edit_json(S["KI_s2"], lambda o: setkey(o, "control_p0_grad_diff", 0.5)), "control"), fails)
    case("KI_s2: refuse an argv without the base pair", refuse("KI_s2", edit_argv("KI_s2", lambda t: t[:-2]), "nobase"), fails)
    case("KI_s2: refuse a second lever flag", refuse("KI_s2", edit_argv("KI_s2", lambda t: t + ["--dagger-label", "kin_v1"]), "label flag"), fails)
    case("B0_s2: refuse a B0 argv that names a dagger bank", refuse("B0_s2", edit_argv("B0_s2", lambda t: t + ["--dagger-bank", "/x"]), "b0 bank"), fails)
    case("RH_s2: refuse an RH argv with another declaration", refuse("RH_s2", edit_argv("RH_s2", lambda t: ["causal_g1" if x == "causal_g1x" else x for x in t]), "decl"), fails)

    def t_wrong_tag():
        root = make_root()
        for tool, tag in ((RHGR, "RH_s0"), (RHGR, "B0_s1"), (RHGR, "KI_s2"), (KIGR, "KI_s0"), (KIGR, "RH_s2"), (KIGR, "KI_s1")):
            rc, out = run(tool, root, "mint", "--tag", tag, "--readout", "causal_g1x")
            assert rc != 0 and "MINTED" not in out, f"{os.path.basename(tool)} {tag}: {out[-200:]}"
    case("each tool serves its own s2 tags only", t_wrong_tag, fails)

    def t_verify_tamper():
        root = make_root()
        for tag, rel in (("RH_s2", "bank/s2/S3_BANK_CHECK.json"), ("RH_s2", "runs/RH_s2/argv.json"), ("KI_s2", "gsuite_result_ki.json"), ("KI_s2", "runs/KI_s2/selfcheck/selfcheck.json"), ("B0_s2", "pilot/PILOT_RH.json")):
            r2 = make_root()
            assert mint(r2, tag)[0] == 0
            with open(f"{r2}/{rel}", "a") as f:
                f.write(" ")
            rc, out = run(TOOLS[tag], r2, "verify", "--tag", tag)
            assert rc != 0 and "VERIFY OK" not in out, f"{tag} {rel}: verify passed after a change"
        root = make_root()
        assert mint(root, "KI_s2")[0] == 0
        p = f"{root}/runs/KI_s2/gate/RH_GATE_RECORD.json"
        o = json.load(open(p))
        o["commit"] = RHC
        json.dump(o, open(p, "w"))
        assert run(KIGR, root, "verify", "--tag", "KI_s2")[0] != 0
    case("verify refuses a record whose bound file or commit changed", t_verify_tamper, fails)

    def t_lite():
        root = make_root()
        for tool, tag in ((RHGR, "B0_s2"), (RHGR, "RH_s2"), (KIGR, "KI_s2")):
            rc, out = run(tool, root, "precheck-lite", "--tag", tag, "--readout", "causal_g1x")
            assert rc == 0 and "PRECHECK OK" in out, out
        os.remove(f"{root}/bank/s2/bank.json")                                  # precheck-lite needs no bank
        assert run(RHGR, root, "precheck-lite", "--tag", "RH_s2", "--readout", "causal_g1x")[0] == 0
        wj(f"{root}/gsuite_result_ki.json", {"new_commit": KIC, "n_regressions": 2})
        assert run(KIGR, root, "precheck-lite", "--tag", "KI_s2", "--readout", "causal_g1x")[0] != 0
        wj(f"{root}/gsuite_result_rh.json", {"new_commit": "0" * 40})
        assert run(RHGR, root, "precheck-lite", "--tag", "B0_s2", "--readout", "causal_g1x")[0] != 0
        assert run(RHGR, root, "precheck-lite", "--tag", "RH_s2", "--readout", "causal_g1")[0] != 0
    case("precheck-lite: needs no bank, refuses a bad G-SUITE / another readout", t_lite, fails)
    def t_tags():
        # a tool serves only its own tags: a foreign tag is refused with the tool's own message, not by a later crash
        root = make_root()
        rc, out = run(RHGR, root, "mint", "--tag", "KI_s2", "--readout", "causal_g1x")
        assert rc != 0 and "serves" in out and "MINTED" not in out, out[-300:]
        rc, out = run(KIGR, root, "mint", "--tag", "RH_s2", "--readout", "causal_g1x")
        assert rc != 0 and "serves" in out and "MINTED" not in out, out[-300:]
        rc, out = run(KIGR, root, "precheck", "--tag", "RH_s2", "--readout", "causal_g1x")
        assert rc != 0 and "serves" in out, out[-300:]
        # verify: a record whose own tag is not the tag asked for is refused although every bound file and the commit match
        for tool, minted, asked in ((RHGR, "B0_s2", "RH_s2"), (KIGR, "KI_s2", "KI_s2")):
            root = make_root()
            assert mint(root, minted)[0] == 0
            p = f"{root}/runs/{minted}/gate/RH_GATE_RECORD.json"
            o = json.load(open(p))
            assert run(tool, root, "verify", "--tag", minted)[0] == 0
            if tool == KIGR:
                o["tag"] = "X_s9"
            if tool == RHGR:
                os.makedirs(f"{root}/runs/RH_s2/gate", exist_ok=True)
                p = f"{root}/runs/RH_s2/gate/RH_GATE_RECORD.json"
            json.dump(o, open(p, "w"))
            rc, out = run(tool, root, "verify", "--tag", asked)
            assert rc != 0 and "VERIFY OK" not in out, f"{tool}: a record of another tag verified: {out[-200:]}"
        # the RH tool does not verify a KI_s2 record, the KI tool does not verify an RH_s2 one (tag outside the tool's own)
        root = make_root()
        assert mint(root, "KI_s2")[0] == 0
        o = json.load(open(f"{root}/runs/KI_s2/gate/RH_GATE_RECORD.json"))
        o["commit"] = RHC
        json.dump(o, open(f"{root}/runs/KI_s2/gate/RH_GATE_RECORD.json", "w"))
        rc, out = run(RHGR, root, "verify", "--tag", "KI_s2")
        assert rc != 0 and "VERIFY OK" not in out
        root = make_root()
        assert mint(root, "RH_s2")[0] == 0
        o = json.load(open(f"{root}/runs/RH_s2/gate/RH_GATE_RECORD.json"))
        o["base"] = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"
        o["commit"] = KIC
        shutil.copytree(f"{root}/runs/RH_s2/gate", f"{root}/runs/KI_s2/gate", dirs_exist_ok=True)
        json.dump(o, open(f"{root}/runs/KI_s2/gate/RH_GATE_RECORD.json", "w"))
        rc, out = run(KIGR, root, "verify", "--tag", "KI_s2")
        assert rc != 0 and "VERIFY OK" not in out
        shutil.rmtree(root, ignore_errors=True)
    case("a tool serves only its own tags (mint, precheck, verify)", t_tags, fails)
    print("RESULT", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
