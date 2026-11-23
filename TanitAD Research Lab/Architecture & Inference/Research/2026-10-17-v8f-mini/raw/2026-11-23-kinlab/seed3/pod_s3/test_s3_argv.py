"""test_s3_argv.py -- the whole file: s3_argv.py renders exactly the card's substitutions and refuses every other change (one regression case per refusal). CPU, stdlib. Run: python test_s3_argv.py (rc 0 = all pass).
Fixtures: the REAL s0 argv files of the pod (fx/), copied here as KIT_FX. S3_TEST_ARGV may name the tool under test (the mutant runner sets it)."""
import contextlib
import hashlib
import io
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.environ.get("S3_FX", os.path.join(HERE, "fx"))
TOOL = os.environ.get("S3_TEST_ARGV", os.path.join(HERE, "s3_argv.py"))
TMPBASE = os.environ.get("S3_TMP") or None


def load_tool(root):
    os.environ["RH_ROOT"] = root
    os.environ["S3_DS"] = root + "/ds"
    spec = importlib.util.spec_from_file_location("s3_argv_under_test", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jl(p):
    return json.load(open(p))["argv"]


def make_root(with_bank=True):
    root = tempfile.mkdtemp(dir=TMPBASE)
    for arm in ("B0_s0", "RH_s0"):
        os.makedirs(f"{root}/runs/{arm}")
        shutil.copy(f"{FX}/{arm}_argv.json", f"{root}/runs/{arm}/argv.json")
    os.makedirs(f"{root}/bank/s2")
    if with_bank:
        open(f"{root}/bank/s2/bank.json", "w").write('{"common": {"ckpt_sha256": "x"}}')
    os.makedirs(f"{root}/ds/argv")
    for s in ("U_s1",):
        shutil.copy(f"{FX}/{s}.B.tmpl.json", f"{root}/ds/argv/{s}.B.tmpl.json")
    return root


def case(name, fn, fails):
    try:
        fn()
        print("PASS", name)
    except AssertionError as e:
        fails.append(name)
        print("FAIL", name, str(e)[:200])


def main():
    fails = []
    root = make_root()
    A = load_tool(root)
    sha = hashlib.sha256(open(f"{root}/bank/s2/bank.json", "rb").read()).hexdigest()
    b0s, rhs = jl(f"{root}/runs/B0_s0/argv.json"), jl(f"{root}/runs/RH_s0/argv.json")

    def t_render_all():
        assert A.main(["render", "--arms", "B0_s2", "RH_s2", "KI_s2"]) == 0
        b0, rh, ki = (jl(f"{root}/runs/{a}/argv.json") for a in ("B0_s2", "RH_s2", "KI_s2"))
        assert len(b0) == len(b0s) and len(rh) == len(rhs) and len(ki) == len(rhs) + 2
        assert ki[:-2] != rh and ki[-2:] == ["--dagger-base", "logged"]
        f = A.flags(rh)
        assert f["--seed"] == ["2"] and f["--dagger-seed"] == ["2"] and f["--dagger-bank"] == [root + "/bank/s2"] and f["--dagger-bank-sha256"] == [sha]
        assert f["--init-ckpt"] == ["/root/dscale10k/runs/UE_s2/A/ckpt_last.pt"] and f["--out"] == [root + "/runs/RH_s2/train"]
        assert A.flags(ki)["--out"] == [root + "/runs/KI_s2/train"]
        assert A.main(["verify", "--arms", "B0_s2", "RH_s2", "KI_s2"]) == 0
        assert A.main(["render", "--arms", "B0_s2", "RH_s2", "KI_s2"]) == 0     # idempotent
        sk = jl(f"{root}/runs/B0_s2/pair_rh_skeleton.json")
        assert A.flags(sk)["--dagger-bank-sha256"] == ["@BANKSHA@"] and [sha if x == "@BANKSHA@" else x for x in sk] == rh
    case("render B0/RH/KI, verify, idempotent, skeleton", t_render_all, fails)

    def refuse(arm, mut, why):
        """mutate the SOURCE or the output; the tool must refuse."""
        def f():
            r2 = make_root()
            A2 = load_tool(r2)
            mut(A2, r2)
            rc = A2.main(["render", "--arms", arm]) if arm == "B0_s2" else A2.main(["render", "--arms", arm])
            assert rc != 0, f"{why}: render returned 0"
            shutil.rmtree(r2, ignore_errors=True)
        return f

    def edit(r2, arm, flag, val=None, drop=False, add=None):
        p = f"{r2}/runs/{arm}/argv.json"
        t = jl(p)
        if add is not None:
            t = t + add
        elif drop:
            i = t.index(flag)
            del t[i:i + 2]
        else:
            t[t.index(flag) + 1] = val
        json.dump({"argv": t}, open(p, "w"))

    # source mutations: the s0 files are pinned and checked
    case("refuse: B0_s0 source changed (--lr)", refuse("B0_s2", lambda A_, r: edit(r, "B0_s0", "--lr", "2e-4"), "lr"), fails)
    case("refuse: RH_s0 source changed (--steps)", refuse("RH_s2", lambda A_, r: edit(r, "RH_s0", "--steps", "3001"), "steps"), fails)
    # output refusals through verify / diff_problems
    def vref(arm, mod, why, msg=None):
        def f():
            r2 = make_root()
            A2 = load_tool(r2)
            assert A2.main(["render", "--arms", "B0_s2", "RH_s2", "KI_s2"]) == 0
            p = f"{r2}/runs/{arm}/argv.json"
            t = jl(p)
            t = mod(t)
            json.dump({"argv": t}, open(p, "w"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = A2.main(["verify", "--arms", arm])
            assert rc != 0, f"{why}: verify returned 0"
            assert msg is None or msg in buf.getvalue(), f"{why}: the refusal does not carry the specific message {msg!r}: {buf.getvalue()[-300:]}"
            shutil.rmtree(r2, ignore_errors=True)
        return f

    def val(flag, v):
        def m(t):
            t = list(t)
            t[t.index(flag) + 1] = v
            return t
        return m
    case("refuse: B0_s2 third flag (--lr)", vref("B0_s2", val("--lr", "2e-4"), "third flag", 'not a card substitution'), fails)
    case("refuse: B0_s2 seed 1", vref("B0_s2", val("--seed", "1"), "seed"), fails)
    case("refuse: B0_s2 init UE_s0", vref("B0_s2", lambda t: [x.replace("UE_s2", "UE_s0") for x in t], "init"), fails)
    case("refuse: B0_s2 init UE_s3", vref("B0_s2", lambda t: [x.replace("UE_s2", "UE_s3") for x in t], "init3"), fails)
    case("refuse: B0_s2 out of B0_s0", vref("B0_s2", lambda t: [x.replace("B0_s2", "B0_s0") for x in t], "out"), fails)
    case("refuse: B0_s2 gets a dagger bank", vref("B0_s2", lambda t: t + ["--dagger-bank", "/x"], "dagger"), fails)
    case("refuse: B0_s2 dagger-p 0.1", vref("B0_s2", val("--dagger-p", "0.1"), "p", 'not the p = 0 control'), fails)
    case("refuse: RH_s2 seed 0", vref("RH_s2", val("--seed", "0"), "seed0"), fails)
    case("refuse: RH_s2 dagger-seed 0", vref("RH_s2", val("--dagger-seed", "0"), "dseed"), fails)
    case("refuse: RH_s2 dagger-seed 3", vref("RH_s2", val("--dagger-seed", "3"), "dseed3"), fails)
    case("refuse: RH_s2 bank s1", vref("RH_s2", lambda t: [x.replace("/bank/s2", "/bank/s1") for x in t], "bank"), fails)
    case("refuse: RH_s2 bank sha not 64 hex", vref("RH_s2", val("--dagger-bank-sha256", "abc"), "sha"), fails)
    case("refuse: RH_s2 bank sha of another bank", vref("RH_s2", val("--dagger-bank-sha256", "e5e44c0156ebc774e3551a54f589dc4fec19a68dc85c2c9838f7ecc0592662cf"), "sha2"), fails)
    case("refuse: RH_s2 dagger-p 0.3", vref("RH_s2", val("--dagger-p", "0.3"), "p3", '--dagger-p reads'), fails)
    case("refuse: RH_s2 readout causal_g1", vref("RH_s2", val("--dagger-bank-readout", "causal_g1"), "ro"), fails)
    case("refuse: RH_s2 out of RH_s0", vref("RH_s2", lambda t: [x.replace("RH_s2", "RH_s0") for x in t], "out"), fails)
    case("refuse: RH_s2 lever flag added", vref("RH_s2", lambda t: t + ["--dagger-mode", "add"], "mode", 'names a lever flag'), fails)
    case("refuse: RH_s2 flag removed (--detach-upper)", vref("RH_s2", lambda t: [x for x in t if x != "--detach-upper"], "removed"), fails)
    case("refuse: RH_s2 flag duplicated", vref("RH_s2", lambda t: t + ["--lr", "1e-4"], "dup", 'names a flag twice'), fails)
    case("refuse: RH_s2 tokens reordered", vref("RH_s2", lambda t: t[2:4] + t[0:2] + t[4:], "order", 'token positions'), fails)
    case("refuse: RH_s2 carries the base flag", vref("RH_s2", lambda t: t + ["--dagger-base", "logged"], "base in RH", 'only KI_s2 does'), fails)
    case("refuse: KI_s2 base own", vref("KI_s2", val("--dagger-base", "own"), "own"), fails)
    case("refuse: KI_s2 without the base pair", vref("KI_s2", lambda t: t[:-2], "nobase"), fails)
    case("refuse: KI_s2 second lever flag", vref("KI_s2", lambda t: t + ["--dagger-label", "kin_v1"], "label"), fails)
    case("refuse: KI_s2 base not last", vref("KI_s2", lambda t: t[-2:] + t[:-2], "base first"), fails)
    case("refuse: KI_s2 out of RH_s2", vref("KI_s2", lambda t: [x.replace("KI_s2", "RH_s2") for x in t], "out"), fails)
    case("refuse: KI_s2 dose 0.05", vref("KI_s2", val("--dagger-p", "0.05"), "dose"), fails)

    def t_nobank():
        r2 = make_root(with_bank=False)
        A2 = load_tool(r2)
        assert A2.main(["render", "--arms", "RH_s2"]) != 0 and not os.path.exists(f"{r2}/runs/RH_s2/argv.json")
        assert A2.main(["render", "--arms", "KI_s2"]) != 0
        assert A2.main(["render", "--arms", "B0_s2"]) == 0
        assert A2.main(["show", "--arm", "RH_s2", "--bank-sha", "@BANKSHA@"]) == 0
    case("RH/KI need the bank; B0 does not; show takes a placeholder", t_nobank, fails)

    def t_bankchanged():
        r2 = make_root()
        A2 = load_tool(r2)
        assert A2.main(["render", "--arms", "RH_s2"]) == 0
        open(f"{r2}/bank/s2/bank.json", "w").write('{"changed": 1}')
        assert A2.main(["verify", "--arms", "RH_s2"]) != 0
        assert A2.main(["render", "--arms", "RH_s2"]) != 0          # an existing argv that differs is never rewritten
    case("a bank that changed after the render is refused", t_bankchanged, fails)

    def t_cpu():
        r2 = make_root()
        A2 = load_tool(r2)
        assert A2.main(["cpu-argv", "--arm", "B0_s2", "--out-dir", r2 + "/cpu"]) == 0
        c, b = jl(r2 + "/cpu/B0_s2_argv_cpu.json"), A2.in_memory("B0_s2", r2)[0]
        assert [i for i in range(len(b)) if b[i] != c[i]] == [b.index("--device") + 1] and c[b.index("--device") + 1] == "cpu"
        assert not os.path.exists(f"{r2}/runs/B0_s2/argv.json")            # cpu-argv writes nothing under runs/
    case("cpu-argv differs only in --device and writes nothing under runs/", t_cpu, fails)

    def t_template():
        r2 = make_root()
        A2 = load_tool(r2)
        out = r2 + "/U_s2.B.tmpl.json"
        assert A2.main(["template", "--out", out]) == 0
        t1, t2 = jl(r2 + "/ds/argv/U_s1.B.tmpl.json"), jl(out)
        assert [i for i in range(len(t1)) if t1[i] != t2[i]] == [3, 7, 53] and t2[3] == "2" and t2[53] == "2" and t2[7].endswith("/U_s2/B")
        edit(r2, "B0_s0", "--lr", "2e-4")                                    # unrelated: template reads only U_s1
        t1[t1.index("--lr") + 1] = "9"
        json.dump({"argv": t1}, open(r2 + "/ds/argv/U_s1.B.tmpl.json", "w"))
        os.remove(out)
        assert A2.main(["template", "--out", out]) == 0                       # a changed U_s1 lr is carried (only 3 positions differ by construction)
        bad = jl(r2 + "/ds/argv/U_s1.B.tmpl.json")
        bad[bad.index("--seed") + 1] = "5"
        json.dump({"argv": bad}, open(r2 + "/ds/argv/U_s1.B.tmpl.json", "w"))
        os.remove(out)
        assert A2.main(["template", "--out", out]) != 0                       # a U_s1 that is not seed 1 is refused
    case("template: U_s2 = U_s1 in exactly seed / dagger-seed / out; a non-seed-1 source is refused", t_template, fails)
    print("RESULT", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
