"""test_ki_argv.py -- Dev-box test of ki_argv.py on RH_s0's REAL argv (read from the pod, pasted below; 62 tokens). stdlib only, no pod.

The well-formed render must differ from RH's in exactly the --out value and the two appended tokens `--dagger-base logged` (64 tokens, token for token). Each deliberate regression (a THIRD difference: another flag, a changed value,
a wrong base value, a wrong bank, dose, init, --out, a lever flag already in RH, the base pair moved / duplicated / missing / valueless) must REFUSE with a reason. Usage: python -I test_ki_argv.py <ki_argv.py> [<scratch dir>: also runs the render / verify / show / json / cpu-argv cases on files]"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys

SCRATCH = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
if SCRATCH:
    os.environ["RH_ROOT"] = SCRATCH + "/R"                                   # the file-level cases below run on a scratch root
spec = importlib.util.spec_from_file_location("ki_argv", sys.argv[1])
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

BSHA = "e5e44c0156ebc774e3551a54f589dc4fec19a68dc85c2c9838f7ecc0592662cf"
RH = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "/root/rollhist/runs/RH_s0/train", "--cache-dir", "/root/data/cache_union_dscale", "--v9-train", "/root/dscale10k/merged/v9_labels_union.npz",
      "--v9-train-md5", "9881b170a889f743c5b0894f9ad13f1d", "--lead-train", "/root/dscale10k/merged/lead_union.npz", "--speed-train", "/root/dscale10k/merged/speed_union.jsonl",
      "--join-train", "/root/dscale10k/merged/join_union.npz", "--p10", "/root/strat1/inputs/p10/p10_prior.pt", "--kin-mode", "yaw", "--tracks", "/root/dscale10k/merged/native_tracks_union.pt",
      "--batch", "64", "--steps-mult", "4", "--device", "cuda", "--init-ckpt", "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100",
      "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0.1", "--dagger-bank", "/root/rollhist/bank/s0", "--dagger-bank-sha256", BSHA, "--dagger-seed", "0",
      "--w-acc", "49.6969", "--detach-upper", "--fit-join", "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2"]
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(name, "PASS" if cond else "FAIL", extra)


def repl(t, old, new):
    t = list(t)
    t[t.index(old)] = new
    return t


def seed1(t):
    m = {"/root/rollhist/bank/s0": "/root/rollhist/bank/s1", "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt": "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt", "/root/rollhist/runs/RH_s0/train": "/root/rollhist/runs/RH_s1/train"}
    return [m.get(x, x) for x in t]


# 1. the well-formed render
check("RH fixture has 62 tokens", len(RH) == 62, len(RH))
ki, bad = M.build(RH, 0, "/root/rollhist", BSHA)
check("WELL-FORMED build: 64 tokens, last two are --dagger-base logged, own --out", not bad and len(ki) == 64 and ki[-2:] == ["--dagger-base", "logged"] and "/root/rollhist/runs/KI_s0/train" in ki, bad)
pos = [i for i in range(len(RH)) if RH[i] != ki[i]]
check("WELL-FORMED differs from RH in exactly 1 token position (the --out value) plus the 2 appended tokens", len(pos) == 1 and RH[pos[0] - 1] == "--out" and ki[:len(RH)][:pos[0]] == RH[:pos[0]] and len(ki) - len(RH) == 2, pos)
check("WELL-FORMED verify (diff_problems) clean", not M.diff_problems(RH, ki, 0, "/root/rollhist"))
ki1, bad1 = M.build(seed1(RH), 1, "/root/rollhist", BSHA)
check("WELL-FORMED seed-1 form (own init, bank, --out)", not bad1 and "/root/rollhist/runs/KI_s1/train" in ki1 and ki1[-2:] == ["--dagger-base", "logged"], bad1)

# 2. deliberate regressions of the KI argv (verify): each must be refused (a THIRD change or a wrong pair)
V = {
    "V1 base value own": repl(ki, "logged", "own"),
    "V2 base value garbage": repl(ki, "logged", "base"),
    "V3 a third flag appended": ki + ["--detach-lower"],
    "V4 a third value changed (--steps)": repl(ki, "3000", "3001"),
    "V5 another bank (bank_k)": repl(ki, "/root/rollhist/bank/s0", "/root/rollhist/bank_k/s0"),
    "V6 --out left RH's": repl(ki, "/root/rollhist/runs/KI_s0/train", "/root/rollhist/runs/RH_s0/train"),
    "V7 dose 0.3": repl(ki, "0.1", "0.3"),
    "V8 dagger-mode swap appended": ki + ["--dagger-mode", "swap"],
    "V9 bank sha changed": repl(ki, BSHA, "0" * 64),
    "V10 base pair moved (not at the end)": ki[:5] + ["--dagger-base", "logged"] + ki[5:-2],
    "V11 init changed": repl(ki, "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt"),
    "V12 --out of RK": repl(ki, "/root/rollhist/runs/KI_s0/train", "/root/rollhist/runs/RK_s0/train"),
    "V13 base flag missing (KI = RH + --out only)": ki[:-2],
    "V14 base flag duplicated": ki + ["--dagger-base", "logged"],
    "V15 base flag valueless": ki[:-1],
    "V16 a label flag added after the base pair (a third change)": ki + ["--dagger-label", "kin_v1"],
    "V17 a cache pair added (a third change)": ki + ["--dagger-label-cache", "/root/rollhist/kin_cache/s0_kin_v1.npz"],
    "V18 --dagger-kin-tb added (a third change)": ki + ["--dagger-kin-tb", "0.5"],
    "V19 the other seed's --out": repl(ki, "/root/rollhist/runs/KI_s0/train", "/root/rollhist/runs/KI_s1/train"),
    "V20 --device changed (the CPU form is not the launch argv)": repl(ki, "cuda", "cpu"),
}
for k, t in V.items():
    b = M.diff_problems(RH, t, 0, "/root/rollhist")
    check(k, bool(b), b[:1])

# 3. deliberate regressions of the SOURCE (build): RH's own argv is wrong
B = {
    "B1 RH already names --dagger-base": RH + ["--dagger-base", "own"],
    "B2 RH carries a label flag": RH + ["--dagger-label", "speed_v1"],
    "B3 RH dose 0.3": repl(RH, "0.1", "0.3"),
    "B4 RH bank sha not bank.json's": repl(RH, BSHA, "1" * 64),
    "B5 RH already has dagger-mode": RH + ["--dagger-mode", "swap"],
    "B6 RH init is another seed's": repl(RH, "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt", "/root/dscale10k/runs/UE_s1/A/ckpt_last.pt"),
    "B7 RH readout is causal_g1": repl(RH, "causal_g1x", "causal_g1"),
    "B8 RH already names a cache": RH + ["--dagger-label-cache", "/root/rollhist/kin_cache/s0_kin_v1.npz"],
    "B9 RH --out is not its own": repl(RH, "/root/rollhist/runs/RH_s0/train", "/root/rollhist/runs/RL_s0/train"),
    "B10 RH names --dagger-kin-tb": RH + ["--dagger-kin-tb", "1.0"],
}
for k, t in B.items():
    b = M.build(t, 0, "/root/rollhist", BSHA)[1]
    check(k, bool(b), b[:1])

# 4. the file-level commands on a scratch root (render / show / json / cpu-argv / verify), when a scratch dir is given
if SCRATCH:
    Rr = SCRATCH + "/R"
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(Rr + "/bank/s0")
    open(Rr + "/bank/s0/bank.json", "w").write('{"bank": "planted"}')
    bsha = hashlib.sha256(open(Rr + "/bank/s0/bank.json", "rb").read()).hexdigest()
    rh_s = [t.replace("/root/rollhist", Rr) if t.startswith("/root/rollhist") else (bsha if t == BSHA else t) for t in RH]
    os.makedirs(Rr + "/runs/RH_s0")
    os.makedirs(Rr + "/runs/B0_s0")
    json.dump({"argv": rh_s}, open(Rr + "/runs/RH_s0/argv.json", "w"), indent=1)
    json.dump({"argv": repl(rh_s[:rh_s.index("--dagger-bank")], "0.1", "0")}, open(Rr + "/runs/B0_s0/argv.json", "w"), indent=1)
    assert M.R == Rr, M.R

    def call(*a):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = M.main(list(a))
        return rc, buf.getvalue(), err.getvalue()

    def tree():
        return sorted(os.path.relpath(os.path.join(d, f), Rr) for d, _, fs in os.walk(Rr) for f in fs)
    before = tree()
    rc, out, _ = call("show", "--seeds", "0")
    check("SHOW passes, 64 tokens, writes nothing", rc == 0 and "ZZ KI-SHOW PASS" in out and "64 tokens" in out and tree() == before, out.strip().splitlines()[-2:])
    rc, out, err = call("json", "--seeds", "0")
    jo = json.loads(out) if rc == 0 else {}
    check("JSON prints the in-memory KI argv (64 tokens, base pair last) and writes nothing", rc == 0 and len(jo.get("argv", [])) == 64 and jo["argv"][-2:] == ["--dagger-base", "logged"] and tree() == before)
    rc, out, _ = call("cpu-argv", "--seeds", "0", "--out-dir", SCRATCH + "/cpu")
    ca = json.load(open(SCRATCH + "/cpu/s0_argv_cpu.json"))["argv"] if rc == 0 else []
    check("CPU-ARGV differs from the KI argv in exactly --device", rc == 0 and len(ca) == 64 and [i for i in range(64) if ca[i] != jo["argv"][i]] == [jo["argv"].index("--device") + 1] and tree() == before)
    rc, out, _ = call("render", "--seeds", "0")
    kp = Rr + "/runs/KI_s0/argv.json"
    check("RENDER writes runs/KI_s0/argv.json", rc == 0 and os.path.isfile(kp) and json.load(open(kp)) == jo, out.strip().splitlines()[-1:])
    rc, out, _ = call("render", "--seeds", "0")
    check("RENDER twice is idempotent", rc == 0 and "ZZ KI-RENDER PASS" in out)
    rc, out, _ = call("verify", "--seeds", "0")
    check("VERIFY passes on the rendered argv", rc == 0 and "ZZ KI-VERIFY PASS" in out)
    hand = json.load(open(kp))
    hand["argv"] = repl(hand["argv"], "3000", "3001")
    json.dump(hand, open(kp, "w"), indent=1)
    rc, out, _ = call("verify", "--seeds", "0")
    check("VERIFY refuses a hand-edited third change (--steps)", rc != 0 and "ZZ KI-VERIFY FAIL" in out and "--steps" in out)
    rc, out, _ = call("render", "--seeds", "0")
    check("RENDER refuses to rewrite a differing KI argv", rc != 0 and "differs from the argv this would write" in out)
    rc, out, err = call("json", "--seeds", "0")
    check("JSON refuses a differing existing KI argv", rc != 0 and "differs" in err and out == "")
    os.remove(kp)
    rh_bad = json.load(open(Rr + "/runs/RH_s0/argv.json"))
    rh_bad["argv"] = rh_bad["argv"] + ["--dagger-label", "kin_v1"]
    json.dump(rh_bad, open(Rr + "/runs/RH_s0/argv.json", "w"))
    rc, out, _ = call("show", "--seeds", "0")
    check("SHOW refuses when RH already carries a lever flag", rc != 0 and "ZZ KI-SHOW FAIL" in out)
    shutil.rmtree(SCRATCH)
print("ZZ KI-ARGV TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
