"""gen_argv_reps2.py -- DSCALE10K REPS2 (pod): writes argv/UE_s2.A.json and argv/UE_s3.A.json = UE_s0.A's argv with EXACTLY --seed and --out changed (seed 2 / 3, out = <root>/runs/UE_s<k>/A).
Then the PAIRING PROOF (blockreps2/PAIRING_PROOF.json): each new argv vs UE_s0.A differs in exactly --seed and --out (nothing added, nothing removed, no other value); the s0 / s1 / s2 / s3 argvs form one family.
Deliberate-regression arms (each MUST be refused by pair_problems, and the checker's own rep_ref_problems must agree where the checker is importable): a second flag changed (--batch, --steps, --kin-mode,
--lr-like value), another data manifest (--v9-train, --lead-train, --speed-train, --tracks, --join-train, --cache-dir, --p10), a flag removed, a flag added, the wrong seed under a name (UE_s2 with --seed 3,
UE_s3 with --seed 2, UE_s2 with --seed 0), --out of another arm (UE_s0, UE_s1, UE_s3), --out outside runs/.
Any third change makes main() exit 1 (the chain's pre-flight reads the proof). Idempotent: an existing argv file must equal what would be written, else REFUSED (assert).
DS_ROOT overrides the root (dry run); DS_REF overrides the reference argv path (default <root>/argv/UE_s0.A.json). usage: gen_argv_reps2.py"""
import json
import os
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")
REF = os.environ.get("DS_REF", D + "/argv/UE_s0.A.json")
OUTD = D + "/blockreps2"
SEEDS = {"UE_s2": 2, "UE_s3": 3}
DATA_FLAGS = ("--v9-train", "--lead-train", "--speed-train", "--tracks", "--join-train", "--cache-dir", "--p10")


def parse(t):
    o, cur = {}, None
    for x in t:
        if x.startswith("--") and len(x) > 2:
            assert x not in o, f"flag {x} twice"
            o[x] = []
            cur = x
        else:
            assert cur is not None, f"value {x!r} before any flag"
            o[cur].append(x)
    return o


def pair_problems(ref_toks, toks, allowed_diff, pinned):
    """toks vs ref: the flag sets equal, the values equal except the flags in ``allowed_diff``, which must read exactly ``pinned[flag]``."""
    try:
        a, b = parse(ref_toks), parse(toks)
    except AssertionError as e:
        return [f"does not parse: {e}"]
    bad = []
    for f in a:
        if f not in b:
            bad.append(f"flag {f} of the reference is missing")
    for f in b:
        if f not in a:
            bad.append(f"flag {f} {b[f]} is not in the reference")
    for f in a:
        if f in b and a[f] != b[f] and f not in allowed_diff:
            bad.append(f"flag {f}: {b[f]} differs from the reference {a[f]}")
    for f, v in pinned.items():
        if b.get(f) != v:
            bad.append(f"flag {f} reads {b.get(f)}, pinned {v}")
    return bad


def derive(ref_toks, arm):
    t = list(ref_toks)
    assert t.count("--seed") == 1 and t.count("--out") == 1, "reference must carry --seed and --out once"
    assert t[t.index("--seed") + 1] == "0", "reference --seed is not 0"
    t[t.index("--seed") + 1] = str(SEEDS[arm])
    t[t.index("--out") + 1] = f"{D}/runs/{arm}/A"
    return t


def setf(t, f, v):
    t = list(t)
    t[t.index(f) + 1] = v
    return t


def mutants(new, root=None):
    """[(name, arm, argv)]: every deliberate regression of the new argvs (each must be refused)."""
    root = root or D
    mut = []
    for a in SEEDS:
        b = new[a]
        mut += [(f"{a}_plus_batch_32", a, setf(b, "--batch", "32")), (f"{a}_steps_61569", a, setf(b, "--steps", "61569")), (f"{a}_kin_mode_raw", a, setf(b, "--kin-mode", "raw")),
                (f"{a}_grid_changed", a, setf(b, "--grid", "pool2x5")), (f"{a}_flag_removed_fit_join", a, [x for x in b if x != "--fit-join"]),
                (f"{a}_flag_added_lr", a, b + ["--lr", "1e-3"]), (f"{a}_arm_changed", a, setf(b, "--arm", "M2a")),
                (f"{a}_seed_0", a, setf(b, "--seed", "0")), (f"{a}_seed_{5 - SEEDS[a]}_under_this_name", a, setf(b, "--seed", str(5 - SEEDS[a]))),
                (f"{a}_out_is_UE_s0", a, setf(b, "--out", f"{root}/runs/UE_s0/A")), (f"{a}_out_is_UE_s1", a, setf(b, "--out", f"{root}/runs/UE_s1/A")),
                (f"{a}_out_is_the_other_new_arm", a, setf(b, "--out", f"{root}/runs/UE_s{5 - SEEDS[a]}/A")), (f"{a}_out_outside_runs", a, setf(b, "--out", f"{root}/tmp/{a}/A"))]
        for fl in DATA_FLAGS:
            if fl in b:
                mut.append((f"{a}_data_manifest_{fl[2:]}_other", a, setf(b, fl, b[b.index(fl) + 1] + ".other")))
    return mut


def main():
    ref = json.load(open(REF))["argv"]
    new = {a: derive(ref, a) for a in SEEDS}
    for a, t in new.items():
        p = f"{D}/argv/{a}.A.json"
        if os.path.exists(p):
            assert json.load(open(p))["argv"] == t, f"{p} exists and differs"
        else:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump({"argv": t}, open(p, "w"), indent=1)
        print(a + ".A", " ".join(t))
    pin = lambda a: {"--seed": [str(SEEDS[a])], "--out": [f"{D}/runs/{a}/A"]}
    pairs = {f"{a}_vs_UE_s0": pair_problems(ref, new[a], {"--seed", "--out"}, pin(a)) for a in SEEDS}
    # the argvs of the other seeds (read-only, when present) join the family: s1 = s0 with --seed 1 and --out only
    s1p = f"{D}/argv/UE_s1.A.json"
    if os.path.exists(s1p):
        s1 = json.load(open(s1p))["argv"]
        pairs["UE_s1_vs_UE_s0 (the reps1 pair, reference)"] = pair_problems(ref, s1, {"--seed", "--out"}, {"--seed": ["1"], "--out": [f"{D}/runs/UE_s1/A"]})
    mut = mutants(new)
    reg = []
    for name, a, t in mut:
        p = pair_problems(ref, t, {"--seed", "--out"}, pin(a))
        reg.append({"arm": name, "refused": bool(p), "problems": p[:2]})
    ctl = all(not v for v in pairs.values())
    proof = {"pairs": {k: {"problems": v, "pass": not v} for k, v in pairs.items()}, "regression": reg, "all_pairs_pass": ctl, "all_regressions_refused": all(r["refused"] for r in reg),
             "n_regression": len(reg), "reference": REF, "reading": "UE_s2 and UE_s3 = UE_s0 with --seed 2 / 3 and --out <root>/runs/<arm>/A, nothing else changed"}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(proof, open(OUTD + "/PAIRING_PROOF.json", "w"), indent=1)
    print(json.dumps({"all_pairs_pass": ctl, "all_regressions_refused": proof["all_regressions_refused"], "n_regression": len(reg)}))
    return 0 if ctl and proof["all_regressions_refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
