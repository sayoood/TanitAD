"""gen_argv_reps1.py -- DSCALE10K REPS1 (pod): writes argv/UE_s1.A.json and argv/UM_s1.A.json = the seed-0 argvs with --seed 1 and --out runs/<ARM>/A, nothing else changed (Amendment UM-A2 DRAFT).
Then the PAIRING PROOF (written to blockreps1/PAIRING_PROOF.json): UE_s1 vs UE_s0 and UM_s1 vs UM_s0 differ in EXACTLY --seed ('0' -> '1') and --out; UM_s1 vs UE_s1 differ in EXACTLY --out plus the three --tac-src-* flags.
Deliberate-regression arms (each must be REFUSED by pair_problems): a second lever changed (--batch 32 / --steps 61569 / --kin-mode raw), a flag removed, --seed 2, a tac flag changed.
Idempotent: an existing argv file must equal what would be written, else REFUSED. DS_ROOT overrides the root (dry run). usage: gen_argv_reps1.py"""
import json
import os
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")
OUTD = D + "/blockreps1"
TAC = ("--tac-src-w", "--tac-src-clips", "--tac-src-clips-sha256")


def parse(t):
    o, cur = {}, None
    for x in t:
        if x.startswith("--") and len(x) > 2:
            assert x not in o, f"flag {x} twice"
            o[x] = []
            cur = x
        else:
            assert cur is not None
            o[cur].append(x)
    return o


def pair_problems(a_toks, b_toks, allowed_diff, pinned, added=()):
    """b vs a: the flag sets equal (plus ``added`` on b), the values equal except the flags in ``allowed_diff``, which must read exactly ``pinned[flag]`` on b."""
    try:
        a, b = parse(a_toks), parse(b_toks)
    except AssertionError as e:
        return [f"does not parse: {e}"]
    bad = []
    for f in a:
        if f not in b:
            bad.append(f"flag {f} of the reference is missing")
    for f in b:
        if f not in a and f not in added:
            bad.append(f"flag {f} {b[f]} is not in the reference and not an allowed addition")
    for f in added:
        if f not in b:
            bad.append(f"allowed addition {f} is absent")
    for f in a:
        if f in b and a[f] != b[f] and f not in allowed_diff:
            bad.append(f"flag {f}: {b[f]} differs from the reference {a[f]}")
    for f, v in pinned.items():
        if b.get(f) != v:
            bad.append(f"flag {f} reads {b.get(f)}, pinned {v}")
    return bad


def derive(s0_toks, arm):
    t = list(s0_toks)
    t[t.index("--seed") + 1] = "1"
    t[t.index("--out") + 1] = f"{D}/runs/{arm}/A"
    return t


def main():
    s0 = {a: json.load(open(f"{D}/argv/{a}_s0.A.json"))["argv"] for a in ("UE", "UM")}
    s1 = {a: derive(s0[a], a + "_s1") for a in s0}
    for a in s1:
        p = f"{D}/argv/{a}_s1.A.json"
        if os.path.exists(p):
            assert json.load(open(p))["argv"] == s1[a], f"{p} exists and differs"
        else:
            json.dump({"argv": s1[a]}, open(p, "w"), indent=1)
        print(a + "_s1.A", " ".join(s1[a]))
    pin = lambda a: {"--seed": ["1"], "--out": [f"{D}/runs/{a}_s1/A"]}
    ok = {"UE_s1_vs_UE_s0": pair_problems(s0["UE"], s1["UE"], {"--seed", "--out"}, pin("UE")),
          "UM_s1_vs_UM_s0": pair_problems(s0["UM"], s1["UM"], {"--seed", "--out"}, pin("UM")),
          "UM_s1_vs_UE_s1": pair_problems(s1["UE"], s1["UM"], {"--out"}, {"--out": [f"{D}/runs/UM_s1/A"], "--seed": ["1"]}, added=TAC),
          "UM_s0_vs_UE_s0 (the s0 pair, reference)": pair_problems(s0["UE"], s0["UM"], {"--out"}, {"--out": [f"{D}/runs/UM_s0/A"], "--seed": ["0"]}, added=TAC)}
    reg = []

    def setf(t, f, v):
        t = list(t)
        t[t.index(f) + 1] = v
        return t

    mut = [("UE_s1_plus_batch_32", "UE", setf(s1["UE"], "--batch", "32")), ("UE_s1_steps_61569", "UE", setf(s1["UE"], "--steps", "61569")),
           ("UE_s1_seed_2", "UE", setf(s1["UE"], "--seed", "2")), ("UE_s1_kin_mode_changed", "UE", setf(s1["UE"], "--kin-mode", "raw")),
           ("UE_s1_flag_removed", "UE", [x for i, x in enumerate(s1["UE"]) if not (x == "--fit-join")]),
           ("UM_s1_plus_batch_32", "UM", setf(s1["UM"], "--batch", "32")), ("UM_s1_tac_w_0.5", "UM", setf(s1["UM"], "--tac-src-w", "0.5")),
           ("UM_s1_out_is_UM_s0", "UM", setf(s1["UM"], "--out", f"{D}/runs/UM_s0/A"))]
    for name, a, t in mut:
        p = pair_problems(s0[a], t, {"--seed", "--out"}, pin(a))
        if a == "UM":
            p += pair_problems(s1["UE"], t, {"--out"}, {"--out": [f"{D}/runs/UM_s1/A"], "--seed": ["1"]}, added=TAC)
        reg.append({"arm": name, "refused": bool(p), "problems": p[:2]})
    ctl = all(not v for k, v in ok.items())
    proof = {"pairs": {k: {"problems": v, "pass": not v} for k, v in ok.items()}, "regression": reg, "all_pairs_pass": ctl, "all_regressions_refused": all(r["refused"] for r in reg),
             "reading": "s1 argv = s0 argv with --seed 1 and --out only (UE_s1, UM_s1); UM_s1 = UE_s1 + the three --tac-src-* flags"}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(proof, open(OUTD + "/PAIRING_PROOF.json", "w"), indent=1)
    print(json.dumps({"all_pairs_pass": ctl, "all_regressions_refused": proof["all_regressions_refused"], "n_regression": len(reg)}))
    return 0 if ctl and proof["all_regressions_refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
