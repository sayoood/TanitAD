#!/usr/bin/env python3
"""rhk_argv.py -- the argv of the ROLLHIST arm RHK (RH's argv with a bank generated under readout K). stdlib only, CPU, sub-second. STAGED (the MM copies it to /root/rollhist/), NOT RUN.

  render [--seeds 0 1]   for each seed k: read $R/runs/RH_s{k}/argv.json (the argv of record, trained) and $R/bank_k/s{k}/bank.json, and write $R/runs/RHK_s{k}/argv.json = that argv with EXACTLY the bank flags
                         changed plus --out:  --dagger-bank -> $R/bank_k/s{k};  --dagger-bank-sha256 -> sha256(bank_k bank.json);  --dagger-bank-readout causal_g1x -> causal_g0x;  --out -> $R/runs/RHK_s{k}/train.
                         (--dagger-bank-kappa stays 0.2, --dagger-p stays 0.1, init, steps, seeds, lr, everything else identical.) Refuses unless
                           * RH_s{k}'s --dagger-p is 0.1, its bank is $R/bank/s{k}, its token causal_g1x and kappa 0.2;
                           * bank_k/s{k}/bank.json exists, its common.generator is the K record {causal_g, acc_gain 0.0, path_exec true, kappa_max "0.2"}, and its sha256 differs from the RH bank's;
                           * the token-list diff against RH is exactly {--dagger-bank, --dagger-bank-sha256, --dagger-bank-readout, --out};  B0_s{k}/argv.json exists with --dagger-p 0 (no new B0).
                         An existing RHK argv.json is kept iff byte-identical to what this would write, else refused (never silently rewritten).
  Last line: `ZZ RHK-RENDER PASS` or `ZZ RHK-RENDER FAIL`."""
import argparse
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
P_RH, TOK_RH, TOK_K = "0.1", "causal_g1x", "causal_g0x"
K_GEN = {"readout": "causal_g", "acc_gain": 0.0, "path_exec": True, "kappa_max": "0.2"}
CHANGED = ["--dagger-bank", "--dagger-bank-readout", "--dagger-bank-sha256", "--out"]


def load(p):
    o = json.load(open(p, encoding="utf-8"))
    return [str(x) for x in (o["argv"] if isinstance(o, dict) else o)]


def setf(t, f, vals):
    t = list(t)
    assert f in t, f"{f} is not in the argv"
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + [str(v) for v in vals] + t[j:]


def flags(t):
    out, i = {}, 0
    while i < len(t):
        if t[i].startswith("--") and len(t[i]) > 2:
            j = i + 1
            while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
                j += 1
            out[t[i]] = t[i + 1:j]
            i = j
        else:
            i += 1
    return out


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build(rh, seed, root=R, k_bank_sha=None, k_generator=None, rh_bank_sha=None):
    """RH argv -> (RHK argv, [problems]). k_bank_sha / k_generator / rh_bank_sha: read from disk when None (tests inject them)."""
    bad = []
    f = flags(rh)
    if f.get("--dagger-p") != [P_RH]:
        bad.append(f"RH_s{seed} --dagger-p reads {f.get('--dagger-p')}, expected [{P_RH!r}]")
    if f.get("--dagger-bank") != [f"{root}/bank/s{seed}"]:
        bad.append(f"RH_s{seed} --dagger-bank reads {f.get('--dagger-bank')}, expected the RH bank {root}/bank/s{seed}")
    if f.get("--dagger-bank-readout") != [TOK_RH] or f.get("--dagger-bank-kappa") != ["0.2"]:
        bad.append("RH declaration flags are not causal_g1x / 0.2")
    if f.get("--out") != [f"{root}/runs/RH_s{seed}/train"]:
        bad.append(f"RH_s{seed} --out reads {f.get('--out')}")
    kb = f"{root}/bank_k/s{seed}"
    kj = f"{kb}/bank.json"
    if k_bank_sha is None:
        if not os.path.isfile(kj):
            bad.append(f"no {kj}: the readout-K bank is not built yet")
            return list(rh), bad
        k_bank_sha = sha256(kj)
        k_generator = (json.load(open(kj, encoding="utf-8")).get("common") or {}).get("generator")
    if k_generator != K_GEN:
        bad.append(f"bank_k bank.json common.generator reads {k_generator!r}, want {K_GEN!r}")
    if rh_bank_sha is None and os.path.isfile(f"{root}/bank/s{seed}/bank.json"):
        rh_bank_sha = sha256(f"{root}/bank/s{seed}/bank.json")
    if rh_bank_sha is not None and k_bank_sha == rh_bank_sha:
        bad.append("the readout-K bank.json is byte-identical to the RH bank's: not a new bank")
    new = list(rh)
    for fl, v in (("--dagger-bank", kb), ("--dagger-bank-sha256", k_bank_sha), ("--dagger-bank-readout", TOK_K), ("--out", f"{root}/runs/RHK_s{seed}/train")):
        new = setf(new, fl, [v])
    a, b = flags(new), f
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != CHANGED:
        bad.append(f"RHK differs from RH in {diff}, expected {CHANGED}")
    return new, bad


def cmd_render(a):
    ok = True
    for s in a.seeds:
        rhp, b0p = f"{R}/runs/RH_s{s}/argv.json", f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rhp, b0p) if not os.path.isfile(p)]
        if not bad:
            new, bad = build(load(rhp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
        if not bad:
            d = f"{R}/runs/RHK_s{s}"
            out = d + "/argv.json"
            text = json.dumps({"argv": new}, indent=1)
            if os.path.isfile(out):
                if open(out, encoding="utf-8").read() != text:
                    bad.append(f"{out} exists and differs from the argv this would write: refused (move it aside by hand)")
            else:
                os.makedirs(d, exist_ok=True)
                open(out, "w", encoding="utf-8").write(text)
            print(f"s{s}: {out} argv_sha256={hashlib.sha256(text.encode()).hexdigest()}")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ RHK-RENDER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("render")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.set_defaults(f=cmd_render)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
