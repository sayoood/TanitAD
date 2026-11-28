#!/usr/bin/env python3
"""rl_argv.py -- the argv of the ROLLHIST next arm RL (the RH argv + exactly `--dagger-label speed_v1`, SAME bank, SAME dose p = 0.1). stdlib only, CPU, sub-second. STAGED (the MM copies it to /root/rollhist/), NOT RUN.

  render [--seeds 0 1]   for each seed k: read $R/runs/RH_s{k}/argv.json (the argv of record, trained and finished) and $R/bank/s{k}/bank.json, and write $R/runs/RL_s{k}/argv.json = that argv with
                         EXACTLY two changes: the added flag `--dagger-label speed_v1` (two tokens) and `--out $R/runs/RL_s{k}/train`. Refuses unless
                           * RH_s{k}'s `--dagger-p` is 0.1, its `--dagger-bank-sha256` equals sha256(bank.json) and `--dagger-bank` is $R/bank/s{k} (same bank, same declaration);
                           * the written argv differs from RH's in exactly {--dagger-p, --out} (token list diff, in order);
                           * B0_s{k}/argv.json exists and has `--dagger-p 0` (the comparator is the EXISTING B0; no new B0).
                         An existing RL argv.json is kept iff it is byte-identical to the one this would write, else refused (never silently rewritten).
  Last line: `ZZ RL-RENDER PASS` or `ZZ RL-RENDER FAIL`.   The stock pair check is run by rl_arm.sh on the filled argv, as for RH."""
import argparse
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
P_RH = "0.1"                                   # RL keeps RH's dose; its ONLY changes are --dagger-label speed_v1 and --out
LABEL = ["speed_v1"]


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


def build(rh, seed, root=R, bank_sha=None):
    """RH argv (list) -> (RL argv, [problems]).  bank_sha: the sha256 of bank.json (None = read it from disk; tests inject it)."""
    bad = []
    f = flags(rh)
    if f.get("--dagger-p") != [P_RH]:
        bad.append(f"RH_s{seed} --dagger-p reads {f.get('--dagger-p')}, expected [{P_RH!r}]")
    if "--dagger-label" in f:
        bad.append(f"RH_s{seed} already names --dagger-label {f['--dagger-label']}: RH is the human-label arm")
    bank = f"{root}/bank/s{seed}"
    if f.get("--dagger-bank") != [bank]:
        bad.append(f"RH_s{seed} --dagger-bank reads {f.get('--dagger-bank')}, expected [{bank!r}]")
    bj = f"{bank}/bank.json"
    if bank_sha is None and not os.path.isfile(bj):
        bad.append(f"no {bj}")
    elif f.get("--dagger-bank-sha256") != [bank_sha or sha256(bj)]:
        bad.append(f"RH_s{seed} --dagger-bank-sha256 is not sha256({bj}): the bank changed since RH trained on it")
    if f.get("--dagger-bank-readout") != ["causal_g1x"] or f.get("--dagger-bank-kappa") != ["0.2"]:
        bad.append("RH declaration flags are not causal_g1x / 0.2")
    if f.get("--out") != [f"{root}/runs/RH_s{seed}/train"]:
        bad.append(f"RH_s{seed} --out reads {f.get('--out')}")
    new = setf(list(rh), "--out", [f"{root}/runs/RL_s{seed}/train"]) + ["--dagger-label", "speed_v1"]
    a, b = flags(new), f
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != ["--dagger-label", "--out"] or a.get("--dagger-label") != LABEL or len(new) != len(rh) + 2:
        bad.append(f"RL differs from RH in {diff} (+{len(new) - len(rh)} tokens), expected ['--dagger-label', '--out'] and exactly two added tokens")
    return new, bad


def cmd_render(a):
    ok = True
    for s in a.seeds:
        rhp = f"{R}/runs/RH_s{s}/argv.json"
        b0p = f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rhp, b0p) if not os.path.isfile(p)]
        if not bad:
            new, bad = build(load(rhp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
        if not bad:
            d = f"{R}/runs/RL_s{s}"
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
    print("ZZ RL-RENDER", "PASS" if ok else "FAIL")
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
