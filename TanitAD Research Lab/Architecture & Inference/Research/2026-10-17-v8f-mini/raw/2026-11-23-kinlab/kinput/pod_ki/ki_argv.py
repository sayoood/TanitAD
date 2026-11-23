#!/usr/bin/env python3
"""ki_argv.py -- The argv of the K-INPUT arm KI (card PREREG_V8F_KINPUT section 2, frozen sha256 05e786a8): RH's argv (RH_s{k}, 62 tokens, trained and finished) with EXACTLY TWO CHANGES:
  * `--out $R/runs/RH_s{k}/train` -> `--out $R/runs/KI_s{k}/train`;
  * the two tokens `--dagger-base logged` appended LAST.
64 tokens. SAME bank, SAME dose p = 0.1, SAME init, swap mode, human label (RH's default: no --dagger-label), no label cache. A third difference is refused. stdlib only, CPU, sub-second.

  render [--seeds 0 1]   for each seed k: read $R/runs/RH_s{k}/argv.json and $R/bank/s{k}/bank.json, and write $R/runs/KI_s{k}/argv.json = that argv with exactly the two changes. Refuses unless
                           * RH_s{k} names none of --dagger-label, --dagger-mode, --dagger-base, --dagger-label-cache, --dagger-kin-tb (RH's own argv has no lever flag), `--dagger-p` is 0.1, its `--dagger-bank` is $R/bank/s{k}
                             and its `--dagger-bank-sha256` equals sha256(bank.json);
                           * the declaration flags read causal_g1x / 0.2 and `--init-ckpt` is /root/dscale10k/runs/UE_s{k}/A/ckpt_last.pt, `--out` is RH's own;
                           * the written argv differs from RH's in exactly {--dagger-base, --out} (token list diff: one value changed, two tokens appended at the END);
                           * B0_s{k}/argv.json exists and has `--dagger-p 0` (the comparator is the EXISTING B0; no new B0).
                         An existing KI argv.json is kept iff it is byte-identical to the one this would write, else refused (never silently rewritten).
  verify [--seeds 0 1]   for each seed k: the EXISTING KI_s{k}/argv.json against RH_s{k}/argv.json: exactly the two differences above, nothing else. Refuses a hand-edited third difference.
  cpu-argv [--seeds 0 1] --out-dir DIR   the SAME argv as `render` would write (built in memory from RH_s{k}; NOTHING is written under runs/), with `--device cuda` -> `cpu` (the ONLY difference), to DIR/s{k}_argv_cpu.json.
                         This is the argv of the DRY MINT (the trainer's --selfcheck-only on the CPU). Refuses if an existing KI_s{k}/argv.json differs from the in-memory argv.
  show [--seeds 0 1]     like `render` but WRITES NOTHING: builds each KI argv in memory from RH_s{k}, checks it (an existing KI_s{k}/argv.json, if any, must be identical) and prints its two differences from RH. Used by the chain's pre-lock validation and by DRY=1.
  json --seeds K         the in-memory KI argv of ONE seed as JSON on stdout (writes nothing; rc 1 and the problems on stderr if it is refused): piped into ki_gate_record.py precheck --argv - by the chain's pre-lock walk.
  Last line: `ZZ KI-RENDER PASS` / `ZZ KI-RENDER FAIL`, `ZZ KI-VERIFY ...`, `ZZ KI-CPUARGV ...`, or `ZZ KI-SHOW ...`.
  The stock pair check is run by ki_arm.sh on the filled argv minus the two trailing tokens (`--dagger-base logged`), against B0."""
import argparse
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
P_RH = "0.1"                                   # KI keeps RH's dose; its ONLY changes are --dagger-base logged and --out
BASE_FLAG = "--dagger-base"
BASE_VAL = "logged"
NOT_IN_RH = ("--dagger-label", "--dagger-mode", "--dagger-base", "--dagger-label-cache", "--dagger-kin-tb")   # the lever flags RH's own argv must not carry


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




def diff_problems(rh, ki, seed, root=R):
    """RH argv vs KI argv (token lists): [problems]. KI must equal RH token for token except the --out value and the two tokens `--dagger-base logged` appended at the end (card section 2)."""
    bad = []
    a, b = flags(ki), flags(rh)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != [BASE_FLAG, "--out"]:
        bad.append(f"KI differs from RH in {diff}, expected exactly ['{BASE_FLAG}', '--out']")
    if a.get(BASE_FLAG) != [BASE_VAL]:
        bad.append(f"KI {BASE_FLAG} reads {a.get(BASE_FLAG)}, expected [{BASE_VAL!r}]")
    if ki.count(BASE_FLAG) != 1:
        bad.append(f"KI names {BASE_FLAG} {ki.count(BASE_FLAG)} times, expected once")
    if a.get("--out") != [f"{root}/runs/KI_s{seed}/train"]:
        bad.append(f"KI --out reads {a.get('--out')}, expected the arm's own {root}/runs/KI_s{seed}/train")
    if len(ki) != len(rh) + 2:
        bad.append(f"KI has {len(ki)} tokens, RH {len(rh)}: the argv adds exactly two ({BASE_FLAG} and its value)")
    else:                                      # token by token: the only position that may differ is the --out value; the last two tokens are the base pair
        pos = [i for i in range(len(rh)) if rh[i] != ki[i]]
        iO = rh.index("--out") + 1 if "--out" in rh else -1
        if any(i != iO for i in pos):
            bad.append(f"KI differs from RH at token positions {pos}, allowed only {[iO]}")
        if ki[-2:] != [BASE_FLAG, BASE_VAL]:
            bad.append(f"KI's last two tokens are {ki[-2:]}, expected [{BASE_FLAG!r}, {BASE_VAL!r}]")
    return bad


def build(rh, seed, root=R, bank_sha=None):
    """RH argv (list) -> (KI argv, [problems]).  bank_sha: the sha256 of bank.json (None = read it from disk; tests inject it)."""
    bad = []
    f = flags(rh)
    for k in NOT_IN_RH:
        if k in f:
            bad.append(f"RH_s{seed} already names {k} {f[k]}: RH has no lever flag, KI adds {BASE_FLAG} only")
    if f.get("--dagger-p") != [P_RH]:
        bad.append(f"RH_s{seed} --dagger-p reads {f.get('--dagger-p')}, expected [{P_RH!r}]")
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
    if f.get("--init-ckpt") != [f"/root/dscale10k/runs/UE_s{seed}/A/ckpt_last.pt"]:
        bad.append(f"RH_s{seed} --init-ckpt reads {f.get('--init-ckpt')}: KI inits from UE_s{seed}.A as RH did")
    if f.get("--out") != [f"{root}/runs/RH_s{seed}/train"]:
        bad.append(f"RH_s{seed} --out reads {f.get('--out')}")
    new = list(rh)
    if "--out" in f:
        new = setf(new, "--out", [f"{root}/runs/KI_s{seed}/train"])
    new = new + [BASE_FLAG, BASE_VAL]
    bad += diff_problems(rh, new, seed, root)
    return new, bad


def _text(argv):
    return json.dumps({"argv": argv}, indent=1)


def _inputs(s):
    rhp, b0p = f"{R}/runs/RH_s{s}/argv.json", f"{R}/runs/B0_s{s}/argv.json"
    return rhp, b0p, [f"no {p}" for p in (rhp, b0p) if not os.path.isfile(p)]


def cmd_render(a):
    ok = True
    for s in a.seeds:
        rhp, b0p, bad = _inputs(s)
        if not bad:
            new, bad = build(load(rhp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
        if not bad:
            d = f"{R}/runs/KI_s{s}"
            out = d + "/argv.json"
            text = _text(new)
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
    print("ZZ KI-RENDER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_cpu_argv(a):
    ok = True
    for s in a.seeds:
        rhp, b0p, bad = _inputs(s)
        if not bad:
            new, bad = build(load(rhp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/KI_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != _text(new):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            if flags(new).get("--device") != ["cuda"]:
                bad.append(f"KI argv --device reads {flags(new).get('--device')}, expected ['cuda'] (the CPU form changes only this)")
            else:
                cpu = setf(new, "--device", ["cpu"])
                pos = [i for i in range(len(new)) if new[i] != cpu[i]]
                if len(cpu) != len(new) or len(pos) != 1 or new[pos[0]] != "cuda" or cpu[pos[0]] != "cpu":
                    bad.append(f"the CPU argv differs from the KI argv at {pos}, not in exactly the --device value")
        if not bad:
            os.makedirs(a.out_dir, exist_ok=True)
            out = f"{a.out_dir}/s{s}_argv_cpu.json"
            text = _text(cpu)
            open(out, "w", encoding="utf-8").write(text)
            print(f"s{s}: {out} argv_sha256={hashlib.sha256(text.encode()).hexdigest()} (KI argv sha256 {hashlib.sha256(_text(new).encode()).hexdigest()})")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ KI-CPUARGV", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_show(a):
    ok = True
    for s in a.seeds:
        rhp, b0p, bad = _inputs(s)
        if not bad:
            rh = load(rhp)
            new, bad = build(rh, s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/KI_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != _text(new):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            pos = [i for i in range(len(rh)) if rh[i] != new[i]]
            print(f"s{s}: KI argv = RH argv with {[(rh[i - 1], rh[i], new[i]) for i in pos]} and the appended {new[len(rh):]}; {len(new)} tokens; sha256 {hashlib.sha256(_text(new).encode()).hexdigest()}")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ KI-SHOW", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):
    ok = True
    for s in a.seeds:
        rhp, kip = f"{R}/runs/RH_s{s}/argv.json", f"{R}/runs/KI_s{s}/argv.json"
        bad = [f"no {p}" for p in (rhp, kip) if not os.path.isfile(p)]
        if not bad:
            bad = diff_problems(load(rhp), load(kip), s, R)
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        print(f"s{s}: {'ok' if not bad else 'REFUSED'}")
        ok &= not bad
    print("ZZ KI-VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_json(a):
    """Print ONE seed's in-memory KI argv as JSON on stdout (nothing else on stdout; problems on stderr, rc 1). Writes nothing: the pre-lock walk pipes it into ki_gate_record.py precheck --argv -."""
    s = a.seeds[0]
    rhp, b0p, bad = _inputs(s)
    if not bad:
        new, bad = build(load(rhp), s, R)
        if flags(load(b0p)).get("--dagger-p") != ["0"]:
            bad.append(f"B0_s{s} --dagger-p is not 0")
        ex = f"{R}/runs/KI_s{s}/argv.json"
        if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != _text(new):
            bad.append(f"{ex} exists and differs from the argv this builds")
    if bad:
        for m in bad:
            print("PROBLEM", f"s{s}", m, file=sys.stderr)
        return 1
    print(_text(new))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    for name, fn in (("render", cmd_render), ("verify", cmd_verify), ("cpu-argv", cmd_cpu_argv), ("show", cmd_show), ("json", cmd_json)):
        r = sp.add_parser(name)
        r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        if name == "cpu-argv":
            r.add_argument("--out-dir", required=True)
        r.set_defaults(f=fn)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
