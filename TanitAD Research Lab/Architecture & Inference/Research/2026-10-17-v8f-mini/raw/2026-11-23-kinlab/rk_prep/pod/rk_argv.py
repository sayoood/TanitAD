#!/usr/bin/env python3
"""rk_argv.py -- REVISION 2. The argv of the KINLAB arm RK: RL's argv with EXACTLY THE THREE CHANGES of Amendment 1 section 2 (frozen, sha256 948670d9): `--dagger-label speed_v1` -> `kin_v1`, its own `--out`, and the added execution-only
token pair `--dagger-label-cache <R>/kin_cache/s<K>_kin_v1.npz` appended LAST (after the label). SAME bank, SAME dose p = 0.1, SAME init, swap mode. A fourth difference is refused. stdlib only, CPU, sub-second.
Card PREREG_V8F_KINLAB section 2 + Amendment 1 section 2.

  render [--seeds 0 1]   for each seed k: read $R/runs/RL_s{k}/argv.json (the argv of record, trained and finished) and $R/bank/s{k}/bank.json, and write $R/runs/RK_s{k}/argv.json = that argv with
                         EXACTLY three changes: the value of `--dagger-label` (speed_v1 -> kin_v1, same position), `--out $R/runs/RK_s{k}/train`, and the two appended tokens `--dagger-label-cache $R/kin_cache/s{k}_kin_v1.npz` (the cache of THIS seed's bank only). Refuses unless
                           * RL_s{k}'s `--dagger-label` is exactly [speed_v1] (the last two tokens), `--dagger-p` is 0.1, no `--dagger-mode`, its `--dagger-bank-sha256` equals sha256(bank.json) and `--dagger-bank` is $R/bank/s{k};
                           * the declaration flags read causal_g1x / 0.2 and `--init-ckpt` is /root/dscale10k/runs/UE_s{k}/A/ckpt_last.pt;
                           * the written argv differs from RL's in exactly {--dagger-label, --dagger-label-cache, --out} (token list diff, in order: two values changed, two tokens appended at the END);
                           * B0_s{k}/argv.json exists and has `--dagger-p 0` (the comparator is the EXISTING B0; no new B0).
                         An existing RK argv.json is kept iff it is byte-identical to the one this would write, else refused (never silently rewritten).
  verify [--seeds 0 1]   for each seed k: the EXISTING RK_s{k}/argv.json against RL_s{k}/argv.json: exactly the three differences above, nothing else. Refuses a hand-edited fourth difference.
  cpu-argv [--seeds 0 1] --out-dir DIR   for each seed k: the SAME argv as `render` would write (built in memory from RL_s{k}, checked; NOTHING is written under runs/), with `--device cuda` -> `cpu` (the ONLY difference), to
                         DIR/s{k}_argv_cpu.json. This is the argv of the cache precompute (the trainer's --selfcheck-only on the CPU). Refuses if an existing RK_s{k}/argv.json differs from the in-memory argv.
  show [--seeds 0 1]     like `render` but WRITES NOTHING: builds each RK argv in memory from RL_s{k}, checks it (and an existing RK_s{k}/argv.json, if any, must be identical) and prints its three differences from RL. Used by the chain's
                         pre-lock validation and by DRY=1.
  Last line: `ZZ RK-RENDER PASS` / `ZZ RK-RENDER FAIL`, `ZZ RK-VERIFY PASS` / `ZZ RK-VERIFY FAIL`, `ZZ RK-CPUARGV PASS` / `ZZ RK-CPUARGV FAIL`, or `ZZ RK-SHOW PASS` / `ZZ RK-SHOW FAIL`.
  The stock pair check is run by rk_arm.sh on the filled argv minus the four trailing tokens (label pair and cache pair), against B0."""
import argparse
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
P_RH = "0.1"                                   # RK keeps RL's (= RH's) dose; its ONLY changes are --dagger-label kin_v1 and --out
OLD_LABEL = "speed_v1"
NEW_LABEL = "kin_v1"
CACHE_FLAG = "--dagger-label-cache"


def cache_path(root, seed):
    """The cache of seed k's bank (Amendment 1): the only path the RK argv may name."""
    return f"{root}/kin_cache/s{seed}_kin_v1.npz"


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


def diff_problems(rl, rk, seed, root=R):
    """RL argv vs RK argv (token lists): [problems]. RK must equal RL token for token except the label value, the --out value and the two tokens `--dagger-label-cache <path>` appended at the end (Amendment 1 section 2)."""
    bad = []
    a, b = flags(rk), flags(rl)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff != ["--dagger-label", CACHE_FLAG, "--out"]:
        bad.append(f"RK differs from RL in {diff}, expected exactly ['--dagger-label', '{CACHE_FLAG}', '--out']")
    want = cache_path(root, seed)
    if a.get(CACHE_FLAG) != [want]:
        bad.append(f"RK {CACHE_FLAG} reads {a.get(CACHE_FLAG)}, expected exactly [{want!r}]: the cache of THIS seed's bank (never the other seed's, never another path or kind)")
    if rk.count(CACHE_FLAG) != 1:
        bad.append(f"RK names {CACHE_FLAG} {rk.count(CACHE_FLAG)} times, expected once")
    if a.get("--dagger-label") != [NEW_LABEL]:
        bad.append(f"RK --dagger-label reads {a.get('--dagger-label')}, expected [{NEW_LABEL!r}]")
    if a.get("--out") != [f"{root}/runs/RK_s{seed}/train"]:
        bad.append(f"RK --out reads {a.get('--out')}, expected the arm's own {root}/runs/RK_s{seed}/train")
    if len(rk) != len(rl) + 2:
        bad.append(f"RK has {len(rk)} tokens, RL {len(rl)}: the label swap keeps the token count and the cache adds exactly two ({CACHE_FLAG} and its path)")
    else:                                      # token by token: the only positions that may differ are the two values; the last two tokens are the cache pair
        pos = [i for i in range(len(rl)) if rl[i] != rk[i]]
        iL = rl.index("--dagger-label") + 1 if "--dagger-label" in rl else -1
        iO = rl.index("--out") + 1 if "--out" in rl else -1
        if any(i not in (iL, iO) for i in pos):
            bad.append(f"RK differs from RL at token positions {pos}, allowed only {[iL, iO]}")
        if rk[-2:] != [CACHE_FLAG, want]:
            bad.append(f"RK's last two tokens are {rk[-2:]}, expected [{CACHE_FLAG!r}, {want!r}]")
    return bad


def build(rl, seed, root=R, bank_sha=None):
    """RL argv (list) -> (RK argv, [problems]).  bank_sha: the sha256 of bank.json (None = read it from disk; tests inject it)."""
    bad = []
    f = flags(rl)
    if f.get("--dagger-label") != [OLD_LABEL] or rl[-2:] != ["--dagger-label", OLD_LABEL]:
        bad.append(f"RL_s{seed} --dagger-label reads {f.get('--dagger-label')} (last two tokens {rl[-2:]}), expected [{OLD_LABEL!r}] as the last two tokens")
    if f.get("--dagger-p") != [P_RH]:
        bad.append(f"RL_s{seed} --dagger-p reads {f.get('--dagger-p')}, expected [{P_RH!r}]")
    if "--dagger-mode" in f:
        bad.append(f"RL_s{seed} already names --dagger-mode {f['--dagger-mode']}: RK is a swap-mode arm")
    bank = f"{root}/bank/s{seed}"
    if f.get("--dagger-bank") != [bank]:
        bad.append(f"RL_s{seed} --dagger-bank reads {f.get('--dagger-bank')}, expected [{bank!r}]")
    bj = f"{bank}/bank.json"
    if bank_sha is None and not os.path.isfile(bj):
        bad.append(f"no {bj}")
    elif f.get("--dagger-bank-sha256") != [bank_sha or sha256(bj)]:
        bad.append(f"RL_s{seed} --dagger-bank-sha256 is not sha256({bj}): the bank changed since RL trained on it")
    if f.get("--dagger-bank-readout") != ["causal_g1x"] or f.get("--dagger-bank-kappa") != ["0.2"]:
        bad.append("RL declaration flags are not causal_g1x / 0.2")
    if f.get("--init-ckpt") != [f"/root/dscale10k/runs/UE_s{seed}/A/ckpt_last.pt"]:
        bad.append(f"RL_s{seed} --init-ckpt reads {f.get('--init-ckpt')}: RK inits from UE_s{seed}.A as RL did")
    if f.get("--out") != [f"{root}/runs/RL_s{seed}/train"]:
        bad.append(f"RL_s{seed} --out reads {f.get('--out')}")
    new = list(rl)
    if "--out" in f:
        new = setf(new, "--out", [f"{root}/runs/RK_s{seed}/train"])
    if "--dagger-label" in f:
        new = setf(new, "--dagger-label", [NEW_LABEL])
    if CACHE_FLAG in f:
        bad.append(f"RL_s{seed} already names {CACHE_FLAG} {f[CACHE_FLAG]}: RK adds it, RL has none")
    new = new + [CACHE_FLAG, cache_path(root, seed)]
    bad += diff_problems(rl, new, seed, root)
    return new, bad


def cmd_render(a):
    ok = True
    for s in a.seeds:
        rlp = f"{R}/runs/RL_s{s}/argv.json"
        b0p = f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, b0p) if not os.path.isfile(p)]
        if not bad:
            new, bad = build(load(rlp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
        if not bad:
            d = f"{R}/runs/RK_s{s}"
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
    print("ZZ RK-RENDER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_cpu_argv(a):
    ok = True
    for s in a.seeds:
        rlp = f"{R}/runs/RL_s{s}/argv.json"
        b0p = f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, b0p) if not os.path.isfile(p)]
        if not bad:
            new, bad = build(load(rlp), s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/RK_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != json.dumps({"argv": new}, indent=1):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            if flags(new).get("--device") != ["cuda"]:
                bad.append(f"RK argv --device reads {flags(new).get('--device')}, expected ['cuda'] (the CPU form changes only this)")
            else:
                cpu = setf(new, "--device", ["cpu"])
                pos = [i for i in range(len(new)) if new[i] != cpu[i]]
                if len(cpu) != len(new) or len(pos) != 1 or new[pos[0]] != "cuda" or cpu[pos[0]] != "cpu":
                    bad.append(f"the CPU argv differs from the RK argv at {pos}, not in exactly the --device value")
        if not bad:
            os.makedirs(a.out_dir, exist_ok=True)
            out = f"{a.out_dir}/s{s}_argv_cpu.json"
            text = json.dumps({"argv": cpu}, indent=1)
            open(out, "w", encoding="utf-8").write(text)
            print(f"s{s}: {out} argv_sha256={hashlib.sha256(text.encode()).hexdigest()} (RK argv sha256 {hashlib.sha256(json.dumps({'argv': new}, indent=1).encode()).hexdigest()})")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ RK-CPUARGV", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_show(a):
    ok = True
    for s in a.seeds:
        rlp, b0p = f"{R}/runs/RL_s{s}/argv.json", f"{R}/runs/B0_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, b0p) if not os.path.isfile(p)]
        if not bad:
            rl = load(rlp)
            new, bad = build(rl, s, R)
            if flags(load(b0p)).get("--dagger-p") != ["0"]:
                bad.append(f"B0_s{s} --dagger-p is not 0")
            ex = f"{R}/runs/RK_s{s}/argv.json"
            if os.path.isfile(ex) and open(ex, encoding="utf-8").read() != json.dumps({"argv": new}, indent=1):
                bad.append(f"{ex} exists and differs from the argv this builds")
        if not bad:
            pos = [i for i in range(len(rl)) if rl[i] != new[i]]
            print(f"s{s}: RK argv = RL argv with {[(rl[i - 1], rl[i], new[i]) for i in pos]} and the appended {new[len(rl):]}; {len(new)} tokens; sha256 {hashlib.sha256(json.dumps({'argv': new}, indent=1).encode()).hexdigest()}")
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        ok &= not bad
    print("ZZ RK-SHOW", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):
    ok = True
    for s in a.seeds:
        rlp, rkp = f"{R}/runs/RL_s{s}/argv.json", f"{R}/runs/RK_s{s}/argv.json"
        bad = [f"no {p}" for p in (rlp, rkp) if not os.path.isfile(p)]
        if not bad:
            bad = diff_problems(load(rlp), load(rkp), s, R)
        for m in bad:
            print("PROBLEM", f"s{s}", m)
        print(f"s{s}: {'ok' if not bad else 'REFUSED'}")
        ok &= not bad
    print("ZZ RK-VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    for name, fn in (("render", cmd_render), ("verify", cmd_verify), ("cpu-argv", cmd_cpu_argv), ("show", cmd_show)):
        r = sp.add_parser(name)
        r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        if name == "cpu-argv":
            r.add_argument("--out-dir", required=True)
        r.set_defaults(f=fn)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
