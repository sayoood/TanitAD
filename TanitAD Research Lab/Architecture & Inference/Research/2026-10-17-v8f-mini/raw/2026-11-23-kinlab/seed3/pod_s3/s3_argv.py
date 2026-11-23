#!/usr/bin/env python3
"""s3_argv.py -- the argv of the three SEED3 stage-B arms (card PREREG_V8F_SEED3, frozen sha256 75f3b191, section 2), rendered from the FINISHED s0 argv files and nothing else:

  B0_s2  = B0_s0's argv with exactly   --seed 2, --init-ckpt = UE_s2.A, --out runs/B0_s2/train
  RH_s2  = RH_s0's argv with exactly   --seed 2, --dagger-seed 2, --init-ckpt = UE_s2.A, --dagger-bank = $R/bank/s2, --dagger-bank-sha256 = sha256($R/bank/s2/bank.json), --out runs/RH_s2/train
  KI_s2  = RH_s2's argv  + `--dagger-base logged` (the two tokens, appended LAST), --out runs/KI_s2/train          (PREREG_V8F_KINPUT section 2)

Any other change is REFUSED: a third flag, a different value, a flag added / removed / duplicated, a moved token. The s0 sources are pinned by sha256 (PINS). stdlib only, CPU, sub-second.
RH_s2 and KI_s2 need the bank (its sha256), so they render AFTER it exists; B0_s2 needs none. B0's pair check (stock rollhist_pair_check.py) needs the RH argv of the same seed, which does not exist before the bank:
`render --arms B0` also writes runs/B0_s2/pair_rh_skeleton.json = RH_s2's argv with the bank sha256 spelled `@BANKSHA@` (the stock check accepts that spelling); `verify` proves the real RH_s2 argv equals the skeleton with the sha filled in.

  show   --arm B0_s2|RH_s2|KI_s2 [--bank-sha HEX|@BANKSHA@]   builds in memory, writes NOTHING, prints the differences from the s0 source (RH / KI need the sha: from bank.json, else --bank-sha)
  json   --arm A [--bank-sha ..]                              the in-memory argv as JSON on stdout
  render --arms B0_s2 [RH_s2 KI_s2]                           writes $R/runs/<arm>/argv.json (an existing file is kept iff byte-identical, else refused); B0 also the pair skeleton
  verify --arms ...                                           the EXISTING argv.json files against the s0 sources: exactly the allowed differences, nothing else
  cpu-argv --arm A --out-dir DIR [--bank-sha ..]              the in-memory argv with `--device cuda` -> `cpu` (the ONLY difference), to DIR/<arm>_argv_cpu.json
  template [--out PATH|-] [--check PATH]                                       $DS/argv/U_s2.B.tmpl.json = U_s1.B.tmpl.json with --seed / --dagger-seed 1 -> 2 and --out .../U_s1/B -> .../U_s2/B only (the study checker's stage-B template of seed 2)
Last line `ZZ S3-ARGV <CMD> PASS|FAIL`. RH_ROOT moves $R (tests)."""
import argparse
import hashlib
import json
import os
import re
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
DS = os.environ.get("S3_DS", "/root/dscale10k")
UE2 = "/root/dscale10k/runs/UE_s2/A/ckpt_last.pt"
UE0 = "/root/dscale10k/runs/UE_s0/A/ckpt_last.pt"
PLACEHOLDER = "@BANKSHA@"
#: the finished s0 argv files the arms are rendered from (sha256 of the file, MEASURED 2026-11-23 on the pod)
PINS = {"B0_s0": "2335965249b8bdd8", "RH_s0": "fe19647471f3555d"}          # sha256 prefixes (16 hex); the full values are in FACTS_POD_S3.md / SHA256_S3.txt
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BASE_FLAG, BASE_VAL = "--dagger-base", "logged"
LEVER_FLAGS = ("--dagger-label", "--dagger-mode", "--dagger-label-cache", "--dagger-kin-tb")
EXPECT = {                                    # arm -> (source arm, the flags whose VALUE may differ from the source, flags that may be appended)
    "B0_s2": ("B0_s0", {"--seed", "--init-ckpt", "--out"}, set()),
    "RH_s2": ("RH_s0", {"--seed", "--dagger-seed", "--init-ckpt", "--dagger-bank", "--dagger-bank-sha256", "--out"}, set()),
    "KI_s2": ("RH_s0", {"--seed", "--dagger-seed", "--init-ckpt", "--dagger-bank", "--dagger-bank-sha256", "--out"}, {BASE_FLAG}),
}


def load(p):
    o = json.load(open(p, encoding="utf-8"))
    return [str(x) for x in (o["argv"] if isinstance(o, dict) else o)]


def text(argv):
    return json.dumps({"argv": argv}, indent=1)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def flags(t):
    """{flag: [values]} (a flag may appear once: a duplicate is reported by `dups`)."""
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


def dups(t):
    names = [x for x in t if x.startswith("--") and len(x) > 2]
    return sorted({n for n in names if names.count(n) > 1})


def setf(t, f, val):
    t = list(t)
    i = t.index(f)
    assert i + 2 == len(t) or (t[i + 2].startswith("--") and len(t[i + 2]) > 2), f"{f} is not a single-value flag"
    t[i + 1] = str(val)
    return t


def bank_dir(root=R):
    return f"{root}/bank/s2"


def bank_sha(root=R):
    p = f"{bank_dir(root)}/bank.json"
    return fsha(p) if os.path.isfile(p) else None


def source_problems(arm, src, kind_src):
    """The s0 argv the arm is rendered from must be the finished, expected one (the pin is separate: file sha)."""
    bad, f = [], flags(src)
    if dups(src):
        bad.append(f"{kind_src} names a flag twice: {dups(src)}")
    if f.get("--seed") != ["0"]:
        bad.append(f"{kind_src} --seed reads {f.get('--seed')}, expected ['0']")
    if f.get("--init-ckpt") != [UE0]:
        bad.append(f"{kind_src} --init-ckpt reads {f.get('--init-ckpt')}, expected UE_s0.A")
    if f.get("--device") != ["cuda"]:
        bad.append(f"{kind_src} --device reads {f.get('--device')}")
    if kind_src == "B0_s0":
        if f.get("--dagger-p") != ["0"] or any(k in f for k in ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed", "--dagger-bank-readout", "--dagger-bank-kappa", BASE_FLAG) + LEVER_FLAGS):
            bad.append("B0_s0 is not the p = 0 control (it names a dagger flag)")
    else:
        if f.get("--dagger-p") != ["0.1"] or f.get("--dagger-seed") != ["0"]:
            bad.append(f"RH_s0 --dagger-p / --dagger-seed read {f.get('--dagger-p')} / {f.get('--dagger-seed')}, expected ['0.1'] / ['0']")
        if f.get("--dagger-bank-readout") != ["causal_g1x"] or f.get("--dagger-bank-kappa") != ["0.2"]:
            bad.append("RH_s0 declaration flags are not causal_g1x / 0.2")
        if any(k in f for k in (BASE_FLAG,) + LEVER_FLAGS):
            bad.append("RH_s0 already names a lever flag")
    return bad


def build(arm, src_tokens, root=R, sha=None, kind_src=None):
    """(argv, problems). src_tokens = the s0 source argv (list). sha = the bank's sha256 (RH / KI) or PLACEHOLDER."""
    bad = []
    src_arm, _allowed, _app = EXPECT[arm]
    bad += source_problems(arm, src_tokens, src_arm)
    seed = "2"
    new = list(src_tokens)
    try:
        new = setf(new, "--seed", seed)
        new = setf(new, "--init-ckpt", UE2)
        new = setf(new, "--out", f"{root}/runs/{arm}/train")
        if arm != "B0_s2":
            if sha is None or not (HEX64.match(sha) or sha == PLACEHOLDER):
                bad.append(f"{arm}: the bank sha256 is {sha!r}: needs the 64-hex sha256 of {bank_dir(root)}/bank.json (the bank does not exist yet?)")
                sha = PLACEHOLDER
            new = setf(new, "--dagger-seed", seed)
            new = setf(new, "--dagger-bank", bank_dir(root))
            new = setf(new, "--dagger-bank-sha256", sha)
        if arm == "KI_s2":
            new = new + [BASE_FLAG, BASE_VAL]
    except (ValueError, AssertionError) as e:
        bad.append(f"{arm}: cannot render from the source: {e}")
    return new, bad + diff_problems(arm, src_tokens, new, root)


def diff_problems(arm, src, new, root=R):
    """An argv (`new`) against the s0 source: exactly the card's substitutions. Independent of `build` (verify uses it on files)."""
    src_arm, allowed, app = EXPECT[arm]
    bad, a, b = [], flags(new), flags(src)
    if dups(new):
        bad.append(f"{arm} names a flag twice: {dups(new)}")
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if diff - allowed - app:
        bad.append(f"{arm} differs from {src_arm} in {sorted(diff - allowed - app)}: not a card substitution")
    gone = sorted(set(b) - set(a))
    if gone:
        bad.append(f"{arm} lacks the {src_arm} flags {gone}")
    extra = sorted(set(a) - set(b) - app)
    if extra:
        bad.append(f"{arm} adds the flags {extra} beyond {sorted(app)}")
    n_add = 2 if app else 0
    if len(new) != len(src) + n_add:
        bad.append(f"{arm} has {len(new)} tokens, {src_arm} {len(src)}: expected {len(src) + n_add}")
    else:
        iv = {src.index(f) + 1 for f in allowed if f in src}
        pos = [i for i in range(len(src)) if src[i] != new[i]]
        if any(i not in iv for i in pos):
            bad.append(f"{arm} differs from {src_arm} at token positions {pos}, allowed only the value positions {sorted(iv)}")
        if app and new[-2:] != [BASE_FLAG, BASE_VAL]:
            bad.append(f"{arm}'s last two tokens are {new[-2:]}, expected {[BASE_FLAG, BASE_VAL]}")
    want = {"--seed": ["2"], "--init-ckpt": [UE2], "--out": [f"{root}/runs/{arm}/train"]}
    if arm != "B0_s2":
        want.update({"--dagger-seed": ["2"], "--dagger-bank": [bank_dir(root)]})
        s = a.get("--dagger-bank-sha256")
        if not (s and len(s) == 1 and (HEX64.match(s[0]) or s[0] == PLACEHOLDER)):
            bad.append(f"{arm} --dagger-bank-sha256 reads {s}: not a 64-hex sha256")
    for k, v in want.items():
        if a.get(k) != v:
            bad.append(f"{arm} {k} reads {a.get(k)}, expected {v}")
    if arm == "KI_s2" and a.get(BASE_FLAG) != [BASE_VAL]:
        bad.append(f"KI_s2 {BASE_FLAG} reads {a.get(BASE_FLAG)}, expected [{BASE_VAL!r}]")
    if arm != "KI_s2" and BASE_FLAG in a:
        bad.append(f"{arm} names {BASE_FLAG}: only KI_s2 does")
    if any(k in a for k in LEVER_FLAGS):
        bad.append(f"{arm} names a lever flag {[k for k in LEVER_FLAGS if k in a]}")
    if arm == "B0_s2" and (a.get("--dagger-p") != ["0"] or any(k.startswith("--dagger-") and k != "--dagger-p" for k in a)):
        bad.append("B0_s2 is not the p = 0 control")
    if arm != "B0_s2" and a.get("--dagger-p") != ["0.1"]:
        bad.append(f"{arm} --dagger-p reads {a.get('--dagger-p')}, expected ['0.1']")
    return bad


def source_path(arm, root=R):
    return f"{root}/runs/{EXPECT[arm][0]}/argv.json"


def load_source(arm, root=R, pins=None):
    p = source_path(arm, root)
    if not os.path.isfile(p):
        return None, [f"no {p}"]
    pins = PINS if pins is None else pins
    want = pins.get(EXPECT[arm][0])
    bad = []
    if want and not fsha(p).startswith(want):
        bad.append(f"{p} sha256 {fsha(p)[:16]} is not the pinned {want}: the s0 argv changed since the kit was built")
    return load(p), bad


def in_memory(arm, root=R, sha=None, pins=None):
    src, bad = load_source(arm, root, pins)
    if src is None:
        return None, bad
    if arm != "B0_s2" and sha is None:
        sha = bank_sha(root)
    new, b2 = build(arm, src, root, sha)
    return new, bad + b2


def skeleton(root=R, pins=None):
    return in_memory("RH_s2", root, PLACEHOLDER, pins)


def emit(path, argv):
    t = text(argv)
    if os.path.isfile(path):
        return None if open(path, encoding="utf-8").read() == t else f"{path} exists and differs from the argv this would write: refused (move it aside by hand)"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(t)
    return None


def cmd_show(a):
    new, bad = in_memory(a.arm, R, a.bank_sha)
    if new is not None and not bad:
        src = load(source_path(a.arm))
        ch = [(src[i - 1], src[i], new[i]) for i in range(len(src)) if src[i] != new[i]]
        print(f"{a.arm}: {len(new)} tokens; differs from {EXPECT[a.arm][0]} in {ch} and the appended {new[len(src):]}; argv sha256 {hashlib.sha256(text(new).encode()).hexdigest()}")
    for m in bad:
        print("PROBLEM", m)
    print("ZZ S3-ARGV SHOW", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def cmd_json(a):
    new, bad = in_memory(a.arm, R, a.bank_sha)
    if bad or new is None:
        for m in bad:
            print("PROBLEM", m, file=sys.stderr)
        return 1
    print(text(new))
    return 0


def cmd_render(a):
    ok = True
    for arm in a.arms:
        new, bad = in_memory(arm, R)
        if not bad:
            err = emit(f"{R}/runs/{arm}/argv.json", new)
            bad += [err] if err else []
            if arm == "B0_s2" and not bad:
                sk, b2 = skeleton(R)
                bad += b2
                if not b2:
                    err = emit(f"{R}/runs/B0_s2/pair_rh_skeleton.json", sk)
                    bad += [err] if err else []
            print(f"{arm}: {R}/runs/{arm}/argv.json argv_sha256={hashlib.sha256(text(new).encode()).hexdigest()}")
        for m in bad:
            print("PROBLEM", arm, m)
        ok &= not bad
    print("ZZ S3-ARGV RENDER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_verify(a):
    ok = True
    for arm in a.arms:
        p = f"{R}/runs/{arm}/argv.json"
        src, bad = load_source(arm, R)
        if not os.path.isfile(p):
            bad = bad + [f"no {p}"]
        if src is not None and os.path.isfile(p):
            new = load(p)
            bad += source_problems(arm, src, EXPECT[arm][0]) + diff_problems(arm, src, new, R)
            if arm != "B0_s2":
                bs = bank_sha(R)
                if bs is None or flags(new).get("--dagger-bank-sha256") != [bs]:
                    bad.append(f"{arm} --dagger-bank-sha256 is not sha256({bank_dir(R)}/bank.json) ({bs})")
            if arm == "RH_s2":
                sk, b2 = skeleton(R)
                skp = f"{R}/runs/B0_s2/pair_rh_skeleton.json"
                if os.path.isfile(skp) and not b2:
                    f = load(skp)
                    filled = [flags(new).get("--dagger-bank-sha256")[0] if x == PLACEHOLDER else x for x in f] if flags(new).get("--dagger-bank-sha256") else f
                    if filled != new:
                        bad.append("RH_s2's argv is not B0_s2's pair skeleton with the bank sha256 filled in")
        for m in bad:
            print("PROBLEM", arm, m)
        print(f"{arm}: {'ok' if not bad else 'REFUSED'}")
        ok &= not bad
    print("ZZ S3-ARGV VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_cpu(a):
    new, bad = in_memory(a.arm, R, a.bank_sha)
    if new is not None and not bad:
        if flags(new).get("--device") != ["cuda"]:
            bad.append("--device is not cuda")
        else:
            cpu = setf(new, "--device", "cpu")
            pos = [i for i in range(len(new)) if new[i] != cpu[i]]
            if len(pos) != 1 or new[pos[0]] != "cuda" or cpu[pos[0]] != "cpu":
                bad.append(f"the CPU argv differs at {pos}, not in exactly the --device value")
            else:
                os.makedirs(a.out_dir, exist_ok=True)
                out = f"{a.out_dir}/{a.arm}_argv_cpu.json"
                open(out, "w", encoding="utf-8").write(text(cpu))
                print(f"{a.arm}: {out} argv_sha256={hashlib.sha256(text(cpu).encode()).hexdigest()} (launch argv sha256 {hashlib.sha256(text(new).encode()).hexdigest()})")
    for m in bad:
        print("PROBLEM", m)
    print("ZZ S3-ARGV CPUARGV", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def cmd_template(a):
    p1 = f"{DS}/argv/U_s1.B.tmpl.json"
    bad = [] if os.path.isfile(p1) else [f"no {p1}"]
    if not bad:
        t = load(p1)
        f = flags(t)
        if f.get("--seed") != ["1"] or f.get("--dagger-seed") != ["1"] or f.get("--out") != ["/root/dscale10k/runs/U_s1/B"]:
            bad.append(f"U_s1.B.tmpl.json reads --seed {f.get('--seed')} / --dagger-seed {f.get('--dagger-seed')} / --out {f.get('--out')}: not the expected seed-1 template")
        else:
            n = setf(setf(setf(t, "--seed", "2"), "--dagger-seed", "2"), "--out", "/root/dscale10k/runs/U_s2/B")
            pos = [i for i in range(len(t)) if t[i] != n[i]]
            if len(pos) != 3:
                bad.append(f"the seed-2 template differs from U_s1's at {pos}, not in exactly seed / dagger-seed / out")
            else:
                if a.check:
                    if not os.path.isfile(a.check) or open(a.check, encoding="utf-8").read() != text(n):
                        bad.append(f"{a.check} is not what this builds from U_s1.B.tmpl.json")
                    else:
                        print(f"{a.check} equals the template built from U_s1 (sha256={hashlib.sha256(text(n).encode()).hexdigest()})")
                elif a.out == "-":
                    print(text(n))
                else:
                    err = emit(a.out, n)
                    bad += [err] if err else []
                    print(f"{a.out} sha256={hashlib.sha256(text(n).encode()).hexdigest()}")
    for m in bad:
        print("PROBLEM", m)
    print("ZZ S3-ARGV TEMPLATE", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    for name, fn in (("show", cmd_show), ("json", cmd_json), ("cpu-argv", cmd_cpu)):
        r = sp.add_parser(name)
        r.add_argument("--arm", required=True, choices=sorted(EXPECT))
        r.add_argument("--bank-sha", default=None)
        if name == "cpu-argv":
            r.add_argument("--out-dir", required=True)
        r.set_defaults(f=fn)
    for name, fn in (("render", cmd_render), ("verify", cmd_verify)):
        r = sp.add_parser(name)
        r.add_argument("--arms", nargs="+", required=True, choices=sorted(EXPECT))
        r.set_defaults(f=fn)
    r = sp.add_parser("template")
    r.add_argument("--out", default=f"{os.path.dirname(os.path.abspath(__file__))}/U_s2.B.tmpl.json")
    r.add_argument("--check", default=None, help="compare PATH with the template this builds (writes nothing)")
    r.set_defaults(f=cmd_template)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
