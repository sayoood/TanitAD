"""PREREG_V8F_LAUNCH_INTENT: the argv comparison that licenses reusing DAG-ORC r2's REP runs as LAUNCH-INTENT's replicate.

LAUNCH-INTENT's REP is DAG-OACC_s's argv at trainer --seed 10 / 11 (A1 of DAG-ORC defines REP the same way). The licence is TOKEN IDENTITY of two argv lists apart from --out (and --seed, which both name 10 / 11):
  oacc  = E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{s}_dagoacc/argv.json          (the banked DAG-OACC run; its argv list, interpreter and script path not included)
  rep   = the argv list DAG-ORC r2's gate PINNED for the REP run (gate/pin.txt: the command line the box-2 chain executes)
Exit 0 iff, for both seeds, the lists have the same length and are equal token for token after removing --out and its value and --seed and its value, and the seeds are 0 / 10 and 1 / 11.
python compare_rep_argv.py [--pin-root <dir holding M2b_s{s}_dagoaccorcrep/gate/pin.txt>] > compare_rep_argv.out
"""
import argparse
import json
import shlex
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--oacc-root", default="E:/Projects/TanitAD-artifacts/v8f-dagoacc")
ap.add_argument("--pin-root", default="E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc")
a = ap.parse_args()


def pin_args(p):
    toks = [t.strip('"') for t in shlex.split(open(p, encoding="utf-8").read().strip(), posix=False)]
    i = next(k for k, t in enumerate(toks) if t.endswith("v8f_mini_train.py"))
    return toks[i + 1:]


def drop(args, flag):
    args = list(args)
    i = args.index(flag)
    v = args[i + 1]
    del args[i:i + 2]
    return args, v


ok = True
for s, want_seed in ((0, "10"), (1, "11")):
    oacc = json.load(open(f"{a.oacc_root}/M2b_s{s}_dagoacc/argv.json", encoding="utf-8"))["argv"]
    rep = pin_args(f"{a.pin_root}/M2b_s{s}_dagoaccorcrep/gate/pin.txt")
    o2, o_out = drop(oacc, "--out")
    r2, r_out = drop(rep, "--out")
    o3, o_seed = drop(o2, "--seed")
    r3, r_seed = drop(r2, "--seed")
    same = o3 == r3 and len(oacc) == len(rep)
    seeds = (o_seed, r_seed) == (str(s), want_seed)
    ok &= same and seeds
    print(f"s{s}: tokens oacc {len(oacc)} / rep {len(rep)}; equal apart from --out and --seed: {same}; seeds oacc {o_seed} / rep {r_seed} (expected {s} / {want_seed}): {seeds}")
    print(f"     --out oacc {o_out}\n     --out rep  {r_out}")
    if not same:
        print("     DIFFERENT TOKENS:", [(x, y) for x, y in zip(o3, r3) if x != y][:6])
print("LICENCE (argv identity apart from --out and --seed): " + ("HOLDS" if ok else "DOES NOT HOLD"))
sys.exit(0 if ok else 1)
