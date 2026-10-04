#!/usr/bin/env python
"""WP-RL: write the CANONICAL argv of every arm (SPEC_RL sec. 2) and audit them (I-11).

    python wprl_argv.py --steps K --batch 32 --micro 8 --out-dir <dir>

Writes ARGV_<arm>.json (a JSON list: the python argv after the interpreter) and ARGV_AUDIT.json
(each arm vs RL: the differing flags must be exactly the arm's named variable plus --out-dir).
The Master Mind's PASS token binds the sha256 of each ARGV file; wprl_arm.sh refuses any other.
"""
import argparse
import hashlib
import json
from pathlib import Path

R = "/home/nvidia/refcv7_post/rl"
LOCK = "/home/nvidia/refcv7_post/thor_gpu.lock"
W = f"{R}/out/windows_train.json"

ARMS = {
    "rl_s0": {"--arm": "rl", "--seed": "0"},
    "rloff_s0": {"--arm": "rloff", "--seed": "0", "--rl-weight": "0"},
    "rlshuf_s0": {"--arm": "rlshuf", "--seed": "0"},
    "rl_s1": {"--arm": "rl", "--seed": "1"},
}
#: the one named variable of each arm (SPEC_RL sec. 2), besides --out-dir
NAMED = {"rl_s0": set(), "rloff_s0": {"--arm", "--rl-weight"}, "rlshuf_s0": {"--arm"},
         "rl_s1": {"--seed"}}


def argv_for(arm, steps, batch, micro):
    flags = {"--arm": None, "--seed": None, "--rl-weight": "1.0", "--steps": str(steps),
             "--batch": str(batch), "--micro": str(micro), "--groups": "4", "--lr": "0.0002",
             "--weight-decay": "0.0001", "--il-form": "matched_anchor", "--grad-clip": "100",
             "--dac-rule": "R2", "--windows": W, "--out-dir": f"{R}/arms/{arm}",
             "--inproc-lock": LOCK, "--segment-minutes": "45", "--min-segment-minutes": "5",
             "--ckpt-every": "25", "--workers": "4", "--log-every": "1"}
    flags.update(ARMS[arm])
    out = ["stack/scripts/ddv2_rl_refcv7.py", "train"]
    for k, v in flags.items():
        out += [k, v]
    return out, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    flags_of, audit = {}, {}
    for arm in ARMS:
        argv, flags = argv_for(arm, a.steps, a.batch, a.micro)
        p = od / f"ARGV_{arm}.json"
        p.write_text(json.dumps(argv), encoding="utf-8")
        flags_of[arm] = flags
        audit[arm] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    ref = flags_of["rl_s0"]
    ok = True
    for arm, f in flags_of.items():
        diff = sorted(k for k in set(ref) | set(f) if ref.get(k) != f.get(k) and k != "--out-dir")
        audit[arm]["differs_from_rl_s0"] = diff
        audit[arm]["named_variable"] = sorted(NAMED[arm])
        audit[arm]["I11_pass"] = set(diff) == NAMED[arm]
        ok &= audit[arm]["I11_pass"]
    rec = {"I11_all_pass": ok, "arms": audit}
    (od / "ARGV_AUDIT.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
