#!/usr/bin/env python3
"""orct1_pullcheck.py -- is one O-RECOVER run's pull to the dev box COMPLETE?   (PREREG_V8F_ORECOVER dev-box T1 chain; CPU only; read-only)

    python orct1_pullcheck.py <ART> <arm> <seed>          arm in {orc, orc0}, seed in {0, 1}
      rc 0   prints  COMPLETE <ckpt sha256> <n files>
      rc 1   prints  INCOMPLETE <the first reason>

The completion signal is the box-2 chain's own (chain_v8f_orc.sh, `runone`): it pulls

    ckpt_last.pt  eval_M2b-<arm>_s<S>.{json,npz}  selfcheck.json  summary.json  argv.json  train_log.jsonl  (+ train.log, eval_*.log)   from box 2 into  <ART>/M2b_s<S>_<arm>/
    (launch_gate_token.json is written there by the chain BEFORE training)

and only AFTER the last `scp` returns it hashes exactly eight files into  <ART>/<run>/SHA256SUMS.txt :

    ckpt_last.pt  eval_<tag>.json  eval_<tag>.npz  selfcheck.json  summary.json  argv.json  train_log.jsonl  launch_gate_token.json

So the run is COMPLETE iff, by CONTENT and not by presence:
  1. SHA256SUMS.txt exists, is not being written right now (not modified in the last MIN_AGE_S seconds), and lists EXACTLY those eight names once each;
  2. every one of the eight files exists, is non-empty, and its streamed sha256 equals the listed one  (a half-pulled ckpt_last.pt, or a SHA256SUMS.txt left by an
     earlier attempt over files that were since replaced, fails here);
  3. the four small JSON records parse, and summary.json says done: true.
Anything else is INCOMPLETE and the caller waits.  Presence of a file, of the directory, or of an old SHA256SUMS.txt is never the signal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

MIN_AGE_S = 3.0          # SHA256SUMS.txt is written by `sha256sum | tee`: a file younger than this may still be filling
ARMS = ("orc", "orc0")
LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def run_name(arm: str, seed: int) -> str:
    return f"M2b_s{seed}_{arm}"


def eval_tag(arm: str, seed: int) -> str:
    return f"M2b-{arm}_s{seed}"


def expected_files(arm: str, seed: int) -> list[str]:
    tg = eval_tag(arm, seed)
    return ["ckpt_last.pt", f"eval_{tg}.json", f"eval_{tg}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(art: str, arm: str, seed: int, min_age_s: float = MIN_AGE_S, now: float | None = None) -> tuple[bool, str, str]:
    """-> (complete, reason, ckpt sha256 ('' unless complete))"""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}", ""
    d = os.path.join(art, run_name(arm, seed))
    if not os.path.isdir(d):
        return False, f"no run directory {d} yet", ""
    sums = os.path.join(d, "SHA256SUMS.txt")
    if not os.path.isfile(sums):
        return False, f"{d} has no SHA256SUMS.txt yet (the box-2 chain writes it after the last scp returns)", ""
    age = (time.time() if now is None else now) - os.path.getmtime(sums)
    if age < min_age_s:
        return False, f"SHA256SUMS.txt was modified {age:.1f} s ago (< {min_age_s} s): it may still be filling", ""
    want = expected_files(arm, seed)
    listed: dict[str, str] = {}
    with open(sums, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    if not text.endswith("\n"):
        return False, "SHA256SUMS.txt does not end with a newline (its last line may be partial)", ""
    for raw in text.splitlines():
        m = LINE.match(raw.rstrip("\r"))
        if not m:
            return False, f"SHA256SUMS.txt has a line that is not '<sha256>  <name>': {raw[:80]!r}", ""
        name = m.group(2).strip()
        if name in listed:
            return False, f"SHA256SUMS.txt lists {name} twice", ""
        listed[name] = m.group(1)
    if sorted(listed) != sorted(want):
        miss, extra = sorted(set(want) - set(listed)), sorted(set(listed) - set(want))
        return False, f"SHA256SUMS.txt lists {len(listed)} names, expected the {len(want)} of the box-2 chain (missing {miss}, unexpected {extra})", ""
    for name in want:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            return False, f"{name} is listed but absent", ""
        if os.path.getsize(p) == 0:
            return False, f"{name} is empty", ""
    for name in want:
        got = sha256_file(os.path.join(d, name))
        if got != listed[name]:
            return False, f"{name} sha256 {got[:12]} != the listed {listed[name][:12]} (half-pulled, or replaced after SHA256SUMS.txt was written)", ""
    for name in ("selfcheck.json", "summary.json", "argv.json", "launch_gate_token.json"):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError) as e:
            return False, f"{name} does not parse: {type(e).__name__}: {e}", ""
        if name == "summary.json" and rec.get("done") is not True:
            return False, f"summary.json done is {rec.get('done')!r}, not true (the training did not finish)", ""
    return True, f"all {len(want)} files present, non-empty, sha256-verified against SHA256SUMS.txt; summary.json done", listed["ckpt_last.pt"]


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3 or not a[2].isdigit():
        print("usage: orct1_pullcheck.py <ART> <orc|orc0> <seed>", file=sys.stderr)
        return 2
    ok, why, sha = check(a[0], a[1], int(a[2]))
    print(f"COMPLETE {sha} {why}" if ok else f"INCOMPLETE {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
