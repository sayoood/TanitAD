"""chain_helpers_r8.py -- the small, testable pieces `chain_battery_r8.sh` needs (no GPU, no heavy imports; run with `python -I`).

    g0state --log <G0 chain log> --step 5000      -> one line `<state>:<marker name or ->:<shell pid or ->`
                                                     state = ended (name = the terminal marker) | running (name = the newest
                                                     marker seen) | nostart
    g0gate  --g0 <g0.json> --ckpt <ckpt> --out <gate.json> [--require-pass]
                                                  -> `ok:<G0 verdict>`  or  `skip:<reason>`   (also writes <gate.json>)

The G0 chain's marker names are assembled from parts below, so this file's own text never contains a literal marker that a
grep over a log could match (CLAUDE.md traps: a polling filter must not contain the pattern it searches for).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

#: the G0 chain (`chain_g0_r8.sh`) ends in exactly one of these; it appends `ZZR8<name>ZZ` to its log
TERMINAL = ("CHAIN" + "END", "PRECOND" + "FAIL", "NO" + "COMMIT", "NO" + "LOCK", "PROBE" + "FAIL")
START = "CHAIN" + "START"


def g0state(log_path: str, step: int) -> str:
    if not os.path.exists(log_path):
        return "nostart:-:-"
    try:
        lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return "nostart:-:-"
    head = f"ZZR8{START}{step}ZZ"
    idx = [i for i, ln in enumerate(lines) if ln.startswith(head)]
    if not idx:
        return "nostart:-:-"
    after = lines[idx[-1]:]
    m = re.search(r"\bpid (\d+)", after[0])
    pid = m.group(1) if m else "-"
    for name in TERMINAL:
        tok = f"ZZR8{name}ZZ"
        if any(ln.startswith(tok) for ln in after):
            return f"ended:{name}:{pid}"
    seen = [m.group(1) for ln in after for m in [re.match(r"ZZR8([A-Z0-9]+)ZZ", ln)] if m]
    last = seen[-1] if seen else "-"                            # the newest marker NAME: lets the waiter poll fast near the end
    return f"running:{last}:{pid}"


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def g0gate(g0: str, ckpt: str, out: str, require_pass: bool = False) -> str:
    rec: dict = {"g0_json": g0, "require_pass": bool(require_pass)}

    def done(res: str) -> str:
        rec["result"] = res
        try:
            json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
        except OSError:
            pass
        return res

    if not os.path.exists(g0) or os.path.getsize(g0) == 0:
        return done("skip:g0.json is missing or empty (G0 produced no artifact)")
    try:
        g = json.load(open(g0, encoding="utf-8"))
    except Exception as exc:                                   # noqa: BLE001 -- recorded
        return done(f"skip:g0.json unreadable ({type(exc).__name__})")
    v = g.get("verdict") or {}
    rec["G0"] = v.get("G0")
    rec["amendment"] = v.get("amendment")
    m1 = ((v.get("mutation_detection") or {}).get("m1") or {}).get("detected")
    rec["M1_detected"] = m1
    rec["ckpt_md5_in_g0"] = g.get("ckpt_md5")
    if g.get("ckpt_md5") != md5_file(ckpt):
        return done("skip:the g0.json is for another checkpoint (md5 differs)")
    if m1 is not True:
        return done(f"skip:M1_detected is {m1!r}: the deliberate-regression arm was not detected, G0 is VOID whatever else reads")
    if require_pass and v.get("G0") != "PASS":
        return done(f"skip:G0 is {v.get('G0')} and R8_REQUIRE_G0_PASS is set")
    return done(f"ok:{v.get('G0')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("g0state")
    a.add_argument("--log", required=True)
    a.add_argument("--step", type=int, required=True)
    b = sub.add_parser("g0gate")
    b.add_argument("--g0", required=True)
    b.add_argument("--ckpt", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--require-pass", action="store_true")
    ns = ap.parse_args()
    if ns.cmd == "g0state":
        print(g0state(ns.log, ns.step))
    else:
        print(g0gate(ns.g0, ns.ckpt, ns.out, ns.require_pass))


if __name__ == "__main__":
    main()
