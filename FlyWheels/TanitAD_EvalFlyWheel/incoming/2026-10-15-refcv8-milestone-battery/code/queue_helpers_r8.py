"""queue_helpers_r8.py -- small, testable pieces for `queue_after_5000.sh` (no GPU, no torch; run with `python -I`).

    battstate  --log <battery chain log> [--step 5000]     one line `<state>:<marker name or ->:<shell pid or ->`
                                                           state = ended | running | nostart ; looks ONLY at the NEWEST segment of
                                                           the log (a segment starts at a `ZZR8BATTSTART..` line; relaunches append)
    metrics_clean --src <copy of Thor metrics.jsonl> --step N --out <dest>
                                                           drop ONE torn trailing line (the file is live), assert every other line
                                                           parses and EXACTLY ONE eval row sits at step N, write <dest> atomically;
                                                           prints `ok:<n_lines>` or `fail:<reason>`
    freegb     --path E:/                                  free GB (floor, 2 dp) on the volume
    md5        <file>                                      md5 hexdigest

Marker names are assembled from parts so this file's own text never contains a literal marker a log grep could match (CLAUDE.md
trap: a polling filter must not contain the pattern it searches for).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

START = "BATT" + "START"
#: a battery-chain segment (`chain_battery_r8.sh`) ends in exactly one of these (each is followed by `exit` or the last line).
#: `FAIL` = pass 1 incomplete after its retry (exit 4); SKIP = the G0 gate said no; NOCOMMIT / NOLOCK = it never got the card.
TERMINAL = ("BATT" + "CHAINEND", "BATT" + "SKIP", "BATT" + "NOCOMMIT", "BATT" + "NOLOCK", "BATT" + "FAIL")


def battstate(log_path: str, step: int | None = None) -> str:
    """State of the NEWEST segment of a battery-chain log. The shell pid is read from the segment's first line (`... pid N step S`).
    `step`, when given, is checked against the segment's own `step S` so another milestone's log never answers for this one."""
    if not os.path.exists(log_path):
        return "nostart:-:-"
    try:
        lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return "nostart:-:-"
    idx = [i for i, ln in enumerate(lines) if ln.startswith(f"ZZR8{START}ZZ")]
    if not idx:
        return "nostart:-:-"
    after = lines[idx[-1]:]
    m = re.search(r"\bpid (\d+)", after[0])
    pid = m.group(1) if m else "-"
    ms = re.search(r"\bstep (\d+)", after[0])
    if step is not None and ms and int(ms.group(1)) != int(step):
        return "nostart:-:-"
    for name in TERMINAL:
        tok = f"ZZR8{name}ZZ"
        if any(ln.startswith(tok) for ln in after):
            return f"ended:{name}:{pid}"
    seen = [mm.group(1) for ln in after for mm in [re.match(r"ZZR8([A-Z0-9]+)ZZ", ln)] if mm]
    return f"running:{seen[-1] if seen else '-'}:{pid}"


def metrics_clean(src: str, step: int, out: str) -> str:
    try:
        raw = open(src, "rb").read()
    except OSError as exc:
        return f"fail:cannot read {src} ({type(exc).__name__})"
    txt = raw.decode("utf-8", errors="replace")
    lines = [ln for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return "fail:empty metrics copy"
    rows, dropped = [], 0
    for i, ln in enumerate(lines):
        try:
            rows.append((ln, json.loads(ln)))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                dropped += 1                    # the live file's torn tail: the ONLY tolerated defect
            else:
                return f"fail:line {i + 1} of {len(lines)} is not JSON (a torn line that is not the last one)"
    n = sum(1 for _, r in rows if isinstance(r, dict) and r.get("step") == step and "eval_loss" in r)
    if n != 1:
        return f"fail:{n} eval rows at step {step} (need exactly 1)"
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".metrics_", suffix=".part", dir=d)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        for ln, _ in rows:
            f.write(ln.rstrip("\r") + "\n")
    os.replace(tmp, out)
    return f"ok:{len(rows)}:dropped_torn={dropped}"


def free_gb(path: str) -> float:
    return round(shutil.disk_usage(path).free / 1e9, 2)


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("battstate")
    a.add_argument("--log", required=True)
    a.add_argument("--step", type=int, default=None)
    b = sub.add_parser("metrics_clean")
    b.add_argument("--src", required=True)
    b.add_argument("--step", type=int, required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("freegb")
    c.add_argument("--path", required=True)
    d = sub.add_parser("md5")
    d.add_argument("file")
    ns = ap.parse_args()
    if ns.cmd == "battstate":
        print(battstate(ns.log, ns.step))
    elif ns.cmd == "metrics_clean":
        print(metrics_clean(ns.src, ns.step, ns.out))
    elif ns.cmd == "freegb":
        print(free_gb(ns.path))
    else:
        print(md5_file(ns.file))


if __name__ == "__main__":
    main()
