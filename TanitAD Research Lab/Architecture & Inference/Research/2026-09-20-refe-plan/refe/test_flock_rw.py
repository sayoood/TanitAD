#!/usr/bin/env python3
"""No chain script may probe a lock with an EXCLUSIVE flock on a READ-ONLY fd.

Why (MEASURED 2026-10-13 on the A40 pod, /workspace = nfs4): `( flock -n 9 ) 9< lockfile` fails with "flock: 9: Bad file descriptor" (rc 65) on NFSv4,
because the kernel emulates flock with POSIX locks there and an exclusive lock needs a writable fd. rls1_pod.sh's p2_over() used that form, so it read
"the P2 chain still holds chain.lock" FOREVER: RL-S1 refused at 13:22:54Z with P2 already gone. `9<>` (read-write), `9>` and `flock -s` all return 0 there.
A local filesystem accepts the read-only form, so no runtime test on the dev box can see this: the check is static, over every shipped .sh.

Prints ZZTEST_PASS <n>.
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
# `( flock [-n] [-x] N ) N< file` or `flock [-n] N ... N< file` with an exclusive (default) lock; `-s` (shared) is allowed on a read-only fd
BAD = re.compile(r"flock\s+(?:-[nwxE]+\s+|-w\s+\S+\s+)*(\d+)\s*\)?\s*(?:;[^\n]*)?\s*\1<(?![>&])")


def offenders(text: str) -> list:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = BAD.search(line)
        if m and "-s" not in line[line.find("flock"):m.end()]:
            out.append((i, line.strip()[:120]))
    return out


def main() -> int:
    checks = []

    def t(name, ok):
        checks.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    # the detector itself: it must flag the defect and pass the fixes (a guard that cannot fail proves nothing)
    t("detector/flags_readonly_exclusive", offenders('if ! ( flock -n 9 ) 9< "$P2/chain.lock"; then') != [])
    t("detector/passes_readwrite", offenders('if ! ( flock -n 9 ) 9<> "$P2/chain.lock"; then') == [])
    t("detector/passes_write", offenders('if ! ( flock -n 9 ) 9> "$P2/chain.lock"; then') == [])
    t("detector/passes_shared_readonly", offenders('( flock -s -n 9 ) 9< "$F"') == [])
    t("detector/passes_path_form", offenders('flock -n "$F" true') == [])
    t("detector/ignores_comments", offenders('# ( flock -n 9 ) 9< "$F" was the defect') == [])

    files = sorted(glob.glob(os.path.join(PKG, "refe", "*.sh")) + glob.glob(os.path.join(PKG, "code", "*.sh")))
    t("scan/found_the_chain_scripts", any(f.endswith("rls1_pod.sh") for f in files) and any(f.endswith("s1aux_pod.sh") for f in files))
    for f in files:
        bad = offenders(open(f, encoding="utf-8", errors="replace").read())
        t(f"scan/{os.path.relpath(f, PKG)}", not bad)
        for i, ln in bad:
            print(f"      {os.path.basename(f)}:{i}: {ln}")

    failed = [n for n, ok in checks if not ok]
    if failed:
        print("ZZTEST_FAIL", len(failed), failed[:5])
        return 1
    print(f"ZZTEST_PASS {len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
