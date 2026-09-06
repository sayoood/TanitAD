#!/usr/bin/env python3
"""DEMONSTRATED RED: the bit-identity assertions are not vacuous.

⛔ WHY THIS FILE EXISTS. A passing "OFF is bit-identical" test is worth nothing
unless the same assertion can be made to FAIL. This script REINTRODUCES THE
DEFECT -- it removes the zero-init from ``MaxSpeedConditioner`` so the edge is
LIVE at step 0 -- reruns the suite, and shows the parity tests going red. Then it
restores the module and shows them green again.

That is the pattern this programme uses because an AST census once read 0
suspects on BOTH the fixed and the broken trainer: a guard needs MUTATION, not
inspection.

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

STACK = Path("C:/Users/Admin/tanitad-wt/stack")
MOD = STACK / "tanitad" / "refs" / "max_speed_input.py"
PY = sys.executable
PARITY = ("test_OFF_is_BIT_IDENTICAL_on_a_fixed_seed",
          "test_ON_at_INIT_is_also_BIT_IDENTICAL",
          "test_the_edge_is_BIT_INERT_at_init")


def _run(tag):
    r = subprocess.run([PY, "-m", "pytest", "tests/test_max_speed_input.py",
                        "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=STACK, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [l for l in r.stdout.splitlines() if l.strip()][-1:]
    fails = sorted(set(re.findall(r"(test_[A-Za-z0-9_]+)", r.stdout))
                   & set(PARITY)) if r.returncode else []
    print("  %-22s exit=%d   %s" % (tag, r.returncode, tail[0] if tail else ""))
    return r.returncode, r.stdout


def main() -> int:
    src = MOD.read_text(encoding="utf-8")
    zero_init = """        for lin in (self.to_tac, self.to_str):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)"""
    assert zero_init in src, ("the zero-init block moved -- this demo would "
                              "have silently mutated nothing, which is exactly "
                              "the vacuous-control failure it exists to rule out")
    mutated = """        for lin in (self.to_tac, self.to_str):
            nn.init.normal_(lin.weight, std=0.5)   # MUTATION: NOT zero-init
            nn.init.normal_(lin.bias, std=0.5)"""

    print("=" * 74)
    print("STEP 1  BASELINE (zero-init intact) -- the parity assertions must PASS")
    rc0, _ = _run("baseline")

    print("STEP 2  MUTATION (zero-init removed) -- they must now FAIL")
    MOD.write_text(src.replace(zero_init, mutated), encoding="utf-8")
    try:
        rc1, out1 = _run("mutated")
    finally:
        MOD.write_text(src, encoding="utf-8")

    print("STEP 3  RESTORED -- green again, so the mutation was the only change")
    rc2, _ = _run("restored")

    print()
    print("=" * 74)
    hit = [t for t in PARITY if re.search(r"%s\b" % t, out1)]
    ok = (rc0 == 0 and rc1 != 0 and rc2 == 0 and hit)
    print("baseline PASS   : %s" % (rc0 == 0))
    print("mutated  FAIL   : %s" % (rc1 != 0))
    print("restored PASS   : %s" % (rc2 == 0))
    print("parity tests that went RED under the mutation: %s" % (hit or "NONE"))
    print()
    print("VERDICT: %s" % ("the bit-identity assertions HAVE TEETH -- an "
                           "equality that can fail" if ok else
                           "INCONCLUSIVE -- the mutation did not turn the "
                           "parity assertions red; do NOT quote the OFF proof"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
