"""H19-STAMP-1 SOURCE MUTATION PROOF -- reintroduce the defect, prove the
instrument REFUSES. A test that can only pass measures nothing.

Runs against the disposable MIRROR (C:/Users/Admin/tanitad-wt), restoring the
pristine file from the repo after every mutation. ASCII-only output.

M0  pristine                      -> the suite must PASS   (the control)
M1  guard removed (man5 always)   -> must FAIL / raise     (guard is load-bearing)
M2  stamp hardcoded to "applied"  -> must FAIL             (the exact defect the
                                                            brief warns about:
                                                            a stamp that reads
                                                            identically on both)
M3  runtime flag hardcoded True   -> must FAIL             (the flag is read, not
                                                            decorative)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
MIRROR = r"C:\Users\Admin\tanitad-wt"
REL = os.path.join("stack", "tanitad", "refs", "refc_v3.py")
SRC = os.path.join(REPO, REL)
DST = os.path.join(MIRROR, REL)
TEST = os.path.join("stack", "tests", "test_h19_prior_stamp.py")

ENV = dict(os.environ)
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["PYTHONPATH"] = MIRROR + r"\stack;" + MIRROR + r"\taniteval"
ENV["OMP_NUM_THREADS"] = "6"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"

GUARD = ('            man5 = (tac.derive_man5_logprobs(lat, lon)\n'
         '                    if self.tac_vocab_version == "kin3" else None)')
GUARD_OFF = '            man5 = tac.derive_man5_logprobs(lat, lon)'
FLAG = '            cache["h19_tactical_feed"] = man5 is not None'
FLAG_OFF = '            cache["h19_tactical_feed"] = True'
STAMP = '''    if vv == "kin3":
        return {"h19_prior": "applied",'''
STAMP_OFF = '''    if True:
        return {"h19_prior": "applied",'''


def restore():
    shutil.copyfile(SRC, DST)


def read():
    with open(DST, encoding="utf-8") as fh:
        return fh.read()


def write(txt):
    with open(DST, "w", encoding="utf-8", newline="") as fh:
        fh.write(txt)


def run():
    p = subprocess.run([PY, "-m", "pytest", TEST, "-q", "--no-header",
                        "-x" if False else "--tb=no"],
                       cwd=MIRROR, env=ENV, capture_output=True, text=True)
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()][-3:]
    return p.returncode, " | ".join(tail)


def mutate(name, old, new, must):
    restore()
    txt = read()
    n = txt.count(old)
    if n != 1:
        print("  %-34s SKIPPED: anchor found %d times (expected 1)" % (name, n))
        return False
    write(txt.replace(old, new))
    rc, tail = run()
    verdict = "PASS" if rc == 0 else "FAIL"
    ok = (verdict == must)
    print("  %-34s -> suite %s  [want %s]  %s" % (name, verdict, must,
                                                  "OK" if ok else "*** WRONG ***"))
    print("       %s" % tail[:180])
    restore()
    return ok


def main():
    print("=" * 74)
    print("H19-STAMP-1 MUTATION PROOF (source-level, mirror, zero GPU)")
    print("=" * 74)
    restore()
    rc, tail = run()
    ok0 = rc == 0
    print("  %-34s -> suite %s  [want PASS]  %s"
          % ("M0 pristine (CONTROL)", "PASS" if ok0 else "FAIL",
             "OK" if ok0 else "*** WRONG ***"))
    print("       %s" % tail[:180])

    results = [ok0]
    results.append(mutate("M1 guard removed (man5 always)", GUARD, GUARD_OFF,
                          "FAIL"))
    results.append(mutate("M2 stamp hardcoded 'applied'", STAMP, STAMP_OFF,
                          "FAIL"))
    results.append(mutate("M3 runtime flag hardcoded True", FLAG, FLAG_OFF,
                          "FAIL"))
    restore()
    print()
    print("=" * 74)
    print("MUTATION PROOF: %d/%d as expected -- %s"
          % (sum(results), len(results),
             "the instrument's failure branches are REACHABLE"
             if all(results) else "*** AN INSTRUMENT THAT CANNOT FAIL ***"))
    print("=" * 74)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
