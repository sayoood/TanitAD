"""seeds_run.py -- runs the frozen STRAT1 readers on the four SEEDS-DISAGREE arms (C0 and S1 at seeds 10 and 11; frozen card PREREG_V8F_STRAT1 section 6).
The readers are used UNEDITED (their sha256 are recorded in the seeds result).  This module only
  * appends the four arms to strat1_lib's arm tuples (C0_s10, S1_s10, C0_s11, S1_s11; S1 arms are text-nav and carry the ctx heads, as S1_s0 / S1_s1), and
  * lets those four arms through the bank-manifest commit check at commit a4357e79.
Why the commit check is widened, and only for these four: a4357e79 is e81bde01 plus two files, stack/scripts/strat1_check.py (the SEEDS table) and
stack/tests/test_v8f_strat1.py (MEASURED 2026-11-16: git diff --name-only e81bde01 a4357e79 lists exactly those two).  The trainer's model, the evaluators and the
readers import neither, so the four arms trained the e81bde01 code of record.  The seven first arms still need e81bde01; any other commit is refused.
Usage:  python seeds_run.py dumps TAG [TAG ...] | t0 [run_t0 args] | strategic [args] | ctx [args]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
if R not in sys.path:
    sys.path.insert(0, R)
import strat1_lib as L  # noqa: E402

NEW = ("C0_s10", "S1_s10", "C0_s11", "S1_s11")
NEW_TEXT = ("S1_s10", "S1_s11")
SEEDS_COMMIT = "a4357e7986f3aa30dab2ed83fde2351fda8d5662"
SEEDS_DIFF = ("stack/scripts/strat1_check.py", "stack/tests/test_v8f_strat1.py")


def patch_lib():
    """extend the arm tuples once (idempotent); must run before run_dumps_strat1 is imported (it builds its tag table at import)"""
    if "C0_s10" not in L.ARMS:
        L.ARMS = tuple(L.ARMS) + NEW
        L.TEXT_ARMS = tuple(L.TEXT_ARMS) + NEW_TEXT
        L.CTX_ARMS = tuple(L.CTX_ARMS) + NEW_TEXT
    return L


def patch_rds():
    """-> run_dumps_strat1 with its refusal() accepting SEEDS_COMMIT for the four new arms only"""
    patch_lib()
    import run_dumps_strat1 as RDS
    if not getattr(RDS, "_seeds_patched", False):
        orig = RDS.refusal

        def refusal(tag):
            if tag[7:] in NEW:
                saved = L.COMMIT
                L.COMMIT = SEEDS_COMMIT
                try:
                    return orig(tag)
                finally:
                    L.COMMIT = saved
            return orig(tag)
        RDS.refusal = refusal
        RDS._seeds_patched = True
    return RDS


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dumps", "t0", "strategic", "ctx"):
        raise SystemExit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    RDS = patch_rds()
    if cmd == "dumps":
        sys.argv = ["run_dumps_strat1.py"] + rest
        RDS.main()
    elif cmd == "t0":
        import run_t0_strat1 as T
        sys.argv = ["run_t0_strat1.py"] + rest
        T.main()
    elif cmd == "strategic":
        import strat1_strategic as S
        sys.argv = ["strat1_strategic.py"] + rest
        S.main()
    else:
        import strat1_ctx_read as C
        sys.argv = ["strat1_ctx_read.py"] + rest
        C.main()


if __name__ == "__main__":
    main()
