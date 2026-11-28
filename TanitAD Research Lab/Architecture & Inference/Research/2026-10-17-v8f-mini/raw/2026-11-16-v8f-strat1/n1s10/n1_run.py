"""n1_run.py -- runs the frozen STRAT1 readers on the two N1 arms of the addendum PREREG_ADDENDUM_N1S10.md (N1_s10 init M2b_s0, N1_s11 init M2b_s1).
Study-local wrapper over ../seeds/seeds_run.py (frozen, unedited).  It
  * appends N1_s10 / N1_s11 to strat1_lib's ARMS and TEXT_ARMS (N1 is text-nav with NO ctx heads, exactly as N1_s0 / N1_s1: they sit in TEXT_ARMS, not in CTX_ARMS), then lets seeds_run append its four,
  * lets the two N1 arms through the bank-manifest commit check at commit a4357e79 (the same widening, for these two tags only; the seven first arms still need e81bde01).
a4357e79 is e81bde01 plus stack/scripts/strat1_check.py and stack/tests/test_v8f_strat1.py only (seeds_run.py docstring, MEASURED 2026-11-16), and N1 was trained by the same chain code as the four seeds arms.
Usage:  python n1_run.py dumps TAG [TAG ...] | t0 [run_t0 args] | strategic [args]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import strat1_lib as L  # noqa: E402
import seeds_run as SRUN  # noqa: E402

N1_NEW = ("N1_s10", "N1_s11")


def patch_lib():
    if "N1_s10" not in L.ARMS:
        L.ARMS = tuple(L.ARMS) + N1_NEW
        L.TEXT_ARMS = tuple(L.TEXT_ARMS) + N1_NEW
    SRUN.patch_lib()
    return L


def patch_rds():
    patch_lib()
    RDS = SRUN.patch_rds()
    if not getattr(RDS, "_n1_patched", False):
        inner = RDS.refusal

        def refusal(tag):
            if tag[7:] in N1_NEW:
                saved = L.COMMIT
                L.COMMIT = SRUN.SEEDS_COMMIT
                try:
                    return inner(tag)
                finally:
                    L.COMMIT = saved
            return inner(tag)
        RDS.refusal = refusal
        RDS._n1_patched = True
    return RDS


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dumps", "t0", "strategic"):
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
    else:
        import strat1_strategic as S
        sys.argv = ["strat1_strategic.py"] + rest
        S.main()


if __name__ == "__main__":
    main()
