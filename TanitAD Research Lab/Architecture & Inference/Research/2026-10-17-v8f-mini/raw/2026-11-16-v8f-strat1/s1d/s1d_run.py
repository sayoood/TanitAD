"""s1d_run.py -- runs the frozen STRAT1 dump and T0 passes on the four S1d arms (S1 with --ctx-detach on; seeds 0 / 1 / 10 / 11).  Study-local wrapper over ../seeds/seeds_run.py and ../n1s10/n1_run.py
(both unedited here).  It
  * appends S1d_s0 / S1d_s1 / S1d_s10 / S1d_s11 to strat1_lib ARMS, TEXT_ARMS and CTX_ARMS (S1d is text-nav WITH the ctx heads, as S1: the heads still train, only their gradient into the adapter is cut),
  * lets the four S1d arms through the bank-manifest commit check at commit 4dedac45 (these tags only; N1 / seeds arms keep their own wideners; the seven first arms still need e81bde01).
4dedac45 differs from e81bde01 in stack/scripts/strat1_check.py, stack/tests/test_strat1_s1d.py and stack/tests/test_v8f_strat1.py only, and from a4357e79 in the first two (MEASURED 2026-11-17,
git diff --name-only in the repo mirror).  The trainer, the models and the evaluators are untouched; the lever is the trainer's existing --ctx-detach flag, set by the checker's argv.
Usage:  python s1d_run.py dumps TAG [TAG ...] | t0 [run_t0 args]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (os.path.join(R, "n1s10"), os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import n1_run as NRUN  # noqa: E402
import strat1_lib as L  # noqa: E402

S1D = ("S1d_s0", "S1d_s1", "S1d_s10", "S1d_s11")
S1D_COMMIT = "4dedac45dc8d4f6922711ed9af63e0ab54e435be"
S1D_DIFF_VS_RECORD = ("stack/scripts/strat1_check.py", "stack/tests/test_strat1_s1d.py", "stack/tests/test_v8f_strat1.py")


def patch_lib():
    NRUN.patch_lib()
    if "S1d_s0" not in L.ARMS:
        L.ARMS = tuple(L.ARMS) + S1D
        L.TEXT_ARMS = tuple(L.TEXT_ARMS) + S1D
        L.CTX_ARMS = tuple(L.CTX_ARMS) + S1D
    return L


def patch_rds():
    patch_lib()
    RDS = NRUN.patch_rds()
    if not getattr(RDS, "_s1d_patched", False):
        inner = RDS.refusal

        def refusal(tag):
            if tag[7:] in S1D:
                saved = L.COMMIT
                L.COMMIT = S1D_COMMIT
                try:
                    return inner(tag)
                finally:
                    L.COMMIT = saved
            return inner(tag)
        RDS.refusal = refusal
        RDS._s1d_patched = True
    return RDS


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dumps", "t0"):
        raise SystemExit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    RDS = patch_rds()
    if cmd == "dumps":
        sys.argv = ["run_dumps_strat1.py"] + rest
        RDS.main()
    else:
        import run_t0_strat1 as T
        sys.argv = ["run_t0_strat1.py"] + rest
        T.main()


if __name__ == "__main__":
    main()
