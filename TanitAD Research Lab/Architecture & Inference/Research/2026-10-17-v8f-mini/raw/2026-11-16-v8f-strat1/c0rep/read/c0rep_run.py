"""c0rep_run.py -- runs the frozen STRAT1 dump and T0 passes on the two C0 same-seed replicate arms of PREREG_ADDENDUM_C0REP.md (C0_s10r, C0_s11r).
Study-local wrapper over ../s1d/s1d_run.py (itself over ../n1s10/n1_run.py and ../seeds/seeds_run.py; all unedited here).  It
  * appends C0_s10r / C0_s11r to strat1_lib ARMS (C0 is v9 nav, no ctx heads: not in TEXT_ARMS, not in CTX_ARMS, exactly as C0_s10 / C0_s11),
  * lets those two tags through the bank-manifest commit check at commit a4357e79 (these tags only; the other widenings keep their own tags; the seven first arms still need e81bde01).
The replicates trained at the commit of the banked C0_s10 / C0_s11 (a4357e79, MEASURED: gate_token.json of both runs, pod chain strat1_c0rep.sh COMMIT=a4357e79...), by the same checker with only --out differing (K-ARGV).
Usage:  python c0rep_run.py dumps TAG [TAG ...] | t0 [run_t0 args]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
for p in (os.path.join(R, "s1d"), os.path.join(R, "n1s10"), os.path.join(R, "seeds"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
import s1d_run as SD  # noqa: E402
import strat1_lib as L  # noqa: E402

REP = ("C0_s10r", "C0_s11r")
REP_COMMIT = "a4357e7986f3aa30dab2ed83fde2351fda8d5662"


def patch_lib():
    SD.patch_lib()
    if "C0_s10r" not in L.ARMS:
        L.ARMS = tuple(L.ARMS) + REP
    return L


def patch_rds():
    patch_lib()
    RDS = SD.patch_rds()
    if not getattr(RDS, "_c0rep_patched", False):
        inner = RDS.refusal

        def refusal(tag):
            if tag[7:] in REP:
                saved = L.COMMIT
                L.COMMIT = REP_COMMIT
                try:
                    return inner(tag)
                finally:
                    L.COMMIT = saved
            return inner(tag)
        RDS.refusal = refusal
        RDS._c0rep_patched = True
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
