"""P7-WIRE-INTEGRATE-2 / Gap 1 probe: is mutant M2 EQUIVALENT?

M2 replaces the early return of gate_checks.check_reserve (`if missing: return verdict(...)`) with `if False:`.  The question is whether the program's OBSERVABLE behaviour differs.
For every argv that omits one of the five audited inputs, this prints, for the tree under test:
    status | number of reasons | how many times the audit RAN | whether the evidence carries an `audit` block | the reasons' first words
usage: python probe_m2.py <refe dir>        (run it once on the unmutated tree and once on the M2-mutated copy; diff the two outputs)
"""
import sys
import tempfile
from pathlib import Path

REFE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REFE))
import test_p7_integration as T  # noqa: E402  (its fixtures: mk_tree, gate_argv, reserve_file)
import gate_checks as C  # noqa: E402
import session_reserve as SR  # noqa: E402

FLAGS = ("--targets", "--metric-cache", "--aux-drv-labels", "--images", "--calib")
calls = []
real = SR.audit_inputs


def counting(reserve, **kw):
    calls.append(sorted(k for k, v in kw.items() if v))
    return real(reserve, **kw)


SR.audit_inputs = counting
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rj = T.reserve_file(td)
    root = T.mk_tree(td / "t")
    base = T.gate_argv(root)
    for flag in (None,) + FLAGS:
        argv = base if flag is None else T.HW.without(base, flag)
        calls.clear()
        e = C.check_reserve(argv, T.P7, rj)
        print(f"{'(none omitted)' if flag is None else flag:18s} | {e['status']:4s} | reasons={len(e['reasons'])} | audit_ran={len(calls)}x | audit_block={'audit' in e['details']} | {'; '.join(r[:46] for r in e['reasons'])[:150]}")
