"""read_rulee1r10k.py -- card PREREG_V8F_PROOF_ADD145 (RULEE1R10K): the read. read_rulefc10k.py (card ADD144) used as a module with the arms, pair, bar,
controls and names of this card; its engines (the landed headway, family and route-goal readers through the card ADD139 reader) are unchanged.
Never run on data by the tool builder. Tier: T1-kin/hb0.2; never driving performance.
Pair RE1 = H_RQ50KE1 - H_RQ50K (bar: spread of H_RQ50K). Controls: r4 runs, r4 partner (sha list of the box-2 H_RQ50K folders), later looks (q = 1..9 in every new run,
q = 5 only in the partner), the landed readers' controls. Name: RULEE1R10K-U-LOWER / -U-HIGHER / -U-NOT-SEPARATED from the pooled interval of U.
Usage: python read_rulee1r10k.py --pull PULL --read-dir D --out OUT.json --partners-sha SHA.txt [--skip-engines]   (--r1 is not used by this card)"""
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import read_rulefc10k as RD  # noqa: E402

RD.CARD, RD.PRE, RD.SPREAD_ARM = "PREREG_V8F_PROOF_ADD145", "RULEE1R10K", "H_RQ50K"
RD.NEW_ARMS, RD.FC_ARMS = ("H_RQ50KE1",), ()
RD.ALL_ARMS = RD.NEW_ARMS + (RD.SPREAD_ARM,)
RD.PAIRS = {"RE1": ("H_RQ50KE1", RD.SPREAD_ARM)}
RD.CONSEQUENCE = ("U LOWER with no family WORSE and C not HIGHER makes every-step the rule of the candidate's reader form (specification section 1, the rule line, via the other "
                  "session); U LOWER with a family WORSE or C HIGHER goes to the PI with its cost; otherwise the reader form keeps rule-every 5 and the reader-form forecast is the next arm.")
RD.DEFAULT_QUEUE = os.path.join(HERE, "box2_rulee1r_queue.py")


def c_later(a, seeds):
    per, ok = {}, True
    for m in seeds:
        for arm, want in (("H_RQ50KE1", list(range(1, 10))), ("H_RQ50K", [5])):
            key = RD.key_of(arm, m)
            p = os.path.join(a.pull, key, "ticks", "rule_log_mid_%s.npz" % key)
            got = None
            if os.path.isfile(p):
                with np.load(p, allow_pickle=True) as z:
                    got = sorted(set(int(x) for x in z["q"]))
            good = got == want
            per[key] = {"q_values": got, "want": want, "PASS": good}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def c_r_none(path):
    return {"PASS": True, "note": "card ADD145 has no r1 run (wrapper rzgfh unchanged, an existing flag)"}


RD.c_r1 = c_r_none
_src = inspect.getsource(RD.evaluate)
_old = 'C = {"r1": c_r1(a.r1), "r4_runs": c_runs(a, seeds, queue), "r4_partner": c_partner(a, seeds)}'
assert _src.count(_old) == 1
_src = _src.replace(_old, 'C = {"r4_runs": c_runs(a, seeds, queue), "r4_partner": c_partner(a, seeds), "later_looks": c_later(a, seeds)}')
RD.c_later = c_later
exec(compile(_src, RD.__file__, "exec"), RD.__dict__)

_jsrc = inspect.getsource(RD.judge)
_jold = '''        if u == "LOWER" and not worse:'''
assert _jsrc.count(_jold) == 1
_jsrc = _jsrc.replace(_jold, '''        c_hi = RD_u_card(b["headway"]["C"]) == "HIGHER"
        if u == "LOWER" and (worse or c_hi):
            cons = "to the PI with its cost (families WORSE: %s; C higher: %s)" % (", ".join(worse) or "none", c_hi)
        elif u == "LOWER" and not worse:''')
_jsrc = _jsrc.replace('''            cons = "the rule of the candidate's label form, to be carried to the reader form (a successor card)"''',
                      '''            cons = "every-step is the rule of the candidate's reader form (specification section 1, via the other session)"''')
_jsrc = _jsrc.replace('''            cons = "no consequence committed by the card for U %s" % u''', '''            cons = "the reader form keeps rule-every 5; the reader-form forecast is the next arm (U %s)" % u''')
RD.RD_u_card = RD.u_card
exec(compile(_jsrc, RD.__file__, "exec"), RD.__dict__)

if __name__ == "__main__":
    sys.argv += [] if "--r1" in sys.argv else ["--r1", "unused"]
    sys.exit(RD.main())
