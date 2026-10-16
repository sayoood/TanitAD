#!/usr/bin/env python3
"""The SH-NO next lever (PREREG_SAFEHEAD.md s.8 (a)): a TTC-penalised TRAINING reward  R' = R - lam_ttc * (1 - ttc)  in the EPO term only.
J, f and every proxy are still MEASURED with the original R (phead_common.Evaluator builds R from SF.epo_reward, which this script never touches: the closures run in a COPY of the
module namespace, adam_common.extract_closures).  Same CLI as sh_closed.py plus SH_TTC_PEN=<lam_ttc> in the environment; the output name gets the tag _ttcpen<lam>.
Kept in its own file so the running queue's harness (sh_common.py / sh_closed.py) is never edited mid-run.
    SH_TTC_PEN=0.25 python sh_ttcpen.py --design FULL --lam 1.0 --regime T --K 250 --snaps 50 100 250 --out-dir out --tag _ttcpen0.25"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
import sh_closed

LAM_TTC = float(os.environ["SH_TTC_PEN"])
_orig = SH.run_closed


def run_closed_pen(SF, a, tr, ev, *args, **kw):
    g = tr.fn["epo_term"].__globals__
    base_reward = g["epo_reward"]
    assert base_reward is SF.epo_reward

    def epo_reward_pen(tg6, recs, rcfg):
        R = base_reward(tg6, recs, rcfg)
        return R - LAM_TTC * (1.0 - np.asarray(tg6, np.float64)[..., 3])      # tg6 column 3 = TTC (rl_reward.COMPONENTS)

    g["epo_reward"] = epo_reward_pen
    try:
        return _orig(SF, a, tr, ev, *args, **kw)
    finally:
        g["epo_reward"] = base_reward


SH.run_closed = run_closed_pen
if __name__ == "__main__":
    sh_closed.main()
