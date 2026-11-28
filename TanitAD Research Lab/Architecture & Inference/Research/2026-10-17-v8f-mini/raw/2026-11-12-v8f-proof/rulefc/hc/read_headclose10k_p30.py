"""read_headclose10k_p30.py -- card PREREG_V8F_PROOF_ADD152 (HEADCLOSE10K): the age read. The read of card ADD147 (rulefc/e1a/read_rulee1a10k.py, itself the age read of
card ADD140 with amendment 2) imported as a module and NOT edited, with the arms of this card: V1P30 = H_RQ50KV1_P30 - H_RQ50KE1_P30 (pre-roll 30 steps, 1,271-window grid;
bar: spread of H_RQ50KE1_P30). Its run checks and engines are used unchanged, except: the wrapper check of q2_argv admits rzgfhqet for the new arm (the card's wrapper);
later looks q = 1..9 in both arms (both act every step); the card's controls t1 to t5 are added. Tier T1-kin/hb0.2; never driving performance.
Usage: python read_headclose10k_p30.py --pull PULL --read-dir D --out OUT.json --partners-sha SHA.txt [--skip-engines]
PULL holds H_RQ50KV1_P30_s*, H_RQ50KE1_P30_s* and H_RLcA_P0_s* (amendment 2's n1 control of card ADD140)."""
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "e1a")))
import hc_common as HC  # noqa: E402
import read_rulee1a10k as E  # noqa: E402

M = E.M
NEW, PART = "H_RQ50KV1_P30", "H_RQ50KE1_P30"
E.CARD, E.PRE, E.NEW, E.PART = "PREREG_V8F_PROOF_ADD152", "HEADCLOSE10K-P30", NEW, PART
E.SPEC = {NEW: ("RRe", "K", 30, "deadreckon", "30", "plan", 1, 0.2), PART: ("RR", "K", 30, "deadreckon", "30", "plan", 1, 0.2)}
M.NEW_ARMS, M.PARTNER_ARMS, M.FRESH_ARMS = (NEW,), (PART,), ()
M.P_ARMS = (NEW, PART)
M.ALL_ARMS = M.P_ARMS
M.R3_PAIR, M.SPREAD_ARM_R3 = (NEW, PART), PART
M.DEFAULT_QUEUE = os.path.join(HERE, "box2_headclose_queue.py")
M.DEFAULT_PARTNER_QUEUE = os.path.normpath(os.path.join(HERE, "..", "e1a", "box2_rulee1a_queue.py"))
M.WRAPPER_OF = {NEW: "v8f_t1_eval_reader_rule_rzgfhqet.py"}

_q2 = inspect.getsource(M.q2_argv)
_old = 'base == ("v8f_t1_eval_reader_rule_rzgfhq.py" if arm in P_ARMS else "v8f_t1_eval_reader_rule_rzgfh.py")'
assert _q2.count(_old) == 1
_q2 = _q2.replace(_old, 'base == (WRAPPER_OF.get(arm, "v8f_t1_eval_reader_rule_rzgfhq.py") if arm in P_ARMS else "v8f_t1_eval_reader_rule_rzgfh.py")')
exec(compile(_q2, "<q2_argv, card ADD152: the new arm's wrapper is rzgfhqet>", "exec"), M.__dict__)


def c_later(a, seeds):
    per, ok = {}, True
    for m in seeds:
        for arm in (NEW, PART):
            key = M.key_of(arm, m)
            p = os.path.join(a.pull, key, "ticks", "rule_log_mid_%s.npz" % key)
            got = None
            if os.path.isfile(p):
                with np.load(p, allow_pickle=True) as z:
                    got = sorted(set(int(x) for x in z["q"]))
            per[key] = {"q_values": got, "PASS": got == list(range(1, 10))}
            ok &= got == list(range(1, 10))
    return {"per_run": per, "PASS": bool(ok and per)}


def c_card(a, seeds):
    q = E.B.load_queue(a.queue)
    return HC.c_tables(os.path.join(HERE, "pull"), head_sha=q.PINNED_SHA["head_table"], zero_sha=q.PINNED_SHA["zero_table"])


def c_fed_card(a, seeds):
    q = E.B.load_queue(a.queue)
    return HC.c_fed(a.pull, seeds, (NEW,), os.path.join(HC.TABLE_DIR, "HEAD_TABLE.npz"), q.PINNED_SHA["head_table"])


E.c_later, E.c_card, E.c_fed_card = c_later, c_card, c_fed_card
_src = inspect.getsource(E.evaluate)
_old = '"q3_partners": M.q3_partners(a, seeds), "later_looks": c_later(a, seeds),'
assert _src.count(_old) == 1
_src = _src.replace(_old, '"q3_partners": M.q3_partners(a, seeds), "later_looks": c_later(a, seeds), "t1_t2_t3": c_card(a, seeds), "t4_t5_fed": c_fed_card(a, seeds),')
exec(compile(_src, E.__file__, "exec"), E.__dict__)


def judge(blk):
    u, c = HC.u_word(blk["headway"]["U"]), HC.u_word(blk["headway"]["C"])
    worse = [f for f in E.FAMS if blk["families"][f] == "WORSE"]
    better = [f for f in E.FAMS if blk["families"][f] == "BETTER"]
    return ({"U": u, "C": c, "F": HC.u_word(blk["headway"]["F"]), "families_WORSE": worse, "families_BETTER": better, "families": blk["families"],
             "consequence": "card section 5: combined with the grid141 read by combine_headclose.py"}, "HEADCLOSE10K-P30-U-%s" % u)


E.judge = judge

if __name__ == "__main__":
    sys.exit(E.main())
