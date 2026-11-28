"""read_headclose10k.py -- card PREREG_V8F_PROOF_ADD152 (HEADCLOSE10K): the grid141 read. read_rulefc10k.py (card ADD144) used as a module with the arms, pairs, bar,
controls and names of this card; its engines (the landed headway, family and route-goal readers through the card ADD139 reader) are unchanged. Tier: T1-kin/hb0.2;
never driving performance. Never run on data by the tool builder before the runs ended.
  --mode main     pair V1 = H_RQ50KV1 - H_RQ50KE1 (bar: spread of H_RQ50KE1)
  --mode ceiling  pairs C1 = H_RQ50KC1 - H_RQ50KE1 and V1C1 = H_RQ50KV1 - H_RQ50KC1 (card section 3, conditional arm; bar: spread of H_RQ50KE1)
Controls: t1 to t3 (hc_common.c_tables), r4 runs (argv equal to the queue row), r4 partner (sha list of the box-2 H_RQ50KE1 dump files), later looks q = 1..9 in
every run, t4/t5 (hc_common.c_fed on the arms fed from the head table), the landed readers' controls.
Usage: python read_headclose10k.py --mode main --pull PULL --read-dir D --out OUT.json --partners-sha SHA.txt [--skip-engines]"""
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import hc_common as HC  # noqa: E402
import read_rulefc10k as RD  # noqa: E402

MODE = "ceiling" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "ceiling" else "main"
if "--mode" in sys.argv:
    i = sys.argv.index("--mode")
    del sys.argv[i:i + 2]

RD.CARD, RD.PRE, RD.SPREAD_ARM = "PREREG_V8F_PROOF_ADD152", "HEADCLOSE10K", "H_RQ50KE1"
RD.FC_ARMS = ()
if MODE == "main":
    RD.NEW_ARMS = ("H_RQ50KV1",)
    RD.PAIRS = {"V1": ("H_RQ50KV1", RD.SPREAD_ARM)}
    FED_ARMS = ("H_RQ50KV1",)
else:
    RD.NEW_ARMS = ("H_RQ50KC1", "H_RQ50KV1")
    RD.PAIRS = {"C1": ("H_RQ50KC1", RD.SPREAD_ARM), "V1C1": ("H_RQ50KV1", "H_RQ50KC1")}
    FED_ARMS = ("H_RQ50KV1",)
RD.ALL_ARMS = RD.NEW_ARMS + (RD.SPREAD_ARM,)
RD.CONSEQUENCE = "card section 5: combined with the P30 read by combine_headclose.py"
RD.DEFAULT_QUEUE = os.path.join(HERE, "box2_headclose_queue.py")


def c_later(a, seeds):
    per, ok = {}, True
    for m in seeds:
        for arm in RD.ALL_ARMS:
            key = RD.key_of(arm, m)
            p = os.path.join(a.pull, key, "ticks", "rule_log_mid_%s.npz" % key)
            got = None
            if os.path.isfile(p):
                with np.load(p, allow_pickle=True) as z:
                    got = sorted(set(int(x) for x in z["q"]))
            good = got == list(range(1, 10))
            per[key] = {"q_values": got, "PASS": good}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def c_card(a, seeds, queue):
    return HC.c_tables(a.pull, head_sha=queue.PINNED_SHA["head_table"], zero_sha=queue.PINNED_SHA["zero_table"])


def c_fed_card(a, seeds, queue):
    return HC.c_fed(a.pull, seeds, FED_ARMS, os.path.join(HC.TABLE_DIR, "HEAD_TABLE.npz"), queue.PINNED_SHA["head_table"])


class _BothQueues:
    """the main and the ceiling rows of the card's queue, so r4 finds every new arm's row"""

    def __init__(self, q):
        self.q = q
        self.PINNED_SHA = q.PINNED_SHA

    def build_runs(self, *a, **k):
        return self.q.build_runs() + self.q.build_runs(ceiling=True)


_orig_c_runs = RD.c_runs


def c_runs_card(a, seeds, queue):
    return _orig_c_runs(a, seeds, _BothQueues(queue))


RD.c_later, RD.c_card, RD.c_fed_card, RD.c_runs_card = c_later, c_card, c_fed_card, c_runs_card
_src = inspect.getsource(RD.evaluate)
_old = 'C = {"r1": c_r1(a.r1), "r4_runs": c_runs(a, seeds, queue), "r4_partner": c_partner(a, seeds)}'
assert _src.count(_old) == 1
_src = _src.replace(_old, 'C = {"t1_t2_t3": c_card(a, seeds, queue), "r4_runs": c_runs_card(a, seeds, queue), "r4_partner": c_partner(a, seeds), '
                          '"later_looks": c_later(a, seeds), "t4_t5_fed": c_fed_card(a, seeds, queue)}')
exec(compile(_src, RD.__file__, "exec"), RD.__dict__)


def judge(blocks):
    out, names = {}, {}
    for nm, b in blocks.items():
        u = HC.u_word(b["headway"]["U"])
        names[nm] = ("HEADCLOSE10K-U-%s" % u) if nm == "V1" else ("HEADCLOSE10K-%s-U-%s" % (nm, u))
        out[nm] = {"U": u, "C": HC.u_word(b["headway"]["C"]), "F": HC.u_word(b["headway"]["F"]),
                   "U_per_seed_signs": ["-" if x < 0 else ("+" if x > 0 else "0") for x in b["headway"]["U"]["per_seed"]],
                   "families_WORSE": [f for f in RD.FAMS if b["families"][f] == "WORSE"], "families_BETTER": [f for f in RD.FAMS if b["families"][f] == "BETTER"],
                   "families": b["families"], "consequence": RD.CONSEQUENCE}
    return out, names


RD.judge = judge

if __name__ == "__main__":
    sys.argv += [] if "--r1" in sys.argv else ["--r1", "unused"]
    sys.exit(RD.main())
