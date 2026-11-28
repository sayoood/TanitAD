"""read_rulee1a10k.py -- card PREREG_V8F_PROOF_ADD147 (RULEE1A10K): the read. The age read of card ADD140 (rhacc10k/read_rhk/read_rhk10k_am2.py, i.e. read_rhk10k.py
with amendment 2, imported as modules and NOT edited) with the arms of this card: R3 = H_RQ50KE1_P30 - H_RQ50K_P30 (reader form RR, readout K, pre-roll 30 steps on the
1,271-window grid; bar: spread of H_RQ50K_P30). Its run checks (q1 complete, q2 models, q2 argv, q3 partners) and its R3 engine (read_longroll10k_n4 grid engines with n1 to n3,
n1 on H_RLcA_P0; route goal with s2 and s3 waived) are used unchanged; the registration gives each arm its own rule interval (1 or 5).
Never run on data by the tool builder. Tier: T1-kin/hb0.2; never driving performance.
Usage: python read_rulee1a10k.py --pull PULL --read-dir D --out OUT.json --partners-sha SHA.txt [--skip-engines]"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(V8, "rhacc10k", "read_rhk"))
import read_rhk10k_am2 as AM2  # noqa: E402

M = AM2.M
B = M.B
CARD, PRE = "PREREG_V8F_PROOF_ADD147", "RULEE1A10K"
NEW, PART = "H_RQ50KE1_P30", "H_RQ50K_P30"
SPEC = {NEW: ("RR", "K", 30, "deadreckon", "30", "plan", 1, 0.2), PART: ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2)}
M.NEW_ARMS, M.PARTNER_ARMS, M.FRESH_ARMS = (NEW,), (PART,), ()
M.P_ARMS = (NEW, PART)
M.ALL_ARMS = M.P_ARMS
M.R3_PAIR, M.SPREAD_ARM_R3 = (NEW, PART), PART
M.DEFAULT_QUEUE = os.path.join(HERE, "box2_rulee1a_queue.py")
M.DEFAULT_PARTNER_QUEUE = os.path.join(V8, "rhacc10k", "kit", "box2_crhc_queue.py")
FAMS = M.FAMS


def register_arms(R):
    for m in M.SEEDS:
        for arm, spec in SPEC.items():
            R.ALL_RUNS[M.key_of(arm, m)] = spec
        for arm in M.N1_ARMS:
            R.ALL_RUNS[M.key_of(arm, m)] = M.N1_SPEC


M.register_arms = register_arms


def c_later(a, seeds):
    per, ok = {}, True
    for m in seeds:
        for arm, want in ((NEW, list(range(1, 10))), (PART, [5])):
            key = M.key_of(arm, m)
            p = os.path.join(a.pull, key, "ticks", "rule_log_mid_%s.npz" % key)
            got = None
            if os.path.isfile(p):
                with np.load(p, allow_pickle=True) as z:
                    got = sorted(set(int(x) for x in z["q"]))
            per[key] = {"q_values": got, "want": want, "PASS": got == want}
            ok &= got == want
    return {"per_run": per, "PASS": bool(ok and per)}


def u_card(u):
    lo, hi = u["interval"]
    return "LOWER" if hi < 0 else ("HIGHER" if lo > 0 else "NOT-SEPARATED")


def judge(blk):
    u, c = u_card(blk["headway"]["U"]), u_card(blk["headway"]["C"])
    worse = [f for f in FAMS if blk["families"][f] == "WORSE"]
    better = [f for f in FAMS if blk["families"][f] == "BETTER"]
    if u != "HIGHER" and c != "HIGHER" and not worse:
        cons = "every-step stays the rule of the candidate's reader form at every age (row 211's consequence stands)"
    else:
        cons = "to the PI with its cost, together with row 211 (U %s, C %s, families WORSE: %s)" % (u, c, ", ".join(worse) or "none")
    return {"U": u, "C": c, "families_WORSE": worse, "families_BETTER": better, "families": blk["families"], "consequence": cons}, "%s-U-%s" % (PRE, u)


def evaluate(a):
    seeds = tuple(a.seeds)
    qnew, qpart = B.load_queue(a.queue), B.load_queue(a.partner_queue)
    inputs = B.jl(a.inputs)
    grid_n = len(json.load(io.open(a.grid_p30, encoding="utf-8"))["order"])
    r3 = B.jl(os.path.join(a.read_dir, "r3", "R3_READ.json"))
    C = {"q1_complete": M.q1_complete(a, seeds, grid_n), "q2_models": M.q2_models(a, seeds, qnew, inputs), "q2_argv": M.q2_argv(a, seeds, qnew, qpart),
         "q3_partners": M.q3_partners(a, seeds), "later_looks": c_later(a, seeds),
         "r3_engines_and_controls": {"PASS": bool(r3 and r3.get("ALL_PASS") is True), "engine_controls_pass": (r3 or {}).get("engine_controls_pass"),
                                     "route_goal_pass": ((r3 or {}).get("strategic") or {}).get("ALL_PASS")}}
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": M.TIER_LINE, "seeds": list(seeds), "spread_arm": PART, "controls": C, "controls_all_pass": bool(ok), "name": None, "checks": None, "pair": None,
           "estimator": "paired episode-cluster bootstrap pooled over the three model seeds (age engines of card ADD140); it answers another draw of episodes"}
    print(M.TIER_LINE)
    for k, v in C.items():
        print("   %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL"))
    if ok:
        pr = r3["pair"]
        blk = {"A": NEW, "B": PART, "headway": B.ucf(pr["headway"]), "family_cells": B.cells_of(pr["family"]), "strategic": r3["strategic"]["pairs"]["R3"]}
        blk["families"] = {f: B.reading(blk["family_cells"], f) for f in M.NAMED}
        blk["families"]["STRATEGIC"] = blk["strategic"]["reading"]
        chk, name = judge(blk)
        res.update(name=name, checks=chk, pair=blk, levels=r3.get("levels"), windows=r3.get("windows"))
        print("== name %s  families %s" % (name, chk["families"]))
        print("== U %s  C %s  F %s" % (blk["headway"]["U"], blk["headway"]["C"], blk["headway"]["F"]))
        print("== consequence: %s" % chk["consequence"])
    else:
        print("== a control did not pass: no name, no level, no difference")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(B.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0 if ok else 2


def main(argv=None):
    a = M.parse(argv)
    if not a.skip_engines:
        os.makedirs(os.path.join(a.read_dir, "r3"), exist_ok=True)
        log = lambda s: print(s, flush=True)                                                    # noqa: E731
        try:
            res = M.r3_engine(a, log)
        except BaseException as ex:                                                             # noqa: BLE001
            res = {"ALL_PASS": False, "error": "%s: %s" % (type(ex).__name__, str(ex)[:400])}
        json.dump(B.conv(res), io.open(os.path.join(a.read_dir, "r3", "R3_READ.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
