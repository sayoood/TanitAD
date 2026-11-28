"""read_rulefc10k.py -- card PREREG_V8F_PROOF_ADD144 (RULEFC10K) with amendment 1: the read. Uses the reader of card ADD139 (rhacc10k/read/read_rhacc10k.py) as a
module, NOT edited: its subprocess helper for the landed readers headway_read_r2.py and q3_read_r3.py, the route-goal engine (strat_roll_read.py), the pair block
and the family rule. Never run on data by the tool builder. Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead
does not react); never driving performance.

Runs (run folder <pull>/<ARM>_s<m>, model seeds 0, 1, 2): H_RLcE1, H_RLcFC, H_RLcE1FC (this card) and the partner H_RLcK (card ADD140).
Pairs: E1 = H_RLcE1 - H_RLcK; FC = H_RLcFC - H_RLcK; E1FC = H_RLcE1FC - H_RLcK. Bar of the families and of the headway cells: spread of H_RLcK over the seeds (card section 5).
Names (card section 5): per arm X, U LOWER if the pooled interval of U lies below zero, HIGHER if above, NOT-SEPARATED otherwise (the per-seed signs are reported beside);
families beside each name; consequence as committed.
Controls (all must pass before any number is opened): r1 (r1_check.py result), r4 (done files, 141 dump files, finite plans, rule logs, forecast logs with a_L on the FC
runs and the zero flag off, argv equal to the queue row, the partner folders equal to the box-2 folders by the sha list), the landed readers' own controls.
Usage: python read_rulefc10k.py --pull PULL --read-dir D --out OUT.json --r1 R1.json --partners-sha SHA.txt [--skip-engines]"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, ".."))
RH10 = os.path.join(V8, "rhacc10k")
sys.path.insert(0, os.path.join(RH10, "read"))
import read_rhacc10k as B  # noqa: E402  (landed ADD139 reader, used as a module, never edited)

CARD = "PREREG_V8F_PROOF_ADD144"
PRE = "RULEFC10K"
TIER_LINE = B.TIER_LINE
SPREAD_ARM = "H_RLcK"
NEW_ARMS = ("H_RLcE1", "H_RLcFC", "H_RLcE1FC")
FC_ARMS = ("H_RLcFC", "H_RLcE1FC")
ALL_ARMS = NEW_ARMS + (SPREAD_ARM,)
PAIRS = {"E1": ("H_RLcE1", SPREAD_ARM), "FC": ("H_RLcFC", SPREAD_ARM), "E1FC": ("H_RLcE1FC", SPREAD_ARM)}
FAMS = B.FAMS
NAMED = B.NAMED
CONSEQUENCE = ("an arm with U LOWER and no family WORSE is the rule of the candidate's label form to be carried to the reader form (a successor card); "
               "an arm with U LOWER and a family WORSE goes to the PI with its cost.")
DEFAULT_QUEUE = os.path.join(HERE, "kit", "box2_rulefc_queue.py")


def key_of(arm, m):
    return B.key_of(arm, m)


def jl(p):
    return B.jl(p)


# ============================================================================================ controls
def c_r1(path):
    r = jl(path) or {}
    return {"file": os.path.basename(path or ""), "PASS": r.get("pass") is True, "checks": {k: v.get("pass") for k, v in (r.get("checks") or {}).items()}}


def c_runs(a, seeds, queue):
    rows = {r["arm"]: r for r in queue.build_runs()}
    per, ok = {}, True
    for m in seeds:
        for arm in ALL_ARMS:
            key = key_of(arm, m)
            d = os.path.join(a.pull, key)
            files = B.dump_files(a.pull, arm, m)
            bad = 0
            for f in files:
                try:
                    with np.load(f, allow_pickle=True) as z:
                        if "cl" not in z.files or not np.isfinite(np.asarray(z["cl"], np.float64)).all():
                            bad += 1
                except Exception:                                                               # noqa: BLE001
                    bad += 1
            done = jl(os.path.join(d, "RUN_DONE.json"))
            tk = os.path.isfile(os.path.join(d, "ticks", "ticks_%s_m10_causal_heads.npz" % key))
            rl = os.path.isfile(os.path.join(d, "ticks", "rule_log_%s.npz" % key)) and os.path.isfile(os.path.join(d, "ticks", "rule_log_mid_%s.npz" % key))
            why = []
            fc_ok = True
            if arm in FC_ARMS:
                p = os.path.join(d, "ticks", "fc_log_%s.npz" % key)
                fc_ok = False
                if os.path.isfile(p):
                    with np.load(p) as z:
                        fc_ok = bool((not bool(z["zero"])) and len(z["a_L"]) > 0 and np.isfinite(z["a_L"]).all() and (z["a_L"] != 0).any())
                if not fc_ok:
                    why.append("forecast log missing, zero flag on, or a_L all zero")
            argv_ok = True
            if arm in NEW_ARMS:
                rec = (jl(os.path.join(d, "argv.json")) or {}).get("argv")
                argv_ok = bool(rec and key in rows and rec == rows[key]["argv"])
                if not argv_ok:
                    why.append("argv differs from the queue row")
            good = bool(done and len(files) == a.expect_files and bad == 0 and tk and rl and fc_ok and argv_ok)
            if len(files) != a.expect_files:
                why.append("%d dump files, expected %d" % (len(files), a.expect_files))
            if bad:
                why.append("%d files with a non-finite plan" % bad)
            if not (done and tk and rl):
                why.append("done file, ticks file or rule logs missing")
            per[key] = {"dump_files": len(files), "nonfinite": bad, "fc_log_ok": fc_ok, "argv_ok": argv_ok, "PASS": good, "why": why}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def c_partner(a, seeds):
    """r4 partner: every dump file of the pulled H_RLcK_s<m> equals the sha list taken on box 2 (sha256, relative path)."""
    want = {}
    if a.partners_sha and os.path.isfile(a.partners_sha):
        for line in io.open(a.partners_sha, encoding="utf-8"):
            p = line.strip().split()
            if len(p) == 2:
                want[p[1].replace("\\", "/")] = p[0]
    per, ok = {}, bool(want)
    for m in seeds:
        key = key_of(SPREAD_ARM, m)
        files = B.dump_files(a.pull, SPREAD_ARM, m)
        diff = 0
        n = 0
        for f in files:
            rel = "%s/%s/%s" % (key, os.path.basename(os.path.dirname(f)), os.path.basename(f))
            n += 1
            if want.get(rel) != B.sha(f):
                diff += 1
        listed = len([k for k in want if k.startswith(key + "/")])
        good = bool(n == a.expect_files and listed == n and diff == 0)
        per[key] = {"files": n, "listed": listed, "differing": diff, "PASS": good}
        ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


# ============================================================================================ engines
def run_engines(a):
    os.makedirs(a.read_dir, exist_ok=True)
    B.SPREAD_ARM = SPREAD_ARM                                    # the bar of this card (section 5); read by B.sub_read
    pairs = ",".join("%s:%s" % (x, y) for x, y in PAIRS.values())
    status = {}
    rc, cmd = B.sub_read(a, a.headway_dir, "headway_read_r2.py", os.path.join(a.read_dir, "hw"), pairs, "hw.log")
    status["headway"] = {"rc": rc, "cmd": cmd}
    rc, cmd = B.sub_read(a, a.q3_dir, "q3_read_r3.py", os.path.join(a.read_dir, "q3"), pairs, "q3.log", extra=("--same-sign",))
    status["families"] = {"rc": rc, "cmd": cmd}
    log = lambda s: print(s, flush=True)                                                        # noqa: E731
    os.makedirs(os.path.join(a.read_dir, "strat"), exist_ok=True)
    try:
        res = B.strategic_engine(a, list(ALL_ARMS), dict(PAIRS), SPREAD_ARM, {}, None, log)
    except BaseException as ex:                                                                 # noqa: BLE001
        res = {"ALL_PASS": False, "error": "%s: %s" % (type(ex).__name__, str(ex)[:400])}
    json.dump(B.conv(res), io.open(os.path.join(a.read_dir, "strat", "STRAT_READ.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    status["route_goal"] = {"ALL_PASS": res.get("ALL_PASS")}
    json.dump(B.conv(status), io.open(os.path.join(a.read_dir, "ENGINES_STATUS.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return status


# ============================================================================================ names
def u_card(u):
    """card section 5: the pooled interval alone decides; per-seed signs beside."""
    lo, hi = u["interval"]
    return "LOWER" if hi < 0 else ("HIGHER" if lo > 0 else "NOT-SEPARATED")


def judge(blocks):
    """names and the committed consequence from the pair blocks (pure function of the numbers)"""
    out, names = {}, {}
    for nm, b in blocks.items():
        u = u_card(b["headway"]["U"])
        worse = [f for f in FAMS if b["families"][f] == "WORSE"]
        better = [f for f in FAMS if b["families"][f] == "BETTER"]
        names[nm] = "%s-%s-U-%s" % (PRE, nm, u)
        if u == "LOWER" and not worse:
            cons = "the rule of the candidate's label form, to be carried to the reader form (a successor card)"
        elif u == "LOWER":
            cons = "to the PI with its cost (families WORSE: %s)" % ", ".join(worse)
        else:
            cons = "no consequence committed by the card for U %s" % u
        out[nm] = {"U": u, "U_per_seed_signs": ["-" if x < 0 else ("+" if x > 0 else "0") for x in b["headway"]["U"]["per_seed"]],
                   "families_WORSE": worse, "families_BETTER": better, "families": b["families"], "consequence": cons}
    return out, names


def evaluate(a):
    seeds = tuple(a.seeds)
    queue = B.load_queue(a.queue)
    hw = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json"))
    q3 = jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    st = jl(os.path.join(a.read_dir, "strat", "STRAT_READ.json"))
    C = {"r1": c_r1(a.r1), "r4_runs": c_runs(a, seeds, queue), "r4_partner": c_partner(a, seeds)}
    cnt = (hw or {}).get("counts", {})
    e = {"headway": bool(hw and (hw.get("controls") or {}).get("ALL_PASS")), "families": bool(q3 and (q3.get("controls") or {}).get("ALL_PASS")),
         "route_goal": bool(st and st.get("ALL_PASS") is True),
         "headway_counts": {"n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "PASS": cnt.get("PASS")}}
    e["PASS"] = bool(e["headway"] and e["families"] and e["route_goal"] and cnt.get("PASS") is True and cnt.get("n_windows_total") == 2399)
    C["reader_controls"] = e
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "seeds": list(seeds), "spread_arm": SPREAD_ARM, "controls": C, "controls_all_pass": bool(ok),
           "names": None, "checks": None, "pairs": None, "levels": None, "consequence_sentence": CONSEQUENCE,
           "estimator": "paired episode-cluster bootstrap pooled over the three model seeds (landed readers); it answers another draw of episodes, not another training or inference run"}
    print(TIER_LINE)
    print("== controls")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(B.conv({q: w for q, w in v.items() if q not in ("per_run", "per_seed")}))[:300]))
    if ok:
        blocks = {nm: B.pair_block(hw, q3, st, x, y, nm) for nm, (x, y) in PAIRS.items()}
        checks, names = judge(blocks)
        lv = hw["levels"]
        res.update(names=names, checks=checks, pairs=blocks,
                   levels={arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in lv})
        print("== names")
        for k, v in names.items():
            print("   %-5s %s  families %s" % (k, v, checks[k]["families"]))
        for nm, b in blocks.items():
            print("== %s: %s minus %s  U %s  C %s  F %s" % (nm, b["A"], b["B"], b["headway"]["U"], b["headway"]["C"], b["headway"]["F"]))
    else:
        print("== a control did not pass: no name, no level, no difference")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(B.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0 if ok else 2


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r1", required=True)
    ap.add_argument("--partners-sha", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-engines", action="store_true")
    ap.add_argument("--expect-files", type=int, default=141)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--join", default=RH10 + "/pins/join_eval141.npz")
    ap.add_argument("--bank", default=V8 + "/res/cache/frames.npz")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--tree-ov", default=RH10 + "/tree")
    ap.add_argument("--code", default="")
    ap.add_argument("--headway-dir", default=V8 + "/headway")
    ap.add_argument("--q3-dir", default=V8 + "/spatial/q3")
    ap.add_argument("--stageb-dir", default=V8 + "/pod/stageB")
    ap.add_argument("--tev-dir", default=RH10 + "/tree/taniteval")
    a = ap.parse_args(argv)
    a.seeds = tuple(int(x) for x in a.seeds.split(",") if x != "")
    if not a.code:
        a.code = os.path.join(a.read_dir, "_code")
    return a


def main(argv=None):
    a = parse(argv)
    if not a.skip_engines:
        run_engines(a)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
