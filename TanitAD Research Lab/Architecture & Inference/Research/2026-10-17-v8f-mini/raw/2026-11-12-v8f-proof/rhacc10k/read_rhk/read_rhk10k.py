"""read_rhk10k.py -- card PREREG_V8F_PROOF_ADD140 (RHK10K): the read. Derived from the reader of card ADD139 (read/read_rhacc10k.py, imported as a module and NOT edited);
its helpers and engines are reused, the arms, pairs, controls and names are those of card ADD140.
Never run on data by the tool builder. Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance.

Runs (run folder <pull>/<ARM>_s<m>, model seeds 0,1,2): new i H_RLcK, j H_RQ50K (eval-141, 2,399 windows), k H_RLcK_P30, l U_RLcK_P30 (grid of card ADD70, pre-roll 30 steps);
reused partners U_RLcK, U_RQ50K of card ADD139 (eval-141).
Pairs: R1 = H_RLcK - U_RLcK; R2 = H_RQ50K - U_RQ50K; R3 = H_RLcK_P30 - U_RLcK_P30 (two PRE-ROLLED arms on the same windows).
Engines (landed readers unchanged): headway_read_r2.py and q3_read_r3.py as subprocesses on R1, R2 (bar: spread of U_RLcK); route goal (strat_roll_read.py, card ADD66) on R1, R2;
R3 with read_longroll10k_n4.py (read_grid, engines, n1 to n3 controls) in memory, bar: spread of U_RLcK_P30, plus the route goal on the same windows.
STATED CHOICES (the card leaves them open):
  R3 route goal: strat_roll_read's controls s2 and s3 need a run that starts at the logged state; both arms of R3 are pre-rolled, so s2 and s3 are WAIVED for R3 and recorded as waived
                 in the result (s1, s4, s6, s9 and the episode map still gate). The headline of the hand-back names this. If the lead rules s2/s3 must run, R3 route goal is BLOCKED.
  FRESH EQUAL also requires that no family reads MIXED (a MIXED family has separated cells); the card text says only 'no family WORSE or BETTER'.
  family reading, WORSE only when the family reads WORSE (as ADD139); STRATEGIC = the three arcs 30, 50, 80 m at the third tick.
  FRESH: families of R1 and R2; U is read from the headway U cell of each pair (HIGHER = separated higher in every seed). AGED: families of R3; U is reported beside, not in the name.
  q3 extra files in a partner folder (present but not in the sha list) are reported and do not fail; a listed file that is missing or differs fails.
  Controls of ADD139 not in section 4 of this card (k4 environment, k5 reader input) are not repeated.
Usage: python read_rhk10k.py --pull PULL --read-dir D --out OUT.json [--tracks native_tracks.pt] [--skip-engines]"""
import argparse
import inspect
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RH10 = "F:/TanitAD-scratch/v8proof/rhacc10k"
sys.path.insert(0, RH10 + "/read")
import read_rhacc10k as B  # noqa: E402  (landed ADD139 reader, used as a module, never edited)

V8, INTEG = B.V8, B.INTEG
CARD = "PREREG_V8F_PROOF_ADD140"
PRE = "RHK10K"
TIER_LINE = B.TIER_LINE
SEEDS = (0, 1, 2)
SPREAD_ARM = "U_RLcK"
SPREAD_ARM_R3 = "U_RLcK_P30"
NEW_ARMS = ("H_RLcK", "H_RQ50K", "H_RLcK_P30", "U_RLcK_P30")           # rows i, j, k, l
PARTNER_ARMS = ("U_RLcK", "U_RQ50K")
FRESH_ARMS = ("H_RLcK", "H_RQ50K", "U_RLcK", "U_RQ50K")
P_ARMS = ("H_RLcK_P30", "U_RLcK_P30")
ALL_ARMS = FRESH_ARMS + P_ARMS
PAIRS = {"R1": ("H_RLcK", "U_RLcK"), "R2": ("H_RQ50K", "U_RQ50K")}
R3_PAIR = ("H_RLcK_P30", "U_RLcK_P30")
FAMS = B.FAMS
NAMED = B.NAMED
CONSEQUENCE = ("FRESH-BETTER or FRESH-EQUAL with AGED-BETTER makes RH under K the candidate of record in place of UE under K (specification section 1, model row); "
               "FRESH-WORSE keeps the line candidate and sends the recipe question back to the other session with the failing cells.")
DEFAULT_QUEUE = RH10 + "/kit/box2_rhk_queue.py"
DEFAULT_PARTNER_QUEUE = RH10 + "/kit/box2_rhacc_queue.py"
DEFAULT_SHA = RH10 + "/PARTNERS_ADD139_SHA.txt"
R3_SPEC = ("RLc", "K", 30, "deadreckon", "30", "plan", 5, 0.2)          # registration for read_longroll10k's ALL_RUNS: form, readout K, pre-roll 30, route, ..., plan, rule every 5, kappa


def key_of(arm, m):
    return B.key_of(arm, m)


def fam_of_arm(arm):
    return "RH" if arm.startswith("H_") else "UE"


# ============================================================================================ q1 to q4
def q1_complete(a, seeds, grid_n):
    per, ok = {}, True
    for m in seeds:
        for arm in ALL_ARMS:
            key = key_of(arm, m)
            rd = B.run_dir(a.pull, arm, m)
            done = B.jl(os.path.join(rd, "RUN_DONE.json"))
            files = B.dump_files(a.pull, arm, m)
            want = grid_n if arm in P_ARMS else a.expect_files
            bad, why = 0, []
            for f in files:
                try:
                    with np.load(f) as z:
                        if "cl" not in z.files or not np.isfinite(np.asarray(z["cl"], np.float64)).all():
                            bad += 1
                except Exception:                                                               # noqa: BLE001
                    bad += 1
            rule_log = os.path.isfile(os.path.join(rd, "ticks", "rule_log_%s.npz" % key))
            pre_json = os.path.isfile(os.path.join(rd, "ticks", "preroll_log_%s.json" % key))
            pre_npz = os.path.isfile(os.path.join(rd, "ticks", "preroll_offsets_%s.npz" % key))
            pre_ok = (pre_json and pre_npz) if arm in P_ARMS else True
            good = bool(done is not None and done.get("rc") == 0 and len(files) == want and bad == 0 and rule_log and pre_ok)
            if done is None or done.get("rc") != 0:
                why.append("done file missing or rc != 0")
            if len(files) != want:
                why.append("dump files %d, expected %d" % (len(files), want))
            if bad:
                why.append("%d files with a non-finite or unreadable plan" % bad)
            if not rule_log:
                why.append("rule log missing")
            if not pre_ok:
                why.append("pre-roll log missing")
            per[key] = {"done": done is not None, "dump_files": len(files), "expected": want, "files_with_nonfinite_plan": bad, "rule_log": rule_log,
                        "preroll_log": pre_ok if arm in P_ARMS else None, "PASS": good, "why": why}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def q2_models(a, seeds, queue, inputs):
    per, ok = {}, True
    table = getattr(queue, "CKPT_SHA", {})
    for m in seeds:
        for arm in ALL_ARMS:
            key = key_of(arm, m)
            rec = B.jl(os.path.join(B.run_dir(a.pull, arm, m), "argv.json"))
            ck = B.argv_flag(rec["argv"], "--ckpt") if rec and rec.get("argv") else None
            want_key = "%s_s%d" % (fam_of_arm(arm), m)
            got_key = os.path.basename(os.path.dirname(ck.replace("\\", "/"))) if ck else None
            rec_sha = ((inputs or {}).get("inputs/%s/ckpt_last.pt" % want_key) or {}).get("sha256")
            rr = B.txt(os.path.join(B.run_dir(a.pull, arm, m), "ticks", "run_%s.txt" % key))
            import re
            mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", rr)
            good = bool(rec_sha and got_key == want_key and rec_sha == table.get(want_key) and rec_sha.startswith(B.CARD_CKPT_PREFIX[want_key]) and (mm is None or mm.group(1) == rec_sha))
            per[key] = {"ckpt_key_in_argv": got_key, "expected_key": want_key, "recorded_sha256_prefix": (rec_sha or "")[:16], "run_record_sha_checked": bool(mm), "PASS": good}
            ok &= good
    return {"per_run": per, "note": "hashes compared as recorded (INPUTS.json vs the queue table and the card prefixes); a checkpoint is not hashed here", "PASS": bool(ok and per)}


def q2_argv(a, seeds, qnew, qpart):
    rows_new = {r["arm"]: (r, qnew.paths()) for r in qnew.build_runs()}
    rows_part = {r["arm"]: (r, qpart.paths()) for r in qpart.build_runs()}
    per, ok = {}, True
    for m in seeds:
        for arm in ALL_ARMS:
            key = key_of(arm, m)
            rec = B.jl(os.path.join(B.run_dir(a.pull, arm, m), "argv.json"))
            row, P = (rows_new if arm in NEW_ARMS else rows_part).get(key, (None, None))
            argv = rec.get("argv") if rec else None
            want_w = P[row.get("wrapper", "wrapper")] if row else None
            base = os.path.basename(want_w or "")
            w_ok = bool(argv and len(argv) > 1 and argv[1] == want_w and base == ("v8f_t1_eval_reader_rule_rzgfhq.py" if arm in P_ARMS else "v8f_t1_eval_reader_rule_rzgfh.py"))
            good = bool(row and argv == row["argv"] and w_ok)
            per[key] = {"recorded": rec is not None, "row_in_queue": row is not None, "queue": "box2_rhk_queue" if arm in NEW_ARMS else "box2_rhacc_queue",
                        "wrapper_in_argv": argv[1] if argv and len(argv) > 1 else None, "wrapper_ok": w_ok, "PASS": good}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def parse_sha_list(path):
    out = {}
    for ln in B.txt(path).splitlines():
        ln = ln.rstrip("\r\n")
        if not ln.strip():
            continue
        h, _, p = ln.partition(" ")
        out[p.lstrip("*").strip()] = h.strip()
    return out


def q3_partners(a, seeds):
    lst = parse_sha_list(a.partners_sha)
    per, ok = {}, bool(lst)
    for m in seeds:
        for arm in PARTNER_ARMS:
            key = key_of(arm, m)
            mine = {p: h for p, h in lst.items() if p.split("/")[0] == key}
            n_bad, missing, differ = 0, [], []
            for p, h in mine.items():
                fp = os.path.join(a.pull, p)
                if not os.path.isfile(fp):
                    missing.append(p)
                elif B.sha(fp) != h:
                    differ.append(p)
            extra = []
            base = os.path.join(a.pull, key)
            for root, _, fs in os.walk(base):
                for f in fs:
                    rel = os.path.relpath(os.path.join(root, f), a.pull).replace("\\", "/")
                    if rel not in mine:
                        extra.append(rel)
            good = bool(mine and not missing and not differ)
            per[key] = {"listed": len(mine), "missing": missing[:5], "n_missing": len(missing), "differ": differ[:5], "n_differ": len(differ), "n_extra_files_reported_only": len(extra), "PASS": good}
            ok &= good
    return {"per_run": per, "note": "sha256 of every listed file vs %s; extra files are reported, not failed" % os.path.basename(a.partners_sha), "PASS": bool(ok)}


def window_list(pull, arm, m):
    pairs = []
    for f in B.dump_files(pull, arm, m):
        with np.load(f, allow_pickle=True) as z:
            if "clip_index" not in z.files or "ws" not in z.files:
                return None
            pairs += [(int(c), int(w)) for c, w in zip(z["clip_index"], z["ws"])]
    return sorted(pairs)


def q4_identity(a, seeds):
    per, ok = {}, True
    for m in seeds:
        fa, fe = B.dump_files(a.pull, "H_RLcK", m), B.dump_files(a.pull, "U_RLcK", m)
        ha = {os.path.basename(f): B.sha(f) for f in fa}
        he = {os.path.basename(f): B.sha(f) for f in fe}
        identical = bool(ha and ha == he)
        wk, wl = window_list(a.pull, "H_RLcK_P30", m), window_list(a.pull, "U_RLcK_P30", m)
        same_w = bool(wk is not None and wl is not None and len(wk) > 0 and wk == wl)
        good = bool(fa and fe and not identical and same_w)
        per["s%d" % m] = {"dump_files_i_and_U_RLcK": [len(fa), len(fe)], "dumps_byte_identical": identical, "k_l_same_window_list": same_w,
                          "n_windows_k_l": [None if wk is None else len(wk), None if wl is None else len(wl)], "PASS": good}
        ok &= good
    return {"per_seed": per, "note": "i = H_RLcK against U_RLcK (every dump file hashed); k and l compared as sorted (clip, t0) lists", "PASS": bool(ok and per)}


# ============================================================================================ engines
_src = inspect.getsource(B.strategic_engine)
for _old, _new in (
        ("def strategic_engine(a, runs, pairs, s3_arm, p_of, keys=None, log=print):", "def strategic_engine_pre(a, runs, pairs, s3_arm, p_of, keys=None, log=print, waive=True):"),
        ('and s2["n_rows_anchor_not_t0"] == 0 and s2["files"])', 'and s2["n_rows_anchor_not_t0"] == 0 and (bool(s2["files"]) or waive))'),
        ("    if nm3 in inputs:\n", '    if waive:\n        C["s3"] = {"pass": True, "waived": True, "reason": "every run of this pair is pre-rolled: no run starts at the logged state, so the anchor checks s2 and s3 cannot be run"}\n    elif nm3 in inputs:\n')):
    assert _src.count(_old) == 1, _old
    _src = _src.replace(_old, _new)
_ns = dict(vars(B))
exec(compile(_src, "<strategic_engine_pre derived from read_rhacc10k.strategic_engine>", "exec"), _ns)
strategic_engine_pre = _ns["strategic_engine_pre"]


def register_arms(R):
    for arm in P_ARMS:
        for m in SEEDS:
            R.ALL_RUNS[key_of(arm, m)] = R3_SPEC


def r3_engine(a, log=print):
    """R3: the 1,271-window grid, two pre-rolled arms (RH and UE, both pre-roll 30 steps, readout K); read_longroll10k_n4 engines, controls n1 to n3, route goal on the same windows"""
    sys.path.insert(0, INTEG)
    env = {"HEADWAY_Q3DIR": a.q3_dir, "HIERREP_STAGEB": a.stageb_dir, "TANITEVAL_ROOT": a.tev_dir, "HEADWAY_JOIN": a.join, "HEADWAY_BANK": a.bank}
    os.environ.update(env)
    for d in (a.headway_dir, a.q3_dir, a.stageb_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    import read_longroll10k_n4 as N4
    R = N4.R
    R.SEEDS = tuple(a.seeds)
    R.key_of = key_of
    register_arms(R)
    ns = argparse.Namespace(pull=a.pull, code=a.code, n_boot=2000, n_boot_control=300, w=a.pull, tree=a.tree_ov)
    E = R.engine(ns)
    grid = json.load(io.open(a.grid_p30, encoding="utf-8"))
    clip_names = [str(c) for c in np.load(a.join, allow_pickle=True)["clip"]]
    keys = R.grid_keys(grid, clip_names)
    out = R.read_grid(ns, E, "30", keys, list(P_ARMS), SPREAD_ARM_R3, {"R3": R3_PAIR}, {}, log)
    cn = N4.controls_n4()
    res = {"restriction": out.get("restriction"), "counts": out.get("counts"), "controls_engines": out.get("controls"), "controls_n4": cn, "spread_arm": SPREAD_ARM_R3,
           "levels": out.get("levels"), "pair": out.get("pairs", {}).get("R3"), "windows": int(len(keys)), "arms": list(P_ARMS), "preroll_steps": {k: 30 for k in P_ARMS}}
    eng_ok = bool((out.get("controls") or {}).get("ALL_PASS") is True and all(v.get("PASS") is True for v in cn.values()))
    res["engine_controls_pass"] = eng_ok
    sres = strategic_engine_pre(a, list(P_ARMS), {"R3": R3_PAIR}, R3_PAIR[0], {k: 30 for k in P_ARMS}, keys=keys, log=log, waive=True)
    res["strategic"] = sres
    res["ALL_PASS"] = bool(eng_ok and sres.get("ALL_PASS") is True)
    return res


def run_engines(a):
    os.makedirs(a.read_dir, exist_ok=True)
    B.SPREAD_ARM = SPREAD_ARM                                    # same value as in ADD139; set so the subprocess helper reads the bar of this card
    pairs = ",".join("%s:%s" % (x, y) for x, y in PAIRS.values())
    status = {}
    rc, cmd = B.sub_read(a, a.headway_dir, "headway_read_r2.py", os.path.join(a.read_dir, "hw"), pairs, "hw.log")
    status["headway"] = {"rc": rc, "cmd": cmd}
    rc, cmd = B.sub_read(a, a.q3_dir, "q3_read_r3.py", os.path.join(a.read_dir, "q3"), pairs, "q3.log", extra=("--same-sign",))
    status["families"] = {"rc": rc, "cmd": cmd}
    log = lambda s: print(s, flush=True)                                                        # noqa: E731
    os.makedirs(os.path.join(a.read_dir, "strat"), exist_ok=True)
    os.makedirs(os.path.join(a.read_dir, "r3"), exist_ok=True)
    for nm, fn, folder, fname in (("route_goal", lambda: B.strategic_engine(a, list(FRESH_ARMS), dict(PAIRS), SPREAD_ARM, {}, None, log), "strat", "STRAT_READ.json"),
                                  ("r3", lambda: r3_engine(a, log), "r3", "R3_READ.json")):
        try:
            res = fn()
        except BaseException as ex:                                                             # noqa: BLE001
            res = {"ALL_PASS": False, "error": "%s: %s" % (type(ex).__name__, str(ex)[:400])}
        json.dump(B.conv(res), io.open(os.path.join(a.read_dir, folder, fname), "w", encoding="utf-8", newline="\n"), indent=1)
        status[nm] = {"ALL_PASS": res.get("ALL_PASS")}
    json.dump(B.conv(status), io.open(os.path.join(a.read_dir, "ENGINES_STATUS.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return status


# ============================================================================================ names
def judge(blocks, r3blk):
    """names of section 5 from the family readings and U of the pair blocks (pure function of the numbers)"""
    out = {}
    fr = {nm: blocks[nm]["families"] for nm in PAIRS}
    u = {nm: B.u_name(blocks[nm]["headway"]["U"]) for nm in PAIRS}
    worse_f = {nm: [f for f in FAMS if fr[nm][f] == "WORSE"] for nm in PAIRS}
    better_f = {nm: [f for f in FAMS if fr[nm][f] == "BETTER"] for nm in PAIRS}
    any_w = any(worse_f.values())
    any_b = any(better_f.values())
    any_sep = any(v != "NOT-SEPARATED" for nm in PAIRS for v in fr[nm].values())
    if not any_w and "HIGHER" not in u.values() and any_b:
        fresh = "BETTER"
    elif not any_w and not any_b and not any_sep and all(x == "NOT-SEPARATED" for x in u.values()):
        fresh = "EQUAL"
    elif any_w:
        fresh = "WORSE"
    else:
        fresh = "MIXED"
    out["FRESH"] = {"R1_R2_families_WORSE": worse_f, "R1_R2_families_BETTER": better_f, "U_R1_R2": u, "name": "%s-FRESH-%s" % (PRE, fresh)}
    f3 = r3blk["families"]
    w3 = [f for f in FAMS if f3[f] == "WORSE"]
    b3 = [f for f in FAMS if f3[f] == "BETTER"]
    if ("TACTICAL" in b3 or "LONGITUDINAL" in b3) and not w3:
        aged = "BETTER"
    elif all(f3[f] == "NOT-SEPARATED" for f in FAMS):
        aged = "EQUAL"
    elif w3 and not b3:
        aged = "WORSE"
    else:
        aged = "MIXED"
    out["AGED"] = {"R3_families": f3, "families_WORSE": w3, "families_BETTER": b3, "U_R3_beside": B.u_name(r3blk["headway"]["U"]), "name": "%s-AGED-%s" % (PRE, aged)}
    names = {"FRESH": out["FRESH"]["name"], "AGED": out["AGED"]["name"]}
    f_ok = fresh in ("BETTER", "EQUAL") and aged == "BETTER"
    cons = ("RH under K becomes the candidate of record in place of UE under K (specification section 1, model row)" if f_ok else
            ("the line candidate stays; the recipe question goes back to the other session with the failing cells" if fresh == "WORSE" else
             "not stated by the card for this combination; the lead writes it from the per-family cells"))
    out["consequence"] = {"card_sentence": CONSEQUENCE, "applied": cons, "fresh": fresh, "aged": aged}
    return out, names


def evaluate(a):
    seeds = tuple(a.seeds)
    qnew, qpart = B.load_queue(a.queue), B.load_queue(a.partner_queue)
    inputs = B.jl(a.inputs)
    grid_n = len(json.load(io.open(a.grid_p30, encoding="utf-8"))["order"]) if os.path.isfile(a.grid_p30) else a.expect_files
    hw = B.jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json"))
    q3 = B.jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    st = B.jl(os.path.join(a.read_dir, "strat", "STRAT_READ.json"))
    r3 = B.jl(os.path.join(a.read_dir, "r3", "R3_READ.json"))
    C = {"q1_complete": q1_complete(a, seeds, grid_n), "q2_models": q2_models(a, seeds, qnew, inputs), "q2_argv": q2_argv(a, seeds, qnew, qpart),
         "q3_partners": q3_partners(a, seeds), "q4_identity": q4_identity(a, seeds)}
    cnt = (hw or {}).get("counts", {})
    e = {"headway": bool(hw and (hw.get("controls") or {}).get("ALL_PASS")), "families": bool(q3 and (q3.get("controls") or {}).get("ALL_PASS")),
         "route_goal": bool(st and st.get("ALL_PASS") is True), "r3_grid_engines_and_tick_controls": bool(r3 and r3.get("ALL_PASS") is True),
         "headway_counts": {"n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "PASS": cnt.get("PASS")}}
    e["PASS"] = bool(e["headway"] and e["families"] and e["route_goal"] and e["r3_grid_engines_and_tick_controls"] and cnt.get("PASS") is True and cnt.get("n_windows_total") == 2399)
    C["e_reader_controls"] = e
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "seeds": list(seeds), "spread_arm": SPREAD_ARM, "spread_arm_r3": SPREAD_ARM_R3,
           "spread_arm_note": "STATED CHOICE: R1, R2 bar = spread of U_RLcK over the seeds; R3 bar = spread of U_RLcK_P30",
           "r3_strategic_waivers": "s2 and s3 of the route-goal reader are waived for R3 (both arms pre-rolled), recorded in r3.strategic.controls; s1, s4, s6, s9 and the episode map gate",
           "c2_reader_note": "row j of model seed 2 uses the reader TK16_s0 and the map QMAP_LR16_s0_q50 (card ADD140 section 2): it shares its reader with model seed 0",
           "controls": C, "controls_all_pass": bool(ok), "names": None, "checks": None, "pairs": None, "levels": None, "r3": None, "consequence": None}
    print(TIER_LINE)
    print("== controls (q1 to q4 and the reader controls)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(B.conv({q: w for q, w in v.items() if q not in ("per_run", "per_seed")}))[:400]))
    if ok:
        blocks = {nm: B.pair_block(hw, q3, st, x, y, nm) for nm, (x, y) in PAIRS.items()}
        pr = r3["pair"]
        r3blk = {"A": R3_PAIR[0], "B": R3_PAIR[1], "headway": B.ucf(pr["headway"]), "family_cells": B.cells_of(pr["family"]), "strategic": r3["strategic"]["pairs"]["R3"]}
        r3blk["families"] = {f: B.reading(r3blk["family_cells"], f) for f in NAMED}
        r3blk["families"]["STRATEGIC"] = r3blk["strategic"]["reading"]
        checks, names = judge(blocks, r3blk)
        lv = hw["levels"]
        res.update(names=names, checks=checks, pairs=dict(blocks, R3=r3blk), consequence=checks["consequence"],
                   r3={"windows": r3.get("windows"), "levels": r3.get("levels"), "arms": r3.get("arms")},
                   levels={arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in lv})
        print("== names")
        for k, v in names.items():
            print("   %-6s %s" % (k, v))
        print("== consequence: %s" % checks["consequence"]["applied"])
        for nm, b in list(blocks.items()) + [("R3", r3blk)]:
            print("== %s: %s minus %s" % (nm, b["A"], b["B"]))
            print("   U %s  C %s  F %s" % (b["headway"]["U"], b["headway"]["C"], b["headway"]["F"]))
            print("   families %s" % b["families"])
    else:
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(B.conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0 if ok else 2


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-engines", action="store_true")
    ap.add_argument("--expect-files", type=int, default=141)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--partner-queue", default=DEFAULT_PARTNER_QUEUE)
    ap.add_argument("--partners-sha", default=DEFAULT_SHA)
    ap.add_argument("--inputs", default=B.DEFAULT_INPUTS)
    ap.add_argument("--grid-p30", default=RH10 + "/pins/grid_p30.json")
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
