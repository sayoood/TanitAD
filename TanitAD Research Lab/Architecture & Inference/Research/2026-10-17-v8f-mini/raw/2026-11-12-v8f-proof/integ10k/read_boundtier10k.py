"""read_boundtier10k.py -- card PREREG_V8F_PROOF_ADD68 (BOUNDTIER10K): the read (numpy; the module strat_roll_read.py of card ADD66 in the same folder, unchanged, for the STRATEGIC family).
Written from read_headbound10k.py (card ADD67: runs, pairs, controls on the dumps and logs, named cells, names) with the runs, pairs, controls and names of this card. Controls e1, g1 to g9 first (g8 includes the
reader controls of the headway and family readers and controls s1 s2 s3 s4 s5 s6 s9 of card ADD66 on the runs read); the names, the levels and the differences only if every control passed (else the read
prints the controls and nothing else). No training. Never run on data by the tool builder.
Tier line: T1-kin (kinematic rollout on the model's own actions); never driving performance.
Runs (heading bound, 10 steps, both model seeds): I10R0C10Hb20 (no rule, causal readout, K 0.2), I10R0C10X (no rule, causal readout with the path of the executed steps, no bound), I10R0C10XHb20 (the same, K 0.2),
I10RLcHb20 (label-fed rule every 10, pinned, 0.2), I10RQ50Hb20 (reader-fed rule every 10, pinned, 0.2), I10R0Hb05, I10R0Hb10, I10R0Hb40 (doses, pinned, no rule).
Controls (model seed 0): I10R0C10HbID (causal, bound 1e9; equals banked I10R0C10), I10RLcHbID (pinned, label-fed rule, bound 1e9; equals banked I10RLc).
Reused, own: I10R0, I10R0C10, I10RLc, I10RQ50. Reused from card ADD67: I10R0Hb20, I10R0KHb20, I10RLcKF5Hb20, I10RQ50KF5Hb20, only if READ_HEADBOUND10K.json has controls_all_pass true; else the pairs
that need them are not named and the reason is recorded.
Pairs named (A minus B, name token): CAUSAL (I10R0C10Hb20, I10R0C10), CAUSALX (I10R0C10XHb20, I10R0C10X), LABEL-RULE (I10RLcHb20, I10RLc), READER-RULE (I10RQ50Hb20, I10RQ50), ACC (I10R0C10XHb20, I10R0KHb20),
READOUT-K (I10R0KHb20, I10R0Hb20), LABEL-CAND (I10RLcKF5Hb20, I10RLcHb20), READER-CAND (I10RQ50KF5Hb20, I10RQ50Hb20).
Beside, naming nothing: I10R0Hb05, I10R0Hb10, I10R0Hb40 each minus I10R0Hb20; I10R0C10X minus I10R0C10; I10R0C10XHb20 minus I10R0Hb20.
Names per pair P (card section 6): BOUNDTIER10K-<P>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED for LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC and BOUNDTIER10K-<P>-U-LOWER|HIGHER|NOT-SEPARATED.
Named cells: LONGITUDINAL speed_mae_mps and along_mae_m, LATERAL heading_mae_deg and yaw_rate_mae_degps, TACTICAL ade_tac_m, STRATEGIC pos at 30, 50, 80 m at the third tick (verdicts per arc of strat_roll_read.pair_read).
Reported without a bar: the share of executed frames at which the bound acted by speed bin (from the readout logs; runs with a bound), the heading offset from the logged heading at the third tick by speed bin for each run read.
--check-tables reads no data file: it prints the run table, the rollouts, the pairs and checks them for internal consistency (last line ZZ CHECK-TABLES ok: True|False).
Usage: python read_boundtier10k.py --read-dir D --pull PULL --reused BOUNDTIER10K_REUSED.json --taken REUSED_CAND_TAKEN.txt --wrapper W.py --test-wrapper W2.py --test-out-head H.txt --test-out-rule R.txt
       --test-out-gain G.txt --e1 ENV_E1.json --add67-json READ_HEADBOUND10K.json --pairs-read PAIRS --w W --tree TREE --out OUT.json"""
import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:                                                                          # --check-tables needs no numpy
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = (0, 1)
M_STEPS = 10
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
WRAPPER_SHA_PREFIX = "373ffe72f2655c35"                                                # sha256 of v8f_t1_eval_reader_rule_rzgfh.py, unchanged since card ADD67
CARD = "PREREG_V8F_PROOF_ADD68"
PRE = "BOUNDTIER10K"
# new run -> (rule form of the wrapper, rule every, readout label, kappa_max or None = no bound); readout labels: pinned | C (--readout causal) | CX (--readout causal_g --acc-gain 1 --path-exec)
RUNS = {"I10R0C10HbID": ("R0", 10, "C", 1e9), "I10RLcHbID": ("RLc", 10, "pinned", 1e9),
        "I10R0C10Hb20": ("R0", 10, "C", 0.2), "I10R0C10X": ("R0", 10, "CX", None), "I10R0C10XHb20": ("R0", 10, "CX", 0.2), "I10RLcHb20": ("RLc", 10, "pinned", 0.2),
        "I10RQ50Hb20": ("RR", 10, "pinned", 0.2), "I10R0Hb05": ("R0", 10, "pinned", 0.05), "I10R0Hb10": ("R0", 10, "pinned", 0.1), "I10R0Hb40": ("R0", 10, "pinned", 0.4)}
CTRL_RUNS = ("I10R0C10HbID", "I10RLcHbID")
NEW_ARMS = ("I10R0C10Hb20", "I10R0C10X", "I10R0C10XHb20", "I10RLcHb20", "I10RQ50Hb20", "I10R0Hb05", "I10R0Hb10", "I10R0Hb40")
BOUND_ARMS = tuple(a for a in NEW_ARMS if RUNS[a][3] is not None)
NEW_RUNS = [(arm, 0) for arm in CTRL_RUNS] + [(arm, k) for arm in NEW_ARMS for k in SEEDS]      # eighteen rollouts
REUSED_OWN = ("I10R0", "I10R0C10", "I10RLc", "I10RQ50")
REUSED_ADD67 = ("I10R0Hb20", "I10R0KHb20", "I10RLcKF5Hb20", "I10RQ50KF5Hb20")                # card ADD67: admitted only if its read has controls_all_pass true
G5_BANK = {"I10R0C10Hb20": "I10R0", "I10R0C10X": "I10R0", "I10R0C10XHb20": "I10R0", "I10RLcHb20": "I10RLc", "I10RQ50Hb20": "I10RQ50", "I10R0Hb05": "I10R0", "I10R0Hb10": "I10R0", "I10R0Hb40": "I10R0"}
N_REUSED = 22                                                                                # 8 result files + 6 rule logs + 8 tick files (BOUNDTIER10K_REUSED.json): see N_REUSED_NOTE
NAMED = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions); never driving performance"
TOL = 1e-6                                                                                   # readout log
TOL_EXC = 1e-9                                                                               # control g3
READER_FLOOR = 0.0196                                                                        # row 106: reader floor on U differences
V_BINS = (("below_2", 0.0, 2.0), ("2_to_8", 2.0, 8.0), ("above_8", 8.0, 1e9))
BIN_KEY = {"below_2": "lt2", "2_to_8": "2to8", "above_8": "gt8"}
# pair name -> (A, B, name token or None); the first eight are named
PAIRS_NAMED = {"CAUSAL": ("I10R0C10Hb20", "I10R0C10", "CAUSAL"), "CAUSALX": ("I10R0C10XHb20", "I10R0C10X", "CAUSALX"), "LABEL_RULE": ("I10RLcHb20", "I10RLc", "LABEL-RULE"),
               "READER_RULE": ("I10RQ50Hb20", "I10RQ50", "READER-RULE"), "ACC": ("I10R0C10XHb20", "I10R0KHb20", "ACC"), "READOUT_K": ("I10R0KHb20", "I10R0Hb20", "READOUT-K"),
               "LABEL_CAND": ("I10RLcKF5Hb20", "I10RLcHb20", "LABEL-CAND"), "READER_CAND": ("I10RQ50KF5Hb20", "I10RQ50Hb20", "READER-CAND")}
PAIRS_BESIDE = {"beside_dose_Hb05_minus_Hb20": ("I10R0Hb05", "I10R0Hb20", None), "beside_dose_Hb10_minus_Hb20": ("I10R0Hb10", "I10R0Hb20", None), "beside_dose_Hb40_minus_Hb20": ("I10R0Hb40", "I10R0Hb20", None),
                "beside_path_C10X_minus_C10": ("I10R0C10X", "I10R0C10", None), "beside_C10XHb20_minus_R0Hb20": ("I10R0C10XHb20", "I10R0Hb20", None)}


def needs_add67(A, B):
    return A in REUSED_ADD67 or B in REUSED_ADD67


def check_tables():
    """no data file is read: prints the run table, the rollouts, the pairs, the reused runs; verifies internal consistency"""
    known = set(RUNS) | set(REUSED_OWN) | set(REUSED_ADD67)
    print(TIER_LINE)
    print("== card %s (%s), wrapper sha256 prefix %s" % (CARD, PRE, WRAPPER_SHA_PREFIX))
    print("== runs (rule form, rule every, readout label, kappa_max):")
    for r, v in RUNS.items():
        print("   %-16s %s" % (r, v))
    print("== rollouts: %d (%d controls at model seed 0, %d arms at both model seeds)" % (len(NEW_RUNS), len(CTRL_RUNS), len(NEW_ARMS)))
    print("   " + " ".join("%s_s%d" % x for x in NEW_RUNS))
    print("== reused own: %s ; reused from card ADD67: %s ; reused files expected: %d" % (", ".join(REUSED_OWN), ", ".join(REUSED_ADD67), N_REUSED))
    print("== first-block bank (g5): " + ", ".join("%s->%s" % kv for kv in G5_BANK.items()))
    print("== pairs named (A minus B, token):")
    for nm, (A, B, t) in PAIRS_NAMED.items():
        print("   %-12s %-16s - %-16s %s%s" % (nm, A, B, t, "  [needs ADD67 admission]" if needs_add67(A, B) else ""))
    print("== pairs beside (name nothing):")
    for nm, (A, B, t) in PAIRS_BESIDE.items():
        print("   %-32s %-16s - %-16s%s" % (nm, A, B, "  [needs ADD67 admission]" if needs_add67(A, B) else ""))
    allp = list(PAIRS_NAMED.values()) + list(PAIRS_BESIDE.values())
    toks = [t for (_, _, t) in PAIRS_NAMED.values()]
    chk = {
        "every pair member is a new, reused or ADD67 run": all(A in known and B in known for A, B, _ in allp),
        "every new arm has a first-block bank": all(r in G5_BANK for r in NEW_ARMS) and all(G5_BANK[r] in REUSED_OWN for r in G5_BANK) and set(G5_BANK) == set(NEW_ARMS),
        "pair keys and name tokens unique": len(PAIRS_NAMED) == len(set(PAIRS_NAMED)) and len(toks) == len(set(toks)) and not (set(PAIRS_NAMED) & set(PAIRS_BESIDE)),
        "pairs unique": len({(A, B) for A, B, _ in allp}) == len(allp),
        "eighteen rollouts, unique": len(NEW_RUNS) == 18 and len(set(NEW_RUNS)) == 18,
        "controls are new runs at seed 0 with bound 1e9": all(c in RUNS and RUNS[c][3] == 1e9 for c in CTRL_RUNS),
        "readout labels known": all(v[2] in ("pinned", "C", "CX") for v in RUNS.values()),
        "no bound only for CX reference": [r for r in NEW_ARMS if RUNS[r][3] is None] == ["I10R0C10X"],
        "names unique": len({"%s-%s-%s-%s" % (PRE, t, f, w) for t in toks for f in FAMS for w in ("BETTER", "WORSE", "MIXED", "NOT-SEPARATED")}) == len(toks) * 16,
        "no run in two tables": not (set(RUNS) & set(REUSED_OWN)) and not (set(RUNS) & set(REUSED_ADD67)) and not (set(REUSED_OWN) & set(REUSED_ADD67)),
    }
    for k, v in chk.items():
        print("   check: %-60s %s" % (k, v))
    ok = all(chk.values())
    print("ZZ CHECK-TABLES ok: %s" % ok)
    return 0 if ok else 1


def jl(p):
    try:
        return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None
    except Exception:                                                                           # noqa: BLE001
        return None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else ""


def txt(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.exists(p) else ""


def npz(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def tick_dir(pull, arm, k):
    return os.path.join(pull, "%s_s%d" % (arm, k), "ticks")


def rlog(pull, arm, k, kind=""):
    """kind '' = tick log, 'mid' = later-look log, 'rd' = reader side file of the tick log"""
    n = "%s_s%d" % (arm, k)
    name = {"": "rule_log_%s.npz", "mid": "rule_log_mid_%s.npz", "rd": "rule_log_%s_rd.npz"}[kind] % n
    return npz(os.path.join(tick_dir(pull, arm, k), name))


def rdlog(pull, arm, k):
    return jl(os.path.join(tick_dir(pull, arm, k), "readout_log_%s_s%d.json" % (arm, k)))


def dump(pull, arm, k, keys=("cl", "g", "fl", "ws", "clip_index")):
    files = sorted(glob.glob(os.path.join(pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k), "ep*.npz")))
    if len(files) != 141:
        return None
    out = {n: [] for n in keys}
    for f in files:
        z = np.load(f)
        for n in keys:
            out[n].append(np.asarray(z[n], np.float64 if n in ("cl", "g", "fl") else np.int64))
    return {n: np.concatenate(v) for n, v in out.items()}


def maxdiff(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape or not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return float("inf")
    f = np.isfinite(x)
    return float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0


def cand_admitted(path):
    cj = jl(path)
    if cj is None:
        return False, "%s absent or unreadable" % os.path.basename(path or "")
    if cj.get("controls_all_pass") is True:
        return True, "READ_HEADBOUND10K.json controls_all_pass is true"
    return False, "READ_HEADBOUND10K.json: controls_all_pass is not true"


# ============================================================================================ controls on the dumps and logs
def controls(a, hw, q3, admitted, why):
    C, R = {}, {}
    e1 = jl(a.e1)
    C["e1_pod_read_reproduces_bank"] = {"file": os.path.basename(a.e1) if e1 else None, "PASS": bool(e1 and e1.get("PASS") is True)}
    for name, fn, banked in (("g1_causal_bound_1e9_equals_banked_R0C10", "CMP_R0C10HbID.txt", "I10R0C10"), ("g2_pinned_label_rule_bound_1e9_equals_banked_RLc", "CMP_RLcHbID.txt", "I10RLc")):
        m = re.search(r"CMP files=(\d+) differing=(\d+)", txt(os.path.join(a.pull, "small_bt", fn)))
        C[name] = {"line": m.group(0) if m else None, "banked": banked, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    cand_arms = REUSED_ADD67 if admitted else ()
    D, B_ = {}, {}
    for arm, k in NEW_RUNS:
        D[(arm, k)] = dump(a.pull, arm, k)
    for arm in REUSED_OWN:
        for k in SEEDS:
            B_[(arm, k)] = dump(a.pull, arm, k)
    # ---- g3: the bound holds and does nothing else (readout log of every run; parameters applied) and the readout log records the readout, the gain, the path and the bound of the table
    h3 = {}
    for arm, k in NEW_RUNS:
        form, every, ro, kap = RUNS[arm]
        key = "%s_s%d" % (arm, k)
        pr = txt(os.path.join(tick_dir(a.pull, arm, k), "run_%s.txt" % key))
        me, mk = re.search(r"rule every (\d+)", pr), re.search(r"kappa (\S+)", pr)
        rec = {"run_record_rule_every": int(me.group(1)) if me else None, "run_record_kappa": mk.group(1) if mk else None}
        if kap is None:
            ok_rec = bool(me and int(me.group(1)) == every and mk and mk.group(1) == "none")
        else:
            ok_rec = bool(me and int(me.group(1)) == every and mk and mk.group(1) != "none" and abs(float(mk.group(1)) - kap) <= 1e-12 * max(1.0, kap))
        RL = rdlog(a.pull, arm, k)
        if RL is None:
            h3[key] = {"PASS": None, "why": "readout log missing"}
            continue
        bins = RL.get("by_speed_bin") or {}
        bt, ba = sum(v.get("frames_total", 0) for v in bins.values()), sum(v.get("frames_bound_acted", 0) for v in bins.values())
        rec.update(readout=RL.get("readout"), acc_gain=RL.get("acc_gain"), path_exec=RL.get("path_exec"), form=RL.get("form"), replan_steps=RL.get("replan_steps"), calls=RL.get("calls"),
                   kappa_max=RL.get("kappa_max"), frames_total=RL.get("frames_total"), frames_bound_acted=RL.get("frames_bound_acted"), max_excess_rad=RL.get("max_excess_rad"),
                   max_abs_diff_where_not_acted=RL.get("max_abs_diff_where_not_acted"), max_abs_diff_xy=RL.get("max_abs_diff_xy"), max_abs_diff_v=RL.get("max_abs_diff_v"))
        dv = RL.get("max_abs_speed_minus_last_step")
        if ro == "pinned":
            ok_rd = bool(RL.get("readout") == "pinned" and RL.get("acc_gain") is None and RL.get("path_exec") is False)
        elif ro == "C":                                                                        # --readout causal: no gain, no path from the executed steps
            da = RL.get("max_abs_acceleration_minus_executed")
            ok_rd = bool(RL.get("readout") == "causal" and RL.get("acc_gain") is None and RL.get("path_exec") is False and dv is not None and da is not None and dv <= TOL and da <= TOL)
        else:                                                                                  # CX: --readout causal_g --acc-gain 1 --path-exec
            dga = RL.get("max_abs_acceleration_minus_gain_executed")
            ok_rd = bool(RL.get("readout") == "causal_g" and RL.get("acc_gain") == 1.0 and RL.get("path_exec") is True and dv is not None and dga is not None and dv <= TOL and dga <= TOL)
        if kap is None:                                                                        # no bound: the log records no bound; the bound checks are skipped
            ok_b = bool(RL.get("kappa_max") is None and RL.get("frames_total") is None and RL.get("form") == form and RL.get("replan_steps") == 10 and (RL.get("calls") or 0) > 0)
        else:
            ok_b = bool(RL.get("kappa_max") is not None and abs(float(RL["kappa_max"]) - kap) <= 1e-12 * max(1.0, kap) and (RL.get("frames_total") or 0) > 0 and RL.get("max_excess_rad") is not None
                        and RL["max_excess_rad"] <= TOL_EXC and RL.get("max_abs_diff_where_not_acted") == 0.0 and RL.get("max_abs_diff_xy") == 0.0 and RL.get("max_abs_diff_v") == 0.0
                        and bt == RL.get("frames_total") and ba == RL.get("frames_bound_acted") and RL.get("form") == form and RL.get("replan_steps") == 10 and (RL.get("calls") or 0) > 0)
        ok_rl = True
        if form != "R0":
            L, Lm = rlog(a.pull, arm, k), rlog(a.pull, arm, k, "mid")
            if L is None:
                h3[key] = {"PASS": None, "why": "rule log missing"}
                continue
            rec["rule_log_form"] = str(np.asarray(L["form"]))
            ok_rl = rec["rule_log_form"] == form
            if every < M_STEPS:
                rec["mid_log_rule_every"] = None if Lm is None else int(np.asarray(Lm["rule_every"]))
                ok_rl = ok_rl and Lm is not None and rec["mid_log_rule_every"] == every
            else:
                ok_rl = ok_rl and Lm is None
        rec["PASS"] = bool(ok_rec and ok_rd and ok_b and ok_rl)
        h3[key] = rec
    C["g3_the_bound_holds_and_does_nothing_else"] = {"per_run": h3, "PASS": (None if any(v["PASS"] is None for v in h3.values()) or len(h3) != len(NEW_RUNS) else bool(all(v["PASS"] for v in h3.values())))}
    # ---- g4: the three tool checks on the pod (planted faults caught), same wrapper everywhere
    t_head, t_rule, t_gain = txt(a.test_out_head), txt(a.test_out_rule), txt(a.test_out_gain)
    lh, lr, lg = t_head.strip().splitlines(), t_rule.strip().splitlines(), t_gain.strip().splitlines()
    head_ok = bool(lh and lh[-1].strip() == "ZZ CHECK-HEAD-BOUND all hold: True" and any(ln.startswith("7 planted faults caught") and ln.rstrip().endswith("caught: True") for ln in lh)
                   and len([ln for ln in lh if re.match(r"^[1-6] ", ln)]) == 6)
    k5 = [ln for ln in t_rule.splitlines() if ln.startswith("k5 ")]
    rule_ok = bool(lr and lr[-1].strip() == "ZZ CHECK-RULE-EVERY all hold: True" and len(k5) == 3 and all(" holds |" in ln for ln in k5))
    f3 = [ln for ln in t_gain.splitlines() if ln.startswith("3 planted fault")]
    gain_ok = bool(lg and lg[-1].strip() == "ZZ CHECK-ACC-GAIN all hold: True" and len(f3) == 1 and f3[0].count("caught: True") == 2)
    w1, w2 = sha(a.wrapper), sha(a.test_wrapper)
    p_head = (re.findall(r"wrapper rzgfh sha256 ([0-9a-f]{64})", t_head) or [""])[-1]
    p_rule = (re.findall(r"wrapper rzgfh sha256 ([0-9a-f]{64})", t_rule) or [""])[-1]
    p_gain = (re.findall(r"wrapper rzgfh sha256 ([0-9a-f]{64})", t_gain) or [""])[-1]
    C["g4_tool_checks"] = {"head_bound_check_last_line": lh[-1][:80] if lh else None, "rule_every_check_last_line": lr[-1][:80] if lr else None, "acc_gain_check_last_line": lg[-1][:80] if lg else None,
                           "wrapper_sha256": w1[:16], "test_wrapper_sha256": w2[:16], "printed_head": p_head[:16], "printed_rule": p_rule[:16], "printed_gain": p_gain[:16],
                           "PASS": bool(head_ok and rule_ok and gain_ok and w1 and w1 == w2 and w1 == p_head and w1 == p_rule and w1 == p_gain)}
    # ---- g5: the first block, known value: the first 10 executed positions equal those of a banked run of the same rule form and model seed
    h5 = {}
    for arm, bank in G5_BANK.items():
        for k in SEEDS:
            key = "%s_s%d" % (arm, k)
            d, bd = D.get((arm, k)), B_.get((bank, k))
            if d is None or bd is None:
                h5[key] = {"banked": bank, "PASS": None, "why": "a dump is missing"}
                continue
            same_w = bool(np.array_equal(d["clip_index"], bd["clip_index"]) and np.array_equal(d["ws"], bd["ws"]))
            dsteps = maxdiff(d["cl"][:, :M_STEPS], bd["cl"][:, :M_STEPS]) if same_w else float("inf")
            h5[key] = {"banked": bank, "max_abs_executed_steps_1_to_10": dsteps, "PASS": bool(same_w and dsteps == 0.0)}
    h5_cand_skipped = []
    h5_rest = list(h5.values())
    C["g5_first_block_equals_banked"] = {"per_run": h5, "PASS": (None if (not h5_rest or any(v["PASS"] is None for v in h5_rest)) else bool(all(v["PASS"] for v in h5_rest))), "skipped_not_admitted": h5_cand_skipped}
    # ---- g6: counts
    nfiles, ntick, nmid, bad = {}, {}, {}, []
    for arm, k in NEW_RUNS:
        form, every, _, _ = RUNS[arm]
        key = "%s_s%d" % (arm, k)
        dd = os.path.join(a.pull, key, "dump_%s_m10_causal_heads" % key)
        nfiles[key] = len([f for f in os.listdir(dd) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(dd) else None
        d = D[(arm, k)]
        nw = None if d is None else len(d["ws"])
        if nw != 2399:
            bad.append(key + ":windows")
        if form == "R0":
            continue
        L, Lm = rlog(a.pull, arm, k), rlog(a.pull, arm, k, "mid")
        ntick[key] = None if L is None else (int(len(L["ci"])), nw * 3 if nw else None)
        if ntick[key] is None or ntick[key][0] != ntick[key][1]:
            bad.append(key + ":tick")
        if every < M_STEPS:
            exp = None if not nw else nw * 3 * (M_STEPS // every - 1)
            nmid[key] = (None if Lm is None else int(len(Lm["q"])), exp)
            if Lm is None or exp is None or len(Lm["q"]) != exp:
                bad.append(key + ":mid")
    cnt = (hw or {}).get("counts", {})
    C["g6_counts"] = {"n_runs": len(nfiles), "not_141": [n for n, v in nfiles.items() if v != 141], "tick_rows_vs_expected": ntick, "later_look_rows_vs_expected": nmid, "mismatch": bad,
                      "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == len(NEW_RUNS) and all(v == 141 for v in nfiles.values()) and not bad and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642
                                   and cnt.get("PASS") is True)}
    # ---- g7: the reused files are the files hashed before the queue started (own reused runs) / taken by the queue after card ADD67 ended
    ru = jl(a.reused) or {"files": {}}
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    badr = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(root, p)) or sha(os.path.join(root, p)) != h]
    taken, absent, badt = {}, [], []
    for ln in txt(a.taken).splitlines():
        ln = ln.strip()
        if ln.startswith("ABSENT "):
            absent.append(ln[7:])
        elif re.match(r"^[0-9a-f]{64}  t1/", ln):
            h, rel = ln[:64], ln[66:]
            taken[rel] = h
            p = os.path.join(a.pull, rel[3:])
            if sha(p) != h:
                badt.append(rel)
    need = {"t1/%s_s%d/" % (r, k) for r in REUSED_ADD67 for k in SEEDS}
    have = {rel.rsplit("/", 2)[0] + "/" for rel in taken}
    cand_ok = (not admitted) or (not absent and need <= have)
    C["g7_reused_files"] = {"n_files": len(ru["files"]), "differing": badr, "taken_after_card_ADD67": {"n_lines": len(taken), "absent": absent, "differing": badt, "admitted": admitted},
                            "PASS": bool(len(ru["files"]) == N_REUSED and not badr and not badt and cand_ok)}
    C["g8_reader_controls_headway_family"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["g8_reader_controls_headway_family"]["PASS"] = bool(C["g8_reader_controls_headway_family"]["headway"] and C["g8_reader_controls_headway_family"]["family"])
    # ---- g9: checkpoints of card ADD46 and the wrapper the runs and the tool checks used
    sh, wsh = {}, {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        t = txt(os.path.join(tick_dir(a.pull, arm, k), "run_%s.txt" % key))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", t)
        sh[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
        mw = re.search(r"wrapper sha256 ([0-9a-f]{64})", t)
        wsh[key] = mw.group(1) if mw else None
    one = {w1, w2, p_head or None, p_rule or None, p_gain or None, *wsh.values()}
    C["g9_model_and_wrapper"] = {"n_runs": len(sh), "ckpt_not_equal": [n for n, v in sh.items() if not v], "wrapper_sha256": w1[:16], "card_prefix": WRAPPER_SHA_PREFIX,
                                 "one_sha_everywhere": bool(len(one) == 1 and None not in one and w1),
                                 "PASS": bool(len(sh) == len(NEW_RUNS) and all(sh.values()) and w1 and w1.startswith(WRAPPER_SHA_PREFIX) and len(one) == 1 and None not in one)}
    # ---- reported without a bar: the bound's frame shares by speed bin per run (both seeds added)
    fr = {}
    for arm in BOUND_ARMS + CTRL_RUNS:
        tot = {b: [0, 0] for b in BIN_KEY}
        allf = [0, 0]
        seeds = (0,) if arm in CTRL_RUNS else SEEDS
        for k in seeds:
            RL = rdlog(a.pull, arm, k)
            if RL is None:
                continue
            allf[0] += RL.get("frames_total") or 0
            allf[1] += RL.get("frames_bound_acted") or 0
            for b, kk in BIN_KEY.items():
                v = (RL.get("by_speed_bin") or {}).get(kk, {})
                tot[b][0] += v.get("frames_total", 0)
                tot[b][1] += v.get("frames_bound_acted", 0)
        fr[arm] = {"frames_total": allf[0], "frames_bound_acted": allf[1], "share_acted": (allf[1] / allf[0]) if allf[0] else None,
                   "by_speed_bin": {b: {"frames_total": t[0], "frames_bound_acted": t[1], "share_acted": (t[1] / t[0]) if t[0] else None} for b, t in tot.items()}}
    R["r1_share_of_executed_frames_where_the_bound_acted_by_speed_bin"] = fr
    return C, R


# ============================================================================================ STRATEGIC: the instrument of card ADD66 (strat_roll_read.py, unchanged)
def sname(arm):
    return arm[3:]                                                                            # I10R0Hb20 -> R0Hb20 (the module adds the prefix)


def strategic(a, admitted, why, log):
    """controls s1 s2 s3 s4 s6 s9 (run_controls_a) on the runs read plus s5; then the cells and the pair reads at the third tick. Returns (controls, cells-or-None, eid, pair reads, runs)."""
    import strat_roll_read as S
    sa = argparse.Namespace(w=a.w, tracks=S.DEF_TRACKS, join=S.DEF_JOIN, tree=a.tree)
    ctx = S.Ctx(sa)
    ctx.load_static()
    runs = [sname(r) for r in REUSED_OWN] + ([sname(r) for r in REUSED_ADD67] if admitted else []) + [sname(r) for r in NEW_ARMS]
    Cs, inputs, avail = S.run_controls_a(ctx, sa, runs, log)
    S.print_controls_a(Cs, log)
    missing = [k for k, v in avail.items() if not v]
    a_ok = bool(all(Cs[k]["pass"] for k in ("s1", "s2", "s4", "s6", "s9")) and Cs["s3"].get("pass") is True and not missing)
    out = {"controls": Cs, "inputs": inputs, "missing_tick_files": missing, "runs": runs, "stage_A_pass": a_ok}
    if not a_ok:
        return out, None, None, None, ctx, S
    cells, anc_ds, T0 = {}, {}, None
    for run in runs:
        for s in S.SEEDS:
            T = S.load_ticks(S.tick_path(ctx.W, run, s))
            if T0 is None:
                T0 = T
            c, st, ds = S.compute_cells(ctx, T)
            cells[(run, s)], anc_ds[(run, s)] = c, ds
            log("cells %s_s%d done" % (run, s))
    eid, n_map = S.episode_map(ctx.W, T0)
    Cs["s6"]["episode_map"] = {"dump_windows": n_map, "all_windows_mapped": bool((eid >= 0).all()), "n_episodes": int(len(np.unique(eid)))}
    Cs["s6"]["pass"] = bool(Cs["s6"]["pass"] and (eid >= 0).all() and len(np.unique(eid)) == S.N_EP)
    d5 = 0.0
    for run in runs:
        for s in S.SEEDS:
            for cell in S.CELLS:
                x, y = cells[(run, s)][cell][0], cells[("R0", s)][cell][0]
                if not (np.isnan(x) == np.isnan(y)).all():
                    d5 = float("inf")
                else:
                    ok = np.isfinite(x)
                    d5 = max(d5, float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0)
    Cs["s5"] = {"pass": bool(d5 == 0.0), "largest_difference": d5, "note": "first-tick cells of every run read equal those of R0 at the same model seed"}
    out["stage_B_pass"] = bool(Cs["s6"]["pass"] and Cs["s5"]["pass"])
    if not out["stage_B_pass"]:
        return out, None, eid, None, ctx, S
    return out, cells, eid, anc_ds, ctx, S


def heading_offsets(ctx, S, run, s):
    """heading offset at the third tick (executed s_pose heading minus logged heading at step j - age_s), degrees absolute, with the speed v0 of the tick row (as strat_roll_decomp.per_run)"""
    p = S.tick_path(ctx.W, run, s)
    T = S.load_ticks(p)
    z = np.load(p, allow_pickle=True)
    v0 = np.asarray(z["v0"], np.float64)
    hs, vs = [], []
    for row in np.nonzero(T["tick"] == 2)[0]:
        ci = int(T["ci"][row])
        G, _ = ctx.grid(str(T["clip_names"][ci]))
        step = int(T["j"][row]) - int(T["age_s"][row])
        if 0 <= step < int(ctx.J["n"][ci]) and step < len(G) and np.isfinite(G[step]).all():
            sp = T["s_pose"][row].astype(np.float64)
            hs.append(abs(float(S.wrap(sp[2] - G[step, 2]))) * 180.0 / math.pi)
            vs.append(float(v0[row]))
    return np.asarray(hs), np.asarray(vs)


def offsets_by_speed(ctx, S, run):
    hs, vs = [], []
    for s in S.SEEDS:
        h, v = heading_offsets(ctx, S, run, s)
        hs.append(h)
        vs.append(v)
    h, v = np.concatenate(hs), np.concatenate(vs)
    out = {}
    for nm, lo, hi in V_BINS:
        g = (v >= lo) & (v < hi)
        out[nm] = {"mean_abs_heading_offset_deg": float(h[g].mean()) if g.any() else None, "median_abs_heading_offset_deg": float(np.median(h[g])) if g.any() else None, "n": int(g.sum())}
    out["all"] = {"mean_abs_heading_offset_deg": float(h.mean()) if len(h) else None, "n": int(len(h))}
    return out


# ============================================================================================ formatting and names
def cells_of(q3, A, B):
    return {c: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for c, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}


def ucf(hw, A, B):
    out = {}
    for c in ("U", "C", "F"):
        v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
        out[c] = {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}
    return out


def family_of(vs):
    """the family rule of the card: BETTER if every named cell reads BETTER, WORSE if every one reads WORSE, NOT-SEPARATED if none reads either, else MIXED"""
    if all(v == "BETTER" for v in vs):
        return "BETTER"
    if all(v == "WORSE" for v in vs):
        return "WORSE"
    if not any(v in ("BETTER", "WORSE") for v in vs):
        return "NOT-SEPARATED"
    return "MIXED"


def reading(cells, fam):
    return family_of([cells[c]["verdict"] for c in NAMED[fam]])


def below(d):
    return bool(all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0)


def above(d):
    return bool(all(x > 0 for x in d["per_seed"]) and d["interval"][0] > 0)


def line(v):
    return "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))


def main(argv=None):
    if "--check-tables" in (sys.argv[1:] if argv is None else argv):
        return check_tables()
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--taken", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--test-wrapper", required=True)
    ap.add_argument("--test-out-head", required=True)
    ap.add_argument("--test-out-rule", required=True)
    ap.add_argument("--test-out-gain", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--add67-json", required=True)
    ap.add_argument("--pairs-read", default="")
    ap.add_argument("--w", default="/workspace/v8proof/integ10k")
    ap.add_argument("--tree", default="/workspace/v8proof/integ10k/tree")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def log(s):
        print(s, flush=True)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    adm0, why = cand_admitted(a.add67_json)
    read_pairs = set(p for p in a.pairs_read.split(",") if p)
    dep = [(A, B) for (A, B, _) in list(PAIRS_NAMED.values()) + list(PAIRS_BESIDE.values()) if needs_add67(A, B)]
    dep_in_read = all(("%s:%s" % (A, B)) in read_pairs for A, B in dep) if read_pairs else adm0
    admitted = bool(adm0 and dep_in_read and hw is not None and q3 is not None and all("%s_minus_%s" % (A, B) in hw["pairs"] for A, B in dep))
    if adm0 and not admitted:
        why = "the pairs that need the runs of card ADD67 are not in the headway / family reads"
    C, R = controls(a, hw, q3, admitted, why)
    SR = None
    try:
        SR, cells, eid, anc_ds, ctx, S = strategic(a, admitted, why, log)
        C["g8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"s1": SR["controls"]["s1"].get("pass"), "s2": SR["controls"]["s2"].get("pass"), "s3": SR["controls"]["s3"].get("pass"), "s4": SR["controls"]["s4"].get("pass"),
                                                          "s5": SR["controls"].get("s5", {}).get("pass"), "s6": SR["controls"]["s6"].get("pass"), "s9": SR["controls"]["s9"].get("pass"),
                                                          "missing_tick_files": SR["missing_tick_files"], "runs": SR["runs"],
                                                          "PASS": bool(SR["stage_A_pass"] and SR.get("stage_B_pass") is True)}
    except SystemExit as ex:
        SR, cells, eid, anc_ds, ctx, S = None, None, None, None, None, None
        C["g8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"PASS": False, "why": "SystemExit: %s" % ex}
    except Exception as ex:                                                                     # noqa: BLE001
        SR, cells, eid, anc_ds, ctx, S = None, None, None, None, None, None
        C["g8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
    C["g8_reader_controls_all"] = {"PASS": bool(C["g8_reader_controls_headway_family"]["PASS"] and C["g8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"]["PASS"])}
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "runs_of_card_ADD67": {"admitted": admitted, "reason": why if not admitted else "READ_HEADBOUND10K.json controls_all_pass is true"},
           "controls": C, "controls_all_pass": bool(ok), "names": None, "levels": None, "pairs": None, "cells": None, "families": None, "strategic": None, "reported_no_bar": R}
    print(TIER_LINE)
    print("== card ADD67 runs (I10R0Hb20, I10R0KHb20, I10RLcKF5Hb20, I10RQ50KF5Hb20) admitted: %s (%s)" % (admitted, res["runs_of_card_ADD67"]["reason"]))
    print("== controls (e1, g1 to g9)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:900]))
    if ok:
        lv = hw["levels"]
        arms_all = REUSED_OWN + (REUSED_ADD67 if admitted else ()) + tuple(NEW_ARMS)
        levels = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in arms_all if arm in lv}
        pairs_def = dict(PAIRS_NAMED)
        pairs_def.update(PAIRS_BESIDE)
        if not admitted:
            pairs_def = {k: v for k, v in pairs_def.items() if not needs_add67(v[0], v[1])}
        pairs, cells_q, fams, strat = {}, {}, {}, {}
        for nm, (A, B, _) in pairs_def.items():
            pairs[nm] = {"A": A, "B": B, "headway": ucf(hw, A, B)}
            if A.startswith("I10RQ50"):                                                          # READER U differences: the floor of row 106 (reported, names nothing)
                pairs[nm]["U_inside_reader_floor_row106"] = bool(abs(pairs[nm]["headway"]["U"]["pooled"]) <= READER_FLOOR)
            cells_q[nm] = cells_of(q3, A, B)
            fams[nm] = {f: reading(cells_q[nm], f) for f in NAMED}
            pr = S.pair_read(cells, sname(A), sname(B), eid, ticks=(2,))
            vs = [c["verdict"] for c in pr["tick2"]["pos"]]
            strat[nm] = {"pos_30_50_80m_third_tick": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pr["tick2"]["pos"])],
                         "reading": family_of(vs)}
            fams[nm]["STRATEGIC"] = strat[nm]["reading"]
        names, cons = {}, {}
        for nm, (A, B, tok) in PAIRS_NAMED.items():
            if nm not in pairs:
                continue
            u = pairs[nm]["headway"]["U"]
            names[nm + "_U"] = "%s-%s-U-%s" % (PRE, tok, "LOWER" if below(u) else ("HIGHER" if above(u) else "NOT-SEPARATED"))
            for f in FAMS:
                names["%s_%s" % (nm, f)] = "%s-%s-%s-%s" % (PRE, tok, f, fams[nm][f])
            better = [f for f in FAMS if fams[nm][f] == "BETTER"]
            worse = [f for f in FAMS if fams[nm][f] == "WORSE"]
            lst = []
            nsep = all(fams[nm][f] == "NOT-SEPARATED" for f in FAMS)
            mixed = [f for f in FAMS if fams[nm][f] == "MIXED"]
            if tok in ("CAUSAL", "CAUSALX", "LABEL-RULE", "READER-RULE"):
                if better and not worse and not above(u):
                    lst.append("BETTER on %s, WORSE on none, U not higher: the unbounded heading is a measured term of the T1-kin errors in this setting too, and the setting joins the proposal for the definition of the tier (decision D of card D16); adopting it is the decision of the PI and the other session" % better)
                if worse or above(u):
                    lst.append("family WORSE %s / U higher %s: reported as written with the table and the share of frames at which the bound acted; the proposal is stated without this setting" % (worse, above(u)))
                if not lst:
                    lst.append("no family BETTER and none WORSE (or BETTER with U higher is not reached): reported as written; this setting does not join the proposal")
            elif tok == "ACC":
                if worse and not better:
                    lst.append("ACC reads WORSE on %s and BETTER on none: the own-acceleration term of row 119 stands on the bounded executor; the training lever of the other session (decision C of card D16) stays the repair at the source and readout K stays a study device inside the candidate" % worse)
                elif nsep and not above(u):
                    lst.append("ACC reads NOT-SEPARATED on all four families and U not higher: on the bounded executor the planner takes the state a car would measure at no measured cost; the readout term of row 119 was carried by the unbounded heading; recommendation A of card D16 is then restated with the causal readout, and a candidate with it gets a card of its own before anything is proposed")
                else:
                    lst.append("ACC reads BETTER on %s / MIXED on %s / U higher %s: reported as written with the table" % (better, mixed, above(u)))
            elif tok == "READOUT-K":
                if better and not worse:
                    lst.append("READOUT-K BETTER on %s and WORSE on none: the gain of readout K holds on the bounded executor" % better)
                elif nsep:
                    lst.append("READOUT-K NOT-SEPARATED on all four families: the gain of readout K was the heading's")
                else:
                    lst.append("READOUT-K BETTER %s / WORSE %s / MIXED %s: reported as written with the table" % (better, worse, mixed))
            else:                                                                              # LABEL-CAND, READER-CAND
                if better and not worse and not above(u):
                    lst.append("%s BETTER on %s, WORSE on none, U not higher: the candidate's reading against the convention with the rule holds when both have the bound, and recommendation A of card D16 stands under decision D" % (tok, better))
                else:
                    lst.append("%s BETTER %s / WORSE %s / U higher %s: reported as written with the table; recommendation A of card D16 is restated for the bounded executor" % (tok, better, worse, above(u)))
            cons[tok] = lst
        for nm, dose_txt in (("beside_dose_Hb05_minus_Hb20", ("0.05", "the point where the bound starts to cut turns a car makes")), ("beside_dose_Hb10_minus_Hb20", ("0.1", "the point where the bound starts to cut turns a car makes")),
                             ("beside_dose_Hb40_minus_Hb20", ("0.4", "the point where the bound stops removing the offset"))):
            if nm not in fams:
                continue
            dw = [f for f in FAMS if fams[nm][f] == "WORSE"]
            cons["DOSE_%s_vs_0.2" % dose_txt[0]] = (["the dose at %s reads worse than 0.2 on %s: said as %s; no value is proposed from the doses by this card" % (dose_txt[0], dw, dose_txt[1])] if dw else
                                                    ["the dose at %s reads worse than 0.2 on no family: nothing said" % dose_txt[0]])
        res.update(names=names, levels=levels, pairs=pairs, cells=cells_q, families=fams, strategic=strat)
        R["consequences_fixed_by_card_section_6_per_pair"] = cons
        # the heading offset at the third tick by speed bin for each run and its pair partner (reported without a bar)
        runs_off = sorted({sname(x) for nm, (A, B, _) in pairs_def.items() for x in (A, B)})
        R["r2_heading_offset_third_tick_by_speed_bin_per_run"] = {"I10" + r: offsets_by_speed(ctx, S, r) for r in runs_off}
        print("== names")
        for k, v in names.items():
            print("   %-26s %s" % (k, v))
        print("== consequences fixed by section 6 of the card (per pair; the card itself changes nothing)")
        for tok, lst in cons.items():
            for t in lst:
                print("   %-14s %s" % (tok, t))
        print("== levels of the runs U C F (per model seed; 562 scored windows, 642 lead windows)")
        for arm, v in levels.items():
            print("   %-16s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== pairs: U C F (pooled, interval, per seed) and the family reading over the named cells (STRATEGIC: pos at 30, 50, 80 m at the third tick)")
        for nm, p in pairs.items():
            fl = "  [U inside the reader floor of row 106]" if p.get("U_inside_reader_floor_row106") else ""
            print("   %-32s %s minus %s | %s | %s%s" % (nm, p["A"], p["B"], line(p["headway"]), fams[nm], fl))
        print("== STRATEGIC pos (m) at the third tick: diff [lo, hi] verdict")
        for nm, s_ in strat.items():
            print("   %-32s " % nm + " ; ".join("%dm %+.4f [%+.4f,%+.4f] %s" % (int(b["arc_m"]), b["mean"], b["lo"], b["hi"], b["verdict"]) for b in s_["pos_30_50_80m_third_tick"]))
        print("== cells (all ten per pair; the named ones decide the readings)")
        for nm, cs in cells_q.items():
            for c, v in cs.items():
                print("   %-32s %-30s D %+8.4f [%+.4f, %+.4f] %s %s%s" % (nm, c, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"], " *" if any(c in NAMED[f] for f in NAMED) else ""))
        print("== reported without a bar")
        for k, v in R.items():
            print("   %s %s" % (k, json.dumps(v)[:3000]))
    else:
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    res["strategic_controls_detail"] = None if SR is None else {k: v for k, v in SR.items() if k != "controls"}
    res["strategic_controls_values"] = None if SR is None else SR["controls"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    try:
        import strat_roll_read as S2
        conv = S2.to_json
    except Exception:                                                                            # noqa: BLE001
        conv = lambda o: o                                                                       # noqa: E731
    json.dump(conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
