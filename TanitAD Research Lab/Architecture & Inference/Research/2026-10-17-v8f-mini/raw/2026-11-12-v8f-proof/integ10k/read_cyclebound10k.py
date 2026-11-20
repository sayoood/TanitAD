"""read_cyclebound10k.py -- card PREREG_V8F_PROOF_ADD69 (CYCLEBOUND10K): the read (numpy; the module strat_roll_read.py of card ADD66 in the same folder, unchanged, for the STRATEGIC family).
Written from read_boundtier10k.py (card ADD68) with the runs, pairs, controls and names of this card. Controls e1 and y1 to y9 first (y8 includes the reader controls of the headway and family
readers and controls s1 s2 s3 s4 s5 s6 s9 of card ADD66 on the runs read); the names, the levels and the differences only if every control passed (else the read prints the controls and
nothing else). No training. Never run on data by the tool builder.
Tier line: T1-kin (kinematic rollout on the model's own actions); never driving performance.
New runs (planning cycle 5 cache steps = 1.0 s, heading bound 0.2 1/m, both model seeds; the rule at every plan): I10R0M5Hb20 (no rule, pinned), I10R0KM5Hb20 (no rule, readout K),
I10RLcM5Hb20 (label-fed, pinned), I10RQ50M5Hb20 (reader-fed, pinned), I10RLcKM5Hb20 (label-fed, K), I10RQ50KM5Hb20 (reader-fed, K). Control (model seed 0): I10R0M5HbID (bound 1e9; equals banked I10R0M5).
Reused, own: I10R0, I10RLc, I10RQ50 (2.0 s), I10R0M5, I10RLcM5, I10RQ50M5 (1.0 s, no bound). Reused from card ADD67 (admitted if its read has controls_all_pass true): I10R0Hb20, I10R0KHb20,
I10RLcKF5Hb20, I10RQ50KF5Hb20 (and I10R0G0X for the known-value check y4 only). Reused from card ADD68 (admitted only if its read has controls_all_pass true): I10RLcHb20, I10RQ50Hb20.
Pairs named (1.0 s run minus 2.0 s run, both with the bound): CYC-PINNED, CYC-K, CYC-LABEL-RULE, CYC-READER-RULE, CYC-LABEL-CAND, CYC-READER-CAND. A pair whose reused run is not admitted is not named
and the reason is printed; the other pairs are read.
Beside, naming nothing: the bound at a 1.0 s cycle (three pairs) and the cycle without a bound re-read on the same reader (three pairs).
Names per pair P: CYCLEBOUND10K-<P>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED for LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC and CYCLEBOUND10K-<P>-U-LOWER|HIGHER|NOT-SEPARATED.
Named cells: LONGITUDINAL speed_mae_mps and along_mae_m, LATERAL heading_mae_deg and yaw_rate_mae_degps, TACTICAL ade_tac_m, STRATEGIC pos at 30, 50, 80 m at the tick t0+20.
STRATEGIC at a 1.0 s cycle: a 1.0 s run writes six ticks per window (j - t0 = 0, 5, ..., 25); before a tick file reaches the instrument only the rows with j - t0 in (0, 10, 20) are kept, in
the original order, for every run, in memory (the tick number is renumbered 0, 1, 2; the speed v0 of the row is kept with it). For a 2.0 s run this selects all rows (asserted, control y4).
Reported without a bar: the share of executed frames at which the bound acted by speed bin, the heading offset at the tick t0+20 by speed bin.
--check-tables reads no data file (last line ZZ CHECK-TABLES ok: True|False).
Usage: python read_cyclebound10k.py --read-dir D --pull PULL --reused CYCLEBOUND10K_REUSED.json --taken REUSED_CAND_TAKEN.txt --taken67 small_bt/REUSED_CAND_TAKEN.txt --wrapper W.py
       --test-wrapper W2.py --test-out-head H.txt --test-out-rule R.txt --test-out-gain G.txt --e1 ENV_E1.json --add67-json READ_HEADBOUND10K.json --add68-json READ_BOUNDTIER10K.json
       --pairs-read PAIRS --w W --tree TREE --out OUT.json"""
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
FIRST_BLOCK = 5                                                                              # executed positions of one plan block of a 1.0 s run (control y5)
WINDOW_STEPS = 30                                                                            # cache steps of a window: 3 ticks at a cycle of 10, 6 ticks at a cycle of 5
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
WRAPPER_SHA_PREFIX = "373ffe72f2655c35"                                                # sha256 of v8f_t1_eval_reader_rule_rzgfh.py, unchanged since card ADD67
CARD = "PREREG_V8F_PROOF_ADD69"
PRE = "CYCLEBOUND10K"
# new run -> (rule form of the wrapper, cycle steps, readout label, kappa_max); readout labels: pinned | K (--readout causal_g --acc-gain 0 --path-exec)
RUNS = {"I10R0M5HbID": ("R0", 5, "pinned", 1e9),
        "I10R0M5Hb20": ("R0", 5, "pinned", 0.2), "I10R0KM5Hb20": ("R0", 5, "K", 0.2), "I10RLcM5Hb20": ("RLc", 5, "pinned", 0.2), "I10RQ50M5Hb20": ("RR", 5, "pinned", 0.2),
        "I10RLcKM5Hb20": ("RLc", 5, "K", 0.2), "I10RQ50KM5Hb20": ("RR", 5, "K", 0.2)}
CTRL_RUNS = ("I10R0M5HbID",)
NEW_ARMS = ("I10R0M5Hb20", "I10R0KM5Hb20", "I10RLcM5Hb20", "I10RQ50M5Hb20", "I10RLcKM5Hb20", "I10RQ50KM5Hb20")
BOUND_ARMS = NEW_ARMS
NEW_RUNS = [(arm, 0) for arm in CTRL_RUNS] + [(arm, k) for arm in NEW_ARMS for k in SEEDS]      # thirteen rollouts
REUSED_OWN = ("I10R0", "I10RLc", "I10RQ50", "I10R0M5", "I10RLcM5", "I10RQ50M5")
REUSED_ADD67 = ("I10R0Hb20", "I10R0KHb20", "I10RLcKF5Hb20", "I10RQ50KF5Hb20")                # card ADD67: admitted only if its read has controls_all_pass true
REUSED_ADD68 = ("I10RLcHb20", "I10RQ50Hb20")                                                  # card ADD68: admitted only if its read has controls_all_pass true
KNOWN_VALUE_RUN = "I10R0G0X"                                                                 # y4 (iv) only: pair I10R0KHb20 minus I10R0G0X of card ADD67
CYCLE = {"I10R0": 10, "I10RLc": 10, "I10RQ50": 10, "I10R0M5": 5, "I10RLcM5": 5, "I10RQ50M5": 5, "I10R0Hb20": 10, "I10R0KHb20": 10, "I10RLcKF5Hb20": 10, "I10RQ50KF5Hb20": 10,
         "I10RLcHb20": 10, "I10RQ50Hb20": 10, KNOWN_VALUE_RUN: 10}
CYCLE.update({r: v[1] for r, v in RUNS.items()})
G5_BANK = {"I10R0M5Hb20": "I10R0M5", "I10R0KM5Hb20": "I10R0M5", "I10RLcM5Hb20": "I10RLcM5", "I10RLcKM5Hb20": "I10RLcM5", "I10RQ50M5Hb20": "I10RQ50M5", "I10RQ50KM5Hb20": "I10RQ50M5"}
N_REUSED = 36                                                                                # 12 result files + 12 tick files + 12 rule logs (CYCLEBOUND10K_REUSED.json)
NAMED = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions); never driving performance"
TOL = 1e-6                                                                                   # readout log
TOL_EXC = 1e-9                                                                               # control y2
TOL_KNOWN = 1e-9                                                                             # control y4 (iv)
READER_FLOOR = 0.0196                                                                        # row 106: reader floor on U differences
V_BINS = (("below_2", 0.0, 2.0), ("2_to_8", 2.0, 8.0), ("above_8", 8.0, 1e9))
BIN_KEY = {"below_2": "lt2", "2_to_8": "2to8", "above_8": "gt8"}
SEL_DJ = (0, 10, 20)                                                                         # j - t0 of the rows the strategic instrument reads
N_ROWS_SEL, N_WIN_ = 7197, 2399
# pair name -> (A, B, name token or None); the first six are named
PAIRS_NAMED = {"PINNED": ("I10R0M5Hb20", "I10R0Hb20", "CYC-PINNED"), "K": ("I10R0KM5Hb20", "I10R0KHb20", "CYC-K"), "LABEL_RULE": ("I10RLcM5Hb20", "I10RLcHb20", "CYC-LABEL-RULE"),
               "READER_RULE": ("I10RQ50M5Hb20", "I10RQ50Hb20", "CYC-READER-RULE"), "LABEL_CAND": ("I10RLcKM5Hb20", "I10RLcKF5Hb20", "CYC-LABEL-CAND"),
               "READER_CAND": ("I10RQ50KM5Hb20", "I10RQ50KF5Hb20", "CYC-READER-CAND")}
PAIRS_BESIDE = {"beside_bound_R0M5": ("I10R0M5Hb20", "I10R0M5", None), "beside_bound_RLcM5": ("I10RLcM5Hb20", "I10RLcM5", None), "beside_bound_RQ50M5": ("I10RQ50M5Hb20", "I10RQ50M5", None),
                "beside_cycle_R0": ("I10R0M5", "I10R0", None), "beside_cycle_RLc": ("I10RLcM5", "I10RLc", None), "beside_cycle_RQ50": ("I10RQ50M5", "I10RQ50", None)}
SECTION6 = {"worse": "WORSE on at least one family or U higher, and BETTER on no family: the 1.0 s cycle costs on the bounded executor too in that setting; coordinator decision D12 (2.0 s for every "
                     "capability number) stands there, now measured on the tier the PI approved",
            "better": "BETTER on at least one family, WORSE on none and U not higher: on the bounded executor the shorter cycle is a gain in that setting, and the cost of rows 114 to 116 was the heading's; "
                      "the 1.0 s cycle then goes to the PI and the other session as a proposal for that setting; adopting it changes the evaluation convention and doubles the planner calls (the latency "
                      "budget on the target hardware is theirs to weigh); nothing is adopted by this card",
            "none": "all four families NOT-SEPARATED and U not separated: the cycle does not move the reads on the bounded executor in that setting; 2.0 s stays (half the planner calls)",
            "else": "anything else (MIXED, or better on one family with U higher): reported as written with the table; nothing is proposed for that setting"}


def needs67(A, B):
    return A in REUSED_ADD67 or B in REUSED_ADD67


def needs68(A, B):
    return A in REUSED_ADD68 or B in REUSED_ADD68


def check_tables():
    """no data file is read: prints the run table, the rollouts, the pairs, the reused runs; verifies internal consistency"""
    known = set(RUNS) | set(REUSED_OWN) | set(REUSED_ADD67) | set(REUSED_ADD68)
    print(TIER_LINE)
    print("== card %s (%s), wrapper sha256 prefix %s" % (CARD, PRE, WRAPPER_SHA_PREFIX))
    print("== runs (rule form, cycle steps, readout label, kappa_max):")
    for r, v in RUNS.items():
        print("   %-16s %s" % (r, v))
    print("== rollouts: %d (%d control at model seed 0, %d arms at both model seeds)" % (len(NEW_RUNS), len(CTRL_RUNS), len(NEW_ARMS)))
    print("   " + " ".join("%s_s%d" % x for x in NEW_RUNS))
    print("== reused own: %s ; ADD67: %s ; ADD68: %s ; known-value run (y4): %s ; reused files expected: %d" % (", ".join(REUSED_OWN), ", ".join(REUSED_ADD67), ", ".join(REUSED_ADD68), KNOWN_VALUE_RUN, N_REUSED))
    print("== cycle steps of every run read: " + ", ".join("%s %d" % kv for kv in CYCLE.items()))
    print("== first-block bank (y5, first %d executed positions): " % FIRST_BLOCK + ", ".join("%s->%s" % kv for kv in G5_BANK.items()))
    print("== pairs named (A minus B, token):")
    for nm, (A, B, t) in PAIRS_NAMED.items():
        print("   %-12s %-16s - %-16s %s%s%s" % (nm, A, B, t, "  [needs ADD67 admission]" if needs67(A, B) else "", "  [needs ADD68 admission]" if needs68(A, B) else ""))
    print("== pairs beside (name nothing):")
    for nm, (A, B, t) in PAIRS_BESIDE.items():
        print("   %-24s %-16s - %-16s" % (nm, A, B))
    allp = list(PAIRS_NAMED.values()) + list(PAIRS_BESIDE.values())
    toks = [t for (_, _, t) in PAIRS_NAMED.values()]
    chk = {
        "every pair member is a new or reused run": all(A in known and B in known for A, B, _ in allp),
        "every new arm has a first-block bank": set(G5_BANK) == set(NEW_ARMS) and all(G5_BANK[r] in REUSED_OWN for r in G5_BANK),
        "pair keys and name tokens unique": len(PAIRS_NAMED) == len(set(PAIRS_NAMED)) and len(toks) == len(set(toks)) and not (set(PAIRS_NAMED) & set(PAIRS_BESIDE)),
        "pairs unique": len({(A, B) for A, B, _ in allp}) == len(allp),
        "thirteen rollouts, unique": len(NEW_RUNS) == 13 and len(set(NEW_RUNS)) == 13,
        "control is a new run at seed 0 with bound 1e9": all(c in RUNS and RUNS[c][3] == 1e9 for c in CTRL_RUNS),
        "readout labels known": all(v[2] in ("pinned", "K") for v in RUNS.values()),
        "every new run has cycle 5 and a bound": all(v[1] == 5 and v[3] is not None for v in RUNS.values()),
        "every named pair is a 1.0 s run minus a 2.0 s run": all(CYCLE[A] == 5 and CYCLE[B] == 10 for A, B, _ in PAIRS_NAMED.values()),
        "names unique": len({"%s-%s-%s-%s" % (PRE, t, f, w) for t in toks for f in FAMS for w in ("BETTER", "WORSE", "MIXED", "NOT-SEPARATED")}) == len(toks) * 16,
        "no run in two tables": len(set(RUNS) | set(REUSED_OWN) | set(REUSED_ADD67) | set(REUSED_ADD68)) == len(RUNS) + len(REUSED_OWN) + len(REUSED_ADD67) + len(REUSED_ADD68),
        "every run read has a cycle": all(r in CYCLE for r in known),
        "known-value run not among the runs read": KNOWN_VALUE_RUN not in known,
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


def admitted_json(path, label):
    cj = jl(path)
    if cj is None:
        return False, "%s absent or unreadable" % os.path.basename(path or label)
    if cj.get("controls_all_pass") is True:
        return True, "%s controls_all_pass is true" % os.path.basename(path)
    return False, "%s: controls_all_pass is not true" % os.path.basename(path)


def select_ticks(z, name):
    """The selection of rows for the strategic instrument: only the rows with j - t0 in (0, 10, 20), in the original order; tick renumbered j - t0 // 10 (0, 1, 2); every row-aligned field,
    and the speed v0 of the row, selected with the same mask. Returns (T, stat). The stat records the control y4 facts of this file (rows before and after, the steps of each window before,
    the ages of the kept rows)."""
    import strat_roll_read as S
    T = {k: z[k] for k in S.FIELDS}
    v0 = np.asarray(z["v0"], np.float64)
    n = len(T["tick"])
    j, t0 = np.asarray(T["j"], np.int64), np.asarray(T["t0"], np.int64)
    dj = j - t0
    keep = np.isin(dj, SEL_DJ)
    win = np.asarray(T["win"], np.int64)
    # the steps of each window before the selection
    o = np.lexsort((dj, win))
    uw, cnt = np.unique(win[o], return_counts=True)
    bnd = np.r_[0, np.cumsum(cnt)]
    djo = dj[o]
    steps_before = sorted({tuple(djo[bnd[i]:bnd[i + 1]].tolist()) for i in range(len(uw))})
    old_tick = np.asarray(T["tick"], np.int64)
    new_tick = (dj[keep] // 10).astype(old_tick.dtype)
    age = np.asarray(T["age_s"], np.int64)[keep]
    age_by_dj = {int(d): sorted(set(age[dj[keep] == d].tolist())) for d in SEL_DJ}
    for k in list(T):
        if k != "clip_names":
            T[k] = T[k][keep]
    T["v0"] = v0[keep]
    T["tick"] = new_tick
    stat = {"rows_before": int(n), "rows_after": int(keep.sum()), "windows": int(len(uw)), "steps_per_window_before": [list(s) for s in steps_before], "ages_after_by_j_minus_t0": age_by_dj,
            "all_rows_kept": bool(keep.all()), "tick_unchanged_when_all_kept": bool(keep.all() and np.array_equal(old_tick, new_tick))}
    return T, stat


def y4_eval(name, stat, cycle):
    """control y4 (i) to (iii) for one tick file"""
    exp_before = N_ROWS_SEL * (10 // cycle)
    want = [list(range(0, WINDOW_STEPS, cycle))]
    ok = bool(stat["rows_before"] == exp_before and stat["rows_after"] == N_ROWS_SEL and stat["steps_per_window_before"] == want and stat["windows"] == N_WIN_)
    if cycle == 10:
        ok = ok and stat["all_rows_kept"] and stat["tick_unchanged_when_all_kept"]
    ages = stat["ages_after_by_j_minus_t0"]
    ok = ok and ages.get(10) == [4] and ages.get(20) == [2] and ages.get(0) == [0]
    return ok


# ============================================================================================ controls on the dumps and logs
def controls(a, hw, q3, adm67, adm68):
    C, R = {}, {}
    e1 = jl(a.e1)
    C["e1_pod_read_reproduces_bank"] = {"file": os.path.basename(a.e1) if e1 else None, "PASS": bool(e1 and e1.get("PASS") is True)}
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt(os.path.join(a.pull, "small_cb", "CMP_R0M5HbID.txt")))
    C["y1_bound_1e9_at_cycle_5_equals_banked_R0M5"] = {"line": m.group(0) if m else None, "banked": "I10R0M5", "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    D, B_ = {}, {}
    for arm, k in NEW_RUNS:
        D[(arm, k)] = dump(a.pull, arm, k)
    for arm in REUSED_OWN:
        for k in SEEDS:
            B_[(arm, k)] = dump(a.pull, arm, k)
    # ---- y2: the bound holds and does nothing else; the readout log records the readout, the gain, the path and the bound of the table; the run record says replan steps 5
    h3 = {}
    for arm, k in NEW_RUNS:
        form, cyc, ro, kap = RUNS[arm]
        key = "%s_s%d" % (arm, k)
        pr = txt(os.path.join(tick_dir(a.pull, arm, k), "run_%s.txt" % key))
        mr, mk = re.search(r"replan steps (\d+)", pr), re.search(r"kappa (\S+)", pr)
        rec = {"run_record_replan_steps": int(mr.group(1)) if mr else None, "run_record_kappa": mk.group(1) if mk else None}
        ok_rec = bool(mr and int(mr.group(1)) == cyc and mk and mk.group(1) != "none" and abs(float(mk.group(1)) - kap) <= 1e-12 * max(1.0, kap))
        RL = rdlog(a.pull, arm, k)
        if RL is None:
            h3[key] = {"PASS": None, "why": "readout log missing"}
            continue
        bins = RL.get("by_speed_bin") or {}
        bt, ba = sum(v.get("frames_total", 0) for v in bins.values()), sum(v.get("frames_bound_acted", 0) for v in bins.values())
        rec.update(readout=RL.get("readout"), acc_gain=RL.get("acc_gain"), path_exec=RL.get("path_exec"), form=RL.get("form"), replan_steps=RL.get("replan_steps"), calls=RL.get("calls"),
                   kappa_max=RL.get("kappa_max"), frames_total=RL.get("frames_total"), frames_bound_acted=RL.get("frames_bound_acted"), max_excess_rad=RL.get("max_excess_rad"),
                   max_abs_diff_where_not_acted=RL.get("max_abs_diff_where_not_acted"), max_abs_diff_xy=RL.get("max_abs_diff_xy"), max_abs_diff_v=RL.get("max_abs_diff_v"))
        if ro == "pinned":
            ok_rd = bool(RL.get("readout") == "pinned" and RL.get("acc_gain") is None and RL.get("path_exec") is False)
        else:                                                                                  # K: --readout causal_g --acc-gain 0 --path-exec
            dv, dga = RL.get("max_abs_speed_minus_last_step"), RL.get("max_abs_acceleration_minus_gain_executed")
            ok_rd = bool(RL.get("readout") == "causal_g" and RL.get("acc_gain") == 0.0 and RL.get("path_exec") is True and dv is not None and dga is not None and dv <= TOL and dga <= TOL)
        ok_b = bool(RL.get("kappa_max") is not None and abs(float(RL["kappa_max"]) - kap) <= 1e-12 * max(1.0, kap) and (RL.get("frames_total") or 0) > 0 and RL.get("max_excess_rad") is not None
                    and RL["max_excess_rad"] <= TOL_EXC and RL.get("max_abs_diff_where_not_acted") == 0.0 and RL.get("max_abs_diff_xy") == 0.0 and RL.get("max_abs_diff_v") == 0.0
                    and bt == RL.get("frames_total") and ba == RL.get("frames_bound_acted") and RL.get("form") == form and RL.get("replan_steps") == cyc and (RL.get("calls") or 0) > 0)
        ok_rl = True
        if form != "R0":
            L, Lm = rlog(a.pull, arm, k), rlog(a.pull, arm, k, "mid")
            if L is None:
                h3[key] = {"PASS": None, "why": "rule log missing"}
                continue
            rec["rule_log_form"] = str(np.asarray(L["form"]))
            rec["rule_log_replan_steps"] = int(np.asarray(L["replan_steps"])) if "replan_steps" in L.files else None
            rec["later_look_log_present"] = Lm is not None
            ok_rl = rec["rule_log_form"] == form and Lm is None and rec["rule_log_replan_steps"] in (None, cyc)     # the rule acts at every plan: no later looks
        rec["PASS"] = bool(ok_rec and ok_rd and ok_b and ok_rl)
        h3[key] = rec
    C["y2_the_bound_holds_and_does_nothing_else"] = {"per_run": h3, "PASS": (None if any(v["PASS"] is None for v in h3.values()) or len(h3) != len(NEW_RUNS) else bool(all(v["PASS"] for v in h3.values())))}
    # ---- y3: the three tool checks on the pod (planted faults caught), same wrapper everywhere
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
    C["y3_tool_checks"] = {"head_bound_check_last_line": lh[-1][:80] if lh else None, "rule_every_check_last_line": lr[-1][:80] if lr else None, "acc_gain_check_last_line": lg[-1][:80] if lg else None,
                           "wrapper_sha256": w1[:16], "test_wrapper_sha256": w2[:16], "printed_head": p_head[:16], "printed_rule": p_rule[:16], "printed_gain": p_gain[:16],
                           "PASS": bool(head_ok and rule_ok and gain_ok and w1 and w1 == w2 and w1 == p_head and w1 == p_rule and w1 == p_gain)}
    # ---- y5: the first block, known value: the first 5 executed positions equal those of the banked 1.0 s run of the same rule form and model seed
    h5 = {}
    for arm, bank in G5_BANK.items():
        for k in SEEDS:
            key = "%s_s%d" % (arm, k)
            d, bd = D.get((arm, k)), B_.get((bank, k))
            if d is None or bd is None:
                h5[key] = {"banked": bank, "PASS": None, "why": "a dump is missing"}
                continue
            same_w = bool(np.array_equal(d["clip_index"], bd["clip_index"]) and np.array_equal(d["ws"], bd["ws"]))
            dsteps = maxdiff(d["cl"][:, :FIRST_BLOCK], bd["cl"][:, :FIRST_BLOCK]) if same_w else float("inf")
            h5[key] = {"banked": bank, "max_abs_executed_steps_1_to_%d" % FIRST_BLOCK: dsteps, "PASS": bool(same_w and dsteps == 0.0)}
    h5_rest = list(h5.values())
    C["y5_first_block_equals_banked"] = {"per_run": h5, "PASS": (None if (not h5_rest or any(v["PASS"] is None for v in h5_rest)) else bool(all(v["PASS"] for v in h5_rest)))}
    # ---- y6: counts
    nfiles, ntick, bad = {}, {}, []
    for arm, k in NEW_RUNS:
        form, cyc, _, _ = RUNS[arm]
        key = "%s_s%d" % (arm, k)
        dd = os.path.join(a.pull, key, "dump_%s_m10_causal_heads" % key)
        nfiles[key] = len([f for f in os.listdir(dd) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(dd) else None
        d = D[(arm, k)]
        nw = None if d is None else len(d["ws"])
        if nw != 2399:
            bad.append(key + ":windows")
        if form == "R0":
            continue
        L = rlog(a.pull, arm, k)
        ntick[key] = None if L is None else (int(len(L["ci"])), nw * (WINDOW_STEPS // cyc) if nw else None)
        if ntick[key] is None or ntick[key][0] != ntick[key][1]:
            bad.append(key + ":tick")
    cnt = (hw or {}).get("counts", {})
    C["y6_counts"] = {"n_runs": len(nfiles), "not_141": [n for n, v in nfiles.items() if v != 141], "tick_rows_vs_expected": ntick, "mismatch": bad,
                      "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == len(NEW_RUNS) and all(v == 141 for v in nfiles.values()) and not bad and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642
                                   and cnt.get("PASS") is True)}
    # ---- y7: the reused files are the files hashed before the queue started (own reused runs) / taken by the queue of card ADD68 (runs of ADD67) / taken by the queue of this card (runs of ADD68)
    ru = jl(a.reused) or {"files": {}}
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    badr = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(root, p)) or sha(os.path.join(root, p)) != h]

    def taken_check(path, runs, needed):
        taken, absent, badt = {}, [], []
        for ln in txt(path).splitlines():
            ln = ln.strip()
            if ln.startswith("ABSENT "):
                absent.append(ln[7:])
            elif re.match(r"^[0-9a-f]{64}  t1/", ln):
                h, rel = ln[:64], ln[66:]
                taken[rel] = h
                if sha(os.path.join(a.pull, rel[3:])) != h:
                    badt.append(rel)
        have = {"t1/" + rel.split("/")[1] + "/" for rel in taken}
        need = {"t1/%s_s%d/" % (r, k) for r in runs for k in SEEDS}
        ok = (not needed) or (bool(taken) and not absent and need <= have and not badt)
        return {"n_lines": len(taken), "absent": absent, "differing": badt, "needed": needed, "PASS": bool(ok)}

    t67 = taken_check(a.taken67, REUSED_ADD67, adm67)
    t68 = taken_check(a.taken, REUSED_ADD68, adm68)
    C["y7_reused_files"] = {"n_files": len(ru["files"]), "differing": badr, "taken_card_ADD67_runs": t67, "taken_card_ADD68_runs": t68,
                            "PASS": bool(len(ru["files"]) == N_REUSED and not badr and t67["PASS"] and t68["PASS"])}
    C["y8_reader_controls_headway_family"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["y8_reader_controls_headway_family"]["PASS"] = bool(C["y8_reader_controls_headway_family"]["headway"] and C["y8_reader_controls_headway_family"]["family"])
    # ---- y9: checkpoints of card ADD46 and the wrapper the runs and the tool checks used
    sh, wsh = {}, {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        t = txt(os.path.join(tick_dir(a.pull, arm, k), "run_%s.txt" % key))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", t)
        sh[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
        mw = re.search(r"wrapper sha256 ([0-9a-f]{64})", t)
        wsh[key] = mw.group(1) if mw else None
    one = {w1, w2, p_head or None, p_rule or None, p_gain or None, *wsh.values()}
    C["y9_model_and_wrapper"] = {"n_runs": len(sh), "ckpt_not_equal": [n for n, v in sh.items() if not v], "wrapper_sha256": w1[:16], "card_prefix": WRAPPER_SHA_PREFIX,
                                 "one_sha_everywhere": bool(len(one) == 1 and None not in one and w1),
                                 "PASS": bool(len(sh) == len(NEW_RUNS) and all(sh.values()) and w1 and w1.startswith(WRAPPER_SHA_PREFIX) and len(one) == 1 and None not in one)}
    # ---- reported without a bar: the bound's frame shares by speed bin per run (both seeds added)
    fr = {}
    for arm in BOUND_ARMS + CTRL_RUNS:
        tot = {b: [0, 0] for b in BIN_KEY}
        allf = [0, 0]
        for k in ((0,) if arm in CTRL_RUNS else SEEDS):
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


# ============================================================================================ STRATEGIC: the instrument of card ADD66 (strat_roll_read.py, unchanged) on the selected rows
def sname(arm):
    return arm[3:]                                                                            # I10R0Hb20 -> R0Hb20 (the module adds the prefix)


def strategic(a, runs_read, log):
    """controls s1 s2 s3 s4 s6 s9 (run_controls_a) on the runs read plus s5 and y4; then the cells. The loader of the module is replaced, in memory, by the selection of the rows
    j - t0 in (0, 10, 20). Returns (out, cells-or-None, eid, anc_ds, ctx, S)."""
    import strat_roll_read as S
    STATS = {}

    def load_sel(path):
        z = np.load(path, allow_pickle=True)
        T, st = select_ticks(z, path)
        STATS[os.path.basename(os.path.dirname(os.path.dirname(path)))] = st
        return T
    S.load_ticks = load_sel
    sa = argparse.Namespace(w=a.w, tracks=S.DEF_TRACKS, join=S.DEF_JOIN, tree=a.tree)
    ctx = S.Ctx(sa)
    ctx.load_static()
    runs = [sname(r) for r in runs_read]
    Cs, inputs, avail = S.run_controls_a(ctx, sa, runs, log)
    S.print_controls_a(Cs, log)
    missing = [k for k, v in avail.items() if not v]
    a_ok = bool(all(Cs[k]["pass"] for k in ("s1", "s2", "s4", "s6", "s9")) and Cs["s3"].get("pass") is True and not missing)
    out = {"controls": Cs, "inputs": inputs, "missing_tick_files": missing, "runs": runs, "stage_A_pass": a_ok, "load_stats": STATS}
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
    # ---- y4: (i) to (iii) on every tick file read, (iv) the known value of card ADD67
    y4 = {"per_file": {}, "known_value": {}}
    ok_files = True
    extras = [r for r in (sname("I10R0KHb20"), sname(KNOWN_VALUE_RUN)) if r not in runs]       # the two runs of the known value: cells now if the controls did not read them
    for run in extras:
        for s in S.SEEDS:
            c, st, ds = S.compute_cells(ctx, S.load_ticks(S.tick_path(ctx.W, run, s)))
            cells[(run, s)] = c
            log("cells %s_s%d done (known value y4)" % (run, s))
    for run in runs + extras:
        for s in S.SEEDS:
            nm = "I10%s_s%d" % (run, s)
            if nm not in STATS:
                y4["per_file"][nm] = {"PASS": False, "why": "tick file not loaded"}
                ok_files = False
                continue
            cyc = CYCLE["I10" + run]
            good = y4_eval(nm, STATS[nm], cyc)
            y4["per_file"][nm] = {"cycle_steps": cyc, "rows_before": STATS[nm]["rows_before"], "rows_after": STATS[nm]["rows_after"], "steps_per_window_before": STATS[nm]["steps_per_window_before"],
                                  "ages": STATS[nm]["ages_after_by_j_minus_t0"], "PASS": good}
            ok_files &= good
    kv = {"PASS": False, "why": "not computed"}
    stored = None
    try:
        j67 = jl(a.add67_json)
        stored = j67["strategic"]["HK"]["pos_30_50_80m_third_tick"] if j67 else None
    except Exception as ex:                                                                   # noqa: BLE001
        kv = {"PASS": False, "why": "stored value unreadable: %s" % type(ex).__name__}
    if stored is not None and out["stage_B_pass"]:
        pr = S.pair_read(cells, sname("I10R0KHb20"), sname(KNOWN_VALUE_RUN), eid, ticks=(2,))
        worst = 0.0
        for b, st_ in zip(pr["tick2"]["pos"], stored):
            for f in ("mean", "lo", "hi"):
                worst = max(worst, abs(float(b[f]) - float(st_[f])))
        same_v = [b["verdict"] for b in pr["tick2"]["pos"]] == [st_["verdict"] for st_ in stored]
        kv = {"largest_difference": worst, "tolerance": TOL_KNOWN, "verdicts_equal": same_v, "PASS": bool(worst <= TOL_KNOWN and same_v and len(stored) == len(pr["tick2"]["pos"]))}
    elif stored is None and "why" not in kv:
        kv = {"PASS": False, "why": "READ_HEADBOUND10K.json has no strategic HK entry"}
    y4["known_value"] = kv
    y4["PASS"] = bool(ok_files and kv.get("PASS") is True)
    out["y4"] = y4
    if not out["stage_B_pass"]:
        return out, None, eid, None, ctx, S
    return out, cells, eid, anc_ds, ctx, S


def heading_offsets(ctx, S, run, s):
    """heading offset at the tick t0+20 (executed s_pose heading minus logged heading at step j - age_s), degrees absolute, with the speed v0 of the tick row; the rows are the selected rows
    (the loader of the module is the selection), tick 2 = j - t0 20"""
    T = S.load_ticks(S.tick_path(ctx.W, run, s))
    v0 = T["v0"]
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


def consequence(fam, u):
    """the four cases of section 6 of the card, chosen by the verdicts"""
    better = [f for f in FAMS if fam[f] == "BETTER"]
    worse = [f for f in FAMS if fam[f] == "WORSE"]
    if (worse or above(u)) and not better:
        return "worse", SECTION6["worse"]
    if better and not worse and not above(u):
        return "better", SECTION6["better"]
    if all(fam[f] == "NOT-SEPARATED" for f in FAMS) and not below(u) and not above(u):
        return "none", SECTION6["none"]
    return "else", SECTION6["else"]


def main(argv=None):
    if "--check-tables" in (sys.argv[1:] if argv is None else argv):
        return check_tables()
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--taken", required=True)
    ap.add_argument("--taken67", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--test-wrapper", required=True)
    ap.add_argument("--test-out-head", required=True)
    ap.add_argument("--test-out-rule", required=True)
    ap.add_argument("--test-out-gain", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--add67-json", required=True)
    ap.add_argument("--add68-json", required=True)
    ap.add_argument("--pairs-read", default="")
    ap.add_argument("--w", default="/workspace/v8proof/integ10k")
    ap.add_argument("--tree", default="/workspace/v8proof/integ10k/tree")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def log(s):
        print(s, flush=True)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    read_pairs = set(p for p in a.pairs_read.split(",") if p)
    adm = {}
    for tag, path, runs_, chk in (("ADD67", a.add67_json, REUSED_ADD67, needs67), ("ADD68", a.add68_json, REUSED_ADD68, needs68)):
        ok0, why = admitted_json(path, tag)
        dep = [(A, B) for (A, B, _) in list(PAIRS_NAMED.values()) + list(PAIRS_BESIDE.values()) if chk(A, B)]
        in_read = all(("%s:%s" % (A, B)) in read_pairs for A, B in dep) if read_pairs else ok0
        ok = bool(ok0 and in_read and hw is not None and q3 is not None and all("%s_minus_%s" % (A, B) in hw["pairs"] for A, B in dep))
        if ok0 and not ok:
            why = "the pairs that need the runs of card %s are not in the headway / family reads" % tag
        adm[tag] = (ok, why if not ok else "%s controls_all_pass is true" % os.path.basename(path))
    adm67, adm68 = adm["ADD67"][0], adm["ADD68"][0]
    C, R = controls(a, hw, q3, adm67, adm68)
    runs_read = list(REUSED_OWN) + (list(REUSED_ADD67) if adm67 else []) + (list(REUSED_ADD68) if adm68 else []) + list(NEW_ARMS)
    SR = None
    try:
        SR, cells, eid, anc_ds, ctx, S = strategic(a, runs_read, log)
        C["y4_strategic_cell_known_values"] = SR.get("y4") or {"PASS": False, "why": "stage A of the strategic controls did not pass"}
        C["y8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"s1": SR["controls"]["s1"].get("pass"), "s2": SR["controls"]["s2"].get("pass"), "s3": SR["controls"]["s3"].get("pass"),
                                                          "s4": SR["controls"]["s4"].get("pass"), "s5": SR["controls"].get("s5", {}).get("pass"), "s6": SR["controls"]["s6"].get("pass"),
                                                          "s9": SR["controls"]["s9"].get("pass"), "missing_tick_files": SR["missing_tick_files"], "runs": SR["runs"],
                                                          "PASS": bool(SR["stage_A_pass"] and SR.get("stage_B_pass") is True)}
    except SystemExit as ex:
        SR, cells, eid, anc_ds, ctx, S = None, None, None, None, None, None
        C["y4_strategic_cell_known_values"] = {"PASS": False, "why": "SystemExit: %s" % ex}
        C["y8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"PASS": False, "why": "SystemExit: %s" % ex}
    except Exception as ex:                                                                     # noqa: BLE001
        SR, cells, eid, anc_ds, ctx, S = None, None, None, None, None, None
        C["y4_strategic_cell_known_values"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
        C["y8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
    C["y8_reader_controls_all"] = {"PASS": bool(C["y8_reader_controls_headway_family"]["PASS"] and C["y8_strategic_controls_s1_s2_s3_s4_s5_s6_s9"]["PASS"])}
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "runs_of_card_ADD67": {"admitted": adm67, "reason": adm["ADD67"][1]}, "runs_of_card_ADD68": {"admitted": adm68, "reason": adm["ADD68"][1]},
           "controls": C, "controls_all_pass": bool(ok), "names": None, "levels": None, "pairs": None, "cells": None, "families": None, "strategic": None, "reported_no_bar": R}
    print(TIER_LINE)
    print("== card ADD67 runs (%s) admitted: %s (%s)" % (", ".join(REUSED_ADD67), adm67, adm["ADD67"][1]))
    print("== card ADD68 runs (%s) admitted: %s (%s)" % (", ".join(REUSED_ADD68), adm68, adm["ADD68"][1]))
    if not adm68:
        print("== the pairs CYC-LABEL-RULE and CYC-READER-RULE are not named: %s" % adm["ADD68"][1])
    if not adm67:
        print("== the pairs CYC-PINNED, CYC-K, CYC-LABEL-CAND and CYC-READER-CAND are not named: %s" % adm["ADD67"][1])
    print("== controls (e1, y1 to y9)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:900]))
    if ok:
        lv = hw["levels"]
        arms_all = REUSED_OWN + (REUSED_ADD67 if adm67 else ()) + (REUSED_ADD68 if adm68 else ()) + tuple(NEW_ARMS)
        levels = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in arms_all if arm in lv}
        pairs_def = dict(PAIRS_NAMED)
        pairs_def.update(PAIRS_BESIDE)
        pairs_def = {k: v for k, v in pairs_def.items() if (adm67 or not needs67(v[0], v[1])) and (adm68 or not needs68(v[0], v[1]))}
        pairs, cells_q, fams, strat = {}, {}, {}, {}
        for nm, (A, B, _) in pairs_def.items():
            pairs[nm] = {"A": A, "B": B, "headway": ucf(hw, A, B)}
            if A.startswith("I10RQ50"):                                                          # READER U differences: the floor of row 106 (reported, names nothing)
                pairs[nm]["U_inside_reader_floor_row106"] = bool(abs(pairs[nm]["headway"]["U"]["pooled"]) <= READER_FLOOR)
            cells_q[nm] = cells_of(q3, A, B)
            fams[nm] = {f: reading(cells_q[nm], f) for f in NAMED}
            pr = S.pair_read(cells, sname(A), sname(B), eid, ticks=(2,))
            vs = [c["verdict"] for c in pr["tick2"]["pos"]]
            strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")}
                                                           for arc, b in zip(S.ARCS, pr["tick2"]["pos"])], "reading": family_of(vs)}
            fams[nm]["STRATEGIC"] = strat[nm]["reading"]
        names, cons = {}, {}
        for nm, (A, B, tok) in PAIRS_NAMED.items():
            if nm not in pairs:
                continue
            u = pairs[nm]["headway"]["U"]
            names[nm + "_U"] = "%s-%s-U-%s" % (PRE, tok, "LOWER" if below(u) else ("HIGHER" if above(u) else "NOT-SEPARATED"))
            for f in FAMS:
                names["%s_%s" % (nm, f)] = "%s-%s-%s-%s" % (PRE, tok, f, fams[nm][f])
            case, sent = consequence(fams[nm], u)
            cons[tok] = {"case": case, "sentence": sent}
        res.update(names=names, levels=levels, pairs=pairs, cells=cells_q, families=fams, strategic=strat)
        R["consequences_fixed_by_card_section_6_per_pair"] = cons
        # the heading offset at the tick t0+20 by speed bin for each run and its pair partner (reported without a bar)
        runs_off = sorted({sname(x) for nm, (A, B, _) in pairs_def.items() for x in (A, B)})
        R["r2_heading_offset_tick_t0_plus_20_by_speed_bin_per_run"] = {"I10" + r: offsets_by_speed(ctx, S, r) for r in runs_off}
        print("== names")
        for k, v in names.items():
            print("   %-26s %s" % (k, v))
        print("== consequences fixed by section 6 of the card (per pair; the card itself changes nothing)")
        for tok, c in cons.items():
            print("   %-16s [%s] %s" % (tok, c["case"], c["sentence"]))
        print("== levels of the runs U C F (per model seed; 562 scored windows, 642 lead windows)")
        for arm, v in levels.items():
            print("   %-16s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== pairs: U C F (pooled, interval, per seed) and the family reading over the named cells (STRATEGIC: pos at 30, 50, 80 m at the tick t0+20)")
        for nm, p in pairs.items():
            fl = "  [U inside the reader floor of row 106]" if p.get("U_inside_reader_floor_row106") else ""
            print("   %-24s %s minus %s | %s | %s%s" % (nm, p["A"], p["B"], line(p["headway"]), fams[nm], fl))
        print("== STRATEGIC pos (m) at the tick t0+20: diff [lo, hi] verdict")
        for nm, s_ in strat.items():
            print("   %-24s " % nm + " ; ".join("%dm %+.4f [%+.4f,%+.4f] %s" % (int(b["arc_m"]), b["mean"], b["lo"], b["hi"], b["verdict"]) for b in s_["pos_30_50_80m_tick_t0_plus_20"]))
        print("== cells (all ten per pair; the named ones decide the readings)")
        for nm, cs in cells_q.items():
            for c, v in cs.items():
                print("   %-24s %-30s D %+8.4f [%+.4f, %+.4f] %s %s%s" % (nm, c, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"], " *" if any(c in NAMED[f] for f in NAMED) else ""))
        print("== reported without a bar")
        for k, v in R.items():
            print("   %s %s" % (k, json.dumps(v)[:3000]))
    else:
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    res["strategic_controls_detail"] = None if SR is None else {k: v for k, v in SR.items() if k not in ("controls", "load_stats")}
    res["strategic_controls_values"] = None if SR is None else SR["controls"]
    res["strategic_load_stats"] = None if SR is None else SR.get("load_stats")
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
