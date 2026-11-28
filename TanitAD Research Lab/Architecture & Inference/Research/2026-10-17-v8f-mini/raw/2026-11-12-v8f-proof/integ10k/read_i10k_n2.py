#!/usr/bin/env python
"""read_i10k.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE) and its note N1: the read (numpy only). Controls k1 to k10 first; the names and the outcome differences only if every control passed.

Inputs: the headway read (RESULT_HEADWAY.json of headway_read_r2.py) and the family read (RESULT_Q3.json of q3_read_r3.py) on the pairs of read_i10k.sh, the pulled run folders
(pull/<NAME>_s<k>/), the pulled small files (pull/small: reader controls, maps, train-log files), K5_BANKED.json (note N1).
Arm names: I10R0, I10RLb, I10RR, I10RRid, I10RR0 (model seed k = reader seed k), I10RRx (model 0 with reader 1, model 1 with reader 0).
A control that did not pass, or that is not evaluable yet (PASS null), withholds every name and every outcome difference.
"""
import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np

TIER_LINE = "T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
STRATEGIC_LINES = (
    "STRATEGIC, T0 at the first tick: the banked read of `UE_s0` and `UE_s1` of the other session, cited with its file, stated as unchanged by construction for an inference-time rule, "
    "and backed by one identity control that the other session runs: its strategic reader on the wrapper with the rule on and with the rule off gives equal outputs bit for bit. "
    "Without that control the family is reported as the banked read of the model without the rule and nothing is said about the candidate.",
    "STRATEGIC along the rule-driven rollout (T1): NOT COMPUTED (n = 0; no instrument exists for it and none is pre-registered). No strategic effect of the rule is claimed in either direction.")
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}
N_WINDOWS = 2399
SEEDS = (0, 1)
STEPS_EXPECTED = 19164


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def cell(c):
    return {"per_seed": [round(x, 4) for x in c["d_s"]], "pooled": round(c["D"], 4), "interval": [round(c["A_lo"], 4), round(c["A_hi"], 4)], "excludes_0": bool(c["A_excludes_0"])}


def pk(x, y):
    return "%s_minus_%s" % (x, y)


def U(hw, x, y):
    return hw["pairs"][pk(x, y)]["cells"]["U"] if hw and pk(x, y) in hw["pairs"] else None


def E6(q3, x, y):
    return q3["pairs"][pk(x, y)]["beside"]["ade6_m|all"] if q3 and pk(x, y) in q3["pairs"] else None


def dumps_equal(dir_a, dir_b):
    """bit-for-bit comparison of the per-episode dump files of two runs: -> (n_files, n_differing). A missing file on either side counts as differing."""
    fa = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, "ep*.npz")))
    fb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_b, "ep*.npz")))
    names = sorted(set(fa) | set(fb))
    diff = 0
    for n in names:
        pa, pb = os.path.join(dir_a, n), os.path.join(dir_b, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            diff += 1
            continue
        A, B = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        if sorted(A.files) != sorted(B.files) or any(A[k].shape != B[k].shape or not np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f") for k in A.files):
            diff += 1
    return len(names), diff


def tri(ok):
    return None if ok is None else bool(ok)


def controls(inp):
    """inp keys: hw, q3, rc (per seed reader controls dict or None), dump_eq (per seed (n, ndiff) or None), r0json (per seed dict of t1v8f json or None), ckpt_sha (per seed str), banked (K5_BANKED dict),
    qmap (per seed dict or None), train (per seed dict {steps, p5_all_pass} or None). Returns {name: {"PASS": True/False/None, ...}}."""
    C = {}
    hw, q3 = inp.get("hw"), inp.get("q3")
    rc = inp.get("rc", {})
    # k1
    if all(rc.get(k) for k in SEEDS):
        mae = float(np.mean([rc[k]["k1"]["gap_mae_m_clipped_40"] for k in SEEDS]))
        aucs = [rc[k]["k1"]["auroc"] for k in SEEDS]
        match = [rc[k]["k1"]["label_match_share"] for k in SEEDS]
        C["k1_reader_good_enough"] = {"gap_mae_m_mean_of_seeds": round(mae, 4), "bar_mae": 5.80, "auroc_per_seed": [round(a, 4) for a in aucs], "bar_auroc": 0.85,
                                      "label_match_share_per_seed": [round(m, 4) for m in match], "PASS": bool(mae <= 5.80 and all(a >= 0.85 for a in aucs))}
    else:
        C["k1_reader_good_enough"] = {"PASS": None, "why": "reader controls not pulled"}
    # k2
    de = inp.get("dump_eq", {})
    if all(de.get(k) for k in SEEDS):
        C["k2_identity_RRid_equals_R0"] = {"per_seed_files_differing": {str(k): list(de[k]) for k in SEEDS}, "PASS": bool(all(de[k][0] > 0 and de[k][1] == 0 for k in SEEDS))}
    else:
        C["k2_identity_RRid_equals_R0"] = {"PASS": None, "why": "dumps not pulled"}
    # k3
    e = E6(q3, "I10RR0", "I10R0")
    C["k3_planted_RR0_raises_6s_error"] = dict(cell(e), PASS=bool(e["D"] > 0 and e["A_excludes_0"])) if e else {"PASS": None, "why": "pair missing"}
    # k4
    u = U(hw, "I10RLb", "I10R0")
    C["k4_label_rule_cuts_U"] = dict(cell(u), PASS=bool(all(d < 0 for d in u["d_s"]) and u["D"] < 0 and u["A_excludes_0"])) if u else {"PASS": None, "why": "pair missing"}
    # k5
    banked, tol = inp.get("banked") or {}, float((inp.get("banked") or {}).get("tolerance_m", 0.02))
    k5 = {}
    for k in SEEDS:
        key = "UE_s%d" % k
        b, j = banked.get(key), inp.get("r0json", {}).get(k)
        sha = inp.get("ckpt_sha", {}).get(k, "")
        sha_ok = bool(sha and sha.startswith(CKPT_SHA_PREFIX[k]))
        if b is None:
            k5[key] = {"PASS": None, "why": "no banked read yet (not evaluable yet)", "ckpt_sha_ok": sha_ok}
        elif j is None:
            k5[key] = {"PASS": None, "why": "R0 json not pulled", "ckpt_sha_ok": sha_ok}
        else:
            m = j["intervals"]["metrics"]
            d1, d2 = abs(m["fde_last_m"] - b["fde_last_m"]), abs(m["ade_dense_m"] - b["ade_dense_m"])
            k5[key] = {"fde_last_m": m["fde_last_m"], "banked_fde": b["fde_last_m"], "ade_dense_m": m["ade_dense_m"], "banked_ade": b["ade_dense_m"], "abs_diff": [round(d1, 4), round(d2, 4)],
                       "banked_file": b.get("file"), "banked_sha256": b.get("sha256"), "ckpt_sha_ok": sha_ok, "PASS": bool(d1 <= tol and d2 <= tol and sha_ok)}
    vals = [v["PASS"] for v in k5.values()]
    C["k5_model_is_the_banked_one"] = {"per_seed": k5, "PASS": (False if False in vals else (None if None in vals else True))}
    # k6
    if all(rc.get(k) for k in SEEDS) and all(inp.get("qmap", {}).get(k) for k in SEEDS):
        C["k6_no_leak"] = {"leak_count": [rc[k]["k6"]["leak_count"] for k in SEEDS], "planted_detected": [rc[k]["k6"]["planted_shared_id_detected"] for k in SEEDS],
                           "fit_check_disjoint": [inp["qmap"][k]["fit_check_clips_disjoint"] for k in SEEDS],
                           "PASS": bool(all(rc[k]["k6"]["leak_count"] == 0 and rc[k]["k6"]["planted_shared_id_detected"] and inp["qmap"][k]["fit_check_clips_disjoint"] for k in SEEDS))}
    else:
        C["k6_no_leak"] = {"PASS": None, "why": "controls or maps not pulled"}
    # k7
    C["k7_causal_reader"] = {"all_unchanged": [rc[k]["k7"]["all_unchanged"] for k in SEEDS], "PASS": bool(all(rc[k]["k7"]["all_unchanged"] for k in SEEDS))} if all(rc.get(k) for k in SEEDS) else {"PASS": None, "why": "reader controls not pulled"}
    # k8
    cnt = (hw or {}).get("counts", {})
    tr = inp.get("train", {})
    if hw and all(tr.get(k) for k in SEEDS):
        C["k8_counts_and_train_log"] = {"train_log_part": "read by note N2 of the card (K8_N2.json, key k8_n2_pass); p5_all_pass is the tool check on every training step: reported, not used", "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "train": {str(k): tr[k] for k in SEEDS},
                                        "PASS": bool(cnt.get("n_windows_total") == N_WINDOWS and all(tr[k].get("steps") == STEPS_EXPECTED and tr[k].get("k8_n2_pass") is True for k in SEEDS))}
    else:
        C["k8_counts_and_train_log"] = {"PASS": None, "why": "headway read or train logs missing"}
    # k9
    qm = inp.get("qmap", {})
    C["k9_map_coverage"] = {"share_below_Q": [qm[k]["coverage"]["share_below_Q"] for k in SEEDS], "band": [0.15, 0.25], "PASS": bool(all(0.15 <= qm[k]["coverage"]["share_below_Q"] <= 0.25 for k in SEEDS))} if all(qm.get(k) for k in SEEDS) else {"PASS": None, "why": "maps not pulled"}
    # k10
    if hw and q3:
        a, b = bool(hw["controls"].get("ALL_PASS")), bool(q3["controls"].get("ALL_PASS"))
        C["k10_headway_h1_h6_and_family_q1_q5"] = {"headway": a, "family": b, "PASS": bool(a and b)}
    else:
        C["k10_headway_h1_h6_and_family_q1_q5"] = {"PASS": None, "why": "headway or family read missing"}
    return C


def names_and_beside(inp):
    hw, q3 = inp["hw"], inp["q3"]
    e1, e2, lb = U(hw, "I10RR", "I10R0"), U(hw, "I10RR", "I10RLb"), U(hw, "I10RLb", "I10R0")
    xr = U(hw, "I10RRx", "I10R0")
    N = {}
    crossed_below = bool(xr and all(d < 0 for d in xr["d_s"]))
    N["cut"] = "INTEG10K-CUTS-UNSAFE" if (all(d < 0 for d in e1["d_s"]) and e1["A_excludes_0"] and crossed_below) else "INTEG10K-NO-CUT"
    share = [(d / l) if l not in (0, 0.0) else float("nan") for d, l in zip(e1["d_s"], lb["d_s"])]
    N["share_reached_per_seed"] = [round(s, 4) for s in share]
    N["reaches_label_rule"] = "INTEG10K-REACHES-LABEL-RULE" if all(np.isfinite(s) and s >= 0.5 for s in share) else "not named"
    q = q3["pairs"][pk("I10RR", "I10R0")]["Q3"]
    N["price"] = "INTEG10K-" + str(q.get("name"))
    N["cost_families"] = q.get("cost_families")
    B = {"E1_U_RR_minus_R0": cell(e1), "E2_U_RR_minus_RLb": cell(e2), "U_RLb_minus_R0": cell(lb), "U_RRx_minus_R0 (crossed, per model seed)": cell(xr) if xr else None,
         "ade6_RR_minus_R0_m": cell(E6(q3, "I10RR", "I10R0")) if E6(q3, "I10RR", "I10R0") else None}
    try:
        B["contact_RR_minus_RLb"] = {k: cell(v) for k, v in hw["pairs"][pk("I10RR", "I10RLb")]["cells"].items() if k.startswith("C") and isinstance(v, dict) and "d_s" in v}
    except Exception as ex:  # noqa: BLE001
        B["contact_RR_minus_RLb"] = "not available: %s" % ex
    return N, B


def evaluate(inp):
    """controls first; the names and the outcome differences only if every control is True (a None, not evaluable, withholds them)."""
    res = {"card": "PREREG_V8F_PROOF_ADD46", "tier": TIER_LINE, "strategic": list(STRATEGIC_LINES), "controls": controls(inp), "names": None}
    res["controls_all_pass"] = all(v.get("PASS") is True for v in res["controls"].values())
    if res["controls_all_pass"]:
        res["names"], res["beside"] = names_and_beside(inp)
    return res


def load_inputs(a):
    inp = {"hw": jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), "q3": jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json")), "banked": jl(a.k5), "rc": {}, "dump_eq": {}, "r0json": {},
           "ckpt_sha": {}, "qmap": {}, "train": {}}
    sm = os.path.join(a.pull, "small_n2")
    for k in SEEDS:
        inp["rc"][k] = jl(os.path.join(sm, "READER_CONTROLS_LR16_s%d.json" % k))
        inp["qmap"][k] = jl(os.path.join(sm, "QMAP_LR16_s%d.json" % k))
        d0 = os.path.join(a.pull, "I10R0_s%d" % k, "dump_I10R0_s%d_m10_causal_heads" % k)
        d1 = os.path.join(a.pull, "I10RRid_s%d" % k, "dump_I10RRid_s%d_m10_causal_heads" % k)
        if os.path.isdir(d0) and os.path.isdir(d1):
            inp["dump_eq"][k] = dumps_equal(d1, d0)
        inp["r0json"][k] = jl(os.path.join(a.pull, "I10R0_s%d" % k, "t1v8f_I10R0_s%d_m10_causal_heads.json" % k))
        run = os.path.join(a.pull, "I10R0_s%d" % k, "ticks", "run_I10R0_s%d.txt" % k)
        if os.path.exists(run):
            mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(run, encoding="utf-8", errors="replace").read())
            inp["ckpt_sha"][k] = mm.group(1) if mm else ""
        td = jl(os.path.join(sm, "TRAIN_DONE.json"))
        if td:
            inp["train"][k] = {"steps": td.get("steps"), "p5_all_pass": bool(td.get("p5_all_pass")) and bool((td.get("p5") or {}).get("TK16_s%d" % k, {}).get("pass", True))}
            inp["train"][k]["k8_n2_pass"] = bool((jl(os.path.join(sm, "K8_N2.json")) or {}).get("PASS") is True)   # note N2: the train-log part of k8
    return inp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)       # holds hw/RESULT_HEADWAY.json and q3/RESULT_Q3.json
    ap.add_argument("--pull", required=True)           # pull/<NAME>_s<k>/ and pull/small
    ap.add_argument("--k5", required=True)             # K5_BANKED.json
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    inp = load_inputs(a)
    print("TIER:", TIER_LINE)
    res = evaluate(inp)
    C = res["controls"]
    print("== controls (k1 to k10)")
    for k, v in C.items():
        print("  ", k, {True: "PASS", False: "FAIL", None: "NOT EVALUABLE"}[v.get("PASS")], json.dumps(v)[:300])
    allp = res["controls_all_pass"]
    if allp:
        N, B = res["names"], res["beside"]
        print("== names")
        print("  ", json.dumps(N))
        for k, v in B.items():
            print("  ", k, json.dumps(v)[:330])
    else:
        print("== names: NOT GIVEN (a control did not pass or is not evaluable yet); the outcome differences are not printed")
    for s in STRATEGIC_LINES:
        print("  *", s)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
