"""read_accwin10k.py -- written by make_read_accwin10k.py from read_handover10k.py. Card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): the read (numpy only). The readouts of this
card: WINDOW (causal_w: the acceleration at every executed step is the executed speed change over the last 1.0 s) and ZERO (causal_z: no acceleration read); both keep
the causal speed. Runs R0W5, R0Z5 (5 steps), R0W10 (10 steps), control R0ZID (causal readout, must equal R0C5 of card ADD54). Controls z1 to z9 first. Text of the parent:
read_handover10k.py -- card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): the read (numpy only). Controls h1 to h9 first; the names and the differences only if every control
passed. No rule in this card. Readouts of the rebuilt ego state after executed steps: pinned (the tool as it is), backward (backward tangent at the last executed knot),
causal (speed column from executed steps only). Runs: R0C5 / R0B5 = 5 steps (1.0 s) with the causal / backward readout, R0C10 = 10 steps with the causal readout;
reused R0 (10 steps) and R0M5 (5 steps), pinned readout. Rules as frozen in the card:
  model, per readout   SP = speed error (5-step run with the readout) - speed error (R0M5);  RS = share of windows with last-step speed above logged + 5 m/s
  prior alone          RS and ADE6 at 5 steps under the readout against the pinned values
  the programme cycle  PR = R0C10 - R0"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

SEEDS = (0, 1)
DT = 0.2
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
WRAPPER_SHA_PREFIX = "fa7207d72311b5cf"                                               # the wrapper the tool check ran with (freeze line of the card in TIMES_NOTE.md)
READOUTS = (("WINDOW", "I10R0W5"), ("ZERO", "I10R0Z5"))
RUNS = {"I10R0ZID": ("causal", 5), "I10R0W5": ("causal_w", 5), "I10R0Z5": ("causal_z", 5), "I10R0W10": ("causal_w", 10)}
NEW_RUNS = [("I10R0ZID", 0)] + [(arm, k) for arm in ("I10R0W5", "I10R0Z5", "I10R0W10") for k in SEEDS]
BANKED_OF = {"I10R0ZID": "I10R0C5", "I10R0W5": "I10R0M5", "I10R0Z5": "I10R0M5", "I10R0W10": "I10R0"}
RS_BAR = (0.1768, 0.1726)                                                             # the card: twice the banked 2.0 s value of the model, per model seed
RS_KNOWN = {"I10R0": (0.0884, 0.0863), "I10R0M5": (0.9075, 0.9825)}
PRIOR_KNOWN = {"I10R0": (0.3014, 3.443), "I10R0M5": (0.4185, 5.407)}                  # RS and ADE6 (m) of the prior alone, pinned readout
SPEED = "LONGITUDINAL:speed_mae_mps"
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; no lead rule in this card); never driving performance"


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else ""


def dump(pull, arm, k, keys=("g", "cl", "fl", "v0", "eid")):
    files = sorted(glob.glob(os.path.join(pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k), "ep*.npz")))
    if len(files) != 141:
        return None
    out = {n: [] for n in keys}
    for f in files:
        z = np.load(f)
        for n in keys:
            out[n].append(np.asarray(z[n], np.float64))
    return {n: np.concatenate(v) for n, v in out.items()}


def replan_of_result(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "t1v8f_%s_s%d_m10_causal_heads.json" % (arm, k))
    m = re.search(r'"replan_steps": (\d+)', io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
    return int(m.group(1)) if m else None


def speeds(tr):
    p = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1) / DT


def rs_of(d, key="cl"):
    s, sg = speeds(d[key]), speeds(d["g"])
    fin = np.isfinite(d[key]).all(axis=(1, 2))
    return round(float(((s[:, -1] - sg[:, -1])[fin] > 5.0).mean()), 4)


def ade_w(d, key):
    return np.linalg.norm(d[key] - d["g"], axis=-1).mean(1)


def boot(diff, eid, n=2000, seed=20261119):
    """episode-cluster bootstrap of the mean of a per-window difference: episodes drawn with replacement, the statistic is the window mean of the drawn episodes."""
    ids = np.unique(eid)
    sums = np.array([diff[eid == e].sum() for e in ids])
    cnts = np.array([(eid == e).sum() for e in ids], np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    st = sums[pick].sum(1) / cnts[pick].sum(1)
    return [round(float(np.quantile(st, 0.025)), 4), round(float(np.quantile(st, 0.975)), 4)]


def controls(a, hw, q3):
    C, R = {}, {}
    p = os.path.join(a.pull, "small_accwin", "CMP_R0ZID.txt")
    m = re.search(r"CMP files=(\d+) differing=(\d+)", io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else "")
    C["z1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    h2, h3, h4, nfiles, shs, rs, D = {}, {}, {}, {}, {}, {}, {}
    for arm in ("I10R0", "I10R0M5", "I10R0C5"):
        for k in SEEDS:
            D[(arm, k)] = dump(a.pull, arm, k)
    pinned_rep = {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        mode, m_steps = RUNS[arm]
        d = dump(a.pull, arm, k)
        D[(arm, k)] = d
        nfiles[key] = None if d is None else 141
        pr = os.path.join(a.pull, key, "ticks", "run_%s.txt" % key)
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(pr, encoding="utf-8", errors="replace").read()) if os.path.exists(pr) else None
        shs[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
        L = jl(os.path.join(a.pull, key, "ticks", "readout_log_%s.json" % key))
        if L is None or d is None:
            h2[key] = h3[key] = h4[key] = {"PASS": None, "why": "readout log or dump missing"}
            continue
        h2[key] = {"readout": L["readout"], "log_steps": L["replan_steps"], "calls": L["calls"], "result_file_steps": replan_of_result(a.pull, arm, k)}
        h2[key]["PASS"] = bool(L["readout"] == mode and L["replan_steps"] == m_steps and L["calls"] > 0 and h2[key]["result_file_steps"] == m_steps)
        dv, da = float(L["max_abs_speed_minus_last_step"]), float(L["max_abs_acceleration_minus_executed"])
        if mode == "causal":
            h3[key] = {"max_speed_diff": dv, "max_acceleration_diff": da, "PASS": bool(dv <= 1e-6 and da <= 1e-6)}
        elif mode == "causal_w":
            dw = float(L["max_abs_acceleration_minus_window"])
            h3[key] = {"max_speed_diff": dv, "max_acceleration_minus_window": dw, "mean_abs_acceleration_read": L["mean_abs_acceleration_read"], "PASS": bool(dv <= 1e-6 and dw <= 1e-6)}
        else:
            az = float(L["max_abs_acceleration_read"])
            h3[key] = {"max_speed_diff": dv, "max_abs_acceleration_read": az, "PASS": bool(dv <= 1e-6 and az <= 1e-6)}
        b = D.get((BANKED_OF[arm], k))
        if b is None:
            h4[key] = {"PASS": None, "why": "banked dump missing"}
        else:
            w = max(float(np.nanmax(np.abs(d[kk][:, :m_steps] - b[kk][:, :m_steps]))) for kk in ("cl", "fl"))
            h4[key] = {"banked": BANKED_OF[arm], "first_block_max_abs_diff": w, "PASS": bool(w == 0.0)}
        rs[key] = {"model": rs_of(d, "cl"), "prior": rs_of(d, "fl")}
    R["pinned_readout_against_executed"] = pinned_rep
    known = {}
    for arm, exp in RS_KNOWN.items():
        got, pri = [], []
        for k in SEEDS:
            d = D.get((arm, k))
            got.append(None if d is None else rs_of(d, "cl"))
            pri.append(None if d is None else [rs_of(d, "fl"), round(float(ade_w(d, "fl").mean()), 3)])
            if d is not None:
                rs["%s_s%d" % (arm, k)] = {"model": got[-1], "prior": pri[-1][0]}
        okp = all(q is not None and abs(q[0] - PRIOR_KNOWN[arm][0]) <= 5e-5 and abs(q[1] - PRIOR_KNOWN[arm][1]) <= 5e-4 for q in pri)
        known[arm] = {"got": got, "card": list(exp), "prior_rs_ade6": pri, "prior_card": list(PRIOR_KNOWN[arm]), "PASS": bool(all(g is not None and abs(g - e) <= 5e-5 for g, e in zip(got, exp)) and okp)}
    for name, dd in (("z2_parameters_applied", h2), ("z3_readout_holds", h3), ("z4_first_block_equal", h4)):
        C[name] = {"per_run": dd, "PASS": (None if any(v["PASS"] is None for v in dd.values()) else bool(all(v["PASS"] for v in dd.values())))}
    cnt = (hw or {}).get("counts", {})
    C["z5_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 7 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or sha(os.path.join(base, p)) != h]
    C["z6_reused_and_known_values"] = {"n_files": len(ru["files"]), "differing": bad, "known_values": known, "PASS": bool(len(ru["files"]) == 4 and not bad and all(v["PASS"] for v in known.values()))}
    C["z7_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["z7_reader_controls"]["PASS"] = bool(C["z7_reader_controls"]["headway"] and C["z7_reader_controls"]["family"])
    C["z8_model"] = {"per_run": shs, "PASS": bool(len(shs) == 7 and all(shs.values()))}
    t_txt = io.open(a.test_out, encoding="utf-8", errors="replace").read() if a.test_out and os.path.exists(a.test_out) else ""
    w1, w2 = sha(a.wrapper), sha(a.test_wrapper)
    C["z9_tool_check"] = {"wrapper_sha256": w1[:16], "test_wrapper_sha256": w2[:16], "check_line": (re.findall(r"ZZ CHECK-TICK-READOUT.*", t_txt) or [None])[-1],
                          "PASS": bool(w1 and w1 == w2 and w1.startswith(WRAPPER_SHA_PREFIX) and "ZZ CHECK-TICK-READOUT-RZ all hold: True" in t_txt)}
    return C, R, rs, D


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def ucf(hw, A, B):
    return {c: cell(hw, A, B, c) for c in ("U", "C", "F")}


def fams(q3, A, B):
    return {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for f, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}


def line(v):
    return "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--test-wrapper", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C, R, rs, D = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD55", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "reported_no_bar": R, "names": None}
    print(TIER_LINE)
    print("== controls (z1 to z9)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:760]))
    print("== reported, no bar")
    for k, v in R.items():
        print("   %s %s" % (k, json.dumps(v)[:600]))
    if ok:
        lv = hw["levels"]
        levels = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in ("I10R0", "I10R0M5", "I10R0C5", "I10R0W5", "I10R0Z5", "I10R0W10")}
        names, sp, gc, prior = {}, {}, {}, {}
        ref = D[("I10R0M5", 0)]
        for rk, arm in READOUTS:
            fm = fams(q3, arm, "I10R0M5")
            d = fm[SPEED]
            rs_a = [rs["%s_s%d" % (arm, k)]["model"] for k in SEEDS]
            sp[rk] = {"speed_error": d, "RS": rs_a, "RS_bar": list(RS_BAR), "headway": ucf(hw, arm, "I10R0M5"), "families": fm}
            if all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0:
                names["model_" + rk] = "ACCWIN10K-MODEL-%s-%s" % (rk, "STABILISES-1S" if all(r <= b for r, b in zip(rs_a, RS_BAR)) else "SLOWS-CLIMB")
            else:
                names["model_" + rk] = "ACCWIN10K-MODEL-%s-NO-EFFECT" % rk
            gc[rk] = {"headway": ucf(hw, arm, "I10R0"), "families": fams(q3, arm, "I10R0")}
            d0, d1 = D[(arm, 0)], D[(arm, 1)]
            a_new, a_ref = ade_w(d0, "fl"), ade_w(ref, "fl")
            pr_rs, pr_ade = rs_of(d0, "fl"), round(float(a_new.mean()), 3)
            ci = boot(a_new - a_ref, d0["eid"])
            prior[rk] = {"RS": pr_rs, "ADE6_m": pr_ade, "ADE6_minus_pinned_5_steps": round(float((a_new - a_ref).mean()), 4), "interval": ci, "speed_error_mps": round(float(np.abs(speeds(d0["fl"]) - speeds(d0["g"])).mean()), 3),
                         "equal_between_model_seeds_max_abs_diff": float(np.nanmax(np.abs(d0["fl"] - d1["fl"])))}
            if pr_rs <= PRIOR_KNOWN["I10R0"][0] and pr_ade <= PRIOR_KNOWN["I10R0"][1]:
                names["prior_" + rk] = "ACCWIN10K-PRIOR-%s-REMOVES-CLIMB" % rk
            elif pr_rs < PRIOR_KNOWN["I10R0M5"][0] and ci[1] < 0:
                names["prior_" + rk] = "ACCWIN10K-PRIOR-%s-REDUCES-CLIMB" % rk
            else:
                names["prior_" + rk] = "ACCWIN10K-PRIOR-%s-CLIMB-STAYS" % rk
        pr = {"headway": ucf(hw, "I10R0W10", "I10R0"), "families": fams(q3, "I10R0W10", "I10R0")}
        worse = any(v["interval"][0] > 0 for v in pr["headway"].values()) or any(v["verdict"] == "WORSE" for v in pr["families"].values())
        better = any(v["interval"][1] < 0 for v in pr["headway"].values()) or any(v["verdict"] == "BETTER" for v in pr["families"].values())
        names["cycle_2s"] = "ACCWIN10K-WINDOW-%s-AT-2S" % ("COSTS" if worse else ("HELPS" if better else "NEUTRAL"))
        d10 = D[("I10R0W10", 0)]
        prior["WINDOW_10_steps"] = {"RS": rs_of(d10, "fl"), "ADE6_m": round(float(ade_w(d10, "fl").mean()), 3)}
        model_ade = {"%s_s%d" % (arm, k): round(float(ade_w(D[(arm, k)], "cl").mean()), 3) for arm in ("I10R0", "I10R0M5", "I10R0C5", "I10R0W5", "I10R0Z5", "I10R0W10") for k in SEEDS}
        cb = {"headway": ucf(hw, "I10R0W5", "I10R0Z5"), "families": fams(q3, "I10R0W5", "I10R0Z5")}
        vc = {rk: {"headway": ucf(hw, arm, "I10R0C5"), "families": fams(q3, arm, "I10R0C5")} for rk, arm in READOUTS}
        res.update(names=names, levels=levels, RS=rs, SP=sp, GC=gc, PR=pr, prior=prior, model_ADE6_m=model_ade, window_minus_zero=cb, against_causal_readout=vc)
        print("== names")
        for k, v in names.items():
            print("   %-16s %s" % (k, v))
        print("== levels of the model alone (per model seed)")
        for arm, v in levels.items():
            print("   %-10s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== RS: share of windows with last-step speed more than 5 m/s above logged (model rollout / prior alone), per model seed; ADE6 of the model (m)")
        for key in sorted(rs):
            print("   %-13s model %.4f  prior %.4f  model ADE6 %s" % (key, rs[key]["model"], rs[key]["prior"], model_ade.get(key, "")))
        print("== SP: the readout at 1.0 s against the pinned readout at 1.0 s (model)")
        for rk, v in sp.items():
            d = v["speed_error"]
            print("   %-8s speed error %+.4f [%+.4f, %+.4f] %s %s | RS %s bar %s | %s" % (rk, d["D"], d["interval"][0], d["interval"][1], d["per_seed"], d["verdict"], v["RS"], v["RS_bar"], line(v["headway"])))
        print("== the prior alone at 5 steps under each readout (pinned: RS 0.4185, ADE6 5.407 m; at 10 steps 0.3014, 3.443 m)")
        for rk, v in prior.items():
            print("   %-16s %s" % (rk, json.dumps(v)))
        print("== GC: the readout at 1.0 s against the banked 2.0 s run")
        for rk, v in gc.items():
            print("   %-8s %s" % (rk, line(v["headway"])))
        print("== PR: the WINDOW readout at the programme's cycle (R0W10 minus R0)")
        print("   %s" % line(pr["headway"]))
        print("== beside: WINDOW minus ZERO at 1.0 s")
        print("   %s" % line(cb["headway"]))
        print("== beside: each readout against the causal readout of card ADD54 at 1.0 s")
        for rk, v in vc.items():
            d = v["families"][SPEED]
            print("   %-8s speed error %+.4f [%+.4f, %+.4f] %s | %s" % (rk, d["D"], d["interval"][0], d["interval"][1], d["per_seed"], line(v["headway"])))
        print("== families")
        groups = [("SP_" + rk, v["families"]) for rk, v in sp.items()] + [("GC_" + rk, v["families"]) for rk, v in gc.items()] + [("PR", pr["families"]), ("window_minus_zero", cb["families"])] + [("vs_causal_" + rk, v["families"]) for rk, v in vc.items()]
        for grp, fm in groups:
            for f, v in fm.items():
                print("   %-22s %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (grp, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
