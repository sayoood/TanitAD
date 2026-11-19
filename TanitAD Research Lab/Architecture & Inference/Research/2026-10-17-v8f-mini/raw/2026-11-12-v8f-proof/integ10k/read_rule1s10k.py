"""read_rule1s10k.py -- card PREREG_V8F_PROOF_ADD59 (RULE1S10K): the read (numpy only). Written from read_cycle10k.py (CY, RU, names, rule-log controls) and
read_accwin10k.py (readout-log controls, first-block control, wrapper sha256). Controls r1 to r9 first; the names and the differences only if every control passed.
Forms: NORULE (R0), LABEL (label-fed rule, RLc), READER (reader-fed rule at map level 0.50, closing 0, RQ50). Readouts: backward (primary), ZERO = causal_z (beside).
Arms per form f (run-folder prefix I10):  bk5 = backward at 5 steps (1.0 s)   bk10 = backward at 10 steps (2.0 s)   p10 = banked pinned readout at 10 steps
                                           z5 = ZERO readout at 5 steps (1.0 s)
Rules as frozen in the card (section 4), all on U; C and F and the four families are printed beside for every pair:
  CY(f) = bk5 - bk10                     RU(rule f) = bk5(f) - bk5(NORULE)   (beside: the same at 2.0 s = bk10(f) - bk10(NORULE))
  GB(rule f) = bk5(f) - p10(f)           RB(f) = bk10(f) - p10(f)
  beside, ZERO: z5(rule f) - z5(NORULE) and z5(rule f) - p10(rule f)
Reused without a rerun (result files hashed before the run, control r6): R0B5, R0Z5, R0, RLc, RQ50, R0M5, RLcM5, RQ50M5, per model seed."""
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
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
WRAPPER_SHA_PREFIX = "fa7207d72311b5cf"                                               # card ADD55 tool check (control r9)
READER_FLOOR = 0.0196                                                                 # row 106: the reader floor on U
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions); never driving performance"
STRATEGIC_LINE = ("STRATEGIC: as note N1 of card ADD46 (at T0 on the first tick the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, "
                  "no instrument). The first tick does not depend on the readout.")
FORMS = (("NORULE", {"bk5": "I10R0B5", "bk10": "I10R0Bk10", "p10": "I10R0", "z5": "I10R0Z5"}),
         ("LABEL", {"bk5": "I10RLcBk5", "bk10": "I10RLcBk10", "p10": "I10RLc", "z5": "I10RLcZ5"}),
         ("READER", {"bk5": "I10RQ50Bk5", "bk10": "I10RQ50Bk10", "p10": "I10RQ50", "z5": "I10RQ50Z5"}))
ARMS = {fk: d for fk, d in FORMS}
RULE_FORMS = ("LABEL", "READER")
# new runs: name -> (readout, replan steps, has a rule log, banked run whose first block and first-tick inputs it must equal)
RUNS = {"I10R0BkID": ("backward", 5, False, "I10R0B5"),
        "I10RLcBk5": ("backward", 5, True, "I10RLcM5"), "I10RQ50Bk5": ("backward", 5, True, "I10RQ50M5"),
        "I10R0Bk10": ("backward", 10, False, "I10R0"), "I10RLcBk10": ("backward", 10, True, "I10RLc"), "I10RQ50Bk10": ("backward", 10, True, "I10RQ50"),
        "I10RLcZ5": ("causal_z", 5, True, "I10RLcM5"), "I10RQ50Z5": ("causal_z", 5, True, "I10RQ50M5")}
NEW_RUNS = [("I10R0BkID", 0)] + [(n, k) for n in RUNS if n != "I10R0BkID" for k in SEEDS]       # fifteen rollouts
REUSED_RUNS = ("I10R0B5", "I10R0Z5", "I10R0", "I10RLc", "I10RQ50", "I10R0M5", "I10RLcM5", "I10RQ50M5")
RULE_RUNS_5 = ("I10RLcBk5", "I10RQ50Bk5", "I10RLcZ5", "I10RQ50Z5")
READER_RUNS = ("I10RQ50Bk5", "I10RQ50Bk10", "I10RQ50Z5")


def pair_groups():
    """the pairs of card section 4 as {group: {form: (A, B)}}; A minus B. podread_rule1s.sh holds the same list (text-checked)."""
    n = ARMS["NORULE"]
    g = {"CY": {fk: (a["bk5"], a["bk10"]) for fk, a in FORMS},
         "RU_1s": {fk: (ARMS[fk]["bk5"], n["bk5"]) for fk in RULE_FORMS},
         "RU_2s": {fk: (ARMS[fk]["bk10"], n["bk10"]) for fk in RULE_FORMS},
         "GB": {fk: (ARMS[fk]["bk5"], ARMS[fk]["p10"]) for fk in RULE_FORMS},
         "RB": {fk: (a["bk10"], a["p10"]) for fk, a in FORMS},
         "ZERO_RU": {fk: (ARMS[fk]["z5"], n["z5"]) for fk in RULE_FORMS},
         "ZERO_GB": {fk: (ARMS[fk]["z5"], ARMS[fk]["p10"]) for fk in RULE_FORMS}}
    return g


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else ""


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def dump(pull, arm, k, keys=("cl", "fl")):
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


def keyed(L, sel=None):
    """rows of a rule log by (clip, window start, cache step)."""
    ci, t0, j = (np.asarray(L[c], np.int64) for c in ("ci", "t0", "j"))
    idx = np.arange(len(ci)) if sel is None else np.flatnonzero(sel)
    return {(int(ci[i]), int(t0[i]), int(j[i])): i for i in idx}


def first_tick_inputs(B, N):
    """source gap, closing speed, ego speed and lead speed of the first tick of every window: banked log B against new log N. returns (n windows, worst abs difference)."""
    kb = keyed(B, np.asarray(B["tick"]) == 0)
    kn = keyed(N, np.asarray(N["tick"]) == 0)
    if set(kb) != set(kn):
        return len(kb), float("inf")
    ib = np.array([kb[q] for q in sorted(kb)])
    im = np.array([kn[q] for q in sorted(kb)])
    worst = 0.0
    for c in ("g_src", "c_src", "v0", "v_L"):
        x, y = np.asarray(B[c], np.float64)[ib], np.asarray(N[c], np.float64)[im]
        if not np.array_equal(np.isfinite(x), np.isfinite(y)):
            return len(kb), float("inf")
        f = np.isfinite(x)
        worst = max(worst, float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0)
    return len(kb), worst


def controls(a, hw, q3):
    C = {}
    # r1: the queue's known-value comparison (marker r1scmp on the pod)
    p = os.path.join(a.pull, "small_rule1s", "CMP_R0BkID.txt")
    m = re.search(r"CMP files=(\d+) differing=(\d+)", io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else "")
    C["r1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    r2, r3, r4, nfiles, shs, r8r = {}, {}, {}, {}, {}, {}
    D = {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        readout, steps, has_rule, banked = RUNS[arm]
        d = dump(a.pull, arm, k)
        D[(arm, k)] = d
        nfiles[key] = None if d is None else 141
        pr = os.path.join(a.pull, key, "ticks", "run_%s.txt" % key)
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(pr, encoding="utf-8", errors="replace").read() if os.path.exists(pr) else "")
        shs[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
        L = jl(os.path.join(a.pull, key, "ticks", "readout_log_%s.json" % key))
        N = npz(a.pull, arm, k) if has_rule else None
        if L is None or d is None or (has_rule and N is None):
            r2[key] = r3[key] = r4[key] = {"PASS": None, "why": "readout log, rule log or dump missing"}
            continue
        # r2: parameters applied
        r2[key] = {"readout": L["readout"], "log_steps": L["replan_steps"], "calls": L["calls"], "result_file_steps": replan_of_result(a.pull, arm, k)}
        ok2 = bool(L["readout"] == readout and L["replan_steps"] == steps and L["calls"] > 0 and r2[key]["result_file_steps"] == steps)
        if has_rule:
            tk = sorted(set(np.asarray(N["tick"], np.int64).tolist()))
            r2[key].update(rule_log_steps=int(N["replan_steps"]), ticks=tk)
            ok2 = ok2 and int(N["replan_steps"]) == steps
            if steps == 5:
                ok2 = ok2 and tk == [0, 1, 2, 3, 4, 5]
        r2[key]["PASS"] = bool(ok2)
        # r3: the readout does what it says (readout log over all committed plans)
        dv = float(L["max_abs_speed_minus_last_step"])
        r3[key] = {"max_speed_diff": dv, "max_abs_acceleration_read": float(L.get("max_abs_acceleration_read", float("nan")))}
        r3[key]["PASS"] = bool(dv <= 1e-6 and (readout != "causal_z" or r3[key]["max_abs_acceleration_read"] <= 1e-6))
        # r4: the first block does not depend on the readout (banked run of the same form and cycle), and the first-tick inputs of the rule log
        b = dump(a.pull, banked, k)
        if b is None:
            r4[key] = {"PASS": None, "why": "banked dump %s missing" % banked}
            continue
        w = max(float(np.nanmax(np.abs(d[kk][:, :steps] - b[kk][:, :steps]))) for kk in ("cl", "fl"))
        r4[key] = {"banked": banked, "first_block_steps": steps, "first_block_max_abs_diff": w}
        ok4 = bool(w == 0.0)
        if has_rule:
            B = npz(a.pull, banked, k)
            if B is None:
                r4[key].update(PASS=None, why="banked rule log missing")
                continue
            n_ft, worst = first_tick_inputs(B, N)
            r4[key].update(n_first_ticks=n_ft, max_abs_first_tick_inputs=worst, rows_new=int(len(N["ci"])), rows_banked=int(len(B["ci"])))
            ok4 = ok4 and n_ft == 2399 and worst <= 1e-6
        r4[key]["PASS"] = bool(ok4)
        # r8 (reader-fed runs): reader checkpoint and map are those of the banked RQ50 run
        if arm in READER_RUNS:
            B = npz(a.pull, "I10RQ50", k)
            if B is None or "reader_sha256" not in N.files or "qmap_sha256" not in N.files or "reader_sha256" not in B.files or "qmap_sha256" not in B.files:
                r8r[key] = {"PASS": None, "why": "reader or map sha256 missing in a rule log"}
            else:
                r8r[key] = {"reader_sha256": str(N["reader_sha256"])[:16], "reader_sha256_banked": str(B["reader_sha256"])[:16], "qmap_sha256": str(N["qmap_sha256"])[:16],
                            "qmap_sha256_banked": str(B["qmap_sha256"])[:16]}
                r8r[key]["PASS"] = bool(str(N["reader_sha256"]) == str(B["reader_sha256"]) and str(N["qmap_sha256"]) == str(B["qmap_sha256"]) and len(str(N["reader_sha256"])) == 64)
    for name, dd in (("r2_parameters_applied", r2), ("r3_readout_holds", r3), ("r4_first_block_and_first_tick_equal", r4)):
        C[name] = {"per_run": dd, "PASS": (None if any(v["PASS"] is None for v in dd.values()) else bool(all(v["PASS"] for v in dd.values())))}
    cnt = (hw or {}).get("counts", {})
    C["r5_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 15 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or sha(os.path.join(base, p)) != h]
    want = {"integ10k/pull/%s_s%d/t1v8f_%s_s%d_m10_causal_heads.json" % (n, k, n, k) for n in REUSED_RUNS for k in SEEDS}
    C["r6_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "names_as_card": bool(set(ru["files"]) == want), "PASS": bool(len(ru["files"]) == 16 and set(ru["files"]) == want and not bad)}
    C["r7_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["r7_reader_controls"]["PASS"] = bool(C["r7_reader_controls"]["headway"] and C["r7_reader_controls"]["family"])
    C["r8_model_reader_map"] = {"checkpoint_per_run": shs, "reader_and_map_per_run": r8r,
                                "PASS": (None if any(v["PASS"] is None for v in r8r.values()) or len(r8r) != 6 else bool(len(shs) == 15 and all(shs.values()) and all(v["PASS"] for v in r8r.values())))}
    w1 = sha(a.wrapper)
    C["r9_tool"] = {"wrapper_sha256": w1[:16], "PASS": bool(w1 and w1.startswith(WRAPPER_SHA_PREFIX))}
    return C, D


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def ucf(hw, A, B):
    return {c: cell(hw, A, B, c) for c in ("U", "C", "F")}


def fams(q3, A, B):
    return {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for f, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}


def floor_note(pooled):
    return ("inside the reader floor of row 106 (|pooled| %.4f <= %.4f)" if abs(pooled) <= READER_FLOOR else "outside the reader floor of row 106 (|pooled| %.4f > %.4f)") % (abs(pooled), READER_FLOOR)


def line(v, reader=False):
    s = "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))
    return s + ("  | U: " + floor_note(v["U"]["pooled"])) if reader else s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C, D = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD59", "tier": TIER_LINE, "reader_floor_U": READER_FLOOR, "controls": C, "controls_all_pass": bool(ok), "names": None, "levels": None}
    print(TIER_LINE)
    print("== controls (r1 to r9)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:760]))
    if ok:
        G = pair_groups()
        lv = hw["levels"]
        levels = {}
        for fk, arms in FORMS:
            for role, arm in arms.items():
                if arm in lv:
                    levels[arm] = {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")}
        cells = {g: {fk: ucf(hw, A, B) for fk, (A, B) in d.items()} for g, d in G.items()}
        names, floor = {}, {}
        for fk, v in cells["CY"].items():
            d = v["U"]
            nm = "FASTER-CYCLE-CUTS-UNSAFE" if (all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0) else ("FASTER-CYCLE-RAISES-UNSAFE" if d["interval"][0] > 0 else "NO-U-CHANGE")
            names["cycle_" + fk] = "RULE1S10K-%s-%s" % (fk, nm)
            floor["cycle_" + fk] = d["pooled"]
        for fk in RULE_FORMS:
            d = cells["RU_1s"][fk]["U"]
            names["rule1s_" + fk] = "RULE1S10K-%s-%s" % (fk, "RULE-CUTS-UNSAFE-AT-1S" if (all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0) else "RULE-NOT-SEPARATED-AT-1S")
            floor["rule1s_" + fk] = d["pooled"]
            d = cells["GB"][fk]["U"]
            nm = "BELOW-BANKED-2S" if (all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0) else ("ABOVE-BANKED-2S" if d["interval"][0] > 0 else "NOT-SEPARATED-FROM-BANKED-2S")
            names["banked_" + fk] = "RULE1S10K-%s-%s" % (fk, nm)
            floor["banked_" + fk] = d["pooled"]
        for fk, v in cells["RB"].items():
            d = v["U"]
            nm = "BACKWARD-COSTS-AT-2S" if d["interval"][0] > 0 else ("BACKWARD-HELPS-AT-2S" if d["interval"][1] < 0 else "BACKWARD-NEUTRAL-AT-2S")
            names["readout2s_" + fk] = "RULE1S10K-%s-%s" % (fk, nm)
            floor["readout2s_" + fk] = d["pooled"]
        floor_notes = {k: floor_note(floor[k]) for k in names if k.split("_", 1)[1] == "READER"}
        fam = {g: {fk: fams(q3, A, B) for fk, (A, B) in d.items()} for g, d in G.items()}
        res.update(names=names, name_floor_notes=floor_notes, levels=levels, pairs={g: {fk: list(p) for fk, p in d.items()} for g, d in G.items()}, cells=cells, families=fam)
        print("== names")
        for k, v in names.items():
            print("   %-18s %s%s" % (k, v, ("   [" + floor_notes[k] + "]") if k in floor_notes else ""))
        print("== levels (per model seed: U, C, F; 562 scored windows per seed for U, 642 lead windows for C and F)")
        for arm, v in levels.items():
            print("   %-12s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== differences: A minus B, pooled [interval] and per model seed (the reader floor remark is on the U cell of every READER difference; C and F carry no floor)")
        for g, d in cells.items():
            for fk, v in d.items():
                A, B = G[g][fk]
                print("   %-8s %-7s %s - %s : %s" % (g, fk, A, B, line(v, reader=(fk == "READER"))))
        print("== families (four families per pair)")
        for g, d in fam.items():
            for fk, fm in d.items():
                for f, v in fm.items():
                    print("   %-8s %-7s %-34s D %+8.4f [%+.4f, %+.4f] %s %s%s" % (g, fk, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"], ("  (reader floor 0.0196 applies to U only)" if fk == "READER" else "")))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print(STRATEGIC_LINE)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
