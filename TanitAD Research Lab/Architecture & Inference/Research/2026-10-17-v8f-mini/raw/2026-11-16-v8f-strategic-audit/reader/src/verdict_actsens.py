"""verdict_actsens.py: card section 6 applied mechanically to the arm JSONs of read_actsens.sh. Tier T0 (oracle goal; a world-model diagnostic, never driving performance). Evidence: MEASURED inputs, rule-derived labels.
usage: python verdict_actsens.py [--arms-dir DIR] [--out verdict.json] [--md verdict.md] [--r35-ia-dir DIR]
Inputs: DIR/arm_{IDM_s0,IDM_s1,DIV_s0,DIV_s1,NULL_s2,NULL_s3}.json (arm_<name>.json); R35 IA-3 / BAND-MET references from the banked ladder_read results (arm_R35_s0/1.json).
Conventions (hwm_eval.contrast): delta = mean(A) - mean(B), lower is better. Deciding quantities, HWM-ORACLE, band 0-2 s: acc_mae (LONGITUDINAL), yaw_mae (LATERAL). STRATEGIC: reported in full (PI 2026-11-16, the pause ended); it is never part of a lever label and never a pass by itself.
Chassis k: IDM_s0 / DIV_s0 / NULL_s2 -> 0, IDM_s1 / DIV_s1 / NULL_s3 -> 1 (the arm JSON carries 'chassis').
Rules (card 4.2 / 4.4 / 6):
  Floor_k(q)  = max(F_T,k(q), F_I(q)); F_T,k = |delta(NULL_k - R35_k)| on the row; F_I = max over {lever arm k, R35_k} of |inference seed 1 - seed 0| on the row.
  BETTER(q)   = at BOTH chassis hi(vs_r35) < 0 and |delta| > Floor_k.   WORSE(q) = at EITHER chassis lo(vs_r35) > 0 and delta > Floor_k.
  MECH-MOVED  = a gain g in {lon, lat} such that at BOTH chassis the arm's L1 2.0 s IA-3 gain point is above R35's, the CIs are disjoint (arm lo > R35 hi) and point gap > |NULL - R35| on that cell.
  BAND-MET lost at a level (HARM) = R35 met the level (beats and healthy and not worse) and the arm does not, at either chassis.
  CEILING     = exists q with hi(vs_hold) < 0 at both chassis, and the other quantity is not WORSE.
  Order: HARM > CEILING > NARROWS > MECH-ONLY > NULL, and SEEDS-DISAGREE when the label differs between the chassis (label evaluated per chassis alone, 'both chassis required' rules then give the weaker).
A missing arm or NULL floor makes the lever INCOMPLETE (no label): the card requires both chassis and the floor. NULL also needs C2/C4 (the lever ran) which this script cannot see: pass --lever-ran IDM=1,DIV=1 only after C2/C4 hold."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.environ.get("STRATEGIC_CODE_DIR", os.path.dirname(os.path.abspath(__file__))))      # strategic_cells.py: the one source of the STRATEGIC text and n/a entries
import strategic_cells as SC

QS = {"acc_mae": "LONGITUDINAL|0-2s|acc_mae", "yaw_mae": "LATERAL|0-2s|yaw_mae"}
LEVERS = ("IDM", "DIV")
NULLS = {0: "NULL_s2", 1: "NULL_s3"}
NEXT = {
    "ACTSENS-CEILING": "The planner beats HOLD at T0 on this world model: this WM replaces R35 as planning WM of record; read full 6 s and 2-6 s rows at once and name any band that also beats HOLD; HWM-PLAN G4/G5 (predicted goal, head-only, box 2) on this WM next; level-2 image constraint amendment for 2-6 s.",
    "ACTSENS-NARROWS": "Lever acts in the right direction. Both levers NARROW: combined arm IDM + DIV (2 runs). Only IDM: per-channel design weight at L1 (gamma_c = 100/var_c = 566 / 1,783). Only DIV: cap 10 -> 30. Each carded by amendment before its windows are scored.",
    "ACTSENS-MECH-ONLY": "Response grew, plan did not improve: next zero-training on dev-box CPU: prior-regularised planner (run_hwm.py --lam > 0, amended dose) on this WM, then the level-2 image constraint.",
    "ACTSENS-NULL": "Lever trained and did nothing. Next box-2 training: a direct action-response objective (DIVRS counterfactual-action target, or a gain-matching term toward IA-3 gain 1.0). Carded next.",
    "ACTSENS-HARM": "Lever stays off. IDM next: gamma_L1 300. DIV next: ACTSENS-NULL's next step.",
}
STRICT = ["ACTSENS-HARM", "ACTSENS-NULL", "ACTSENS-MECH-ONLY", "ACTSENS-NARROWS", "ACTSENS-CEILING"]     # weaker to stronger: SEEDS-DISAGREE reports the less favourable


def cell(arm, key, slot, src="hwm"):
    return arm[src]["cells"][key][slot]


def floors(arms, lever, k):
    """Per quantity: Floor_k = max(F_T,k, F_I) with its parts.
    Card 4.4 subscripts the training floor by chassis (F_T,k) and NOT the inference floor: F_I = max over {arm, R35} of |seed 1 - seed 0| per row,
    so F_I is ONE value per row, the max over both chassis of the lever arm's and R35's replicate deltas (MM 2026-11-15, fixed before any ACTSENS arm was read;
    the per-chassis value is kept as F_I_k for the report). Both chassis must be present (label_lever checks that first)."""
    out = {}
    nl = arms.get(NULLS[k])
    for q, key in QS.items():
        ft = abs(cell(nl, key, "vs_r35")["delta"]) if nl else None
        fik = {}
        for j in (0, 1):
            a = arms[f"{lever}_s{j}"]
            fik[j] = max(abs(cell(a, key, "rep_arm")["delta"]), abs(cell(a, key, "rep_r35")["delta"]))
        fi = max(fik.values())
        out[q] = {"F_T": ft, "F_I": fi, "F_I_k": fik[k], "Floor": None if ft is None else max(ft, fi)}
    return out


def strategic_summary(arms):
    """The STRATEGIC section of the verdict, computed from the arm JSONs: per arm the report's block (rates with n, logged-path ceiling and rate) and the STRATEGIC cells, plus the n/a entries. Reads, never labels."""
    per = {}
    for n, a in sorted(arms.items()):
        blk = a.get("strategic") if isinstance(a.get("strategic"), dict) else None
        cells = {k: v for k, v in a.get("hwm", {}).get("cells", {}).items() if k.startswith("STRATEGIC|")}
        per[n] = {"state": blk.get("state") if blk else "not in this arm's read: the arm JSON carries no STRATEGIC block", "arms": blk.get("arms") if blk else None,
                  "logged_path": blk.get("logged_path") if blk else None, "cells": cells}
    return {"state": SC.STRATEGIC_STATE, "tier_stamp": SC.STRATEGIC_TIER_STAMP, "role": "reported beside the lever labels; not an input of any label", "not_computed": SC.STRATEGIC_NOT_COMPUTED, "per_arm": per}


def r35_ref(dirp, k):
    p = os.path.join(dirp, f"arm_R35_s{k}.json")
    j = json.load(open(p, encoding="utf-8"))
    return {"ia3": j["beside"]["ia3"], "passes": j["passes"]}


def label_lever(arms, refs, lever):
    ks = (0, 1)
    need = [f"{lever}_s{k}" for k in ks] + [NULLS[k] for k in ks]
    miss = [n for n in need if n not in arms]
    if miss:
        return {"label": "INCOMPLETE", "missing_arms": miss, "why": "both chassis and both NULL floors are required (card section 6)"}
    fl = {k: floors(arms, lever, k) for k in ks}
    per = {}
    for q, key in QS.items():
        d = {k: cell(arms[f"{lever}_s{k}"], key, "vs_r35") for k in ks}
        h = {k: cell(arms[f"{lever}_s{k}"], key, "vs_hold") for k in ks}
        per[q] = {"vs_r35": d, "vs_hold": h,
                  "BETTER": all(d[k]["hi"] < 0 and abs(d[k]["delta"]) > fl[k][q]["Floor"] for k in ks),
                  "WORSE": any(d[k]["lo"] > 0 and d[k]["delta"] > fl[k][q]["Floor"] for k in ks),
                  "BEATS_HOLD_both": all(h[k]["hi"] < 0 for k in ks), "BEATS_HOLD_k": {k: h[k]["hi"] < 0 for k in ks}}
    mech = {}
    for g in ("gain_lon", "gain_lat"):
        row = {}
        ok = True
        for k in ks:
            a = arms[f"{lever}_s{k}"]["ia"]["ia3"]["1|10"][g]; r = refs[k]["ia3"]["1|10"][g]; n = arms[NULLS[k]]["ia"]["ia3"]["1|10"][g]
            floor = abs(n["point"] - r["point"])
            c = a["point"] > r["point"] and a["lo"] > r["hi"] and (a["point"] - r["point"]) > floor
            row[k] = {"arm": a["point"], "arm_lo": a["lo"], "r35": r["point"], "r35_hi": r["hi"], "null_minus_r35_abs": floor, "moved": bool(c)}
            ok &= bool(c)
        row["MECH_MOVED"] = ok
        mech[g] = row
    mech_moved = any(v["MECH_MOVED"] for v in mech.values())
    lost = {}
    for k in ks:
        a = arms[f"{lever}_s{k}"]["ia"]["band_met"]
        for lv in ("1", "2", "3"):
            met = lambda p: bool(p["beats"]) and bool(p["healthy"]) and not bool(p["worse"])
            if met(refs[k]["passes"][lv]) and not met(a[lv]):
                lost[f"chassis{k}_L{lv}"] = {"r35": refs[k]["passes"][lv], "arm": a[lv]}
    better = any(per[q]["BETTER"] for q in QS)
    worse = any(per[q]["WORSE"] for q in QS)
    ceiling = any(per[q]["BEATS_HOLD_both"] and not per[o]["WORSE"] for q, o in (("acc_mae", "yaw_mae"), ("yaw_mae", "acc_mae")))
    return {"floors": fl, "per_quantity": per, "mech": mech, "MECH_MOVED": mech_moved, "band_met_lost": lost, "BETTER_any": better, "WORSE_any": worse, "CEILING": ceiling}


def outcome(better, worse, ceiling, mech_moved, lost, lever_ran=True):
    """One label from one set of flags. Order HARM > CEILING > NARROWS > MECH-ONLY > NULL."""
    if worse or lost:
        return "ACTSENS-HARM"
    if ceiling and not worse:
        return "ACTSENS-CEILING"
    if better and not worse:
        return "ACTSENS-NARROWS"
    if mech_moved:
        return "ACTSENS-MECH-ONLY"
    return "ACTSENS-NULL" if lever_ran else "UNDECIDED (lever did not run: C2/C4)"


def per_chassis_outcome(arms, refs, lever, k, L):
    """The same rule on ONE chassis alone, to find SEEDS-DISAGREE (a label that holds at one chassis only)."""
    f = L["floors"][k]; res = {}
    qs = {}
    for q in QS:
        d = L["per_quantity"][q]["vs_r35"][k]
        qs[q] = {"better": d["hi"] < 0 and abs(d["delta"]) > f[q]["Floor"], "worse": d["lo"] > 0 and d["delta"] > f[q]["Floor"], "hold": L["per_quantity"][q]["BEATS_HOLD_k"][k]}
    mech = any(L["mech"][g][k]["moved"] for g in L["mech"])
    lost = any(key.startswith(f"chassis{k}_") for key in L["band_met_lost"])
    ceiling = any(qs[q]["hold"] and not qs[o]["worse"] for q, o in (("acc_mae", "yaw_mae"), ("yaw_mae", "acc_mae")))
    return outcome(any(v["better"] for v in qs.values()), any(v["worse"] for v in qs.values()), ceiling, mech, lost)


def run(arms, refs, lever_ran=None):
    lever_ran = lever_ran or {}
    out = {"tier": "T0", "evidence": "MEASURED inputs; labels are rule-derived (card section 6)", "strategic": strategic_summary(arms), "levers": {}}
    labels = {}
    for lv in LEVERS:
        L = label_lever(arms, refs, lv)
        if "label" in L:
            out["levers"][lv] = L; labels[lv] = L["label"]; continue
        ran = lever_ran.get(lv, True)
        lab = outcome(L["BETTER_any"], L["WORSE_any"], L["CEILING"], L["MECH_MOVED"], bool(L["band_met_lost"]), ran)
        pc = {k: per_chassis_outcome(arms, refs, lv, k, L) for k in (0, 1)}
        L["per_chassis_label"] = pc
        if pc[0] != pc[1] and lab != "ACTSENS-HARM":      # HARM is an either-chassis rule (card section 6), so it is never a one-chassis disagreement
            weaker = min((pc[0], pc[1]), key=lambda x: STRICT.index(x) if x in STRICT else -1)
            L["label"] = "SEEDS-DISAGREE"; L["next_label_applied"] = weaker
        else:
            L["label"] = lab
            L["next_label_applied"] = lab
        L["next_step"] = NEXT.get(L["next_label_applied"], "none (undecided)")
        out["levers"][lv] = L; labels[lv] = L["label"]
    out["labels"] = labels
    done = [lv for lv in LEVERS if labels[lv] != "INCOMPLETE"]
    if len(done) == 2:
        same = {lv: out["levers"][lv]["next_label_applied"] for lv in LEVERS}
        if same["IDM"] == same["DIV"] and out["levers"]["IDM"]["label"] != "SEEDS-DISAGREE":
            mean = {}
            for lv in LEVERS:
                v = [cell(arms[f"{lv}_s{k}"], QS["acc_mae"], "vs_r35")["delta"] for k in (0, 1)]
                mean[lv] = -sum(v) / 2.0
            out["lever_of_record"] = {"rule": "larger mean -delta over both chassis on 0-2 s acc_mae", "mean_minus_delta": mean, "lever": max(mean, key=mean.get)}
    return out


def load(dirp):
    arms = {}
    for p in glob.glob(os.path.join(dirp, "arm_*.json")):
        n = os.path.basename(p)[4:-5]
        if n.split("_")[0] in ("IDM", "DIV", "NULL"):
            arms[n] = json.load(open(p, encoding="utf-8"))
    return arms


def md(out):
    L = ["# ACTSENS verdict (T0, oracle goal: a world-model diagnostic, never driving performance)", "", "| lever | label | next label applied | per chassis |", "|---|---|---|---|"]
    for lv, v in out["levers"].items():
        L.append(f"| {lv} | {v['label']} | {v.get('next_label_applied', '-')} | {v.get('per_chassis_label', v.get('missing_arms', '-'))} |")
    for lv, v in out["levers"].items():
        if "floors" in v:
            L += ["", f"## {lv}", ""]
            for q, p in v["per_quantity"].items():
                L.append(f"* {q}: BETTER {p['BETTER']}, WORSE {p['WORSE']}, beats HOLD at both chassis {p['BEATS_HOLD_both']}; floors " + ", ".join(f"c{k}: {v['floors'][k][q]['Floor']:.4f}" for k in (0, 1)))
            L.append(f"* MECH-MOVED {v['MECH_MOVED']}; BAND-MET lost {list(v['band_met_lost'])}")
            L.append(f"* next: {v.get('next_step')}")
    L += ["", f"## STRATEGIC: {out['strategic']['state']}", "", f"* tier: {out['strategic']['tier_stamp']}", f"* role: {out['strategic']['role']}"]
    for k, v in out["strategic"]["not_computed"].items():
        L.append(f"* {k}: {v}")
    for n, v in out["strategic"]["per_arm"].items():
        cs = v["cells"]
        if not cs:
            L.append(f"* {n}: {v['state']}")
            continue
        for k, c in sorted(cs.items()):
            for slot, r in sorted(c.items()):
                L.append(f"* {n} {k} {slot}: delta {r['delta']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n {r.get('n_windows')} episodes {r.get('n_episodes')} label {r.get('label')}")
    if "lever_of_record" in out:
        L += ["", f"Lever of record: {out['lever_of_record']['lever']} ({out['lever_of_record']['rule']})"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", default="F:/TanitAD-scratch/actsens/read/out"); ap.add_argument("--out"); ap.add_argument("--md")
    ap.add_argument("--r35-ia-dir", default="E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read/results")
    ap.add_argument("--lever-ran", default="IDM=1,DIV=1", help="set only after C2 / C4 hold (the lever ran); a lever with 0 cannot be labelled NULL")
    a = ap.parse_args()
    arms = load(a.arms_dir)
    refs = {k: r35_ref(a.r35_ia_dir, k) for k in (0, 1)}
    ran = {kv.split("=")[0]: kv.split("=")[1] == "1" for kv in a.lever_ran.split(",")}
    res = run(arms, refs, ran)
    res["arms_read"] = sorted(arms)
    s = json.dumps(res, indent=1, default=float)
    if a.out:
        if os.path.exists(a.out):
            sys.exit(f"REFUSED: {a.out} exists")
        open(a.out, "w", encoding="utf-8").write(s)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(md(res))
    print(json.dumps({"labels": res["labels"], "arms_read": res["arms_read"], "lever_of_record": res.get("lever_of_record")}, indent=1))
