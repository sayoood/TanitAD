"""t1blind_read: the READ of PREREG_T1_BLIND.md (T1-kin on eval-141, scene `reindexed`). Arms BLIND1_s{0,1} (C0 with every frame's scene tokens := the pinned mean). Comparators (sha256-checked): the banked RIC0_s{0,1} arrays and the
P10 arrays inside every array file. The instrument is the LF / A-series reader's, loaded BY FILE PATH (lf_read.py -> llr_read.py -> the RECOVER-NEXT reader), not copied: paired episode-cluster bootstrap B 2,000 seed 0
(gapdecomp.decompose), the 19 rows in four families (+ the PRIMARY row ADE6 and the crawl row M1). STRATEGIC: PAUSED (PI ruling, PI_DECISION_QUEUE ITEM 25): rows listed beside, never decisive.
REFUSES to read any row unless raw/GATES.json (G-ID, G-MEAN, G-EGO at both seeds) is PASS.
Outcome rules = the card's section 5, made operational here (the card does not say how to apply "At both seeds" to its disjunction; this reader pins the reading BEFORE any BLIND1 window is scored, and reports the alternative readings beside):
  per seed s:  ADE_s = (ADE6 B separated-worse in the pinned rows' direction AND the ADE6 B point > F)   F = 0.12074 m;   M1_s = (M1 B separated, either direction: CI excludes 0)
  VISION-USED   iff (ADE_s or M1_s) at BOTH seeds            [the card: "At both seeds, ADE6 B is separated-worse with a point above F, or M1 B is separated in either direction"]
  VISION-UNUSED iff at BOTH seeds: ADE6 B not separated in either direction AND M1 B not separated AND |ADE6 B point| < F
  MIXED         otherwise.
Beside (decide nothing): the same call under the stricter reading "ADE_s at both seeds, or M1_s at both seeds", and under "at least one seed".
usage: t1blind_read.py --out-root F:/TanitAD-scratch/motion_c/t1_blind/out --work-dir F:/TanitAD-scratch/motion_c/t1_blind/work --raw <dir>/raw [--out-dir <dir>] [--workers 2]   (CUDA_VISIBLE_DEVICES=-1)"""
import os
import sys
import json
import time
import argparse
import importlib.util
import concurrent.futures as cf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
LF_PATH = f"{ART}/dagland/motion_c/lf_read/code/lf_read.py"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SEEDS = (0, 1)
F_M = 0.12074                       # the training-run spread of M2b on ADE6 (D-V8F-LORA-FPRIME-1)
ADE, M1 = "ADE6", "M1 crawl share"
BANK_OUT = "F:/TanitAD-scratch/motion_c/ri/out"
C0_CKPT_SHA = {0: "bc3f926b7d5bc034ebb7043e5ef186a9dcad29ef404341fcd0493c7a22ca9fdf", 1: "43a303980f063a495f4d745e66efb870cf7f2f9c25acc0a7998562fb4978be93"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sep_any(r):
    return bool(r["lo"] > 0 or r["hi"] < 0)


def seed_conditions(rows, SR, GD, F=F_M):
    """The card's section 5 conditions for ONE seed from the 19 rows of B = BLIND1 - RIC0."""
    a, m = rows[ADE], rows[M1]
    worse = bool(SR.worse(GD, ADE, a))
    c = {"ade6_delta": a["delta"], "ade6_lo": a["lo"], "ade6_hi": a["hi"], "ade6_separated_worse": worse, "ade6_separated_any": sep_any(a), "ade6_point_above_F": bool(a["delta"] > F),
         "ade6_abs_point_below_F": bool(abs(a["delta"]) < F), "m1_delta": m["delta"], "m1_lo": m["lo"], "m1_hi": m["hi"], "m1_separated_any": sep_any(m),
         "m1_separated_better": bool(SR.better(GD, M1, m)), "m1_separated_worse": bool(SR.worse(GD, M1, m))}
    c["ADE_cond"] = bool(worse and c["ade6_point_above_F"])
    c["M1_cond"] = c["m1_separated_any"]
    c["USED_cond"] = bool(c["ADE_cond"] or c["M1_cond"])
    c["UNUSED_cond"] = bool((not c["ade6_separated_any"]) and (not c["m1_separated_any"]) and c["ade6_abs_point_below_F"])
    return c


def outcome(conds):
    """``conds`` = {seed: seed_conditions}. -> (label, reading beside)."""
    s = [conds[k] for k in sorted(conds)]
    used = all(c["USED_cond"] for c in s)
    unused = all(c["UNUSED_cond"] for c in s)
    label = "VISION-USED" if used else ("VISION-UNUSED" if unused else "MIXED")
    strict = all(c["ADE_cond"] for c in s) or all(c["M1_cond"] for c in s)
    anyone = any(c["USED_cond"] for c in s)
    return label, {"primary_reading": "VISION-USED iff (ADE6 cond or M1 cond) at BOTH seeds", "strict_reading_ADE_both_or_M1_both": "VISION-USED" if strict else ("VISION-UNUSED" if unused else "MIXED"),
                   "at_least_one_seed_reading": "a USED condition holds at >= 1 seed" if anyone else "no USED condition at either seed"}


def status_of(SR, GD, lab, r):
    return "better" if SR.better(GD, lab, r) else ("worse" if SR.worse(GD, lab, r) else ("separated" if sep_any(r) else "n.s."))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gates", default="")
    a = ap.parse_args(argv)
    gpath = a.gates or f"{a.raw}/GATES.json"
    if not os.path.isfile(gpath) or not json.load(open(gpath)).get("PASS"):
        print(f"REFUSED: {gpath} is absent or not PASS (card section 3: each gate must pass before any row is read)")
        return 3
    gates = json.load(open(gpath))
    LF = _load("lf_read_for_t1blind", LF_PATH)
    LF.init()
    rr, LR, SR, GD = LF.rr, LF.LR, LF.SR, LF.GD
    rr.PK = HERE
    rr.WORK = f"{a.work_dir}/rig"
    arr_dir = f"{a.raw}/arrays"
    t0 = time.time()
    res = {"card": "PREREG_T1_BLIND.md", "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2); arm BLIND1 = C0 with every frame's scene tokens := the pinned mean",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose) over 141 episodes / 2,399 windows; answers another draw of EPISODES only; BLIND1 is a deterministic function of C0 (no training seed in B); the seed question is answered by reading both seeds",
           "F_m": F_M, "strategic_family": "PAUSED (PI ruling, PI_DECISION_QUEUE ITEM 25): rows listed beside, never decisive", "gates": gates, "gates_file": gpath, "reader_sha256": LF.sha256_file(__file__), "lf_read_sha256": LF.sha256_file(LF_PATH),
           "controls": {}, "seeds": {}}
    base, ctl = {}, {}
    for s in SEEDS:
        z, c = LF.reused_arrays(s)
        base[s] = z["RIC0"]
        ctl.update({k: v for k, v in c.items() if k.startswith("RIC0_s")})
    res["controls"]["reused_arrays"] = {"PASS": all(v["PASS"] for v in ctl.values()), "files": ctl}
    done = {}
    for s in SEEDS:
        tag = f"BLIND1_s{s}"
        d = json.load(open(f"{a.out_root}/done_{tag}.txt"))
        ric = json.load(open(f"{BANK_OUT}/done_RIC0_s{s}.txt"))
        done[tag] = {"flags_ok": d["flags"] == ric["flags"], "ckpt_same_as_RIC0": d["ckpt_sha256"] == ric["ckpt_sha256"] == C0_CKPT_SHA[s], "tool_same_as_RIC0": d["tool_sha256"] == ric["tool_sha256"], "threads_same": d["threads"] == ric["threads"],
                     "done_marker": d, "ric0_marker": ric}
        done[tag]["PASS"] = bool(done[tag]["flags_ok"] and done[tag]["ckpt_same_as_RIC0"] and done[tag]["tool_same_as_RIC0"])
    res["controls"]["rollout_markers"] = done
    if not (res["controls"]["reused_arrays"]["PASS"] and all(v["PASS"] for v in done.values())):
        res["refused"] = "a reused array differs from its pinned sha256, or a rollout marker differs from RIC0's (flags / ckpt / tool)"
        json.dump(res, open(f"{a.raw}/READ_T1BLIND.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print("REFUSED:", res["refused"])
        return 3
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(rr.extract, f"BLIND1_s{s}", s, a.out_root, arr_dir) for s in SEEDS]
        for f in futs:
            f.result()
    conds = {}
    for s in SEEDS:
        tag = f"BLIND1_s{s}"
        ar = {"BL": rr.read_arrays(f"{arr_dir}/{tag}.npz"), "RIC0": base[s]}
        eid = ar["RIC0"]["eid"]
        assert np.array_equal(ar["BL"]["eid"], eid) and np.array_equal(ar["BL"]["ws"], ar["RIC0"]["ws"]) and np.array_equal(ar["BL"]["v0"], ar["RIC0"]["v0"]), f"{tag}: windows differ from RIC0's"
        cls, _ = GD.classify(ar["RIC0"]["v0"], ar["RIC0"]["g"])
        t = np.load(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
        b = {}
        b["BLIND1_minus_RIC0"], b["M1_per_class_split"] = rr.rows_of(rr.Aof(ar["BL"]), rr.Aof(ar["RIC0"]), eid, cls, a.n_boot, M1)
        b["BLIND1_minus_P10"] = rr.rows_of(rr.Aof(ar["BL"]), rr.Bof(ar["BL"]), eid, cls, a.n_boot)[0]
        b["RIC0_minus_P10"] = rr.rows_of(rr.Aof(ar["RIC0"]), rr.Bof(ar["RIC0"]), eid, cls, a.n_boot)[0]
        ft = {c: LR.family_table(b[c]) for c in ("BLIND1_minus_RIC0", "BLIND1_minus_P10", "RIC0_minus_P10")}
        B = b["BLIND1_minus_RIC0"]
        cond = seed_conditions(B, SR, GD)
        conds[s] = cond
        strat = sorted(lab for lab in B if SR.FAMILY[lab] == "STRATEGIC")
        status = {lab: status_of(SR, GD, lab, B[lab]) for lab in B}
        status_p10 = {lab: status_of(SR, GD, lab, b["BLIND1_minus_P10"][lab]) for lab in B}
        status_ric0_p10 = {lab: status_of(SR, GD, lab, b["RIC0_minus_P10"][lab]) for lab in B}
        res["seeds"][f"s{s}"] = {"tag": tag, "conditions": cond, "rows_BLIND1_minus_RIC0": B, "rows_BLIND1_minus_P10": b["BLIND1_minus_P10"], "rows_RIC0_minus_P10": b["RIC0_minus_P10"],
                                 "status_BLIND1_minus_RIC0": status, "status_BLIND1_minus_P10": status_p10, "status_RIC0_minus_P10": status_ric0_p10,
                                 "separated_better_rows": sorted(l for l, v in status.items() if v == "better"), "separated_worse_rows": sorted(l for l, v in status.items() if v == "worse"),
                                 "strategic_rows_beside_PAUSED": {lab: {"B": B[lab], "status": status[lab]} for lab in strat}, "family_tables": ft, "M1_per_class_split": b["M1_per_class_split"],
                                 "ticks_n": int(len(t[t.files[0]])), "family_of": {lab: SR.FAMILY[lab] for lab in B}}
        print(f"s{s}: ADE6 B {B[ADE]['delta']:+.4f} [{B[ADE]['lo']:+.4f}, {B[ADE]['hi']:+.4f}]  M1 B {B[M1]['delta']:+.4f} [{B[M1]['lo']:+.4f}, {B[M1]['hi']:+.4f}]  ADE_cond {cond['ADE_cond']} M1_cond {cond['M1_cond']} UNUSED_cond {cond['UNUSED_cond']}", flush=True)
    label, beside = outcome(conds)
    res["outcome"] = label
    res["outcome_reading_beside"] = beside
    res["outcome_rule_reading"] = "'At both seeds' applies to the whole disjunction: (ADE6 B separated-worse with point > F) or (M1 B separated either direction) must hold at EACH seed; pinned in this reader before any BLIND1 window was scored"
    res["next_committed_by_card"] = {
        "VISION-USED": "RENDER-GAP-BINDS (read with T2's BLIND ~ C0): the LoRA dominance card gains a gate before LC's T2 read counts (L-BLIND at T2 separated from L on ADE6 or M1); a render-robustness lever gets its own card (adapt the encoder to NuRec renders, or render-domain augmentation), aimed at T2.",
        "VISION-UNUSED": "L stays the route; its card gains a T1-kin BLIND gate (L-BLIND - L separated on ADE6 or M1 before any L number counts); the theory note records that the shrunken a0 gain is not offset by vision.",
        "MIXED": "the BLIND gate enters the LoRA card at both tiers; the render-robustness card is drafted, not run."}[label]
    res["wall_s"] = time.time() - t0
    os.makedirs(a.raw, exist_ok=True)
    json.dump(res, open(f"{a.raw}/READ_T1BLIND.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    print("OUTCOME T1-BLIND:", label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
