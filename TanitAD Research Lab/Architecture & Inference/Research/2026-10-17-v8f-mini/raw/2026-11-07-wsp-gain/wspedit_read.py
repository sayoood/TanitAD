"""wspedit_read: READ 2 of PREREG_WSP_GAIN.md section 3 (T1-kin on eval-141, scene `reindexed`). The instrument is the LF / A-series reader's, loaded BY FILE PATH (lf_read.py -> llr_read.py -> the RECOVER-NEXT reader), not copied:
paired episode-cluster bootstrap B 2,000 seed 0 (gapdecomp.decompose), the 19 rows in four families, M1 = "M1 crawl share", share_closed = -(WSP - RIC0)[M1] / (RIC0 - P10)[M1].
Arms: RIWSP{05,10}_s{0,1} (the edited C0). Comparators (sha256-checked): RIC0_s{0,1} (C0 on `reindexed`, same flags) and the P10 arrays inside every array file.
Outcome rules = the card's section 4 table made operational in the card's section 6 PIN entry (written before any edited-model score):
  per (dose): HARMS if, at EITHER seed, a non-M1 non-STRATEGIC row is separated-worse in WSP - RIC0, or separated-worse in WSP - P10 while not separated-worse in RIC0 - P10 (a NEW separated-worse row);
              else MOVES-M1 if M1 of WSP - RIC0 is separated-better at BOTH seeds; else NO-EFFECT (a note says when it was separated-better at exactly one seed).
  STRATEGIC: PAUSED (PI ruling, PI_DECISION_QUEUE ITEM 25): its rows are listed beside, never decisive, never a pass.
usage: wspedit_read.py --out-root F:/TanitAD-scratch/motion_c/wsp_edit/out --work-dir ... --raw .../raw [--workers 4]   (CUDA_VISIBLE_DEVICES=-1)"""
import os
import sys
import json
import time
import argparse
import importlib.util
import concurrent.futures as cf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
LF_PATH = f"{ART}/dagland/motion_c/lf_read/code/lf_read.py"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SEEDS = (0, 1)
DOSES = {"05": 0.5, "10": 1.0}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def status(SR, GD, rows):
    better = [lab for lab, r in rows.items() if SR.better(GD, lab, r)]
    worse = [lab for lab, r in rows.items() if SR.worse(GD, lab, r)]
    return better, worse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True); ap.add_argument("--work-dir", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    LF = _load("lf_read_for_wsp", LF_PATH)
    LF.init()
    rr, LR, SR, GD, M1 = LF.rr, LF.LR, LF.SR, LF.GD, LF.M1
    rr.PK = HERE
    rr.WORK = f"{a.work_dir}/rig"
    arr_dir = f"{a.raw}/arrays"
    os.makedirs(a.raw, exist_ok=True)
    t0 = time.time()
    res = {"card": "PREREG_WSP_GAIN.md", "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only (the edit is a deterministic function of C0, so no training seed is in a WSP - RIC0 contrast)",
           "strategic_family": "PAUSED (PI ruling, PI_DECISION_QUEUE ITEM 25): rows listed beside, never decisive", "reader_sha256": LF.sha256_file(__file__), "lf_read_sha256": LF.sha256_file(LF_PATH),
           "doses_read": list(DOSES), "disclosure": "Two doses (alpha 0.5, 1.0) and two seeds were read; the card's prediction names alpha 1.0; a counted outcome names its dose and says two were read.", "controls": {}, "doses": {}}
    base, ctl = {}, {}
    for s in SEEDS:
        z, c = LF.reused_arrays(s)
        base[s] = {"C0": z["C0"], "RIC0": z["RIC0"]}
        ctl.update({k: v for k, v in c.items() if k.startswith(("C0_", "RIC0_"))})
    res["controls"]["reused_arrays"] = {"PASS": all(v["PASS"] for v in ctl.values()), "files": ctl}
    if not res["controls"]["reused_arrays"]["PASS"]:
        res["refused"] = "a reused array differs from its pinned sha256"
        json.dump(res, open(f"{a.raw}/READ2.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print("REFUSED:", res["refused"]); return 3
    jobs = [(f"RIWSP{d}_s{s}", s) for d in DOSES for s in SEEDS]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(rr.extract, tag, s, a.out_root, arr_dir): tag for tag, s in jobs}
        for f in cf.as_completed(futs):
            f.result()
    for d, alpha in DOSES.items():
        out = {"alpha": alpha, "seeds": {}}
        for s in SEEDS:
            tag = f"RIWSP{d}_s{s}"
            ar = {"WSP": rr.read_arrays(f"{arr_dir}/{tag}.npz"), **base[s]}
            eid = ar["C0"]["eid"]
            assert np.array_equal(ar["WSP"]["eid"], eid) and np.array_equal(ar["RIC0"]["eid"], eid), f"{tag}: windows differ from C0's"
            cls, _ = GD.classify(ar["C0"]["v0"], ar["C0"]["g"])
            t = np.load(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            b = {}
            b["WSP_minus_RIC0"], b["M1_per_class_split"] = rr.rows_of(rr.Aof(ar["WSP"]), rr.Aof(ar["RIC0"]), eid, cls, a.n_boot, M1)
            b["RIC0_minus_P10"] = rr.rows_of(rr.Aof(ar["RIC0"]), rr.Bof(ar["RIC0"]), eid, cls, a.n_boot)[0]
            b["WSP_minus_P10"] = rr.rows_of(rr.Aof(ar["WSP"]), rr.Bof(ar["WSP"]), eid, cls, a.n_boot)[0]
            cell = rr.m1_cell(b["WSP_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])
            ft = {c: LR.family_table(b[c]) for c in ("WSP_minus_RIC0", "WSP_minus_P10", "RIC0_minus_P10")}
            strat = {lab for lab in b["WSP_minus_RIC0"] if SR.FAMILY[lab] == "STRATEGIC"}
            bw_x, ww_x = status(SR, GD, b["WSP_minus_RIC0"])
            _, ww_p = status(SR, GD, b["WSP_minus_P10"])
            _, ww_r = status(SR, GD, b["RIC0_minus_P10"])
            drop = {M1} | strat
            new_worse = sorted((set(ww_x) - drop) | ((set(ww_p) - set(ww_r)) - drop))
            out["seeds"][f"s{s}"] = {"tag": tag, "m1_cell": cell, "M1_WSP_minus_RIC0": b["WSP_minus_RIC0"][M1], "M1_RIC0_minus_P10": b["RIC0_minus_P10"][M1], "M1_WSP_minus_P10": b["WSP_minus_P10"][M1],
                                     "share_closed": cell["share_closed"], "new_separated_worse_rows": new_worse, "separated_better_rows_WSP_minus_RIC0": [x for x in bw_x],
                                     "separated_worse_rows_WSP_minus_RIC0_all": ww_x, "n_worse_vs_P10": {"WSP": len(set(ww_p) - drop), "RIC0": len(set(ww_r) - drop)},
                                     "strategic_rows_beside_PAUSED": {lab: b["WSP_minus_RIC0"][lab] for lab in sorted(strat)}, "family_tables": ft, "rows_WSP_minus_RIC0": b["WSP_minus_RIC0"],
                                     "rows_WSP_minus_P10": b["WSP_minus_P10"], "rows_RIC0_minus_P10": b["RIC0_minus_P10"], "M1_per_class_split": b["M1_per_class_split"],
                                     "ticks_n": int(len(t[t.files[0]])), "max_abs_plan_dev_vs_RIC0_m": float(max(np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) for u, v in zip(rr.Bof(ar["WSP"]), rr.Bof(ar["RIC0"]))))}
            print(f"alpha {alpha} s{s}: M1 WSP-RIC0 {b['WSP_minus_RIC0'][M1]['delta']:+.4f} [{b['WSP_minus_RIC0'][M1]['lo']:+.4f}, {b['WSP_minus_RIC0'][M1]['hi']:+.4f}] better {cell['better']} share {cell['share_closed']} | new worse {new_worse}", flush=True)
        harms = any(v["new_separated_worse_rows"] for v in out["seeds"].values())
        n_better = sum(bool(v["m1_cell"]["better"]) for v in out["seeds"].values())
        out["n_seeds_M1_separated_better"] = n_better
        out["label"] = "HARMS" if harms else ("MOVES-M1" if n_better == 2 else "NO-EFFECT")
        out["note"] = ("M1 separated-better at exactly one seed" if (n_better == 1 and not harms) else "")
        res["doses"][d] = out
    res["outcome_alpha_1p0"] = res["doses"]["10"]["label"]
    res["outcome_alpha_0p5"] = res["doses"]["05"]["label"]
    res["wall_s"] = time.time() - t0
    json.dump(res, open(f"{a.raw}/READ2.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    md = ["# READ 2 (WSP-GAIN edit, T1-kin reindexed, eval-141)", "", "| dose | seed | M1 WSP-RIC0 [CI] | share_closed | new separated-worse rows | label |", "|---|---|---|---|---|---|"]
    for d, o in res["doses"].items():
        for k, v in o["seeds"].items():
            m = v["M1_WSP_minus_RIC0"]
            md.append(f"| {o['alpha']} | {k} | {m['delta']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | {v['share_closed']} | {', '.join(v['new_separated_worse_rows']) or 'none'} | {o['label']} |")
    md += ["", "STRATEGIC family: PAUSED (PI item 25). Estimator: paired episode-cluster bootstrap B 2,000 seed 0. Tier T1-kin."]
    open(f"{a.raw}/RESULTS_READ2.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("OUTCOME alpha 1.0:", res["outcome_alpha_1p0"], "| alpha 0.5:", res["outcome_alpha_0p5"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
