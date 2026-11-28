#!/usr/bin/env python3
"""The refcv7-50,400 LEGAL NavSim row, summarised (TANITAD VENV).  BAR-R8-N4's baseline.

    python code/vmaxoff_legal_summary7.py --root raw/milestones/step50400

Reads ``<root>/vmaxoff_legal/scores_{navtest,navhard}`` (R7_VMAXOFF, the LEGAL-input arm: bare command,
no route checkpoint, no turn distance, the UNKNOWN max-speed row) and, when they read PASS, R7_A1 / R7_A1_s1
from the runner's ``<root>/scores_{navtest,navhard}`` (the MAP-speed-limit arm: PRIVILEGED on NavSim, not a
legal baseline). It re-uses the suite's own statistics: ``parse_navtest7.py`` helpers + the settled
``taniteval/adapters/navsim_ci.py`` for navtest, and ``parse7.py`` itself (run on a STAGED copy) for navhard.

⛔ NAMING TRAP, MEASURED 2026-10-04: ``parse_navtest7.py`` calls its ``--arm-csv`` argument ``R7_A1`` in every
output key whatever CSV it is given -- feeding it the VMAXOFF CSV would publish the LEGAL arm under the
PRIVILEGED arm's name. This script therefore never uses that argument for VMAXOFF: every key is the arm's
own name, and the file header repeats which input set each arm had.

Writes ``<root>/vmaxoff_legal/summary_navtest_vmaxoff_legal.json`` and ``summary_navhard_vmaxoff_legal.json``.
A CSV beside a count guard that does not read PASS is REFUSED (never a partial result).
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parse_navtest7 as PN  # noqa: E402  (read, FLOORS, INPUTS, TERMS; importing runs nothing)

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
INPUT_SETS = {
    "R7_VMAXOFF": "LEGAL: frames + ego pose/velocity at t0-1.0..t0 + BARE driving_command; max speed = UNKNOWN row "
                  "(v_max_valid=0); no route checkpoint, no turn distance",
    "R7_A1": "PRIVILEGED on NavSim: as LEGAL but max speed = the MAP posted limit (NavSim AgentInput has no map)",
    "R7_A1_s1": "PRIVILEGED on NavSim (map speed limit), inference seed 1",
}


def _status(csv_path: str) -> str:
    cnt = csv_path[:-4] + ".counts.json"
    try:
        return json.load(open(cnt, encoding="utf-8")).get("status", "NO_STATUS")
    except OSError:
        return "NO_COUNTS"


def _load_ci():
    spec = importlib.util.spec_from_file_location("navsim_ci7", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
    ci = importlib.util.module_from_spec(spec)
    sys.modules["navsim_ci7"] = ci
    spec.loader.exec_module(ci)
    return ci


def navtest(root: str, out_path: str) -> dict:
    ci = _load_ci()
    vdir = os.path.join(root, "vmaxoff_legal", "scores_navtest")
    arm_csv = {"R7_VMAXOFF": os.path.join(vdir, "r7s50400_R7_VMAXOFF", "r7s50400_R7_VMAXOFF.csv")}
    for a in ("R7_A1", "R7_A1_s1"):
        arm_csv[a] = os.path.join(root, "scores_navtest", f"r7s50400_{a}", f"r7s50400_{a}.csv")
    arms, refused = {}, {}
    for k, p in arm_csv.items():
        if not os.path.exists(p):
            refused[k] = "csv absent"
            continue
        st = _status(p)
        if st != "PASS":
            refused[k] = f"count guard {st}"
            continue
        arms[k] = PN.read(p)
    if "R7_VMAXOFF" not in arms:
        res = {"status": "UNAVAILABLE", "refused": refused}
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
        return res
    toks = sorted(arms["R7_VMAXOFF"].index)
    for k, v in PN.FLOORS.items():
        arms[k] = PN.read(v)
    for k, df in arms.items():
        miss = set(toks) - set(df.index)
        if miss:
            raise SystemExit(f"{k} lacks {len(miss)} of the VMAXOFF tokens -- not paired")
    tok2log = {t: r["log_name"] for t, r in
               json.load(gzip.open(PN.INPUTS, "rt", encoding="utf-8"))["tokens"].items()}
    clusters = [tok2log[t] for t in toks]
    out = {"_label": "refcv7-50400 LEGAL-input NavSim row (BAR-R8-N4 baseline)", "protocol": "PDMS_v1_navtest",
           "evidence": "MEASURED (official devkit PDMS, W3 harness; CSVs under raw/milestones/step50400/vmaxoff_legal)",
           "tier": "NavSim PDM scorer on the model's emitted plan (open-loop plan scoring; EVAL_DOCTRINE: not T1 closed loop)",
           "n_tokens": len(toks), "n_logs": len(set(clusters)), "input_sets": INPUT_SETS, "refused": refused,
           "estimator": {"name": ci.ESTIMATOR, "paired": ci.PAIRED_ESTIMATOR, "unit": ci.CLUSTER_UNIT,
                         "aggregation": ci.AGG_SINGLE_STAGE,
                         "answers": "another draw of LOGS? (blind to training and inference variance)"},
           "arms": {}, "pairs": {}}
    for k, df in arms.items():
        d = df.loc[toks]
        rec = {t: round(100 * float(d[c].mean()), 4) for t, c in PN.TERMS.items()}
        rec["zeroed_by_NC_or_DAC"] = int(((d[PN.TERMS["NC"]] == 0) | (d[PN.TERMS["DAC"]] == 0)).sum())
        rec["input_set"] = INPUT_SETS.get(k, "banked floor (W3, model-free or log replay)")
        rec["interval"] = ci.log_cluster_bootstrap(d["score"].to_numpy(dtype=float), clusters,
                                                   aggregation=ci.AGG_SINGLE_STAGE,
                                                   official_value=float(d["score"].mean()))
        out["arms"][k] = rec
    pairs = [("R7_VMAXOFF", y) for y in ("STOP", "CV", "HUMAN") if y in arms]
    pairs += [("R7_VMAXOFF", y) for y in ("R7_A1", "R7_A1_s1") if y in arms]
    pairs += [("R7_A1", "STOP")] if "R7_A1" in arms else []
    pairs += [("R7_A1", "R7_A1_s1")] if "R7_A1" in arms and "R7_A1_s1" in arms else []
    for x, y in pairs:
        ax = arms[x].loc[toks, "score"].to_numpy(dtype=float)
        ay = arms[y].loc[toks, "score"].to_numpy(dtype=float)
        dd = ax - ay
        out["pairs"][f"{x}__minus__{y}"] = {
            "delta_x100": round(100 * float(dd.mean()), 4), "wins": int((dd > 1e-12).sum()),
            "ties": int((np.abs(dd) <= 1e-12).sum()), "losses": int((dd < -1e-12).sum()),
            "interval": ci.paired_log_cluster_bootstrap(ax, ay, clusters, aggregation=ci.AGG_SINGLE_STAGE,
                                                        official_a=float(ax.mean()),
                                                        official_b=float(ay.mean()))}
    if "R7_A1" in arms and "R7_A1_s1" in arms:
        out["inference_seed_floor_x100"] = abs(out["pairs"]["R7_A1__minus__R7_A1_s1"]["delta_x100"])
        out["inference_seed_floor_note"] = ("|PDMS(A1) - PDMS(A1_s1)| on the full split; a separated paired interval "
                                            "answers 'another draw of logs?', NOT 'another inference draw?'")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    return out


def navhard(root: str, out_path: str) -> dict:
    vroot = os.path.join(root, "vmaxoff_legal")
    vdir = os.path.join(vroot, "scores_navhard")
    sfx = "__navhard_two_stage"
    stage = os.path.join(vroot, "_parse_stage_navhard")
    st_sc, st_br = os.path.join(stage, "scores"), os.path.join(stage, "bridge")
    for d in (st_sc, st_br):
        os.makedirs(d, exist_ok=True)

    def stage_arm(arm: str, src_scores: str, src_bridge: str) -> bool:
        csv = os.path.join(src_scores, f"score_{arm}{sfx}.csv")
        if not (os.path.exists(csv) and _status(csv) == "PASS"):
            return False
        for n in (f"score_{arm}{sfx}.csv", f"score_{arm}{sfx}.counts.json"):
            shutil.copyfile(os.path.join(src_scores, n), os.path.join(st_sc, n))
        wr = os.path.join(src_scores, f"score_{arm}{sfx}_wrapper")
        dst = os.path.join(st_sc, f"score_{arm}{sfx}_wrapper")
        os.makedirs(dst, exist_ok=True)
        fn = f"{arm}{sfx}_final_scores_frame.csv"
        shutil.copyfile(os.path.join(wr, fn), os.path.join(dst, fn))
        for n in (f"seam_{arm}.npz", f"seam_{arm}.manifest.json"):
            shutil.copyfile(os.path.join(src_bridge, n), os.path.join(st_br, n))
        return True

    have = {"R7_VMAXOFF": stage_arm("R7_VMAXOFF", vdir, os.path.join(vroot, "bridge_navhard"))}
    for a in ("R7_A1", "R7_A1_s1"):
        have[a] = stage_arm(a, os.path.join(root, "scores_navhard"), os.path.join(root, "bridge_navhard"))
    if not have["R7_VMAXOFF"]:
        res = {"status": "UNAVAILABLE", "staged": have}
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
        return res
    full = os.path.join(vroot, "summary_navhard_parse7_full.json")
    pkg = os.path.dirname(HERE)
    inputs = ("C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/"
              "navsim_agent_inputs.json")
    env = dict(os.environ, PYTHONPATH="C:/Users/Admin/ev7nav/stack;C:/Users/Admin/ev7nav/taniteval",
               TANITAD_REPO="C:/Users/Admin/ev7nav", PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, os.path.join(HERE, "parse7.py"), "--split", "navhard_two_stage", "--scores", st_sc,
                        "--floors", os.path.join(pkg, "raw", "floors", "navhard_two_stage"), "--bridge", st_br,
                        "--inputs", inputs, "--label", "refcv7-50400 LEGAL-input row (R7_VMAXOFF) vs R7_A1 (map speed)",
                        "--out", full, "--csv-suffix", sfx], capture_output=True, text=True, env=env)
    full_doc = json.load(open(full, encoding="utf-8")) if os.path.exists(full) else {}
    out = {"_label": "refcv7-50400 LEGAL-input NavSim navhard two-stage row (BAR-R8-N4 baseline)",
           "evidence": "MEASURED (official NavSim v2 two-stage devkit scorer; CSVs under raw/milestones/step50400/"
                       "vmaxoff_legal/scores_navhard)",
           "input_sets": INPUT_SETS, "staged_arms": have, "parse7_rc": r.returncode,
           "parse7_stderr_tail": r.stderr[-400:], "parse7_full": os.path.basename(full),
           "arms": {k: full_doc.get("arms", {}).get(k) for k in
                    ("R7_VMAXOFF", "R7_A1", "R7_A1_s1", "STOP_zero", "CV_official", "ECHO_ha0_ext")
                    if full_doc.get("arms", {}).get(k)},
           "intervals": {k: v for k, v in full_doc.get("intervals", {}).items()
                         if k in ("R7_VMAXOFF", "R7_A1", "R7_A1_s1", "STOP_zero", "CV_official", "ECHO_ha0_ext")},
           "pairs": {k: v for k, v in full_doc.get("pairs", {}).items() if "R7_VMAXOFF" in k},
           "paired_intervals": {k: v for k, v in full_doc.get("paired_intervals", {}).items()
                                if "R7_VMAXOFF" in k},
           "note": "VMAXOFF vs STOP/CV/ECHO/A1 pairs exist only where parse7.PAIRS lists them; the "
                   "VMAXOFF - STOP delta is computed below from the staged per-scene scores when absent."}
    # the pairs parse7 does not list for VMAXOFF (it lists A1 - VMAXOFF only): VMAXOFF vs the floors
    try:
        import pandas as pd
        ps = importlib.util.spec_from_file_location("e2_parse_scores7", os.path.join(
            pkg, "..", "2026-09-19-navsim-refcv4b-bridge", "code", "parse_scores.py"))
        pm = importlib.util.module_from_spec(ps)
        sys.modules["e2_parse_scores7"] = pm
        ps.loader.exec_module(pm)
        ci = _load_ci()
        doc = json.load(open(inputs, encoding="utf-8"))
        mapping = doc["reactive_all_mapping"]
        log_of = {t: r_["log_name"] for t, r_ in doc["tokens"].items()}
        clusters = [log_of[str(m[0])] for m in mapping]
        contrib = {}
        for arm, d in (("R7_VMAXOFF", st_sc), ("STOP_zero", os.path.join(pkg, "raw", "floors", "navhard_two_stage")),
                       ("CV_official", os.path.join(pkg, "raw", "floors", "navhard_two_stage")),
                       ("ECHO_ha0_ext", os.path.join(pkg, "raw", "floors", "navhard_two_stage")),
                       ("R7_A1", st_sc)):
            if arm == "R7_A1" and not have["R7_A1"]:
                continue
            tag = f"{arm}{sfx}"
            rows, _ = ci.rows_from_score_frame(os.path.join(d, f"score_{tag}_wrapper", f"{tag}_final_scores_frame.csv"))
            contrib[arm] = ci.two_stage_key_contributions(rows, mapping, "score")
        off = {k: (full_doc.get("arms", {}).get(k) or {}).get("official_two_stage_EPDMS") for k in contrib}
        extra = {}
        for y in [k for k in contrib if k != "R7_VMAXOFF"]:
            if isinstance(off.get("R7_VMAXOFF"), float) and isinstance(off.get(y), float):
                extra[f"R7_VMAXOFF__minus__{y}"] = ci.paired_log_cluster_bootstrap(
                    contrib["R7_VMAXOFF"], contrib[y], clusters, aggregation=ci.AGG_TWO_STAGE,
                    official_a=off["R7_VMAXOFF"], official_b=off[y])
        out["paired_intervals_vmaxoff_vs_each"] = extra
    except Exception as e:                                                   # noqa: BLE001
        out["paired_intervals_vmaxoff_vs_each"] = {"status": "ERROR", "reason": repr(e)[:400]}
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(HERE), "raw", "milestones", "step50400"))
    ap.add_argument("--splits", default="navtest,navhard")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    vroot = os.path.join(os.path.abspath(a.root), "vmaxoff_legal")
    for sk in [s for s in a.splits.split(",") if s]:
        outp = os.path.join(vroot, f"summary_{sk}_vmaxoff_legal.json")
        res = navtest(os.path.abspath(a.root), outp) if sk == "navtest" else navhard(os.path.abspath(a.root), outp)
        if res.get("status") == "UNAVAILABLE":
            print(f"{sk}: UNAVAILABLE {res}")
            continue
        print(f"{sk}: written {outp}")
        if sk == "navtest":
            for k, v in res["arms"].items():
                iv = v["interval"]
                print(f"  {k:12s} PDMS={v['PDMS']:.4f} NC={v['NC']:.2f} DAC={v['DAC']:.2f} EP={v['EP']:.2f} "
                      f"CI=[{iv.get('lo')}, {iv.get('hi')}] {iv.get('status')}")
            for k, v in res["pairs"].items():
                iv = v["interval"]
                print(f"  {k:30s} d={v['delta_x100']:+.4f} W/T/L={v['wins']}/{v['ties']}/{v['losses']} "
                      f"CI=[{iv.get('lo')}, {iv.get('hi')}] {iv.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
