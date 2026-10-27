#!/usr/bin/env python3
"""n0_ticks.py -- BESIDE rows of N0 from the TICK dumps (never decisive): STRATEGIC S-POS@30/50/80 vs CTRV (within run, unpaired) and the TACTICAL T-head macro-F1 (lat / lon, ticks >= 1, matched cells),
N0 (DAG nav-logged) vs DAG (nav-deadreckon), via dag_read.RealBackend.analyze_t1 (t1hd_read's own pipeline, REUSED).  Also re-derives DAG's own S-POS from DAG's banked ticks and compares it with DAG_READ.json
(a control that the tick path reproduces the banked strategic rows).  Output: N0_ticks_beside.json.
usage: n0_ticks.py --out <json>"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ART = "E:/Projects/TanitAD-artifacts"
N0 = f"{ART}/dag-orc-n0"
sys.path.insert(0, f"{ART}/dag2t1")
import dag2_read as D2  # noqa: E402

TT = f"{ART}/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SF = f"{ART}/sfresh_inputs"
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"


def main():
    out_p = sys.argv[sys.argv.index("--out") + 1]
    t0 = time.time()
    DR = D2.load_dag_read("C:/Users/Admin/dagread/dag_read.py")
    T = DR.load_t1hd(f"{TT}/{PK}/raw/2026-10-26-t1hdread/t1hd_read.py", TT)
    np = DR.np
    work = f"{N0}/_ticks_work"
    os.makedirs(work, exist_ok=True)
    ns = T.parse(["--mode", "real", "--tool-root", TT, "--eval-cache", f"{SF}/cache", "--tracks", f"{SF}/inputs/native_tracks.pt", "--eval-grid", f"{TT}/{PK}/raw/eval_grid_windows.json",
                  "--eval-join", f"{TT}/{PK}/raw/join_eval141.npz", "--banked-dir", ".", "--out", f"{work}/unused.json"])
    be = DR.RealBackend(T, ns, 2000, work)
    banked = json.load(open(f"{ART}/v8f-dagger-t1/read/DAG_READ.json", encoding="utf-8"))
    res = {"note": "BESIDE rows; S-POS is read against CTRV WITHIN a run (unpaired), the T-head rows are paired matched-cell macro-F1 at ticks >= 1; none of this decides NAVMOVE", "seeds": {}}
    for s in (0, 1):
        ent_n = {"ticks": f"{N0}/ticks/ticks_N0DAG_s{s}_m10_causal_heads_navlogged.npz", "dump_dir": f"{N0}/dump_N0DAG_s{s}_m10_causal_heads_navlogged", "k7": {"mode": "x1"}}
        ent_b = {"ticks": f"{ART}/v8f-dagger-t1/ticks/ticks_DAG_s{s}_m10_causal_heads.npz", "dump_dir": f"{ART}/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", "k7": {"mode": "x1"}}
        pn = be.analyze_t1(ent_n, f"N0_s{s}", s)
        print(f"s{s}: N0 pack done {time.time() - t0:.0f}s", flush=True)
        pb = be.analyze_t1(ent_b, f"DAGdr_s{s}", s)
        print(f"s{s}: DAG pack done {time.time() - t0:.0f}s", flush=True)
        d = {"controls_N0": {k: {"PASS": v[0], "detail": v[1]} for k, v in pn.controls.items()}, "controls_DAGdr": {k: {"PASS": v[0], "detail": v[1]} for k, v in pb.controls.items()}}
        sp_n, sp_b = pn.extra[SPOS_KEY], pb.extra[SPOS_KEY]
        sp_banked = banked["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"]["DAG"][SPOS_KEY]
        rc = {k: {x: abs(float(sp_b[k][x]) - float(sp_banked[k][x])) for x in ("delta", "lo", "hi")} for k in sp_banked}
        d["S-POS_vs_CTRV_within_run"] = {"N0_nav_logged": sp_n, "DAG_nav_deadreckon_rederived": sp_b, "DAG_banked_DAG_READ": sp_banked, "rederived_vs_banked_max_abs_diff": max(max(v.values()) for v in rc.values()),
                                         "note": "delta = arm minus CTRV position error (negative = arm closer to the logged path than CTRV), unpaired between runs; the paired N0 - DAG S-POS contrast is NOT computed (PREREG section 5)"}
        d["crawl_fraction_ticks_ge1"] = {"N0": pn.extra["crawl_fraction_ticks_ge1"], "DAGdr": pb.extra["crawl_fraction_ticks_ge1"]}
        d["off_route"] = {"N0": pn.extra["off_route"], "DAGdr": pb.extra["off_route"]}
        d["goal_macroF1"] = {"N0": pn.goal, "DAGdr": pb.goal}
        d["T_head_macroF1_N0_minus_DAGdr"] = {}
        for hd in ("lat", "lon"):
            for matched in (True, False):
                row, _ = DR.f1_pair_row(T, pn, pb, hd, 2000, matched=matched)
                d["T_head_macroF1_N0_minus_DAGdr"][f"{hd}_{'matched' if matched else 'own'}"] = {k: row[k] for k in ("delta", "lo", "hi", "separated", "better", "f1_a", "f1_b", "n_cells_a", "n_cells_b",
                                                                                                            "n_cells_matched_pairs") if k in row}
        res["seeds"][f"s{s}"] = d
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out_p, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", out_p, D2.sha256_file(out_p), flush=True)


if __name__ == "__main__":
    main()
