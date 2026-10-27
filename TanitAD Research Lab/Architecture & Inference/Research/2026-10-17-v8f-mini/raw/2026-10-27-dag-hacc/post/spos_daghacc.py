"""STRATEGIC rows of Read A for the DAG-HACC arm: S-POS@30/50/80 against CTRV, within run, via the SAME path DAG_READ.json used (dag_read.RealBackend.analyze_t1 -> t1hd_read.run_one),
arm swapped. Control: the same driver run on DAG's own ticks + dump reproduces DAG_READ.json's S-POS rows for DAG exactly.
usage: spos_daghacc.py <out json> [--repro-dag]      (--repro-dag: run DAG s0/s1 (decisive) instead and compare to DAG_READ.json)"""
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
RS = "E:/Projects/TanitAD-artifacts/dagreread/reader_src"
PK = "E:/Projects/TanitAD-artifacts/daghacc"
RAW = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
out_p = sys.argv[1]
repro = "--repro-dag" in sys.argv
spec = importlib.util.spec_from_file_location("dag_read", f"{RS}/dag_read.py")
DR = importlib.util.module_from_spec(spec)
sys.modules["dag_read"] = DR
spec.loader.exec_module(DR)
T = DR.load_t1hd(f"{RAW}/2026-10-26-t1hdread/t1hd_read.py", TREE)
work = f"{PK}/pt_tmp/spos_work"
os.makedirs(work, exist_ok=True)
ns = T.parse(["--mode", "real", "--tool-root", TREE, "--eval-cache", "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache", "--tracks", "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt",
              "--eval-grid", f"{RAW}/eval_grid_windows.json", "--eval-join", f"{RAW}/join_eval141.npz", "--banked-dir", ".", "--out", f"{work}/t1hd_unused.json"])
backend = DR.RealBackend(T, ns, 2000, work)
res = {"tier": "T1-kin (self-action OPEN loop)", "estimator": "t1hd_read STRATEGIC decisive rows (S-POS vs CTRV within run, ticks>=1, unpaired vs the CTRV floor; episode-cluster bootstrap B=2000)", "runs": {}}
runs = []
if repro:
    for s in (0, 1):
        runs.append((f"DAG_s{s}_decisive", f"E:/Projects/TanitAD-artifacts/v8f-dagger-t1/ticks/ticks_DAG_s{s}_m10_causal_heads.npz", f"E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", s))
else:
    for s in (0, 1):
        runs.append((f"DAGHACC_s{s}_decisive", f"{PK}/t1/ticks/ticks_DAGHACC_s{s}_m10_causal_heads.npz", f"{PK}/t1/dump_DAGHACC_s{s}_m10_causal_heads", s))
        runs.append((f"DAGHACCSTRICT_s{s}_beside", f"{PK}/t1/ticks/ticks_DAGHACCSTRICT_s{s}_m10_frozen_causal_heads.npz", f"{PK}/t1/dump_DAGHACCSTRICT_s{s}_m10_frozen_causal_heads", s))
for label, ticks, dump, s in runs:
    try:
        pack = backend.analyze_t1({"ticks": ticks, "dump_dir": dump, "k7": {"mode": "x1"}}, label, s)
        res["runs"][label] = {"S-POS": pack.extra[SPOS_KEY], "controls": {k: [bool(v[0]), v[1]] for k, v in pack.controls.items()},
                              "crawl_fraction_ticks_ge1": pack.extra["crawl_fraction_ticks_ge1"], "off_route": pack.extra["off_route"], "tick0_f1": pack.tick0_f1, "goal": pack.goal}
        print(label, {k: (round(v["delta"], 3), v["separated"]) for k, v in pack.extra[SPOS_KEY].items()}, "controls", {k: v[0] for k, v in pack.controls.items()}, flush=True)
    except Exception as e:                       # a row that cannot be computed is reported with its reason
        res["runs"][label] = {"status": "NOT COMPUTED", "reason": f"{type(e).__name__}: {e}"}
        print(label, "NOT COMPUTED", type(e).__name__, str(e)[:300], flush=True)
if repro:
    dr = json.load(open("E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json", encoding="utf-8"))
    rep = {}
    for s in (0, 1):
        banked = dr["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"]["DAG"][SPOS_KEY]
        mine = res["runs"][f"DAG_s{s}_decisive"]["S-POS"]
        rep[f"s{s}"] = {k: {"banked": [banked[k][x] for x in ("delta", "lo", "hi")], "mine": [mine[k][x] for x in ("delta", "lo", "hi")],
                            "max_abs_diff": max(abs(banked[k][x] - mine[k][x]) for x in ("delta", "lo", "hi"))} for k in banked}
    res["reproduces_DAG_READ"] = rep
    res["PASS"] = bool(all(v["max_abs_diff"] <= 1e-9 for s in rep.values() for v in s.values()))
    print("REPRO PASS", res["PASS"])
json.dump(res, open(out_p, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("ZZSPOS-DONE", out_p)
