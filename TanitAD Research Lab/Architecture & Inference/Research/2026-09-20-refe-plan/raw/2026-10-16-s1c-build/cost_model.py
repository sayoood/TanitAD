"""S1c cost model from S1's OWN arm logs (MEASURED).  usage: cost_model.py <s1_read_cache/arms> <out.json>

S1's nine arms (400 updates; 8 micro-batches x 8 rows = 64 rows per update; accum 8) give, per arm:
  pre   = start event -> step0 eval event          (the held-out eval before any update)
  s_upd = mean of the `s_per_update` field of the 16 train rows (one update = 8 micro-batches)
  post  = last train row -> done event             (the final held-out eval, dumps, the saves)
  gap   = PREFLIGHT_PASS `at` -> start event       (the gate + data check between the preflight and the trainer; only arms whose PREFLIGHT_PASS.json is banked)
A micro-batch (8 rows) costs m_mb = s_upd / 8 (an S1c arm at accum 1 runs the SAME micro-batch shape).  The per-optimiser-step overhead x is NOT identifiable from a
single accumulation setting: x in [0, 0.5] s is bracketed (ESTIMATED; Adam on the scorer + head, ~10^7 parameters, is milliseconds on an A40).
  hours(N updates, accum a, E evals) = (N * (a * (s_upd - x) / 8 + x) + E * eval_s + save_s) / 3600
"""
import json
import os
import statistics as st
import sys
from datetime import datetime

root, out = sys.argv[1], sys.argv[2]


def T(s):
    return datetime.fromisoformat(s)


rows_by_arm = {}
for arm in sorted(os.listdir(root)):
    p = os.path.join(root, arm, "sft.jsonl")
    if os.path.exists(p):
        rows_by_arm[arm] = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
res = {"arms": {}}
for arm, rows in rows_by_arm.items():
    ev = {}
    for r in rows:
        ev.setdefault(r["event"], []).append(r)
    tr = ev["train"]
    start, done = ev["start"][0], ev["done"][0]
    step0 = [e for e in ev["eval"] if e["tag"] == "step0"][0]
    d = {"s_per_update_mean": st.mean(r["s_per_update"] for r in tr), "s_per_update_min": min(r["s_per_update"] for r in tr), "s_per_update_max": max(r["s_per_update"] for r in tr),
         "rows_per_update": start["micro_per_update"] * start["batch"], "pre_eval_min": (T(step0["at"]) - T(start["at"])).total_seconds() / 60,
         "post_min": (T(done["at"]) - T(tr[-1]["at"])).total_seconds() / 60, "hours_logged": done["hours"], "wall_start_to_done_h": (T(done["at"]) - T(start["at"])).total_seconds() / 3600, "n_train_rows": len(tr)}
    pf = os.path.join(root, arm, "PREFLIGHT_PASS.json")
    if os.path.exists(pf):
        j = json.load(open(pf, encoding="utf-8"))
        d["preflight_to_start_min"] = (T(start["at"]) - T(j["at"])).total_seconds() / 60
        d["preflight_s_per_microbatch_estimate"] = j.get("s_per_microbatch")
        d["preflight_gpu_mem_gb"] = j.get("gpu_mem_gb")
    res["arms"][arm] = d
H = [a for a in res["arms"] if a.startswith("H1")]
S = st.mean(res["arms"][a]["s_per_update_mean"] for a in H)
PRE = st.mean(v["pre_eval_min"] for v in res["arms"].values())
POST = st.mean(v["post_min"] for v in res["arms"].values())
GAP = [v["preflight_to_start_min"] for v in res["arms"].values() if "preflight_to_start_min" in v]
res["pooled"] = {"H1_arms": H, "s_per_update_H1_mean": S, "eval_min_pre_mean": PRE, "eval_min_post_mean": POST, "preflight_to_start_min": GAP,
                 "head_cost_s_per_update_vs_N": S - st.mean(v["s_per_update_mean"] for a, v in res["arms"].items() if a.startswith("N-"))}
EVAL_S, SAVE_S = PRE * 60, (POST - PRE) * 60
GAP_H = sorted(res["arms"][a]["preflight_to_start_min"] for a in H if "preflight_to_start_min" in res["arms"][a])      # the three H1 arms (the chain version S1c descends from): 14.7-16.1 min
GAPM = st.median(GAP_H) + 2.0                                                                                            # + about 2 min: test_s1c.py joins G-SUITE (dev box 1m52s)
pts = {}
for name, N, acc, E in (("u2k_x64 (2,000 updates, 8x8 rows)", 2000, 8, 3), ("u10k_x8 (10,000 updates, 8 rows)", 10000, 1, 3), ("S1 itself (400 updates, 8x8 rows)", 400, 8, 2)):
    row = {}
    for x in (0.0, 0.05, 0.5):
        train_s = N * (acc * (S - x) / 8 + x)
        hours = (train_s + E * EVAL_S + SAVE_S) / 3600
        row[f"x={x}"] = {"train_h": round(train_s / 3600, 3), "run_h": round(hours, 3), "with_preflight_gate_gap_h": round(hours + GAPM / 60, 3)}
    pts[name] = {"rows": N * acc * 8, "evals": E, **row}
res["model"] = {"eval_s_each": EVAL_S, "save_s": SAVE_S, "preflight_to_start_min_used": GAPM, "preflight_to_start_min_H1_arms": GAP_H, "points": pts,
                "check_against_S1": f"S1 400 updates: wall start->done {st.mean(v['wall_start_to_done_h'] for v in res['arms'].values()):.3f} h per arm (n=9, the `hours` field, {st.mean(v['hours_logged'] for v in res['arms'].values()):.3f} h, excludes the step-0 eval); the model's S1 line (x-independent: accum 8 is S1's own shape) reads {pts['S1 itself (400 updates, 8x8 rows)']['x=0.0']['run_h']} h"}
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print(json.dumps(res["pooled"], indent=1))
for k, v in pts.items():
    print(k, json.dumps(v))
print(res["model"]["check_against_S1"])
