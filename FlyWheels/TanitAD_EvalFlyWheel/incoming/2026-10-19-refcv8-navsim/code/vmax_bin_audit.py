"""vmax_bin_audit.py -- does refcv7's RECORDED speed bin (float64 ``speed_max_bin`` -> ``bin_kmh`` / ``ceiling_ms``) ever differ
from the bin the MODEL computes (float32 ``speed_max_bin_tensor``, ``MaxSpeedOneHotEncoder``) on the navtest / navhard map limits?

Context: the refcv8 NavSim bridge review (A-R8-NS1 AMENDMENT 1) found ``refcv7_bridge.ceiling_ms`` can put a SNAPPED 50 km/h value in the
100 km/h bin while the model's tensor rule keeps it at 50.  refcv6/7's ``max_speed_input`` feeds the RAW m/s and the model bins it ONCE
(``refcv6_bridge.py:172-200``), so the model INPUT is never the recorded bin; the records (``bin_kmh``, ``ceiling_expected_ms``) feed only
descriptive strata (``parse_navtest7.py:166``, ``plan_deltas7.py:73``).  This counts the tokens where record and model disagree, and the
limits that sit within 2e-3 m/s of a bin edge.   python -I vmax_bin_audit.py --out <json>   (PYTHONPATH=<stack>)
"""
import argparse, collections, json
import torch
from tanitad.refs.refcv6_max_speed import speed_max_bin, speed_max_bin_tensor, SPEED_MAX_STEPS_KMH_V6, SPEED_MAX_STEPS_MS_V6

SRC = {"navtest": "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/inputs/speed_limits_navtest.json",
       "navhard": "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/speed_limits_navhard_two_stage.json"}
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = {"steps_ms": [repr(x) for x in SPEED_MAX_STEPS_MS_V6], "sources": SRC}
for split, p in SRC.items():
    toks = json.load(open(p, encoding="utf-8"))["tokens"]
    lim = dis = 0; pairs = collections.Counter(); edge = collections.Counter()
    for r in toks.values():
        if r.get("status") != "limit":
            continue
        lim += 1
        v = float(r["speed_limit_mps"])
        b64 = speed_max_bin(v)[0]
        b32 = int(speed_max_bin_tensor(torch.tensor([v], dtype=torch.float32))[0][0])
        if b64 != b32:
            dis += 1; pairs[f"{SPEED_MAX_STEPS_KMH_V6[b64]}->{SPEED_MAX_STEPS_KMH_V6[b32]} @ {v!r}"] += 1
        if any(abs(v - s) < 2e-3 for s in SPEED_MAX_STEPS_MS_V6):
            edge[repr(v)] += 1
    out[split] = {"n_tokens": len(toks), "n_limit": lim, "n_recorded_bin_ne_model_bin": dis, "disagreements": dict(pairs),
                  "n_limits_within_2e-3_of_a_step": sum(edge.values()), "edge_values": dict(edge)}
    print(split, json.dumps(out[split]))
json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
