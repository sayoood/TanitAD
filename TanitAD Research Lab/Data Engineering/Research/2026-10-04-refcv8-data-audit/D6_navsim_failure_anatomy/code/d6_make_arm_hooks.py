"""D6 P2 -- turn a bridge arm's rows_<arm>.jsonl into a hooks-style JSON that d6_rescore.py / d6_rescore --geom-only can consume.

    python d6_make_arm_hooks.py <banked A1 30k hooks.json> <rows_<arm>.jsonl> <out hooks.json>

Copies scene_type / log_name from the banked A1 hooks and the BANKED A1 row (so ``repro`` is meaningful ONLY for the A1 control arm K7);
``agent_poses`` = the arm's emitted ``poses``.  A scene whose row is a CV stand-in (source != refcv7) is refused (listed, not silently dropped).
"""
import json
import sys

banked_path, rows_path, out_path = sys.argv[1:4]
banked = {c["token"]: c for c in json.load(open(banked_path, encoding="utf-8"))["pdm_score_calls"]}
calls, bad = [], []
for ln in open(rows_path, encoding="utf-8"):
    try:
        r = json.loads(ln)
    except Exception:                                               # noqa: BLE001
        continue
    t = r["token"]
    if r.get("source") != "refcv7":
        bad.append(t)
        continue
    b = banked[t]
    calls.append({"token": t, "scene_type": b["scene_type"], "log_name": b["log_name"], "agent_poses": r["poses"],
                  "agent_sampling": [8, 0.5], "human_poses": None, "v0_mps": b.get("v0_mps"), "row": b["row"]})
json.dump({"pdm_score_calls": calls, "cache_scenes": [], "refused_non_model_rows": bad}, open(out_path, "w"))
print(f"{len(calls)} calls, {len(bad)} refused -> {out_path}")
