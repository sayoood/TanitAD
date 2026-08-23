"""Measured probe of the yaak-ai/L2D LeRobot dataset semantic-label taxonomy (G-H, backlog P1 #2d).

Answers, with real bytes off the HF hub (no full clone of the 90 TB corpus):
  1. What is the nav-command / task-instruction taxonomy depth?  (how many distinct tasks,
     what do the instructions look like, are they route-level strategic commands?)
  2. Are ego actions present and in what form?  (continuous / discrete — the labeled bridge)
  3. Camera / contract compatibility with our episode contract (6-ch 2-frame 256 px stacks).
  4. Cost to first batch (what has to be downloaded to build one episode).

Only the small `meta/` sidecar files + ONE data parquet are fetched (LeRobot v2 layout),
so this runs in minutes on the dev box.  Reaches HF through truststore (certifi fails behind
the corporate TLS proxy — see hub memory note).  Writes a JSON result next to this script.
"""
import json, os, sys, collections, pathlib

import truststore
truststore.inject_into_ssl()

from huggingface_hub import hf_hub_download, HfApi

REPO = "yaak-ai/L2D"
OUT = pathlib.Path(__file__).with_name("l2d_taxonomy_result.json")
TOKEN = None
# token from Keys.txt (line after "huggingface key:")
keys = pathlib.Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
for i, ln in enumerate(keys):
    if "huggingface key" in ln.lower() and i + 1 < len(keys):
        TOKEN = keys[i + 1].strip()
        break

res = {"repo": REPO, "steps": {}}

def dl(path):
    return hf_hub_download(REPO, path, repo_type="dataset", token=TOKEN)

# ---- 1. info.json: schema, fps, camera keys, feature shapes ----------------
info = json.load(open(dl("meta/info.json"), encoding="utf-8"))
feats = info.get("features", {})
cam_keys = [k for k, v in feats.items() if v.get("dtype") in ("video", "image")]
res["steps"]["info"] = {
    "total_episodes": info.get("total_episodes"),
    "total_frames": info.get("total_frames"),
    "fps": info.get("fps"),
    "robot_type": info.get("robot_type"),
    "camera_keys": cam_keys,
    "camera_shapes": {k: feats[k].get("shape") for k in cam_keys},
    "action_features": {k: feats[k].get("shape") for k in feats if k.startswith("action")},
    "state_features": [k for k in feats if k.startswith("observation.state")],
    "all_feature_keys": sorted(feats.keys()),
}

# ---- 2. tasks.jsonl: the instruction taxonomy -------------------------------
try:
    tasks_path = dl("meta/tasks.jsonl")
    tasks = [json.loads(l) for l in open(tasks_path, encoding="utf-8") if l.strip()]
except Exception:
    # newer LeRobot packs tasks in parquet
    import pandas as pd
    tasks_path = dl("meta/tasks.parquet")
    tasks = pd.read_parquet(tasks_path).to_dict("records")

instr = [str(t.get("task", t.get("instructions", t))) for t in tasks]
# crude taxonomy: leading verb/phrase = the maneuver primitive
def head(s):
    w = s.lower().strip().replace(",", " ").split()
    return " ".join(w[:2]) if w else ""
verbs = collections.Counter(head(s) for s in instr)
res["steps"]["tasks"] = {
    "n_distinct_tasks": len(tasks),
    "sample_instructions": instr[:12],
    "top_instruction_heads": verbs.most_common(15),
    "has_speed_limit_token": sum("speed limit" in s.lower() for s in instr),
    "has_distance_token": sum(any(u in s.lower() for u in (" km", " m ", "meter")) for s in instr),
    "has_road_class_token": sum(any(r in s.lower() for r in ("motorway", "secondary", "primary", "residential", "roundabout")) for s in instr),
}

# ---- 3. one data parquet: decode real action + waypoint columns -------------
try:
    import pandas as pd
    # episode 0 chunk 0 (LeRobot v2.1 path convention)
    for cand in ("data/chunk-000/episode_000000.parquet",
                 "data/chunk-000/file-000.parquet",
                 "data/chunk-000/file_000.parquet"):
        try:
            dp = dl(cand); break
        except Exception:
            dp = None
    if dp:
        df = pd.read_parquet(dp)
        cols = list(df.columns)
        row = df.iloc[0].to_dict()
        def shp(v):
            try: return list(getattr(v, "shape", [len(v)]))
            except Exception: return None
        res["steps"]["data_sample"] = {
            "parquet": os.path.basename(dp),
            "n_rows": int(len(df)),
            "columns": cols,
            "action_continuous_dim": shp(row.get("action.continuous")),
            "action_discrete_dim": shp(row.get("action.discrete")),
            "waypoints_dim": shp(row.get("observation.state.waypoints")),
            "task_index_value": int(row["task_index"]) if "task_index" in row else None,
        }
    else:
        res["steps"]["data_sample"] = {"note": "no data parquet path matched; meta sufficient for taxonomy"}
except Exception as e:
    res["steps"]["data_sample"] = {"error": repr(e)}

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str)[:4000])
print("\nWROTE", OUT)
