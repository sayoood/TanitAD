"""HF structure probe (Data-Eng G-H, 2026-07-15).

Two measured questions, both via the HF *tree* API (memory: tree, not
list_repo_files, which hangs on huge repos):

  Q1 (BACKLOG P0.1, the GATE): does PhysicalAI-WorldModel-Synthetic-Scenarios
     ship an ego-pose / action field? -> decides loader path (cosmos-mirror vs
     IDM/H7 vs video-only).
  Q2 (OWN_DATASET_PLAN §7 #2): PandaSet HF mirror structure — front camera,
     GPS/IMU pose, camera intrinsics present? -> grounds the pandaset adapter.

Read-only. No downloads. Prints a compact JSON verdict.
"""
import json
import re
import sys

import truststore
truststore.inject_into_ssl()  # certifi fails behind the dev-box TLS proxy

from huggingface_hub import HfApi

TOKEN = sys.argv[1] if len(sys.argv) > 1 else None
api = HfApi(token=TOKEN)


def sample_tree(repo_id, repo_type="dataset", root_paths=("",), max_per=4000):
    """Walk a few top-level paths of the tree, collect file paths (bounded)."""
    seen = []
    try:
        for rp in root_paths:
            n = 0
            for item in api.list_repo_tree(repo_id, path_in_repo=rp or None,
                                           repo_type=repo_type, recursive=True):
                seen.append(item.path)
                n += 1
                if n >= max_per:
                    break
    except Exception as e:  # noqa
        return {"error": f"{type(e).__name__}: {e}", "paths": seen}
    return {"paths": seen}


def classify(paths):
    """Bucket a path list by keyword to answer the structure questions."""
    ext = {}
    kw = {"pose": [], "ego": [], "vehicle": [], "can": [], "oxts": [],
          "intrinsic": [], "calib": [], "front": [], "camera": [], "gps": [],
          "imu": [], "steer": [], "action": [], "trajectory": [], "meta": []}
    for p in paths:
        pl = p.lower()
        m = re.search(r"\.([a-z0-9]+)$", pl)
        if m:
            ext[m.group(1)] = ext.get(m.group(1), 0) + 1
        for k in kw:
            if k in pl:
                kw[k].append(p)
    return {"ext_counts": ext,
            "keyword_hits": {k: {"n": len(v), "examples": v[:3]}
                             for k, v in kw.items() if v}}


def probe(repo_id, root_paths=("",)):
    print(f"\n### {repo_id}", flush=True)
    tree = sample_tree(repo_id, root_paths=root_paths)
    if "error" in tree:
        print("  ERROR:", tree["error"], flush=True)
        return {"repo": repo_id, "error": tree["error"],
                "partial_paths": tree["paths"][:20]}
    paths = tree["paths"]
    cls = classify(paths)
    print(f"  files sampled: {len(paths)}", flush=True)
    print(f"  ext_counts: {json.dumps(cls['ext_counts'])}", flush=True)
    print(f"  keyword_hits: {json.dumps(cls['keyword_hits'], indent=0)}",
          flush=True)
    print(f"  first 15 paths: {json.dumps(paths[:15], indent=0)}", flush=True)
    return {"repo": repo_id, "n_sampled": len(paths),
            "ext_counts": cls["ext_counts"],
            "keyword_hits": cls["keyword_hits"],
            "sample_paths": paths[:40]}


out = {}
# Q1: WorldModel-Synthetic-Scenarios — the pose gate.
out["worldmodel_synth"] = probe(
    "nvidia/PhysicalAI-Autonomous-Vehicle-WorldModel-Synthetic-Autonomous-Driving-Scenarios")
# Q2: PandaSet HF mirror.
out["pandaset"] = probe("georghess/pandaset")

with open(sys.argv[2] if len(sys.argv) > 2 else "hf_probe_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nHF_PROBE_DONE", flush=True)
