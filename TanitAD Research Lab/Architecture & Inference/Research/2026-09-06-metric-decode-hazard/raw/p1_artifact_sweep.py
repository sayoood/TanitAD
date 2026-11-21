"""P1c - EVERY banked artifact recording the `grounding.step['op']` decoder,
classified by CHECKPOINT LINEAGE.

The decoder name is shared by two different modules:
  * flagship / v5f / v5.8f : `ck['grounding']` carries a SEPARATELY TRAINED
    `step_readout` (trained by train_stage_a / finetune_traj) -- NOT exposed.
  * v6 / v7 (`ck['stack']`): `V6Grounding.step['op'] is stack.step_readout_op`
    -- the module measured to be at random init under the O1=0 recipe.

So the artifact list alone over-claims. This classifies by the ckpt path/lineage
recorded IN the artifact, and reports UNKNOWN rather than guessing.
"""
import glob
import json
import os
import re

CLONE = r"C:\Users\Admin\refcv4b_repo"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p1_artifact_sweep.json"

V7_MARK = re.compile(r"v7tiny|/v7tiny/|v6-staged|v7f", re.I)
FLAGSHIP_MARK = re.compile(r"flagship|v5f|v58f|v5\.8|refa|4b", re.I)

rows = []
n_read = n_unreadable = 0
for p in glob.glob(os.path.join(CLONE, "**", "*.json"), recursive=True):
    try:
        raw = open(p, "rb").read()
    except Exception as e:                                  # unreadable != absent
        n_unreadable += 1
        rows.append({"path": os.path.relpath(p, CLONE), "status": f"UNREADABLE {e}"})
        continue
    n_read += 1
    if b"grounding.step" not in raw:
        continue
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        rows.append({"path": os.path.relpath(p, CLONE), "status": "NOT_JSON"})
        continue
    if not isinstance(d, dict):
        continue
    ck = d.get("ckpt") or d.get("model_ckpt") or ""
    prov = json.dumps(d.get("rollout_provenance") or {})[:400]
    dec = d.get("decoder") or (d.get("rollout_provenance") or {}).get("decoder")
    if isinstance(dec, dict):
        dec = dec.get("kind")
    blob = f"{ck} {json.dumps(d.get('run_config') or {})[:2000]}"
    if V7_MARK.search(blob):
        lineage = "V6_V7_STACK -> step_readout_op (EXPOSED)"
    elif FLAGSHIP_MARK.search(blob):
        lineage = "FLAGSHIP/v5f -> separately TRAINED step_readout (not exposed)"
    else:
        lineage = "UNKNOWN"
    rows.append({"path": os.path.relpath(p, CLONE).replace("\\", "/"),
                 "ckpt": ck, "decoder": dec, "lineage": lineage,
                 "prov_head": prov[:160]})

res = {"n_json_read": n_read, "n_json_unreadable": n_unreadable,
       "artifacts": [r for r in rows if r.get("decoder") or "status" in r]}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)

print(f"[scan] json files read={n_read} unreadable={n_unreadable}")
print(f"[hits] {len([r for r in rows if r.get('decoder')])} artifacts record a "
      f"grounding.step decoder")
for r in rows:
    if "status" in r:
        print("  !!", r["path"], r["status"])
for r in sorted((r for r in rows if r.get("decoder")),
                key=lambda x: x["lineage"]):
    print(f"  [{r['lineage'][:34]:34s}] {r['path']}")
    print(f"        ckpt={r['ckpt']}")
print("[wrote]", OUT)
