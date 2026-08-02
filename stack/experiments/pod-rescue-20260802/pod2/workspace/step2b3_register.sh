#!/bin/bash
# RUNBOOK STEP 2b (rename) + 2c (register, --write-manifest) for BOTH splits.
set -eu
export PYTHONPATH=/workspace/v5eval/stack
cd /workspace/v5eval/stack
MAN=/workspace/v5eval/stack/tanitad/data/parity_manifest.json
OLD=/workspace/data/pai_wide120_v2png_train
NEW=/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
VAL=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl

echo "############ 2b. RENAME (no _v2manifest.pt exists -> costs nothing) ############"
if [ -d "$OLD" ]; then
  ls "$OLD"/_v2manifest.pt 2>/dev/null && { echo "REFUSING: a _v2manifest.pt exists"; exit 3; }
  mv "$OLD" "$NEW"
  echo "renamed: $OLD -> $NEW"
else
  echo "already renamed (source dir absent)"
fi
ls -d "$NEW" && echo "clips: $(ls "$NEW" | grep -c 'v2ep.pt$')"

echo "############ manifest BEFORE ############"
python3 -c "import json;print(list(json.load(open('$MAN'))['corpora'].keys()))"

echo "############ 2c. REGISTER TRAIN ############"
python3 scripts/register_v2_sibling.py \
  --cache        "$NEW" \
  --new-key      physicalai-train-e438721ae894-w120-256x640cyl \
  --source-key   physicalai-train-e438721ae894 \
  --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --out          /workspace/v5eval/raw/sibling_entry_train.json \
  --write-manifest 2>&1 | grep -E "WROTE|NOW STAGE|only on this host|V2_SIBLING" | head -5

echo "############ 2c. REGISTER VAL ############"
python3 scripts/register_v2_sibling.py \
  --cache        "$VAL" \
  --new-key      physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --source-key   physicalai-val-0c5f7dac3b11 \
  --expect-clips /workspace/wfov/paritysplit/parity_val_clips.txt \
  --out          /workspace/v5eval/raw/sibling_entry_val.json \
  --write-manifest 2>&1 | grep -E "WROTE|NOW STAGE|only on this host|V2_SIBLING" | head -5

echo "############ manifest AFTER ############"
python3 - <<PY
import json
d = json.load(open("$MAN"))
print("corpora:", list(d["corpora"].keys()))
for k in ("physicalai-train-e438721ae894-w120-256x640cyl",
          "physicalai-val-0c5f7dac3b11-w120-256x640cyl"):
    e = d["corpora"][k]
    g = (e.get("provenance") or {}).get("geometry") or {}
    gc = g.get("geometry_check") or {}
    print("---", k)
    print("  episode_count      :", e.get("episode_count"))
    print("  uid_kind           :", e.get("uid_kind"))
    print("  episode_uid_sha256 :", e.get("episode_uid_sha256"))
    print("  derived_from       :", (e.get("provenance") or {}).get("derived_from"))
    print("  geometry.frame     :", json.dumps(g.get("frame")))
    print("  geometry.frame_tag :", g.get("frame_tag"))
    print("  observed_frac      :", gc.get("observed_frac"))
    print("  observed_frac_superseded:", gc.get("observed_frac_superseded"))
    ro = g.get("rig_observability") or {}
    print("  rig_observability keys:", sorted(ro.keys()) if isinstance(ro, dict) else ro)
PY

echo "############ resolution check: corpus_key_of ############"
python3 - <<PY
import sys; sys.path.insert(0, "/workspace/v5eval/stack")
from tanitad.data import parity
for p in ("$NEW", "$VAL"):
    print(p, "->", parity.corpus_key_of(p, "$MAN"))
PY
