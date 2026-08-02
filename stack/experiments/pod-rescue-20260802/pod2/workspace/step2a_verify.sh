#!/bin/bash
# RUNBOOK STEP 2a — prove membership. Writes NOTHING to the manifest.
set -u
export PYTHONPATH=/workspace/v5eval/stack
cd /workspace/v5eval/stack
mkdir -p /workspace/v5eval/raw

echo "############ TRAIN (pre-rename, verify-only) ############"
python3 scripts/register_v2_sibling.py --verify-only \
  --cache        /workspace/data/pai_wide120_v2png_train \
  --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --source-key   physicalai-train-e438721ae894 \
  --out          /workspace/v5eval/raw/verify_train.json 2>&1 | tail -5
echo "TRAIN exit: ${PIPESTATUS[0]}"

echo "############ VAL (verify-only) ############"
python3 scripts/register_v2_sibling.py --verify-only \
  --cache        /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --expect-clips /workspace/wfov/paritysplit/parity_val_clips.txt \
  --source-key   physicalai-val-0c5f7dac3b11 \
  --out          /workspace/v5eval/raw/verify_val.json 2>&1 | tail -5
echo "VAL exit: ${PIPESTATUS[0]}"

echo "############ PASS CRITERIA (fixed BEFORE the numbers existed) ############"
python3 - <<'PY'
import json
for tag, p in (("train", "/workspace/v5eval/raw/verify_train.json"),
               ("val", "/workspace/v5eval/raw/verify_val.json")):
    try:
        r = json.load(open(p))
    except Exception as e:
        print(tag, "NO ARTIFACT:", e); continue
    keep = ("mode", "n_built", "n_expected", "missing_count", "extra_count",
            "membership_identical", "shortfall_matches_recorded_skips",
            "shortfall_identity_checked", "clip_sha256", "expect_clips_count",
            "source_key", "n_clips_expected", "clips_built")
    print(tag, {k: r[k] for k in keep if k in r})
    extra = r.get("extra_count")
    miss = r.get("missing_count")
    ok = (extra == 0) and (miss == 0 or r.get("shortfall_matches_recorded_skips") is True)
    print(tag, "PASS_CRITERIA_MET:", ok)
PY
