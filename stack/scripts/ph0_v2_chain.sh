#!/bin/bash
# PH0 v2 — production chain. Grammar-constrained 4-call extraction.
#
# ⛔ NO GIT IN THIS CHAIN. Pods have no git credentials: `git fetch` HANGS, and a
# failed fetch followed by a successful checkout RESETS the tree to an ancient
# commit, destroying shipped files. Files arrive md5-verified and are checked
# here by grep + a real import before anything launches.
#
# Every branch emits V2_EXIT= so silence can never read as success.
set -u
REPO="${REPO:-/workspace/TanitAD_head}"
S="$REPO/stack"
CLIPS="${CLIPS:-/workspace/ph0_mini/clips.json}"
VIDEOS="${VIDEOS:-/workspace/ph0_mini/videos}"
EGO="${EGO:-/workspace/ph0_mini/ego}"
# none = v2.1 control (prompts byte-identical) | past = production
# | full = leak-measurement arm, B2 sees the speedometer
EGO_MODE="${EGO_MODE:-past}"
ARM="${ARM:-Qwen/Qwen3.5-9B}"
OUT="${OUT:-/workspace/ph0_mini/v2}"
N="${N:-8}"
RESUME="${RESUME:---resume}"
LOG="${LOG:-/tmp/ph0v2.log}"
: > "$LOG"

export PYTHONPATH="$S"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
# torch spawns ~113 threads PER PROCESS; ffmpeg/swscale sizes its pool to the
# HOST's nproc (96 here, not the container's allowance) and fails thread
# creation with EAGAIN, which presents as "[swscaler] Failed initializing
# scaling graph" and looks exactly like a broken model. Both pinned.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PH0_DECODER_THREADS="${PH0_DECODER_THREADS:-1}"

PY="${PY:-/workspace/a2venv/bin/python}"
[ -x "$PY" ] || { echo "[v2] 09 no interpreter at $PY" >> "$LOG"
                  echo "V2_EXIT=09" >> "$LOG"; exit 1; }

# ---- gate 10: inputs exist ------------------------------------------------ #
for f in "$S/scripts/ph0_v2.py" "$S/scripts/ph0_pilot.py" "$CLIPS" "$VIDEOS"; do
  [ -e "$f" ] || { echo "[v2] 10 MISSING $f — ship it (xz+b64, per-file md5)" \
                     >> "$LOG"; echo "V2_EXIT=10" >> "$LOG"; exit 1; }
done

# ---- gate 11: the shipped code is CURRENT, by content --------------------- #
# `git log` on a pod proves nothing — its HEAD is weeks stale while the working
# tree is current. Each token below is a fix that must not silently regress.
for tok in "validate_v2" "force_json_field_order" "max_consecutive_whitespaces" \
           "allowed_tokens" "AutoModelForImageTextToText" "ego_past_state" \
           "_EGO_SPEED_KEYS"; do
  grep -q "$tok" "$S/scripts/ph0_v2.py" || {
    echo "[v2] 11 STALE ph0_v2.py — missing '$tok'" >> "$LOG"
    echo "V2_EXIT=11" >> "$LOG"; exit 1; }
done
grep -q "_decoder_threads" "$S/scripts/ph0_pilot.py" || {
  echo "[v2] 11 STALE ph0_pilot.py — decode thread pin absent" >> "$LOG"
  echo "V2_EXIT=11" >> "$LOG"; exit 1; }

# ---- gate 12: REAL imports + a validator smoke ---------------------------- #
# Greps passing while imports failed is how the T1 run burned 11 min/arm and
# then died in analyze().
cd "$S" || { echo "V2_EXIT=12_NO_STACK" >> "$LOG"; exit 1; }
"$PY" - >> "$LOG" 2>&1 <<'PY'
import sys
try:
    sys.path.insert(0, "scripts")
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer  # noqa: F401
    from lmformatenforcer.characterlevelparser import \
        CharacterLevelParserConfig
    from ph0_v2 import S_B3, validate_v2
    # the measured defect must still be caught: bbox beyond a 448 px frame
    bad = validate_v2("B3_ground_0", {"visible": True, "frame_idx": 0,
                                      "bbox": [952, 100, 975, 160]}, px=448)
    assert bad, "validator no longer catches an out-of-frame bbox"
    CharacterLevelParserConfig(max_consecutive_whitespaces=1,
                               force_json_field_order=True,
                               max_json_array_length=6)
    # v2.2: the B2 speed redaction is the one thing standing between the sign
    # channel and an unfalsifiable "read" of the ego speedometer. Gate on it.
    from ph0_v2 import ego_past_state, ego_section
    _st = ego_past_state([[i * 1.39, 0.0, 0.0, 13.9] for i in range(200)], 150)
    assert _st and _st["v_now_kmh"] == 50.0, "ego state not computed"
    _b2 = ego_section("B2_signs", _st, "past")
    assert "50.0" not in _b2 and "v_now_ms" not in _b2, \
        "B2 ego block is NOT speed-redacted — sign text could be transcribed "\
        "from the speedometer"
    assert "v_now_kmh" in ego_section("B1_scene", _st, "past")
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    print("V2_IMPORTS_OK")
except Exception as e:                                    # noqa: BLE001
    print(f"V2_IMPORT_FAIL {type(e).__name__}: {e}")
PY
grep -q V2_IMPORTS_OK "$LOG" || { echo "V2_EXIT=12" >> "$LOG"; exit 1; }

# ---- gate 15: real disk, never df ---------------------------------------- #
# df reports the 965 TB cluster and hides the per-pod MooseFS quota.
mkdir -p "$OUT"
dd if=/dev/zero of="$OUT/.ddtest" bs=1M count=64 >/dev/null 2>&1 || {
  echo "[v2] 15 DISK — 64 MB dd write FAILED (quota, not df)" >> "$LOG"
  echo "V2_EXIT=15" >> "$LOG"; rm -f "$OUT/.ddtest"; exit 1; }
rm -f "$OUT/.ddtest"

# ---- run ------------------------------------------------------------------ #
"$PY" -u scripts/ph0_v2.py \
  --clips "$CLIPS" --video-root "$VIDEOS" --ego-root "$EGO" \
  --arm "$ARM" --out "$OUT" --n "$N" --ego-in-prompt "$EGO_MODE" \
  $RESUME >> "$LOG" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { echo "V2_EXIT=RUN_$rc" >> "$LOG"; exit "$rc"; }
grep -q PH0V2_DONE "$LOG" || { echo "V2_EXIT=NO_DONE_MARKER" >> "$LOG"; exit 1; }
echo "V2_EXIT=0" >> "$LOG"
