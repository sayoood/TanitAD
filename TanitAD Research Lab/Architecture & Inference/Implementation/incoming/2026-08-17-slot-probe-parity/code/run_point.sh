#!/usr/bin/env bash
# run_point.sh <label> <ckpt> <stride> <want_tokens:0|1> [nq]
# ⭐ IDEMPOTENT + LOCKED: three chains run in parallel over a shared point list, so
# a point already banked is SKIPPED and a point in flight is not started twice.
set -eu
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
LBL="$1"; CK="$2"; STRIDE="$3"; TOK="$4"; NQ="${5:-74}"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=6
if [ -f "$S/raw/results_$LBL.json" ]; then echo "POINTSKIP $LBL (already banked)"; exit 0; fi
if ! mkdir "$S/.lock_$LBL" 2>/dev/null; then echo "POINTSKIP $LBL (in flight elsewhere)"; exit 0; fi
trap 'rmdir "$S/.lock_$LBL" 2>/dev/null || true' EXIT
EXTRA=""; [ "$TOK" = "1" ] && EXTRA="--want-tokens"
ARMS="cells"; [ "$TOK" = "1" ] && ARMS="cells tokens"
"$PY" "$S/sp1_cache_latents.py" \
  --ckpt "$CK" --config-json "$S/v6F_config.json" \
  --v2-cache "$S/cache/slotprobe-lead130-w120-256x640cyl" \
  --join-file "$S/lead130_agents.jsonl" \
  --out "$S/cache_$LBL" --stride "$STRIDE" --batch 4 --v2-lru 6 $EXTRA
"$PY" "$S/sp2_probe.py" \
  --cache "$S/cache_$LBL/latents.pt" --out "$S/out_$LBL" \
  --split-json "$S/p3_selection.json" --arms $ARMS \
  --n-queries "$NQ" --steps 3000 --batch 32 --lr 1e-3 --seed 0
cp "$S/cache_$LBL/sp1_meta.json" "$S/raw/cache_meta_$LBL.json"
cp "$S/out_$LBL/slot_probe_results.json" "$S/raw/results_$LBL.json"
echo "POINTDONE $LBL"
