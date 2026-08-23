#!/usr/bin/env bash
# ⭐ THE STEP TREND — does S-W training REMOVE linear readability, or add it?
#
# §7.4.1 measured a RANDOM-INIT copy of our encoder reading 3.6x BETTER than the
# TRAINED one at step 11250. A single checkpoint cannot separate "training
# subtracts" from "11250 is a transient". This re-encodes the SAME 2809 banked
# windows with the encoder weights from each locally banked checkpoint.
#
# ⛔ THE FIRST RUN IS A GATE, NOT A DATA POINT. `trained@11250` must reproduce
# the BANKED token cache's numbers (raw/fals_ours.json). If it does not, the
# `trained` mode is wrong and nothing below it is readable (C94: an instrument
# must agree with the PRODUCER, not merely with itself).
#
# ⚠️ Only three checkpoints are local (9250 / 10000 / 11250) — a 2000-step
# window inside a 30000-step run. `randenc` (§7.4) is the STEP-0 point.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-experiments"
E="$REPO/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-pooling-ladder-ER10"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export OMP_NUM_THREADS=6 PYTHONUTF8=1
for s in 011250 010000 009250; do
  T="$SP/eenc/tok_trained_$s.pt"
  "$PY" "$D/code/eenc_falsifier_cache.py" --mode trained \
    --row-index "$SP/eenc/row_index.pt" \
    --episodes-dir "$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl" \
    --ckpt "$SP/sp2/ck/v6F_sw_step$s.fp16.pt" --out "$T" --batch 8 \
    > "$D/raw/log_build_trained_$s.txt" 2>&1
  echo "TREND build $s rc=$?"
  "$PY" "$E/code/er10_pool_ladder.py" --cache "$T" \
    --split-json "$SP/sp2/p3_selection.json" \
    --episodes-dir "$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl" \
    --join-file "$SP/sp2/lead130_agents.jsonl" \
    --out "$D/raw/trend_trained_$s.json" --label "v6F-SW-30k@$s TRAINED re-encode" \
    --arms p40 --targets ego_v0 lead_gap lead_closing --proj-seeds 0 1 2 \
    > "$D/raw/log_trend_trained_$s.txt" 2>&1
  echo "TREND ladder $s rc=$?"
  rm -f "$T"
done
echo "CHAIN_DONE_steptrend"
