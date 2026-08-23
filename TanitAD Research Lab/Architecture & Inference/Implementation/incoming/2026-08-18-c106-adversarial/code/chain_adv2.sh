#!/usr/bin/env bash
# CHAIN 2 — the MECHANISM half: is the trained token field RANK-COLLAPSED, and
# does whitening recover what C106 says was "subtracted"?  ZERO TRAINING.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-c106-adversarial"
F="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-experiments"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
CK="$SP/sp2/ck/v6F_sw_step011250.fp16.pt"
RIDX="$SP/eenc/row_index.pt"
BANKED="$SP/sp2/cache_tok11250/latents.pt"
W="$SP/c106"
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$D/raw" "$W"

case "${1:-all}" in
  rank)
    "$PY" "$D/code/c106_rank.py" --ckpt "$CK" --row-index "$RIDX" \
      --episodes-dir "$EPS" --n-rows 384 --batch 4 \
      --out "$D/raw/rank.json" > "$D/raw/log_rank.txt" 2>&1
    echo "ADV2 rank rc=$?" ;;
  whiten)
    "$PY" "$D/code/c106_whiten.py" --cache "$BANKED" --tag ours \
      --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
      --out "$D/raw/whiten_ours.json" > "$D/raw/log_whiten_ours.txt" 2>&1
    echo "ADV2 whiten ours rc=$?"
    T="$W/tok_randenc_s0.pt"
    "$PY" "$F/code/eenc_falsifier_cache.py" --mode randenc --seed 0 \
      --row-index "$RIDX" --episodes-dir "$EPS" --ckpt "$CK" --out "$T" \
      --batch 8 > "$D/raw/log_build_randenc_s0.txt" 2>&1
    echo "ADV2 build randenc s0 rc=$?"
    "$PY" "$D/code/c106_whiten.py" --cache "$T" --tag randenc_s0 \
      --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
      --out "$D/raw/whiten_randenc_s0.json" \
      > "$D/raw/log_whiten_randenc_s0.txt" 2>&1
    echo "ADV2 whiten randenc_s0 rc=$?"
    rm -f "$T" ;;
  all)
    bash "$0" rank; bash "$0" whiten ;;
esac
echo "CHAIN2_DONE_${1:-all}"
