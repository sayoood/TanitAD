#!/usr/bin/env bash
# CHAIN — the C104 falsifier battery. ZERO TRAINING, ZERO THOR, ZERO POD.
# One frozen banked window cache + the dev-box RTX 4060; the ridge is CPU.
#
# ⛔ ORDER IS NOT ARBITRARY. `ours` runs FIRST because it is the only arm that
# can show this invocation reproduces the BANKED v6 numbers rather than agreeing
# with itself (C94). Every falsifier is read against it, on the same command.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
D="$REPO/TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-experiments"
E="$REPO/TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-pooling-ladder-ER10"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
EPS="$SP/sp2/cache/slotprobe-lead130-w120-256x640cyl"
JOIN="$SP/sp2/lead130_agents.jsonl"
SPLIT="$SP/sp2/p3_selection.json"
CK="$SP/sp2/ck/v6F_sw_step011250.fp16.pt"
RIDX="$SP/eenc/row_index.pt"
BANKED="$SP/sp2/cache_tok11250/latents.pt"
# ⛔ torch spawns ~113 threads PER PROCESS; concurrent arms then make NO
# progress (CLAUDE.md, MEASURED 2026-07-27). Capped here, not remembered.
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$D/raw" "$SP/eenc"

# ⚠️ Read at the DEPLOYED pool only: E-R1-0 settled the pooling axis (C104), so
# spending on the other three rungs buys nothing this battery needs.
TARGETS="ego_v0 lead_gap lead_closing"

ladder () {  # <tag> <cache> <label>
  "$PY" "$E/code/er10_pool_ladder.py" --cache "$2" --split-json "$SPLIT" \
    --episodes-dir "$EPS" --join-file "$JOIN" --out "$D/raw/fals_$1.json" \
    --label "$3" --arms p40 --targets $TARGETS --proj-seeds 0 1 2 \
    > "$D/raw/log_fals_$1.txt" 2>&1
  echo "FALS ladder $1 rc=$?"
}

build () {   # <mode> <seed> <out> [extra...]
  local m="$1" s="$2" o="$3"; shift 3
  "$PY" "$D/code/eenc_falsifier_cache.py" --mode "$m" --seed "$s" \
    --row-index "$RIDX" --episodes-dir "$EPS" --ckpt "$CK" --out "$o" \
    --batch 8 "$@" > "$D/raw/log_build_${m}_s${s}.txt" 2>&1
  echo "FALS build $m s$s rc=$?"
}

case "${1:-all}" in
  ours)     # the REPRO BASELINE: the banked v6 tokens, this command, this arm
    ladder ours "$BANKED" "v6F-SW-30k@11250 OURS (repro baseline)" ;;
  dino1f)   # isolates the 3-view CONCATENATION (width 768 == ours)
    build dino1f 0 "$SP/eenc/tok_dino1f.pt" --sub-frame 0
    ladder dino1f "$SP/eenc/tok_dino1f.pt" "C-1FRAME DINOv2-B/14 one sub-frame"
    rm -f "$SP/eenc/tok_dino1f.pt" ;;
  dinorand) # isolates PRETRAINING from ARCHITECTURE + INPUT FORMAT
    build dinorand 0 "$SP/eenc/tok_dinorand.pt"
    ladder dinorand "$SP/eenc/tok_dinorand.pt" "C-DINORAND untrained DINOv2 arch"
    rm -f "$SP/eenc/tok_dinorand.pt" ;;
  randenc)  # ⭐ the load-bearing one: OUR architecture, NEVER TRAINED, 3 seeds
    for s in 0 1 2; do
      build randenc "$s" "$SP/eenc/tok_randenc_s$s.pt"
      ladder "randenc_s$s" "$SP/eenc/tok_randenc_s$s.pt" \
        "C-RANDENC our ViT5Encoder RANDOM INIT seed $s"
      rm -f "$SP/eenc/tok_randenc_s$s.pt"
    done ;;
  all)
    bash "$0" ours; bash "$0" dino1f; bash "$0" dinorand; bash "$0" randenc ;;
esac
echo "CHAIN_DONE_${1:-all}"
