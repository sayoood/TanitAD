#!/usr/bin/env bash
# CHAIN — the C106 ADVERSARIAL battery. ZERO TRAINING, ZERO THOR, ZERO POD.
# Dev-box RTX 4060 for the frozen encoder forwards; the ridge is CPU.
#
# ⛔ ORDER IS NOT ARBITRARY. `gate` runs FIRST: this harness must reproduce the
# BANKED numbers under the BANKED alpha grid before any widened grid is read
# (C94 — agree with the PRODUCER, not with yourself). It reproduced
# 0.05207 / 0.00490 / 0.00000 exactly.
#
# ⚠️ The randenc token caches are 2.76 GB each and are REBUILT here (the
# producing chain deleted them). Build -> fit -> delete, one seed at a time.
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
# ⛔ torch spawns ~113 threads PER PROCESS (CLAUDE.md, MEASURED 2026-07-27).
export OMP_NUM_THREADS=6 PYTHONUTF8=1
mkdir -p "$D/raw" "$W"

fit () {  # <tag> <cache> <grid> [extra...]
  local tag="$1" cache="$2" grid="$3"; shift 3
  "$PY" "$D/code/c106_adv.py" --stage fit --cache "$cache" --tag "$tag" \
    --split-json "$SPLIT" --episodes-dir "$EPS" --join-file "$JOIN" \
    --alpha-grid "$grid" --proj-seeds 0 1 2 --ridge-seeds 0 1 2 \
    --out "$D/raw/adv_${tag}.json" --out-preds "$W/pred_${tag}.npz" "$@" \
    > "$D/raw/log_adv_${tag}.txt" 2>&1
  echo "ADV fit $tag rc=$?"
}

build () {  # <mode> <seed> <out> [extra...]
  local m="$1" s="$2" o="$3"; shift 3
  "$PY" "$F/code/eenc_falsifier_cache.py" --mode "$m" --seed "$s" \
    --row-index "$RIDX" --episodes-dir "$EPS" --ckpt "$CK" --out "$o" \
    --batch 8 "$@" > "$D/raw/log_build_${m}_s${s}.txt" 2>&1
  echo "ADV build $m s$s rc=$?"
}

delta () {  # <a> <b>
  "$PY" "$D/code/c106_adv.py" --stage delta \
    --preds-a "$W/pred_$1.npz" --preds-b "$W/pred_$2.npz" \
    --name-a "$1" --name-b "$2" --out "$D/raw/delta_$1_vs_$2.json" \
    > "$D/raw/log_delta_$1_vs_$2.txt" 2>&1
  echo "ADV delta $1 vs $2 rc=$?"
}

case "${1:-all}" in
  ours)
    fit ours_base   "$BANKED" base
    fit ours_wide   "$BANKED" wide ;;
  randenc)
    for s in 0 1 2; do
      T="$W/tok_randenc_s$s.pt"
      build randenc "$s" "$T"
      fit "randenc_s${s}_base" "$T" base
      fit "randenc_s${s}_wide" "$T" wide
      rm -f "$T"
    done ;;
  null)   # the matched-random NULL through the identical path, per arm
    fit ours_null "$BANKED" wide --randomise-features 4242
    T="$W/tok_randenc_s0.pt"; build randenc 0 "$T"
    fit randenc_s0_null "$T" wide --randomise-features 4242
    rm -f "$T" ;;
  pc)     # ⛔ POSITIVE CONTROL, planted into the SAME tokens.
    # ⚠️ PC-2OBJ IS THE WRONG CONTROL HERE AND THIS RUN PROVES IT, WHICH IS WHY
    # IT IS KEPT: its two planted tokens have OPPOSITE SIGN inside ONE deployed
    # 4x10 cell, so the deployed pool CANCELS them exactly. It is a
    # POOLING-RATIO contrast (p40 0.0000 -> p1 0.9998 in E-R1-0), and at p40 —
    # the only arm this battery reads — it is INERT BY CONSTRUCTION. MEASURED:
    # `adv_ours_pc2obj` reproduces `adv_ours_wide` to 5e-05. A control that
    # cannot fire is not a control.
    # ⇒ PC-LOCAL (a 2x2 token block wholly INSIDE one deployed cell, same sign,
    # so it survives the 40:1 average at 4/40 amplitude) and PC-DIST (planted in
    # every token) are the controls that CAN fire at p40, and both arms get them.
    fit ours_pc2obj "$BANKED" wide --oracle local2 --oracle-amp 1.0
    fit ours_pclocal "$BANKED" wide --oracle local --oracle-amp 1.0
    fit ours_pcdist "$BANKED" wide --oracle dist --oracle-amp 1.0
    T="$W/tok_randenc_s0.pt"; build randenc 0 "$T"
    fit randenc_s0_pc2obj "$T" wide --oracle local2 --oracle-amp 1.0
    fit randenc_s0_pclocal "$T" wide --oracle local --oracle-amp 1.0
    fit randenc_s0_pcdist "$T" wide --oracle dist --oracle-amp 1.0
    rm -f "$T" ;;
  deltas)
    for s in 0 1 2; do
      delta "randenc_s${s}_base" ours_base
      delta "randenc_s${s}_wide" ours_wide
    done
    delta ours_wide ours_null
    delta randenc_s0_wide randenc_s0_null ;;
  all)
    bash "$0" ours; bash "$0" randenc; bash "$0" null; bash "$0" pc
    bash "$0" deltas ;;
esac
echo "CHAIN_DONE_${1:-all}"
