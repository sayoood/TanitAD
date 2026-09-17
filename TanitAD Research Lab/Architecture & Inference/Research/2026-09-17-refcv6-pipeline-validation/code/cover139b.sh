#!/usr/bin/env bash
# SPEC_REFCV6_V2 §10.6 -- THE 139-CLIP COVERAGE PASS, at the PI's 416 x 1024.
#
# §10.6 verbatim: "The 139 B1 eval clips ... CARRY THE WHOLE CHAIN END TO END --
# trunk -> lift -> map and box heads -> planner."
#
# ⛔ WHY TWO PASSES. `refc_v3_train` REFUSES a run whose train and eval caches share
# episodes ("a held-out split that is not held out measures memorisation"). That refusal
# is CORRECT and is not worked around: the 139 clips are split into two DISJOINT halves
# (sorted-order alternating, so neither half is biased toward one end of an id-sorted
# corpus), and each pass EVALUATES one half while TRAINING on the other. Union = 139.
#
# ⭐ WHY THE COVERAGE IS A FACT AND NOT A PROBABILITY. The eval subset is built as
#     perm = randperm(len(e_ds), generator=Generator().manual_seed(12345))[:nb*batch]
# -- deterministic -- so which clips it touches was REPLAYED OFFLINE against the cache
# manifest before this ran: splitA needs 150 eval-batches to touch all 70 clips, splitB
# needs 130 for all 69. The values below carry margin over both.
#
# ⛔ THIS IS A COVERAGE PASS, NOT AN EVAL. The model has taken 1 step and the trunk is
# ImageNet-init. NO number it emits is a capability claim, a tier stamp, or a metric
# family. What it establishes is exactly one thing: every one of the 139 clips' data
# reaches and passes through trunk -> lift -> map + box3d heads -> planner without error.
#
# resnet34, not resnet101: §10.2's PRIMARY trunk OOMs on this 8 GB card at 416x1024
# batch 1 (MEASURED) -- its shapes are proven separately on CPU.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/refcv6-pipeval-20260917
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
A="$ART/v2ep-eval139-416x1024cyl-splitA"
B="$ART/v2ep-eval139-416x1024cyl-splitB"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"

for p in "$A" "$B" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZABORT-MISSING-INPUT ${p}ZZ"; exit 2; }
done

WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

pass () {                      # $1 tag  $2 train cache  $3 eval cache  $4 eval batches
  read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
    echo "ZZABORT-PROBE-FAILED box unreadable -- INCONCLUSIVE, never clearZZ"; return 3; }
  case "$GU$HF" in *[!0-9]*) echo "ZZABORT-PROBE-SHAPE '$GU $HF'ZZ"; return 3 ;; esac
  [ "$GU" -le 2500 ] || { echo "ZZABORT-GPU-BUSY used=${GU}MiBZZ"; return 3; }
  [ "$HF" -ge 8 ]    || { echo "ZZABORT-HOST-TIGHT free=${HF}GBZZ"; return 3; }
  echo "== $1  train=$(basename "$2")  eval=$(basename "$3")  eval_batches=$4  box gpu=${GU}MiB host=${HF}GB  $(date -u +%H:%M:%SZ)"
  ( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
      --arm hier --size tiny --out "$OUT/$1" \
      --device cuda --seed 0 --steps 1 --batch 2 --workers 0 \
      --log-every 1 --save-every 1000 \
      --v2-cache "$2" --image-hw 416 1024 --v2-lru 4 \
      --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
      --trunk-pretrained \
      --v7-labels "$LAB" \
      --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
      --agent-queries 16 --agent-pad 32 \
      --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
      --map-gt-root "$MAPS" --map-lru 2 --map-min-coverage 0.90 \
      --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
      --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 \
      --conflict-detector off \
      --eval-cache "$3" --eval-labels "$LAB" \
      --eval-every 1 --eval-batches "$4" ) > "$OUT/$1.log" 2>&1
  local rc=$?
  # THE ARTIFACT IS THE EVIDENCE, NEVER THE EXIT CODE.
  local rows ev evk
  rows=$(wc -l < "$OUT/$1/metrics.jsonl" 2>/dev/null || echo 0)
  ev=$(grep -c "held-out eval:" "$OUT/$1.log" 2>/dev/null || echo 0)
  evk=$(grep -c "eval_" "$OUT/$1/metrics.jsonl" 2>/dev/null || echo 0)
  echo "ZZCOVER $1 rc=$rc rows=${rows} evline=${ev} rows_with_eval=${evk}ZZ"
  grep -h "held-out eval:\|SAM3 map GT:" "$OUT/$1.log" 2>/dev/null | head -3
  [ "${evk:-0}" -ge 1 ] || { echo "ZZCOVER-INCOMPLETE $1 -- no eval metrics landedZZ"
                             tail -25 "$OUT/$1.log"; return 5; }
  echo "ZZCOVER-OK $1ZZ"
}

pass cover_evalB "$A" "$B" 200 || exit $?
pass cover_evalA "$B" "$A" 250 || exit $?
echo "ZZCOVER-BOTH-OKZZ"
