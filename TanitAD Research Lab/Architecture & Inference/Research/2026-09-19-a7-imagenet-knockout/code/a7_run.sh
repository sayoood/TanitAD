#!/usr/bin/env bash
# GATE RAISED 2,500 -> 4,300 MiB on the PI's direct authorisation (Sayed, 2026-09-20),
# on MEASURED evidence, not to make a blocked chain move. The desktop alone holds
# 3,111-3,958 MiB (19 samples), so the old ceiling could NEVER clear (0/19 over ~9.4 h)
# while rejecting a configuration that in fact runs. The arm's own peak is 2,573 MiB
# allocated / ~2,939 MiB on the card, so worst-case 3,958 + 2,939 = 6,897 of 8,188 leaves
# 1,291 MiB. Evidence: .../2026-09-19-s1-collision-gate/raw/vram_probe/ .
# DO NOT restore the old ceiling: on this box it is not safer, it is unsatisfiable.
# A7 -- resnet34 trunk, ImageNet-init vs random-init, 2 seeds each = 4 arms, ONE AT A TIME.
# Pre-registered BEFORE any data: Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md,
# "A7 AMENDMENT, 2026-09-19" (A7.1-A7.8). Configuration = C:/Users/Admin/qland/a3_heldout_read.sh
# with EXACTLY the A7.5 changes; nothing else differs between the four arms (audited after the
# panel from each config.json argv by a7_analyze.py).
#
# RUN TREE (pinned, never edited under a live arm): C:/Users/Admin/tanitad-a7-run
#   = `git archive` of HEAD 37645fcc61b1 (stack/ + taniteval/) + the two STAGED blobs
#     stack/tanitad/models/timm_trunk.py d74ad535c41f, stack/scripts/refc_v3_train.py 87e222388736
#
# ⛔ GPU RULE (the Master Mind's): before EACH launch boxstat.py must read GPU <= 4,300 MiB and
# host free >= 8 GB. An unreadable probe is INCONCLUSIVE, never clear -- the gate keeps waiting.
# ⭐ RESUMABLE: an arm whose a7_arm_check.json says VALID is skipped; a partial arm directory is
# moved aside (never deleted) and the arm re-runs from scratch.
# ⭐ ORDER: IN-s0, RND-s0, IN-s1, RND-s1 -- conditions interleaved, so a panel cut short after
# two arms still holds one seed-matched IN/RND pair.
# Markers are ZZA7...ZZ tokens: a monitor must parse those, never grep the words it searches for.
set -u
RUNT=/c/Users/Admin/tanitad-a7-run
RUNW='C:\Users\Admin\tanitad-a7-run'
OUT=/c/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
BOX=/c/Users/Admin/qland/boxstat.py
ART=/d/Projects/TanitAD-artifacts
LAB_SRC=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
LAB_MD5=eefc38d1453bd1c73802d44d45affced
A=$ART/v2ep-eval124clean-416x1024cyl-halfA
B=$ART/v2ep-eval124clean-416x1024cyl-halfB
MAPS=$ART/sam3-maps-eval
JOIN=$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz
EXTR=$ART/refcv5v2_final/extrinsics141.json
CODE_SRC="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$RUNW\\stack;$RUNW;$RUNW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

mkdir -p "$OUT/inputs" "$OUT/code"
for p in "$A" "$B" "$MAPS" "$JOIN" "$EXTR" "$RUNT/stack/scripts/refc_v3_train.py" "$BOX"; do
  [ -e "$p" ] || { echo "ZZA7-ABORT-MISSING-INPUT ${p}ZZ"; exit 2; }
done
# the v8 eval labels live only in the tanitad-wt MIRROR, whose resync deletes repo-absent
# files -- pin a verified copy so arm 4 reads the same bytes as arm 1.
LAB="$OUT/inputs/s2_labels_v8_eval.jsonl.gz"
if [ ! -f "$LAB" ]; then cp "$LAB_SRC" "$LAB" || { echo "ZZA7-ABORT-LABEL-COPYZZ"; exit 2; }; fi
m=$(md5sum < "$LAB" | cut -c1-32)
[ "$m" = "$LAB_MD5" ] || { echo "ZZA7-ABORT-LABEL-MD5 ${m}ZZ"; exit 2; }
# pin the gate + verdict code next to the outputs
cp "$CODE_SRC/a7_check_arm.py" "$CODE_SRC/a7_analyze.py" "$OUT/code/" || {
  echo "ZZA7-ABORT-CODE-COPYZZ"; exit 2; }
( cd "$OUT/code" && md5sum a7_check_arm.py a7_analyze.py > MD5SUMS.txt )
md5sum "$RUNT/stack/scripts/refc_v3_train.py" "$RUNT/stack/tanitad/models/timm_trunk.py" \
  > "$OUT/code/RUNTREE_MD5SUMS.txt"

gate() {   # $1 = max wait in seconds
  local waited=0 bad=0 GU HF line
  while :; do
    line=$("$PY" "$BOX" 2>/dev/null)
    read -r GU HF <<< "$line"
    case "${GU:-x}${HF:-x}" in
      *[!0-9]*) bad=$((bad + 1)); echo "ZZA7-GATE-PROBE-INCONCLUSIVE '${line}'ZZ" ;;
      *) if [ "$GU" -le 4300 ] && [ "$HF" -ge 8 ]; then
           echo "ZZA7-GATE-CLEAR gpu=${GU}MiB host=${HF}GB $(date -u +%H:%M:%SZ)ZZ"; return 0
         fi
         [ $((waited % 900)) -eq 0 ] && echo "ZZA7-GATE-WAIT gpu=${GU}MiB host=${HF}GB waited=${waited}sZZ" ;;
    esac
    [ "$waited" -ge "$1" ] && { echo "ZZA7-GATE-TIMEOUT after ${waited}sZZ"; return 1; }
    sleep 60; waited=$((waited + 60))
  done
}

run_arm() {   # $1 arm  $2 pretrained(0|1)  $3 seed  $4 max gate wait (s)
  local arm=$1 pre=$2 seed=$3 D="$OUT/$1" flag st RC T0 T1
  if [ -f "$D/a7_arm_check.json" ] && grep -q '"status": "VALID"' "$D/a7_arm_check.json"; then
    echo "ZZA7-SKIP-${arm}-ALREADY-VALIDZZ"; return 0
  fi
  if [ -e "$D" ]; then mv "$D" "$D.aborted-$(date +%s)"; echo "ZZA7-MOVED-ASIDE-${arm}ZZ"; fi
  gate "$4" || return 3
  mkdir -p "$D"
  flag=--no-trunk-pretrained; [ "$pre" = 1 ] && flag=--trunk-pretrained
  echo "ZZA7-LAUNCH-${arm} $(date -u +%H:%M:%SZ)ZZ"
  T0=$(date +%s)
  ( cd "$RUNT/stack" && "$PY" -u scripts/refc_v3_train.py \
      --arm hier --size tiny --out "$D/run" \
      --device cuda --seed "$seed" --steps 2000 --batch 2 --workers 0 \
      --log-every 10 --save-every 250 \
      --v2-cache "$A" --image-hw 416 1024 --v2-lru 4 \
      --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
      "$flag" \
      --trunk-chunk-ckpt 1 --trunk-frozen-bn \
      --trunk-bn-recalib 256 --trunk-bn-recalib-seed 0 \
      --v7-labels "$LAB" \
      --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
      --agent-queries 16 --agent-pad 32 \
      --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
      --map-gt-root "$MAPS" --map-lru 4 --map-min-coverage 0.90 \
      --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
      --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 \
      --conflict-detector off --eval-cache "$B" --eval-labels "$LAB" \
      --eval-every 2000 --eval-batches 500 \
      --eval-window-dump "$D/eval_windows.jsonl" ) > "$D/train.log" 2>&1
  RC=$?; T1=$(date +%s)
  echo "ZZA7-TRAIN-RC-${arm} ${RC} wall=$((T1 - T0))sZZ"
  "$PY" "$OUT/code/a7_check_arm.py" "$D" --pretrained "$pre" --seed "$seed" > "$D/check.out" 2>&1
  # ⛔ the admissible evidence is the ARTIFACT, never the exit code
  [ -f "$D/a7_arm_check.json" ] || { echo "ZZA7-CHECK-MISSING-${arm}ZZ"; return 4; }
  st=$(grep -o '"status": "[A-Z]*"' "$D/a7_arm_check.json" | head -1 | cut -d'"' -f4)
  echo "ZZA7-ARM-${arm}-${st:-UNKNOWN}ZZ"
  cat "$D/check.out"
  [ "$st" = "VALID" ] || return 5
}

echo "ZZA7-PANEL-START $(date -u +%Y-%m-%dT%H:%M:%SZ)ZZ"
run_arm A7-IN-s0  1 0 36000 || { echo "ZZA7-PANEL-STOPPED-AT-A7-IN-s0ZZ";  exit 10; }
run_arm A7-RND-s0 0 0 7200  || { echo "ZZA7-PANEL-STOPPED-AT-A7-RND-s0ZZ"; exit 11; }
run_arm A7-IN-s1  1 1 7200  || { echo "ZZA7-PANEL-STOPPED-AT-A7-IN-s1ZZ";  exit 12; }
run_arm A7-RND-s1 0 1 7200  || { echo "ZZA7-PANEL-STOPPED-AT-A7-RND-s1ZZ"; exit 13; }
"$PY" "$OUT/code/a7_analyze.py" "$OUT" --out "$OUT/a7_verdict.json" > "$OUT/verdict.out" 2>&1
[ -f "$OUT/a7_verdict.json" ] && echo "ZZA7-VERDICT-WRITTENZZ" || echo "ZZA7-VERDICT-MISSINGZZ"
cat "$OUT/verdict.out"
echo "ZZA7-PANEL-END $(date -u +%Y-%m-%dT%H:%M:%SZ)ZZ"
