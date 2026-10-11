#!/bin/bash
# SFT-4R labels on a fresh pod (eval/PREREG_SFT4R.md "Changed 1"; STOP_CONSEQUENCES_2026-10-11.md "what has to be redone").
# Deploy: copy refe/ to /workspace/refe-sft4r/run (the dir sft4r_pod.sh runs from) and the two held-out lane .xz files
# below to /workspace/refe-sft4r/raw/2026-10-04-lane-discipline/, restore the banks from HF
# Sayood/tanitad-refe-navtrain-banks, then:  nohup bash sft4r_relabel.sh > /workspace/data/refe_sft2/relabel.out 2>&1 &
#
#  1. held-out lane labels: RESTORED from the repo (raw/2026-10-04-lane-discipline/heldout_{navsim,teacher}_lane_labels
#     .jsonl.xz, 3,137 rows each, the labels SFT-4 trained against), not recomputed. The trainer globs lane_*.jsonl and
#     teacher_lane_r*.jsonl; G8/G10 then prove they are served (stale <= 5 %).
#  2. PDM train + held-out, --served-only (one label per served set, the newest ckpt_step). The relabeller resumes by
#     (key, ckpt_step), so a restart redoes nothing; its last DONE marker is what pdm_label_complete.py audits.
#  3. the 15,000-set lane subset (select_bank_subset.py, seed 20261004) and the two-rank teacher-lane relabel.
#     NAMED GAP: the --queue source of SFT-4's subset was never written into a banked script. The HF banks carry an
#     `onpolicy_queue`; LANE_QUEUE_SRC must name the restored dir, and nothing is guessed when it is unset.
# The PDM held-out pass is --served-only by ASSUMPTION (SFT-4's own command line was not banked); the chain guards it
# with --expect 3137, so a wrong assumption stops the chain instead of training on it.
set -u
D=/workspace/data/refe_sft2
L=$D/relabel_sft4r.log
REPO_RAW=/workspace/refe-sft4r/raw/2026-10-04-lane-discipline
log() { echo "$(date -u +%FT%T) $*" >> $L; }
mkdir -p $D
source /workspace/teacher_env.sh
export NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs OMP_NUM_THREADS=1
echo 1000 > /proc/self/oom_score_adj
cd /workspace/refe-sft4r/run
SETS=/workspace/data/refe_navtrain/onpolicy/sets
HSETS=/workspace/data/refe_heldout/sets
for p in $SETS $HSETS /workspace/data/refe_heldout/heldout_logs.txt /workspace/data/navtrain_dbs; do
  [ -e "$p" ] || { log "REFUSED: $p missing (restore the banks first)"; exit 1; }
done
log "relabel armed: sets $(ls $SETS | wc -l) files, held-out $(ls $HSETS | wc -l) files"

# 1. held-out lane labels from the repo
for pair in "heldout_navsim_lane_labels heldout_lane lane_heldout_restored" \
            "heldout_teacher_lane_labels heldout_teacher_lane teacher_lane_r0_restored"; do
  set -- $pair
  mkdir -p $D/$2
  if ls $D/$2/*.jsonl >/dev/null 2>&1; then log "$2 present ($(cat $D/$2/*.jsonl | wc -l) rows), not restored"
  else
    [ -f $REPO_RAW/$1.jsonl.xz ] || { log "REFUSED: $REPO_RAW/$1.jsonl.xz missing"; exit 1; }
    xz -dc $REPO_RAW/$1.jsonl.xz > $D/$2/$3.jsonl.part && mv $D/$2/$3.jsonl.part $D/$2/$3.jsonl
    log "$2 restored from the repo: $(wc -l < $D/$2/$3.jsonl) rows"
  fi
done

# 2. PDM labels (train, then held-out)
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_pdm.py --sets $SETS --out $D/train_pdm --served-only \
  >> $D/pdm_train.log 2>&1 200>&-
echo "ZZPDM_TRAIN_EXIT $?" >> $D/pdm_train.log
log "PDM train: $(tail -2 $D/pdm_train.log | tr '\n' ' ')"
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_pdm.py --sets $HSETS --out $D/heldout_pdm --served-only \
  >> $D/pdm_heldout.log 2>&1 200>&-
echo "ZZPDM_HELDOUT_EXIT $?" >> $D/pdm_heldout.log
log "PDM held-out: $(tail -2 $D/pdm_heldout.log | tr '\n' ' ')"

# 3. teacher-lane train labels on the 15,000-set subset
Q=$D/train_subset_queue
O=$D/train_teacher_lane
if ! ls $Q/props_r*_subset.jsonl >/dev/null 2>&1; then
  [ -n "${LANE_QUEUE_SRC:-}" ] && [ -d "$LANE_QUEUE_SRC" ] || {
    log "REFUSED: lane subset queue absent and LANE_QUEUE_SRC unset or missing (named gap, see header)"
    echo "ZZLANE_TRAIN_EXIT 9" >> $D/tl_train.log; exit 1; }
  $DRIVERL_EVAL_PYTHON select_bank_subset.py --sets $SETS --queue "$LANE_QUEUE_SRC" \
    --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --n 15000 --seed 20261004 --out $Q >> $L 2>&1
  log "subset: $(grep ZZSUBSET $L | tail -1)"
fi
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_teacher_lane.py --queue $Q --files 'props_r0_*' --rank 0 --out $O \
  --workers 5 --repair-last-heading >> $D/tl_train_r0.log 2>&1 200>&- &
P0=$!
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_teacher_lane.py --queue $Q --files 'props_r1_*' --rank 1 --out $O \
  --workers 4 --repair-last-heading >> $D/tl_train_r1.log 2>&1 200>&- &
P1=$!
wait $P0; R0=$?; wait $P1; R1=$?
echo "ZZLANE_TRAIN_EXIT $R0 $R1" >> $D/tl_train.log
log "lane train: rc $R0 $R1, rows $(cat $O/*.jsonl 2>/dev/null | wc -l)"
log "ZZRELABEL_SFT4R_DONE"
