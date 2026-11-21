#!/usr/bin/env bash
# PREREG_V8F_SPEEDREAD s8 verbatim (dev-box CPU), started by pull_speedread.sh after ZZSPEEDREAD-PULL-DONE. Tree = a git archive of the
# promotion commit f3971165 (stack/ + SPEEDCAL's banked run/eval). Each stage is skipped when its .done exists: resume = run again.
# Markers (in F:/TanitAD-scratch/speedread/run.out): ZZSPEEDREAD-RUN-DONE <outcome line> | ZZSPEEDREAD-RUN-STOP <why>.
set -u
export CUDA_VISIBLE_DEVICES=-1 SPEEDREAD_THREADS=8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/speedread; SF=E:/Projects/TanitAD-artifacts/sfresh_inputs
TREE=F:/TanitAD-scratch/speedread/tree_f3971165
T=$TREE/stack/scripts/v8f_mini_eval_speedread.py
WRAP=E:/Projects/TanitAD-artifacts/speedread_run/run_with_taniteval.py      # run mechanics: puts $TREE/taniteval on sys.path under -I (the first extract stopped on "No module named taniteval")
PK="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
X0=C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt
H0=c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d
X1=C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt
H1=68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
stop() { log "ZZSPEEDREAD-RUN-STOP $*"; exit 3; }
log "run start"
grep -aq ZZSPEEDREAD-PULL-DONE /f/TanitAD-scratch/speedread/pull.out || stop "the pull has not written ZZSPEEDREAD-PULL-DONE"
[ "$(sha256sum < $T | cut -c1-64)" = d554575169ea393cf5783959d354ca6a5a0acd859e583b0b4e1f520fbb776119 ] || stop "instrument sha256 differs from the PREREG's pinned file"
[ "$(sha256sum < $X0 | cut -c1-64)" = $H0 ] && [ "$(sha256sum < $X1 | cut -c1-64)" = $H1 ] || stop "an X_s checkpoint differs from its sha256"
free=$(df -m /f | tail -1 | awk '{print $4}'); [ "$free" -gt 15000 ] || stop "F: has $free MB free"
stage() {   # stage <name> <cmd...>
  local n=$1; shift
  [ -f $W/.done_$n ] && { log "$n: done earlier, skipped"; return 0; }
  log "$n: start"
  "$@" > $W/$n.log 2>&1 || stop "$n rc=$? (log $W/$n.log)"
  touch $W/.done_$n; log "$n: done ($(tail -1 $W/$n.log | cut -c1-200))"
}
stage extract "$PY" -I $WRAP $TREE $T extract --cache $W/cache --join C:/Users/Admin/ar3_work/dc/join_train.npz --tracks $SF/inputs/native_tracks.pt --out $W/train
stage fit "$PY" -I $WRAP $TREE $T fit --train $W/train --eval-manifest $SF/cache/eval/manifest.jsonl --out $W/fit
stage evals_s0 "$PY" -I $WRAP $TREE $T evals --seed 0 --x $X0 --expect-sha $H0 --readout $W/fit/readout.pt --fit-json $W/fit/fit.json \
  --eval-dir $W/eval_s0 --eval-cache $SF/cache --tracks $SF/inputs/native_tracks.pt --v9-eval $SF/inputs/v9/v9_labels_eval139.npz \
  --lead-eval $SF/inputs/v9/lead_eval139.npz --speed-eval $SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl --p10 $SF/inputs/p10/p10_prior.pt
stage evals_s1 "$PY" -I $WRAP $TREE $T evals --seed 1 --x $X1 --expect-sha $H1 --readout $W/fit/readout.pt --fit-json $W/fit/fit.json \
  --eval-dir $W/eval_s1 --eval-cache $SF/cache --tracks $SF/inputs/native_tracks.pt --v9-eval $SF/inputs/v9/v9_labels_eval139.npz \
  --lead-eval $SF/inputs/v9/lead_eval139.npz --speed-eval $SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl --p10 $SF/inputs/p10/p10_prior.pt
stage read "$PY" -I $WRAP $TREE $T read --fit-json $W/fit/fit.json --eval-dir-s0 $W/eval_s0 --eval-dir-s1 $W/eval_s1 \
  --speedcal-dir-s0 "$PK/raw/2026-10-25-speedcal/run/eval" --speedcal-dir-s1 "$PK/raw/2026-10-25-speedcal/run/eval" --out $W/read
log "ZZSPEEDREAD-RUN-DONE $(cat $W/read/OUTCOME.txt 2>/dev/null | head -3 | tr '\n' ' ')"
