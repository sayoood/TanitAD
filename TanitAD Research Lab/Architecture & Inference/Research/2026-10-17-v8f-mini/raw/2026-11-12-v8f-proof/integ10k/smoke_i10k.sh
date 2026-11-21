#!/usr/bin/env bash
# smoke_i10k.sh -- ON THE POD, after staging: wrapper forms R0 and RRid on UE_s0, a planted constant reader (presence 0.88, gap 25 m), at most 20 windows, at most 300 s each. Output only under build/smoke.
set -u
W=/workspace/v8proof/integ10k; B=$W/build; S=$B/smoke3; TREE=$B/tree_ov; V=/workspace/v8proof; PY=/workspace/venv/bin/python
mkdir -p "$S" || exit 2
[ -e "$S/R0" ] && { echo "REFUSED: smoke output exists"; exit 2; }
CUDA_VISIBLE_DEVICES= $PY $B/make_planted_reader.py $S/planted.pt >> $S/smoke.log 2>&1 || { echo "STOP planted" >> $S/smoke.log; echo STOP > $S/SMOKE_DONE.txt; exit 3; }
for form in R0 RRid; do
  out=$S/$form; mkdir -p $out/ticks
  extra=(); [ $form = RRid ] && extra=(--reader $S/planted.pt --rule-log $out/ticks/rule_log_$form.npz)
  echo "[$(date -u +%H:%M:%SZ)] $form start" >> $S/smoke.log
  DISTRULE_DIR=$B PYTHONPATH=$TREE/stack timeout 300 $PY $B/v8f_t1_eval_reader_rule.py --tree $TREE --form $form "${extra[@]}" run --ckpt $W/ckpt/UE_s0/A/ckpt_last.pt --tag SMK${form}_m10_causal_heads \
    --tracks $V/data/inputs/native_tracks.pt --eval-cache $W/eval_cache --eval-grid $V/hierrep/pins/eval_grid_windows.json --eval-join $V/hierrep/pins/join_eval141.npz --out-dir $out \
    --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks $out/ticks --dump-heads --device cuda --max-windows 20 > $out/t1.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] $form rc $?" >> $S/smoke.log
done
echo DONE > $S/SMOKE_DONE.txt
