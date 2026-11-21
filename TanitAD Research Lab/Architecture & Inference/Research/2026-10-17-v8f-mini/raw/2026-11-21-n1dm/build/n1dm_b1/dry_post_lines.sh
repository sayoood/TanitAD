#!/bin/bash
# dry_post_lines.sh TINYDIR -- run the THREE post-training lines of pod/n1dm_arm.sh (post check, zero-grad gate, K-CLIP) ONCE, as the script writes them (extracted with sed, evaluated), on the REAL finished tiny arm made by
# mk_tiny_arm.py (N1DH Amendments 4 and 5: a line that runs only after training is reached by neither bash -n nor a dry mint). CPU, one python at a time (commit gate first). Output: logs/dry_post_lines.log.
T=$1; B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; L=$B/logs/dry_post_lines.log; : > $L
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 >> $L 2>&1 || exit 9
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 TMP=$B/tmp TEMP=$B/tmp
TREE=$B/tree; N=$T/podN; RUNS=$T/podruns; tag=N1DM_s0; kind=N1DM; seed=s0
CHK=$TREE/stack/scripts/strat1_check.py; ZG=$TREE/stack/scripts/zerograd_gate.py; init=$T/init/ckpt_last.pt
mkdir -p $N $RUNS/$tag/train && cp $B/pod_ref/kclip.py $N/kclip.py && cp $T/arm/ckpt_last.pt $T/arm/train_log.jsonl $T/arm/summary.json $RUNS/$tag/train/
fail() { echo "ZZ ARM $tag FAIL ZZ $1" >> $L; exit 1; }
A=$B/pod/n1dm_arm.sh
for pat in 'CHK post \$kind' 'ZG --ckpt' 'kclip.py'; do
  ln=$(grep -F -- "$(echo "$pat" | sed 's/\//g')" $A | head -1)
  [ -n "$ln" ] || { echo "no line for $pat" >> $L; exit 2; }
  echo "LINE: $ln" | cut -c1-200 >> $L
  eval "$ln" >> $L 2>&1; rc=$?
  echo "rc=$rc  <- $pat" >> $L
done
echo "--- post log:"; cat $RUNS/post_$tag.log 2>/dev/null | tail -3 >> $L; echo "--- zerograd:" >> $L; tail -8 $RUNS/zerograd_$tag.log >> $L; echo "--- kclip:" >> $L; head -c 400 $RUNS/$tag/kclip_$tag.json >> $L; echo >> $L
echo "ZZ DRYPOST DONE ZZ" >> $L
