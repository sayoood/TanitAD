#!/bin/bash
# dry_post_lines2.sh TINYDIR -- MM 2026-11-21, the repaired dry_post_lines.sh. The first version extracted the lines with `sed 's/\//g'`, which is an
# unterminated s command: the pattern came out EMPTY, `grep -F ""` matched line 1 (#!/bin/bash), and the script evaluated that three times and wrote rc=0.
# This version greps the literal command prefix, refuses a line that does not contain it, fails on any non-zero rc, and requires the three outputs to exist.
# It runs the THREE post-training lines of pod/n1dm_arm.sh (post check, zero-grad gate, K-CLIP) ONCE, as the script writes them, on the REAL finished tiny arm.
T=$1; B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; L=$B/logs/dry_post_lines2.log; : > $L
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 >> $L 2>&1 || exit 9
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 TMP=$B/tmp TEMP=$B/tmp
TREE=$B/tree; N=$T/podN; RUNS=$T/podruns; tag=N1DM_s0; kind=N1DM; seed=s0
CHK=$TREE/stack/scripts/strat1_check.py; ZG=$TREE/stack/scripts/zerograd_gate.py; init=$T/init/ckpt_last.pt
rm -rf "${T:?}/podruns"
mkdir -p $N $RUNS/$tag/train && cp $B/pod_ref/kclip.py $N/kclip.py && cp $T/arm/ckpt_last.pt $T/arm/train_log.jsonl $T/arm/summary.json $RUNS/$tag/train/ || { echo "ZZ DRYPOST2 FAIL ZZ staging" >> $L; exit 5; }
fail() { echo "ZZ ARM $tag FAIL ZZ $1" >> $L; echo "ZZ DRYPOST2 FAIL ZZ a post line called fail" >> $L; exit 1; }
A=$B/pod/n1dm_arm.sh
for pat in '$PY $CHK post $kind' '$PY $ZG --ckpt' '$PY $N/kclip.py'; do
  ln=$(grep -F -- "$pat" $A | head -1)
  case "$ln" in *"$pat"*) ;; *) echo "ZZ DRYPOST2 FAIL ZZ no line containing: $pat" >> $L; exit 2;; esac
  echo "LINE: $ln" | cut -c1-240 >> $L
  eval "$ln" >> $L 2>&1; rc=$?
  echo "rc=$rc  <- $pat" >> $L
  [ $rc = 0 ] || { echo "ZZ DRYPOST2 FAIL ZZ rc $rc on: $pat" >> $L; exit 3; }
done
for f in $RUNS/post_$tag.log $RUNS/$tag/zerograd_n1dm.json $RUNS/$tag/kclip_$tag.json; do
  [ -s "$f" ] || { echo "ZZ DRYPOST2 FAIL ZZ missing or empty output $f" >> $L; exit 4; }
done
echo "--- post log:" >> $L; tail -5 $RUNS/post_$tag.log >> $L
echo "--- zerograd:" >> $L; tail -12 $RUNS/zerograd_$tag.log >> $L
echo "--- zerograd json:" >> $L; head -c 1500 $RUNS/$tag/zerograd_n1dm.json >> $L; echo >> $L
echo "--- kclip:" >> $L; head -c 600 $RUNS/$tag/kclip_$tag.json >> $L; echo >> $L
echo "ZZ DRYPOST2 PASS ZZ" >> $L
