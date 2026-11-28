#!/bin/bash
# n1dm_dumps.sh -- the D1-HO CPU dumps of both N1DM arms, run by n1dm_chain.sh after both arms, GPU lock still held (so nothing else trains meanwhile). Adapted from n1dh_dumps.sh (which ran).
# First the two checkpoint copies per arm (n1dm_zuse_ckpt.py; the trained checkpoints are not touched):
#   zuse = the K = 3 branches zeroed (plan = head_o alone: the branch residual removed, the "posterior zeroed" ablation), as runs/N1DM_s<k>_zuse/train/ckpt_last.pt
#   unif = route_head weight and bias zeroed (p = uniform 1/3 on every row: ONE shared mixture of the 3 branches, the "no per-row selection" control), as runs/N1DM_s<k>_unif/train/ckpt_last.pt
# Then 10 dumps, three at a time (2 threads each):
#   executor of record hb0.2: N1DM_s0/s1 follow (the real arm), N1DM_s0/s1 zero (the D-channel zero twin, --nav-rb-variant zero), N1DM_s0_zuse/s1_zuse follow, N1DM_s0_unif/s1_unif follow;   old executor (kappa none), beside: N1DM_s0/s1 follow.
# A dump with DONE_<arm> is skipped (resume). Exit 0 = all 10 DONE. Sentinel ZZ DUMPS DONE ZZ in out/DUMPS_DONE.
N=/root/strat1/n1dm
mkdir -p $N/out
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
for S in s0 s1; do
  for MODE in zuse unif; do
    D=$N/runs/N1DM_${S}_${MODE}/train; mkdir -p $D
    if [ ! -s $D/ckpt_last.pt ]; then
      taskset -c 48-95 nice -n 10 /root/venv/bin/python $N/n1dm_zuse_ckpt.py $MODE $N/runs/N1DM_$S/train/ckpt_last.pt $D/ckpt_last.pt > $N/out/${MODE}_$S.log 2>&1 \
        || { rm -f $D/ckpt_last.pt; echo "$MODE COPY FAIL N1DM_$S: $(tail -c 300 $N/out/${MODE}_$S.log)"; exit 1; }
      cat $N/out/${MODE}_$S.log
    fi
  done
done
# the dump list: "ARM VARIANT KAPPA | out dir (relative, for the DONE check)"
need() { [ -e $N/out/$2/DONE_$1 ]; }
LIST=$N/out/dumplist.txt; : > $LIST
ALL=$N/out/dumpall.txt; : > $ALL
for S in s0 s1; do
  echo "N1DM_$S follow 0.2 hb0.2/t1 N1DM_$S" >> $ALL
  echo "N1DM_$S zero 0.2 hb0.2/t1_zero N1DM_$S" >> $ALL
  echo "N1DM_${S}_zuse follow 0.2 hb0.2/t1 N1DM_${S}_zuse" >> $ALL
  echo "N1DM_${S}_unif follow 0.2 hb0.2/t1 N1DM_${S}_unif" >> $ALL
  echo "N1DM_$S follow none hbnone/t1 N1DM_$S" >> $ALL
done
while read a v k d n; do need $n $d || echo "$a $v $k" >> $LIST; done < $ALL
echo "$(date -u +%T) $(wc -l < $LIST) of $(wc -l < $ALL) dumps to run"
[ -s $LIST ] && { xargs -P 3 -L 1 bash $N/hb_dump_n1dm.sh < $LIST; }
bad=0
while read a v k d n; do need $n $d || { echo "MISSING DONE $d $n"; bad=1; }; done < $ALL
[ $bad = 0 ] || exit 1
echo "ZZ DUMPS DONE ZZ" > $N/out/DUMPS_DONE
