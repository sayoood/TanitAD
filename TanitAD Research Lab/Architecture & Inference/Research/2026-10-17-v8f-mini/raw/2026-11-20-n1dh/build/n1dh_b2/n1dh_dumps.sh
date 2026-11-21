#!/bin/bash
# n1dh_dumps.sh -- the D1-HO CPU dumps of both N1DH arms, run by n1dh_chain.sh after both arms, GPU lock still held (so nothing else trains meanwhile). Adapted from n1dc_dumps.sh (which ran).
# First the Z-USE checkpoint copies (route_emb.weight = 0, n1dh_zuse_ckpt.py; the trained checkpoints are not touched) as runs/N1DH_s<k>_zuse/train/ckpt_last.pt. Then 8 dumps, three at a time (2 threads each):
#   executor of record hb0.2: N1DH_s0/s1 follow (the real arm), N1DH_s0/s1 zero (the D-channel zero twin, --nav-rb-variant zero), N1DH_s0_zuse/s1_zuse follow (the posterior zeroed);   old executor (kappa none), beside: N1DH_s0/s1 follow.
# A dump with DONE_<arm> is skipped (resume). Exit 0 = all 8 DONE. Sentinel ZZ DUMPS DONE ZZ in out/DUMPS_DONE.
N=/root/strat1/n1dh
mkdir -p $N/out
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
for S in s0 s1; do
  D=$N/runs/N1DH_${S}_zuse/train; mkdir -p $D
  if [ ! -s $D/ckpt_last.pt ]; then
    taskset -c 48-95 nice -n 10 /root/venv/bin/python $N/n1dh_zuse_ckpt.py $N/runs/N1DH_$S/train/ckpt_last.pt $D/ckpt_last.pt > $N/out/zuse_$S.log 2>&1 \
      || { rm -f $D/ckpt_last.pt; echo "ZUSE COPY FAIL N1DH_$S: $(tail -c 300 $N/out/zuse_$S.log)"; exit 1; }
    cat $N/out/zuse_$S.log
  fi
done
# the dump list: "ARM VARIANT KAPPA | out dir (relative, for the DONE check)"
need() { [ -e $N/out/$2/DONE_$1 ]; }
LIST=$N/out/dumplist.txt; : > $LIST
ALL=$N/out/dumpall.txt; : > $ALL
for S in s0 s1; do
  echo "N1DH_$S follow 0.2 hb0.2/t1 N1DH_$S" >> $ALL
  echo "N1DH_$S zero 0.2 hb0.2/t1_zero N1DH_$S" >> $ALL
  echo "N1DH_${S}_zuse follow 0.2 hb0.2/t1 N1DH_${S}_zuse" >> $ALL
  echo "N1DH_$S follow none hbnone/t1 N1DH_$S" >> $ALL
done
while read a v k d n; do need $n $d || echo "$a $v $k" >> $LIST; done < $ALL
echo "$(date -u +%T) $(wc -l < $LIST) of $(wc -l < $ALL) dumps to run"
[ -s $LIST ] && { xargs -P 3 -L 1 bash $N/hb_dump_n1dh.sh < $LIST; }
bad=0
while read a v k d n; do need $n $d || { echo "MISSING DONE $d $n"; bad=1; }; done < $ALL
[ $bad = 0 ] || exit 1
echo "ZZ DUMPS DONE ZZ" > $N/out/DUMPS_DONE
