#!/bin/bash
# MM 2026-11-21: pull the N1DM + N1DH D1-HO dumps and the two N1DM checkpoints (HANDOFF_READ_N1DM "Commands"). Pod: tar + pscp only; the pod tars are removed after the
# local copy is verified by sha256 (the pod sources stay). Local tars removed after extraction. Every removal logged in F:/TanitAD-scratch/n1dc/DELETIONS.log.
set -euo pipefail
source C:/Users/Admin/lorapod/podenv.sh
H=F:/TanitAD-scratch/n1dc/read_n1dm
LOGD=F:/TanitAD-scratch/n1dc/DELETIONS.log
mkdir -p $H/in/n1dm $H/in/n1dh $H/ckpts/N1DM_s0 $H/ckpts/N1DM_s1 $H/tmp_pull
pssh 'set -e; mkdir -p /root/strat1/n1dm_readtmp; X="--exclude=ticks_* --exclude=log_* --exclude=DONE_* --exclude=hb_* --exclude=t1v8f_*"
  cd /root/strat1/n1dm/out && tar cf /root/strat1/n1dm_readtmp/n1dm.tar $X hb0.2 hbnone
  cd /root/strat1/n1dh/out && tar cf /root/strat1/n1dm_readtmp/n1dh.tar $X hb0.2 hbnone
  cd /root/strat1/n1dm_readtmp && sha256sum n1dm.tar n1dh.tar > tars.sha256
  cd /root/strat1/n1dm/runs && sha256sum N1DM_s0/train/ckpt_last.pt N1DM_s1/train/ckpt_last.pt > /root/strat1/n1dm_readtmp/SHA256_PODS.txt
  ls -la /root/strat1/n1dm_readtmp' | tr -d '\r'
for f in n1dm.tar n1dh.tar tars.sha256 SHA256_PODS.txt; do pscp "$POD_TARGET:/root/strat1/n1dm_readtmp/$f" $H/tmp_pull/$f; done
pscp "$POD_TARGET:/root/strat1/n1dm/runs/N1DM_s0/train/ckpt_last.pt" $H/ckpts/N1DM_s0/ckpt_last.pt
pscp "$POD_TARGET:/root/strat1/n1dm/runs/N1DM_s1/train/ckpt_last.pt" $H/ckpts/N1DM_s1/ckpt_last.pt
(cd $H/tmp_pull && sha256sum -c tars.sha256)
(cd $H/ckpts && sed 's|N1DM_s\([01]\)/train/ckpt_last.pt|N1DM_s\1/ckpt_last.pt|' $H/tmp_pull/SHA256_PODS.txt > SHA256_PODS.txt && sha256sum -c SHA256_PODS.txt)
(cd $H/in/n1dm && tar --force-local -xf $H/tmp_pull/n1dm.tar)
(cd $H/in/n1dh && tar --force-local -xf $H/tmp_pull/n1dh.tar)
echo "n1dm files $(find $H/in/n1dm -type f | wc -l), n1dh files $(find $H/in/n1dh -type f | wc -l)"
s1=$(stat -c %s $H/tmp_pull/n1dm.tar); s2=$(stat -c %s $H/tmp_pull/n1dh.tar)
pssh 'rm -f /root/strat1/n1dm_readtmp/n1dm.tar /root/strat1/n1dm_readtmp/n1dh.tar /root/strat1/n1dm_readtmp/tars.sha256 /root/strat1/n1dm_readtmp/SHA256_PODS.txt && rmdir /root/strat1/n1dm_readtmp && echo pod-tmp-removed' | tr -d '\r'
rm -f "$H/tmp_pull/n1dm.tar" "$H/tmp_pull/n1dh.tar"
echo "$(date -u +%FT%TZ) MM pull_n1dm.sh: removed pod /root/strat1/n1dm_readtmp (n1dm.tar $s1 B, n1dh.tar $s2 B, two sha lists; sources stay in /root/strat1/n1dm/out, /root/strat1/n1dh/out) and local tmp_pull/n1dm.tar, n1dh.tar (extracted into read_n1dm/in/)" >> $LOGD
echo "ZZ PULL N1DM OK ZZ"
