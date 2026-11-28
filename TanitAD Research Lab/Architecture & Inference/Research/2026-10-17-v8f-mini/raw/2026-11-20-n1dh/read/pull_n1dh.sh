#!/bin/bash
# MM 2026-11-21: pull the N1DH + N1DC D1-HO dumps and the two N1DH checkpoints (HANDOFF_READ_N1DH "Commands"). Pod: tar + pscp only; the pod tars are removed after the
# local copy is verified by sha256 (the pod sources stay). Local tars removed after extraction. Every removal logged in F:/TanitAD-scratch/n1dc/DELETIONS.log.
set -euo pipefail
source C:/Users/Admin/lorapod/podenv.sh
H=F:/TanitAD-scratch/n1dc/read_n1dh
LOGD=F:/TanitAD-scratch/n1dc/DELETIONS.log
mkdir -p $H/in/n1dh $H/in/n1dc $H/ckpts/N1DH_s0 $H/ckpts/N1DH_s1 $H/tmp_pull
pssh 'set -e; mkdir -p /root/strat1/n1dh_readtmp; X="--exclude=ticks_* --exclude=log_* --exclude=DONE_* --exclude=hb_* --exclude=t1v8f_*"
  cd /root/strat1/n1dh/out && tar cf /root/strat1/n1dh_readtmp/n1dh.tar $X hb0.2 hbnone
  cd /root/strat1/n1dc/out && tar cf /root/strat1/n1dh_readtmp/n1dc.tar $X hb0.2 hbnone
  cd /root/strat1/n1dh_readtmp && sha256sum n1dh.tar n1dc.tar > tars.sha256
  cd /root/strat1/n1dh/runs && sha256sum N1DH_s0/train/ckpt_last.pt N1DH_s1/train/ckpt_last.pt > /root/strat1/n1dh_readtmp/SHA256_PODS.txt
  ls -la /root/strat1/n1dh_readtmp' | tr -d '\r'
for f in n1dh.tar n1dc.tar tars.sha256 SHA256_PODS.txt; do pscp "$POD_TARGET:/root/strat1/n1dh_readtmp/$f" $H/tmp_pull/$f; done
pscp "$POD_TARGET:/root/strat1/n1dh/runs/N1DH_s0/train/ckpt_last.pt" $H/ckpts/N1DH_s0/ckpt_last.pt
pscp "$POD_TARGET:/root/strat1/n1dh/runs/N1DH_s1/train/ckpt_last.pt" $H/ckpts/N1DH_s1/ckpt_last.pt
(cd $H/tmp_pull && sha256sum -c tars.sha256)
(cd $H/ckpts && sed 's|N1DH_s\([01]\)/train/ckpt_last.pt|N1DH_s\1/ckpt_last.pt|' $H/tmp_pull/SHA256_PODS.txt > SHA256_PODS.txt && sha256sum -c SHA256_PODS.txt)
(cd $H/in/n1dh && tar --force-local -xf $H/tmp_pull/n1dh.tar)
(cd $H/in/n1dc && tar --force-local -xf $H/tmp_pull/n1dc.tar)
echo "n1dh files $(find $H/in/n1dh -type f | wc -l), n1dc files $(find $H/in/n1dc -type f | wc -l)"
s1=$(stat -c %s $H/tmp_pull/n1dh.tar); s2=$(stat -c %s $H/tmp_pull/n1dc.tar)
pssh 'rm -f /root/strat1/n1dh_readtmp/n1dh.tar /root/strat1/n1dh_readtmp/n1dc.tar /root/strat1/n1dh_readtmp/tars.sha256 /root/strat1/n1dh_readtmp/SHA256_PODS.txt && rmdir /root/strat1/n1dh_readtmp && echo pod-tmp-removed' | tr -d '\r'
rm -f "$H/tmp_pull/n1dh.tar" "$H/tmp_pull/n1dc.tar"
echo "$(date -u +%FT%TZ) MM pull_n1dh.sh: removed pod /root/strat1/n1dh_readtmp (n1dh.tar $s1 B, n1dc.tar $s2 B, two sha lists; sources stay in /root/strat1/n1dh/out, /root/strat1/n1dc/out) and local tmp_pull/n1dh.tar, n1dc.tar (extracted into read_n1dh/in/)" >> $LOGD
echo "ZZ PULL N1DH OK ZZ"
