#!/bin/bash
# parallel pixel shards (same per-shard commands and .done_N markers as pod_dataprep.sh STAGES=pix, 4 at a time)
set -u
DATA=/workspace/data; S=/workspace/ops/stage_data; mkdir -p $DATA/navtrain_pixels $S
one() {
  n=$1; PIX=/workspace/data/navtrain_pixels; S=/workspace/ops/stage_data
  HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim
  [ -f $PIX/.done_$n ] && exit 0
  tgz=$PIX/navtrain_current_$n.tgz
  curl -sSL -C - --retry 10 --retry-all-errors -o "$tgz" "$HF/navtrain_current_$n.tgz" || { echo "ZZPIX_FAIL dl $n" >> $S/pix_par.log; exit 1; }
  if tar -xzf "$tgz" --no-same-owner -C $PIX --wildcards '*/CAM_F0/*' '*/CAM_B0/*' '*/CAM_L0/*' '*/CAM_R0/*'; then
    touch $PIX/.done_$n; rm -f "$tgz"; echo "ZZPIX_OK $n $(date -u +%T)" >> $S/pix_par.log
  else echo "ZZPIX_FAIL extract $n" >> $S/pix_par.log; rm -f "$tgz"; fi
}
export -f one
seq 1 32 | xargs -P 4 -I{} bash -c 'one {}'
echo "ZZPIX_PAR_DONE $(ls $DATA/navtrain_pixels/.done_* 2>/dev/null | wc -l) of 32" >> $S/pix_par.log
