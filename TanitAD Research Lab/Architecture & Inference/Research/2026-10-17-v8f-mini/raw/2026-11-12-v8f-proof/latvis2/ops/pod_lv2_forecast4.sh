#!/bin/bash
# pod_lv2_forecast4.sh [ARM_sK ...] : card PREREG_V8F_PROOF_ADD38, note N4 (the token-level read at equal calibration on the new seeds 2 and 3 of G and RR).
# The run of pod_lv2_forecast.sh for the new arms, the tool unchanged, plus the k5 probe of note N2 for each of them. Pod CPU, one arm after the other.
# Starts no GPU job and no training. Refuses below 24 GB of free container memory.
set -u
W=/workspace/v8proof; L=$W/probe/lv2; Z=$W/runs/lv2_z; PY=/workspace/venv/bin/python
mkdir -p $Z/work
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ck() { case $1 in G_s2) echo $W/runs/spatialrep_a/G_s2/ckpt_last.pt;; G_s3) echo $W/runs/spatialrep_a/G_s3/ckpt_last.pt;; RR_s2) echo $W/runs/motion_e/RR_s2/ckpt_last.pt;; RR_s3) echo $W/runs/motion_e/RR_s3/ckpt_last.pt;; *) echo NONE;; esac; }
free_gb() { local mx an; mx=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); an=$(grep "^anon " /sys/fs/cgroup/memory.stat 2>/dev/null | awk '{print $2}'); [ -n "$mx" ] && [ "$mx" != max ] && [ -n "$an" ] && echo $(( (mx - an) / 1000000000 )) || echo 0; }
[ $# -gt 0 ] || set -- G_s2 G_s3 RR_s2 RR_s3
echo "[$(ts)] tool sha256 $(sha256sum $L/lv2_pod_tokens.py | cut -c1-64) probe sha256 $(sha256sum $L/lv2_k5_probe.py | cut -c1-64)"
for A in "$@"; do
  C=$(ck $A); [ -f "$C" ] || { echo "[$(ts)] ZZ LV2F4 NO-CKPT $A"; continue; }
  [ -f $L/banked/LATENTS_CTX_${A}_R3g.npz ] || { echo "[$(ts)] ZZ LV2F4 NO-BANKED-ROLL $A"; continue; }
  O=$Z/DUMP_LV2F_$A.npz
  if [ -f "$O" ] && [ -f "$O.json" ]; then echo "[$(ts)] $A: dump exists, kept"; else
    F=$(free_gb); [ "$F" -ge 24 ] || { echo "[$(ts)] ZZ LV2F4 DUMP-NOT-RUN $A: free memory $F GB"; continue; }
    echo "[$(ts)] $A: start (free memory $F GB)"
    ( cd $L && CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/tree_leadsup/stack OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY lv2_pod_tokens.py \
        --ckpt "$C" --tree $W/tree_leadsup --windows $W/probe/imag_in/LATENTS_CTX_G_s0_R1.npz --pack $W/probe/imag_in/eval_pack.npz \
        --states $W/probe/states/${A}_eval.npz --cache $W/data/cache_pod --banked-roll $L/banked/LATENTS_CTX_${A}_R3g.npz \
        --panel-windows $L/WINDOWS_LV2.json --out "$O" --workdir $Z/work --device cpu --threads 4 --imag-dir $W/probe --forecast ) > $Z/lv2f_$A.log 2>&1
    rc=$?; echo "[$(ts)] $A: rc $rc : $(tail -n 1 $Z/lv2f_$A.log | cut -c1-240)"
    [ $rc = 0 ] || continue
  fi
  if [ ! -s $Z/k5_probe_$A.out ]; then
    ( cd $L && CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/tree_leadsup/stack OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY lv2_k5_probe.py $A "$C" ) > $Z/k5_probe_$A.out 2> $Z/k5_probe_$A.err
    echo "[$(ts)] $A: k5 probe rc $? : $(tail -n 1 $Z/k5_probe_$A.out | cut -c1-200)"
  fi
  echo "   SHA $A $(sha256sum "$O" | cut -c1-64) $(sha256sum "$O.json" | cut -c1-64) $(sha256sum $Z/k5_probe_$A.out | cut -c1-64) $(stat -c %s "$O")"
done
echo "[$(ts)] ZZ LV2F4 POD DONE"
