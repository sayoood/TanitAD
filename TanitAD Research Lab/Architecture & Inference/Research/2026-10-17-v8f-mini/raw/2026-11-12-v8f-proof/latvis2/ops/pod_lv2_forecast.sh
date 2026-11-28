#!/bin/bash
# pod_lv2_forecast.sh [ARM_sK ...] : card PREREG_V8F_PROOF_ADD38, note N3 (the token-level read at equal calibration). Pod CPU, one arm after the other.
# The same run as pod_lv2_tokens.sh plus --forecast; writes DUMP_LV2F_<ARM>.npz. Starts no GPU job and no training. Refuses below 24 GB of free container memory.
set -u
W=/workspace/v8proof; L=$W/probe/lv2; Z=$W/runs/lv2_z; PY=/workspace/venv/bin/python
mkdir -p $Z/work
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ck() { case $1 in G_s0) echo $W/runs/spatial/G_s0/ckpt_last.pt;; G_s1) echo $W/runs/spatial/G_s1/ckpt_last.pt;; RR_s0) echo $W/runs/motion_a/RR_s0/ckpt_last.pt;; RR_s1) echo $W/runs/motion_d/RR_s1/ckpt_last.pt;; *) echo NONE;; esac; }
free_gb() { local mx an; mx=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); an=$(grep "^anon " /sys/fs/cgroup/memory.stat 2>/dev/null | awk '{print $2}'); [ -n "$mx" ] && [ "$mx" != max ] && [ -n "$an" ] && echo $(( (mx - an) / 1000000000 )) || echo 0; }
[ $# -gt 0 ] || set -- G_s0 G_s1 RR_s0 RR_s1
echo "[$(ts)] tool sha256 $(sha256sum $L/lv2_pod_tokens.py | cut -c1-64)"
for A in "$@"; do
  C=$(ck $A); [ -f "$C" ] || { echo "[$(ts)] ZZ LV2F NO-CKPT $A"; continue; }
  O=$Z/DUMP_LV2F_$A.npz
  if [ -f "$O" ] && [ -f "$O.json" ]; then echo "[$(ts)] $A: dump exists, kept"; else
    F=$(free_gb); [ "$F" -ge 24 ] || { echo "[$(ts)] ZZ LV2F DUMP-NOT-RUN $A: free memory $F GB"; continue; }
    echo "[$(ts)] $A: start (free memory $F GB)"
    ( cd $L && CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/tree_leadsup/stack OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY lv2_pod_tokens.py \
        --ckpt "$C" --tree $W/tree_leadsup --windows $W/probe/imag_in/LATENTS_CTX_G_s0_R1.npz --pack $W/probe/imag_in/eval_pack.npz \
        --states $W/probe/states/${A}_eval.npz --cache $W/data/cache_pod --banked-roll $L/banked/LATENTS_CTX_${A}_R3g.npz \
        --panel-windows $L/WINDOWS_LV2.json --out "$O" --workdir $Z/work --device cpu --threads 4 --imag-dir $W/probe --forecast ) > $Z/lv2f_$A.log 2>&1
    rc=$?; echo "[$(ts)] $A: rc $rc : $(tail -n 1 $Z/lv2f_$A.log | cut -c1-240)"
    [ $rc = 0 ] || continue
  fi
  echo "   SHA $A $(sha256sum "$O" | cut -c1-64) $(sha256sum "$O.json" | cut -c1-64) $(stat -c %s "$O")"
done
echo "[$(ts)] ZZ LV2F POD DONE"
