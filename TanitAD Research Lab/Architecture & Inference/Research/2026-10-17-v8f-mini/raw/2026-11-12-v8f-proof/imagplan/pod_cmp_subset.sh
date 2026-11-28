# pod_cmp_subset.sh <ARM_sK>... : runs ON THE POD (CPU, read-only). Card PREREG_V8F_PROOF_ADD40, note N5 section 2: the subset form of control p3 of cand-imagkin.
# The sequences rlft0m1, rlam0m1, rlft1m1, rlam1m1 made the two runs on the first 600 windows (IPkm, IPknm) but did not compare them: the compare of the sequence asks for
# T1_DONE, and a run with IP_MAX_WINDOWS writes none. This script makes that compare with the same tool the sequence uses (distrule/cmp_dumps.py). It changes nothing.
W=/workspace/v8proof; R=$W/runs/imagplan
for B in "$@"; do
  a=$R/t1/IPknm_$B/dump_IPknm_${B}_m10_causal_heads; b=$R/t1/IPkm_$B/dump_IPkm_${B}_m10_causal_heads
  if [ -d "$a" ] && [ -d "$b" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $B cmp IPknm against IPkm (p3 of cand-imagkin, subset form of note N5 on the first 600 windows, must be equal): $(/workspace/venv/bin/python $W/distrule/cmp_dumps.py "$a" "$b" 2>&1 | tail -n 1 | cut -c1-200) ; episode files noise run $(ls "$a" | wc -l), reference run $(ls "$b" | wc -l)"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $B ZZ CMP-SUBSET MISSING: a dump folder of the subset runs is not on the pod"
  fi
done
