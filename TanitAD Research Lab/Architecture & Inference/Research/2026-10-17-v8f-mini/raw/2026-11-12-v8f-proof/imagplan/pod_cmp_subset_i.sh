# pod_cmp_subset_i.sh <ARM_sK>... : runs ON THE POD (CPU, read-only). Card PREREG_V8F_PROOF_ADD40, note N5 section 2: the subset form of control p3, here for
# source cand-imag (runs IPim and IPnm on the first 600 windows; a run with IP_MAX_WINDOWS writes no T1_DONE, so the sequence does not compare them).
# The compare uses the tool the sequence uses (distrule/cmp_dumps.py). It changes nothing.
W=/workspace/v8proof; R=$W/runs/imagplan
for B in "$@"; do
  a=$R/t1/IPnm_$B/dump_IPnm_${B}_m10_causal_heads; b=$R/t1/IPim_$B/dump_IPim_${B}_m10_causal_heads
  if [ -d "$a" ] && [ -d "$b" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $B cmp IPnm against IPim (p3 of cand-imag, subset form of note N5 on the first 600 windows, must be equal): $(/workspace/venv/bin/python $W/distrule/cmp_dumps.py "$a" "$b" 2>&1 | tail -n 1 | cut -c1-200) ; episode files noise run $(ls "$a" | wc -l), reference run $(ls "$b" | wc -l)"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $B ZZ CMP-SUBSET MISSING: a dump folder of the subset runs is not on the pod"
  fi
done
