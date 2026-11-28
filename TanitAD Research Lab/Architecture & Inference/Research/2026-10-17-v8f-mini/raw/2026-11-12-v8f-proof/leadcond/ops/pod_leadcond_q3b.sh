# pod_leadcond_q3b.sh : runs ON THE POD (nohup). Control q3b-ii of note PREREG_V8F_PROOF_ADD35_N1: the checkpoint LS_s0 with the layer of --lead-cond built and
# both of its tensors zero is rolled with the pinned T1-kin tool of tree_leadcond, and so is the unchanged checkpoint LS_s0, the same way; every array of the two
# dumps is compared. PASS (largest absolute difference 0.0) writes runs/leadcond_z/Q3B_DONE.txt, anything else Q3B_FAIL.txt. Trains nothing.
# It waits until fewer than three GPU jobs run, read twice 45 s apart, and no state extraction runs (never a fourth GPU job).
set -u
W=/workspace/v8proof; Z=$W/runs/leadcond_z; PY=/workspace/venv/bin/python; TREE=$W/tree_leadcond
INP=$W/data/inputs; PINS=$W/hierrep/pins
export PYTHONPATH=$TREE/stack:$TREE/taniteval TREE   # the environment the chain gives its T1-kin dumps (chain_arms_ft6.sh line 108)
mkdir -p $Z/t1 $Z/ck/LS0Z || exit 1
LOG=$Z/q3b.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -e $Z/Q3B_DONE.txt ] || [ -e $Z/Q3B_FAIL.txt ] && { echo "[$(ts)] a result exists, nothing done" >> $LOG; exit 0; }
echo "[$(ts)] q3b start" >> $LOG
k=0
while :; do
  k=$((k+1))
  if true; then   # the state extraction of the motion waiters is a CPU job (extract_states.py has no device code), so only the GPU job count is waited for
    n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45
    n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ "$(pgrep -f '[e]xtract_states.py' | wc -l)" = 0 ] && break
  fi
  [ $k -gt 480 ] && { echo "[$(ts)] no free slot in 8 h" >> $LOG; echo "no free slot in 8 h ($(ts))" > $Z/Q3B_FAIL.txt; exit 1; }
  sleep 60
done
echo "[$(ts)] slot free (base states present: $(ls $W/probe/states 2>/dev/null | grep -c 'G_s3_.*npz.json')); building the zero-layer checkpoint" >> $LOG
CUDA_VISIBLE_DEVICES=-1 $PY - >> $LOG 2>&1 <<'PYEOF'
import hashlib, torch
src = "/workspace/v8proof/runs/leadsup_a/LS_s0/ckpt_last.pt"
dst = "/workspace/v8proof/runs/leadcond_z/ck/LS0Z/ckpt_last.pt"
ck = torch.load(src, map_location="cpu", weights_only=False)
sd, cfg = ck["model"], ck["cfg"]
assert "lead_cond.weight" not in sd and cfg.get("lead_sup") == "on" and cfg.get("lead_cond", "off") == "off"
d = int(cfg["d"]); assert tuple(sd["e_ego.weight"].shape)[0] == d
sd["lead_cond.weight"] = torch.zeros(d, 2, dtype=sd["e_ego.weight"].dtype)
sd["lead_cond.bias"] = torch.zeros(d, dtype=sd["e_ego.bias"].dtype)
cfg["lead_cond"] = "on"
torch.save(ck, dst)
print("zero-layer checkpoint written; source sha256", hashlib.sha256(open(src, "rb").read()).hexdigest(), "tensors", len(sd))
PYEOF
[ -f $Z/ck/LS0Z/ckpt_last.pt ] || { echo "the zero-layer checkpoint was not written ($(ts))" > $Z/Q3B_FAIL.txt; exit 1; }
dump() { # dump <ckpt> <name>
  local out=$Z/t1/$2 tag=${2}_m10_causal_heads
  mkdir -p "$out/ticks"
  flock -w 5400 $W/chains/heavy.lock timeout 2700 $PY $TREE/stack/scripts/v8f_t1_eval.py run --ckpt "$1" --tag "$tag" --tracks $INP/native_tracks.pt --eval-cache $W/data/cache_pod --eval-grid $PINS/eval_grid_windows.json --eval-join $PINS/join_eval141.npz --out-dir "$out" \
    --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
  local rc=$?
  echo "[$(ts)] dump $2 rc $rc, ep files $(ls $out/dump_$tag 2>/dev/null | grep -c '^ep.*npz$')" >> $LOG
  return $rc
}
dump $W/runs/leadsup_a/LS_s0/ckpt_last.pt LS0REF || { echo "dump LS0REF failed ($(ts))" > $Z/Q3B_FAIL.txt; exit 1; }
dump $Z/ck/LS0Z/ckpt_last.pt LS0Z || { echo "dump LS0Z failed ($(ts))" > $Z/Q3B_FAIL.txt; exit 1; }
CUDA_VISIBLE_DEVICES=-1 $PY - > $Z/q3b_compare.txt 2>&1 <<'PYEOF'
import glob, os, sys
import numpy as np
Z = "/workspace/v8proof/runs/leadcond_z/t1"
def files(name, sub):
    d = os.path.join(Z, name, sub)
    return {os.path.basename(p).replace(name, "X"): p for p in sorted(glob.glob(os.path.join(d, "*.npz")))}
worst, n_arr, n_files, bad = 0.0, 0, 0, []
for sub_a, sub_b in (("dump_LS0REF_m10_causal_heads", "dump_LS0Z_m10_causal_heads"), ("ticks", "ticks")):
    A, B = files("LS0REF", sub_a), files("LS0Z", sub_b)
    if sorted(A) != sorted(B) or not A:
        bad.append("file lists differ or empty in %s: %d vs %d" % (sub_a, len(A), len(B))); continue
    for k in sorted(A):
        a, b = np.load(A[k], allow_pickle=True), np.load(B[k], allow_pickle=True)
        if sorted(a.files) != sorted(b.files):
            bad.append("keys differ in " + k); continue
        n_files += 1
        for f in a.files:
            x, y = a[f], b[f]
            if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
                if x.shape != y.shape:
                    bad.append("shape %s %s" % (k, f)); continue
                n_arr += 1
                if x.size:
                    dd = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if not (np.isnan(x.astype(np.float64)) != np.isnan(y.astype(np.float64))).any() else float("inf")
                    if dd > worst:
                        worst = dd
                    if dd != 0.0 and len(bad) < 8:
                        bad.append("diff %.3e in %s %s" % (dd, k, f))
print("files compared %d, numeric arrays %d, largest absolute difference %r" % (n_files, n_arr, worst))
for b_ in bad[:10]:
    print("  ", b_)
print("Q3B", "PASS" if (worst == 0.0 and not bad and n_files >= 141) else "FAIL")
PYEOF
cat $Z/q3b_compare.txt >> $LOG
if grep -q "^Q3B PASS" $Z/q3b_compare.txt; then { echo "PASS $(ts)"; cat $Z/q3b_compare.txt; } > $Z/Q3B_DONE.txt; else { echo "FAIL $(ts)"; cat $Z/q3b_compare.txt; } > $Z/Q3B_FAIL.txt; fi
echo "[$(ts)] q3b end" >> $LOG
