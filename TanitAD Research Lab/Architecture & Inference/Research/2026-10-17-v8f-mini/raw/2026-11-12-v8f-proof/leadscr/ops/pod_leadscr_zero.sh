# pod_leadscr_zero.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD39, sections 2 and 5: the checkpoints of LCS_s0 and LCS_s1 with the two tensors of the
# lead-input layer set to zero (LCSZ), rolled with the pinned T1-kin tool of tree_leadcond the way the chain rolls its arms; control s4: LCSZ_s0 rolled a second
# time (LCSZ_s0r) and every array of the two dumps compared. Trains nothing. Waits for the end of chain leadscr_a, then for a free GPU slot before every dump
# (never a fourth GPU job). Writes runs/leadscr_z/LCSZ_DONE.txt.
set -u
W=/workspace/v8proof; Z=$W/runs/leadscr_z; PY=/workspace/venv/bin/python; TREE=$W/tree_leadcond
INP=$W/data/inputs; PINS=$W/hierrep/pins
export PYTHONPATH=$TREE/stack:$TREE/taniteval TREE
mkdir -p $Z/t1 $Z/ck || exit 1
LOG=$Z/lcsz.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
[ -e $Z/LCSZ_DONE.txt ] && { say "a result exists, nothing done"; exit 0; }
say "lcsz start: waits for chain leadscr_a"
k=0
while :; do
  a=0
  [ -f $W/chains/leadscr_a/leadscr_a_DONE.txt ] && a=1; [ -f $W/chains/leadscr_a/leadscr_a_STOP.txt ] && a=2
  [ $a != 0 ] && break
  k=$((k+1)); [ $k -gt 1440 ] && { say "chain not ended in 24 h"; break; }
  sleep 60
done
say "chain ended: a=$a (1 done, 2 stopped)"
slot() { local j=0 n1 n2; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && return 0; j=$((j+1)); [ $j -gt 480 ] && return 1; sleep 45; done; }
one() { # one <name> <source arm>
  local name=$1 src=$W/runs/leadscr_a/$2/ckpt_last.pt out=$Z/t1/$1 tag=${1}_m10_causal_heads
  [ -f "$src" ] || { say "$name: source checkpoint absent ($2), skipped"; return 1; }
  [ -f "$out/T1_DONE" ] && { say "$name: dump exists"; return 0; }
  mkdir -p $Z/ck/$name "$out/ticks"
  CUDA_VISIBLE_DEVICES=-1 $PY - "$src" "$Z/ck/$name/ckpt_last.pt" >> $LOG 2>&1 <<'PYEOF'
import sys, hashlib, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd, cfg = ck["model"], ck["cfg"]
assert cfg.get("lead_cond") == "on" and "lead_cond.weight" in sd and "lead_cond.bias" in sd, "the source has no lead_cond layer"
w, b = sd["lead_cond.weight"], sd["lead_cond.bias"]
print("   source", src, "sha256", hashlib.sha256(open(src, "rb").read()).hexdigest()[:16], "step", ck.get("step"), "| layer before zeroing: weight abs max %.6g, abs mean %.6g, column norms %s, bias abs max %.6g" % (
    float(w.abs().max()), float(w.abs().mean()), [round(float(x), 5) for x in w.norm(dim=0)], float(b.abs().max())))
sd["lead_cond.weight"] = torch.zeros_like(w); sd["lead_cond.bias"] = torch.zeros_like(b)
torch.save(ck, dst)
PYEOF
  [ -f $Z/ck/$name/ckpt_last.pt ] || { say "$name: zeroed checkpoint not written"; return 1; }
  slot || { say "$name: no free GPU slot in 12 h"; return 1; }
  flock -w 5400 $W/chains/heavy.lock timeout 2700 $PY $TREE/stack/scripts/v8f_t1_eval.py run --ckpt $Z/ck/$name/ckpt_last.pt --tag "$tag" --tracks $INP/native_tracks.pt --eval-cache $W/data/cache_pod --eval-grid $PINS/eval_grid_windows.json --eval-join $PINS/join_eval141.npz --out-dir "$out" \
    --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
  local rc=$? n; n=$(ls $out/dump_$tag 2>/dev/null | grep -c '^ep.*npz$')
  say "$name: dump rc $rc, ep files $n"
  [ $rc = 0 ] && [ "$n" = 141 ] && touch "$out/T1_DONE"
}
ok=0
one LCSZ_s0 LCS_s0 && ok=$((ok+1))
one LCSZ_s1 LCS_s1 && ok=$((ok+1))
# control s4: the same zeroed checkpoint rolled a second time
if [ -f $Z/t1/LCSZ_s0/T1_DONE ]; then
  mkdir -p $Z/ck/LCSZ_s0r; [ -f $Z/ck/LCSZ_s0r/ckpt_last.pt ] || cp $Z/ck/LCSZ_s0/ckpt_last.pt $Z/ck/LCSZ_s0r/ckpt_last.pt
  out=$Z/t1/LCSZ_s0r; tag=LCSZ_s0r_m10_causal_heads; mkdir -p "$out/ticks"
  if [ ! -f "$out/T1_DONE" ] && slot; then
    flock -w 5400 $W/chains/heavy.lock timeout 2700 $PY $TREE/stack/scripts/v8f_t1_eval.py run --ckpt $Z/ck/LCSZ_s0r/ckpt_last.pt --tag "$tag" --tracks $INP/native_tracks.pt --eval-cache $W/data/cache_pod --eval-grid $PINS/eval_grid_windows.json --eval-join $PINS/join_eval141.npz --out-dir "$out" \
      --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$out/ticks" --dump-heads --device cuda >> "$out/t1.log" 2>&1
    rc=$?; n=$(ls $out/dump_$tag 2>/dev/null | grep -c '^ep.*npz$'); say "LCSZ_s0r: dump rc $rc, ep files $n"; [ $rc = 0 ] && [ "$n" = 141 ] && touch "$out/T1_DONE"
  fi
  if [ -f "$out/T1_DONE" ]; then
    CUDA_VISIBLE_DEVICES=-1 $PY - $Z/t1/LCSZ_s0/dump_LCSZ_s0_m10_causal_heads $out/dump_$tag > $Z/S4.txt 2>&1 <<'PYEOF'
import sys, os, numpy as np
a, b = sys.argv[1], sys.argv[2]
fa = sorted(f for f in os.listdir(a) if f.endswith(".npz")); fb = sorted(f for f in os.listdir(b) if f.endswith(".npz"))
worst = 0.0; where = ""; n_arr = 0; skipped = 0
for f in fa:
    if f not in fb: continue
    A, B = np.load(os.path.join(a, f), allow_pickle=True), np.load(os.path.join(b, f), allow_pickle=True)
    for k in A.files:
        if k not in B.files: continue
        x, y = A[k], B[k]
        if x.dtype.kind not in "fiub" or x.shape != y.shape: skipped += 1; continue
        n_arr += 1
        d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        if d > worst: worst, where = d, f + ":" + k
print("S4 files %d against %d, arrays compared %d, skipped %d, largest absolute difference %.6g %s" % (len(fa), len(fb), n_arr, skipped, worst, where))
PYEOF
    say "s4: $(tail -n 1 $Z/S4.txt | cut -c1-240)"
  else
    say "s4: second dump not made"
  fi
fi
say "lcsz end: $ok of 2 dumps"
echo "$ok of 2 dumps ($(ts))" > $Z/LCSZ_DONE.txt
