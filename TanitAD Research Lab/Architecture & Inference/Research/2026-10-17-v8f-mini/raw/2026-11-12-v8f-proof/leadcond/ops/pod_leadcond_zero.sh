# pod_leadcond_zero.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD35, section 3: the checkpoints of LC and LCO with the two tensors of the new layer
# set to zero (LCZ, LCOZ), rolled with the pinned T1-kin tool of tree_leadcond the way the chain rolls its arms. Trains nothing.
# Waits for the end of chains leadcond_a and leadcond_b, then for a free GPU slot before every dump (never a fourth GPU job). Writes runs/leadcond_z/LCZ_DONE.txt.
set -u
W=/workspace/v8proof; Z=$W/runs/leadcond_z; PY=/workspace/venv/bin/python; TREE=$W/tree_leadcond
INP=$W/data/inputs; PINS=$W/hierrep/pins
export PYTHONPATH=$TREE/stack:$TREE/taniteval TREE   # the environment the chain gives its T1-kin dumps
mkdir -p $Z/t1 $Z/ck || exit 1
LOG=$Z/lcz.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
[ -e $Z/LCZ_DONE.txt ] && { say "a result exists, nothing done"; exit 0; }
say "lcz start: waits for chains leadcond_a and leadcond_b"
k=0
while :; do
  a=0; b=0
  [ -f $W/chains/leadcond_a/leadcond_a_DONE.txt ] && a=1; [ -f $W/chains/leadcond_a/leadcond_a_STOP.txt ] && a=2
  [ -f $W/chains/leadcond_b/leadcond_b_DONE.txt ] && b=1; [ -f $W/chains/leadcond_b/leadcond_b_STOP.txt ] && b=2; [ -f $Z/LCB_NOT_STARTED.txt ] && b=2
  [ $a != 0 ] && [ $b != 0 ] && break
  k=$((k+1)); [ $k -gt 1440 ] && { say "chains not ended in 24 h (a=$a b=$b)"; break; }
  sleep 60
done
say "chains ended: a=$a b=$b (1 done, 2 stopped or not started)"
slot() { local j=0 n1 n2; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && return 0; j=$((j+1)); [ $j -gt 480 ] && return 1; sleep 45; done; }
one() { # one <name> <chain> <source arm>
  local name=$1 src=$W/runs/$2/$3/ckpt_last.pt out=$Z/t1/$1 tag=${1}_m10_causal_heads
  [ -f "$src" ] || { say "$name: source checkpoint absent ($2/$3), skipped"; return 1; }
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
one LCZ_s0 leadcond_a LC_s0 && ok=$((ok+1))
one LCZ_s1 leadcond_a LC_s1 && ok=$((ok+1))
one LCOZ_s0 leadcond_b LCO_s0 && ok=$((ok+1))
one LCOZ_s1 leadcond_b LCO_s1 && ok=$((ok+1))
say "lcz end: $ok of 4 dumps"
echo "$ok of 4 dumps ($(ts))" > $Z/LCZ_DONE.txt
