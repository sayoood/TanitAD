#!/bin/bash
# rh_bank.sh -- one DAgger bank as bank4.sh builds it (4 shards AT THE SAME TIME on the one GPU, then merge), with the tree and the generator flags as parameters.
# usage: rh_bank.sh TREE CKPT M2B_ARGV CACHE TRACKS JOIN OUT_DIR EXPECT_CLIPS EXPECT_WINDOWS LIMIT_CLIPS "EXTRA GENERATOR FLAGS"
#   LIMIT_CLIPS > 0 = a PILOT (--limit-clips N, no --expect-*, the bank is marked partial); EXTRA = "" is today's generator (the identity pilot, or the unpatched tree).
# Exit 0 only when every shard AND the merge succeeded and OUT_DIR/bank.json exists.
T=$1; CKPT=$2; MARGV=$3; CACHE=$4; TRACKS=$5; JOIN=$6; OUT=$7; EC=$8; EW=$9; LIM=${10}; EXTRA=${11}
PY=${RH_PY:-/root/venv/bin/python}; N_SHARDS=4
export PYTHONPATH="$T/stack:$T/stack/scripts:$T" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT"
COMMON="--ckpt $CKPT --m2b-argv $MARGV --cache-dir $CACHE --tracks $TRACKS --join-train $JOIN --eval-grid /root/strat1/eval_grid_windows.json --out-dir $OUT --n-shards $N_SHARDS --device cuda"
if [ "${LIM:-0}" -gt 0 ]; then COMMON="$COMMON --limit-clips $LIM"; else COMMON="$COMMON --expect-clips $EC --expect-windows $EW"; fi
t0=$(date +%s)
echo "$(date -u +%FT%TZ) rh_bank start tree $T n_shards $N_SHARDS parallel limit ${LIM:-0} extra [$EXTRA] out $OUT"
for k in 0 1 2 3; do
  $PY $T/stack/scripts/v8f_dagger_bank.py gen $COMMON $EXTRA --shard $k > "$OUT/gen_$k.log" 2>&1 &
done
fails=0
for job in $(jobs -p); do wait "$job" || fails=$((fails+1)); done
echo "gen wall_s $(( $(date +%s)-t0 )) failed_shards $fails"
[ "$fails" -eq 0 ] || { for k in 0 1 2 3; do tail -n 2 "$OUT/gen_$k.log"; done; echo "RH_BANK FAIL $fails shard(s) failed"; exit 1; }
$PY $T/stack/scripts/v8f_dagger_bank.py merge --out-dir "$OUT" || { echo "RH_BANK FAIL merge"; exit 1; }
[ -f "$OUT/bank.json" ] || { echo "RH_BANK FAIL no bank.json (a control failed: see $OUT/PROVENANCE.json)"; exit 1; }
echo "$(date -u +%FT%TZ) rh_bank done total_wall_s $(( $(date +%s)-t0 ))"
