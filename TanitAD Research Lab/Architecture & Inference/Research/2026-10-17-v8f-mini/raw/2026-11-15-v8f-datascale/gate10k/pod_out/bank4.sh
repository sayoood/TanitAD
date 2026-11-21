#!/bin/bash
# bank4.sh -- one DAgger bank, built AS THE RECIPE OF RECORD BUILDS BANKS: n_shards 4, the 4 shard processes AT THE SAME TIME on the one GPU (one chain step), then `merge`.
# (M2b's bank of record /root/strat1/inputs/dagger_bank/s0/bank.json has n_shards 4; MM ruling 2026-11-17: 4 parallel shards 111 s vs 358 s for 1 shard on 200 clips.)
# usage: bank4.sh CKPT M2B_ARGV_JSON CACHE_DIR TRACKS JOIN OUT_DIR EXPECT_CLIPS EXPECT_WINDOWS
# Exit 0 only when every shard AND the merge succeeded and OUT_DIR/bank.json exists. The shard count is fixed here (N_SHARDS=4) and recorded in bank.json (n_shards) for the checker (C-BANK).
D=/root/dscale10k; PY=/root/venv/bin/python; TREE=$D/tree
CKPT=$1; MARGV=$2; CACHE=$3; TRACKS=$4; JOIN=$5; OUT=$6; EC=$7; EW=$8
N_SHARDS=4
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT"
COMMON="--ckpt $CKPT --m2b-argv $MARGV --cache-dir $CACHE --tracks $TRACKS --join-train $JOIN --eval-grid /root/strat1/eval_grid_windows.json --out-dir $OUT --n-shards $N_SHARDS --expect-clips $EC --expect-windows $EW --device cuda"
t0=$(date +%s)
echo "$(date -u +%FT%TZ) bank4 start n_shards $N_SHARDS parallel out $OUT"
for k in 0 1 2 3; do
  $PY $TREE/stack/scripts/v8f_dagger_bank.py gen $COMMON --shard $k > "$OUT/gen_$k.log" 2>&1 &
done
fails=0
for job in $(jobs -p); do wait "$job" || fails=$((fails+1)); done
t1=$(date +%s)
echo "gen wall_s $((t1-t0)) failed_shards $fails"
[ "$fails" -eq 0 ] || { for k in 0 1 2 3; do tail -n 2 "$OUT/gen_$k.log"; done; echo "BANK4 FAIL $fails shard(s) failed"; exit 1; }
$PY $TREE/stack/scripts/v8f_dagger_bank.py merge --out-dir "$OUT" || { echo "BANK4 FAIL merge"; exit 1; }
[ -f "$OUT/bank.json" ] || { echo "BANK4 FAIL no bank.json (a control failed: see $OUT/PROVENANCE.json)"; exit 1; }
echo "$(date -u +%FT%TZ) bank4 done total_wall_s $(( $(date +%s)-t0 )) n_shards $N_SHARDS"
