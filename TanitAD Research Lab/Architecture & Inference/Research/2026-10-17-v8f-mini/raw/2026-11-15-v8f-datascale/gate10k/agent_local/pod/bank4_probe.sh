#!/bin/bash
# bank4_probe.sh -- MM timing probe (2026-11-16): the pilot's 200-clip DAgger bank rebuilt as the recipe of record builds banks, 4 clip shards (M2b's bank of record
# /root/strat1/inputs/dagger_bank/s0 has n_shards 4), the 4 shard processes run AT THE SAME TIME on the one GPU, then the generator's own merge.  Same checkpoint, argv and
# clips as pilot/bank200 (n_shards 1, 358 s).  Measures wall time only; the bank is a timing artifact, never used for training.  Sentinel: ZZ BANK4 PROBE DONE|FAIL ZZ.
D=/root/dscale10k
PY=/root/venv/bin/python
TREE=$D/tree
OUT=$D/pilot/bank200_4sh
LOG=$D/pilot/bank200_4sh.log
mkdir -p "$OUT"
COMMON="--ckpt $D/pilot/pB/ckpt_last.pt --m2b-argv $D/argv/B_s0.A.json --cache-dir /root/data/cache_pod --tracks /root/data/cache_pod/native_tracks.pt --join-train /root/strat1/inputs/join_train.npz --eval-grid /root/strat1/eval_grid_windows.json --out-dir $OUT --limit-clips 200 --n-shards 4 --device cuda"
t0=$(date +%s)
echo "$(date -u +%FT%TZ) start 4 shards in parallel" > "$LOG"
for k in 0 1 2 3; do
  $PY $TREE/stack/scripts/v8f_dagger_bank.py gen $COMMON --shard $k > "$OUT/gen_$k.log" 2>&1 &
done
fails=0
for job in $(jobs -p); do wait "$job" || fails=$((fails+1)); done
t1=$(date +%s)
echo "gen wall_s $((t1-t0)) failed_shards $fails" >> "$LOG"
for k in 0 1 2 3; do tail -n 2 "$OUT/gen_$k.log" >> "$LOG"; done
[ "$fails" -eq 0 ] || { echo "ZZ BANK4 PROBE FAIL ZZ $fails shard(s) failed" >> "$LOG"; exit 1; }
$PY $TREE/stack/scripts/v8f_dagger_bank.py merge --out-dir "$OUT" >> "$LOG" 2>&1 || { echo "ZZ BANK4 PROBE FAIL ZZ merge (see log)" >> "$LOG"; exit 1; }
t2=$(date +%s)
echo "total wall_s $((t2-t0))" >> "$LOG"
echo "ZZ BANK4 PROBE DONE ZZ" >> "$LOG"
