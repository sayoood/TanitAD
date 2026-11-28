#!/usr/bin/env bash
# H-NT-DECIDE-1 E2/E3 over the FULL A1 token universe on Thor's CPU (no GPU; nice 19; <= 4 cores, 1 thread each).
# Independent of E1's progress: labels + joins are built from the OpenScene log pickles; clip ids are the declared
# "navtrain-<token>" (INTERFACE.md sec. 4). thor_finalize_a1_v2.sh re-runs the (cheap) labels with E1's manifest once
# DONE_A1.json exists. Every step writes an EXIT_<step> file; the admissible evidence is the artifact, not the code.
#
# v2 (2026-10-21 ~23:10 Berlin), after run 2's gather took Thor into a GLOBAL OOM (22:50:38, 122.9 GB anon RSS; the
# kernel killed only that process; nt_joins.py's per-clip NpzFile reads, fixed + pinned by tests/test_nt_code.py):
#   * EVERY python step runs inside a KERNEL memory cap (systemd-run --user --scope MemoryMax, swap 0): a runaway is
#     killed in its own cgroup and can never again reach the shared box's global OOM; peak.py records each step's peak;
#   * every step is GATED on the previous step's EXIT (run 2 ran the merge after a SIGKILLed gather);
#   * RESUMABLE: a step whose EXIT file reads 0 and whose artifact exists is not re-run.
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts"
# the builders' own tree guard (nt_joins/_stack, nt_controls, nt_heldout) resolves the tree from NT_TREE; unset, it
# defaulted to <code>/../nt_tree, which does not exist on Thor -> the guard REFUSED all 4 join shards (run 1)
export NT_TREE="$W/tree"
cd "$W" || exit 2
mkdir -p "$L/a1_prov" "$L/a1_joins" "$L/a1" logs
cap() {  # $1 = MemoryMax, rest = command: kernel-capped (own cgroup, no swap), niced; rc preserved
  local M=$1; shift
  systemd-run --user --scope -q -p MemoryMax="$M" -p MemorySwapMax=0 nice -n 19 "$@" 200>&-
}
done_ok() { [ "$(cat "logs/EXIT_$1" 2>/dev/null)" = 0 ]; }
shards_ok() { local k; for k in 0 1 2 3; do [ -s "$L/a1_joins/shard_$k/JOINS_REPORT.json" ] || return 1; done; }
# 1. provisional labels (the token table the joins read)
if done_ok labels_prov && [ -s "$L/a1_prov/token_table_nt.json" ]; then
  echo "$(date +%T) step 1 complete ($(md5sum "$L/a1_prov/token_table_nt.json" | cut -c1-12)) -- skipped" >> logs/build.resume
else
  cap 16G taskset -c 10 $PY code/peak.py labels_prov logs -- $PY -X utf8 code/nt_labels.py --tree "$W/tree" \
    --logs-dir "$LOGS" --navtrain-filter "$W/inputs/navtrain_filter_v11.json" \
    --heldout "$W/tree/stack/tanitad/configs/refcv8_nt/nt_heldout.json" \
    --merge-clock /home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl --out "$L/a1_prov" > logs/labels_prov.log 2>&1
  echo $? > logs/EXIT_labels_prov
fi
done_ok labels_prov && [ -s "$L/a1_prov/token_table_nt.json" ] || exit 3
# 2. joins: 4 shards, one core each
if done_ok joins && shards_ok; then
  echo "$(date +%T) step 2 complete (4 shard reports) -- skipped" >> logs/build.resume
else
  for k in 0 1 2 3; do
    cap 12G taskset -c $((10 + k)) $PY code/peak.py joins_shard_$k logs -- $PY -X utf8 code/nt_joins.py build \
      --logs-dir "$LOGS" --token-table "$L/a1_prov/token_table_nt.json" --shard "$k/4" --out "$L/a1_joins/shard_$k" \
      > logs/joins_shard_$k.log 2>&1 &
  done
  wait
  shards_ok || { echo "a shard produced no report" > logs/EXIT_joins; exit 4; }
  echo 0 > logs/EXIT_joins
fi
# 3. gather (GATED: run 2 merged after a SIGKILLed gather)
if ! { done_ok gather && [ -s "$L/a1_joins/gathered/JOINS_REPORT.json" ]; }; then
  rm -rf "$L/a1_joins/gathered"
  cap 16G taskset -c 10 $PY code/peak.py gather logs -- $PY -X utf8 code/nt_joins.py gather \
    --shards "$L/a1_joins/shard_0" "$L/a1_joins/shard_1" "$L/a1_joins/shard_2" "$L/a1_joins/shard_3" \
    --out "$L/a1_joins/gathered" > logs/joins_gather.log 2>&1
  echo $? > logs/EXIT_gather
fi
{ done_ok gather && [ -s "$L/a1_joins/gathered/JOINS_REPORT.json" ]; } || exit 5
# 4. merge with eval139 (the PhysicalAI in-run eval = the forgetting monitor)
if ! { done_ok merge && [ -s "$L/a1/MERGE_REPORT.json" ]; }; then
  rm -f "$L/a1"/nt_train_plus_eval139_* "$L/a1/MERGE_REPORT.json"
  cap 24G taskset -c 10 $PY code/peak.py merge logs -- $PY -X utf8 code/nt_joins.py merge \
    --parts "$L/a1_joins/gathered" \
    --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139 \
    --pai-join /home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz \
    --pai-join3d /home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz \
    --pai-vis1 /home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz \
    --pai-extrinsics /home/nvidia/data/refcv8_train4572_eval139_extrinsics.json \
    --tag nt_train_plus_eval139 --out "$L/a1" > logs/joins_merge.log 2>&1
  echo $? > logs/EXIT_merge
fi
{ done_ok merge && [ -s "$L/a1/MERGE_REPORT.json" ]; } || exit 6
# 5. the provisional clock reference (G-CLOCK's independent instrument)
cap 8G taskset -c 10 $PY code/peak.py clockref_prov logs -- $PY -X utf8 code/nt_clock_reference.py \
  --logs-dir "$LOGS" --token-table "$L/a1_prov/token_table_nt.json" \
  --out "$L/a1_prov/nt_clock_reference_a1_prov.json" > logs/clockref_prov.log 2>&1
echo $? > logs/EXIT_clockref_prov
echo DONE > logs/BUILD_DONE
