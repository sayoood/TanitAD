#!/bin/bash
# dry re-gate of nt-a1-s0 (read-only on the run dir; outputs to a scratch dir) -- MM 2026-10-26
set -u
RUN=/home/nvidia/r8ns/nt/runs/nt-a1-s0; OUT=/home/nvidia/r8ns/nt/regate_20261026; mkdir -p "$OUT"
PY=/home/nvidia/venvs/tanitad-train/bin/python; ZG=/home/nvidia/r8ns/nt/launch_tree/stack/scripts/zerograd_gate.py
source "$OUT/allow_strict.sh"
A=(); for p in "${ZEROGRAD_ALLOW[@]}"; do A+=(--allow "$p"); done
$PY "$ZG" --ckpt "$RUN/ckpt.pt" "${A[@]}" --json-out "$OUT/dry_strict.json" > "$OUT/dry_strict.txt" 2>&1; echo "strict_exit=$?"
$PY "$ZG" --ckpt "$RUN/ckpt.pt" "${A[@]}" --allow 'core\.decoder\.r8_rc_gate' --json-out "$OUT/dry_plus_rcgate.json" > "$OUT/dry_plus_rcgate.txt" 2>&1; echo "plus_exit=$?"
for f in dry_strict dry_plus_rcgate; do $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(sys.argv[2],d['verdict'],'offenders',[o['names'] for o in d['offenders']],'allowed',len(d['allowed']),'unused',len(d['unused_allow']))" "$OUT/$f.json" "$f"; done
