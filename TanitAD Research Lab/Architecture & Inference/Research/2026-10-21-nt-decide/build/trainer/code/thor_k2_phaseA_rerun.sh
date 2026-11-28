#!/usr/bin/env bash
# H-NT-DECIDE-1: re-run of phase A's K2 pair after thor_controls_a1_v3.sh's devkit pull died (2026-10-21 23:30 Berlin:
# code/thor_pull_devkit_inputs.py had never been shipped to Thor, so devkit_inputs_sample.json was never written and
# both K2 calls raised FileNotFoundError). Same caps / cores / inputs as run_phase A_prov; K1 is NOT re-run (it read
# PASS, mutation RED). Writes devkit_inputs_sample.json (which phase B, still waiting in the v3 script, then reads),
# K2_A_prov{,_mutation}.json and rewrites CONTROLS_A_prov.json from the ARTIFACTS.
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
C=$L/a1_controls
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
TAG=nt_train_plus_eval139
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts" NT_TREE="$W/tree"
cap() { local M=$1; shift; systemd-run --user --scope -q -p MemoryMax="$M" -p MemorySwapMax=0 nice -n 19 "$@" 200>&-; }
cd "$W" || exit 2
P=A_prov; LAB=$L/a1_prov
cap 8G taskset -c 10 $PY code/peak.py devkit_pull "$C/logs" -- $PY code/thor_pull_devkit_inputs.py \
  "$NT/inputs/a1_inputs.json.gz" "$C/sample_tokens.txt" "$C/devkit_inputs_sample.json" > "$C/logs/devkit.log" 2>&1
echo $? > "$C/logs/EXIT_devkit"
[ -s "$C/devkit_inputs_sample.json" ] || { echo "no devkit sample" > "$C/logs/EXIT_k2rerun"; exit 3; }
K2="--cache $C/cache --token-table $LAB/token_table_nt.json --logs-dir $LOGS --e1-manifest $C/manifest_sample.jsonl
    --devkit-inputs $C/devkit_inputs_sample.json"
cap 16G taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --out "$C/K2_${P}.json" > "$C/logs/k2_$P.log" 2>&1
cap 16G taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --mutate --out "$C/K2_${P}_mutation.json" > "$C/logs/k2_${P}_mut.log" 2>&1
$PY - "$C" "$P" "$LAB" "$L/a1/${TAG}_extrinsics.json" <<'PYEOF' > "$C/CONTROLS_${P}.json"
import json, os, sys
c, p, lab, x = sys.argv[1:5]
def rd(n):
    q = os.path.join(c, n)
    return json.load(open(q)).get("PASS") if os.path.isfile(q) else "MISSING"
r = {"phase": p, "labels": lab, "extrinsics": x,
     "K1": rd(f"K1_{p}.json"), "K1_mutation_PASS_field": rd(f"K1_{p}_mutation.json"),
     "K2": rd(f"K2_{p}.json"), "K2_mutation_PASS_field": rd(f"K2_{p}_mutation.json"),
     "note": "K2 pair re-run by thor_k2_phaseA_rerun.sh (devkit pull script had not been shipped)"}
r["ALL_OK"] = (r["K1"] is True and r["K2"] is True
               and r["K1_mutation_PASS_field"] is False and r["K2_mutation_PASS_field"] is False)
print(json.dumps(r, indent=1))
PYEOF
echo DONE > "$C/logs/EXIT_k2rerun"
