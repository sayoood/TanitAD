#!/usr/bin/env bash
# H-NT-DECIDE-1 controls K1 (train feed == inference feed) and K2 (targets == devkit future) AT SCALE, on Thor's CPU:
# 200 PRODUCTION A1 train tokens (E1's v2ep_a1/), a SEEDED sample of E1's manifest taken at run time. Each control runs
# twice: the real check (must PASS) and its deliberate-regression arm (--mutate, must go RED). 0 GPU, nice 19, cores
# 10-13, 4 threads; starts only after thor_build_a1.sh wrote BUILD_DONE (its 4 join shards hold the same 4 cores).
#
# Inputs are independent on purpose: TRAIN side = E1's files (frames/poses) + MY labels (a1_prov v9/speed) + MY joins
# rig table (gathered/nt_rig_extrinsics.json); INFERENCE side = the bridge's own functions/literals + E1's RIGS_A1.json
# + the devkit export (a1_inputs.json.gz = SceneLoader.get_agent_input_from_token).
# ⛔ The sample gets its OWN cache dir of symlinks: build_v2_providers() writes _v2manifest.pt INTO the cache dir, and
# E1's live v2ep_a1/ must never receive a file from this job.
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
C=$L/a1_controls
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
N=${N_SAMPLE:-200}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts" NT_TREE="$W/tree"
mkdir -p "$C/logs"
cd "$W" || exit 2
while [ ! -s logs/BUILD_DONE ]; do sleep 60 200>&-; done
[ "$(cat logs/EXIT_gather 2>/dev/null)" = 0 ] || { echo "gather not 0: $(cat logs/EXIT_gather 2>/dev/null)" > "$C/logs/EXIT"; exit 3; }
X="$L/a1_joins/gathered/nt_rig_extrinsics.json"
[ -s "$X" ] || { echo "no rig table $X" > "$C/logs/EXIT"; exit 3; }
# 1. the sample: complete manifest lines whose file exists at the declared size and whose clip is in the token table;
#    ranked by sha256(token) (seed-free, reproducible), first N
rm -rf "$C/cache"; mkdir -p "$C/cache"
nice -n 19 taskset -c 10 $PY - "$NT/MANIFEST_A1.jsonl" "$NT/v2ep_a1" "$L/a1_prov/token_table_nt.json" "$C" "$N" <<'PYEOF' > "$C/logs/sample.log" 2>&1
import hashlib, json, os, sys
man, d, tt, out, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
cids = {r["clip_id"] for r in json.load(open(tt))["rows"]}
rows, bad = [], 0
for l in open(man):
    try:
        r = json.loads(l)
    except Exception:
        bad += 1                                  # a line E1 is still writing
        continue
    p = os.path.join(d, r["file"])
    if r["clip_id"] in cids and os.path.isfile(p) and os.path.getsize(p) == int(r["bytes"]):
        rows.append(r)
rows.sort(key=lambda r: hashlib.sha256(r["token"].encode()).hexdigest())
rows = rows[:n]
with open(os.path.join(out, "manifest_sample.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        os.symlink(os.path.join(d, r["file"]), os.path.join(out, "cache", r["file"]))
open(os.path.join(out, "sample_tokens.txt"), "w").write("".join(r["token"] + "\n" for r in rows))
print(json.dumps({"n_eligible_seen": len(rows), "n_sample": len(rows), "unparsable_lines": bad}))
PYEOF
echo $? > "$C/logs/EXIT_sample"
[ "$(wc -l < "$C/sample_tokens.txt")" -ge 1 ] || { echo "empty sample" > "$C/logs/EXIT"; exit 4; }
nice -n 19 taskset -c 10 $PY code/thor_pull_devkit_inputs.py "$NT/inputs/a1_inputs.json.gz" "$C/sample_tokens.txt" \
  "$C/devkit_inputs_sample.json" > "$C/logs/devkit.log" 2>&1
echo $? > "$C/logs/EXIT_devkit"
K1="--cache $C/cache --e1-manifest $C/manifest_sample.jsonl --token-table $L/a1_prov/token_table_nt.json --logs-dir $LOGS
    --extrinsics $X --rigs $NT/RIGS_A1.json --v9 $L/a1_prov/v9_labels_nt_train.npz
    --speed $L/a1_prov/speed_label_nt_train.jsonl --n-lift 10"
K2="--cache $C/cache --token-table $L/a1_prov/token_table_nt.json --logs-dir $LOGS --e1-manifest $C/manifest_sample.jsonl
    --devkit-inputs $C/devkit_inputs_sample.json"
nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k1-feed $K1 --out "$C/K1_feed_A1_200.json" > "$C/logs/k1.log" 2>&1
echo $? > "$C/logs/EXIT_k1"
nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k1-feed $K1 --mutate --out "$C/K1_feed_A1_200_mutation.json" > "$C/logs/k1_mut.log" 2>&1
echo $? > "$C/logs/EXIT_k1_mut"
nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --out "$C/K2_targets_A1_200.json" > "$C/logs/k2.log" 2>&1
echo $? > "$C/logs/EXIT_k2"
nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --mutate --out "$C/K2_targets_A1_200_mutation.json" > "$C/logs/k2_mut.log" 2>&1
echo $? > "$C/logs/EXIT_k2_mut"
# the verdict is read from the ARTIFACTS (PASS fields), never the exit codes
$PY - "$C" <<'PYEOF' > "$C/CONTROLS_A1.json"
import json, os, sys
c = sys.argv[1]
def rd(n):
    p = os.path.join(c, n)
    return json.load(open(p)).get("PASS") if os.path.isfile(p) else "MISSING"
r = {"K1": rd("K1_feed_A1_200.json"), "K1_mutation_PASS_field": rd("K1_feed_A1_200_mutation.json"),
     "K2": rd("K2_targets_A1_200.json"), "K2_mutation_PASS_field": rd("K2_targets_A1_200_mutation.json")}
r["ALL_OK"] = r["K1"] is True and r["K2"] is True and r["K1_mutation_PASS_field"] is False and r["K2_mutation_PASS_field"] is False
print(json.dumps(r, indent=1))
PYEOF
echo DONE > "$C/logs/CONTROLS_DONE"
