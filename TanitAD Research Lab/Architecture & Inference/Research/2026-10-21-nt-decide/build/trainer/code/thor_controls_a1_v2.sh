#!/usr/bin/env bash
# H-NT-DECIDE-1 controls K1 (train feed == inference feed) and K2 (targets == devkit future) AT SCALE, on Thor's CPU:
# 200 PRODUCTION A1 train tokens (E1's v2ep_a1/), a seed-free sample of E1's manifest (ranked by sha256(token)).
# Each control runs twice: the real check (must PASS) and its deliberate-regression arm (--mutate, must go RED).
# 0 GPU, nice 19, cores 10-13, 4 threads.
#
#   phase A -- after thor_build_a1.sh's BUILD_DONE (its 4 join shards hold the same 4 cores): the PROVISIONAL labels
#              (a1_prov) + the MERGED rig table the trainer reads (--agent-rig-extrinsics);
#   phase B -- after thor_finalize_a1.sh's FINAL_DONE (E1 complete): EXACTLY the files the A1 argv names
#              (labels/a1/{token_table_nt.json, v9_labels_nt_train.npz, speed_label_nt_train.jsonl, ..._extrinsics.json}).
#
# Inputs are independent on purpose: TRAIN side = E1's files (frames/poses) + MY labels + MY joins rig table;
# INFERENCE side = the bridge's own functions/literals + E1's RIGS_A1.json + the devkit export (a1_inputs.json.gz).
# ⛔ The sample gets its OWN cache dir of symlinks: build_v2_providers() writes _v2manifest.pt INTO the cache dir, and
# E1's live v2ep_a1/ must never receive a file from this job.
# v2 of thor_controls_a1.sh (v1 read the shard-GATHERED rig table, which the trainer does not read; never ran a step).
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
C=$L/a1_controls
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
N=${N_SAMPLE:-200}
TAG=nt_train_plus_eval139
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts" NT_TREE="$W/tree"
mkdir -p "$C/logs"
cd "$W" || exit 2

run_phase() {  # $1 = phase tag, $2 = labels dir (token table / v9 / speed)
  local P=$1 LAB=$2 X="$L/a1/${TAG}_extrinsics.json" K1 K2
  [ -s "$X" ] && [ -s "$LAB/token_table_nt.json" ] && [ -s "$LAB/v9_labels_nt_train.npz" ] \
    || { echo "phase $P: an input is missing" > "$C/logs/EXIT_$P"; return 3; }
  K1="--cache $C/cache --e1-manifest $C/manifest_sample.jsonl --token-table $LAB/token_table_nt.json --logs-dir $LOGS
      --extrinsics $X --rigs $NT/RIGS_A1.json --v9 $LAB/v9_labels_nt_train.npz
      --speed $LAB/speed_label_nt_train.jsonl --n-lift 10"
  K2="--cache $C/cache --token-table $LAB/token_table_nt.json --logs-dir $LOGS --e1-manifest $C/manifest_sample.jsonl
      --devkit-inputs $C/devkit_inputs_sample.json"
  nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k1-feed $K1 --out "$C/K1_${P}.json" > "$C/logs/k1_$P.log" 2>&1
  nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k1-feed $K1 --mutate --out "$C/K1_${P}_mutation.json" > "$C/logs/k1_${P}_mut.log" 2>&1
  nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --out "$C/K2_${P}.json" > "$C/logs/k2_$P.log" 2>&1
  nice -n 19 taskset -c 10-13 $PY -X utf8 code/nt_controls.py k2-targets $K2 --mutate --out "$C/K2_${P}_mutation.json" > "$C/logs/k2_${P}_mut.log" 2>&1
  # the verdict is read from the ARTIFACTS (PASS fields), never from exit codes
  $PY - "$C" "$P" "$LAB" "$X" <<'PYEOF' > "$C/CONTROLS_${P}.json"
import json, os, sys
c, p, lab, x = sys.argv[1:5]
def rd(n):
    q = os.path.join(c, n)
    return json.load(open(q)).get("PASS") if os.path.isfile(q) else "MISSING"
r = {"phase": p, "labels": lab, "extrinsics": x,
     "K1": rd(f"K1_{p}.json"), "K1_mutation_PASS_field": rd(f"K1_{p}_mutation.json"),
     "K2": rd(f"K2_{p}.json"), "K2_mutation_PASS_field": rd(f"K2_{p}_mutation.json")}
r["ALL_OK"] = (r["K1"] is True and r["K2"] is True
               and r["K1_mutation_PASS_field"] is False and r["K2_mutation_PASS_field"] is False)
print(json.dumps(r, indent=1))
PYEOF
  echo DONE > "$C/logs/EXIT_$P"
}

# ---- phase A ------------------------------------------------------------------------------------------------------
while [ ! -s logs/BUILD_DONE ]; do sleep 60 200>&-; done
[ "$(cat logs/EXIT_merge 2>/dev/null)" = 0 ] || { echo "merge not 0: $(cat logs/EXIT_merge 2>/dev/null)" > "$C/logs/EXIT"; exit 3; }
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
n_elig = len(rows)
rows.sort(key=lambda r: hashlib.sha256(r["token"].encode()).hexdigest())
rows = rows[:n]
with open(os.path.join(out, "manifest_sample.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        os.symlink(os.path.join(d, r["file"]), os.path.join(out, "cache", r["file"]))
open(os.path.join(out, "sample_tokens.txt"), "w").write("".join(r["token"] + "\n" for r in rows))
print(json.dumps({"n_eligible": n_elig, "n_sample": len(rows), "unparsable_lines": bad}))
PYEOF
[ "$(wc -l < "$C/sample_tokens.txt" 2>/dev/null || echo 0)" -ge 1 ] || { echo "empty sample" > "$C/logs/EXIT"; exit 4; }
nice -n 19 taskset -c 10 $PY code/thor_pull_devkit_inputs.py "$NT/inputs/a1_inputs.json.gz" "$C/sample_tokens.txt" \
  "$C/devkit_inputs_sample.json" > "$C/logs/devkit.log" 2>&1
run_phase A_prov "$L/a1_prov"
# ---- phase B ------------------------------------------------------------------------------------------------------
while [ ! -s logs/FINAL_DONE ]; do sleep 120 200>&-; done
run_phase B_final "$L/a1"
echo DONE > "$C/logs/CONTROLS_DONE"
