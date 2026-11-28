#!/usr/bin/env bash
# pod_selftest_v8proof.sh -- lead's smoke of the ADD45 tools on the v8proof pod (the pod of this fork), CPU only, before the kit goes to the other session.
# This pod has the corpus only, so a stand-in "union" is used: the corpus cache as the union, the first half of its manifest as the "corpus list", the other half as "new"
# clips. It exercises every code path of the precheck (counts, d1, the planted d1, d2, d2j, d2n, d2g) and a 20-step training of the port with the clip list, the evaluation
# guard and the closing weight 0, plus one refusal each. It is NOT a result of the card: no number of it is a read. Usage: bash pod_selftest_v8proof.sh > SELFTEST.txt 2>&1
set -u
D=/workspace/v8proof/dscale_lead; PY=/workspace/venv/bin/python; TREE=/workspace/v8proof/tree_leadsup; DATA=/workspace/v8proof/data
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
W=$D/selftest_$(date -u +%H%M%S); mkdir -p "$W"
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 20 ] || { echo "ZZ SELFTEST STOP memory"; exit 3; }
"$PY" - "$DATA/cache_pod/train/manifest.jsonl" "$DATA/cache_pod/eval/manifest.jsonl" "$W" <<'EOF'
import json, sys
ids = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()))
ev = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[2], encoding="utf-8") if l.strip()))
half = ids[: len(ids) // 2]
open(sys.argv[3] + "/half.txt", "w", encoding="utf-8", newline="\n").write("\n".join(half) + "\n")
open(sys.argv[3] + "/small.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[600:760]) + "\n")
open(sys.argv[3] + "/eval_ids.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ev) + "\n")
open(sys.argv[3] + "/bad_guard.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[600:602]) + "\n")
open(sys.argv[3] + "/missing.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[600:610] + ["not_a_clip_id"]) + "\n")
print("manifest clips", len(ids), "half", len(half), "eval", len(ev))
EOF
N=$(wc -l < "$W/half.txt"); T=$(grep -c . "$DATA/cache_pod/train/manifest.jsonl")
LEAD=${LEAD:-$D/standin/lead_train.npz}   # stand-in: the dev box's corpus lead file (sha256 f0440deb...), not the 4,572-clip file of the other pod (25494fd9...)
echo "lead file $(sha256sum "$LEAD" | cut -c1-16)"
PC=("$PY" "$D/dscale_precheck.py" --union-cache "$DATA/cache_pod" --bset-list "$W/half.txt" --lead-union "$LEAD" --lead-corpus "$LEAD"
    --eval-manifest "$DATA/cache_pod/eval/manifest.jsonl" --join-union "$DATA/inputs/join_train.npz" --expect-union "$T" --expect-corpus "$N")
"${PC[@]}" --out "$W/PRECHECK.json"; echo "precheck rc $? (expected 0)"
"${PC[@]}" --plant --out "$W/PRECHECK_PLANT.json"; echo "precheck with a planted shared id rc $? (expected 2)"
"$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
for k,v in d['checks'].items(): print('  ', k, json.dumps(v)[:420])
" "$W/PRECHECK.json"
TARGV="--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --out X --cache-dir $DATA/cache_pod --v9-train $DATA/v9/v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train $DATA/v9/lead_train4572.npz --speed-train $DATA/v9/refcv6_speed_max_v8_train.jsonl --join-train $DATA/inputs/join_train.npz --p10 $DATA/inputs/p10_prior.pt --kin-mode yaw --tracks $DATA/inputs/native_tracks.pt --batch 64 --steps-mult 4 --device cuda"
trn() { local name=$1; shift; "$PY" "$D/tok_probe_u.py" train --grid pool4x10 --tree "$TREE" --out "$W/$name" --arms TK16_s0 --trainer-argv "$TARGV" --device cpu --threads 3 "$@" > "$W/$name.log" 2>&1; echo "train $name rc $?"; tail -n 2 "$W/$name.log" | cut -c1-300; }
trn ok        --steps 20 --expect-steps 20 --w-clos 0.0 --clip-list "$W/small.txt" --eval-guard "$W/eval_ids.txt"
trn shuf      --steps 20 --expect-steps 20 --w-clos 0.0 --clip-list "$W/small.txt" --shuffle-gap-labels
trn guard     --steps 20 --expect-steps 20 --w-clos 0.0 --clip-list "$W/small.txt" --eval-guard "$W/bad_guard.txt"      # must refuse
trn missing   --steps 20 --expect-steps 20 --w-clos 0.0 --clip-list "$W/missing.txt"                                    # must refuse
trn nosteps   --w-clos 0.0 --clip-list "$W/small.txt"                                                                   # must refuse
trn badw      --steps 20 --expect-steps 20 --w-clos 0.5 --clip-list "$W/small.txt"                                      # must refuse
ls "$W/ok" 2>/dev/null | head -n 12 | tr '\n' ' '; echo
"$PY" -c "
import json,glob,sys
for p in sorted(glob.glob(sys.argv[1] + '/*/**/train_meta.json', recursive=True))[:2]:
    d=json.load(open(p)); print(p.split('selftest_')[-1], {k: d[k] for k in d if any(s in k for s in ('clip','guard','clos','shuffle','steps','n_win','windows'))})
" "$W"
grep -h "lead_sup" "$W/ok"/*/train_log.jsonl 2>/dev/null | tail -n 1 | cut -c1-300
echo "ZZ SELFTEST DONE $W"
