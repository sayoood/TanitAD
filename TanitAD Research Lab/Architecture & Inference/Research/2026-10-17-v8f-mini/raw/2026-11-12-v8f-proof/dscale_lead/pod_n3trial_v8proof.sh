#!/usr/bin/env bash
# pod_n3trial_v8proof.sh -- lead's trial of note N3 of card ADD45 (the evaluation view) on the v8proof pod, CPU only, in a folder of its own: the kit folder that the
# running card ADD60 uses is not touched. Not a result of the card.
#   1. the tests of dscale_evalview.py and its four mutations, under the stand-in runner (it is not pytest and says so in its last line);
#   2. the kit stages precheck and cpucheck, run for real on stand-in data, with an evaluation cache whose manifest is this pod's own manifest REVERSED:
#      the check of the kit as landed (hold 77) fails on that source, the N3 kit must pass, and the view manifest must equal this pod's manifest (the study's) byte for byte.
# Usage: bash pod_n3trial_v8proof.sh > N3TRIAL.txt 2>&1
set -u
N=/workspace/v8proof/dscale_lead_n3; OLD=/workspace/v8proof/dscale_lead; DATA=/workspace/v8proof/data; PY=/workspace/venv/bin/python
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat); free=$(( (mm - an) / 1073741824 )); echo "container free ${free} GB"
[ "$free" -ge 20 ] || { echo "ZZ N3TRIAL STOP memory"; exit 3; }
cd "$N" || exit 3
W=$N/trial_$(date -u +%H%M%S); mkdir -p "$W"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
echo "== 1. tests and mutations of the evaluation view"
PYTHONPATH=$OLD/pytest_shim "$PY" -m pytest -q -p no:cacheprovider test_dscale_evalview.py --basetemp "$W/pt_ev" > "$W/test_ev.out" 2>&1; rc=$?
echo "test_dscale_evalview rc $rc :: $(tail -n 1 "$W/test_ev.out" | cut -c1-200)"
[ $rc -eq 0 ] || { grep -n "FAILED\|Error\|assert\|SKIP" "$W/test_ev.out" | head -n 20 | cut -c1-260; }
grep -c "SKIP" "$W/test_ev.out" | sed 's/^/skipped lines: /'
PYTHONPATH=$OLD/pytest_shim "$PY" run_mutations_u.py --targets evalview > "$W/mut.out" 2>&1; echo "mutation runner rc $?"
cut -c1-220 "$W/mut.out" | tail -n 8
echo "== 2. the kit on a reversed evaluation manifest"
S=$W/revcache; mkdir -p "$S/eval"
tac "$DATA/cache_pod/eval/manifest.jsonl" > "$S/eval/manifest.jsonl"
ln -s "$DATA/cache_pod/eval/clips" "$S/eval/clips"; ln -s "$DATA/cache_pod/eval/meta.json" "$S/eval/meta.json"
"$PY" - "$DATA/cache_pod/train/manifest.jsonl" "$W" <<'EOF'
import json, sys
ids = list(dict.fromkeys(json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()))
open(sys.argv[2] + "/half.txt", "w", encoding="utf-8", newline="\n").write("\n".join(ids[: len(ids) // 2]) + "\n")
print("manifest clips", len(ids), "half", len(ids) // 2)
EOF
"$PY" "$OLD/dscale_tokens_same.py" --eval-cache "$S" --union-cache "$DATA/cache_pod" --bset-list "$W/half.txt" --study-eval-tsv "$OLD/STUDY_EVAL_MD5.tsv" --study-train-tsv "$OLD/STUDY_TRAIN_MD5.tsv" --out "$W/EVALSET_reversed_source.json" > "$W/evalset_reversed_source.out" 2>&1
echo "the landed check on the reversed source: rc $? (must be 2) :: $(sed -n '2p' "$W/evalset_reversed_source.out" | cut -c1-200)"
LEAD=$OLD/standin/lead_train.npz
export KIT=$N TREE=$OLD/ship_b/tree OUT=$W/out PY
export CACHE=$DATA/cache_pod EVAL_CACHE=$S
export V9_UNION=$DATA/v9/v9_labels_train.npz V9_MD5=83e30bd3791f0767707eade5f4671759 LEAD_UNION=$LEAD LEAD_CORPUS=$LEAD SPEED_UNION=$DATA/v9/refcv6_speed_max_v8_train.jsonl JOIN_UNION=$DATA/inputs/join_train.npz
export BSET=$W/half.txt P10=$DATA/inputs/p10_prior.pt TRACKS=$DATA/inputs/native_tracks.pt
export SHA_LEAD_UNION=$(sha256sum "$LEAD" | cut -c1-8) SHA_LEAD_CORPUS=$(sha256sum "$LEAD" | cut -c1-8) SHA_BSET=$(sha256sum "$BSET" | cut -c1-8)
export EXPECT_UNION=4572 EXPECT_CORPUS=2286 MIN_FREE_GB=0 THREADS=3
for st in precheck cpucheck; do
  bash "$N/pod_dscale_lead_run.sh" "$st" > "$W/$st.out" 2>&1; echo "stage $st rc $?"
  grep "evaluation view\|evaluation-set check\|pins ok\|STOP\|REFUSED" "$W/$st.out" | cut -c1-300
done
cmp "$W/out/eval_view/eval/manifest.jsonl" "$DATA/cache_pod/eval/manifest.jsonl" && echo "the view manifest equals this pod's study manifest byte for byte"
cmp -s "$S/eval/manifest.jsonl" "$DATA/cache_pod/eval/manifest.jsonl" || echo "the reversed source differs from it (as built)"
ls -la "$W/out/eval_view/eval" | cut -c1-150
"$PY" - "$W/out" <<'EOF'
import json, sys
o = sys.argv[1]
for st in ("precheck", "cpucheck"):
    d = json.load(open(o + "/eval_view/EVALVIEW.%s.json" % st, encoding="utf-8"))
    print(st, {k: d.get(k) for k in ("pass", "source_in_study_order", "positions_that_differ", "manifest_written_by_this_call", "view_in_study_order", "lines_unchanged", "links")})
e = json.load(open(o + "/precheck/EVALSET.json", encoding="utf-8")); p = json.load(open(o + "/precheck/EVALSET_PLANTED.json", encoding="utf-8"))
print("d2e on the view:", e["d2e_eval_set_is_the_study_set"]["pass"], "planted swap:", p["d2e_eval_set_is_the_study_set"]["pass"], "d3t eval same md5:", e["d3t_tokens_same"]["eval_clips_same_content_md5"])
EOF
echo "extract under DRY=1 reads the evaluation set from:"
DRY=1 bash "$N/pod_dscale_lead_run.sh" extract 2>&1 | grep -o -- "--cache [^ ]*" | sort | uniq -c
ls "$W/out/markers" | tr '\n' ' '; echo
echo "ZZ N3TRIAL DONE $W"
