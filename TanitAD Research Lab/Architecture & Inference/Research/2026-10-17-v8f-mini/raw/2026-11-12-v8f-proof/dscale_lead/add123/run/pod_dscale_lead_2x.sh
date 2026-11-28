#!/usr/bin/env bash
# pod_dscale_lead_2x.sh -- run kit of card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X), generated from pod_dscale_lead_run.sh by make_2x_script.py, for the pod of the other v8F session. Tier T0; no model is trained.
#   usage: bash pod_dscale_lead_2x.sh <stage>      stage = precheck | cpucheck | smoke | run | extract | checklog      (DRY=1 prints every command and runs nothing)
#   precheck : CPU only. tree pins, input sha256 prefixes, dscale_precheck.py (counts, d1, d2, d2j, d2n, d2g) and its planted d1 control (must exit 2), dscale_clockcheck.py (d2c) and its planted shift (must exit 2)       -> $OUT/precheck/
#   cpucheck : CPU only, no GPU lock. Two training steps on the union and two with the corpus list: shows that the study tree reads the union files, before any GPU slot is taken      -> $OUT/cpucheck_U, $OUT/cpucheck_C
#   smoke    : 200 steps of the four arms of folder runQ form (banked tool), under the flock; prints the projected GPU hours of the ONE training process (seconds/200 x 38328); exit 4 ESTIMATE-ABOVE-0.9H if above $MAX_H (0.9)
#   run      : runQ = the arms TK16_s0 to TK16_s3 at 38328 steps, ONE GPU process under the flock (threads <= 4) through tok_probe_u_ck.py (--ck-steps 19164,28746), under timeout TRAIN_LIMIT_S (3300); rc 124 = stop TRAIN-ABOVE-TIME-LIMIT
#   extract  : evaluation split only, tokens from $EVAL_CACHE, per arm THREE extracts under the flock (the last state; steps 19164 and 28746 through --ckpt into states/runQ_s19164 and states/runQ_s28746)                                                         -> $OUT/states/<folder>/
#   checklog : control p5 / d4 (train-log check) per folder                                                                                       -> $OUT/checklog/
# Every stage: refuses under $MIN_FREE_GB free container memory (cgroup v1: memory.limit_in_bytes minus total_rss of memory.stat; usage_in_bytes counts page cache and is not used),
# is resumable through marker files ($OUT/markers/), never overwrites an existing output folder, and writes $OUT/DSCALE_2X_STOP.txt (with the reason) on a stop, $OUT/DSCALE_2X_DONE.txt after checklog.
# Nothing is deleted or moved. Exit codes: 0 ok, 1 stop, 3 refused to start, 4 estimate above the bar.
set -u
# Note N3: the evaluation set is read from a VIEW folder ($OUT/eval_view) whose manifest holds the lines of the pod's own evaluation manifest in the study's order
# (dscale_evalview.py; clips linked, nothing copied, no line edited). Every stage builds or re-checks the view; control d2e is checked on the view.
STAGE=${1:-}
PY=${PY:-/root/venv/bin/python}
KIT=${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TREE=${TREE:-}                                   # SET-ME: the shipped study tree (folder holding stack/)
OUT=${OUT:-}                                     # SET-ME: a NEW output folder
CACHE=${CACHE:-/root/data/cache_union_dscale}
EVAL_CACHE=${EVAL_CACHE:-/root/data/cache_pod}
V9_UNION=${V9_UNION:-/root/dscale10k/merged/v9_labels_union.npz}
V9_MD5=${V9_MD5:-9881b170a889f743c5b0894f9ad13f1d}
LEAD_UNION=${LEAD_UNION:-/root/dscale10k/merged/lead_union.npz}
SPEED_UNION=${SPEED_UNION:-/root/dscale10k/merged/speed_union.jsonl}
JOIN_UNION=${JOIN_UNION:-/root/dscale10k/merged/join_union.npz}
LEAD_CORPUS=${LEAD_CORPUS:-/root/strat1/inputs/data/v9/lead_train4572.npz}
BSET=${BSET:-/root/dscale10k/mix/bset_clips.txt}
P10=${P10:-}                                     # SET-ME: p10 prior (.pt) for the union clips
TRACKS=${TRACKS:-}                               # SET-ME: native tracks (.pt) that cover the 14,512 union clips
EVAL_IDS=${EVAL_IDS:-}                           # optional: file with the 48 evaluation episodes' clip ids (the eval cache manifest is always used)
CLOCK_CHECK_JSON=${CLOCK_CHECK_JSON:-}           # optional: result file of the tree owner session's same-clock check (recorded with its sha256)
SHA_LEAD_UNION=${SHA_LEAD_UNION:-e96c2d5e}; SHA_LEAD_CORPUS=${SHA_LEAD_CORPUS:-25494fd9}; SHA_BSET=${SHA_BSET:-37f4f9ee}
PINS=${PINS:-$KIT/TREE_PINS_2X.txt}
LOCK=${LOCK:-/root/gpu_queue.lock}
THREADS=${THREADS:-4}
STEPS=${STEPS:-19164}; STEPS_U2=${STEPS_U2:-38328}; SMOKE_STEPS=${SMOKE_STEPS:-200}; TRAIN_LIMIT_S=${TRAIN_LIMIT_S:-3300}
MAX_H=${MAX_H:-0.9}
EXPECT_UNION=${EXPECT_UNION:-14512}; EXPECT_CORPUS=${EXPECT_CORPUS:-4512}     # clip counts the precheck requires (the card: 14,512 union clips, 4,512 corpus clips)
MIN_FREE_GB=${MIN_FREE_GB:-20}
CG_LIMIT=${CG_LIMIT:-/sys/fs/cgroup/memory/memory.limit_in_bytes}
CG_STAT=${CG_STAT:-/sys/fs/cgroup/memory/memory.stat}
EVAL_RANGE=${EVAL_RANGE:-0:141}
DEVICE=${DEVICE:-cuda}
DRY=${DRY:-0}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
TP=$KIT/tok_probe_u.py
TPCK=$KIT/tok_probe_u_ck.py
PC=$KIT/dscale_precheck.py
CK=$KIT/dscale_clockcheck.py
TS=$KIT/dscale_tokens_same.py
EV=$KIT/dscale_evalview.py
EVAL_SRC=$EVAL_CACHE                            # the pod's own evaluation cache; after resolve_eval $EVAL_CACHE is the view folder (note N3)
CACHE_T=$CACHE                                  # the folder the trainer is given: $CACHE, or a view folder of two links (resolve_cache)
# the trainer argv of every arm (no --exclude: the merge already dropped the 60 holdout clips; --out, --device, --grid are replaced by the tool)
set_targv() { TARGV="--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --out X --cache-dir $CACHE_T --v9-train $V9_UNION --v9-train-md5 $V9_MD5 --lead-train $LEAD_UNION --speed-train $SPEED_UNION --join-train $JOIN_UNION --p10 $P10 --kin-mode yaw --tracks $TRACKS --batch 64 --steps-mult 4 --device cuda"; }
set_targv
# folder -> arms, steps, extra flags (the card's arm names: P16Q_sK = runQ/TK16_sK)
FOLDERS="runQ"
folder_arms()  { case $1 in runQ) echo "TK16_s0,TK16_s1,TK16_s2,TK16_s3";; *) echo "TK16_s0";; esac; }
folder_steps() { case $1 in runQ) echo "$STEPS_U2";; *) echo "$STEPS";; esac; }
folder_flags() { case $1 in *) echo "";; esac; }

say() { local l="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] dscale_2x: $*"; echo "$l"; if [ "$DRY" != 1 ] && [ -n "$OUT" ] && [ -d "$OUT" ]; then echo "$l" >> "$OUT/dscale_2x.log"; fi; }
stop() { say "STOP: $1"; if [ "$DRY" != 1 ] && [ -n "$OUT" ]; then mkdir -p "$OUT"; printf 'DSCALE_2X_STOP %s\nstage %s\n%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$STAGE" "$1" > "$OUT/DSCALE_2X_STOP.txt"; fi; exit "${2:-1}"; }
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi; say "run: $*"; "$@"; }
gpu() {  # name, then the command: ONE GPU process, blocking, under the flock
  local name=$1; shift
  local job; job=$(printf '%q ' "$@")
  if [ "$DRY" = 1 ]; then echo "DRY: flock $LOCK bash -c '$job'"; return 0; fi
  say "run under flock ($name): $job"
  flock "$LOCK" bash -c "$job"
}
free_gb() {
  local lim rss
  lim=$(cat "$CG_LIMIT" 2>/dev/null || echo 0)
  rss=$(awk '$1=="total_rss"{print $2}' "$CG_STAT" 2>/dev/null)
  [ -z "$rss" ] && rss=$(awk '$1=="rss"{print $2}' "$CG_STAT" 2>/dev/null)
  if [ -z "$rss" ] || [ "$lim" = 0 ]; then echo 0; return; fi
  echo $(( (lim - rss) / 1073741824 ))
}
need_mem() { [ "$DRY" = 1 ] && { echo "DRY: memory free check (need >= $MIN_FREE_GB GB)"; return 0; }; local g; g=$(free_gb); say "container memory free: ${g} GB (need >= $MIN_FREE_GB)"; [ "$g" -ge "$MIN_FREE_GB" ] || stop "memory free ${g} GB is under $MIN_FREE_GB GB (or the cgroup v1 files were unreadable)" 1; }
marker() { echo "$OUT/markers/$1.done"; }
is_done() { [ -f "$(marker "$1")" ]; }
mark() { [ "$DRY" = 1 ] || { mkdir -p "$OUT/markers"; printf '%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$(marker "$1")"; }; }
placeholder() { case "$1" in ""|SET-ME*|*PLACEHOLDER*) return 0;; esac; return 1; }

preflight() {
  case "$STAGE" in precheck|cpucheck|smoke|run|extract|checklog) ;; *) echo "usage: bash $0 precheck|cpucheck|smoke|run|extract|checklog"; exit 3;; esac
  placeholder "$OUT" && { echo "REFUSED: set OUT (a new output folder)"; exit 3; }
  if [ "$STAGE" != checklog ] && [ "$STAGE" != precheck ]; then
    for v in TREE P10 TRACKS; do placeholder "${!v}" && { echo "REFUSED: SET-ME variable $v is empty"; exit 3; }; done
  fi
  placeholder "$TREE" && { echo "REFUSED: set TREE"; exit 3; }
  if [ "$DRY" != 1 ]; then
    mkdir -p "$OUT/markers"
    [ -f "$OUT/DSCALE_2X_DONE.txt" ] && { echo "REFUSED: $OUT/DSCALE_2X_DONE.txt exists"; exit 3; }
    [ -f "$OUT/DSCALE_2X_STOP.txt" ] && { echo "REFUSED: $OUT/DSCALE_2X_STOP.txt exists (read it, then use a new OUT)"; exit 3; }
    check_pins
  else
    echo "DRY: check_pins $PINS (refuses placeholders and any sha256 that differs)"
  fi
}
check_pins() {   # lines: <sha256> <base>:<relpath>   base = tree (under $TREE) or kit (under $KIT); a placeholder or a difference refuses the start
  [ -f "$PINS" ] || { echo "REFUSED: no pins file $PINS"; exit 3; }
  local n=0 sha ref base rel p got
  while read -r sha ref; do
    case "$sha" in ""|\#*) continue;; esac
    placeholder "$sha" && { echo "REFUSED: pins file still holds a placeholder ($ref)"; exit 3; }
    base=${ref%%:*}; rel=${ref#*:}
    case $base in tree) p=$TREE/$rel;; kit) p=$KIT/$rel;; *) echo "REFUSED: bad pin line $ref"; exit 3;; esac
    got=$(sha256sum "$p" 2>/dev/null | cut -c1-64)
    [ "$got" = "$sha" ] || { echo "REFUSED: sha256 of $p is '$got', the pin says '$sha'"; exit 3; }
    n=$((n+1))
  done < "$PINS"
  [ "$n" -ge 10 ] || { echo "REFUSED: the pins file holds $n pins, need the three tree files and seven kit files (10)"; exit 3; }
  say "pins ok: $n files"
}

resolve_eval() {   # note N3: the kit reads the evaluation set from a view folder: the lines of $EVAL_SRC's manifest in the order of STUDY_EVAL_MD5.tsv, every other entry linked
  if [ "$DRY" = 1 ]; then echo "DRY: resolve_eval: $PY $EV --eval-cache $EVAL_SRC --study-eval-tsv $KIT/STUDY_EVAL_MD5.tsv --view $OUT/eval_view --out $OUT/eval_view/EVALVIEW.$STAGE.json ; then EVAL_CACHE=$OUT/eval_view"; EVAL_CACHE=$OUT/eval_view; return 0; fi
  mkdir -p "$OUT/eval_view"
  "$PY" "$EV" --eval-cache "$EVAL_SRC" --study-eval-tsv "$KIT/STUDY_EVAL_MD5.tsv" --view "$OUT/eval_view" --out "$OUT/eval_view/EVALVIEW.$STAGE.json" > "$OUT/eval_view/evalview.$STAGE.out" 2>&1
  local rc=$?
  say "evaluation view rc=$rc: $(tail -n 1 "$OUT/eval_view/evalview.$STAGE.out" | cut -c1-220)"
  [ $rc -eq 0 ] || stop "the evaluation view was not built, or differs from the view of an earlier stage (note N3, rc $rc): read $OUT/eval_view/EVALVIEW.$STAGE.json" 1
  EVAL_CACHE=$OUT/eval_view
}

resolve_cache() {   # the trainer reads <cache>/train/manifest.jsonl and <cache>/train/clips/; a cache with manifest.jsonl and clips/ at its root gets a view folder of two links (nothing is copied)
  if [ "$DRY" = 1 ]; then echo "DRY: resolve_cache: $CACHE/train if it holds a manifest, else the view folder $OUT/cache_view (links train -> $CACHE, eval -> $EVAL_CACHE/eval)"; return 0; fi
  if [ -f "$CACHE/train/manifest.jsonl" ]; then CACHE_T=$CACHE
  elif [ -f "$CACHE/manifest.jsonl" ]; then
    local v=$OUT/cache_view
    mkdir -p "$v"
    [ -e "$v/train" ] || ln -s "$CACHE" "$v/train"
    [ -e "$v/eval" ] || ln -s "$EVAL_CACHE/eval" "$v/eval"
    { [ -f "$v/train/manifest.jsonl" ] && [ -f "$v/eval/manifest.jsonl" ]; } || stop "the cache view $v does not resolve (train -> $CACHE, eval -> $EVAL_CACHE/eval)" 1
    CACHE_T=$v
  else stop "no manifest.jsonl under $CACHE/train or under $CACHE" 1; fi
  set_targv
  say "training cache folder: $CACHE_T"
}

st_precheck() {
  is_done precheck && { say "precheck done (marker)"; return 0; }
  need_mem
  [ -d "$OUT/precheck" ] && [ "$DRY" != 1 ] && stop "folder $OUT/precheck exists without a marker: not overwritten" 1
  [ "$DRY" = 1 ] || mkdir -p "$OUT/precheck"
  if [ "$DRY" != 1 ]; then
    for pair in "$LEAD_UNION:$SHA_LEAD_UNION" "$LEAD_CORPUS:$SHA_LEAD_CORPUS" "$BSET:$SHA_BSET"; do
      f=${pair%:*}; w=${pair##*:}; got=$(sha256sum "$f" | cut -c1-8)
      [ "$got" = "$w" ] || stop "sha256 prefix of $f is $got, expected $w" 1
    done
  else echo "DRY: sha256 prefix checks of $LEAD_UNION $LEAD_CORPUS $BSET"; fi
  local ev=(); [ -n "$EVAL_IDS" ] && ev=(--eval-ids "$EVAL_IDS")
  local cc=(); [ -n "$CLOCK_CHECK_JSON" ] && cc=(--clock-check-json "$CLOCK_CHECK_JSON")
  local base=(--union-cache "$CACHE" --bset-list "$BSET" --lead-union "$LEAD_UNION" --lead-corpus "$LEAD_CORPUS" --eval-manifest "$EVAL_CACHE/eval/manifest.jsonl" "${ev[@]}" "${cc[@]}" --join-union "$JOIN_UNION" --expect-union "$EXPECT_UNION" --expect-corpus "$EXPECT_CORPUS"
          --hash "v9_union=$V9_UNION" --hash "speed_union=$SPEED_UNION" --hash "join_union=$JOIN_UNION")
  CUDA_VISIBLE_DEVICES=-1 run "$PY" "$PC" "${base[@]}" --out "$OUT/precheck/PRECHECK.json"
  local rc=$?
  [ "$DRY" = 1 ] || { say "precheck rc=$rc"; [ $rc -eq 0 ] || stop "precheck failed (rc $rc): read $OUT/precheck/PRECHECK.json" 1; }
  CUDA_VISIBLE_DEVICES=-1 run "$PY" "$PC" "${base[@]}" --plant --out "$OUT/precheck/PRECHECK_PLANTED.json"
  rc=$?
  [ "$DRY" = 1 ] || { say "planted precheck rc=$rc (must be 2)"; [ $rc -eq 2 ] || stop "the planted shared evaluation id did not make d1 fail (rc $rc)" 1; }
  local clk=(--union-cache "$CACHE" --bset-list "$BSET" --lead-union "$LEAD_UNION")
  CUDA_VISIBLE_DEVICES=-1 run "$PY" "$CK" "${clk[@]}" --out "$OUT/precheck/CLOCK.json"
  rc=$?
  [ "$DRY" = 1 ] || { say "clock check rc=$rc"; [ $rc -eq 0 ] || stop "clock check d2c failed (rc $rc): read $OUT/precheck/CLOCK.json" 1; }
  CUDA_VISIBLE_DEVICES=-1 run "$PY" "$CK" "${clk[@]}" --plant-shift 1 --out "$OUT/precheck/CLOCK_PLANTED.json"
  rc=$?
  [ "$DRY" = 1 ] || { say "planted clock check rc=$rc (must be 2)"; [ $rc -eq 2 ] || stop "the planted shift of one token frame did not make the clock check fail (rc $rc)" 1; }
  local evs=(--eval-cache "$EVAL_CACHE" --union-cache "$CACHE" --bset-list "$BSET" --study-eval-tsv "$KIT/STUDY_EVAL_MD5.tsv" --study-train-tsv "$KIT/STUDY_TRAIN_MD5.tsv")
  run "$PY" "$TS" "${evs[@]}" --out "$OUT/precheck/EVALSET.json"
  rc=$?
  [ "$DRY" = 1 ] || { say "evaluation-set check rc=$rc"; [ $rc -eq 0 ] || stop "the evaluation cache does not hold the study evaluation set in its order (d2e, rc $rc): read $OUT/precheck/EVALSET.json" 1; }
  run "$PY" "$TS" "${evs[@]}" --plant-swap --out "$OUT/precheck/EVALSET_PLANTED.json"
  rc=$?
  [ "$DRY" = 1 ] || { say "planted evaluation-set check rc=$rc (must be 2)"; [ $rc -eq 2 ] || stop "two swapped evaluation ids did not make d2e fail (rc $rc)" 1; }
  mark precheck
}

build_train() {  # steps arms out flags... -> the array CMD (the trainer argv stays ONE word)
  local steps=$1 arms=$2 out=$3; shift 3
  CMD=("$PY" "$TP" train --grid pool4x10 --input-norm ln --tree "$TREE" --out "$out" --arms "$arms" --steps "$steps" --w-clos 0.0 --eval-guard "${EVAL_GUARD_FILE:-EVAL_GUARD_FILE}" --trainer-argv "$TARGV" --device "$DEVICE" --threads "$THREADS" "$@")
}

eval_guard_file() {   # the evaluation ids used by control d1 in training: the eval cache manifest ids, written once
  EVAL_GUARD_FILE=$OUT/eval_ids_guard.txt
  if [ "$DRY" = 1 ]; then echo "DRY: write $EVAL_GUARD_FILE from $EVAL_CACHE/eval/manifest.jsonl (+ $EVAL_IDS)"; return 0; fi
  if [ ! -f "$EVAL_GUARD_FILE" ]; then
    "$PY" - "$EVAL_CACHE/eval/manifest.jsonl" "$EVAL_IDS" "$EVAL_GUARD_FILE" <<'PYEOF'
import json, os, sys
ids = [json.loads(l)["clip"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
if sys.argv[2] and os.path.exists(sys.argv[2]):
    ids += [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
open(sys.argv[3], "w", encoding="utf-8", newline="\n").write("\n".join(dict.fromkeys(ids)) + "\n")
PYEOF
  fi
}

st_cpucheck() {   # CPU only, no GPU lock: two steps on the union and two with the corpus list
  is_done cpucheck && { say "cpucheck done (marker)"; return 0; }
  is_done precheck || [ "$DRY" = 1 ] || stop "run the precheck stage first" 1
  need_mem
  eval_guard_file
  local f d rc dev=$DEVICE
  for f in U C; do
    d=$OUT/cpucheck_$f
    [ -d "$d" ] && [ "$DRY" != 1 ] && stop "folder $d exists without a marker: not overwritten" 1
    DEVICE=cpu
    if [ "$f" = C ]; then build_train 2 TK16_s0 "$d" --expect-steps 2 --clip-list "$BSET"; else build_train 2 TK16_s0 "$d" --expect-steps 2; fi
    DEVICE=$dev
    CUDA_VISIBLE_DEVICES=-1 run "${CMD[@]}"
    rc=$?
    [ "$DRY" = 1 ] && continue
    { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || stop "cpucheck $f exited $rc: the study tree did not train two steps on these inputs (read the output above)" 1
    [ -f "$d/TRAIN_DONE.json" ] || stop "cpucheck $f wrote no TRAIN_DONE.json" 1
    say "cpucheck $f ok (rc $rc; 2 = the train-log check is not met at two steps, as expected)"
  done
  mark cpucheck
}

st_smoke() {
  is_done smoke && { say "smoke done (marker)"; return 0; }
  is_done cpucheck || [ "$DRY" = 1 ] || stop "run the cpucheck stage first" 1
  need_mem
  eval_guard_file
  local d=$OUT/smoke
  [ -d "$d" ] && [ "$DRY" != 1 ] && stop "folder $d exists without a marker: not overwritten" 1
  build_train "$SMOKE_STEPS" "$(folder_arms runQ)" "$d"
  local t0=$SECONDS
  gpu smoke "${CMD[@]}"
  local rc=$?
  [ "$DRY" = 1 ] && { echo "DRY: read $d/TRAIN_DONE.json seconds; projection = seconds/$SMOKE_STEPS x $STEPS_U2 steps (the ONE training process), bar ${MAX_H} h"; return 0; }
  [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "smoke exited $rc" 1
  [ -f "$d/TRAIN_DONE.json" ] || stop "smoke wrote no TRAIN_DONE.json" 1
  local sec; sec=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['seconds'])" "$d/TRAIN_DONE.json")
  local proj; proj=$(awk -v s="$sec" -v n="$SMOKE_STEPS" -v a="$STEPS" -v b="$STEPS_U2" 'BEGIN{printf "%.2f", (s/n)*b/3600}')
  say "smoke: ${sec} s for $SMOKE_STEPS steps of 4 arms (wall $((SECONDS - t0)) s incl. data build); projected GPU hours of the ONE training process of $STEPS_U2 steps (four arms): ${proj} h (data build and extracts not included)"
  printf '%s\n' "{\"smoke_seconds\": $sec, \"smoke_steps\": $SMOKE_STEPS, \"projected_gpu_h_upper_bound\": $proj, \"bar_h\": $MAX_H}" > "$OUT/SMOKE_ESTIMATE.json"
  if awk -v p="$proj" -v m="$MAX_H" 'BEGIN{exit !(p>m)}'; then stop "ESTIMATE-ABOVE-0.9H: projected ${proj} h is above ${MAX_H} h; the run goes back to the PI" 4; fi
  mark smoke
}

st_run() {
  is_done smoke || [ "$DRY" = 1 ] || stop "run the smoke stage first (it measures the time and the 6 h bar)" 1
  eval_guard_file
  local f
  for f in $FOLDERS; do
    is_done "train_$f" && { say "$f done (marker)"; continue; }
    need_mem
    local d=$OUT/$f
    if [ "$DRY" != 1 ] && [ -d "$d" ]; then
      if [ -f "$d/TRAIN_DONE.json" ]; then mark "train_$f"; continue; fi
      stop "folder $d exists without TRAIN_DONE.json: not overwritten (move it aside, then restart this stage)" 1
    fi
    build_train "$(folder_steps "$f")" "$(folder_arms "$f")" "$d" --expect-steps "$(folder_steps "$f")" $(folder_flags "$f")
    CMD[1]="$TPCK"; CMD+=(--ck-steps 19164,28746); CMD=(timeout "$TRAIN_LIMIT_S" "${CMD[@]}")   # the saving tool, under the time limit (the wait for the lock is not counted)
    gpu "train $f" "${CMD[@]}"
    local rc=$?
    if [ "$DRY" != 1 ]; then
      [ $rc -eq 124 ] && stop "TRAIN-ABOVE-TIME-LIMIT: train $f ran past ${TRAIN_LIMIT_S} s and was stopped (rc 124); the run goes back to the PI" 1
      [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "train $f exited $rc" 1
      [ -f "$d/TRAIN_DONE.json" ] || stop "train $f wrote no TRAIN_DONE.json (rc $rc)" 1
      [ -f "$d/CK_RECORD.json" ] || stop "train $f wrote no CK_RECORD.json (rc $rc): the saving tool did not run to its end" 1
      say "train $f done rc=$rc (0 = p5 met on every arm, 2 = not; the reader applies p5)"
      mark "train_$f"
    fi
  done
}

st_extract() {
  local f a rc tag xsrc xdir
  for f in $FOLDERS; do
    is_done "train_$f" || [ "$DRY" = 1 ] || stop "train $f is not done" 1
    is_done "extract_$f" && { say "extract $f done (marker)"; continue; }
    need_mem
    local sd=$OUT/states/$f
    [ "$DRY" != 1 ] && [ -d "$sd" -o -d "${sd}_s19164" -o -d "${sd}_s28746" ] && stop "a states folder of $f ($sd, ${sd}_s19164 or ${sd}_s28746) exists without a marker: not overwritten" 1
    for a in $(folder_arms "$f" | tr ',' ' '); do
      for tag in last s19164 s28746; do
        if [ "$tag" = last ]; then xsrc=(--run-dir "$OUT/$f"); xdir=$sd; else xsrc=(--ckpt "$OUT/$f/$a/ckpt_step${tag#s}.pt"); xdir=${sd}_$tag; fi
        gpu "extract $f $a $tag" "$PY" "$TP" extract --grid pool4x10 --tree "$TREE" "${xsrc[@]}" --arm "$a" --cache "$EVAL_CACHE" --states-dir "$xdir" --splits eval --eval-range "$EVAL_RANGE" --device "$DEVICE" --threads "$THREADS"
        rc=$?
        [ "$DRY" = 1 ] || [ $rc -eq 0 ] || stop "extract $f $a $tag failed (rc $rc)" 1
      done
    done
    mark "extract_$f"
  done
}

st_checklog() {
  local f
  [ "$DRY" = 1 ] || mkdir -p "$OUT/checklog"
  for f in $FOLDERS; do
    if [ "$DRY" = 1 ]; then echo "DRY: CUDA_VISIBLE_DEVICES=-1 $PY $TP check-log --run-dir $OUT/$f > $OUT/checklog/$f.json"; continue; fi
    [ -d "$OUT/$f" ] || stop "no folder $OUT/$f" 1
    [ -f "$OUT/checklog/$f.json" ] && continue
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$TP" check-log --run-dir "$OUT/$f" > "$OUT/checklog/$f.json"
    say "check-log $f rc=$? (2 = p5 not met on an arm; the reader applies it)"
  done
  if [ "$DRY" != 1 ]; then
    { echo "DSCALE_2X_DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "tok_probe_u.py sha256: $(sha256sum "$TP" | cut -c1-64)"
      echo "tok_probe_u_ck.py sha256: $(sha256sum "$TPCK" | cut -c1-64)"
      echo "dscale_precheck.py sha256: $(sha256sum "$PC" | cut -c1-64)"
      echo "dscale_clockcheck.py sha256: $(sha256sum "$CK" | cut -c1-64)"
      echo "precheck: $OUT/precheck/PRECHECK.json  clock: $OUT/precheck/CLOCK.json  smoke: $OUT/SMOKE_ESTIMATE.json  training cache folder: $CACHE_T"
      for f in $FOLDERS; do for a in $(folder_arms "$f" | tr ',' ' '); do for c in ckpt_last ckpt_step19164 ckpt_step28746; do echo "ckpt $f/$a/$c $(sha256sum "$OUT/$f/$a/$c.pt" | cut -c1-64)"; done; done; done
      echo "train logs: $OUT/<folder>/<ARM>/train_log.jsonl  batch logs: $OUT/<folder>/r1_batches.json  states (eval only): $OUT/states/<folder>/"
    } > "$OUT/DSCALE_2X_DONE.txt"
    say "DONE: $OUT/DSCALE_2X_DONE.txt"
  fi
}

preflight
[ "$STAGE" = checklog ] || resolve_eval
[ "$STAGE" = checklog ] || resolve_cache
case "$STAGE" in
  precheck) st_precheck;;
  cpucheck) st_cpucheck;;
  smoke)    st_smoke;;
  run)      st_run;;
  extract)  st_extract;;
  checklog) st_checklog;;
esac
exit 0
