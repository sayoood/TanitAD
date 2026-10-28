#!/usr/bin/env bash
# run_speedsub_t1.sh -- the chain of PREREG_V8F_SPEEDSUB_T1 (section 9): fit -> pin -> smoke -> 12 decision rollouts -> reader -> 4 beside rollouts.
#
#   bash run_speedsub_t1.sh --dry-run          NO rollout, NO model: validates every input, pins into a fresh out_dry/<stamp>/, renders and PARSES (wrapper + pinned tool) the command line of every
#                                              rollout the chain would start, and shows that the reader refuses an empty manifest. Exit 0 iff all of it holds.
#   bash run_speedsub_t1.sh                    the full chain (dev box, CPU, one rollout at a time, OMP/MKL 8). Resumable: a rollout with a matching done record is skipped, an incomplete one is MOVED
#                                              ASIDE (<dir>.incomplete.<stamp>), never deleted.
#   bash run_speedsub_t1.sh --stage <name>     one stage: fit pin static smoke ctrl0 preflight ctrl1 rep0 rep1 read beside
#
# Stages and markers (stdout and $OUT/chain.log): ZZSST1-STAGE-DONE <stage> | ZZSST1-STOP <stage> <reason> | ZZSST1-CHAIN-DONE | ZZSST1-DRY-DONE.
#   fit        the read-out `mlp_ego_c` (+ C-FIT / C-CAUSAL / C-COVER / C-DISJ / C-PARITY / C-INSTR); skipped if fit_c.json says FITTED and the container exists
#   pin        pins.json (checkpoints against the PREREG prefixes, read-out, tool, wrapper, instrument, reader)
#   static     the reader's own check that the PREREG text is the pinned one
#   smoke      CTRL s0 identity + egoc at --max-windows 200, then the reader's --smoke-check (K-NONSPEED K-REC K-REACH K-SHA)           [the PREREG's registered smoke]
#   ctrl0      CTRL s0: identity, egoc, egoshufc
#   preflight  the reader's --preflight on CTRL s0 (K-ID K-REC K-REACH K-SHUF K-FLOOR K-WIN K-SHA K-NONSPEED): the chain STOPS here if any control fails
#   ctrl1 rep0 rep1   the other three runs, identity + egoc + egoshufc each
#   read       --make-manifest, then the reader on all four runs (REFUSED if a control fails)
#   beside     the logged lane, CTRL only, identity_l + egol at both seeds (4 rollouts), then the reader's --beside. Decides nothing.
#
# Environment (all optional): SST1_CODE (dir with the scripts), SST1_READER, SST1_PREREG, SST1_OUT, SST1_FIT, SST1_TREE, SST1_PYTHON, SST1_MIN_RAM_GB (4), SST1_MIN_DISK_GB (5).
shopt -u patsub_replacement 2>/dev/null || true          # `&` in "Architecture & Inference" paths
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYV="${SST1_PYTHON:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
CODE="${SST1_CODE:-$HERE/stack/scripts}"
READER="${SST1_READER:-$HERE/read/speedsub_t1_read.py}"
ARGCHECK="${SST1_ARGCHECK:-$HERE/chain_argcheck.py}"
PREREG="${SST1_PREREG:-F:/TanitAD-scratch/speedsub_t1/ref/PREREG_V8F_SPEEDSUB_T1.md}"
FIT="${SST1_FIT:-F:/TanitAD-scratch/speedsub_t1/fit}"
TREE="${SST1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}"
MIN_RAM_GB="${SST1_MIN_RAM_GB:-4}"
MIN_DISK_GB="${SST1_MIN_DISK_GB:-5}"
WRAP="$CODE/v8f_t1_speedsub.py"; PINPY="$CODE/v8f_speedsub_t1_pin.py"; FITPY="$CODE/v8f_speedsub_t1_fit.py"; INSTR="$CODE/v8f_mini_eval_speedread.py"
SMOKE_N=200
DRY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --stage) ONLY="$2"; shift ;;
    -h|--help) sed -n 2,26p "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 64 ;;
  esac
  shift
done
if [ "$DRY" = 1 ]; then
  OUT="${SST1_OUT:-F:/TanitAD-scratch/speedsub_t1/out_dry/$(date +%Y%m%d-%H%M%S)}"
else
  OUT="${SST1_OUT:-F:/TanitAD-scratch/speedsub_t1/out}"
fi
PINS="$OUT/pins.json"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$OUT/logs" || { echo "ZZSST1-STOP init cannot create $OUT"; exit 1; }
LOGF="$OUT/chain.log"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOGF"; }
stop() { say "ZZSST1-STOP $1 $2"; exit 1; }
stage_done() { say "ZZSST1-STAGE-DONE $1"; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
pin_get() { "$PYV" "$PINPY" get --pins "$PINS" --key "$1" | tr -d '\r'; }

free_ram_gb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" | tr -d '\r' | tr ',' '.'; }
guard() {   # RAM and disk, before every heavy step
  local r d
  r="$(free_ram_gb)"; d="$(df -Pk "$OUT" | awk 'NR==2{printf "%.1f", $4/1048576}')"
  awk -v v="${r:-0}" -v m="$MIN_RAM_GB" 'BEGIN{exit !(v+0>=m+0)}' || stop "$1" "free RAM ${r} GB < ${MIN_RAM_GB} GB"
  awk -v v="${d:-0}" -v m="$MIN_DISK_GB" 'BEGIN{exit !(v+0>=m+0)}' || stop "$1" "free disk ${d} GB < ${MIN_DISK_GB} GB on the output drive"
}

# ---- lock: one chain per output dir
LOCK="$OUT/.chain.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "ZZSST1-STOP init $LOCK exists: another chain runs on $OUT, or one died (remove the directory by hand after checking)"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say "run_speedsub_t1 start dry=$DRY stage=${ONLY:-all} out=$OUT code=$CODE reader=$READER"
for f in "$PYV" "$WRAP" "$PINPY" "$FITPY" "$INSTR" "$READER" "$ARGCHECK" "$PREREG" "$TREE/stack/scripts/v8f_t1_eval.py"; do
  [ -e "$f" ] || stop init "missing input: $f"
done

# ---- one rollout: run mode [smoke_n]   (dry run: only appends the command line to the plan file)
run_one() {
  local run="$1" mode="$2" smoke="${3:-0}" sm="" tag d sha ro rosha t0 rc out
  [ "$smoke" != 0 ] && sm="SMOKE/"
  tag="SST1_$([ "$smoke" != 0 ] && echo SMOKE_)${run}_${mode}"
  d="$OUT/${sm}${run}/${mode}"
  sha="$(pin_get "ckpts.${run}.sha256")" || stop "$tag" "no pinned checkpoint sha for $run"
  if "$PYV" "$PINPY" check-done --done "$d/done_${tag}.txt" --tag "$tag" --ckpt-sha "$sha"; then say "skip $tag (done record matches)"; return 0; fi
  local smoke_args=(); [ "$smoke" != 0 ] && smoke_args=(--smoke "$smoke")
  out="$("$PYV" "$PINPY" render-argv --run "$run" --mode "$mode" --out-root "$OUT" "${smoke_args[@]}" | tr -d '\r')" || stop "$tag" "render-argv refused"
  local TOOLARGV=(); mapfile -t TOOLARGV <<< "$out"
  local wargs=(--mode "$mode" --expect-ckpt-sha "$sha" --tool-root "$TREE" --tool-sha "$(pin_get tool_sha256)" --instr "$INSTR" --instr-sha "$(pin_get instrument_sha256)")
  case "$mode" in
    egoc|egoshufc) wargs+=(--readout "$(pin_get readout_c.path)" --readout-sha "$(pin_get readout_c.sha256)") ;;
    egol)          wargs+=(--readout "$(pin_get readout_banked.path)" --readout-sha "$(pin_get readout_banked.sha256)") ;;
  esac
  [ "$mode" = egoshufc ] && wargs+=(--shuffle-source "$OUT/${sm}${run}/egoc/ticks/speedsub_SST1_$([ "$smoke" != 0 ] && echo SMOKE_)${run}_egoc.npz")
  if [ "$DRY" = 1 ]; then
    # the shuffle source does not exist yet in a dry run: the parse check only needs the flag to be present
    { printf '%s
' "${wargs[@]}" "${TOOLARGV[@]}" "---"; } >> "$OUT/logs/plan.txt"
    say "DRY plan $tag -> $d"
    return 0
  fi
  [ -e "$d" ] && { mv "$d" "$d.incomplete.$(date +%Y%m%d-%H%M%S)" && say "moved an incomplete $d aside"; }
  guard "$tag"
  say "START $tag"
  t0=$(date +%s)
  "$PYV" "$WRAP" "${wargs[@]}" "${TOOLARGV[@]}" > "$OUT/logs/${tag}.log" 2>&1; rc=$?
  [ $rc = 0 ] || stop "$tag" "the rollout exited $rc (log: $OUT/logs/${tag}.log)"
  "$PYV" "$PINPY" write-done --run "$run" --mode "$mode" --out-root "$OUT" "${smoke_args[@]}" --sha-before "$sha" --wall-s "$(( $(date +%s) - t0 ))" | tee -a "$LOGF"
  [ "${PIPESTATUS[0]}" = 0 ] || stop "$tag" "write-done refused: the rollout's files are incomplete"
  say "DONE  $tag in $(( $(date +%s) - t0 )) s"
}

trio() { local run="$1"; run_one "$run" identity; run_one "$run" egoc; run_one "$run" egoshufc; }

# ---- stages
DRY_FAIL=0
if want fit; then
  if grep -q '"outcome": "FITTED"' "$FIT/fit_c.json" 2>/dev/null && [ -f "$FIT/readout_c.pt" ]; then say "fit: reusing $FIT (fit_c.json FITTED)"
  elif [ "$DRY" = 1 ]; then stop fit "no fitted read-out in $FIT: run the fit stage first (a dry run does not fit)"
  else
    guard fit
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYV" "$FITPY" fit --out-dir "$FIT" --instr "$INSTR" > "$OUT/logs/fit.log" 2>&1 || stop fit "the fit exited non-zero (log: $OUT/logs/fit.log)"
    grep -q '"outcome": "FITTED"' "$FIT/fit_c.json" || stop fit "fit_c.json does not say FITTED"
  fi
  stage_done fit
fi
if want pin; then
  "$PYV" "$PINPY" pin --pins "$PINS" --code-dir "$CODE" --reader "$READER" --fit-dir "$FIT" | tee -a "$LOGF"
  [ "${PIPESTATUS[0]}" = 0 ] || stop pin "the pin step refused or the existing pins differ"
  stage_done pin
fi
[ -f "$PINS" ] || stop init "no $PINS: run the pin stage first"
if want static; then
  "$PYV" "$READER" --static-check --prereg "$PREREG" | tail -1 | tee -a "$LOGF" | grep -q "STATIC-OK" || stop static "the PREREG text is not the pinned one"
  stage_done static
fi
if want smoke; then
  run_one CTRL_s0 identity "$SMOKE_N"; run_one CTRL_s0 egoc "$SMOKE_N"
  if [ "$DRY" = 0 ]; then
    "$PYV" "$READER" --smoke-check --pins "$PINS" --out-root "$OUT" --run CTRL_s0 --instr "$INSTR" --out "$OUT/smoke_check_CTRL_s0.json" | tee -a "$LOGF"
    [ "${PIPESTATUS[0]}" = 0 ] || stop smoke "a smoke control failed (see $OUT/smoke_check_CTRL_s0.json)"
  fi
  stage_done smoke
fi
if want ctrl0; then trio CTRL_s0; stage_done ctrl0; fi
if want preflight; then
  if [ "$DRY" = 0 ]; then
    "$PYV" "$READER" --preflight --pins "$PINS" --out-root "$OUT" --run CTRL_s0 --instr "$INSTR" --out "$OUT/preflight_CTRL_s0.json" | tee -a "$LOGF"
    [ "${PIPESTATUS[0]}" = 0 ] || stop preflight "a preflight control failed (see $OUT/preflight_CTRL_s0.json)"
  fi
  stage_done preflight
fi
if want ctrl1; then trio CTRL_s1; stage_done ctrl1; fi
if want rep0; then trio REP_s0; stage_done rep0; fi
if want rep1; then trio REP_s1; stage_done rep1; fi
if want read; then
  "$PYV" "$READER" --make-manifest --pins "$PINS" --out-root "$OUT" --out "$OUT/runs_manifest.json" | tee -a "$LOGF"
  if [ "$DRY" = 1 ]; then
    "$PYV" "$READER" --manifest "$OUT/runs_manifest.json" --prereg "$PREREG" --instr "$INSTR" --work-dir "$OUT/read_work" --out "$OUT/SPEEDSUB_T1_READ.json" > "$OUT/logs/dry_read.out" 2>&1; rc=$?
    [ $rc = 4 ] && [ ! -e "$OUT/SPEEDSUB_T1_READ.json" ] && say "DRY ok   the reader refuses a manifest with no rollouts (rc 4, nothing decisive written)" || { say "DRY FAIL the reader returned rc $rc on an empty manifest"; DRY_FAIL=1; }
  else
    guard read
    "$PYV" "$READER" --manifest "$OUT/runs_manifest.json" --prereg "$PREREG" --instr "$INSTR" --work-dir "$OUT/read_work" --out "$OUT/SPEEDSUB_T1_READ.json" | tee -a "$LOGF"
    rc="${PIPESTATUS[0]}"
    [ "$rc" = 0 ] || stop read "the reader returned rc $rc (2 = REFUSED on a control or the PREREG pin, 4 = a rollout is missing); see $OUT/SPEEDSUB_T1_READ.json"
  fi
  stage_done read
fi
if want beside; then
  for r in CTRL_s0 CTRL_s1; do run_one "$r" identity_l; run_one "$r" egol; done
  if [ "$DRY" = 0 ]; then
    "$PYV" "$READER" --beside --manifest "$OUT/runs_manifest.json" --instr "$INSTR" --work-dir "$OUT/read_work" --out "$OUT/SPEEDSUB_T1_BESIDE.json" | tee -a "$LOGF"
    [ "${PIPESTATUS[0]}" = 0 ] || stop beside "the beside reader failed"
  fi
  stage_done beside
fi

if [ "$DRY" = 1 ]; then
  # ONE python process parses every planned command line with the wrapper's parser and the pinned tool's parser and cross-checks the pins against the files named (checkpoint, read-out, instrument)
  "$PYV" "$ARGCHECK" --wrapper "$WRAP" --plan "$OUT/logs/plan.txt" | tee -a "$LOGF" | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || DRY_FAIL=1
  [ "$DRY_FAIL" = 0 ] || stop dry "at least one command line failed its parse check (or the reader did not refuse an empty manifest)"
  say "ZZSST1-DRY-DONE  (no rollout was started; pins and manifest are in $OUT)"
else
  say "ZZSST1-CHAIN-DONE  result: $OUT/SPEEDSUB_T1_READ.json  beside: $OUT/SPEEDSUB_T1_BESIDE.json"
fi
