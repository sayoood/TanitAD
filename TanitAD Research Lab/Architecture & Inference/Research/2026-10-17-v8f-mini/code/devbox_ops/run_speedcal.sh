#!/bin/bash
# run_speedcal.sh -- PREREG_V8F_SPEEDCAL on the DEV BOX (CPU, zero training): is T's speed error a head bias? A map per speed cell, fitted on TRAINING windows, scored on eval-141.
#
#   bash run_speedcal.sh [--dry-run] [stage]    stage = check | fit0 | fit1 | evals0 | evals1 | evalsB | pairfam | read | bank | all   (default: all)
#
#   check    inputs, shas, disk                                                                                                -> $OUT/check.json
#   fit0/1   `stack/scripts/v8f_mini_eval_speedcal.py fit`: X_s's T forward on HBLEND's 141 training clips (9,582 windows, checked equal to HBLEND's list), closed-form masked least squares per speed cell, and the
#            permuted-target map of control 4                                                                                     -> $OUT/fit/fit_*.json, fit_cells_*.npz   (~1 min each on 8 CPU threads)
#   evals0/1 `... evals`: the six modes of X_s (own, identity, gt, cal, perm, ceiling), resumable (a finished mode is skipped)   -> $OUT/eval/                                (~40 s per mode)
#   evalsB   the baseline B = M2b_s<s> (own, cal with X_s's map): DESCRIPTIVE, skipped when B's checkpoint is absent
#   pairfam  `stack/scripts/v8f_paired_families.py` (cal - own, per seed): the paired TACTICAL / STRATEGIC families        -> $OUT/pairfam/
#   read     `v8f_speedcal_read.py --check-inputs`, then the read (controls 1-4 FIRST; PRIMARY / CEILING / CAPTURE / SECONDARY / outcome; beside)   -> $OUT/READ.json, READ.md, READ_controls.json
#   bank     copy the run's small outputs into $SPEEDCAL_BANK (the deliverable tree's raw/2026-10-25-speedcal/run), when set
#   all      check, fit0, fit1, evals0, evals1, evalsB, pairfam, read, bank, then ONE marker line:  ZZSPEEDCAL-DONE <OUTCOME>   or   ZZSPEEDCAL-STOP <reason>
#
# ~25-30 min wall. LAUNCH DETACHED (a Claude session's child dies with the session): WMI `Win32_Process.Create` (CreateFlags 0x08000200) with `cmd /c "... < NUL > $OUT/logs/run.out 2>&1"`
# (the memory note `devbox-drive-letter-and-detached-jobs`); poll `$OUT/logs/run.out` for the ZZSPEEDCAL- marker with a script, never by model turns.
# CPU ONLY: CUDA_VISIBLE_DEVICES=-1, torch threads SPEEDCAL_THREADS (default 8), BLAS 1. Never run it while another torch job is pinning the CPU (the threads stall each other).
# It writes ONLY under $OUT (a scratch dir) and, in `bank`, $SPEEDCAL_BANK. It READS box2_ship/logs (the two checkpoints, KINDX's seed-1 evals) and the scratchpad inputs; it never touches box 2.
#
# Inputs (each overridable; the defaults are the layout of this session's scratchpad and of the box2_ship logs):
#   SPEEDCAL_X0 / SPEEDCAL_X1        X_s0 = M2b-mixo_s0, X_s1 = M2b-mixo_s1 (default: box2_ship copies; a missing one STOPs, nothing is fetched)
#   SPEEDCAL_X0_SHA / _X1_SHA        their sha256 (default: X_s0's committed c6a62876...; X_s1's from $KX1DIR/ckpt_sha256.json key M2b-mixo_s1_trained)
#   SPEEDCAL_B0 / _B1, _B0_SHA/_B1_SHA   the baseline M2b_s0 / M2b_s1 (the sfresh stand-ins) and their sha256 (601cb435... / 6104fb15...)
#   SPEEDCAL_HB                      HBLEND's fit set: fit_clips.json, cache/train/clips/, box2/join_train.npz, fit_windows_M2b_s<s>.npz
#   SPEEDCAL_SF                      the eval-141 inputs (cache, v9 release, lead, speed sidecar, P10 prior, native tracks)
#   SPEEDCAL_KINDX_S0 / _S1          the KINDX evals (own, only-speed) of each seed
#   SPEEDCAL_BANKED_S0 / _S1         the banked STANDARD eval npz (optional second reference of the own-reproduction guard)
#   SPEEDCAL_OUT, SPEEDCAL_BANK, SPEEDCAL_PY, SPEEDCAL_THREADS
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd "$HERE/.." && pwd)
REPO=$(cd "$CODE/../../../../.." && pwd)
if command -v cygpath >/dev/null 2>&1; then REPO=$(cygpath -m "$REPO"); CODE=$(cygpath -m "$CODE"); fi       # C:/... for the native python (Git-bash would hand it /tmp/... otherwise)
PK="$REPO/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
STAGE=${1:-all}

PY=${SPEEDCAL_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
SC=${SPEEDCAL_SC:-C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad}
SF=${SPEEDCAL_SF:-$SC/sfresh}
HB=${SPEEDCAL_HB:-$SC/hblend}
OUT=${SPEEDCAL_OUT:-$SC/speedcal_run}
BANK=${SPEEDCAL_BANK:-}
B2=${SPEEDCAL_B2:-C:/Users/Admin/box2_ship}
KX1DIR=${SPEEDCAL_KX1DIR:-$B2/logs/v8f_kindx_s1}
X0=${SPEEDCAL_X0:-$B2/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt}
X1=${SPEEDCAL_X1:-$KX1DIR/runs/M2b_s1_mixo/ckpt_last.pt}
X0_SHA=${SPEEDCAL_X0_SHA:-c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d}
B0=${SPEEDCAL_B0:-$SF/inputs/M2b_s0/ckpt_last.pt}
B1=${SPEEDCAL_B1:-$SF/inputs/M2b_s1/ckpt_last.pt}
B0_SHA=${SPEEDCAL_B0_SHA:-601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8}
B1_SHA=${SPEEDCAL_B1_SHA:-6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6}
KX0=${SPEEDCAL_KINDX_S0:-$PK/raw/2026-10-25-kindx/eval}
KX1=${SPEEDCAL_KINDX_S1:-$KX1DIR/eval}
BK0=${SPEEDCAL_BANKED_S0:-$B2/logs/v8f_mixo/eval/eval_M2b-mixo_s0.npz}
BK1=${SPEEDCAL_BANKED_S1:-$KX1DIR/eval/eval_M2b-mixo_s1.npz}
# the paired-families tool's SUBSTITUTE control (coordinator, 2026-10-25): our `own`-mode npz against the banked X_s npz, BOTH read against the banked eval json (which carries the reference families block)
PFB0=${SPEEDCAL_PFBANK_S0:-$B2/logs/v8f_mixo/eval/eval_M2b-mixo_s0}                # .npz and .json
PFB1=${SPEEDCAL_PFBANK_S1:-$KX1DIR/eval/eval_M2b-mixo_s1-kindx-own}
DRV="$REPO/stack/scripts/v8f_mini_eval_speedcal.py"
PFT="$REPO/stack/scripts/v8f_paired_families.py"
RDR="$CODE/v8f_speedcal_read.py"
LEAD_BLOCK=${SPEEDCAL_LEAD_BLOCK:-}               # default: the reader's own (the repo's b1_eval_lead_block.npz)
THREADS=${SPEEDCAL_THREADS:-8}
export CUDA_VISIBLE_DEVICES=-1 SPEEDCAL_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
MIN_FREE_KB=4194304                               # abort if C: has < 4 GB free

say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
have() {   # have <path>: "present" / "absent"; a DRY RUN does not stat anything under box2_ship (the other session's tree): it says so instead
  if [ "$DRY" = 1 ] && [[ "$1" == "$B2"* ]]; then echo "not probed in a dry run (under box2_ship)"; elif [ -e "$1" ]; then echo present; else echo absent; fi
}
stop() { say "ZZSPEEDCAL-STOP $*"; [ "$DRY" = 1 ] || bank_files > /dev/null 2>&1; exit 3; }          # a STOP still banks what exists: a failed control is evidence
x1sha() {   # X_s1's sha256 as measured on box 2 after training: the env value, else the KINDX chain's json
  if [ -n "${SPEEDCAL_X1_SHA:-}" ]; then echo "$SPEEDCAL_X1_SHA"; return; fi
  [ -f "$KX1DIR/ckpt_sha256.json" ] && $PY -I -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('M2b-mixo_s1_trained',''))" "$KX1DIR/ckpt_sha256.json"
}
sha_of() { $PY -I "$DRV" sha "$1"; }
ck_of()  { if [ "$1" = 0 ]; then echo "$X0"; else echo "$X1"; fi; }
sha_for() { if [ "$1" = 0 ]; then echo "$X0_SHA"; else x1sha; fi; }

# ---- the data arguments the driver needs (KINDX's flags with this session's paths)
DATA=(--eval-dir "$OUT/eval" --eval-cache "$SF/cache" --tracks "$SF/inputs/native_tracks.pt" --v9-eval "$SF/inputs/v9/v9_labels_eval139.npz" --lead-eval "$SF/inputs/v9/lead_eval139.npz"
      --speed-eval "$SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl" --p10 "$SF/inputs/p10/p10_prior.pt")

stage_check() {
  say "== check"
  local free; free=$(df -k /c | tail -1 | awk '{print $4}')
  if [ "$DRY" = 1 ]; then echo "  C: free ${free} KB (a real run needs >= ${MIN_FREE_KB} KB and stops otherwise)"
  else [ "$free" -ge "$MIN_FREE_KB" ] || stop "C: has less than 4 GB free (${free} KB)"; fi
  local d; for d in "$DRV" "$PFT" "$RDR" "$SF/cache" "$SF/inputs/native_tracks.pt" "$SF/inputs/p10/p10_prior.pt" "$SF/inputs/v9/lead_eval139.npz" "$SF/inputs/v9/v9_labels_eval139.npz" \
                    "$HB/fit_clips.json" "$HB/cache/train/clips" "$HB/box2/join_train.npz" "$HB/fit_windows_M2b_s0.npz" "$HB/fit_windows_M2b_s1.npz" \
                    "$KX0/eval_M2b-mixo_s0-kindx-own.npz" "$KX0/eval_M2b-mixo_s0-kindx-only-speed.npz" "$X0" "$X1" \
                    "${LEAD_BLOCK:-$REPO/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz}"; do
    [ "$DRY" = 1 ] && { echo "  $d: $(have "$d")"; continue; }
    [ -e "$d" ] || stop "missing input: $d"
  done
  if [ "$DRY" = 1 ]; then
    echo "  KINDX s1 evals (own, only-speed) in $KX1: $(have "$KX1/eval_M2b-mixo_s1-kindx-own.npz")"
    echo "  X_s0 sha256 expected $X0_SHA; X_s1 expected ${SPEEDCAL_X1_SHA:-the key M2b-mixo_s1_trained of $KX1DIR/ckpt_sha256.json}"
    echo "  baseline B (descriptive, skipped when absent): $B0: $(have "$B0"); $B1: $(have "$B1")"
    return 0
  fi
  { [ -e "$KX1/eval_M2b-mixo_s1-kindx-own.npz" ] && [ -e "$KX1/eval_M2b-mixo_s1-kindx-only-speed.npz" ]; } || stop "KINDX's seed-1 evals (own, only-speed) are not in $KX1"
  local X1S; X1S=$(x1sha)
  [ -n "$X1S" ] || stop "no sha256 for X_s1 (set SPEEDCAL_X1_SHA or provide $KX1DIR/ckpt_sha256.json)"
  mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/fit" "$OUT/pairfam"
  local g0 g1; g0=$(sha_of "$X0"); g1=$(sha_of "$X1")
  [ "$g0" = "$X0_SHA" ] || stop "sha256 of $X0 is $g0, expected $X0_SHA"
  [ "$g1" = "$X1S" ] || stop "sha256 of $X1 is $g1, expected $X1S"
  [ "$g0" != "$g1" ] || stop "X_s0 and X_s1 are the same file: no second seed"
  $PY -I -c "import json,sys;json.dump({'x0':sys.argv[1],'x1':sys.argv[2],'x0_sha':sys.argv[3],'x1_sha':sys.argv[4],'kindx_s0':sys.argv[5],'kindx_s1':sys.argv[6],'hblend':sys.argv[7],'driver_sha256':sys.argv[8],'reader_sha256':sys.argv[9]},open(sys.argv[10],'w'),indent=1)" \
     "$X0" "$X1" "$X0_SHA" "$X1S" "$KX0" "$KX1" "$HB" "$(sha_of "$DRV")" "$(sha_of "$RDR")" "$OUT/check.json"
  say "CHECK OK (X_s0 $g0, X_s1 $g1)"
}

stage_fit() {   # stage_fit <seed>
  local s=$1
  say "== fit seed $s"
  local ck sh; ck=$(ck_of "$s"); sh=$(sha_for "$s")
  local args=(fit --seed "$s" --x "$ck" --expect-sha "${sh:-<X${s}_SHA>}" --fit-clips "$HB/fit_clips.json" --fit-cache "$HB/cache" --fit-join "$HB/box2/join_train.npz"
              --tracks "$SF/inputs/native_tracks.pt" --hblend-windows "$HB/fit_windows_M2b_s$s.npz" --out-dir "$OUT/fit")
  if [ "$DRY" = 1 ]; then echo "  [dry-run] $PY -I $DRV ${args[*]}"; $PY -I "$DRV" "${args[@]}" --dry-run | sed 's/^/    /'; return 0; fi
  mkdir -p "$OUT/logs" "$OUT/fit"
  $PY -I "$DRV" "${args[@]}" > "$OUT/logs/fit$s.log" 2>&1 || { tail -5 "$OUT/logs/fit$s.log"; stop "fit seed $s failed (see $OUT/logs/fit$s.log)"; }
  tail -1 "$OUT/logs/fit$s.log"
}

stage_evals() {   # stage_evals <seed>
  local s=$1
  say "== evals seed $s (own, identity, gt, cal, perm, ceiling)"
  local ck sh; ck=$(ck_of "$s"); sh=$(sha_for "$s")
  local args=(evals --seed "$s" --x "$ck" --expect-sha "${sh:-<X${s}_SHA>}" --fit-json "$OUT/fit/fit_M2b-mixo_s$s-speedcal-fit.json" "${DATA[@]}")
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] $PY -I $DRV ${args[*]}"
    echo "  the driver's own dry-run (the v8f_mini_eval.py argv of each mode):"
    $PY -I "$DRV" "${args[@]}" --dry-run | sed 's/^/    /'
    return 0
  fi
  mkdir -p "$OUT/logs" "$OUT/eval"
  $PY -I "$DRV" "${args[@]}" > "$OUT/logs/evals$s.log" 2>&1 || { tail -5 "$OUT/logs/evals$s.log"; stop "evals seed $s failed (see $OUT/logs/evals$s.log)"; }
  grep -E "^\[(ok|skip)\]|SPEEDCAL evals done" "$OUT/logs/evals$s.log"
}

stage_evalsB() {   # DESCRIPTIVE: X_s's fitted map applied to the baseline B = M2b_s<s>; a missing B checkpoint skips it (it decides nothing)
  local s b bsha
  for s in 0 1; do
    if [ "$s" = 0 ]; then b=$B0; bsha=$B0_SHA; else b=$B1; bsha=$B1_SHA; fi
    say "== evalsB seed $s (baseline own, cal)"
    local args=(evals --seed "$s" --target b --x "$b" --expect-sha "$bsha" --modes own,cal --fit-json "$OUT/fit/fit_M2b-mixo_s$s-speedcal-fit.json" "${DATA[@]}" --eval-dir "$OUT/evalB")
    if [ "$DRY" = 1 ]; then echo "  [dry-run] $PY -I $DRV ${args[*]}   (baseline $b: $(have "$b"))"; continue; fi
    if [ ! -e "$b" ]; then say "baseline $b is absent: skipped (descriptive)"; continue; fi
    mkdir -p "$OUT/logs" "$OUT/evalB"
    $PY -I "$DRV" "${args[@]}" > "$OUT/logs/evalsB$s.log" 2>&1 || { say "baseline evals seed $s failed (descriptive; see $OUT/logs/evalsB$s.log): continuing"; continue; }
    grep -E "^\[(ok|skip)\]|SPEEDCAL evals done" "$OUT/logs/evalsB$s.log"
  done
}

stage_pairfam() {
  local s
  for s in 0 1; do
    say "== pairfam seed $s (SPEEDCAL - X_s, TACTICAL / STRATEGIC)"
    local args=(--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-cal.npz" --b "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz" --out "$OUT/pairfam/pairfam_speedcal_s$s.json" --md "$OUT/pairfam/pairfam_speedcal_s$s.md"
                --label-a SPEEDCAL --label-b "X_s$s")
    local pb; if [ "$s" = 0 ]; then pb=$PFB0; else pb=$PFB1; fi
    if [ "$DRY" = 1 ]; then
      echo "  [dry-run] $PY -I $PFT ${args[*]}"
      echo "  [dry-run] substitute control: $PY -I $PFT --a $OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz --a-json $pb.json --b $pb.npz --b-json $pb.json --out $OUT/pairfam/subctl_own_vs_banked_s$s.json   (banked json: $(have "$pb.json"))"
      continue
    fi
    mkdir -p "$OUT/pairfam" "$OUT/logs"
    $PY -I "$PFT" "${args[@]}" > "$OUT/pairfam/pairfam_s$s.out" 2>&1; local rc=$?
    say "pairfam seed $s rc=$rc (0 ok; 3 = one of ITS controls failed, the json is still written and the reader reports it; 2 = refused)"
    # the SUBSTITUTE control: our `own`-mode npz vs the banked X_s npz, both against the banked json (PASS = controls_ok, 119/119 values reproduced, paired max |delta| <= 1e-6)
    local pb; if [ "$s" = 0 ]; then pb=$PFB0; else pb=$PFB1; fi
    local sargs=(--a "$OUT/eval/eval_M2b-mixo_s$s-speedcal-own.npz" --a-json "$pb.json" --b "$pb.npz" --b-json "$pb.json" --out "$OUT/pairfam/subctl_own_vs_banked_s$s.json"
                 --md "$OUT/pairfam/subctl_own_vs_banked_s$s.md" --label-a own_via_speedcal_driver --label-b "banked_X_s$s")
    $PY -I "$PFT" "${sargs[@]}" > "$OUT/pairfam/subctl_s$s.out" 2>&1; rc=$?
    say "pairfam substitute control seed $s rc=$rc"
  done
}

stage_read() {
  say "== read"
  local X1S; X1S=$(x1sha)
  local args=(--eval-dir "$OUT/eval" --fit-dir "$OUT/fit" --kindx-dir-s0 "$KX0" --kindx-dir-s1 "$KX1" --expect-sha0 "$X0_SHA" --expect-sha1 "${X1S:-<X1_SHA>}" --out "$OUT/READ.json"
              --pairfam-dir "$OUT/pairfam" --b-eval-dir "$OUT/evalB" --expect-sha-b0 "$B0_SHA" --expect-sha-b1 "$B1_SHA")
  [ -e "$BK0" ] && args+=(--banked-own-s0 "$BK0")
  [ -e "$BK1" ] && args+=(--banked-own-s1 "$BK1")
  [ -n "$LEAD_BLOCK" ] && args+=(--lead-block "$LEAD_BLOCK")
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] $PY -I $RDR ${args[*]} --check-inputs   (then the read without --check-inputs)"
    return 0
  fi
  $PY -I "$RDR" "${args[@]}" --check-inputs > "$OUT/logs/read_check.log" 2>&1 || { tail -5 "$OUT/logs/read_check.log"; stop "reader --check-inputs refused (see $OUT/logs/read_check.log)"; }
  $PY -I "$RDR" "${args[@]}" > "$OUT/logs/read.log" 2>&1; local rc=$?
  tail -4 "$OUT/logs/read.log"
  [ $rc -eq 0 ] || stop "the reader refused or failed rc=$rc (a failed CONTROL reads no outcome: see $OUT/READ_controls.json and $OUT/logs/read.log)"
  local oc; oc=$($PY -I -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['OUTCOME'])" "$OUT/READ.json")
  echo "$oc" > "$OUT/OUTCOME.txt"
  say "read OK: $oc"
}

bank_files() {   # copy everything under $OUT to $BANK (no stop: also called from stop())
  [ -n "$BANK" ] && [ -d "$OUT" ] || return 0
  $PY -I - "$OUT" "$BANK" <<'PYEOF'
import os, shutil, sys
def lp(p):   # extended-length path: the bank sits under a deep tree and Windows' MAX_PATH (260) fails the copy otherwise
    p = os.path.abspath(p)
    return ("\\\\?\\" + p) if os.name == "nt" and not p.startswith("\\\\?\\") else p
src, dst = sys.argv[1], sys.argv[2]
n = 0
for root, _dirs, files in os.walk(src):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), src)
        out = lp(os.path.join(dst, rel))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(os.path.join(root, f), out)
        n += 1
print("banked", n, "files ->", dst)
PYEOF
}

stage_bank() {
  say "== bank"
  if [ "$DRY" = 1 ]; then echo "  [dry-run] copy everything under $OUT (READ.json READ.md READ_controls.json check.json OUTCOME.txt, fit/, eval/, evalB/, pairfam/, logs/) -> ${BANK:-<unset: skipped>}"; return 0; fi
  [ -n "$BANK" ] || { say "SPEEDCAL_BANK is unset: nothing banked (the outputs stay in $OUT)"; return 0; }
  bank_files || stop "bank failed"
}

if [ "$DRY" = 1 ]; then
  echo "== run_speedcal.sh --dry-run (nothing is read, written or launched; no torch is imported)"
  echo "repo $REPO"; echo "out  $OUT"; echo "bank ${BANK:-<unset>}"; echo "threads $THREADS (CPU only)"
fi
case "$STAGE" in
  check)   stage_check ;;
  fit0)    stage_fit 0 ;;
  fit1)    stage_fit 1 ;;
  evals0)  stage_evals 0 ;;
  evals1)  stage_evals 1 ;;
  evalsB)  stage_evalsB ;;
  pairfam) stage_pairfam ;;
  read)    stage_read ;;
  bank)    stage_bank ;;
  all)     stage_check; stage_fit 0; stage_fit 1; stage_evals 0; stage_evals 1; stage_evalsB; stage_pairfam; stage_read
           if [ "$DRY" = 1 ]; then stage_bank; echo "== dry-run done: the stages above are what 'all' runs, in this order; the final marker is ZZSPEEDCAL-DONE <OUTCOME> or ZZSPEEDCAL-STOP <reason>"
           else stage_bank; say "ZZSPEEDCAL-DONE $(cat "$OUT/OUTCOME.txt")"; fi ;;
  *) echo "unknown stage $STAGE (check | fit0 | fit1 | evals0 | evals1 | evalsB | pairfam | read | bank | all)"; exit 2 ;;
esac
exit 0
