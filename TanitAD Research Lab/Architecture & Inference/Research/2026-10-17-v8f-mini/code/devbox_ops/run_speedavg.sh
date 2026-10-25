#!/bin/bash
# run_speedavg.sh -- PREREG_V8F_SPEEDAVG on the DEV BOX (CPU, zero training). CONDITIONAL: run it only if KINDX's seed-1 read makes SPEED a carrier
# (gain_speed(X_s1) >= 2F = 0.2415 m and separated; `READ_s1.json` OUTCOME CARRIER with `speed` in `carriers`). Refuses to run otherwise.
#
#   bash run_speedavg.sh [--dry-run] [stage]        stage = check | fetch | evals0 | evals1 | read | all   (default: all)
#
#   check   inputs, shas, disk, the KINDX condition (READ_s1.json), the KINDX evals the controls read      -> $OUT/check.json
#   fetch   an X checkpoint absent locally is pulled READ-ONLY from box 2 (scp -o BatchMode=yes), then sha256-checked; box 2 is otherwise untouched
#   evals0  the four modes (own, self, gt, avg) of X_s0  } `stack/scripts/v8f_mini_eval_speedavg.py evals`, resumable (a finished mode is skipped); ~5-7 min each on 8 CPU threads
#   evals1  the four modes of X_s1                       }
#   read    `v8f_speedavg_read.py --check-inputs`, then the read (controls first; PRIMARY / CAPTURE / SECONDARY / outcome; beside)  -> $OUT/READ.json, READ.md
#   all     check, fetch, evals0, evals1, read, then ONE marker line:  ZZSPEEDAVG-DONE <OUTCOME>   or   ZZSPEEDAVG-STOP <reason>
#
# Total ~15-20 min wall. LAUNCH DETACHED (a Claude session's child dies with the session): WMI `Win32_Process.Create` with `cmd /c "... < NUL > $OUT/logs/run.out 2>&1"`
# (the memory note `devbox-drive-letter-and-detached-jobs`); poll `$OUT/logs/run.out` for the ZZSPEEDAVG- marker with a script, never by model turns.
# CPU ONLY: CUDA_VISIBLE_DEVICES=-1, torch threads SPEEDAVG_THREADS (default 8), BLAS 1. Never run it while another torch job is pinning the CPU (the threads stall each other).
# It writes ONLY under $OUT (a scratch dir). It reads box2_ship/ and the sfresh inputs read-only; it touches box 2 only for the optional read-only scp/ssh hash of a missing checkpoint.
#
# Inputs (each overridable; the defaults are the layout of the KINDX seed-1 chain and of this session's scratchpad):
#   SPEEDAVG_X0 / SPEEDAVG_X1          the checkpoints X_s0 = M2b-mixo_s0, X_s1 = M2b-mixo_s1 (default: box2_ship copies; absent -> fetch from box 2)
#   SPEEDAVG_X1_SHA                    X_s1's sha256 as MEASURED on box 2 after training (default: $KX1DIR/ckpt_sha256.json key M2b-mixo_s1_trained). X_s0's is the committed c6a62876...
#   SPEEDAVG_KINDX_S0 / _S1            the KINDX evals (own, only-speed) of each seed (s0: the repo's raw/2026-10-25-kindx/eval; s1: the chain's pulled eval dir)
#   SPEEDAVG_BANKED_S0 / _S1           the banked STANDARD eval npz (control 3's second reference; optional)
#   SPEEDAVG_SF                        the eval-141 inputs (cache, v9 release, lead, speed sidecar, P10 prior, native tracks)
#   SPEEDAVG_OUT                       scratch output dir
#   SPEEDAVG_SKIP_CONDITION=1          run although KINDX's s1 read is absent (a rehearsal; the marker then says REHEARSAL and the outcome is NOT a result)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(cd "$HERE/.." && pwd)
REPO=$(cd "$CODE/../../../../.." && pwd)
if command -v cygpath >/dev/null 2>&1; then REPO=$(cygpath -m "$REPO"); CODE=$(cygpath -m "$CODE"); fi       # C:/... for the native python (Git-bash would hand it /tmp/... otherwise)
PK="$REPO/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
STAGE=${1:-all}

PY=${SPEEDAVG_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
SC=${SPEEDAVG_SC:-C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad}
SF=${SPEEDAVG_SF:-$SC/sfresh}
OUT=${SPEEDAVG_OUT:-$SC/speedavg_run}
B2=${SPEEDAVG_B2:-C:/Users/Admin/box2_ship}
KX1DIR=${SPEEDAVG_KX1DIR:-$B2/logs/v8f_kindx_s1}
X0=${SPEEDAVG_X0:-$B2/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt}
X1=${SPEEDAVG_X1:-$KX1DIR/runs/M2b_s1_mixo/ckpt_last.pt}
X0_SHA=${SPEEDAVG_X0_SHA:-c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d}
KX0=${SPEEDAVG_KINDX_S0:-$PK/raw/2026-10-25-kindx/eval}
KX1=${SPEEDAVG_KINDX_S1:-$KX1DIR/eval}
BK0=${SPEEDAVG_BANKED_S0:-$B2/logs/v8f_mixo/eval/eval_M2b-mixo_s0.npz}
BK1=${SPEEDAVG_BANKED_S1:-$KX1DIR/eval/eval_M2b-mixo_s1.npz}
READ_S1=${SPEEDAVG_READ_S1:-$KX1DIR/READ_s1.json}
R2=D:/tanitad/runs/v8f_mini_full                  # box 2's run root (read-only pulls only)
DRV="$REPO/stack/scripts/v8f_mini_eval_speedavg.py"
RDR="$CODE/v8f_speedavg_read.py"
LEAD_BLOCK=${SPEEDAVG_LEAD_BLOCK:-}               # default: the reader's own (the repo's b1_eval_lead_block.npz)
THREADS=${SPEEDAVG_THREADS:-8}
export CUDA_VISIBLE_DEVICES=-1 SPEEDAVG_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
MIN_FREE_KB=2097152                               # abort if C: has < 2 GB free

say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
have() {   # have <path>: "present" / "absent" for a path; a DRY RUN does not stat anything under box2_ship (the other session's tree): it says so instead
  if [ "$DRY" = 1 ] && [[ "$1" == "$B2"* ]]; then echo "not probed in a dry run (under box2_ship)"; elif [ -e "$1" ]; then echo present; else echo absent; fi
}
stop() { say "ZZSPEEDAVG-STOP $*"; exit 3; }
run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }

# ---- the data arguments the driver needs (KINDX's flags with this session's paths)
DATA=(--eval-dir "$OUT/eval" --eval-cache "$SF/cache" --tracks "$SF/inputs/native_tracks.pt" --v9-eval "$SF/inputs/v9/v9_labels_eval139.npz" --lead-eval "$SF/inputs/v9/lead_eval139.npz"
      --speed-eval "$SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl" --p10 "$SF/inputs/p10/p10_prior.pt")
x1sha() {   # X_s1's sha256 as measured on box 2 after training: the env value, else the KINDX chain's json
  if [ -n "${SPEEDAVG_X1_SHA:-}" ]; then echo "$SPEEDAVG_X1_SHA"; return; fi
  [ -f "$KX1DIR/ckpt_sha256.json" ] && $PY -I -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('M2b-mixo_s1_trained',''))" "$KX1DIR/ckpt_sha256.json"
}
sha_of() { $PY -I "$DRV" sha "$1"; }

stage_check() {
  say "== check"
  local free; free=$(df -k /c | tail -1 | awk '{print $4}')
  if [ "$DRY" = 1 ]; then echo "  C: free ${free} KB (a real run needs >= ${MIN_FREE_KB} KB and stops otherwise)"
  else [ "$free" -ge "$MIN_FREE_KB" ] || stop "C: has less than 2 GB free (${free} KB)"; fi
  local d; for d in "$DRV" "$RDR" "$SF/cache" "$SF/inputs/native_tracks.pt" "$SF/inputs/p10/p10_prior.pt" "$SF/inputs/v9/lead_eval139.npz" "$KX0/eval_M2b-mixo_s0-kindx-own.npz" "$KX0/eval_M2b-mixo_s0-kindx-only-speed.npz"; do
    [ "$DRY" = 1 ] && { echo "  $d: $(have "$d")"; continue; }
    [ -e "$d" ] || stop "missing input: $d"
  done
  # the PREREG's condition: KINDX's seed-1 read must make SPEED a carrier
  if [ "$DRY" = 1 ]; then
    echo "  KINDX s1 read (the PREREG's condition): $READ_S1: $(have "$READ_S1"); the stage requires OUTCOME CARRIER with speed among the carriers"
  elif [ -f "$READ_S1" ]; then
    echo "  KINDX s1 read: $READ_S1"
    [ "$DRY" = 1 ] || $PY -I -c "
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
ok = r.get('OUTCOME')=='CARRIER' and 'speed' in (r.get('carriers') or [])
print('KINDX s1 OUTCOME', r.get('OUTCOME'), 'carriers', r.get('carriers'))
sys.exit(0 if ok else 4)" "$READ_S1" || { [ "${SPEEDAVG_SKIP_CONDITION:-0}" = 1 ] && say "REHEARSAL: the KINDX condition is NOT met" || stop "the PREREG's condition is not met: KINDX s1 did not make SPEED a carrier ($READ_S1)"; }
  else
    echo "  KINDX s1 read ABSENT: $READ_S1"
    [ "$DRY" = 1 ] || [ "${SPEEDAVG_SKIP_CONDITION:-0}" = 1 ] || stop "KINDX's seed-1 read ($READ_S1) does not exist: this arm is CONDITIONAL on it (SPEEDAVG_SKIP_CONDITION=1 for a rehearsal)"
  fi
  if [ "$DRY" = 1 ]; then
    echo "  KINDX s1 evals (own, only-speed) in $KX1: $(have "$KX1/eval_M2b-mixo_s1-kindx-own.npz")"
  else
    { [ -e "$KX1/eval_M2b-mixo_s1-kindx-own.npz" ] && [ -e "$KX1/eval_M2b-mixo_s1-kindx-only-speed.npz" ]; } || stop "KINDX's seed-1 evals (own, only-speed) are not in $KX1"
  fi
  local X1S; X1S=$(x1sha)
  if [ "$DRY" = 1 ]; then
    echo "  X_s0: $X0 (expect sha256 $X0_SHA): $(have "$X0"); if absent, fetch reads it from box 2 with: $SCP tanitad-box2:$R2/M2b_s0_mixo/ckpt_last.pt"
    echo "  X_s1: $X1 (expect sha256 ${X1S:-UNKNOWN-until-the-KINDX-json-exists-else-the-box-side-hash}): $(have "$X1"); if absent, fetch reads it from box 2 with: $SCP tanitad-box2:$R2/M2b_s1_mixo/ckpt_last.pt"
    return 0
  fi
  mkdir -p "$OUT/logs" "$OUT/eval"
  $PY -I -c "import json,sys;json.dump({'x0':sys.argv[1],'x1':sys.argv[2],'x0_sha':sys.argv[3],'x1_sha':sys.argv[4],'kindx_s0':sys.argv[5],'kindx_s1':sys.argv[6]},open(sys.argv[7],'w'),indent=1)" "$X0" "$X1" "$X0_SHA" "$X1S" "$KX0" "$KX1" "$OUT/check.json"
  say "CHECK OK"
}

fetch_one() {   # fetch_one <local path> <box-2 dir name> <expected sha256 or ''>
  local dst=$1 name=$2 want=$3
  if [ ! -e "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    say "pull (read-only) tanitad-box2:$R2/$name/ckpt_last.pt -> $dst"
    $SCP "tanitad-box2:$R2/$name/ckpt_last.pt" "$dst" || stop "scp of $name failed"
  fi
  local got; got=$(sha_of "$dst")
  if [ -z "$want" ]; then   # no committed / measured value: the box-side hash is the reference (a read-only ssh call)
    want=$($SSH "powershell -NoProfile -Command \"(Get-FileHash -Algorithm SHA256 -LiteralPath '$R2/$name/ckpt_last.pt').Hash.ToLower()\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{64}$')
    [ -n "$want" ] || stop "no reference sha256 for $name (no env value, no KINDX json, box 2 unreachable)"
  fi
  [ "$got" = "$want" ] || stop "sha256 of $dst is $got, expected $want"
  say "sha256 ok $name $got"
}

stage_fetch() {
  say "== fetch"
  if [ "$DRY" = 1 ]; then
    echo "  X_s0: if $X0 is absent -> $SCP tanitad-box2:$R2/M2b_s0_mixo/ckpt_last.pt $OUT/ckpt/M2b_s0_mixo/ckpt_last.pt ; then sha256 == $X0_SHA"
    echo "  X_s1: if $X1 is absent -> $SCP tanitad-box2:$R2/M2b_s1_mixo/ckpt_last.pt $OUT/ckpt/M2b_s1_mixo/ckpt_last.pt ; then sha256 == the KINDX json's M2b-mixo_s1_trained, else the box-side Get-FileHash (read-only ssh)"
    return 0
  fi
  [ -e "$X0" ] || X0="$OUT/ckpt/M2b_s0_mixo/ckpt_last.pt"
  [ -e "$X1" ] || X1="$OUT/ckpt/M2b_s1_mixo/ckpt_last.pt"
  fetch_one "$X0" M2b_s0_mixo "$X0_SHA"
  local X1S; X1S=$(x1sha)
  fetch_one "$X1" M2b_s1_mixo "$X1S"
  [ "$(sha_of "$X0")" != "$(sha_of "$X1")" ] || stop "X_s0 and X_s1 are the same file: no second seed"
  printf '%s\n%s\n' "$X0" "$X1" > "$OUT/ckpt_paths.txt"
  say "fetch OK"
}

resolve() {   # after fetch: the checkpoint paths actually used
  if [ -f "$OUT/ckpt_paths.txt" ]; then X0=$(sed -n 1p "$OUT/ckpt_paths.txt"); X1=$(sed -n 2p "$OUT/ckpt_paths.txt"); fi
}

stage_evals() {   # stage_evals <seed>
  local s=$1
  say "== evals seed $s (own, self, gt, avg)"
  resolve
  local X1S; X1S=$(x1sha)
  local args=(evals --seed "$s" --x0 "$X0" --x1 "$X1" --expect-sha0 "$X0_SHA" --expect-sha1 "${X1S:-<X1_SHA>}" "${DATA[@]}")
  if [ "$DRY" = 1 ]; then
    echo "  [dry-run] $PY -I $DRV ${args[*]}"
    echo "  the driver's own dry-run (the v8f_mini_eval.py argv of each mode):"
    $PY -I "$DRV" "${args[@]}" --dry-run | sed 's/^/    /'
    return 0
  fi
  mkdir -p "$OUT/logs" "$OUT/eval"
  $PY -I "$DRV" "${args[@]}" > "$OUT/logs/evals$s.log" 2>&1 || { tail -5 "$OUT/logs/evals$s.log"; stop "evals seed $s failed (see $OUT/logs/evals$s.log)"; }
  grep -E "^\[(ok|skip)\]|SPEEDAVG evals done" "$OUT/logs/evals$s.log"
}

stage_read() {
  say "== read"
  resolve
  local X1S; X1S=$(x1sha)
  local args=(--eval-dir "$OUT/eval" --kindx-dir-s0 "$KX0" --kindx-dir-s1 "$KX1" --expect-sha0 "$X0_SHA" --expect-sha1 "${X1S:-<X1_SHA>}" --out "$OUT/READ.json")
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
  if [ "${SPEEDAVG_SKIP_CONDITION:-0}" = 1 ] && [ ! -f "$READ_S1" ]; then say "ZZSPEEDAVG-DONE REHEARSAL (the KINDX condition was not met: the outcome $oc is NOT a result)"; else say "ZZSPEEDAVG-DONE $oc"; fi
}

if [ "$DRY" = 1 ]; then
  echo "== run_speedavg.sh --dry-run (nothing is read, written or launched; no torch is imported)"
  echo "repo $REPO"; echo "out  $OUT"; echo "threads $THREADS (CPU only)"
fi
case "$STAGE" in
  check)  stage_check ;;
  fetch)  stage_fetch ;;
  evals0) stage_evals 0 ;;
  evals1) stage_evals 1 ;;
  read)   stage_read ;;
  all)    stage_check; stage_fetch; stage_evals 0; stage_evals 1; stage_read ;;
  *) echo "unknown stage $STAGE (check | fetch | evals0 | evals1 | read | all)"; exit 2 ;;
esac
[ "$DRY" = 1 ] && echo "== dry-run done: the stages above are what 'all' runs, in this order; the final marker is ZZSPEEDAVG-DONE <OUTCOME> or ZZSPEEDAVG-STOP <reason>"
exit 0
