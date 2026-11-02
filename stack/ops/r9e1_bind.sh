#!/usr/bin/env bash
# r9e1_bind.sh -- THOR-SIDE producer of the two BINDING overfit records for the refcv9 E1 perception rig
# (launch-gate profile `refcv9-e1rig`; C12-E1 blocker 2).
#
#   r9e1_bind.sh [--dry-run] <map|box> <C0|C1> --commit SHA40 [--root DIR] [--code DIR] [--inputs DIR]
#
# WHY IT EXISTS. G-MAP-OVERFIT and G-BOX-OVERFIT bind the LAUNCH ARGV sha256 (launch_gate.py _overfit_binding_reasons:
# schema + `launch_argv_sha256`). The sha is over the WHOLE argv -- `--seed` and `--out` included -- so C0 and C1 need ONE
# RECORD EACH (their shas differ). The binding was NOT redesigned (MM decision): the records are produced under the rig argv
# `stack/ops/runs.d/r9e1_<ARM>.argv.json` of the shipped tree, through the SAME harness, templates and bars as refcv8's
# (`stack/scripts/refcv8_bind_{map,box}.args.tpl`, `closure_run.py --binding`); the argv sha is the only thing that changes.
#
# Mirrors the `bind` mode of r9w_prelaunch.sh (refcv9-warm, the precedent that worked on this Thor), with:
#   * the argv read from the SHIPPED TREE (the committed file, never a copy);
#   * ONE kind x ONE arm per call, so a long job is a detached call of its own (CLAUDE.md token rule 1);
#   * the GPU protocol of r9e1_run.sh: the Thor lock (blocking, <= 2 h) THEN an EMPTY compute-apps list;
#   * every verdict read from an ARTIFACT (closure json, record json, the assert-* markers), never from $?.
#
# Outputs (BD = $ROOT/ovf_<ARM>_<C12>):  BD/run_<kind>/{g_map_overfit.json | gbo_binding.json}, BD/<kind>_closure.json,
#   BD/<kind>.{out,render.txt,closure.txt,record.txt}, BD/<kind>.TIMING (start/end epoch, wall s).
# Markers:  ZZR9E1BIND-<KIND>-<ARM>-OKZZ  ZZR9E1BIND-REFUSEDZZ  ZZR9E1BIND-DRYOKZZ
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
KIND="${1:-}"; ARM="${2:-}"; shift 2 || true
ROOT=/home/nvidia/r8ns/r9e1; CODE=""; COMMIT=""; INPUTS=""
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
while [ $# -gt 0 ]; do case "$1" in
  --commit) COMMIT="$2"; shift 2;; --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --inputs) INPUTS="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 2;; esac; done
CODE="${CODE:-$ROOT/launch_tree}"
INPUTS="${INPUTS:-/home/nvidia/refcv7_bind/run_map/inputs}"      # the A19 MAIN spec + frameset (refcv8's measured location)
say() { echo "[r9e1-bind:$KIND:$ARM] $*"; }
die() { say "REFUSED: $*"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 10; }
pyval() { "$PY" -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }
case "$KIND" in map|box) ;; *) die "kind must be map|box";; esac
case "$ARM" in C0|C1) ;; *) die "arm must be C0|C1";; esac
[ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
C12="${COMMIT:0:12}"; BD="$ROOT/ovf_${ARM}_$C12"; HERE="$CODE/stack/scripts"
ARGV="$CODE/stack/ops/runs.d/r9e1_$ARM.argv.json"
[ -f "$ARGV" ] || die "no rig argv $ARGV in the shipped tree"
[ -f "$HERE/refcv8_assemble_argv.py" ] && [ -f "$HERE/closure_run.py" ] || die "no refcv8_assemble_argv.py / closure_run.py under $HERE"
# the rig's own template where one exists (box: the --candidate text and the data roots; map: refcv8's, unchanged)
TPL="$HERE/r9e1_bind_$KIND.args.tpl"; [ -f "$TPL" ] || TPL="$HERE/refcv8_bind_$KIND.args.tpl"; [ -f "$TPL" ] || die "no harness template $TPL"
mkdir -p "$BD" 2>/dev/null
# the argv sha: the GATE'S OWN function on the GATE'S OWN loader (no second implementation)
ASHA="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(L.argv_sha256(L.load_argv_file(sys.argv[2])))' "$CODE" "$ARGV")"
[ "${#ASHA}" = 64 ] || die "could not hash the launch argv $ARGV"
say "argv $ARGV sha256 $ASHA"
# the BOX harness READS its prereg + frame set from {OUT}/audit/raw (refcv8_prelaunch.sh bind): copies, md5 LITERALS (blob md5s)
if [ "$KIND" = box ] && [ "$DRY" != 1 ]; then
  BAUD="$BD/run_box/audit/raw"; BSRC="$CODE/stack/ops/refcv8_box_audit/raw"
  mkdir -p "$BAUD/visibility" 2>/dev/null
  cp -p "$BSRC/PREREG_G_BOX_OVERFIT.md" "$BAUD/" && cp -p "$BSRC/visibility/gbo_frameset.json" "$BAUD/visibility/" || die "could not prepare the box audit inputs from $BSRC"
  echo "594c71196cc5bbd527fee40b2cb0e3f1  $BAUD/PREREG_G_BOX_OVERFIT.md" | md5sum -c - > /dev/null 2>&1 || die "box audit prereg md5"
  echo "b291404c36f83c3e397e3b90367e8e7b  $BAUD/visibility/gbo_frameset.json" | md5sum -c - > /dev/null 2>&1 || die "box audit frame-set md5"
fi
"$PY" "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
    --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$KIND" --sub "INPUTS=$INPUTS" \
    --args-out "$BD/$KIND.args" --data-roots-out "$BD/$KIND.roots" > "$BD/$KIND.render.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^RENDERED' "$BD/$KIND.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$KIND.render.txt)"
if [ "$DRY" = 1 ]; then
  say "DRY: would hold $LOCKF (blocking <= 2 h), assert compute-apps EMPTY, then closure_run.py --binding --commit $C12 --tree $CODE -- $(wc -l < "$BD/$KIND.args") harness tokens (argv sha ${ASHA:0:16})"
  echo "ZZR9E1BIND-DRYOKZZ"; exit 0
fi
# ---- GPU protocol (r9e1_run.sh): the blocking lock, then an EMPTY GPU
exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
[ "$NAPPS" = 0 ] || die "the GPU is not empty under the lock ($NAPPS compute app(s)): never share the GPU"
mapfile -t HARGS < "$BD/$KIND.args"; mapfile -t ROOTS < "$BD/$KIND.roots"
DR=(); for r in "${ROOTS[@]}"; do [ -n "$r" ] && DR+=(--data-root "$r"); done
mkdir -p "$BD/run_$KIND" 2>/dev/null
if [ "$KIND" = map ]; then REC="$BD/run_map/g_map_overfit.json"; else REC="$BD/run_box/gbo_binding.json"; fi
T0=$(date +%s); echo "$T0" > "$BD/$KIND.TIMING"; say "lock held, GPU empty; closure_run --binding starts at $(date -u +%T)"
( cd "$CODE" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
  "$PY" "$CODE/stack/scripts/closure_run.py" --binding --commit "$COMMIT" --tree "$CODE" \
    --out "$BD/${KIND}_closure.json" --result "$REC" "${DR[@]}" -- "${HARGS[@]}" > "$BD/$KIND.out" 2>&1 9>&- 200>&- )
T1=$(date +%s); echo "$T0 $T1 $((T1 - T0))" > "$BD/$KIND.TIMING"; say "harness finished, wall $((T1 - T0)) s"
"$PY" "$HERE/refcv8_assemble_argv.py" --assert-closure "$BD/${KIND}_closure.json" > "$BD/$KIND.closure.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^ASSERT-OK' "$BD/$KIND.closure.txt")" = 1 ] || die "$KIND closure refused: $(tr -d '\r' < "$BD/$KIND.closure.txt" | head -3 | tr '\n' ' ')"
"$PY" "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$KIND" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^ASSERT-OK' "$BD/$KIND.record.txt")" = 1 ] || die "$KIND record refused: $(tr -d '\r' < "$BD/$KIND.record.txt" | head -3 | tr '\n' ' ')"
say "$KIND binding run asserted: closure $BD/${KIND}_closure.json, record $REC (argv sha ${ASHA:0:16})"
echo "ZZR9E1BIND-${KIND}-${ARM}-OKZZ"
