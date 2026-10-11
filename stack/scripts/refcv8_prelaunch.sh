#!/usr/bin/env bash
# refcv8 PRE-LAUNCH kit: argv assembly -> Thor gate (G-LIVE smoke AS LAUNCHED + every refcv8 check) -> verify -> fire.
#
#   DEV BOX  : refcv8_prelaunch.sh assemble --stage measure --l1-recipe F --l2-k K --l3-w W --b1-ref-ap2m V
#                  -> stack/ops/runs.d/refcv8-r101-s0.measure.argv.json (100 steps, --grad-share-every 50 --grad-share-at 50)
#              refcv8_prelaunch.sh assemble --l1-recipe F --l2-k K --l3-w W --b1-ref-ap2m V --gs-measure gs_measure.json
#                  prints the SET lines (each flag traced to its SPEC clause) and writes
#                  stack/ops/runs.d/refcv8-r101-s0.argv.json  -> MM COMMITS IT in the launch commit, then ships CODE.
#                  There is NO unmeasured-reading escape: --gs-measure is required (SPEC_REFCV8 10.3, ruled 2026-10-11).
#   THOR     : refcv8_prelaunch.sh gs-measure --code DIR --commit SHA40     (G-LIVE measurement smoke: 100 steps, ONE full-size
#                  gradient-share reading; writes gs_measure.json = {reading_s, s_per_step} from the metrics rows)
#              refcv8_prelaunch.sh bind --code DIR --commit SHA40           (BINDING G-MAP-OVERFIT then G-BOX-OVERFIT, MAIN-only
#                  per A19, closure_run.py --binding, sequential under the GPU lock, BEFORE the gate; ~34 + ~82 min INHERITED)
#              refcv8_prelaunch.sh gate  --code DIR --commit SHA40 [--devbox-evidence DIR] [--suite-verdict F] [--pi-cost-approval F]
#                  (hours; run under nohup; the overfit records come from `bind`)
#              refcv8_prelaunch.sh fire  --code DIR --commit SHA40   (verify == MATCH, write manifest, start sup_refcv7.sh)
#   ANY      : refcv8_prelaunch.sh --dry-run <mode> ...   prints every command it WOULD run; touches nothing.
#
# Rules this script keeps (each cost a run): the verdict is read from the gate's ARTIFACT (token / verify json), never
# from $? and never through a pipe; counts are computed in python and read back from a FILE; every child closes the
# lock fds (9, 200); no `pgrep -f`/`pkill -f`; the ladder must not be running; nothing is written to G:.
# REFUSED BY NAME (PI 2026-10-11, ladder limited to 11 runs): --r8-speed-enc8, --r8-critic-drivable, --w-r8-drivable.
# Marker tokens are disjoint from anything they are grepped for (ZZ...ZZ).
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MODE="${1:-}"; shift || true
PY="${PYTHON:-python3}"
# dev box (Git-Bash + Windows python): hand python a native path for any argument that EXISTS here; leave
# /home/nvidia/... untouched (MSYS would otherwise rewrite it to C:/Program Files/Git/home/...). No-op on Thor.
export MSYS_NO_PATHCONV=1
py() { local a=() x; for x in "$@"; do
  if command -v cygpath >/dev/null 2>&1 && case "$x" in /*) [ -e "$x" ] || [ -d "$(dirname "$x")" ];; *) false;; esac; then
    a+=("$(cygpath -m "$x")"); else a+=("$x"); fi; done; "$PY" "${a[@]}"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=refcv8-r101-s0; OUTROOT="${OUT_ROOT:-/home/nvidia/refcv8_run}"; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
CODE=""; COMMIT=""; L1=""; K=""; W=""; B1REF=""; GSM=""; STAGE="launch"; DEVEV=""; SUITE=""; PICOST=""
while [ $# -gt 0 ]; do case "$1" in
  --l1-recipe) L1="$2"; shift 2;; --l2-k) K="$2"; shift 2;; --l3-w) W="$2"; shift 2;;
  --b1-ref-ap2m) B1REF="$2"; shift 2;; --gs-measure) GSM="$2"; shift 2;; --stage) STAGE="$2"; shift 2;;
  --code) CODE="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;; --devbox-evidence) DEVEV="$2"; shift 2;;
  --suite-verdict) SUITE="$2"; shift 2;; --pi-cost-approval) PICOST="$2"; shift 2;;
  --out-root) OUTROOT="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done

say() { echo "[prelaunch:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZPRELAUNCH-REFUSEDZZ"; exit 10; }
run() { if [ "$DRY" = 1 ]; then say "DRY: $*"; else "$@"; fi; }
need() { for v in "$@"; do [ -n "${!v}" ] || die "--$(echo "$v" | tr 'A-Z_' 'a-z-') is required for '$MODE'"; done; }
pyval() { py -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }   # a value, never a status

case "$MODE" in
# ------------------------------------------------------------------------------------------------------ assemble
assemble)
  need L1 K W B1REF
  [ "$STAGE" = measure ] || [ "$STAGE" = launch ] || die "--stage must be measure|launch"
  [ "$STAGE" = measure ] || [ -n "$GSM" ] || die "--gs-measure is required for the launch stage (no unmeasured-reading escape; run 'gs-measure' on Thor)"
  CODE="${CODE:-$(cd "$HERE/../.." && pwd)}"
  BASE="$CODE/stack/ops/runs.d/refcv8-wpb-smoke.argv.json"; OUTJ="$CODE/stack/ops/runs.d/$ARM.argv.json"
  [ "$STAGE" = measure ] && OUTJ="$CODE/stack/ops/runs.d/$ARM.measure.argv.json"
  [ -f "$BASE" ] || die "no base smoke argv at $BASE"
  [ -f "$L1" ] || die "ladder L1 recipe file $L1 missing"
  [ "$DRY" = 1 ] && OUTJ="${TMPDIR:-/tmp}/$ARM.argv.dryrun.json"
  LOG="${OUTJ}.assemble.log"
  py "$HERE/refcv8_assemble_argv.py" --self-test > "$LOG" 2>&1 9>&- 200>&-
  # the launch tree on Thor has a FIXED name (the argv embeds it; a sha-named dir would be a chicken-and-egg)
  CODE_ON_THOR="${CODE_ON_THOR:-/home/nvidia/refcv8_run/launch_tree}"
  RUNOUT="$OUTROOT/runs/$ARM"; [ "$STAGE" = measure ] && RUNOUT="$OUTROOT/runs/${ARM}_gsmeasure"
  GSARG=(); [ -n "$GSM" ] && GSARG=(--gs-measure "$GSM")
  py "$HERE/refcv8_assemble_argv.py" --base "$BASE" --code "$CODE_ON_THOR" --run-out "$RUNOUT" --stage "$STAGE" \
      --recipe-json "$L1" --k-tac "$K" --w-map-hires "$W" --b1-ref-ap2m "$B1REF" "${GSARG[@]}" --out-json "$OUTJ" >> "$LOG" 2>&1 9>&- 200>&-
  cat "$LOG" | tr -d '\r'
  [ "$(grep -c '^SELFTEST-OK' "$LOG")" = 1 ] || die "assembler self-test did not print SELFTEST-OK"
  [ "$(grep -c '^ASSEMBLED' "$LOG")" = 1 ] && [ -s "$OUTJ" ] || die "no assembled argv artifact at $OUTJ"
  if [ "$STAGE" = measure ]; then say "measurement argv at $OUTJ. NEXT: commit it with the tree, ship CODE to Thor, run 'gs-measure', then assemble the launch stage with --gs-measure."
  else say "argv at $OUTJ (sha256 $(sha256sum "$OUTJ" | cut -c1-16)...). NEXT: commit it, ship CODE (git archive of that commit) to Thor, run 'bind' then 'gate'."; fi
  echo "ZZASSEMBLE-OKZZ";;
# ------------------------------------------------------------------------------------------------------ gs-measure
gs-measure)
  need CODE COMMIT
  C12="${COMMIT:0:12}"; ARGVM="$CODE/stack/ops/runs.d/$ARM.measure.argv.json"; MD="$OUTROOT/gs_measure_$C12"
  mkdir -p "$MD" 2>/dev/null; MOUT="$OUTROOT/runs/${ARM}_gsmeasure"
  [ -f "$ARGVM" ] || die "no measurement argv at $ARGVM (run 'assemble --stage measure')"
  py "$HERE/refcv8_assemble_argv.py" --audit "$ARGVM" --stage measure > "$MD/audit.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^AUDIT-OK' "$MD/audit.txt")" = 1 ] || die "measurement argv audit failed ($MD/audit.txt)"
  pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print("\n".join(a))' "$ARGVM" > "$MD/args.txt"
  if [ "$DRY" = 1 ]; then say "DRY: flock $LOCKF; $CODE/stack/scripts/refc_v3_train.py with $(wc -l < "$MD/args.txt") args (100 steps, --grad-share-every 50 --grad-share-at 50) -> $MOUT/metrics.jsonl -> gs_measure.json"; echo "ZZGSMEASURE-DRYOKZZ"; exit 0; fi
  [ ! -e "$MOUT/metrics.jsonl" ] || die "$MOUT/metrics.jsonl already exists (a measurement is one run; move it aside)"
  exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h; the ladder/evals release it between jobs)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
  mapfile -t ARGS < "$MD/args.txt"
  ( cd "$CODE/stack" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
    /home/nvidia/venvs/tanitad-train/bin/python "$CODE/stack/scripts/refc_v3_train.py" "${ARGS[@]}" > "$MD/train.log" 2>&1 9>&- 200>&- )
  # the verdict is the ARTIFACT: a gs_measure.json built from metrics rows, not the trainer's exit status
  [ -s "$MOUT/metrics.jsonl" ] || die "no metrics.jsonl at $MOUT (see $MD/train.log)"
  py "$HERE/refcv8_assemble_argv.py" --read-gs-measure "$MOUT/metrics.jsonl" --out-json "$MD/gs_measure.json" > "$MD/read.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^GSMEASURE' "$MD/read.txt")" = 1 ] && [ -s "$MD/gs_measure.json" ] || die "no usable reading: $(tr -d '\r' < "$MD/read.txt" | tail -1)"
  say "$(tr -d '\r' < "$MD/read.txt")  -> $MD/gs_measure.json"; echo "ZZGSMEASURE-OKZZ";;
# ------------------------------------------------------------------------------------------------------ bind
bind)
  need CODE COMMIT
  [ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
  C12="${COMMIT:0:12}"; ARGV="$CODE/stack/ops/runs.d/$ARM.argv.json"; BD="$OUTROOT/bind_$C12"
  [ -f "$ARGV" ] || die "no launch argv at $ARGV"
  mkdir -p "$BD" 2>/dev/null
  # the harness records bind to sha256(json.dumps(argv, separators=(",",":"))) (refcv7_bind gbo_binding.json launch_argv_sha256_form)
  ASHA="$(pyval 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(hashlib.sha256(json.dumps(a,separators=(",",":")).encode()).hexdigest())' "$ARGV")"
  [ "${#ASHA}" = 64 ] || die "could not hash the launch argv"
  for K2 in map box; do
    TPL="$HERE/refcv8_bind_$K2.args.tpl"; [ -f "$TPL" ] || die "no harness template $TPL"
    py "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
        --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$K2" --sub "INPUTS=${BIND_INPUTS:-/home/nvidia/refcv7_bind/run_$K2/inputs}" \
        --args-out "$BD/$K2.args" --data-roots-out "$BD/$K2.roots" > "$BD/$K2.render.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^RENDERED' "$BD/$K2.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$K2.render.txt)"
  done
  if [ "$DRY" = 1 ]; then say "DRY: flock $LOCKF; map ($(wc -l < "$BD/map.args") args) then box ($(wc -l < "$BD/box.args") args) via closure_run.py --binding (MAIN arm only, A19), then assert the closure + record JSONs"; echo "ZZBIND-DRYOKZZ"; exit 0; fi
  NL="$(ps -eo args | grep -c 'ladder_chai[n]')"; [ "$NL" = 0 ] || die "the ladder is running"
  exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h; the ladder/evals release it between jobs)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
  for K2 in map box; do
    mapfile -t HARGS < "$BD/$K2.args"; mapfile -t ROOTS < "$BD/$K2.roots"; DR=(); for r in "${ROOTS[@]}"; do [ -n "$r" ] && DR+=(--data-root "$r"); done
    mkdir -p "$BD/run_$K2" 2>/dev/null
    if [ "$K2" = map ]; then REC="$BD/run_map/g_map_overfit.json"; else REC="$BD/run_box/gbo_binding.json"; fi
    ( cd "$CODE" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
      /home/nvidia/venvs/tanitad-train/bin/python "$CODE/stack/scripts/closure_run.py" --binding --commit "$COMMIT" --tree "$CODE" \
        --out "$BD/${K2}_closure.json" --result "$REC" "${DR[@]}" -- "${HARGS[@]}" > "$BD/$K2.out" 2>&1 9>&- 200>&- )
    # assert on the ARTIFACTS (closure + record JSON), never on $?
    py "$HERE/refcv8_assemble_argv.py" --assert-closure "$BD/${K2}_closure.json" > "$BD/$K2.closure.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.closure.txt")" = 1 ] || die "$K2 closure refused: $(tr -d '\r' < "$BD/$K2.closure.txt" | head -3 | tr '\n' ' ')"
    py "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$K2" > "$BD/$K2.record.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.record.txt")" = 1 ] || die "$K2 record refused: $(tr -d '\r' < "$BD/$K2.record.txt" | head -3 | tr '\n' ' ')"
    say "$K2 binding run asserted: closure $BD/${K2}_closure.json, record $REC"
  done
  say "records for the gate: --map-overfit-record $BD/run_map/g_map_overfit.json --map-overfit-closure $BD/map_closure.json --box-overfit-record $BD/run_box/gbo_binding.json --box-overfit-closure $BD/box_closure.json"
  echo "ZZBIND-OKZZ";;
# ------------------------------------------------------------------------------------------------------ gate
gate)
  need CODE COMMIT
  [ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
  C12="${COMMIT:0:12}"; GATE_OUT="$OUTROOT/gate_$C12"; WORK="$GATE_OUT/prelaunch"; ARGV="$CODE/stack/ops/runs.d/$ARM.argv.json"
  GATE="$CODE/stack/scripts/launch_gate.py"; PF="$WORK/preflight.txt"
  mkdir -p "$WORK" 2>/dev/null; : > "$PF"
  pf() { echo "$1 $2" >> "$PF"; }                                   # PF-OK / PF-FAIL <name>
  [ -f "$GATE" ] && pf PF-OK gate-present || pf PF-FAIL gate-present
  [ -f "$ARGV" ] && pf PF-OK argv-committed || pf PF-FAIL argv-committed
  [ -f "$CODE/stack/tanitad/configs/refcv8_join_label_defects.json" ] && pf PF-OK joindefects || pf PF-FAIL joindefects
  if [ "$DRY" = 1 ] && [ -z "$GSM" ]; then printf 'AUDIT-SKIPPED(dry-run, no --gs-measure)\nAUDIT-OK\n' > "$WORK/audit.txt"
  else GSF="${GSM:-$(ls -1 "$OUTROOT"/gs_measure_*/gs_measure.json 2>/dev/null | tail -1)}"
       py "$HERE/refcv8_assemble_argv.py" --audit "$ARGV" --gs-measure "${GSF:-/nonexistent}" > "$WORK/audit.txt" 2>&1 9>&- 200>&-; fi
  [ "$(grep -c '^AUDIT-OK' "$WORK/audit.txt")" = 1 ] && pf PF-OK argv-audit || pf PF-FAIL argv-audit
  JD="$(pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(a[a.index("--join-defect-masks")+1])' "$ARGV")"
  [ "$JD" = "$CODE/stack/tanitad/configs/refcv8_join_label_defects.json" ] && pf PF-OK argv-code-path || pf PF-FAIL argv-code-path
  if [ "$DRY" != 1 ]; then
    CK=/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt
    [ "$(stat -c %s "$CK" 2>/dev/null)" = 1172814977 ] && pf PF-OK ckpt-size || pf PF-FAIL ckpt-size
    M="$(md5sum "$CK" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = d5f104ee54ba6b2861e38030b4f6fcf1 ] && pf PF-OK ckpt-md5 || pf PF-FAIL ckpt-md5
    for f in v9_labels_train:f63ece410b725febb8a5242cf2b01d3c v9_labels_eval139:6b5c7f207cffc3b7eb3cd527fd433599; do
      M="$(md5sum /home/nvidia/refcv8_v9labels/release/${f%%:*}.npz 2>/dev/null | cut -c1-32)"
      [ "${#M}" = 32 ] && [ "$M" = "${f##*:}" ] && pf PF-OK "md5-${f%%:*}" || pf PF-FAIL "md5-${f%%:*}"; done
    [ -s /home/nvidia/data/refcv6_pose_sync_sidecar.jsonl ] && pf PF-OK pose-sync || pf PF-FAIL pose-sync
    [ ! -e "$OUTROOT/runs/$ARM/ckpt.pt" ] && [ ! -e "$OUTROOT/runs/$ARM/summary.json" ] && pf PF-OK run-dir-fresh || pf PF-FAIL run-dir-fresh
    exec 9>"$LOCKF"; flock -n 9 && pf PF-OK gpu-lock || pf PF-FAIL gpu-lock
    NL="$(ps -eo args | grep -c 'ladder_chai[n]')"; [ "$NL" = 0 ] && pf PF-OK ladder-stopped || pf PF-FAIL ladder-stopped
    FREE="$(df -BG --output=avail "$OUTROOT" 2>/dev/null | tail -1 | tr -dc 0-9)"; [ "${FREE:-0}" -ge 60 ] && pf PF-OK disk-60G || pf PF-FAIL disk-60G
  else say "DRY: skipping Thor-only preflights (ckpt md5, v9 md5, sidecars, lock, ladder, disk)"; fi
  NF="$(grep -c '^PF-FAIL' "$PF")"; cat "$PF"
  [ "$NF" = 0 ] || die "$NF preflight failure(s) in $PF"
  A=(run --profile refcv8 --stage thor --tree "$CODE" --commit "$COMMIT" --argv-file "$ARGV" --out-dir "$GATE_OUT" --eval-kit /home/nvidia)
  [ -n "$DEVEV" ] && A+=(--import-evidence "$DEVEV"); [ -n "$SUITE" ] && A+=(--suite-verdict "$SUITE")
  [ -n "$PICOST" ] && A+=(--pi-cost-approval "$PICOST")
  BD="$OUTROOT/bind_$C12"
  MAPREC="$BD/run_map/g_map_overfit.json"; MAPCLO="$BD/map_closure.json"; BOXREC="$BD/run_box/gbo_binding.json"; BOXCLO="$BD/box_closure.json"
  if [ "$DRY" != 1 ]; then
    for k2 in map box; do
      [ "$k2" = map ] && { R="$MAPREC"; C="$MAPCLO"; } || { R="$BOXREC"; C="$BOXCLO"; }
      [ -s "$R" ] && [ -s "$C" ] || die "no binding $k2 overfit record/closure under $BD (run 'bind' on THIS commit first; Colab/off-Thor runs are NON-binding)"
      py "$HERE/refcv8_assemble_argv.py" --assert-closure "$C" 9>&- 200>&- | grep -q '^ASSERT-OK' || die "$k2 closure not binding-clean: $C"
      py "$HERE/refcv8_assemble_argv.py" --assert-record "$R" --kind "$k2" 9>&- 200>&- | grep -q '^ASSERT-OK' || die "$k2 record not PASS: $R"
    done
    GSMJ="$(ls -1 "$OUTROOT"/gs_measure_*/gs_measure.json 2>/dev/null | tail -1)"
    [ -s "${GSMJ:-/nonexistent}" ] || die "no gs_measure.json: run 'gs-measure' on Thor (the launch schedule is derived from it)"
    py "$HERE/refcv8_assemble_argv.py" --audit "$ARGV" --gs-measure "$GSMJ" 9>&- 200>&- | grep -q '^AUDIT-OK' || die "the committed argv does not match the MEASURED gradient-share schedule ($GSMJ)"
  fi
  A+=(--map-overfit-record "$MAPREC" --map-overfit-closure "$MAPCLO" --box-overfit-record "$BOXREC" --box-overfit-closure "$BOXCLO")
  say "gate command: $PY $GATE ${A[*]}   (G-LIVE smokes THIS argv: same --log-every/--eval-every as the run; see RESULT B-GS-SMOKE for the grad-share rows)"
  [ "$DRY" = 1 ] && { echo "ZZGATE-DRYOKZZ"; exit 0; }
  py "$GATE" "${A[@]}" > "$WORK/gate.log" 2>&1 9>&- 200>&-          # stdout to a FILE: no pipe, no $? to trust
  # ---- assert on the ARTIFACT: exactly one current token per commit; verdict read from its JSON ----
  TOK="$GATE_OUT/PASS_$C12.json"
  V="$(pyval 'import json,glob,sys; fs=sorted(glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json")); d=json.load(open(fs[-1])) if fs else {}; print(d.get("verdict","NO-TOKEN"))' "$GATE_OUT" "$C12")"
  if [ "$V" != PASS ] || [ ! -s "$TOK" ]; then
    pyval 'import json,glob,sys; [print(" -",r[:300]) for f in glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json") for r in json.load(open(f)).get("reasons",[])[:12]]' "$GATE_OUT" "$C12"
    die "token verdict is '$V', not PASS (see $GATE_OUT)"; fi
  # ---- smoke reach: the memory note 'smoke the launch as launched' -- did the smoke reach each periodic instrument? ----
  MJ="$(find "$GATE_OUT" -name metrics.jsonl -size +0 2>/dev/null | head -1)"
  NE=0   # NOT `grep -c || echo 0` (that prints the zero twice)
  [ -n "$MJ" ] && NE="$(grep -c 'eval_' "$MJ")"
  say "smoke metrics: ${MJ:-NONE} eval rows=$NE (the gradient-share reading's reach is asserted by 'gs-measure', whose smoke carries --grad-share-every 50 --grad-share-at 50)"
  [ -n "$MJ" ] || die "INCONCLUSIVE: no smoke metrics.jsonl found under $GATE_OUT (cannot assert instrument reach)"
  [ "${NE:-0}" -ge 1 ] || die "the smoke wrote no in-run eval row"
  say "token $TOK is PASS"; echo "ZZGATE-PASSZZ";;
# ------------------------------------------------------------------------------------------------------ fire
fire)
  need CODE COMMIT
  C12="${COMMIT:0:12}"; GATE_OUT="$OUTROOT/gate_$C12"; WORK="$GATE_OUT/prelaunch"; ARGV="$CODE/stack/ops/runs.d/$ARM.argv.json"
  TOK="$GATE_OUT/PASS_$C12.json"; VJ="$WORK/verify.json"; ENVF="$OUTROOT/runs.d/$ARM.env"
  [ -s "$TOK" ] || die "no PASS token at $TOK"
  run rm -f "$VJ"; run py "$CODE/stack/scripts/launch_gate.py" verify --token "$TOK" --argv-file "$ARGV" --tree "$CODE" --commit "$COMMIT" --json "$VJ" 9>&- 200>&-
  [ "$DRY" = 1 ] || [ "$(pyval 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","NONE"))' "$VJ")" = MATCH ] || die "verify did not MATCH ($VJ)"
  run mkdir -p "$OUTROOT/runs.d"
  say "manifest -> $ENVF"
  if [ "$DRY" != 1 ]; then cat > "$ENVF" <<EOF
# $ARM -- launch manifest for stack/ops/sup_refcv7.sh (generated by refcv8_prelaunch.sh; sourced ONCE by the supervisor)
export CODE=$CODE
export COMMIT=$COMMIT
PYTHON=/home/nvidia/venvs/tanitad-train/bin/python
GATE_TOKEN=$TOK
GATE_ARGV_FILE=$ARGV
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
MAX_RELAUNCH=4
POLL_S=300
EOF
  fi
  say "starting supervisor (lock fds closed on the child; kill later by explicit pid from train.pid)"
  if [ "$DRY" = 1 ]; then say "DRY: nohup bash $CODE/stack/ops/sup_refcv7.sh $ARM $OUTROOT/runs.d > $OUTROOT/sup_$ARM.out 2>&1 9>&- 200>&- &"; echo "ZZFIRE-DRYOKZZ"; exit 0; fi
  nohup bash "$CODE/stack/ops/sup_refcv7.sh" "$ARM" "$OUTROOT/runs.d" > "$OUTROOT/sup_$ARM.out" 2>&1 9>&- 200>&- &
  for i in 1 2 3 4 5 6; do sleep 20 9>&- 200>&-; if grep -q 'ZZGATEOK-' "$OUTROOT/runs/$ARM/sup_$ARM.log" 2>/dev/null; then break; fi; done
  NS="$(ps -eo args | grep -c 'sup_refcv[7].sh refcv8-r101-s0')"
  [ "$NS" -ge 1 ] || die "no supervisor process after start (see $OUTROOT/sup_$ARM.out)"
  say "supervisor up; watch the log for ZZ${ARM}-<step>-<steps>-<errs>-<launch>ZZ"; echo "ZZFIRE-OKZZ";;
*) echo "usage: refcv8_prelaunch.sh [--dry-run] assemble|gate|fire ... (see header)"; exit 2;;
esac
