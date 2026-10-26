#!/usr/bin/env bash
# mapft_gsuite.sh -- DEV-BOX driver for the two suite artifacts the refcv8-mapft launch gate CONSUMES (modelled on the NT scripts
# final_suite.sh / final_pinned_f7.sh / wait_suite2.sh and LAUNCH_COMMANDS.md (1)+(2), 2026-10-22..25).
#
#   G-SUITE (Thor, CPU only, ~45 min, argv-INDEPENDENT -- bound to commit + tree):      --suite-verdict F
#       mapft_gsuite.sh plan         --tip T --cand C0 [--repo R]     build tree_tip.tar + overlay.tar + plan_thor.json (+ PLAN_CHECK.json; it cross-checks
#                                                                     TIP+overlay == git archive C0 for every path, and REFUSES deletions / mode changes)
#       mapft_gsuite.sh thor-start   --cand C0                        ship the three files (md5 at both ends) and start the runner (nice 19, CUDA hidden)
#       mapft_gsuite.sh thor-wait    --cand C0                        poll results/ALLDONE (opaque marker)
#       mapft_gsuite.sh thor-verdict --cand C0                        gate_verdict.py -> suite-bind -> mapft_gsuite_precheck.py; F = $ROOT/suite_<c12>/suite_bound.json
#   G-SUITE-PINNED (dev box, ~25 min, bound to commit + tree + ARGV SHA -> run it AFTER `mapft_prelaunch.sh budget` has made the final argv):  --devbox-evidence DIR
#       mapft_gsuite.sh devbox-pinned --cand C0 --argv-file <the fetched refcv8-mapft-s0.argv.json>   run, read the evidence, tar + scp + md5 to Thor
#   ANY: --stage DIR (default F:/refcv9_mapft/gate)  --dry-run (builds/prints only, touches nothing on Thor)
#
# Hard rules kept: Thor gets files by md5-verified scp only (never git); the Thor runner starts ONLY after NT A1's done-marker (it uses RAM, disk and CPU);
# every verdict is read from an ARTIFACT; the `ps | grep` liveness probe is in its OWN ssh call (a pattern inside the launching command self-matches it).
# Markers: ZZGS-<NAME>ZZ.
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MODE="${1:-}"; shift || true
TIP=""; CAND=""; REPO=C:/Users/Admin/tanitad-push; STAGE=F:/refcv9_mapft/gate; ARGVF=""; SKIPNT=0
while [ $# -gt 0 ]; do case "$1" in
  --tip) TIP="$2"; shift 2;; --cand) CAND="$2"; shift 2;; --repo) REPO="$2"; shift 2;; --stage) STAGE="$2"; shift 2;;
  --argv-file) ARGVF="$2"; shift 2;; --skip-nt-done) SKIPNT=1; shift;;
  *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done
HOST=tanitad-thor; ROOT="${MAPFT_ROOT:-/home/nvidia/r8ns/mapft}"; NT_SUM="${MAPFT_NT_SUMMARY:-/home/nvidia/r8ns/nt/runs/nt-a1-s0/summary.json}"
THOR_PY="${MAPFT_THOR_PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
# MAPFT_LOCAL=1 (dev-box REHEARSAL ONLY): the "Thor" is a local directory -- rsh runs the command in a local bash, rcp is cp.  Never set on launch day.
DEVPY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
PLANGEN=C:/Users/Admin/refcv8_pf/nt_build/trainer/code/make_thor_suite_plan.py
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MSYS_NO_PATHCONV=1
say() { echo "[mapft-gsuite:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZGS-REFUSEDZZ"; exit 10; }
rsh() { if [ "$DRY" = 1 ]; then say "DRY ssh: $1"; return 0; fi; if [ "${MAPFT_LOCAL:-0}" = 1 ]; then bash -c "$1"; return $?; fi; ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "$@"; }
rcp() { local d="${@: -1}"; local -a f=("${@:1:$#-1}"); if [ "${MAPFT_LOCAL:-0}" = 1 ]; then mkdir -p "$d" && cp -p "${f[@]}" "$d/"; else scp -q -o BatchMode=yes "${f[@]}" "$HOST:$d/"; fi; }
md5of() { md5sum "$1" 2>/dev/null | cut -c1-32; }
[ -n "$MODE" ] || { sed -n 2,19p "${BASH_SOURCE[0]}"; exit 2; }
if [ -n "$CAND" ]; then [ "${#CAND}" = 40 ] || die "--cand must be 40 hex chars"; C12="${CAND:0:12}"; G="$ROOT/suite_$C12"; NS="$STAGE/nts_$C12"; fi

case "$MODE" in
plan)
  [ -n "$TIP" ] && [ -n "$CAND" ] || die "usage: plan --tip T --cand C0 [--repo R]"
  [ "${#TIP}" = 40 ] || die "--tip must be 40 hex chars"
  mkdir -p "$STAGE" || die "cannot create $STAGE"
  rm -rf "$NS" 2>/dev/null
  "$DEVPY" -I "$PLANGEN" --repo "$REPO" --tip "$TIP" --cand "$CAND" --out "$NS" > "$STAGE/plan_$C12.log" 2>&1
  for f in tree_tip.tar overlay.tar plan_thor.json PLAN_CHECK.json; do [ -s "$NS/$f" ] || die "the plan generator produced no $f (see $STAGE/plan_$C12.log): $(tail -c 300 "$STAGE/plan_$C12.log" | tr '\n' ' ')"; done
  say "plan ok: $(tail -c 300 "$STAGE/plan_$C12.log" | tr '\n' ' ')"
  for f in tree_tip.tar overlay.tar plan_thor.json; do say "  $f md5 $(md5of "$NS/$f") $(stat -c %s "$NS/$f") bytes"; done
  echo "ZZGS-PLAN-OKZZ";;

thor-start)
  [ -n "$CAND" ] || die "usage: thor-start --cand C0"
  for f in tree_tip.tar overlay.tar plan_thor.json; do [ -s "$NS/$f" ] || die "no $NS/$f (run 'plan' first)"; done
  if [ "$DRY" != 1 ] && [ "$SKIPNT" != 1 ]; then   # NT A1's done-marker is the trigger: the suite uses RAM, disk and CPU
    rsh "grep -q '\"done\"[[:space:]]*:[[:space:]]*true' $NT_SUM && echo NTDONE" | tr -d '\r' | grep -q NTDONE || die "NT A1 has no done-marker ($NT_SUM): the suite would load a training box"
  fi
  WANT=""; for f in tree_tip.tar overlay.tar plan_thor.json; do WANT="$WANT $(md5of "$NS/$f")"; done
  rsh "mkdir -p $G" || die "ssh mkdir failed"
  if [ "$DRY" = 1 ]; then say "DRY scp: tree_tip.tar overlay.tar plan_thor.json -> $HOST:$G/"; else rcp "$NS/tree_tip.tar" "$NS/overlay.tar" "$NS/plan_thor.json" "$G" || die "scp failed"; fi
  GOT="$([ "$DRY" = 1 ] && echo "$WANT" || rsh "cd $G && md5sum tree_tip.tar overlay.tar plan_thor.json | cut -c1-32 | tr '\n' ' '" | tr -d '\r')"
  [ "$(echo $WANT)" = "$(echo $GOT)" ] || die "md5 mismatch after the ship: want [$WANT] got [$GOT]"
  # start the runner (the refcv8 G-SUITE (a) runner UNCHANGED); CPU only, lowest priority
  rsh "cp -p /home/nvidia/gate_fb47a583763f/thor_gate_runner2.py /home/nvidia/gate_fb47a583763f/gate_verdict.py $G/ && cd $G && (nohup nice -n 19 $THOR_PY $G/thor_gate_runner2.py $G /home/nvidia/refcv8_audit/pyt > $G/runner.out 2> $G/runner.err < /dev/null &) ; sleep 2" || die "could not start the runner"
  # liveness = an ARTIFACT (the runner writes <stage>.json as it goes) within 3 min, and a process count from a SEPARATE ssh call
  ok=0; for i in 1 2 3 4 5 6; do
    if [ "$DRY" = 1 ]; then ok=1; break; fi
    rsh "ls $G/results/TIP.json $G/results/CAND.json 2>/dev/null | wc -l" | tr -d '\r' | grep -q '^[12]$' && { ok=1; break; }
    sleep 30; done
  [ "$ok" = 1 ] || die "the runner wrote no progress file under $G/results in 3 min (see $G/runner.err)"
  [ "$DRY" = 1 ] || say "runner processes: $(rsh "ps -eo args | grep -c 'thor_gate_runne[r]2.py'" | tr -d '\r')"
  say "runner started for $C12 (about 45 min, CPU only)"; echo "ZZGS-STARTED-OKZZ";;

thor-wait)
  [ -n "$CAND" ] || die "usage: thor-wait --cand C0"
  t0=$(date +%s)
  while :; do
    r="$(rsh "a=0; [ -e $G/results/ALLDONE ] && a=1; n=\$(ls $G/results/CAND 2>/dev/null | wc -l); echo \"\$a-\$n\"" 2>/dev/null | tr -d '\r')"
    case "$r" in 1-*) say "ALLDONE $r"; echo "ZZGS-ALLDONE-${r}ZZ"; exit 0;; esac
    [ "$DRY" = 1 ] && { echo "ZZGS-WAIT-DRYZZ"; exit 0; }
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && die "no ALLDONE after 2 h ($r)"
    sleep 120
  done;;

thor-verdict)
  [ -n "$CAND" ] || die "usage: thor-verdict --cand C0"
  [ "$DRY" = 1 ] && { say "DRY: gate_verdict.py -> suite-bind --verdict ... --cand-tree $G/CAND --commit $CAND --out $G/suite_bound.json -> mapft_gsuite_precheck.py $ROOT/launch_tree"; echo "ZZGS-VERDICT-DRYZZ"; exit 0; }
  rsh "for f in TIP CAND; do grep -q '\"done\"[[:space:]]*:[[:space:]]*true' $G/results/\$f.json && echo \"\$f-done\"; done; [ -e $G/results/ALLDONE ] && echo ALLDONE" | tr -d '\r' > "$STAGE/verdict_pre_$C12.txt"
  [ "$(grep -c -E '^(TIP-done|CAND-done|ALLDONE)$' "$STAGE/verdict_pre_$C12.txt")" = 3 ] || die "the suite is not finished (need TIP-done, CAND-done, ALLDONE): $(tr '\n' ' ' < "$STAGE/verdict_pre_$C12.txt")"
  rsh "mkdir -p $G/v/res && ln -sfn $G/results $G/v/res/results && cd $G && $THOR_PY gate_verdict.py $G/v > $G/verdict.out 2>&1" || true
  V="$(rsh "$THOR_PY -c 'import json; d=json.load(open(\"$G/v/verdict.json\")); print(\"VERDICT\", d.get(\"verdict\", \"NONE\"), \"regressions\", len(d.get(\"regressions\", [])), \"new_fail\", len(d.get(\"new_fail\", [])))'" 2>/dev/null | tr -d '\r' | grep '^VERDICT')"
  say "gate_verdict: $V"
  echo "$V" | grep -q '^VERDICT PASS regressions 0 new_fail 0' || die "the suite verdict is not a clean PASS: $V (read $G/v/verdict.json on Thor)"
  rsh "cd $G/CAND && PYTHONPATH=$G/CAND/stack $THOR_PY stack/scripts/launch_gate.py suite-bind --verdict $G/v/verdict.json --cand-tree $G/CAND --commit $CAND --out $G/suite_bound.json > $G/suite_bind.out 2>&1" || true
  rsh "grep -c 'ZZSUITE-BOUND-${C12}ZZ' $G/suite_bind.out; test -s $G/suite_bound.json && echo BOUND-FILE" | tr -d '\r' | tr '\n' ' ' | grep -q '^1 BOUND-FILE' || die "suite-bind produced no bound file (read $G/suite_bind.out on Thor)"
  rsh "test -s $ROOT/kit/mapft_gsuite_precheck.py && echo PRECHECK-PRESENT" | tr -d '\r' | grep -q PRECHECK-PRESENT || die "no $ROOT/kit/mapft_gsuite_precheck.py on Thor (mapft_ship.sh kit ships it)"
  rsh "cd /tmp && $THOR_PY $ROOT/kit/mapft_gsuite_precheck.py $ROOT/launch_tree $G/suite_bound.json $CAND refcv8-mapft > $G/precheck.out 2>&1; tail -1 $G/precheck.out" | tr -d '\r' > "$STAGE/precheck_$C12.txt"
  grep -q '^G-SUITE PRECHECK: PASS' "$STAGE/precheck_$C12.txt" || die "G-SUITE precheck did not PASS ($(cat "$STAGE/precheck_$C12.txt")): the launch tree and the suite's candidate tree differ, or a failure is outside the pinned set -- read $G/precheck.out"
  say "F = $G/suite_bound.json  (pass to:  mapft_prelaunch.sh gate --suite-verdict $G/suite_bound.json ...)"; echo "ZZGS-SUITE-BOUND-OKZZ";;

devbox-pinned)
  [ -n "$CAND" ] && [ -n "$ARGVF" ] || die "usage: devbox-pinned --cand C0 --argv-file <final argv json>"
  [ -s "$ARGVF" ] || die "no argv file $ARGVF"
  T="$STAGE/ntlt_$C12"; SW="$STAGE/gsw"; OUT="$STAGE/devbox_evidence_${C12}"; LOG="$STAGE/devbox_pinned_$C12.log"; mkdir -p "$STAGE" "$SW" || die "cannot create $STAGE"
  if [ ! -s "$T/stack/scripts/launch_gate.py" ]; then
    mkdir -p "$T" && git --git-dir="$REPO/.git" -c core.autocrlf=false archive "$CAND" stack taniteval | tar -x -C "$T" || die "git archive $CAND failed"
  fi
  [ -s "$T/stack/scripts/launch_gate.py" ] || die "the extracted tree has no launch_gate.py"
  AH="$("$DEVPY" -I -c "import json,hashlib,sys; d=json.load(open(sys.argv[1])); a=d['argv'] if isinstance(d,dict) else d; print(hashlib.sha256(json.dumps(a,separators=(',',':')).encode()).hexdigest())" "$ARGVF")"
  say "argv sha256 ${AH:0:16}...  tree $T  work $SW  out $OUT"
  [ "$DRY" = 1 ] && { say "DRY: launch_gate.py run --profile refcv8-mapft --stage devbox --tree $T --commit $CAND --git-dir $REPO/.git --argv-file $ARGVF --suite-work $SW --out-dir $OUT (timeout 5400)"; echo "ZZGS-PINNED-DRYZZ"; exit 0; }
  # the NT env (final_pinned_f7.sh): the CUDA-gated tests, the lidar-BEV GT dir, the calibration dir; >= 6 GB free commit (the Windows COMMIT LIMIT bit PILOT-S)
  export TANITAD_REFAV1_CUDA_TEST=1 TANITAD_LIDAR_BEV_GT_DIR=E:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913 TANITAD_CALIB_DIR=E:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration
  t0=$(date +%s); until [ "$(powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r')" -ge 6 ] 2>/dev/null; do
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && die "free commit stayed below 6 GB for 1 h"; sleep 30; done
  rm -rf "$OUT" 2>/dev/null
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T/stack;$T/stack/scripts" timeout 5400 "$DEVPY" "$T/stack/scripts/launch_gate.py" run \
    --profile refcv8-mapft --stage devbox --tree "$T" --commit "$CAND" --git-dir "$REPO/.git" \
    --argv-file "$ARGVF" --suite-work "$SW" --out-dir "$OUT" > "$LOG" 2>&1
  EV="$OUT/evidence/G-SUITE-PINNED.json"; [ -s "$EV" ] || EV="$OUT/G-SUITE-PINNED.json"
  ST="$([ -s "$EV" ] && "$DEVPY" -I -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8')).get('status'))" "$EV" 2>/dev/null)"
  say "G-SUITE-PINNED evidence: ${ST:-MISSING} ($EV)"
  [ "$ST" = PASS ] || die "the dev-box pinned suite did not produce a PASS record (tail of $LOG: $(tail -c 300 "$LOG" | tr '\n' ' '))"
  # --force-local: GNU tar reads an archive name with a colon (F:/...) as host:path and refuses "Cannot connect to F"
  # (MEASURED 2026-10-27 00:14, the first time a PASS reached this line; the operator shipped by hand)
  TAR="$STAGE/ev_${C12}.tar"; tar --force-local -cf "$TAR" -C "$OUT" . || die "tar of the evidence failed"
  LM="$(md5of "$TAR")"; D="$ROOT/devbox_evidence_$C12"
  rcp "$TAR" "$ROOT" || die "scp of the evidence failed"
  RM="$(rsh "md5sum $ROOT/ev_${C12}.tar | cut -c1-32" | tr -d '\r')"; [ "$RM" = "$LM" ] || die "evidence md5 mismatch local $LM remote $RM"
  rsh "mkdir -p $D && tar -xf $ROOT/ev_${C12}.tar -C $D && ls $D/evidence/G-SUITE-PINNED.json $D/G-SUITE-PINNED.json 2>/dev/null | head -1" | tr -d '\r' | grep -q G-SUITE-PINNED || die "the evidence is not readable on Thor under $D"
  say "evidence on Thor: $D   (pass to:  mapft_prelaunch.sh gate --devbox-evidence $D ...)"; echo "ZZGS-PINNED-OKZZ";;
*) echo "usage: mapft_gsuite.sh [--dry-run] plan|thor-start|thor-wait|thor-verdict|devbox-pinned ... (see header)"; exit 2;;
esac
