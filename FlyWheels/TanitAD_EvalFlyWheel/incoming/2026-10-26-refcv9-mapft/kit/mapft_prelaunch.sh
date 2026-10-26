#!/usr/bin/env bash
# mapft_prelaunch.sh -- THOR-SIDE launch kit for the refcv8 MAP-ONLY FINE-TUNE (MAPFT).  PREREG_MAPFT sealed f8774a7b...
# PI D-R9-MAPHEAD-PRECISION (a), 2026-10-26: cap 6 Thor-hours, Thor after NT A1, 50-step timing smoke FIRST, stop if over the cap.
# Modelled on stack/scripts/nt_prelaunch.sh (GPU-lock protocol, artifact-read verdicts, ZZ markers) and refcv8_prelaunch.sh
# (refcv8's bind templates + box audit: MAPFT trains on refcv8's own corpus, not the NavSim one).
#
#   THOR (this file lives at $ROOT/kit/ after mapft_ship.sh kit; $ROOT = /home/nvidia/r8ns/mapft):
#     mapft_prelaunch.sh preflight                      READ-ONLY: is NT A1 done, is the GPU empty, do the inputs exist
#     mapft_prelaunch.sh smoke   [--code DIR]           50-step TIMING smoke (GPU lock) -> $ROOT/smoke/SMOKE_RESULT.json
#     mapft_prelaunch.sh budget  [--code DIR]           PREREG_MAPFT s1 rule -> $ROOT/argv/refcv8-mapft-s0.argv.json  | STOP (exit 11)
#     mapft_prelaunch.sh bind    --commit SHA40         BINDING G-MAP-OVERFIT then G-BOX-OVERFIT for THIS argv (GPU, ~34 + ~82 min INHERITED)
#     mapft_prelaunch.sh gate    --commit SHA40 --devbox-evidence DIR --suite-verdict F [--pi-cost-approval F]
#                                                       launch_gate.py run --profile refcv8-mapft --stage thor -> PASS_<c12>.json
#     mapft_prelaunch.sh redarms --commit SHA40 ...     the TWO RED arms (mapft_planner_trainable, mapft_map_frozen) must NOT pass
#     mapft_prelaunch.sh fire    --commit SHA40 [--zerograd-off]  the MASTER MIND's step: verify == MATCH, zerograd rehearsal PASS, manifest, sup_refcv7.sh
#   ANY: mapft_prelaunch.sh --dry-run <mode> --root <scratch> ...   prints what it WOULD do; takes no lock, touches no GPU.
#
# Rules kept (each cost a run, see CLAUDE.md): every verdict is read from an ARTIFACT (token / record / closure / JSON), never from
# $? and never through a pipe; every child closes the lock fds (9, 200); no `pgrep -f` / `pkill -f` (self-matching); the markers
# are opaque ZZ...ZZ tokens disjoint from anything grepped; the GPU is never shared (lock + EMPTY compute-apps list).
# ⛔ THOR GPU PROTOCOL (MM 2026-10-21): every GPU step holds /home/nvidia/refcv7_post/thor_gpu.lock AND finds
# `nvidia-smi --query-compute-apps` EMPTY under it.  A timing measurement is meaningful ONLY with the GPU exclusive
# (memory thor-gpu-sharing-is-net-negative: 3 jobs ran 3.9x slower EACH).
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MODE="${1:-}"; shift || true
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
# dev-box dry-runs only: hand a native path to python for any argument that EXISTS here (no-op on Thor, where cygpath is absent)
py() { local a=() x; for x in "$@"; do
  if command -v cygpath >/dev/null 2>&1 && case "$x" in /*) [ -e "$x" ] || [ -d "$(dirname "$x")" ];; *) false;; esac; then
    a+=("$(cygpath -m "$x")"); else a+=("$x"); fi; done; "$PY" "${a[@]}"; }
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/nvidia/r8ns/mapft; ARM=refcv8-mapft-s0; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
NT=/home/nvidia/r8ns/nt; NT_RUN=$NT/runs/nt-a1-s0; NT_PENDING=$NT/NT_LAUNCH_PENDING
INIT=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt; INIT_BYTES=403200712; INIT_MD5=b4364029b68f29a33eeb4aebbbbd002b
V9_MD5=83e30bd3791f0767707eade5f4671759; V9_EVAL_MD5=6b5c7f207cffc3b7eb3cd527fd433599
CODE=""; COMMIT=""; DEVEV=""; SUITE=""; PICOST=""; SKIPNT=0; CHECKMD5=0; ZGOFF=0
# the post-run zero-gradient gate (sup_refcv7.sh G5) reads the OPTIMIZER state; the positive control = at least one TRAINED map-only parameter
MAPBF=""
ZG_CONTROL='_map_hires\.(refine|near|near_refine)\..*'
while [ $# -gt 0 ]; do case "$1" in
  --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;;
  --devbox-evidence) DEVEV="$2"; shift 2;; --suite-verdict) SUITE="$2"; shift 2;; --pi-cost-approval) PICOST="$2"; shift 2;;
  --skip-nt-done) SKIPNT=1; shift;; --check-init-md5) CHECKMD5=1; shift;;
  --zerograd-off) ZGOFF=1; shift;;
  --map-bind-from) MAPBF="$2"; shift 2;;   # C1 (MM option A): reuse a prior bind dir's MAP record+closure; the gate re-asserts the closure against THIS tree
  *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done
CODE="${CODE:-$ROOT/launch_tree}"
SARGV="$ROOT/argv/mapft_argv_smoke.json"; ARGV="$ROOT/argv/$ARM.argv.json"; SD="$ROOT/smoke"; SRES="$SD/SMOKE_RESULT.json"

say() { echo "[mapft-prelaunch:$MODE] $*"; }
die() { say "REFUSED: $*"; echo "ZZMAPFTPRELAUNCH-REFUSEDZZ"; exit 10; }
need() { for v in "$@"; do [ -n "${!v}" ] || die "--$(echo "$v" | tr 'A-Z_' 'a-z-') is required for '$MODE'"; done; }
pyval() { py -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }            # a value, never a status
argv_get() { pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(a[a.index(sys.argv[2])+1] if sys.argv[2] in a else "")' "$1" "$2"; }
argv_lines() { pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print("\n".join(a))' "$1"; }
gpu_take() {  # the protocol: a blocking lock, then an EMPTY GPU.  The yield marker NT left for the L7 chain stays in place
  if [ "$DRY" = 1 ]; then say "DRY: touch $NT_PENDING; flock $LOCKF (blocking, <= 2 h); assert nvidia-smi compute-apps EMPTY"; return 0; fi
  touch "$NT_PENDING" 9>&- 200>&- || die "cannot write the yield marker $NT_PENDING"
  exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
  NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
  [ "$NAPPS" = 0 ] || die "the GPU is not empty under the lock ($NAPPS compute app(s)): never share the GPU"
}
grade_arm() {   # $1 arm  $2 dir holding the arm run's token  $3 the arm-free MAIN token (the CONTROL) -> one ARM-... line, read from the TOKENS
  pyval 'import json,glob,sys
arm,d,main=sys.argv[1],sys.argv[2],sys.argv[3]
def tok_of(path):
    t=None
    for f in sorted(glob.glob(path)):
        try: j=json.load(open(f))
        except Exception: continue
        if j.get("kind")=="launch-gate-token": t=j
    return t
tok=tok_of(d+"/*.json"); ctl=tok_of(main)
if tok is None or ctl is None: print("ARM-"+arm+"-UNGRADED(no arm token or no control token)"); raise SystemExit
ck=tok.get("checks",{}); cc=ctl.get("checks",{})
bad=[c for c in ("G-DVB","G-LIVE") if ck.get(c,{}).get("status") in ("FAIL","ERROR")]
# CAUGHT = the arm token is not PASS, one of the two checks is RED WITH A MAP-ONLY REASON of its own, and the arm-free MAIN token read PASS on that same check
sig=[c for c in bad if any("map-only" in r.lower() for r in ck[c].get("reasons",[]))]
ctl_ok=all(cc.get(c,{}).get("status")=="PASS" for c in sig) and bool(sig)
print("ARM-"+arm+("-CAUGHT" if (tok.get("verdict")!="PASS" and sig and ctl_ok) else "-ESCAPED-OR-MISCREDITED") + " verdict=" + str(tok.get("verdict")) + " red_checks=" + ",".join(bad) + " map_only_reason_on=" + ",".join(sig) + " control_pass=" + str(ctl_ok))' "$1" "$2" "$3"
}
nt_finished() {   # NT A1's supervisor/trainer DONE-MARKER is the trigger (CLAUDE.md: the marker, not a guess)
  [ -s "$NT_RUN/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$NT_RUN/summary.json" 2>/dev/null
}
[ -n "$MODE" ] || { sed -n 3,17p "${BASH_SOURCE[0]}"; exit 2; }

# =========================================================================================================== preflight
run_preflight() {   # prints PF-OK / PF-FAIL lines to $1 ; the caller counts the FAILs
  local PF="$1" f; : > "$PF"; pf() { echo "$1 $2" >> "$PF"; }
  [ "$SKIPNT" = 1 ] && pf PF-OK "nt-a1-done(SKIPPED by --skip-nt-done)" || { nt_finished && pf PF-OK nt-a1-done || pf PF-FAIL "nt-a1-done:no summary.json{done:true} at $NT_RUN"; }
  local NTP; NTP="$(ps -eo args 2>/dev/null | grep -c 'refc_v3_trai[n]')"
  [ "$SKIPNT" = 1 ] || { [ "$NTP" = 0 ] && pf PF-OK no-trainer-process || pf PF-FAIL "trainer-process-count:$NTP"; }
  local NS; NS="$(ps -eo args 2>/dev/null | grep -c 'sup_refcv[7].sh')"
  [ "$SKIPNT" = 1 ] || { [ "$NS" = 0 ] && pf PF-OK no-supervisor-process || pf PF-FAIL "supervisor-process-count:$NS (a live supervisor may RESURRECT a run)"; }
  if [ "$DRY" != 1 ]; then
    local NA; NA="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
    [ "$NA" = 0 ] && pf PF-OK gpu-empty || pf PF-FAIL "gpu-compute-apps:$NA"
    [ -f "$LOCKF" ] && pf PF-OK gpu-lock-file || pf PF-FAIL gpu-lock-file
    local SZ; SZ="$(stat -c %s "$INIT" 2>/dev/null)"
    [ "$SZ" = "$INIT_BYTES" ] && pf PF-OK "init-from-size:$SZ" || pf PF-FAIL "init-from-size:${SZ:-missing} (want $INIT_BYTES) $INIT"
    if [ "$CHECKMD5" = 1 ]; then M="$(md5sum "$INIT" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = "$INIT_MD5" ] && pf PF-OK init-from-md5 || pf PF-FAIL "init-from-md5:${M:-unreadable}"
    else pf PF-OK "init-from-md5(deferred: the launch gate md5s it by CONTENT; pass --check-init-md5 to do it now)"; fi
    "$PY" -c "import torch, timm" > /dev/null 2>&1 9>&- 200>&- && pf PF-OK python-imports-torch-timm || pf PF-FAIL "python-imports-torch-timm:$PY"
    local FREE; FREE="$(df -BG --output=avail "$(dirname "$ROOT")" 2>/dev/null | tail -1 | tr -dc 0-9)"
    [ "${FREE:-0}" -ge 60 ] && pf PF-OK "disk-60G(${FREE}G)" || pf PF-FAIL "disk-60G(${FREE:-?}G)"
  else pf PF-OK "(dry-run: Thor-only probes skipped: gpu, lock, init size, imports, disk)"; fi
  # every absolute data path of the refcv8 launch argv exists (metadata only; the gate md5s/sha256s what it binds)
  if [ -f "$KIT/refcv8_launch_argv_base.json" ]; then
    local MISS=0 N=0 p
    for p in $(pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"]; print("\n".join(x for x in a if x.startswith("/home/nvidia/") and "/runs/" not in x and "launch_tree" not in x and "refcv7_run" not in x))' "$KIT/refcv8_launch_argv_base.json"); do
      N=$((N+1)); [ "$DRY" = 1 ] || { [ -e "$p" ] || { MISS=$((MISS+1)); pf PF-FAIL "argv-path-missing:$p"; }; }; done
    [ "$MISS" = 0 ] && pf PF-OK "argv-data-paths($N)"
  else pf PF-FAIL "no $KIT/refcv8_launch_argv_base.json"; fi
}
case "$MODE" in
preflight)
  mkdir -p "$ROOT/logs" 2>/dev/null; PF="$ROOT/logs/preflight_$(date +%H%M%S).txt"; [ "$DRY" = 1 ] && PF="${TMPDIR:-/tmp}/mapft_preflight_dry.txt"
  run_preflight "$PF"; cat "$PF"; NF="$(grep -c '^PF-FAIL' "$PF")"
  [ "$NF" = 0 ] || die "$NF preflight failure(s)"
  say "preflight clean"; echo "ZZMAPFTPREFLIGHT-OKZZ";;
# =========================================================================================================== smoke
smoke)
  [ -f "$SARGV" ] || die "no smoke argv at $SARGV (ship the kit: mapft_ship.sh kit)"
  [ -f "$CODE/stack/scripts/refc_v3_train.py" ] || die "no trainer under --code $CODE"
  SOUT="$(argv_get "$SARGV" --out)"; [ -n "$SOUT" ] || die "the smoke argv carries no --out"
  mkdir -p "$SD" 2>/dev/null
  [ "$DRY" = 1 ] || [ ! -e "$SOUT/metrics.jsonl" ] || die "$SOUT/metrics.jsonl already exists: a timing smoke is ONE run (move it aside)"
  argv_lines "$SARGV" > "$SD/smoke.args" || die "cannot read $SARGV"
  mapfile -t SARGS < "$SD/smoke.args"
  [ "${SARGS[0]:-}" = "--arm" ] || die "smoke argv does not start with --arm"
  if [ "$DRY" = 1 ]; then say "DRY: gpu_take; (cd $CODE/stack && PYTHONPATH=$CODE/stack:$CODE/taniteval OMP_NUM_THREADS=8 $PY $CODE/stack/scripts/refc_v3_train.py <${#SARGS[@]} smoke args>) timed with date +%s; then make_mapft_argv.py derive --from-smoke-metrics $SOUT/metrics.jsonl --smoke-wall-s <wall>"; echo "ZZMAPFTSMOKE-DRYOKZZ"; exit 0; fi
  gpu_take
  { echo "gpu before: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"; date -u +%FT%TZ; } > "$SD/gpu_before.txt" 2>&1
  T0=$(date +%s)
  ( cd "$CODE/stack" && export PYTHONPATH="$CODE/stack:$CODE/taniteval" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token &&
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN &&
    "$PY" "$CODE/stack/scripts/refc_v3_train.py" "${SARGS[@]}" > "$SD/train.log" 2> "$SD/train.stderr.log" 9>&- 200>&- )
  T1=$(date +%s); WALL=$((T1-T0))
  { echo "gpu after: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"; } >> "$SD/gpu_before.txt" 2>&1
  # the verdict is the ARTIFACT: the metrics file must REACH step 50 (a crash, an OOM or a FrozenStateGuard refusal does not)
  [ -s "$SOUT/metrics.jsonl" ] || die "no metrics.jsonl at $SOUT (see $SD/train.log, $SD/train.stderr.log)"
  LAST="$(pyval 'import json,sys
m=-1
for l in open(sys.argv[1],encoding="utf-8"):
    try: r=json.loads(l)
    except Exception: continue
    if isinstance(r,dict) and isinstance(r.get("step"),int) and "loss" in r: m=max(m,r["step"])
print(m)' "$SOUT/metrics.jsonl")"
  [ "$LAST" = 50 ] || die "the smoke reached step '$LAST', not 50 (see $SD/train.log)"
  # DRESS REHEARSAL of the supervisor's post-run zero-gradient gate (G5) on a MAP-ONLY checkpoint -- never exercised on one before: the optimizer
  # holds ONLY the 105,032 map parameters, so every registered parameter must have a non-zero state, and the positive control must match one.
  ZGJ="$SD/zerograd_smoke.json"; rm -f "$ZGJ"
  if [ -s "$SOUT/ckpt.pt" ]; then
    ( cd "$CODE/stack" && PYTHONPATH="$CODE/stack:$CODE/taniteval" CUDA_VISIBLE_DEVICES="" "$PY" "$CODE/stack/scripts/zerograd_gate.py" --ckpt "$SOUT/ckpt.pt" --control "$ZG_CONTROL" --json-out "$ZGJ" > "$SD/zerograd_smoke.txt" 2>&1 9>&- 200>&- )
    ZGV="$(pyval 'import json,sys
d=json.load(open(sys.argv[1])); o=d.get("optimizer",{}); c=d.get("controls",[])
ok = d.get("verdict")=="PASS" and o.get("n_trained",0)>=1 and o.get("n_case_a_no_state",1)==0 and o.get("n_case_b_zero_state",1)==0 and len(c)>=1 and all(x.get("ok") for x in c)
print(("PASS" if ok else "NOT-PASS:"+str(d.get("verdict"))), "registered=%s trained=%s noState=%s zeroState=%s controls=%s" % (o.get("n_registered"), o.get("n_trained"), o.get("n_case_a_no_state"), o.get("n_case_b_zero_state"), [bool(x.get("ok")) for x in c]))' "$ZGJ" 2>/dev/null)"
  else ZGV="NOT-PASS:no ckpt.pt at $SOUT (the smoke saved nothing)"; fi
  [ -n "$ZGV" ] || ZGV="NOT-PASS:no readable report $ZGJ"
  say "zerograd dress rehearsal on the smoke ckpt: $ZGV"; echo "$ZGV" > "$SD/ZEROGRAD_SMOKE.txt"
  py "$KIT/make_mapft_argv.py" derive --from-smoke-metrics "$SOUT/metrics.jsonl" --smoke-wall-s "$WALL" > "$SD/derive.txt" 2>&1 9>&- 200>&-
  pyval 'import json,sys; json.dump({"smoke_wall_s": int(sys.argv[2]), "metrics": sys.argv[1], "last_step": 50, "t0": int(sys.argv[3]), "t1": int(sys.argv[4]), "host": __import__("platform").node()}, open(sys.argv[5],"w"), indent=1)' "$SOUT/metrics.jsonl" "$WALL" "$T0" "$T1" "$SRES" > /dev/null
  [ -s "$SRES" ] || die "could not write $SRES"
  say "smoke wall ${WALL}s; $(tr -d '\r' < "$SD/derive.txt" | grep -E 'WARN|NOTE' | head -3 | tr '\n' ' ')"
  if [ "$(grep -c 'ZZMAPFT-STOPZZ' "$SD/derive.txt")" = 1 ]; then say "$(grep 'ZZMAPFT-STOPZZ' "$SD/derive.txt" | tr -d '\r')"; die "STOP (PREREG_MAPFT s1): N < 1000 -- do NOT launch; report s_step to the PI (cap 6 Thor-h). Details: $SD/derive.txt"; fi
  [ "$(grep -c '^ZZMAPFT-N-[0-9]*ZZ' "$SD/derive.txt")" = 1 ] || die "no budget marker in $SD/derive.txt"
  say "$(grep -o '^ZZMAPFT-N-[0-9]*ZZ' "$SD/derive.txt" | tr -d '\r')  (s_step / N details: $SD/derive.txt)"
  echo "ZZMAPFTSMOKE-OKZZ";;
# =========================================================================================================== budget
budget)
  [ -s "$SRES" ] || die "no $SRES: run 'smoke' first"
  W="$(pyval 'import json,sys; print(json.load(open(sys.argv[1]))["smoke_wall_s"])' "$SRES")"
  M="$(pyval 'import json,sys; print(json.load(open(sys.argv[1]))["metrics"])' "$SRES")"
  [ -n "$W" ] && [ -s "$M" ] || [ "$DRY" = 1 ] || die "SMOKE_RESULT.json names no readable metrics"
  mkdir -p "$ROOT/argv" 2>/dev/null
  [ "$DRY" = 1 ] && ARGV="${TMPDIR:-/tmp}/$ARM.argv.dryrun.json"
  [ "$DRY" = 1 ] || [ ! -e "$ARGV" ] || die "$ARGV already exists (one budget per smoke; move it aside deliberately)"
  py "$KIT/make_mapft_argv.py" build --from-smoke-metrics "$M" --smoke-wall-s "$W" --base "$KIT/refcv8_launch_argv_base.json" \
      --out "$ARGV" --launch-tree "$CODE" --run-out "$ROOT/runs/$ARM" --init-from "$INIT" > "$SD/build.txt" 2>&1 9>&- 200>&-
  tr -d '\r' < "$SD/build.txt" | grep -v '^ZZ' | head -6
  if [ "$(grep -c 'ZZMAPFT-STOPZZ' "$SD/build.txt")" = 1 ]; then say "$(grep 'ZZMAPFT-STOPZZ' "$SD/build.txt" | tr -d '\r')"; echo "ZZMAPFTBUDGET-STOPZZ"; exit 11; fi
  [ "$(grep -c '^ZZMAPFT-BUILT-[0-9]*ZZ' "$SD/build.txt")" = 1 ] && [ -s "$ARGV" ] || die "no built argv at $ARGV (see $SD/build.txt)"
  # re-derive from the same two numbers and compare (DETERMINISM, not correctness: the formula's correctness is the literal unit tests')
  py "$KIT/make_mapft_argv.py" audit --argv "$ARGV" --from-smoke-metrics "$M" --smoke-wall-s "$W" > "$SD/audit.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^AUDIT-OK' "$SD/audit.txt")" = 1 ] || die "argv audit failed: $(tr -d '\r' < "$SD/audit.txt" | head -3 | tr '\n' ' ')"
  say "argv -> $ARGV (file sha256 $(sha256sum "$ARGV" | cut -c1-16)...). NEXT: bind --commit <C0>, then gate."; echo "ZZMAPFTBUDGET-OKZZ";;
# =========================================================================================================== bind
bind)
  need COMMIT; [ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
  C12="${COMMIT:0:12}"; BD="$ROOT/bind_$C12"; HERE="$CODE/stack/scripts"
  [ -f "$ARGV" ] || die "no launch argv at $ARGV (run 'budget')"
  [ -f "$HERE/refcv8_assemble_argv.py" ] || die "no refcv8_assemble_argv.py under $HERE"
  mkdir -p "$BD" 2>/dev/null
  # the BOX harness READS its prereg + frame set from {OUT}/audit (refcv8_prelaunch.sh bind): ship copies, md5 LITERALS (blob md5s)
  BAUD="$BD/run_box/audit/raw"; BSRC="$CODE/stack/ops/refcv8_box_audit/raw"
  if [ "$DRY" != 1 ]; then
    mkdir -p "$BAUD/visibility" 2>/dev/null
    cp -p "$BSRC/PREREG_G_BOX_OVERFIT.md" "$BAUD/" && cp -p "$BSRC/visibility/gbo_frameset.json" "$BAUD/visibility/" || die "could not prepare the box audit inputs from $BSRC"
    echo "594c71196cc5bbd527fee40b2cb0e3f1  $BAUD/PREREG_G_BOX_OVERFIT.md" | md5sum -c - > /dev/null 2>&1 || die "box audit prereg md5 (a CRLF-converted export? use git -c core.autocrlf=false archive)"
    echo "b291404c36f83c3e397e3b90367e8e7b  $BAUD/visibility/gbo_frameset.json" | md5sum -c - > /dev/null 2>&1 || die "box audit frame-set md5 (a CRLF-converted export?)"
  fi
  ASHA="$(pyval 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(hashlib.sha256(json.dumps(a,separators=(",",":")).encode()).hexdigest())' "$ARGV")"
  [ "${#ASHA}" = 64 ] || die "could not hash the launch argv"
  for K2 in map box; do
    TPL="$HERE/refcv8_bind_$K2.args.tpl"; [ -f "$TPL" ] || die "no harness template $TPL"
    py "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
        --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$K2" --sub "INPUTS=${BIND_INPUTS:-/home/nvidia/refcv7_bind/run_$K2/inputs}" \
        --args-out "$BD/$K2.args" --data-roots-out "$BD/$K2.roots" > "$BD/$K2.render.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^RENDERED' "$BD/$K2.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$K2.render.txt)"
  done
  if [ "$DRY" = 1 ]; then say "DRY: gpu_take; map ($(wc -l < "$BD/map.args") args) then box ($(wc -l < "$BD/box.args") args) via closure_run.py --binding for argv sha ${ASHA:0:16}"; echo "ZZMAPFTBIND-DRYOKZZ"; exit 0; fi
  gpu_take
  for K2 in map box; do
    mapfile -t HARGS < "$BD/$K2.args"; mapfile -t ROOTS < "$BD/$K2.roots"; DR=(); for r in "${ROOTS[@]}"; do [ -n "$r" ] && DR+=(--data-root "$r"); done
    mkdir -p "$BD/run_$K2" 2>/dev/null
    if [ "$K2" = map ]; then REC="$BD/run_map/g_map_overfit.json"; else REC="$BD/run_box/gbo_binding.json"; fi
    ( cd "$CODE" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
      "$PY" "$CODE/stack/scripts/closure_run.py" --binding --commit "$COMMIT" --tree "$CODE" \
        --out "$BD/${K2}_closure.json" --result "$REC" "${DR[@]}" -- "${HARGS[@]}" > "$BD/$K2.out" 2>&1 9>&- 200>&- )
    py "$HERE/refcv8_assemble_argv.py" --assert-closure "$BD/${K2}_closure.json" > "$BD/$K2.closure.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.closure.txt")" = 1 ] || die "$K2 closure refused: $(tr -d '\r' < "$BD/$K2.closure.txt" | head -3 | tr '\n' ' ')"
    py "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$K2" > "$BD/$K2.record.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.record.txt")" = 1 ] || die "$K2 record refused: $(tr -d '\r' < "$BD/$K2.record.txt" | head -3 | tr '\n' ' ')"
    say "$K2 binding run asserted: closure $BD/${K2}_closure.json, record $REC"
  done
  echo "ZZMAPFTBIND-OKZZ";;
# =========================================================================================================== gate / redarms (shared preflight + argument list)
gate|redarms)
  need COMMIT; [ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
  C12="${COMMIT:0:12}"; GATE_OUT="$ROOT/gate_$C12"; WORK="$GATE_OUT/prelaunch"; BD="$ROOT/bind_$C12"
  GATE="$CODE/stack/scripts/launch_gate.py"; PF="$WORK/preflight.txt"; HERE="$CODE/stack/scripts"
  EVO="$CODE/stack/ops/refcv8_eval_remap_a2_thor.json"
  mkdir -p "$WORK" 2>/dev/null; : > "$PF"; pf() { echo "$1 $2" >> "$PF"; }
  [ -f "$GATE" ] && pf PF-OK gate-present || pf PF-FAIL gate-present
  [ -f "$ARGV" ] && pf PF-OK argv-present || pf PF-FAIL "argv-present:$ARGV"
  # the profile must be CLOSED in THIS tree (MAPFT-BUDGET): at 70211d6 it is not, and the token could only ever be INCOMPLETE
  NOI="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(len(L.PROFILES["refcv8-mapft"]["open_items"]))' "$CODE")"
  [ "$NOI" = 0 ] && pf PF-OK profile-closed || pf PF-FAIL "profile-open-items:${NOI:-unreadable} (apply mapft_close_budget.py and commit BEFORE this tree is gated)"
  # the budget flags in the argv re-derive from the recorded smoke (determinism check)
  if [ -s "$SRES" ]; then
    W="$(pyval 'import json,sys; print(json.load(open(sys.argv[1]))["smoke_wall_s"])' "$SRES")"; M="$(pyval 'import json,sys; print(json.load(open(sys.argv[1]))["metrics"])' "$SRES")"
    py "$KIT/make_mapft_argv.py" audit --argv "$ARGV" --from-smoke-metrics "$M" --smoke-wall-s "$W" > "$WORK/audit.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^AUDIT-OK' "$WORK/audit.txt")" = 1 ] && pf PF-OK argv-audit || pf PF-FAIL argv-audit
  else pf PF-FAIL "argv-audit:no $SRES"; fi
  # every repo-relative input the profile names (G-CLOCK references, the FIX-4 tau record) is IN the shipped tree
  NPI="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; p = L.PROFILES["refcv8-mapft"]; print("\n".join([*p["clock_reference"], p["nav_tau_record_default"]]))' "$CODE")"
  n_pi=0; while IFS= read -r rel; do [ -n "$rel" ] || continue; n_pi=$((n_pi+1))
    [ -f "$CODE/$rel" ] && pf PF-OK "profile-input:$(basename "$rel")" || pf PF-FAIL "profile-input:$(basename "$rel")"; done <<< "$NPI"
  [ "$n_pi" -eq 4 ] && pf PF-OK profile-inputs-4 || pf PF-FAIL "profile-inputs-$n_pi-of-4"
  # the argv's code paths are THIS tree; init-from is the shipped EMA model's path; the eval remap equals the argv's own 4 files
  JD="$(argv_get "$ARGV" --join-defect-masks)"
  [ "$JD" = "$CODE/stack/tanitad/configs/refcv8_join_label_defects_train4572.json" ] && { [ "$DRY" = 1 ] || [ -s "$JD" ]; } && pf PF-OK argv-code-path || pf PF-FAIL "argv-code-path:$JD"
  [ "$(argv_get "$ARGV" --init-from)" = "$INIT" ] && pf PF-OK argv-init-from-path || pf PF-FAIL "argv-init-from-path:$(argv_get "$ARGV" --init-from)"
  [ -f "$EVO" ] && pf PF-OK eval-remap-override || pf PF-FAIL eval-remap-override
  NEV="$(pyval 'import json,sys; o=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); a=d["argv"] if isinstance(d,dict) else d; print(sum(1 for f,p in o.items() if f in a and a[a.index(f)+1]==p), len(o))' "$EVO" "$ARGV" 2>/dev/null)"
  [ "$NEV" = "4 4" ] && pf PF-OK eval-remap-is-argv || pf PF-FAIL "eval-remap-is-argv:${NEV:-unreadable}"
  if [ "$DRY" != 1 ]; then
    SZ="$(stat -c %s "$INIT" 2>/dev/null)"; [ "$SZ" = "$INIT_BYTES" ] && pf PF-OK init-from-size || pf PF-FAIL "init-from-size:${SZ:-missing}"
    for f in --r8-v9-labels:$V9_MD5 --r8-v9-labels-eval:$V9_EVAL_MD5; do
      M5="$(md5sum "$(argv_get "$ARGV" "${f%%:*}")" 2>/dev/null | cut -c1-32)"
      [ "${#M5}" = 32 ] && [ "$M5" = "${f##*:}" ] && pf PF-OK "md5${f%%:*}" || pf PF-FAIL "md5${f%%:*}:${M5:-missing}"; done
    for p in $(pyval 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).values()))' "$EVO"); do
      [ -s "$p" ] && pf PF-OK "eval-remap-file:$(basename "$p")" || pf PF-FAIL "eval-remap-file:$(basename "$p")"; done
    RO="$(argv_get "$ARGV" --out)"; [ -n "$RO" ] && [ ! -e "$RO/ckpt.pt" ] && [ ! -e "$RO/summary.json" ] && pf PF-OK run-dir-fresh || pf PF-FAIL "run-dir-fresh:$RO"
    "$PY" -c "import torch, timm" > /dev/null 2>&1 9>&- 200>&- && pf PF-OK python-imports-torch-timm || pf PF-FAIL "python-imports-torch-timm:$PY"
    FREE="$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc 0-9)"; [ "${FREE:-0}" -ge 60 ] && pf PF-OK disk-60G || pf PF-FAIL disk-60G
  else pf PF-OK "(dry-run: Thor-only preflights skipped: sizes, md5s, remap files, run dir, imports, disk)"; fi
  NF="$(grep -c '^PF-FAIL' "$PF")"; cat "$PF"
  [ "$NF" = 0 ] || die "$NF preflight failure(s) in $PF"
  MBD="${MAPBF:-$BD}"; MAPREC="$MBD/run_map/g_map_overfit.json"; MAPCLO="$MBD/map_closure.json"; BOXREC="$BD/run_box/gbo_binding.json"; BOXCLO="$BD/box_closure.json"
  A=(run --profile refcv8-mapft --stage thor --tree "$CODE" --commit "$COMMIT" --argv-file "$ARGV" --eval-kit /home/nvidia
     --eval-remap-overrides "$EVO" --map-overfit-record "$MAPREC" --map-overfit-closure "$MAPCLO"
     --box-overfit-record "$BOXREC" --box-overfit-closure "$BOXCLO")
  # launch_gate's _collect_evidence globs G-*.json DIRECTLY in each --import-evidence dir; the dev-box out-dir keeps them under evidence/
  if [ -n "$DEVEV" ]; then [ -d "$DEVEV/evidence" ] && DEVEV="$DEVEV/evidence"
    [ "$DRY" = 1 ] || [ -s "$DEVEV/G-SUITE-PINNED.json" ] || die "no G-SUITE-PINNED.json in $DEVEV (the gate's dev-box half)"
    A+=(--import-evidence "$DEVEV"); fi
  [ -n "$SUITE" ] && A+=(--suite-verdict "$SUITE")
  [ -n "$PICOST" ] && A+=(--pi-cost-approval "$PICOST")
  if [ "$MODE" = gate ]; then
    A+=(--out-dir "$GATE_OUT")
    say "gate command: $PY $GATE ${A[*]}"
    [ "$DRY" = 1 ] && { echo "ZZMAPFTGATE-DRYOKZZ"; exit 0; }
    [ -n "$DEVEV" ] && [ -n "$SUITE" ] || die "--devbox-evidence and --suite-verdict are required (G-SUITE-PINNED and G-SUITE are required checks)"
    NEEDBOX="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(int("G-BOX-OVERFIT" in L.PROFILES["refcv8-mapft"]["required_checks"]))' "$CODE")"
    case "$NEEDBOX" in 1) KINDS="map box";; 0) KINDS="map";; *) die "cannot read refcv8-mapft required_checks from $CODE";; esac
    say "binding records checked: $KINDS (G-BOX-OVERFIT required=$NEEDBOX)"
    for k2 in $KINDS; do
      [ "$k2" = map ] && { R="$MAPREC"; C="$MAPCLO"; } || { R="$BOXREC"; C="$BOXCLO"; }
      [ -s "$R" ] && [ -s "$C" ] || die "no binding $k2 record/closure under $BD (run 'bind' on THIS commit and THIS argv first)"
      py "$HERE/refcv8_assemble_argv.py" --assert-closure "$C" > "$WORK/$k2.closure.txt" 2>&1 9>&- 200>&-
      [ "$(grep -c '^ASSERT-OK' "$WORK/$k2.closure.txt")" = 1 ] || die "$k2 closure not binding-clean: $C"
      py "$HERE/refcv8_assemble_argv.py" --assert-record "$R" --kind "$k2" > "$WORK/$k2.record.txt" 2>&1 9>&- 200>&-
      [ "$(grep -c '^ASSERT-OK' "$WORK/$k2.record.txt")" = 1 ] || die "$k2 record not PASS: $R"
    done
    gpu_take                                  # G-LIVE / G-CKPT / G-EVAL run on the GPU: the lock is held to the end
    "$PY" "$GATE" "${A[@]}" > "$WORK/gate.log" 2>&1 9>&- 200>&-          # stdout to a FILE: no pipe, no $? to trust
    TOK="$GATE_OUT/PASS_$C12.json"
    V="$(pyval 'import json,glob,sys; fs=sorted(glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json")); d=json.load(open(fs[-1])) if fs else {}; print(d.get("verdict","NO-TOKEN"))' "$GATE_OUT" "$C12")"
    if [ "$V" != PASS ] || [ ! -s "$TOK" ]; then
      pyval 'import json,glob,sys; [print(" -",r[:300]) for f in glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json") for r in json.load(open(f)).get("reasons",[])[:12]]' "$GATE_OUT" "$C12"
      die "token verdict is '$V', not PASS (see $GATE_OUT; a PI-DECISION on cost is lifted with --pi-cost-approval, nothing else is)"; fi
    MJ="$(find "$GATE_OUT" -name metrics.jsonl -size +0 2>/dev/null | head -1)"
    NE=0; [ -n "$MJ" ] && NE="$(grep -c 'eval_' "$MJ")"
    [ -n "$MJ" ] || die "INCONCLUSIVE: no smoke metrics.jsonl under $GATE_OUT (cannot assert instrument reach)"
    [ "${NE:-0}" -ge 1 ] || die "the gate smoke wrote no in-run eval row"
    say "token $TOK is PASS"; echo "ZZMAPFTGATE-PASSZZ"; exit 0
  fi
  # ------------------------------------------------------------------------------------- redarms
  # PREREG_MAPFT s1: "launch only on a PASS of profile refcv8-mapft (incl. its two RED arms)".  Each arm re-installs ONE defect through the
  # real gate CLI (`--arm X`, never an edited file) on G-DVB + G-LIVE and must come out NOT-PASS with a map-only reason, while the MAIN
  # gate token (already PASS) read PASS on the same two checks (the control).
  TOKM="$GATE_OUT/PASS_$C12.json"
  [ "$DRY" = 1 ] || [ -s "$TOKM" ] || die "run 'gate' first: the MAIN token $TOKM is the arm-free CONTROL"
  RED="$ROOT/red_$C12"; mkdir -p "$RED" 2>/dev/null
  # launch_gate.py run puts --stage and --checks in ONE mutually-exclusive group: "--stage thor" + "--checks" is an argparse
  # error (rc 2, no arm token -> UNGRADED -> a false "freeze guard is decorative").  MEASURED 2026-10-26 on C1 7e14cad3.  The arms
  # therefore take A WITHOUT its --stage pair; --checks G-DVB,G-LIVE selects the checks.
  AR=(); i=0; while [ $i -lt ${#A[@]} ]; do if [ "${A[$i]}" = --stage ]; then i=$((i+2)); else AR+=("${A[$i]}"); i=$((i+1)); fi; done
  for ARMN in mapft_planner_trainable mapft_map_frozen; do
    AA=("${AR[@]}" --out-dir "$RED/$ARMN" --checks G-DVB,G-LIVE --arm "$ARMN")
    say "RED arm $ARMN: $PY $GATE ${AA[*]}"
  done
  [ "$DRY" = 1 ] && { echo "ZZMAPFTRED-DRYOKZZ"; exit 0; }
  gpu_take
  NBAD=0
  for ARMN in mapft_planner_trainable mapft_map_frozen; do
    AA=("${AR[@]}" --out-dir "$RED/$ARMN" --checks G-DVB,G-LIVE --arm "$ARMN")
    "$PY" "$GATE" "${AA[@]}" > "$RED/$ARMN.log" 2>&1 9>&- 200>&-
    grade_arm "$ARMN" "$RED/$ARMN" "$TOKM" > "$RED/$ARMN.grade.txt"
    say "$(cat "$RED/$ARMN.grade.txt")"
    [ "$(grep -c -- "-CAUGHT" "$RED/$ARMN.grade.txt")" = 1 ] || NBAD=$((NBAD+1))
  done
  [ "$NBAD" = 0 ] || die "$NBAD RED arm(s) NOT caught: the freeze guard is decorative -- do NOT launch (see $RED/*.grade.txt)"
  say "both RED arms caught"; echo "ZZMAPFTRED-OKZZ";;
grade-arm)   # read-only: grade an arm run from its token + the control token (the rehearsal / audit entry point)
  [ -n "${GRADE_ARM:-}" ] && [ -n "${GRADE_DIR:-}" ] && [ -n "${GRADE_MAIN:-}" ] || die "set GRADE_ARM, GRADE_DIR, GRADE_MAIN"
  grade_arm "$GRADE_ARM" "$GRADE_DIR" "$GRADE_MAIN";;
# =========================================================================================================== fire (MM)
fire)
  need COMMIT; C12="${COMMIT:0:12}"; GATE_OUT="$ROOT/gate_$C12"; WORK="$GATE_OUT/prelaunch"; TOK="$GATE_OUT/PASS_$C12.json"
  VJ="$WORK/verify.json"; ENVF="$ROOT/runs.d/$ARM.env"
  if [ "$DRY" != 1 ]; then
    [ -s "$TOK" ] || die "no PASS token at $TOK"
    for a2 in mapft_planner_trainable mapft_map_frozen; do   # PREREG_MAPFT s1: the PASS includes its two RED arms -- CAUGHT, graded from artifacts
      [ "$(grep -c -- '-CAUGHT' "$ROOT/red_$C12/$a2.grade.txt" 2>/dev/null)" = 1 ] || die "RED arm $a2 has no CAUGHT grade under $ROOT/red_$C12 (run 'redarms')"; done
  fi
  if [ "$DRY" = 1 ]; then say "DRY: verify $TOK; manifest $ENVF; gpu_take; nohup bash $CODE/stack/ops/sup_refcv7.sh $ARM $ROOT/runs.d"; echo "ZZMAPFTFIRE-DRYOKZZ"; exit 0; fi
  if [ "$ZGOFF" != 1 ]; then   # the post-run gate would otherwise REJECT a finished run (ZEROGRAD_FAIL is final): find out BEFORE 6 Thor-hours
    [ "$(head -c 4 "$SD/ZEROGRAD_SMOKE.txt" 2>/dev/null)" = PASS ] || die "the zero-gradient dress rehearsal on the smoke ckpt did not PASS ($(cat "$SD/ZEROGRAD_SMOKE.txt" 2>/dev/null)): fix the control/allow list, or pass --zerograd-off (logged and recorded in the done-marker)"
  fi
  rm -f "$VJ"; "$PY" "$CODE/stack/scripts/launch_gate.py" verify --token "$TOK" --argv-file "$ARGV" --tree "$CODE" --commit "$COMMIT" --json "$VJ" > "$WORK/verify.log" 2>&1 9>&- 200>&-
  [ "$(pyval 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","NONE"))' "$VJ")" = MATCH ] || die "verify did not MATCH ($VJ)"
  mkdir -p "$ROOT/runs.d" "$ROOT/runs"
  cat > "$ENVF" <<EOF
# $ARM -- launch manifest for stack/ops/sup_refcv7.sh (generated by mapft_prelaunch.sh; sourced ONCE by the supervisor)
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
  if [ "$ZGOFF" = 1 ]; then echo "ZEROGRAD_GATE=off" >> "$ENVF"; else echo "ZEROGRAD_CONTROL=('$ZG_CONTROL')" >> "$ENVF"; fi
  gpu_take
  # the RUN holds the GPU lock too: the supervisor INHERITS fd 9 on purpose (released when the LAST process of the run exits); 200 is closed
  nohup bash "$CODE/stack/ops/sup_refcv7.sh" "$ARM" "$ROOT/runs.d" > "$ROOT/sup_$ARM.out" 2>&1 200>&- &
  RO="$(argv_get "$ARGV" --out)"
  for i in 1 2 3 4 5 6; do sleep 20 9>&- 200>&-; grep -q 'ZZGATEOK-' "$RO/sup_$ARM.log" 2>/dev/null && break; done
  NS="$(ps -eo args | grep -c "sup_refcv[7].sh $ARM")"
  [ "$NS" -ge 1 ] || die "no supervisor process after start (see $ROOT/sup_$ARM.out)"
  say "supervisor up; progress marker ZZ${ARM}-<step>-<steps>-<errs>-<launch>ZZ in $RO/sup_$ARM.log"; echo "ZZMAPFTFIRE-OKZZ";;
*) echo "usage: mapft_prelaunch.sh [--dry-run] preflight|smoke|budget|bind|gate|redarms|fire|grade-arm (see header)"; exit 2;;
esac
