#!/usr/bin/env bash
# H-NT-DECIDE-1 PRE-LAUNCH kit (profile refcv8-nt): bind -> gate -> fire. refcv8_prelaunch.sh's structure, with NT's
# arm, profile, argv (COMMITTED by nt_argv.py / closed by nt_close.py: no assembly, no gs-measure) and NT preflights.
#
#   THOR : nt_prelaunch.sh bind --code /home/nvidia/r8ns/nt/launch_tree --commit SHA40 [--arm nt-a1-s0]
#              G-MAP-OVERFIT then G-BOX-OVERFIT binding records (closure_run.py --binding, MAIN arm only per A19) for
#              THIS arm's argv sha -- refcv8's templates (refcv8_bind_{map,box}.args.tpl), GPU, ~34 + ~82 min INHERITED
#          nt_prelaunch.sh gate --code ... --commit SHA40 [--arm A] [--devbox-evidence DIR] [--suite-verdict F]
#              preflights, then launch_gate.py run --profile refcv8-nt --stage thor ... -> PASS_<c12>.json (hours)
#          nt_prelaunch.sh fire --code ... --commit SHA40 [--arm A]       (the MASTER MIND's step: verify == MATCH,
#              manifest, sup_refcv7.sh) -- never run by an agent that was told not to launch
#   ANY  : nt_prelaunch.sh --dry-run <mode> ...   prints what it WOULD run; touches nothing, takes no lock.
#
# ⛔ THOR GPU PROTOCOL (MM 2026-10-21): every GPU step takes /home/nvidia/refcv7_post/thor_gpu.lock (blocking, <= 2 h)
# AND requires `nvidia-smi --query-compute-apps` to be EMPTY once it holds the lock; the GPU is never shared.
# Kept from refcv8_prelaunch.sh: verdicts are read from ARTIFACTS (token / closure / record JSON), never from $? and
# never through a pipe; every child closes the lock fds (9, 200); no pgrep -f / pkill -f; markers are ZZ...ZZ.
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MODE="${1:-}"; shift || true
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=nt-a1-s0; OUTROOT="${OUT_ROOT:-/home/nvidia/r8ns/nt}"; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
NT=/home/nvidia/r8ns/nt; LAB=$NT/labels/a1; ACK=$NT/inputs/AMENDMENT_ACK_NT.txt
ACK_MD5=bd49cf77b455671ad03527d6c897b666                     # = git blob 38175992... (profile amendment_ack)
THETA_DEC=/home/nvidia/r8ns/navtrain_sel/wiseft/ckpt_mod2/ckpt_wiseft_mod2_DEC.pt; THETA_DEC_MD5=dad4cb5b5fb316922b4e9a7e85cee8f8
V9_EVAL_MD5=6b5c7f207cffc3b7eb3cd527fd433599
CODE=""; COMMIT=""; DEVEV=""; SUITE=""; PICOST=""
while [ $# -gt 0 ]; do case "$1" in
  --code) CODE="$2"; shift 2;; --commit) COMMIT="$2"; shift 2;; --arm) ARM="$2"; shift 2;;
  --devbox-evidence) DEVEV="$2"; shift 2;; --suite-verdict) SUITE="$2"; shift 2;; --pi-cost-approval) PICOST="$2"; shift 2;;
  --out-root) OUTROOT="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; exit 2;; esac; done
case "$ARM" in nt-a1-s0|nt-a1r-s1) ;; *) echo "REFUSED: --arm must be nt-a1-s0 or nt-a1r-s1"; exit 2;; esac

say() { echo "[nt-prelaunch:$MODE:$ARM] $*"; }
die() { say "REFUSED: $*"; echo "ZZNTPRELAUNCH-REFUSEDZZ"; exit 10; }
need() { for v in "$@"; do [ -n "${!v}" ] || die "--$(echo "$v" | tr 'A-Z_' 'a-z-') is required for '$MODE'"; done; }
pyval() { "$PY" -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }   # a value, never a status
argv_get() { pyval 'import json,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(a[a.index(sys.argv[2])+1] if sys.argv[2] in a else "")' "$ARGV" "$1"; }
NT_PENDING=$NT/NT_LAUNCH_PENDING                 # MM 2026-10-22 00:45: the L7 capture chain yields to NT when this exists
gpu_take() {  # the protocol: the yield marker, then a blocking lock, then an EMPTY GPU
  # the marker FIRST so a queued chain exits before its next step (we then wait at most one step); it is LEFT in place
  # -- the MM clears it after the A1 launch
  touch "$NT_PENDING" 9>&- 200>&- || die "cannot write the yield marker $NT_PENDING"
  say "yield marker $NT_PENDING present (left in place for the MM)"
  exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
  NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
  [ "$NAPPS" = 0 ] || die "the GPU is not empty under the lock ($NAPPS compute app(s)): never share the GPU"
}

[ -n "$MODE" ] || { echo "usage: nt_prelaunch.sh [--dry-run] bind|gate|fire --code DIR --commit SHA40 [--arm A]"; exit 2; }
need CODE COMMIT
[ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
C12="${COMMIT:0:12}"; ARGV="$CODE/stack/ops/runs.d/$ARM.argv.json"; BD="$OUTROOT/bind_${C12}_$ARM"
GATE_OUT="$OUTROOT/gate_${C12}_$ARM"; WORK="$GATE_OUT/prelaunch"; EVO="$CODE/stack/ops/refcv8nt_eval_remap_thor.json"
[ -f "$ARGV" ] || die "no committed argv at $ARGV"

case "$MODE" in
# ------------------------------------------------------------------------------------------------------ bind
bind)
  mkdir -p "$BD" 2>/dev/null
  BAUD="$BD/run_box/audit/raw"; BSRC="$CODE/stack/ops/refcv8_box_audit/raw"
  if [ "$DRY" != 1 ]; then
    mkdir -p "$BAUD/visibility" 2>/dev/null
    cp -p "$BSRC/PREREG_G_BOX_OVERFIT.md" "$BAUD/" && cp -p "$BSRC/visibility/gbo_frameset.json" "$BAUD/visibility/" \
      || die "could not prepare the box audit inputs from $BSRC"
    echo "594c71196cc5bbd527fee40b2cb0e3f1  $BAUD/PREREG_G_BOX_OVERFIT.md" | md5sum -c - > /dev/null 2>&1 || die "box audit prereg md5"
    echo "b291404c36f83c3e397e3b90367e8e7b  $BAUD/visibility/gbo_frameset.json" | md5sum -c - > /dev/null 2>&1 || die "box audit frame-set md5"
  fi
  # the harness records bind to sha256(json.dumps(argv, separators=(",",":"))) -- THIS arm's argv
  ASHA="$(pyval 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); a=d["argv"] if isinstance(d,dict) else d; print(hashlib.sha256(json.dumps(a,separators=(",",":")).encode()).hexdigest())' "$ARGV")"
  [ "${#ASHA}" = 64 ] || die "could not hash the launch argv"
  [ "$(argv_get --r8-v9-md5)" != PENDING-NT-CORPUS ] || die "the argv still carries PENDING-NT-CORPUS: close NT-CORPUS (nt_close.py) and commit BEFORE binding (the records bind the argv sha)"
  for K2 in map box; do
    TPL="$HERE/refcv8_bind_$K2.args.tpl"; [ -f "$TPL" ] || die "no harness template $TPL"
    "$PY" "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
        --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$K2" --sub "INPUTS=${BIND_INPUTS:-/home/nvidia/refcv7_bind/run_$K2/inputs}" \
        --args-out "$BD/$K2.args" --data-roots-out "$BD/$K2.roots" > "$BD/$K2.render.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^RENDERED' "$BD/$K2.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$K2.render.txt)"
  done
  if [ "$DRY" = 1 ]; then say "DRY: lock+empty-GPU; map ($(wc -l < "$BD/map.args") args) then box ($(wc -l < "$BD/box.args") args) via closure_run.py --binding for argv sha ${ASHA:0:16}"; echo "ZZNTBIND-DRYOKZZ"; exit 0; fi
  gpu_take
  for K2 in map box; do
    mapfile -t HARGS < "$BD/$K2.args"; mapfile -t ROOTS < "$BD/$K2.roots"; DR=(); for r in "${ROOTS[@]}"; do [ -n "$r" ] && DR+=(--data-root "$r"); done
    mkdir -p "$BD/run_$K2" 2>/dev/null
    if [ "$K2" = map ]; then REC="$BD/run_map/g_map_overfit.json"; else REC="$BD/run_box/gbo_binding.json"; fi
    ( cd "$CODE" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
      /home/nvidia/venvs/tanitad-train/bin/python "$CODE/stack/scripts/closure_run.py" --binding --commit "$COMMIT" --tree "$CODE" \
        --out "$BD/${K2}_closure.json" --result "$REC" "${DR[@]}" -- "${HARGS[@]}" > "$BD/$K2.out" 2>&1 9>&- 200>&- )
    "$PY" "$HERE/refcv8_assemble_argv.py" --assert-closure "$BD/${K2}_closure.json" > "$BD/$K2.closure.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.closure.txt")" = 1 ] || die "$K2 closure refused: $(tr -d '\r' < "$BD/$K2.closure.txt" | head -3 | tr '\n' ' ')"
    "$PY" "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$K2" > "$BD/$K2.record.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$BD/$K2.record.txt")" = 1 ] || die "$K2 record refused: $(tr -d '\r' < "$BD/$K2.record.txt" | head -3 | tr '\n' ' ')"
    say "$K2 binding run asserted: closure $BD/${K2}_closure.json, record $REC"
  done
  echo "ZZNTBIND-OKZZ";;
# ------------------------------------------------------------------------------------------------------ gate
gate)
  GATE="$CODE/stack/scripts/launch_gate.py"; PF="$WORK/preflight.txt"
  mkdir -p "$WORK" 2>/dev/null; : > "$PF"
  pf() { echo "$1 $2" >> "$PF"; }
  [ -f "$GATE" ] && pf PF-OK gate-present || pf PF-FAIL gate-present
  # the profile is CLOSED (NT-CORPUS + NT-B1-GUARD): open items make the token INCOMPLETE forever
  NOI="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(len(L.PROFILES["refcv8-nt"]["open_items"]))' "$CODE")"
  [ "$NOI" = 0 ] && pf PF-OK profile-closed || pf PF-FAIL "profile-open-items:${NOI:-unreadable}"
  V9="$(argv_get --r8-v9-md5)"; [ "${#V9}" = 32 ] && pf PF-OK argv-v9-md5-literal || pf PF-FAIL "argv-v9-md5:$V9"
  # every repo-relative input the profile names (G-CLOCK x4, FIX-4 tau) is IN the shipped tree
  NPI="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; p = L.PROFILES["refcv8-nt"]; print("\n".join([*p["clock_reference"], p["nav_tau_record_default"]]))' "$CODE")"
  n_pi=0; while IFS= read -r rel; do [ -n "$rel" ] || continue; n_pi=$((n_pi+1))
    [ -f "$CODE/$rel" ] && pf PF-OK "profile-input:$(basename "$rel")" || pf PF-FAIL "profile-input:$(basename "$rel")"; done <<< "$NPI"
  [ "$n_pi" -eq 5 ] && pf PF-OK profile-inputs-5 || pf PF-FAIL "profile-inputs-$n_pi-of-5"
  # the argv's code paths are THIS tree (the argv embeds the launch_tree path)
  for f in --join-defect-masks --nt-heldout; do P="$(argv_get $f)"; case "$P" in "$CODE"/*) [ -s "$P" ] && pf PF-OK "argv-code-path$f" || pf PF-FAIL "argv-code-path$f:missing";; *) pf PF-FAIL "argv-code-path$f:$P-not-under-$CODE";; esac; done
  # G-EVAL: the eval loader's recorded override names exactly THIS argv's four files (refcv8's names 1 of 4)
  [ -f "$EVO" ] && pf PF-OK eval-remap-override || pf PF-FAIL eval-remap-override
  NEV="$(pyval 'import json,sys; o=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); a=d["argv"] if isinstance(d,dict) else d; print(sum(1 for f,p in o.items() if f in a and a[a.index(f)+1]==p), len(o))' "$EVO" "$ARGV" 2>/dev/null)"
  [ "$NEV" = "4 4" ] && pf PF-OK eval-remap-is-argv || pf PF-FAIL "eval-remap-is-argv:${NEV:-unreadable}"
  if [ "$DRY" != 1 ]; then
    M="$(md5sum "$ACK" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = "$ACK_MD5" ] && pf PF-OK amendment-ack || pf PF-FAIL "amendment-ack:${M:-missing}"
    M="$(md5sum "$THETA_DEC" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = "$THETA_DEC_MD5" ] && [ "$(argv_get --init-from)" = "$THETA_DEC" ] && pf PF-OK init-from-theta-dec || pf PF-FAIL "init-from:${M:-missing}"
    M="$(md5sum "$(argv_get --r8-v9-labels)" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = "$V9" ] && pf PF-OK md5--r8-v9-labels || pf PF-FAIL "md5--r8-v9-labels:${M:-missing}"
    M="$(md5sum "$(argv_get --r8-v9-labels-eval)" 2>/dev/null | cut -c1-32)"; [ "${#M}" = 32 ] && [ "$M" = "$V9_EVAL_MD5" ] && pf PF-OK md5--r8-v9-labels-eval || pf PF-FAIL "md5--r8-v9-labels-eval:${M:-missing}"
    S="$(sha256sum "$(argv_get --nt-heldout)" 2>/dev/null | cut -c1-64)"; [ "${#S}" = 64 ] && [ "$S" = "$(argv_get --nt-heldout-sha256)" ] && pf PF-OK nt-heldout-sha || pf PF-FAIL "nt-heldout-sha:${S:-missing}"
    [ -s "$NT/DONE_A1.json" ] && pf PF-OK e1-done || pf PF-FAIL e1-done
    [ -s "$NT/labels/a1_build/logs/FINAL_DONE" ] && pf PF-OK labels-final-done || pf PF-FAIL labels-final-done
    OKC="$(pyval 'import json,sys; print(json.load(open(sys.argv[1])).get("ALL_OK"))' "$NT/labels/a1_controls/CONTROLS_B_final.json" 2>/dev/null)"
    [ "$OKC" = True ] && pf PF-OK k1-k2-at-scale-final || pf PF-FAIL "k1-k2-at-scale-final:${OKC:-missing}"
    for f in $(pyval 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).values()))' "$EVO"); do
      [ -s "$f" ] && pf PF-OK "eval-remap-file:$(basename "$f")" || pf PF-FAIL "eval-remap-file:$(basename "$f")"; done
    RO="$(argv_get --out)"; [ -n "$RO" ] && [ ! -e "$RO/ckpt.pt" ] && [ ! -e "$RO/summary.json" ] && pf PF-OK run-dir-fresh || pf PF-FAIL "run-dir-fresh:$RO"
    "$PY" -c "import torch, timm" > /dev/null 2>&1 9>&- 200>&- && pf PF-OK python-imports-torch-timm || pf PF-FAIL "python-imports-torch-timm:$PY"
    FREE="$(df -BG --output=avail "$OUTROOT" 2>/dev/null | tail -1 | tr -dc 0-9)"; [ "${FREE:-0}" -ge 60 ] && pf PF-OK disk-60G || pf PF-FAIL disk-60G
  else say "DRY: skipping Thor-only preflights (ACK, theta_DEC, v9 md5s, held-out sha, DONE markers, K1/K2, run dir, imports, disk)"; fi
  NF="$(grep -c '^PF-FAIL' "$PF")"; cat "$PF"
  [ "$NF" = 0 ] || die "$NF preflight failure(s) in $PF"
  MAPREC="$BD/run_map/g_map_overfit.json"; MAPCLO="$BD/map_closure.json"; BOXREC="$BD/run_box/gbo_binding.json"; BOXCLO="$BD/box_closure.json"
  A=(run --profile refcv8-nt --stage thor --tree "$CODE" --commit "$COMMIT" --argv-file "$ARGV" --out-dir "$GATE_OUT" --eval-kit /home/nvidia
     --eval-remap-overrides "$EVO" --map-overfit-record "$MAPREC" --map-overfit-closure "$MAPCLO"
     --box-overfit-record "$BOXREC" --box-overfit-closure "$BOXCLO")
  [ -n "$DEVEV" ] && A+=(--import-evidence "$DEVEV"); [ -n "$SUITE" ] && A+=(--suite-verdict "$SUITE")
  [ -n "$PICOST" ] && A+=(--pi-cost-approval "$PICOST")
  say "gate command: $PY $GATE ${A[*]}"
  [ "$DRY" = 1 ] && { echo "ZZNTGATE-DRYOKZZ"; exit 0; }
  for k2 in map box; do
    [ "$k2" = map ] && { R="$MAPREC"; C="$MAPCLO"; } || { R="$BOXREC"; C="$BOXCLO"; }
    [ -s "$R" ] && [ -s "$C" ] || die "no binding $k2 record/closure under $BD (run 'bind' on THIS commit and arm first)"
    "$PY" "$HERE/refcv8_assemble_argv.py" --assert-closure "$C" > "$WORK/$k2.closure.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$WORK/$k2.closure.txt")" = 1 ] || die "$k2 closure not binding-clean: $C"
    "$PY" "$HERE/refcv8_assemble_argv.py" --assert-record "$R" --kind "$k2" > "$WORK/$k2.record.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^ASSERT-OK' "$WORK/$k2.record.txt")" = 1 ] || die "$k2 record not PASS: $R"
  done
  gpu_take                                   # the gate's G-LIVE / G-CKPT / G-EVAL run on the GPU: the lock is held to the end
  "$PY" "$GATE" "${A[@]}" > "$WORK/gate.log" 2>&1 9>&- 200>&-          # stdout to a FILE: no pipe, no $? to trust
  TOK="$GATE_OUT/PASS_$C12.json"
  V="$(pyval 'import json,glob,sys; fs=sorted(glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json")); d=json.load(open(fs[-1])) if fs else {}; print(d.get("verdict","NO-TOKEN"))' "$GATE_OUT" "$C12")"
  if [ "$V" != PASS ] || [ ! -s "$TOK" ]; then
    pyval 'import json,glob,sys; [print(" -",r[:300]) for f in glob.glob(sys.argv[1]+"/*_"+sys.argv[2]+".json") for r in json.load(open(f)).get("reasons",[])[:12]]' "$GATE_OUT" "$C12"
    die "token verdict is '$V', not PASS (see $GATE_OUT)"; fi
  MJ="$(find "$GATE_OUT" -name metrics.jsonl -size +0 2>/dev/null | head -1)"
  NE=0; [ -n "$MJ" ] && NE="$(grep -c 'eval_' "$MJ")"
  [ -n "$MJ" ] || die "INCONCLUSIVE: no smoke metrics.jsonl under $GATE_OUT (cannot assert instrument reach)"
  [ "${NE:-0}" -ge 1 ] || die "the smoke wrote no in-run eval row"
  say "token $TOK is PASS"; echo "ZZNTGATE-PASSZZ";;
# ------------------------------------------------------------------------------------------------------ fire (MM)
fire)
  TOK="$GATE_OUT/PASS_$C12.json"; VJ="$WORK/verify.json"; ENVF="$OUTROOT/runs.d/$ARM.env"
  [ -s "$TOK" ] || die "no PASS token at $TOK"
  if [ "$DRY" = 1 ]; then say "DRY: verify $TOK; manifest $ENVF; lock+empty-GPU; nohup bash $CODE/stack/ops/sup_refcv7.sh $ARM $OUTROOT/runs.d"; echo "ZZNTFIRE-DRYOKZZ"; exit 0; fi
  rm -f "$VJ"; "$PY" "$CODE/stack/scripts/launch_gate.py" verify --token "$TOK" --argv-file "$ARGV" --tree "$CODE" --commit "$COMMIT" --json "$VJ" > "$WORK/verify.log" 2>&1 9>&- 200>&-
  [ "$(pyval 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict","NONE"))' "$VJ")" = MATCH ] || die "verify did not MATCH ($VJ)"
  mkdir -p "$OUTROOT/runs.d"
  cat > "$ENVF" <<EOF
# $ARM -- launch manifest for stack/ops/sup_refcv7.sh (generated by nt_prelaunch.sh; sourced ONCE by the supervisor)
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
  gpu_take
  # ⛔ the RUN takes the GPU lock too (MM protocol: "the run itself"). The supervisor INHERITS fd 9 on purpose: the lock
  # is released only when the LAST process of the run (supervisor, trainer, workers) has exited -- the fd-inheritance
  # that is a trap for a SUPERVISOR lock is the right semantics for an exclusive GPU. 200 is still closed.
  nohup bash "$CODE/stack/ops/sup_refcv7.sh" "$ARM" "$OUTROOT/runs.d" > "$OUTROOT/sup_$ARM.out" 2>&1 200>&- &
  for i in 1 2 3 4 5 6; do sleep 20 9>&- 200>&-; grep -q 'ZZGATEOK-' "$OUTROOT/runs/$ARM/sup_$ARM.log" 2>/dev/null && break; done
  NS="$(ps -eo args | grep -c "sup_refcv[7].sh $ARM")"
  [ "$NS" -ge 1 ] || die "no supervisor process after start (see $OUTROOT/sup_$ARM.out)"
  say "supervisor up"; echo "ZZNTFIRE-OKZZ";;
*) echo "usage: nt_prelaunch.sh [--dry-run] bind|gate|fire --code DIR --commit SHA40 [--arm A]"; exit 2;;
esac
