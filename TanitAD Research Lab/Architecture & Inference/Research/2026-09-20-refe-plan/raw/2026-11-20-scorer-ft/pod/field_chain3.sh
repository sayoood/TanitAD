#!/bin/bash
# MM 2026-11-20: the REST of field_chain2.sh from F's K6 watch onward, with ONE change: what runs after F is decided AT THAT MOMENT, not at chain start.
#   Why: PI go 2026-11-20 ("Follow your recommendations, acclerate progress."): SCORER-FT goes right after F, and replaces the field Cs/Fs replicates
#   if F is refuted. field_chain2.sh (pid 108789) would have started Cs the minute F posted. This script takes over from it while F trains:
#   F's trainer is setsid'd (PPID 1), so stopping field_chain2's bash leaves F's training untouched.
#   Unchanged from field_chain2.sh, byte for byte in the functions: K6 on F (field_k6.py, pre-registered), wait_arm F, post_arm F (other out = C), the C0r block.
#   After the C0r block: if $ROOT/after_f.sh exists and is non-empty at that moment -> exec it (the MM writes it before F ends: SCORER-FT stage 1, then the
#   field replicates only per FIELD_REP_DECISION.txt). Otherwise the replicates run exactly as field_chain2.sh would have run them (the pod is not idled
#   on the MM).
# usage: field_chain3.sh <PKG = code_10848403 dir>.  Status: $ROOT/field_chain3.status (ZZFC3_DONE | ZZFC3_FAIL <stage> <why> | ZZFC3_K6_FIRE | ZZFC3_HANDOFF)
set -u
PKG="$1"
D=/workspace/data; ROOT=$D/refe_p5c; PY=/workspace/venv-teacher/bin/python; PKG5=$ROOT/code_33824420; RUNS=$D/refe_runs
S=$ROOT/field_chain3.status
log() { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
fail() { log "ZZFC3_FAIL $*"; exit 1; }
aside() {
  [ -e "$1/ckpt_last.pt" ] && mv -n "$1/ckpt_last.pt" "$1/ckpt_last.post_done.pt"
  [ -e "$1/ckpt_last.pt" ] && fail aside "$1/ckpt_last.pt still present"
  log "ASIDE $1/ckpt_last.pt -> ckpt_last.post_done.pt"
}
done2000() { [ -f "$1/summary.json" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==2000 else 1)" "$1/summary.json" 2>/dev/null; }
wait_arm() {
  local A="$1" O="$2" P; P="$(cut -d' ' -f1 "$ROOT/train_$A.pid")"
  while ! done2000 "$O"; do
    if ! kill -0 "$P" 2>/dev/null; then sleep 60; done2000 "$O" && break; fail "wait_$A" "pid $P gone, summary not done ($(tail -n 1 $ROOT/train_$A.log | cut -c1-160))"; fi
    sleep 300
  done
  for i in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 20; done
  kill -0 "$P" 2>/dev/null && fail "wait_$A" "pid $P alive 10 min after summary done"
  log "${A}_DONE"
}
run_arm() {    # <ARM> <PROFILE> <PKG> <OUT>
  local A="$1" PROF="$2" PK="$3" O="$4" r la; la="$(echo $A | tr A-Z a-z)"
  bash "$ROOT/arm_evidence.sh" "$A" "$PK"; grep -q "^ZZ${A}_EVIDENCE_DONE" "$ROOT/evidence_$la/status.txt" || fail "evidence_$A" "$(tail -n 1 $ROOT/evidence_$la/status.txt)"
  bash "$ROOT/arm_egr_gate.sh" "$A" "$PROF" "$PK"
  grep -q "^ZZEXIT 0" "$ROOT/launch_gate_$A.log" || fail "gate_$A" "$(grep -v '^ZZGATE' $ROOT/launch_gate_$A.log | tail -n 1 | cut -c1-200)"
  grep -q "^VERIFY_RC 0" "$ROOT/token_verify_$A.log" || fail "gate_$A" "verify: $(tail -n 2 $ROOT/token_verify_$A.log | tr '\n' ' ' | cut -c1-200)"
  grep -q "^PREFLIGHT_RC 0" "$ROOT/preflight_$A.log" || fail "gate_$A" "preflight: $(tail -n 2 $ROOT/preflight_$A.log | tr '\n' ' ' | cut -c1-200)"
  log "${A}_GATE_OK $(ls $ROOT/lg_$A/PASS_*.json | head -n 1 | xargs -n1 basename)"
  r="$(bash "$ROOT/arm_launch.sh" "$A" "$PROF" "$PK" "$O")"; log "$r"
  case "$r" in ZZ${A}_LAUNCHED*) ;; *) fail "launch_$A" "$r" ;; esac
}
post_arm() {   # <ARM> <PKG> <OUT> <HF prefix> [<OTHER out>]
  local A="$1"
  bash "$ROOT/arm_post.sh" "$@"; grep -q "^ZZ${A}POST_OK" "$ROOT/post_$A.status" || fail "post_$A" "$(tail -n 1 $ROOT/post_$A.status)"
  log "${A}_POST_OK $(cut -c1-16 $ROOT/model_final_sha256_$A.txt)"
  aside "$3"
}
[ -d "$PKG/refe" ] || { echo "no package $PKG"; exit 2; }
log "START chain pid $$ PKG $PKG (takes over from field_chain2.sh at F's K6 watch)"
grep -q "if FIELD_P7_ARM_W\[arm\] > 0.0 else ()" "$PKG/refe/launch_gate.py" || fail pkg "launch_gate.py lacks the live-term fix"
[ "$(sha256sum $PKG/refe/field_loss.py | cut -c1-64)" = da7b43030cbc89b43da74c3c9ce48b12450d3080c702374dc2e62dea929563b3 ] || fail pkg "field_loss.py is not the deterministic lookup (sha256 != da7b4303)"
[ -s "$ROOT/train_F.pid" ] || fail start "no train_F.pid: F is not launched, this script only takes over a RUNNING F"
grep -q "ZZF_LAUNCHED $(cut -d' ' -f1 $ROOT/train_F.pid)" "$ROOT/field_chain2.status" || fail start "train_F.pid is not the F that field_chain2 launched"
# ---- F (already launched by field_chain2.sh): K6, wait, post
while :; do
  k="$("$PY" "$ROOT/field_k6.py" "$RUNS/p7field_f/metrics.jsonl" 2.0 2>&1 | tail -n 1)"
  case "$k" in ZZK6_OK*) log "F $k"; break ;;
    ZZK6_FIRE*|ZZK6_INVALID*) log "F $k"; P="$(cut -d' ' -f1 $ROOT/train_F.pid)"; kill "$P"; sleep 120; kill -0 "$P" 2>/dev/null && kill -9 "$P"; log "ZZFC3_K6_FIRE F stopped (pid $P); the MM decides"; exit 3 ;;
  esac
  kill -0 "$(cut -d' ' -f1 $ROOT/train_F.pid)" 2>/dev/null || fail k6_F "F pid gone before step 525 ($(tail -n 1 $ROOT/train_F.log | cut -c1-160))"
  sleep 300
done
wait_arm F "$RUNS/p7field_f"
post_arm F "$PKG" "$RUNS/p7field_f" p7field/f "$RUNS/p7field_c"
# ---- C0r (only if the P5-C read triggered it AND the PI approved: C0R_DECISION.txt first word RUN)
dec="$(head -n 1 $ROOT/C0R_DECISION.txt 2>/dev/null | cut -d' ' -f1)"
case "$dec" in
  RUN)
    log "C0R_DECISION RUN: $(head -n 1 $ROOT/C0R_DECISION.txt | cut -c1-200)"
    "$PY" "$ROOT/c0r_mkargv.py" > "$ROOT/c0r_argv.log" 2>&1; grep -q "^ZZC0R_ARGV_OK" "$ROOT/c0r_argv.log" || fail c0r_argv "$(tail -n 1 $ROOT/c0r_argv.log | cut -c1-200)"
    log "$(grep '^ZZC0R_ARGV_OK' $ROOT/c0r_argv.log)"
    run_arm C0r refe_p5c_c0r "$PKG5" "$RUNS/p5c_c0r"
    wait_arm C0r "$RUNS/p5c_c0r"
    post_arm C0r "$PKG5" "$RUNS/p5c_c0r" p5c/c0r "$RUNS/p5c_c0" ;;
  SKIP) log "C0R_DECISION SKIP: $(head -n 1 $ROOT/C0R_DECISION.txt | cut -c1-200)" ;;
  *) log "C0R_DECISION ABSENT: no C0r now (the pod is not idled on the MM; a C0r decided later runs after F-seed, by hand)" ;;
esac
# ---- what runs next is read NOW (never at chain start)
if [ -s "$ROOT/after_f.sh" ]; then
  log "ZZFC3_HANDOFF after_f.sh sha256 $(sha256sum $ROOT/after_f.sh | cut -c1-16)"
  exec bash "$ROOT/after_f.sh" "$PKG"
fi
log "NO after_f.sh: the field replicates run as field_chain2.sh would have run them"
"$PY" "$ROOT/rep_argv.py" > "$ROOT/rep_argv.log" 2>&1; grep -q "^ZZREP_ARGV_WRITTEN" "$ROOT/rep_argv.log" || fail rep_argv "$(tail -n 1 $ROOT/rep_argv.log | cut -c1-200)"
log "REP_ARGV_OK $(grep '^ARGV' $ROOT/rep_argv.log | tr '\n' ' ')"
run_arm Cs refe_field_p7_ctrl_s1 "$PKG" "$RUNS/p7field_cs"
wait_arm Cs "$RUNS/p7field_cs"
post_arm Cs "$PKG" "$RUNS/p7field_cs" p7field/cs "$RUNS/p7field_c"
run_arm Fs refe_field_p7_s1 "$PKG" "$RUNS/p7field_fs"
wait_arm Fs "$RUNS/p7field_fs"
post_arm Fs "$PKG" "$RUNS/p7field_fs" p7field/fs "$RUNS/p7field_f"
log "ZZFC3_DONE"
