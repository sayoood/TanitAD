#!/bin/bash
# MM 2026-11-21: what the pod runs after after_f.sh (v2) ends: exec'd by after_f.sh's handoff() after ZZAF_DONE or ZZAF_FAIL; read only at that moment.
# PI go 2026-11-20 ("Follow your recommendations, acclerate progress."): the C1 seed-1 replicate C1r (refe_p5c_c1r), queued after the field chain.
#   C1r = P5-C's C1 exactly (ARGS_C1.txt) with --seed 0 -> 1 and --out p5c_c1 -> p5c_c1r, nothing else (checked here token by token).
#   It reads C1's +1.17 PDMS (P5-C, T1, one seed) against the rig's own training-seed noise (H-ESTIM-SEED-1).
#   1. package: $ROOT/c1r/PKG_C1R.txt (the dir) verified against $ROOT/c1r/expect_C1R.sha256; its launch_gate.py must register refe_p5c_c1r.
#      Absent at handoff -> wait up to 12 h (the MM ships it), then end.
#   2. argv: $ROOT/ARGS_C1r.txt, sha256 == $ROOT/c1r/ARGS_C1r.sha256, and its diff to ARGS_C1.txt == exactly the two tokens above.
#   3. fresh out: $RUNS/p5c_c1r must not exist; the GPU must hold no compute process.
#   4. run_arm (evidence -> launch gate PASS + token verify + preflight -> launch) -> wait 2,000 steps -> arm_post.sh (G-ADAM, other = p5c_c1) -> HF p5c/c1r.
# usage: after_af.sh <field PKG>.  Status: $ROOT/after_af.status (ZZAAF_DONE | ZZAAF_FAIL <stage> <why> | ZZAAF_SKIP <why>)
set -u
FPKG="${1:-}"
D=/workspace/data; ROOT=$D/refe_p5c; PY=/workspace/venv-teacher/bin/python; RUNS=$D/refe_runs; CR=$ROOT/c1r
S=$ROOT/after_af.status
log() { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
fail() { log "ZZAAF_FAIL $*"; exit 1; }
aside() {
  [ -e "$1/ckpt_last.pt" ] && mv -n "$1/ckpt_last.pt" "$1/ckpt_last.post_done.pt"
  [ -e "$1/ckpt_last.pt" ] && fail aside "$1/ckpt_last.pt still present"
  log "ASIDE $1/ckpt_last.pt -> ckpt_last.post_done.pt"
}
doneN() { [ -f "$1/summary.json" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('done') is True and d.get('steps')==int(sys.argv[2]) else 1)" "$1/summary.json" "$2" 2>/dev/null; }
wait_arm() {   # <ARM> <OUT> <STEPS>   (after_f.sh's wait_arm, byte for byte apart from fail)
  local A="$1" O="$2" N="$3" P; P="$(cut -d' ' -f1 "$ROOT/train_$A.pid")"
  while ! doneN "$O" "$N"; do
    if ! kill -0 "$P" 2>/dev/null; then sleep 60; doneN "$O" "$N" && break; fail "wait_$A" "pid $P gone, summary not done ($(tail -n 1 $ROOT/train_$A.log | cut -c1-160))"; fi
    sleep 300
  done
  for i in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 20; done
  kill -0 "$P" 2>/dev/null && fail "wait_$A" "pid $P alive 10 min after summary done"
  log "${A}_DONE"
}
run_arm() {    # <ARM> <PROFILE> <PKG> <OUT>   (field_chain3's run_arm)
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
post_arm() {   # <ARM> <PKG> <OUT> <HF prefix> [<OTHER out>]   (field_chain3's post_arm)
  local A="$1"
  bash "$ROOT/arm_post.sh" "$@"; grep -q "^ZZ${A}POST_OK" "$ROOT/post_$A.status" || fail "post_$A" "$(tail -n 1 $ROOT/post_$A.status)"
  log "${A}_POST_OK $(cut -c1-16 $ROOT/model_final_sha256_$A.txt)"
  aside "$3"
}
pkg_ok() { [ -d "$1/refe" ] && [ -s "$2" ] && (cd "$1" && sha256sum --quiet -c "$2") > /dev/null 2>&1; }
log "START pid $$ (after after_f.sh; field PKG ${FPKG:-none})"
# ---- 1. the package (wait up to 12 h for the MM to ship it)
i=0
while [ ! -s "$CR/PKG_C1R.txt" ] || [ ! -s "$CR/expect_C1R.sha256" ] || [ ! -s "$ROOT/ARGS_C1r.txt" ] || [ ! -s "$CR/ARGS_C1r.sha256" ]; do
  i=$((i+1)); [ $i -gt 72 ] && { log "ZZAAF_SKIP C1r package / argv not shipped 12 h after the handoff: the chain ends (C1r runs by hand later)"; exit 0; }
  [ $i = 1 ] && log "WAIT package C1R ($CR/PKG_C1R.txt + expect_C1R.sha256) + ARGS_C1r.txt + ARGS_C1r.sha256"
  sleep 600
done
PKG="$(cat $CR/PKG_C1R.txt)"
pkg_ok "$PKG" "$CR/expect_C1R.sha256" || fail pkg "package $PKG is not the shipped bytes ($CR/expect_C1R.sha256)"
grep -q '"C1r": "refe_p5c_c1r"' "$PKG/refe/launch_gate.py" || fail pkg "$PKG/refe/launch_gate.py does not register refe_p5c_c1r"
log "PKG_OK $PKG"
# ---- 2. the argv: the shipped sha AND exactly C1's argv with --seed 0 -> 1, --out p5c_c1 -> p5c_c1r
[ "$(sha256sum $ROOT/ARGS_C1r.txt | cut -c1-64)" = "$(cut -c1-64 $CR/ARGS_C1r.sha256)" ] || fail argv "ARGS_C1r.txt sha256 != $CR/ARGS_C1r.sha256"
d="$("$PY" - "$ROOT/ARGS_C1.txt" "$ROOT/ARGS_C1r.txt" "$RUNS" <<'EOF'
import sys
c1, c1r, runs = open(sys.argv[1]).read().split(), open(sys.argv[2]).read().split(), sys.argv[3]
assert len(c1) == len(c1r), ("length", len(c1), len(c1r))
diff = [(i, a, b) for i, (a, b) in enumerate(zip(c1, c1r)) if a != b]
want = {("0", "1"), (runs + "/p5c_c1", runs + "/p5c_c1r")}
got = {(a, b) for _, a, b in diff}
assert len(diff) == 2 and got == want, ("diff", diff)
for i, a, b in diff:
    assert c1[i - 1] == {"0": "--seed", runs + "/p5c_c1": "--out"}[a], ("flag", c1[i - 1], a)
assert c1.count("--seed") == 1 and c1.count("--out") == 1 and "--traj-label-table" in c1
print("ZZC1R_DIFF_OK", diff)
EOF
)" || fail argv "C1 -> C1r diff is not exactly --seed 0->1 and --out p5c_c1->p5c_c1r: $(echo "$d" | tail -n 1 | cut -c1-200)"
log "$d"
# ---- 3. a fresh out and a free GPU
[ -e "$RUNS/p5c_c1r" ] && fail fresh "$RUNS/p5c_c1r already exists"
[ -d "$RUNS/p5c_c1" ] || fail other "$RUNS/p5c_c1 (C1, the post-gate's other arm) is missing"
g="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+[[:space:]]*$')"
[ "$g" = 0 ] || fail gpu "the GPU holds $g compute process(es)"
# ---- 4. run, wait, post
run_arm C1r refe_p5c_c1r "$PKG" "$RUNS/p5c_c1r"
wait_arm C1r "$RUNS/p5c_c1r" 2000
post_arm C1r "$PKG" "$RUNS/p5c_c1r" p5c/c1r "$RUNS/p5c_c1"
log "ZZAAF_DONE (the C1r read is the MM's, on the dev box)"
