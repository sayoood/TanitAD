#!/usr/bin/env bash
# exec_chain.sh: the EXEC-STANDSTILL production rollouts (PREREG_EXEC_STANDSTILL.md section 3): C0 s0/s1 and DAGOACC s0/s1, each under the base executor
# (--exec-eps 1e-3) and under hold (--exec-eps 0.1), on ALL 2,399 eval-141 windows, CPU, ONE rollout at a time, 16 threads (the banked C0 runs used 16).
# Flags = the banked C0_s* flags (run_tool.sh: --smooth sg72 --scene logged, replan 10, causal bucket, ticks + heads) + --no-analyze + --dump-exec-ch.
# RESUME = launch exec_chain.cmd again (a rollout with done_<tag>.txt is skipped; an incomplete out dir is MOVED aside). Markers: raw/CHAIN.running (pid), raw/CHAIN.done, raw/CHAIN.fail.
# Gates (refuse -> CHAIN.fail): the exec tool's sha256 equals the pin; every checkpoint equals its SHA256SUMS.txt line.
set -u
EXW=E:/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill
EXP=/e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill
S=C:/Users/Admin/exec_standstill_scratch
SP=/c/Users/Admin/exec_standstill_scratch
OUT=$S/out
OUTP=$SP/out
TOOL=$S/tree/stack/scripts/v8f_t1_eval_exec.py
TOOLP=$SP/tree/stack/scripts/v8f_t1_eval_exec.py
TOOL_SHA=e042a1bd869d35ecb9de9af9811b8cecc9adbcb149e7096717c5ff66ce6ef42a
ART=E:/Projects/TanitAD-artifacts
ARTP=/e/Projects/TanitAD-artifacts
DATA="$ARTP/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
GRID="$DATA/eval_grid_windows.json"
GRIDW="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
export EXEC_TH=16
mkdir -p "$OUTP" "$EXP/logs" "$EXP/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$EXP/logs/chain.out"; }
fail() { log "ZZEXEC-FAIL $1"; echo "$(date +%FT%T) $1" > "$EXP/raw/CHAIN.fail"; exit "${2:-3}"; }
if [ -f "$EXP/raw/CHAIN.running" ]; then
  op=$(tr -d '\r\n ' < "$EXP/raw/CHAIN.running")
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZEXEC-ALREADY-RUNNING pid $op"; exit 9; fi
  mv "$EXP/raw/CHAIN.running" "$EXP/raw/_stale_CHAIN.running_$(date +%H%M%S)"
fi
echo $$ > "$EXP/raw/CHAIN.running"
trap 'echo "exited $(date +%FT%T)" > "$EXP/raw/CHAIN.running"' EXIT
for m in CHAIN.done CHAIN.fail; do [ -f "$EXP/raw/$m" ] && mv "$EXP/raw/$m" "$EXP/raw/_old_${m}_$(date +%H%M%S)"; done
got=$(sha256sum "$TOOLP" | cut -c1-64)
[ "$got" = "$TOOL_SHA" ] || fail "exec tool sha $got != pin $TOOL_SHA" 2
declare -A CK
for s in 0 1; do
  CK["XC0_s$s"]="$ART/v8f-recover/M2b_s${s}_recft_C0/ckpt_last.pt"
  CK["XDAG_s$s"]="$ART/v8f-dagoacc/M2b_s${s}_dagoacc/ckpt_last.pt"
done
for r in v8f-recover/M2b_s0_recft_C0 v8f-recover/M2b_s1_recft_C0 v8f-dagoacc/M2b_s0_dagoacc v8f-dagoacc/M2b_s1_dagoacc; do
  f="$ARTP/$r/ckpt_last.pt"
  [ -f "$f" ] || fail "checkpoint $r absent" 5
  g=$(sha256sum "$f" | cut -c1-64)
  w=$(awk '$2 ~ /ckpt_last.pt$/ {print $1}' "$ARTP/$r/SHA256SUMS.txt" | head -n 1 | tr -d '\r')
  [ -n "$w" ] && [ "$g" = "$w" ] || fail "$r ckpt sha $g != SHA256SUMS.txt $w" 6
done
log "ZZEXEC-START tool $TOOL_SHA; checkpoints match SHA256SUMS.txt; threads=$EXEC_TH"
run_one() {   # tag model-key eps
  local tag=$1 key=$2 eps=$3
  [ -f "$OUTP/done_$tag.txt" ] && { log "ZZEXEC-SKIP $tag"; return 0; }
  [ -d "$OUTP/$tag" ] && mv "$OUTP/$tag" "$OUTP/_incomplete_${tag}_$(date +%H%M%S)"
  local t0=$(date +%s)
  log "ZZEXEC-RUN $tag key=$key eps=$eps"
  bash "$EXP/code/run_tool.sh" "$TOOL" "$tag" "${CK[$key]}" "$OUT" "$GRIDW" --no-analyze --dump-exec-ch "$OUT/$tag/exec_ch_$tag.npz" --exec-eps "$eps"
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "model": "%s", "ckpt": "%s", "exec_eps": %s, "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$key" "${CK[$key]}" "$eps" "$TOOL_SHA" "$EXEC_TH" "$(( $(date +%s) - t0 ))" > "$OUTP/done_$tag.txt"
    log "ZZEXEC-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZEXEC-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
run_one XC0_s0_base XC0_s0 1e-3 || nfail=$((nfail+1))
run_one XC0_s0_hold XC0_s0 0.1 || nfail=$((nfail+1))
run_one XC0_s1_base XC0_s1 1e-3 || nfail=$((nfail+1))
run_one XC0_s1_hold XC0_s1 0.1 || nfail=$((nfail+1))
run_one XDAG_s0_base XDAG_s0 1e-3 || nfail=$((nfail+1))
run_one XDAG_s0_hold XDAG_s0 0.1 || nfail=$((nfail+1))
run_one XDAG_s1_base XDAG_s1 1e-3 || nfail=$((nfail+1))
run_one XDAG_s1_hold XDAG_s1 0.1 || nfail=$((nfail+1))
log "ZZEXEC-ROLLOUTS-DONE nfail=$nfail"
missing=""
for t in XC0_s0_base XC0_s0_hold XC0_s1_base XC0_s1_hold XDAG_s0_base XDAG_s0_hold XDAG_s1_base XDAG_s1_hold; do [ -f "$OUTP/done_$t.txt" ] || missing="$missing $t"; done
[ -z "$missing" ] || fail "missing done markers:$missing (see $OUT/<tag>.out)" 3
echo "$(date +%FT%T) eight rollouts done" > "$EXP/raw/CHAIN.done"
log "ZZEXEC-CHAIN-DONE"
