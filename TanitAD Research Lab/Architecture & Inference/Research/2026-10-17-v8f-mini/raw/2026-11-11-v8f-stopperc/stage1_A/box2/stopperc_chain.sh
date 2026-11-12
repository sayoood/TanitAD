#!/usr/bin/env bash
# stopperc_chain.sh -- STOPPERC stage 1 on box 2 (PREREG_V8F_STOPPERC section 5; code of record 3e264bbf3fa2f6f17bea448de09aa972faaf7725).  NOT RUN: written and unit-tested only (see README.md, "What is verified").
# Order: busy-marker check (REFUSE, never wait, while any other owner holds box 2) -> claim -> ship (tree at the commit + camera table, every put sha-checked) -> W smoke (A_s0 selfcheck-only, W read by stop_aux_w_from_smoke.py)
#        -> per run [argv pin, gate (stopperc_check.py stage: G-DVB / G-LIVE + token), train under the token, post, pull sha-checked, bank, cleanup]:  NULL_s0, (reuse check), NULL_rep2, A_s0, A_s1, then NULL_s1 / NULL_rep ONLY if the reuse check says DIFFERS.
# Cost (ESTIMATED from the card, ~40 min per NULL run and ~44 min per A run): 40 + 40 + 2 x 44 = 168 min = 2.8 h; with the two conditional NULL runs 248 min = 4.1 h.  Plus ship, W smoke, gates and pulls.  Refused above 12 h (PROJ_H below).
# Sentinels live in their own files only: $STATE/STOP.txt ("ZZ STOPPERC-STOP"), $STATE/DONE.txt ("ZZ STOPPERC-DONE").  A refused gate of one run (stage rc 11) stops the chain: every arm is read against the NULL, so a hole is not skipped.
# Launch:  STOPPERC_SSH=$(command -v ssh) STOPPERC_SFTP=$(command -v sftp) bash stopperc_chain.sh      (relaunch: "C:\Program Files\Git\usr\bin\bash.exe" -l, see README)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; HERE="$(cygpath -m "$HERE" 2>/dev/null || echo "$HERE")"
COMMIT="3e264bbf3fa2f6f17bea448de09aa972faaf7725"
[ "${STOPPERC_DRY:-0}" = 1 ] && [ -n "${STOPPERC_TEST_COMMIT:-}" ] && COMMIT="$STOPPERC_TEST_COMMIT"      # TEST HOOK (dry run only: stub transport + STOPPERC_DRY=1): a throw-away repo cannot carry the code-of-record sha; a real launch never sets STOPPERC_DRY
TABLE_SHA="77b6e8dc63d07d90548d09182fd5b8073e81079e224e8ce672dda11d2c3ae361"; TABLE_DEV="${STOPPERC_TABLE_DEV:-C:/Users/Admin/stopaux_extr/stopaux_extr_join4572.json}"
WSCRIPT_SHA="064baba6e7385ee1df475e9e3206eb5cf74360bf6b3c8d3637823ba27bd4de63"      # stack/scripts/stop_aux_w_from_smoke.py at the commit (MEASURED on the export)
BASE_CHECK_SHA="70e260a697a6e3e23b33a37e6781e34ef4d482bd80b5e70d0e3964c19a8e5212"   # theory/stopbal/box2/stopbal_check.py, shipped unchanged beside stopperc_check.py
BAL_DIR="${STOPPERC_BAL_DIR:-E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/box2}"
BAL_NULL_CKPT="${STOPPERC_BAL_NULL_CKPT:-C:/Users/Admin/t2v8f_pulls/stopbal_BAL-NULL_s0/ckpt_last.pt}"
REPO_DEV="${STOPPERC_REPO:-C:/Users/Admin/tanitad-push}"
export STOPBAL_STATE="${STOPPERC_STATE:-C:/Users/Admin/stopperc_state}"; STATE="$STOPBAL_STATE"
export STOPBAL_PULLS="${STOPPERC_PULLS:-C:/Users/Admin/t2v8f_pulls}"; export B2_ROOT="D:/tanitad/stopperc"; export STOPBAL_COMMIT="$COMMIT"
export STOPBAL_SSH="${STOPPERC_SSH:-}" STOPBAL_SFTP="${STOPPERC_SFTP:-}" STOPBAL_PKG="$HERE" STOPBAL_DRY="${STOPPERC_DRY:-0}"
BUSY_DEV="${STOPPERC_BUSY_DEV:-C:/Users/Admin/box2_ship/BOX2_BUSY.txt}"; ASIDE_DEV="${STOPPERC_ASIDE_DEV:-C:/Users/Admin/box2_ship/busy_aside}"
PYV="${STOPPERC_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
mkdir -p "$STATE/done" "$ASIDE_DEV" || exit 2
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
refuse_early() { echo "ZZ STOPPERC-STOP start $1 rc=$2" > "$STATE/STOP.txt"; echo "REFUSED: $1" >&2; exit "$2"; }
for f in stopperc_check.py argvs_stopperc.txt jobs/stopperc_stage_job.cmd jobs/stopperc_ship_job.cmd; do [ -e "$HERE/$f" ] || refuse_early "package-incomplete ($f missing in $HERE)" 8; done
for f in stopbal_check.py stopbal_ship_tree.py stopbal_refresh.py stages/stopbal_common.sh; do [ -e "$BAL_DIR/$f" ] || refuse_early "stopbal-package-incomplete ($f missing in $BAL_DIR)" 8; done
[ "$(sha256sum "$BAL_DIR/stopbal_check.py" | cut -d' ' -f1)" = "$BASE_CHECK_SHA" ] || refuse_early "stopbal_check.py differs from the pinned base" 8
[ "$(sha256sum "$TABLE_DEV" | cut -d' ' -f1)" = "$TABLE_SHA" ] || refuse_early "camera table $TABLE_DEV does not hash to $TABLE_SHA" 8
if [ -e "$STATE/STOP.txt" ] && [ "${STOPPERC_RESUME:-0}" != "1" ]; then echo "REFUSED: $STATE/STOP.txt exists ($(cat "$STATE/STOP.txt")); inspect, then rerun with STOPPERC_RESUME=1" >&2; exit 2; fi
[ "${STOPPERC_RESUME:-0}" = "1" ] && [ -e "$STATE/STOP.txt" ] && mv "$STATE/STOP.txt" "$STATE/STOP.$(date +%Y%m%d_%H%M%S).txt"
# ---- box 2 must be free: another owner's marker is a REFUSAL (the SLOWWM chain holds it until ~23:30 on 2026-11-12, ESTIMATED), not a wait
if ! ( set -o noclobber; echo "stopperc chain $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then refuse_early "box2-busy ($BUSY_DEV exists: $(head -c 120 "$BUSY_DEV" 2>/dev/null))" 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.stopperc.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
# the STOPBAL transport/guard functions (ssh/sftp absolute paths, put_verified, pull_verified, need_*): sourced in place, driven by the exports above
source "$BAL_DIR/stages/stopbal_common.sh"
PKG="$HERE"; PYC="$PYV -I $HERE/stopperc_check.py"
render() { local t="$1" o="$2"; shift 2; local a=() kv; for kv in "$@"; do a+=(--set "$kv"); done
  "$PYV" -I "$HERE/stopperc_check.py" render --template "$HERE/jobs/$t" --out "$o" --set "ROOT=$B2_WIN_ROOT" "${a[@]}" || die 16 "could not render $t"; }
stop() { local d=""; [ -s "$STATE/stage_detail.txt" ] && d=" :: $(tr '\r\n' '  ' < "$STATE/stage_detail.txt" | head -c 500)"; echo "ZZ STOPPERC-STOP $1 $2 rc=$3$d" > "$STATE/STOP.txt"; log "stop at $1: $2 (rc $3)"; release_box; exit "$3"; }
stage_rc() { case "$2" in 11) stop "$1" "gate-or-pin-refused" 4;; 12) stop "$1" "job-or-transfer-failed" 5;; 13) stop "$1" "disk-low" 6;; 14) stop "$1" "box2-busy-or-gpu-not-idle" 12;; 15) stop "$1" "pull-sha-mismatch" 7;;
  16) stop "$1" "package-commit-or-transport" 8;; 17) stop "$1" "ship-refused" 9;; 19) stop "$1" "box2-timeout-or-unreadable" 11;; 20) stop "$1" "wsmoke-refused (probe_n_pos 0 or selfcheck unreadable; the sampler is not changed here)" 14;; *) stop "$1" "stage-crashed" 5;; esac; }
ORDER=(NULL_s0 NULL_rep2 A_s0 A_s1)
PROJ_H="$("$PYV" -I -c "print(round((40+40+44+44)/60, 2))")"; log "projected ${PROJ_H} h (ESTIMATED, card section 5), limit 12 h"
"$PYV" -I -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 12 else 1)" "$PROJ_H" || stop projection "above-12h-rule (goes to the PI)" 3

ship_stage() {
  STOPBAL_STAGE="ship"; need_free_gb "$NEED_GB_SHIP"
  local S="$STATE/ship/$COMMIT"; mkdir -p "$S"
  if [ ! -s "$S/stopbal_tree.tar" ]; then
    "$PYV" -I "$BAL_DIR/stopbal_ship_tree.py" build --repo "$REPO_DEV" --commit "$COMMIT" --export-list "${STOPPERC_EXPORT_LIST:-E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
      || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
  fi
  local T; T="$(sha256sum "$S/stopbal_tree.tar" | cut -d' ' -f1)"
  grep -q "\"tar_sha256\": \"$T\"" "$S/stopbal_tree_manifest.json" && grep -q "\"commit\": \"$COMMIT\"" "$S/stopbal_tree_manifest.json" || die 16 "the tree tar does not match its manifest / commit"
  b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/data" "$B2_ROOT/runs" "$B2_ROOT/runs/W_SMOKE"
  put_verified "$S/stopbal_tree.tar" "$B2_ROOT/ship/stopbal_tree.tar"; put_verified "$S/stopbal_tree_manifest.json" "$B2_ROOT/ship/stopbal_tree_manifest.json"
  put_verified "$BAL_DIR/stopbal_refresh.py" "$B2_ROOT/ship/stopbal_refresh.py"
  put_verified "$BAL_DIR/stopbal_check.py" "$B2_ROOT/check/stopbal_check.py"; put_verified "$HERE/stopperc_check.py" "$B2_ROOT/check/stopperc_check.py"; put_verified "$HERE/argvs_stopperc.txt" "$B2_ROOT/check/argvs_stopperc.txt"
  put_verified "$TABLE_DEV" "$B2_ROOT/data/stopaux_extr_join4572.json"
  [ "$(b2_sha256 "$B2_ROOT/data/stopaux_extr_join4572.json")" = "$TABLE_SHA" ] || die 12 "the camera table on box 2 does not hash to $TABLE_SHA"
  local G; G="$(date +%Y%m%d_%H%M%S)"; render stopperc_ship_job.cmd "$S/ship.$G.cmd" "COMMIT=$COMMIT" "GID=$G"; b2_put "$S/ship.$G.cmd" "$B2_ROOT/ship/ship.cmd"; b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
  local r; r="$(b2_wait "$B2_ROOT/ship/SHIP_$G.done" "$B2_ROOT/ship/SHIP_$G.fail" "${STOPPERC_POLLS_SHIP:-30}")"; say "ship job marker=$r"
  b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$G.json" 2> /dev/null
  [ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r)"
  "$PYV" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); assert r.get('ok') is True and r.get('commit') == sys.argv[2] and not r.get('verify_failures'), r" "$S/SHIP_RESULT.$G.json" "$COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $COMMIT"
}

wsmoke_stage() {      # A_s0 selfcheck-only at W 1.0 on box 2; W = goal / stop_aux from the probe, rounded to 4 decimals by the commit's own script; probe_n_pos 0 -> rc 20, STOP
  STOPBAL_STAGE="wsmoke"; need_free_gb "$NEED_GB_RUN"; need_gpu_idle
  local R="$STATE/runs/W_SMOKE" RR="$B2_ROOT/runs/W_SMOKE" G; G="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$R"
  render stopperc_stage_job.cmd "$R/w.$G.cmd" "COMMIT=$COMMIT" "GID=$G" "ARGS=wsmoke --gid $G" "MARKDIR=@ROOT@\\stopperc\\runs\\W_SMOKE" "MARK=STAGE" "W=" "REUSE_FAILED="
  b2_put "$R/w.$G.cmd" "$RR/w.cmd"; b2_launch "$RR/w.cmd" "$RR/w.log"
  local r; r="$(b2_wait "$RR/STAGE_$G.done" "$RR/STAGE_$G.fail" "${STOPPERC_POLLS_WSMOKE:-60}")"; say "wsmoke marker=$r"
  [ "$r" != TIMEOUT ] || die 19 "the W smoke did not finish ($RR/w.log)"
  [ "$r" = DONE ] || die 20 "W smoke refused: $(b2_cat "$RR/STAGE_$G.fail" | head -c 400 | tr '\n' ' ')"
  pull_verified "$RR/selfcheck.json" "$R/selfcheck.json" || die 15 "pull of the W smoke selfcheck.json"
  mkdir -p "$STATE/wscript"; git -C "$REPO_DEV" show "$COMMIT:stack/scripts/stop_aux_w_from_smoke.py" > "$STATE/wscript/stop_aux_w_from_smoke.py" 2> /dev/null || die 16 "cannot read the W script at $COMMIT"
  [ "$(sha256sum "$STATE/wscript/stop_aux_w_from_smoke.py" | cut -d' ' -f1)" = "$WSCRIPT_SHA" ] || die 16 "the W script differs from the pinned one"
  local W; W="$("$PYV" -I "$STATE/wscript/stop_aux_w_from_smoke.py" --selfcheck "$R/selfcheck.json" --digits 4)" || die 20 "the W script refused the smoke selfcheck"
  W="$(echo "$W" | tr -d '\r\n ')"; echo "$W" | grep -Eq '^[0-9]+\.[0-9]{4}$' || die 20 "W '$W' is not a 4-decimal number"
  "$PYV" -I -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.0 else 1)" "$W" || die 20 "W '$W' is not positive (the A argv would be refused at A_s0, after two NULL runs)"
  echo "$W" > "$STATE/W.txt"; log "W = $W (goal / stop_aux at step 0, probe_n_pos in $R/selfcheck.json)"
}

run_one() {           # run_one <tag>: argv -> box 2 -> ONE job (gate + token + train + post) -> sha-checked pull -> bank -> cleanup
  local TAG="$1"; STOPBAL_STAGE="run:$TAG"
  local W=""; case "$TAG" in A_*) W="$(cat "$STATE/W.txt" 2> /dev/null)"; [ -n "$W" ] || die 16 "no W: the W smoke has not run";; esac
  local REUSE_FAILED="0"; [ -e "$STATE/NULL_REUSE_DIFFERS" ] && REUSE_FAILED="1"
  local R="$STATE/runs/$TAG" RR="$B2_ROOT/runs/$TAG"; mkdir -p "$R"
  need_free_gb "$NEED_GB_RUN"; need_gpu_idle
  for f in stopperc_check.py stopbal_check.py argvs_stopperc.txt; do
    [ "$(b2_sha256 "$B2_ROOT/check/$f")" = "$(sha256sum "$( [ "$f" = stopbal_check.py ] && echo "$BAL_DIR" || echo "$HERE")/$f" | cut -d' ' -f1)" ] || die 16 "$f on box 2 differs from the package: re-run the ship stage"; done
  STOPPERC_W="$W" STOPPERC_NULL_REUSE_FAILED="$REUSE_FAILED" "$PYV" -I "$HERE/stopperc_check.py" argv --tag "$TAG" --out-txt "$R/argv.txt"; local rc=$?
  [ "$rc" -ne 23 ] || die 11 "argv refused for $TAG (blocked tag or unfilled W)"; [ "$rc" -eq 0 ] || die 16 "argv refused for $TAG (rc $rc)"
  b2_mkdir "$RR"; put_verified "$R/argv.txt" "$RR/argv.txt"
  local G; G="$(date +%Y%m%d_%H%M%S)"
  render stopperc_stage_job.cmd "$R/stage.$G.cmd" "COMMIT=$COMMIT" "GID=$G" "ARGS=stage --tag $TAG --gid $G" "MARKDIR=@ROOT@\\stopperc\\runs\\$TAG" "MARK=STAGE" "W=$W" "REUSE_FAILED=$REUSE_FAILED"
  b2_put "$R/stage.$G.cmd" "$RR/stage.cmd"; b2_launch "$RR/stage.cmd" "$RR/stage.log"
  local r; r="$(b2_wait "$RR/STAGE_$G.done" "$RR/STAGE_$G.fail" "${STOPPERC_POLLS_RUN:-1100}")"; say "stage marker=$r"
  [ "$r" != TIMEOUT ] || die 19 "the box-2 job did not finish within the poll limit; it may still run ($RR/stage.log)"
  if [ "$r" != DONE ]; then local why; why="$(b2_cat "$RR/STAGE_$G.fail" | head -c 400 | tr '\n' ' ')"
    case "$why" in preflight-fail*|pin-fail*|gate-fail*|token-fail*) die 11 "refused ($why)";; disk-low*) die 13 "$why";; *) die 12 "box-2 job failed ($why)";; esac; fi
  [ "$(c_free_mb)" -ge "${STOPPERC_NEED_C_MB:-2000}" ] || die 13 "dev box C: free $(c_free_mb) MB: the bank pull would fill it"
  local BD="$STOPBAL_PULLS/stopperc_$TAG" OUTW="D:/tanitad/runs/v8f_stopperc/$TAG"; mkdir -p "$BD"
  for f in ckpt_last.pt summary.json train_log.jsonl; do pull_verified "$OUTW/$f" "$BD/$f" || die 15 "bank of $f"; done
  for f in RUN.done gate/gate_token.json gate/selfcheck.json; do pull_verified "$RR/$f" "$BD/$(basename "$f")" || die 15 "bank of $f"; done
  cp "$R/argv.txt" "$BD/argv.txt"
  STOPPERC_W="$W" "$PYV" -I "$HERE/stopperc_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag "$TAG" --commit "$COMMIT" > /dev/null || die 11 "the token does not bind this commit and argv"
  "$PYV" -I - "$BD" "$TAG" "$COMMIT" << 'PYEOF' || die 15 "bank manifest: the RUN.done evidence does not match the banked files"
import hashlib, json, os, sys
bd, tag, commit = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ev = json.load(open(os.path.join(bd, "RUN.done"), encoding="utf-8"))["evidence"]
files = {f: {"bytes": os.path.getsize(os.path.join(bd, f)), "sha256": sha(os.path.join(bd, f))} for f in sorted(os.listdir(bd)) if os.path.isfile(os.path.join(bd, f)) and f not in ("BANK_MANIFEST.json", "BANK.done", "CLEANUP.json")}
assert files["ckpt_last.pt"]["sha256"] == ev["ckpt_sha256"] and files["summary.json"]["sha256"] == ev["summary_sha256"] and files["train_log.jsonl"]["sha256"] == ev["train_log_sha256"], "banked sha256 != RUN.done evidence"
json.dump({"tag": tag, "commit": commit, "files": files}, open(os.path.join(bd, "BANK_MANIFEST.json"), "w"), indent=1)
open(os.path.join(bd, "BANK.done"), "w").write("verified against box 2 by sha256\n")
PYEOF
  local CG; CG="$(date +%Y%m%d_%H%M%S)c"; printf 'bank verified on the dev box %s\n' "$(date '+%F %T')" > "$R/BANK_VERIFIED_$CG"; put_verified "$R/BANK_VERIFIED_$CG" "$RR/BANK_VERIFIED_$CG"
  render stopperc_stage_job.cmd "$R/clean.$CG.cmd" "COMMIT=$COMMIT" "GID=$CG" "ARGS=cleanup --tag $TAG --gid $CG" "MARKDIR=@ROOT@\\stopperc\\runs\\$TAG" "MARK=CLEAN" "W=$W" "REUSE_FAILED=$REUSE_FAILED"
  b2_put "$R/clean.$CG.cmd" "$RR/clean.cmd"; b2_launch "$RR/clean.cmd" "$RR/clean.log"
  [ "$(b2_wait "$RR/CLEAN_$CG.done" "$RR/CLEAN_$CG.fail" "${STOPPERC_POLLS_CLEAN:-30}")" = DONE ] || die 12 "the cleanup job did not finish ($RR/clean.log)"
  pull_verified "$RR/CLEANUP.json" "$BD/CLEANUP.json" || die 15 "bank of CLEANUP.json"
  say "banked $TAG in $BD"
}

reuse_check() {       # the card's NULL reuse rule, after NULL_s0 is banked: NULL_s0's model tensors against BAL-NULL_s0's (pinned sha256). Reported; DIFFERS appends the two conditional NULL runs.
  [ -s "$STATE/NULL_REUSE.json" ] && return 0
  "$PYV" -I "$HERE/stopperc_check.py" reuse --ckpt "$STOPBAL_PULLS/stopperc_NULL_s0/ckpt_last.pt" --bal-ckpt "$BAL_NULL_CKPT" --out "$STATE/NULL_REUSE.json" > "$STATE/reuse.txt" 2>&1
  log "NULL reuse: $(tr '\n' ' ' < "$STATE/reuse.txt" | head -c 200)"
  grep -q '"null_reuse": "DIFFERS"' "$STATE/NULL_REUSE.json" && : > "$STATE/NULL_REUSE_DIFFERS"
  grep -q '"null_reuse": "UNKNOWN"' "$STATE/NULL_REUSE.json" && stop reuse "null-reuse-unknown (an input was unreadable; the PI/MM decides whether NULL_s1 and NULL_rep run)" 13
  return 0
}

[ -e "$STATE/done/ship.done" ] || { log "start ship"; ( ship_stage ); rc=$?; [ "$rc" -eq 0 ] && echo "ship $(date '+%F %T')" > "$STATE/done/ship.done" || stage_rc ship "$rc"; }
[ -e "$STATE/done/wsmoke.done" ] || { log "start wsmoke"; ( wsmoke_stage ); rc=$?; [ "$rc" -eq 0 ] && echo "wsmoke $(date '+%F %T')" > "$STATE/done/wsmoke.done" || stage_rc wsmoke "$rc"; }
i=0
while [ "$i" -lt "${#ORDER[@]}" ]; do
  t="${ORDER[$i]}"; i=$((i+1))
  if [ ! -e "$STATE/done/$t.done" ]; then log "start $t"; ( run_one "$t" ); rc=$?; if [ "$rc" -eq 0 ]; then echo "$t $(date '+%F %T')" > "$STATE/done/$t.done"; log "done $t"; else stage_rc "$t" "$rc"; fi; fi
  if [ "$t" = NULL_s0 ]; then reuse_check; [ -e "$STATE/NULL_REUSE_DIFFERS" ] && ORDER+=(NULL_s1 NULL_rep); fi
done
echo "ZZ STOPPERC-DONE $(date '+%F %T') commit=$COMMIT runs=${ORDER[*]}" > "$STATE/DONE.txt"; log "all runs banked"
exit 0
