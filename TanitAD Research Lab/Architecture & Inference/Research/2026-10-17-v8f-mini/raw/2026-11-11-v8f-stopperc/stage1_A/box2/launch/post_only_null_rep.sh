#!/bin/bash
# post_only_null_rep.sh (MM 2026-11-13 ~21:30). STOPPERC stage 1, arm NULL_rep: the box-2 job FINISHED (STAGE_20261113_192855.done, RUN.done at 20:08:50),
# then the chain stopped BEFORE its dev-box steps ("dev box C: free 275 MB: the bank pull would fill it", rc 6, 20:09:40). A chain resume would relaunch the
# box-2 job, and the trainer would RESUME from the existing step-3,000 ckpt_last.pt ("resume wins", v8f_mini_train.py:243) and append to its log: a mixed record.
# This script runs ONLY the dev-box tail of run_one, with the chain's own setup and the same STOPBAL helper library, line for line (stopperc_chain.sh run_one,
# from the C: check to "banked"). It writes done/NULL_rep.done only if every step passes; the chain is then resumed and writes its own DONE.txt.
set -u
TAG=NULL_rep; G=20261113_192855
HERE=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/stage1_A/box2
COMMIT="3e264bbf3fa2f6f17bea448de09aa972faaf7725"
BAL_DIR=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/box2
export STOPBAL_STATE=C:/Users/Admin/stopperc_state; STATE="$STOPBAL_STATE"
export STOPBAL_PULLS=C:/Users/Admin/t2v8f_pulls; export B2_ROOT="D:/tanitad/stopperc"; export STOPBAL_COMMIT="$COMMIT"
export STOPBAL_SSH="$(command -v ssh)" STOPBAL_SFTP="$(command -v sftp)" STOPBAL_PKG="$HERE" STOPBAL_DRY=0
BUSY_DEV=C:/Users/Admin/box2_ship/BOX2_BUSY.txt; ASIDE_DEV=C:/Users/Admin/box2_ship/busy_aside
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/chain.log"; }
[ -e "$STATE/done/$TAG.done" ] && { echo "already done"; exit 0; }
if ! ( set -o noclobber; echo "stopperc post-only $TAG $$ $(date '+%F %T')" > "$BUSY_DEV" ) 2>/dev/null; then echo "REFUSED: box2 busy ($(head -c 120 "$BUSY_DEV"))"; exit 12; fi
OWN_MARK="$(cat "$BUSY_DEV")"
release_box() { if [ -e "$BUSY_DEV" ] && [ "$(cat "$BUSY_DEV" 2>/dev/null)" = "$OWN_MARK" ]; then mv "$BUSY_DEV" "$ASIDE_DEV/BOX2_BUSY.stopperc_postonly.$(date +%Y%m%d_%H%M%S).txt"; fi; }
trap 'release_box' EXIT
source "$BAL_DIR/stages/stopbal_common.sh"
render() { local t="$1" o="$2"; shift 2; local a=() kv; for kv in "$@"; do a+=(--set "$kv"); done
  "$PYV" -I "$HERE/stopperc_check.py" render --template "$HERE/jobs/$t" --out "$o" --set "ROOT=$B2_WIN_ROOT" "${a[@]}" || die 16 "could not render $t"; }
STOPBAL_STAGE="post-only:$TAG"
W=""; REUSE_FAILED="0"; [ -e "$STATE/NULL_REUSE_DIFFERS" ] && REUSE_FAILED="1"
R="$STATE/runs/$TAG" RR="$B2_ROOT/runs/$TAG"
# the box-2 job's own success marker and evidence must exist (what b2_wait returned DONE on)
b2_exists "$RR/STAGE_$G.done" || die 12 "no STAGE_$G.done on box 2"
b2_exists "$RR/RUN.done" || die 12 "no RUN.done on box 2"
log "post-only $TAG: box-2 job $G finished earlier (STAGE_$G.done, RUN.done); running the dev-box steps of run_one"
# ---- from here: stopperc_chain.sh run_one, verbatim
  [ "$(c_free_mb)" -ge "${STOPPERC_NEED_C_MB:-2000}" ] || die 13 "dev box C: free $(c_free_mb) MB: the bank pull would fill it"
  BD="$STOPBAL_PULLS/stopperc_$TAG" OUTW="D:/tanitad/runs/v8f_stopperc/$TAG"; mkdir -p "$BD"
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
  CG="$(date +%Y%m%d_%H%M%S)c"; printf 'bank verified on the dev box %s\n' "$(date '+%F %T')" > "$R/BANK_VERIFIED_$CG"; put_verified "$R/BANK_VERIFIED_$CG" "$RR/BANK_VERIFIED_$CG"
  render stopperc_stage_job.cmd "$R/clean.$CG.cmd" "COMMIT=$COMMIT" "GID=$CG" "ARGS=cleanup --tag $TAG --gid $CG" "MARKDIR=@ROOT@\\stopperc\\runs\\$TAG" "MARK=CLEAN" "W=$W" "REUSE_FAILED=$REUSE_FAILED"
  b2_put "$R/clean.$CG.cmd" "$RR/clean.cmd"; b2_launch "$RR/clean.cmd" "$RR/clean.log"
  [ "$(b2_wait "$RR/CLEAN_$CG.done" "$RR/CLEAN_$CG.fail" "${STOPPERC_POLLS_CLEAN:-30}")" = DONE ] || die 12 "the cleanup job did not finish ($RR/clean.log)"
  pull_verified "$RR/CLEANUP.json" "$BD/CLEANUP.json" || die 15 "bank of CLEANUP.json"
  say "banked $TAG in $BD"
# ---- end of the verbatim part
echo "$TAG $(date '+%F %T') (dev-box steps by post_only_null_rep.sh after the C:-low stop)" > "$STATE/done/$TAG.done"
log "done $TAG (post-only: box-2 job $G + dev-box bank steps)"
