#!/usr/bin/env bash
# GENERATED from stages/slowwm_run.sh by the MM (2026-11-12) to adopt the finished pilot job 20261112_011153: only the source path, the GID and the launch line differ.
# slowwm_run.sh <PILOT_SLOW_s0|SLOW_s0|PCA-B_s0|SLOW_s1|PCA-B_s1|SLOW_rep>: argv -> box 2 -> ONE box-2 job (preflight, pin, gate with BUILT-lever assertions (wm_basis_readback) and the token, then the trainer under the token, then post)
# -> pull and verify -> bank -> only then the cleanup job (the run's own temporary gate dirs, by literal path, logged). Never a checkpoint, never a basis cache.
# PILOT: 200 steps of SLOW_s0 with its exact argv plus --steps 200, a separate out dir; the same job also gates the five full-length argvs; PILOT.json carries the timing and the projection.
# A full tag banks ckpt / summary / log / token / selfcheck / RUN.done to $PULLS/slowwm_<tag>/, each sha256-checked against box 2's own copy.
source E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/box2/stages/slowwm_common.sh; TAG="${1:?tag}"; SLOWWM_STAGE="run:$TAG"
ALL_TAGS="SLOW_s0 PCA-B_s0 SLOW_s1 PCA-B_s1 SLOW_rep"
case "$TAG" in
  PILOT_SLOW_s0) PILOT=1;;
  SLOW_s0|PCA-B_s0|SLOW_s1|PCA-B_s1|SLOW_rep) PILOT=0;;
  *) die 16 "unknown tag $TAG";;
esac
need_commit; need_pkg slowwm_check.py frozen/argvs_slowwm.json jobs/slowwm_stage_job.cmd
R="$STATE/runs/$TAG"; RR="$B2_ROOT/runs/$TAG"; mkdir -p "$R"
need_free_gb "$NEED_GB_RUN"
need_gpu_idle
b2_exists "$B2_ROOT/check/slowwm_check.py" || die 16 "$B2_ROOT/check/slowwm_check.py is not on box 2: run the ship stage first"
[ "$(b2_sha256 "$B2_ROOT/check/slowwm_check.py")" = "$(sha256sum "$PKG/slowwm_check.py" | cut -d' ' -f1)" ] || die 16 "the checker on box 2 differs from the package: re-run the ship stage"
[ "$(b2_sha256 "$B2_ROOT/check/argvs_slowwm.json")" = "$(sha256sum "$PKG/frozen/argvs_slowwm.json" | cut -d' ' -f1)" ] || die 16 "the frozen argv file on box 2 differs from the package: re-run the ship stage"
if [ "$PILOT" = 1 ]; then PF="--pilot"; else PF="--tag $TAG"; fi
"$PY" -I "$PKG/slowwm_check.py" argv $PF --out-txt "$R/argv.txt"; rc=$?
[ "$rc" -ne 23 ] || die 23 "argv refused for $TAG: blocked tag or placeholder path"
[ "$rc" -eq 0 ] || die 16 "argv refused for $TAG (rc $rc)"
b2_mkdir "$RR"; put_verified "$R/argv.txt" "$RR/argv.txt"
GID=20261112_011153   # ADOPT (MM 2026-11-12): the stage job of this GID finished DONE on box 2 at 01:58:43 after the dev-box driver died in the 01:32 Windows Update restart; it is adopted, not launched again
render slowwm_stage_job.cmd "$R/stage.$GID.cmd" "COMMIT=$SLOWWM_COMMIT" "GID=$GID" "ARGS=stage --tag $TAG --gid $GID" "MARKDIR=@ROOT@\\slowwm\\runs\\$TAG" "MARK=STAGE"
: adopt: no launch, the job already ran   # was: b2_put "$R/stage.$GID.cmd" "$RR/stage.cmd"; b2_launch "$RR/stage.cmd" "$RR/stage.log"
r="$(b2_wait "$RR/STAGE_$GID.done" "$RR/STAGE_$GID.fail" "${SLOWWM_POLLS_RUN:-1100}")"; say "stage marker=$r"
[ "$r" != TIMEOUT ] || die 19 "the box-2 job did not finish within the poll limit; it may still run ($RR/stage.log, $RR/logs)"
if [ "$r" != DONE ]; then
  why="$(b2_cat "$RR/STAGE_$GID.fail" | head -c 400 | tr '\n' ' ')"
  case "$why" in preflight-fail*|pin-fail*|gate-fail*|token-fail*) die 11 "refused ($why)";; disk-low*) die 13 "$why";; post-fail*) die 12 "post check failed ($why)";; *) die 12 "box-2 job failed ($why) (log $RR/stage.log)";; esac
fi
clean_after_bank() {   # BANK_VERIFIED_<gid> proves (on box 2) that the dev box pulled and sha-checked the bank; the cleanup job refuses without it
  local BD="$1" CG; CG="$(date +%Y%m%d_%H%M%S)c"
  printf 'bank verified on the dev box %s\n' "$(date '+%F %T')" > "$R/BANK_VERIFIED_$CG"; put_verified "$R/BANK_VERIFIED_$CG" "$RR/BANK_VERIFIED_$CG"
  render slowwm_stage_job.cmd "$R/clean.$CG.cmd" "COMMIT=$SLOWWM_COMMIT" "GID=$CG" "ARGS=cleanup --tag $TAG --gid $CG" "MARKDIR=@ROOT@\\slowwm\\runs\\$TAG" "MARK=CLEAN"
  b2_put "$R/clean.$CG.cmd" "$RR/clean.cmd"; b2_launch "$RR/clean.cmd" "$RR/clean.log"
  local c; c="$(b2_wait "$RR/CLEAN_$CG.done" "$RR/CLEAN_$CG.fail" "${SLOWWM_POLLS_CLEAN:-30}")"
  [ "$c" = DONE ] || die 12 "the cleanup job did not finish ($c): the temporary gate dirs on box 2 are still there (log $RR/clean.log)"
  pull_verified "$RR/CLEANUP.json" "$BD/CLEANUP.json" || die 15 "bank of CLEANUP.json"
  local d t
  if [ "$PILOT" = 1 ]; then
    for t in $TAG $ALL_TAGS; do d="D:/tanitad/slowwm/gate_out/$t"; b2_exists "$d" && die 12 "temporary dir $d still exists on box 2 after the cleanup"; done
    b2_exists "D:/tanitad/slowwm/pilot_out/SLOW_s0" && die 12 "temporary pilot out dir still exists on box 2"
  else
    d="D:/tanitad/slowwm/gate_out/$TAG"; b2_exists "$d" && die 12 "temporary dir $d still exists on box 2 after the cleanup"
  fi
  say "cleanup logged in $BD/CLEANUP.json"
}
if [ "$PILOT" = 1 ]; then
  PD="$STATE/pilot"; mkdir -p "$PD"
  for f in PILOT.json train_log.jsonl summary.json selfcheck.json gate_token.json; do pull_verified "$B2_ROOT/pilot/$f" "$PD/$f" || die 15 "pilot file $f: the pulled copy differs from box 2's (3 attempts)"; done
  "$PY" -I "$PKG/slowwm_check.py" project --pilot-dir "$PD" --out "$PD/PILOT.verified.json" --commit "$SLOWWM_COMMIT" || die 18 "the pilot cannot be re-derived here from its pulled log"
  "$PY" -I "$PKG/slowwm_check.py" verify-token --token "$PD/gate_token.json" --argv-file "$R/argv.txt" --tag "$TAG" --commit "$SLOWWM_COMMIT" > /dev/null || die 11 "the pilot token does not bind this commit and argv"
  clean_after_bank "$PD"
  say "pilot ok: files in $PD; temporary dirs removed on box 2"; exit 0
fi
[ "$(c_free_mb)" -ge "${SLOWWM_NEED_C_MB:-2000}" ] || die 13 "dev box C: free $(c_free_mb) MB < ${SLOWWM_NEED_C_MB:-2000} MB: the bank pull would fill it (disk-low)"
BD="$PULLS/slowwm_$TAG"; mkdir -p "$BD"; OUTW="D:/tanitad/runs/v8f_slowwm/$TAG"
for f in ckpt_last.pt summary.json train_log.jsonl; do pull_verified "$OUTW/$f" "$BD/$f" || die 15 "bank of $f: the pulled copy differs from box 2's (3 attempts)"; done
pull_verified "$RR/RUN.done" "$BD/RUN.done" || die 15 "bank of RUN.done"
pull_verified "$RR/gate/gate_token.json" "$BD/gate_token.json" || die 15 "bank of the token"; pull_verified "$RR/gate/selfcheck.json" "$BD/selfcheck.json" || die 15 "bank of selfcheck.json"
cp "$R/argv.txt" "$BD/argv.txt"
"$PY" -I "$PKG/slowwm_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag "$TAG" --commit "$SLOWWM_COMMIT" > /dev/null || die 11 "the token does not bind this commit and argv"
"$PY" -I - "$BD" "$TAG" "$SLOWWM_COMMIT" << 'PYEOF' || die 15 "bank manifest: the RUN.done evidence does not match the banked files"
import hashlib, json, os, sys
bd, tag, commit = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ev = json.load(open(os.path.join(bd, "RUN.done"), encoding="utf-8"))["evidence"]
files = {f: {"bytes": os.path.getsize(os.path.join(bd, f)), "sha256": sha(os.path.join(bd, f))} for f in sorted(os.listdir(bd)) if os.path.isfile(os.path.join(bd, f)) and f not in ("BANK_MANIFEST.json", "BANK.done", "CLEANUP.json")}
assert files["ckpt_last.pt"]["sha256"] == ev["ckpt_sha256"] and files["summary.json"]["sha256"] == ev["summary_sha256"] and files["train_log.jsonl"]["sha256"] == ev["train_log_sha256"], "banked sha256 != RUN.done evidence"
json.dump({"tag": tag, "commit": commit, "files": files}, open(os.path.join(bd, "BANK_MANIFEST.json"), "w"), indent=1)
open(os.path.join(bd, "BANK.done"), "w").write("verified against box 2 by sha256\n")
PYEOF
clean_after_bank "$BD"
say "banked $TAG in $BD"; say "ok"
