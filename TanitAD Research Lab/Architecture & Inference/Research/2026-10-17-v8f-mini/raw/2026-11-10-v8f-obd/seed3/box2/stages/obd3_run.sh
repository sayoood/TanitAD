#!/usr/bin/env bash
# obd3_run.sh <OBD-CTL_s10|OBD-BAND_s10|OBD-CTL_s11|OBD-BAND_s11>: argv -> box 2 -> ONE box-2 job (preflight, pin, gate with BUILT-lever assertions and the token, then the trainer under the token, then post)
# -> pull and verify -> bank -> only then the cleanup job (the run's own temporary gate dir, by literal path, logged). Never a checkpoint.
# Disk (>= 5 GB free on box 2's D:), GPU idle (no python.exe, used memory below OBD3_GPU_IDLE_MIB, default 3500) and the shipped checker files are asserted before the job starts.
# A tag banks ckpt / summary / log / token / selfcheck / RUN.done to $PULLS/obd_<tag>/, each sha256-checked against box 2's own copy.
source "$(dirname "${BASH_SOURCE[0]}")/obd3_common.sh"; TAG="${1:?tag}"; OBD3_STAGE="run:$TAG"
case "$TAG" in
  OBD-CTL_s10|OBD-BAND_s10|OBD-CTL_s11|OBD-BAND_s11) ;;
  *) die 23 "unknown tag $TAG (this package trains OBD-CTL_s10, OBD-BAND_s10, OBD-CTL_s11, OBD-BAND_s11)";;
esac
need_commit; need_pkg obd3_check.py frozen/argvs_obd_seed3.json jobs/obd3_stage_job.cmd
BASE="$(base_dir)" || die 16 "the pinned OBD chain files are not found or one of them changed"
R="$STATE/runs/$TAG"; RR="$B2_ROOT/runs/$TAG"; mkdir -p "$R"
need_free_gb "$NEED_GB_RUN"
need_gpu_idle
for pair in "obd3_check.py:$PKG/obd3_check.py" "obd_check.py:$BASE/obd_check.py" "argvs_obd_seed3.json:$PKG/frozen/argvs_obd_seed3.json"; do
  n="${pair%%:*}"; l="${pair#*:}"
  b2_exists "$B2_ROOT/check/$n" || die 16 "$B2_ROOT/check/$n is not on box 2: run the ship stage first"
  [ "$(b2_sha256 "$B2_ROOT/check/$n")" = "$(sha256sum "$l" | cut -d' ' -f1)" ] || die 16 "$n on box 2 differs from the package: re-run the ship stage"
done
"$PY" -I "$PKG/obd3_check.py" argv --tag "$TAG" --out-txt "$R/argv.txt"; rc=$?
[ "$rc" -ne 23 ] || die 23 "argv refused for $TAG: blocked tag or placeholder path"
[ "$rc" -eq 0 ] || die 16 "argv refused for $TAG (rc $rc)"
b2_mkdir "$RR"; put_verified "$R/argv.txt" "$RR/argv.txt"
GID="$(date +%Y%m%d_%H%M%S)"
render obd3_stage_job.cmd "$R/stage.$GID.cmd" "COMMIT=$OBD3_COMMIT" "GID=$GID" "ARGS=stage --tag $TAG --gid $GID" "MARKDIR=@ROOT@\\obd3\\runs\\$TAG" "MARK=STAGE"
b2_put "$R/stage.$GID.cmd" "$RR/stage.cmd"; b2_launch "$RR/stage.cmd" "$RR/stage.log"
r="$(b2_wait "$RR/STAGE_$GID.done" "$RR/STAGE_$GID.fail" "${OBD3_POLLS_RUN:-1100}")"; say "stage marker=$r"
[ "$r" != TIMEOUT ] || die 19 "the box-2 job did not finish within the poll limit; it may still run ($RR/stage.log, $RR/logs)"
if [ "$r" != DONE ]; then
  why="$(b2_cat "$RR/STAGE_$GID.fail" | head -c 400 | tr '\n' ' ')"
  case "$why" in preflight-fail*|pin-fail*|gate-fail*|token-fail*) die 11 "refused ($why)";; disk-low*) die 13 "$why";; post-fail*) die 12 "post check failed ($why)";; *) die 12 "box-2 job failed ($why) (log $RR/stage.log)";; esac
fi
clean_after_bank() {   # BANK_VERIFIED_<gid> proves (on box 2) that the dev box pulled and sha-checked the bank; the cleanup job refuses without it
  local BD="$1" CG; CG="$(date +%Y%m%d_%H%M%S)c"
  printf 'bank verified on the dev box %s\n' "$(date '+%F %T')" > "$R/BANK_VERIFIED_$CG"; put_verified "$R/BANK_VERIFIED_$CG" "$RR/BANK_VERIFIED_$CG"
  render obd3_stage_job.cmd "$R/clean.$CG.cmd" "COMMIT=$OBD3_COMMIT" "GID=$CG" "ARGS=cleanup --tag $TAG --gid $CG" "MARKDIR=@ROOT@\\obd3\\runs\\$TAG" "MARK=CLEAN"
  b2_put "$R/clean.$CG.cmd" "$RR/clean.cmd"; b2_launch "$RR/clean.cmd" "$RR/clean.log"
  local c; c="$(b2_wait "$RR/CLEAN_$CG.done" "$RR/CLEAN_$CG.fail" "${OBD3_POLLS_CLEAN:-30}")"
  [ "$c" = DONE ] || die 12 "the cleanup job did not finish ($c): the temporary gate dir on box 2 is still there (log $RR/clean.log)"
  pull_verified "$RR/CLEANUP.json" "$BD/CLEANUP.json" || die 15 "bank of CLEANUP.json"
  b2_exists "$B2_ROOT/gate_out/$TAG" && die 12 "temporary dir $B2_ROOT/gate_out/$TAG still exists on box 2 after the cleanup"
  say "cleanup logged in $BD/CLEANUP.json"
}
[ "$(c_free_mb)" -ge "${OBD3_NEED_C_MB:-2000}" ] || die 13 "dev box C: free $(c_free_mb) MB < ${OBD3_NEED_C_MB:-2000} MB: the bank pull would fill it (disk-low)"
BD="$PULLS/obd_$TAG"; mkdir -p "$BD"; OUTW="D:/tanitad/runs/v8f_obd/$TAG"
for f in ckpt_last.pt summary.json train_log.jsonl; do pull_verified "$OUTW/$f" "$BD/$f" || die 15 "bank of $f: the pulled copy differs from box 2's (3 attempts)"; done
pull_verified "$RR/RUN.done" "$BD/RUN.done" || die 15 "bank of RUN.done"
pull_verified "$RR/gate/gate_token.json" "$BD/gate_token.json" || die 15 "bank of the token"; pull_verified "$RR/gate/selfcheck.json" "$BD/selfcheck.json" || die 15 "bank of selfcheck.json"
cp "$R/argv.txt" "$BD/argv.txt"
"$PY" -I "$PKG/obd3_check.py" verify-token --token "$BD/gate_token.json" --argv-file "$BD/argv.txt" --tag "$TAG" --commit "$OBD3_COMMIT" > /dev/null || die 11 "the token does not bind this commit and argv"
"$PY" -I - "$BD" "$TAG" "$OBD3_COMMIT" << 'PYEOF' || die 15 "bank manifest: the RUN.done evidence does not match the banked files"
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
