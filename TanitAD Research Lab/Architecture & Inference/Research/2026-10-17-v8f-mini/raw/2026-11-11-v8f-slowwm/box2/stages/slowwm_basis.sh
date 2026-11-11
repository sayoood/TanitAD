#!/usr/bin/env bash
# slowwm_basis.sh: the BASIS stage (card control 3). ONE box-2 job (the generic stage job, `slowwm_check.py basis`): slow_basis.py fit (train, --expect-windows 40358, --exclude 540:600, --spool), cache (slow, pca) x (train, eval) into the ROOT D:\tanitad\zbasis
# (cache appends <kind>\<split>), check3 once per split of clips8.json (eval 6 + train 2 clips, combined into one CHECK3.json with the cov identity; AMEND_SLOWWM_2), then the counts (4572 train / 141 eval per kind, a BASIS_MANIFEST.json per directory) -> BASIS.json, FIT_RECORD.json, SPECTRA.json, CHECK3.json.
# Refusals: D: free < 6 GB BEFORE anything is built; a failed check3, fit or count stops the chain (rc 11). The pulled records are re-judged on the dev box (`verify-basis`).
# CPU only (CUDA_VISIBLE_DEVICES=-1 is set by the job). Nothing is deleted: the basis caches are the input of every run.
source "$(dirname "${BASH_SOURCE[0]}")/slowwm_common.sh"; SLOWWM_STAGE="basis"
need_commit; need_pkg slowwm_check.py jobs/slowwm_stage_job.cmd
C="$STATE/basis"; CB="$B2_ROOT/basis"; mkdir -p "$C"
if [ "$(c_free_mb)" -lt "${SLOWWM_NEED_C_MB_BASIS:-200}" ]; then die 13 "dev box C: free $(c_free_mb) MB is too low to bank the basis record"; fi
need_free_gb "$NEED_GB_BASIS"
need_gpu_idle
b2_exists "$B2_ROOT/check/slowwm_check.py" || die 16 "$B2_ROOT/check/slowwm_check.py is not on box 2: run the ship stage first"
[ "$(b2_sha256 "$B2_ROOT/check/slowwm_check.py")" = "$(sha256sum "$PKG/slowwm_check.py" | cut -d' ' -f1)" ] || die 16 "the checker on box 2 differs from the package: re-run the ship stage"
[ "$(b2_sha256 "$B2_ROOT/basiscode/slow_basis.py")" = "$(sha256sum "$BASIS_SRC" | cut -d' ' -f1)" ] || die 16 "basiscode/slow_basis.py on box 2 differs from theory/slowwm: re-run the ship stage"
[ "$(b2_sha256 "$B2_ROOT/basiscode/clips8.json")" = "$(sha256sum "$CLIPS_SRC" | cut -d' ' -f1)" ] || die 16 "basiscode/clips8.json on box 2 differs from theory/ladder: re-run the ship stage"
GID="$(date +%Y%m%d_%H%M%S)"
render slowwm_stage_job.cmd "$C/basis.$GID.cmd" "COMMIT=$SLOWWM_COMMIT" "GID=$GID" "ARGS=basis --gid $GID" "MARKDIR=@ROOT@\slowwm\basis" "MARK=BASIS"
b2_mkdir "$CB"; b2_put "$C/basis.$GID.cmd" "$CB/basis.cmd"; b2_launch "$CB/basis.cmd" "$CB/basis.job.log"
r="$(b2_wait "$CB/BASIS_$GID.done" "$CB/BASIS_$GID.fail" "${SLOWWM_POLLS_BASIS:-180}")"; say "basis marker=$r"
[ "$r" != TIMEOUT ] || die 19 "the box-2 basis job did not finish within the poll limit; it may still run ($CB/basis.log, $CB/basis.job.log)"
if [ "$r" != DONE ]; then
  why="$(b2_cat "$CB/BASIS_$GID.fail" | head -c 400 | tr '\n' ' ')"
  case "$why" in *disk-low*) die 13 "$why";; preflight:*|check3-refused*|counts:*) die 11 "basis refused ($why)";; *) die 12 "basis job failed ($why) (log $CB/basis.log)";; esac
fi
for f in BASIS.json FIT_RECORD.json CHECK3.json SPECTRA.json; do pull_verified "$CB/$f" "$C/$f" || die 15 "bank of $f"; done
"$PY" -I "$PKG/slowwm_check.py" verify-basis --basis-json "$C/BASIS.json" --fit-json "$C/FIT_RECORD.json" --check3-json "$C/CHECK3.json" --commit "$SLOWWM_COMMIT" || die 11 "the pulled basis record does not pass on the dev box (check3, fit, counts, manifests)"
say "basis ok: $C (caches stay on box 2 under D:/tanitad/zbasis)"; say "ok"
