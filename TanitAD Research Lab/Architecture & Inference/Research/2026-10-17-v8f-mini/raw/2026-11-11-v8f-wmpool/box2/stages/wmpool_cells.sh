#!/usr/bin/env bash
# wmpool_cells.sh: the CELL CACHE stage (card control 3). ONE box-2 job (the generic stage job, `wmpool_check.py cells`): zcells.py build (M2b_s0 and M2b_s1, splits eval and train, resumable), zcells.py manifest, zcells.py check3 on the 8 clips of clips8.json
# (mean of the cached cells == today's token-mean cache; a wrong-seed adapter must FAIL the identity), then the counts (4572 train / 141 eval per seed, no .tmp, a CELL_MANIFEST.json per directory) -> CELLS.json.
# Refusals: D: free < cache + 5 GB (need 2 x 2.36 + 5 = 9.72 GB, ESTIMATED; the stage asks for 10) BEFORE anything is built; a failed check3 or count stops the chain (rc 11). The pulled CELLS.json / ZCELLS_CHECK3.json are re-judged on the dev box.
# CPU only (CUDA_VISIBLE_DEVICES=-1 is set by the job). Nothing is deleted: the cell cache is the input of every run and re-creatable only by rebuilding.
source "$(dirname "${BASH_SOURCE[0]}")/wmpool_common.sh"; WMPOOL_STAGE="cells"
need_commit; need_pkg wmpool_check.py frozen/zcells.py frozen/zcache.py frozen/clips8.json jobs/wmpool_stage_job.cmd
C="$STATE/cells"; CB="$B2_ROOT/cells"; mkdir -p "$C"
if [ "$(c_free_mb)" -lt "${WMPOOL_NEED_C_MB_CELLS:-200}" ]; then die 13 "dev box C: free $(c_free_mb) MB is too low to bank the cells record"; fi
need_free_gb "$NEED_GB_CELLS"
need_gpu_idle
b2_exists "$B2_ROOT/check/wmpool_check.py" || die 16 "$B2_ROOT/check/wmpool_check.py is not on box 2: run the ship stage first"
[ "$(b2_sha256 "$B2_ROOT/check/wmpool_check.py")" = "$(sha256sum "$PKG/wmpool_check.py" | cut -d' ' -f1)" ] || die 16 "the checker on box 2 differs from the package: re-run the ship stage"
for f in zcells.py zcache.py clips8.json; do
  [ "$(b2_sha256 "$B2_ROOT/cellcode/$f")" = "$(sha256sum "$PKG/frozen/$f" | cut -d' ' -f1)" ] || die 16 "cellcode/$f on box 2 differs from the package: re-run the ship stage"
done
GID="$(date +%Y%m%d_%H%M%S)"
render wmpool_stage_job.cmd "$C/cells.$GID.cmd" "COMMIT=$WMPOOL_COMMIT" "GID=$GID" "ARGS=cells --gid $GID" "MARKDIR=@ROOT@\wmpool\cells" "MARK=CELLS"
b2_mkdir "$CB"; b2_put "$C/cells.$GID.cmd" "$CB/cells.cmd"; b2_launch "$CB/cells.cmd" "$CB/cells.job.log"
r="$(b2_wait "$CB/CELLS_$GID.done" "$CB/CELLS_$GID.fail" "${WMPOOL_POLLS_CELLS:-180}")"; say "cells marker=$r"
[ "$r" != TIMEOUT ] || die 19 "the box-2 cells job did not finish within the poll limit; it may still run ($CB/cells.log, $CB/cells.job.log); the build is resumable"
if [ "$r" != DONE ]; then
  why="$(b2_cat "$CB/CELLS_$GID.fail" | head -c 400 | tr '\n' ' ')"
  case "$why" in *disk-low*) die 13 "$why";; preflight:*|check3-refused*|counts:*) die 11 "cells refused ($why)";; *) die 12 "cells job failed ($why) (log $CB/cells.log)";; esac
fi
pull_verified "$CB/CELLS.json" "$C/CELLS.json" || die 15 "bank of CELLS.json"
pull_verified "$CB/ZCELLS_CHECK3.json" "$C/ZCELLS_CHECK3.json" || die 15 "bank of ZCELLS_CHECK3.json"
"$PY" -I "$PKG/wmpool_check.py" verify-cells --cells-json "$C/CELLS.json" --check3-json "$C/ZCELLS_CHECK3.json" --commit "$WMPOOL_COMMIT" || die 11 "the pulled cells record does not pass on the dev box (check3 identity, counts, manifests)"
say "cells ok: $C (cache stays on box 2 under D:/tanitad/zcells)"; say "ok"
