#!/usr/bin/env bash
# r9e2_s6c_resume.sh -- DEV-BOX side of the refcv9 E2 read: score the four MAPSEL traces (K2, K0 x inference seeds 0, 1) with s6c_dev.py, then INPUTS -> the verdict.  E2-S6C-WIRE, 2026-11-08.
# NEW file.  Pairs with r9e2_roll.sh, which STOPS (exit 5, marker ZZR9E2ROLL-AWAIT-S6CZZ) when the four S6c files are missing.
#
#   Why off Thor: s6c_dev.py -> mapsel_score.py opens the banked refcv8 eval artifacts (battery/step48852ema_pf: dump_s*/refcv6_extras.npz, s6_s*, panel_s*; the
#   DEPLOY8_POS_*_s*.npz + refcv8_retime_deploy_gains_pf.json under pfpkg/raw/retime_deploy), the retime_stage.py, the v9 label npz and the GT map root through ABSOLUTE dev-box
#   paths hard-coded in l2_common.py / mapsel_data.py / mapsel_score.py.  They exist on the dev box only; mirroring ~8 absolute paths on Thor (no python allowed there) is the
#   larger risk, so the traces and rasters (~1 GB per arm per seed, DEV windows) are pulled here instead.
#
#   Pull first:  <R9E2_OUT>/read/ (EVAL_*.json, families_s*.json, camblind_s*.json, distkeep_s*.json ...) -> R9E2_RD ;  <R9E2_OUT>/E2_{K2,K0}_mapsel/{trace_s0,trace_s1}.npz + rast_s0/ rast_s1/ -> R9E2_MS_ROOT
#   ENV (all REQUIRED, exit 4 when unset):  R9E2_RD  R9E2_MS_ROOT  R9E2_SPLIT (DEV_HOLD_SPLIT.json)  R9E2_MAPSEL_CODE (.../2026-10-25-refcv9-mapsel/code)  R9E2_OPS (the r9e2_read dir)
#                                           R9E2_VERDICT (stack/scripts/r9_e2_verdict.py)
#   OPTIONAL:  R9E2_L2_CODE (.../2026-10-25-refcv9-l2/code, s6c_dev --l2-code), R9E2_PY, R9E2_DRY=1 (print every command; touch nothing)
#   Output: $R9E2_RD/s6c/S6C_{K2,K0}_s{0,1}.json (one s6c_dev flat JSON per arm per seed), $R9E2_RD/INPUTS.json, E2_VERDICT.json / .md.   EXIT: 0 GO | 1 NO-GO | 2 REFUSED | 3 abort | 4 refused to start.
#   HOLD discipline: s6c_dev.py REFUSES a trace that holds any HOLD / non-DEV window; this script adds nothing that reads a HOLD window.
set -u
DRY="${R9E2_DRY:-0}"
PINS="R9E2_RD R9E2_MS_ROOT R9E2_SPLIT R9E2_MAPSEL_CODE R9E2_OPS R9E2_VERDICT"
MISS=""
for _p in $PINS; do [ -n "${!_p:-}" ] || MISS="$MISS $_p"; done
if [ -n "$MISS" ]; then echo "[r9e2_s6c_resume] refusing: unset env pin(s):$MISS" >&2; exit 4; fi
RD="$R9E2_RD"; MS="$R9E2_MS_ROOT"; PY="${R9E2_PY:-python}"; OPS="$R9E2_OPS"; S6D="$RD/s6c"
L2=(); [ -n "${R9E2_L2_CODE:-}" ] && L2=(--l2-code "$R9E2_L2_CODE")
q() { printf '%q ' "$@"; }
X() { if [ "$DRY" = 1 ]; then echo "[dry] $(q "$@")"; return 0; fi; "$@"; }
ev() { echo "ZZR9E2S6CZZ-$1 $(date -u +%H:%M:%SZ) ${*:2}"; }
abort() { echo "ZZR9E2S6CZZ-ABORTED $1 -- ${2:-}" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=
[ "$DRY" = 1 ] || mkdir -p "$S6D" || exit 4
ev START "dry=$DRY rd=$RD ms=$MS"
for A in K2 K0; do
  for S in 0 1; do
    T="$MS/E2_${A}_mapsel/trace_s$S.npz"; RA="$MS/E2_${A}_mapsel/rast_s$S"; F="$S6D/S6C_${A}_s$S.json"
    if [ "$DRY" != 1 ]; then
      [ -s "$T" ] || abort NO-TRACE "$A seed $S: $T is missing (pull E2_${A}_mapsel from Thor)"
      [ -d "$RA" ] || abort NO-RAST "$A seed $S: $RA is missing"
    fi
    X env PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 "$PY" -X utf8 "$OPS/s6c_dev.py" --mapsel-code "$R9E2_MAPSEL_CODE" ${L2[@]+"${L2[@]}"} --split "$R9E2_SPLIT" --seed "$S" \
        --trace "$T" --rast "$RA" --out "$S6D/score_${A}_s$S" --flat-out "$F"; rc=$?
    if [ "$DRY" != 1 ]; then [ "$rc" = 0 ] && [ -s "$F" ] || abort S6C-FAILED "$A seed $S rc=$rc: no $F (s6c_dev refuses an empty set, a failing control or a HOLD window)"; fi
    ev S6C-DONE "$A seed $S $F"
  done
done
X "$PY" -X utf8 "$OPS/r9e2_inputs.py" --rd "$RD" --out "$RD/INPUTS.json" --k2-label K2 --k0-label K0 --s6c-dir "$S6D"; rc=$?
if [ "$DRY" != 1 ]; then [ "$rc" = 0 ] && [ -s "$RD/INPUTS.json" ] || abort INPUTS-FAILED "rc=$rc"; fi
ev INPUTS-WRITTEN "$RD/INPUTS.json"
X "$PY" -X utf8 "$R9E2_VERDICT" --inputs "$RD/INPUTS.json" --out-dir "$RD"; VRC=$?
if [ "$DRY" = 1 ]; then ev DRY-DONE "nothing was run"; exit 0; fi
ev VERDICT-DONE "rc=$VRC $RD/E2_VERDICT.json (0 GO / 1 NO-GO / 2 REFUSED)"
case "$VRC" in 0|1) exit "$VRC";; *) exit 2;; esac
