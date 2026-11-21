#!/bin/bash
# chain_after_build.sh -- FRESH2 unattended tail after the v2 episode build: chain_build_rest.sh (refusals, split, kit, lead labels) -> chain_roll.sh (GPU z_tac roll
# under the shared lock) -> fresh2_derive.py --set fresh2 (a0 / psi_dot0) -> fresh2_content_check.py -> fresh2_lhg_fit.py --fit (LH_G frozen + sha registered).
# Stage 1 is NOT run here (it is run once, by hand, after the content checks are read).  Each step judged on its ARTIFACT.  Markers ZZF2A<NAME>ZZ.  No git.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_fresh2; CODE=$W/code; L=$W/logs/chain_after_build.log
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mk() { echo "ZZF2A$1ZZ $(date +%FT%T%z) $2" >> "$L"; }
mk START "pid $$"
bash "$CODE/chain_build_rest.sh"
grep -q "ZZF2ENDZZ" "$W/logs/chain_build_rest.log" || { mk STOP "chain_build_rest did not reach END (see logs/chain_build_rest.log)"; exit 3; }
mk BUILDREST "ok"
bash "$CODE/chain_roll.sh"
grep -q "ZZF2RENDZZ\|ZZF2RHAVEZZ" "$W/logs/chain_roll.log" || { mk STOP "chain_roll did not reach END (see logs/chain_roll.log)"; exit 4; }
mk ROLL "$(grep ZZF2RROLLZZ "$W/logs/chain_roll.log" | tail -1 | cut -c1-160)"
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/fresh2_derive.py" --set fresh2 > "$W/logs/derive_fresh2.log" 2>&1 )
[ -s F:/l1c/fresh2/features/result/lha0_features.npz ] && grep -q "ZZCALZZ CALIBRATED" "$W/logs/derive_fresh2.log" || { mk STOP "derive (see logs/derive_fresh2.log)"; exit 5; }
mk DERIVE "$(tail -1 "$W/logs/derive_fresh2.log" | cut -c1-200)"
( export CUDA_VISIBLE_DEVICES=-1; "$PY" -X utf8 "$CODE/fresh2_content_check.py" > "$W/logs/content_check.log" 2>&1 )
grep -q ZZCONTENTPASSZZ "$W/logs/content_check.log" || { mk STOP "content check FAILED (see result/CONTENT_CHECKS.json)"; exit 6; }
mk CONTENT "$(tail -1 "$W/logs/content_check.log" | cut -c1-300)"
( export CUDA_VISIBLE_DEVICES=-1 R8_LAUNCH_TREE=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58; "$PY" -X utf8 "$CODE/fresh2_lhg_fit.py" --fit > "$W/logs/lhg_fit.log" 2>&1 )
[ -s F:/l1c/fresh2/lhg/LH_G_FROZEN.sha256 ] && grep -q ZZLHGFITZZ "$W/logs/lhg_fit.log" || { mk STOP "LH_G fit (see logs/lhg_fit.log)"; exit 7; }
mk LHG "$(grep ZZLHGFITZZ "$W/logs/lhg_fit.log" | cut -c1-200)"
mk END "ready for stage 1"
