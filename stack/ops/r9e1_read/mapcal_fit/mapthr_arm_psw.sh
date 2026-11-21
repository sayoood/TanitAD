#!/usr/bin/env bash
# mapthr_arm_psw.sh -- MM 2026-11-06 (Thor, CPU only). Re-run of mapthr_read.sh's step 6 for the DECISIVE psw variant: its first attempt (20:00:01Z)
# never started -- kit_mapcal_2408e921 had been exported with core.autocrlf=true, so bash refused r9e1_arm_read.sh at line 33. The kit path now
# holds a byte-exact LF export of 2408e921 (3,357 blobs = git ls-tree; the CRLF copy is kit_mapcal_2408e921_crlf).
#   CONTROL: re-fit thr_C0_psw with the LF kit; its levels must equal the file the CRLF kit fitted (the Python steps cannot depend on line endings).
#   ARM:     r9e1_arm_read.sh M1 on read_v3 with arm thr_M1_psw, reference thr_C0_psw, floor floor_thr_psw, FLOOR v3's lift-valid mask.
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; K=$R/kit_mapcal_2408e921; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LV=$RD/LIFT_VALID_M1.npz; AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$F/MAPTHR_STATUS.txt"; echo "ZZMAPTHRARM-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
[ "$(grep -c $'\r' $AR)" = 0 ] || die CRLF-KIT "$AR still has CR"
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES=$F/fit_remap.json CUDA_VISIBLE_DEVICES= "$@"; }
PPC="$K/stack:$K/stack/scripts:$K/taniteval:$K/stack/ops/r9e1_read/shim_scipy"
mkdir -p $F/control_lf
( cd $K/stack && penv "$PPC" $PY -X utf8 scripts/r9_map_fit_thresholds.py fit --rast-dir $F/dump_C0/rast --dev-hold $SPLIT --arm C0 \
    --planes psw --out $F/control_lf/thr_C0_psw.json > $F/control_lf/fit_C0_psw.log 2>&1 < /dev/null ) || die CONTROL-FIT "(read $F/control_lf/fit_C0_psw.log)"
CMP=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8")); b=json.load(open(sys.argv[2],encoding="utf-8"))
def lv(d): return json.dumps([d["planes"], d["bands_m"], d["criterion"], d["provenance"]["rast_dir_names_sha256"]], sort_keys=True)
print(("EQUAL" if lv(a)==lv(b) else "DIFFER") + " tool_sha256 crlf " + a["provenance"]["tool_sha256"][:12] + " lf " + b["provenance"]["tool_sha256"][:12])' $F/thr_C0_psw.json $F/control_lf/thr_C0_psw.json)
[ "${CMP%% *}" = EQUAL ] || die CONTROL-DIFFER "LF-kit re-fit of thr_C0_psw differs from the CRLF-kit fit"
ev CONTROL-OK "LF-kit re-fit of thr_C0_psw: planes / bands / criterion / raster set EQUAL to the CRLF-kit fit ($CMP)"
( cd "$RD" && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_M1/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60 \
    R9E1_MAP_THR_ARM="$F/thr_M1_psw.json" R9E1_MAP_THR_REF="$F/thr_C0_psw.json" R9E1_MAP_THR_FLOOR_DIR="$F/floor_thr_psw" \
    bash "$AR" M1 > "$F/arm_M1_thr_psw.rerun.out" 2>&1 < /dev/null )
ev ARM "psw-rerun M1: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_M1_thr_psw.rerun.out | tail -1)"
