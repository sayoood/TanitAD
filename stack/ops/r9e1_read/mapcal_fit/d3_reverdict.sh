#!/usr/bin/env bash
# d3_reverdict.sh <ARM> <REF> <THRDIR> -- MM 2026-11-07 (Thor, CPU): the banked DECISIVE pswpedgeF read of <ARM> vs <REF> (read_v3/arm_<ARM>/<THRDIR>)
# re-run with kit_d3_4e49ebb5 (hold_r9_d3all) and --map-beside-d3: M1's two design beside rows (lane tolerance-IoU 0-20 m, drivable-boundary recall
# @0.3 m 0-20 m) from the verdict's own paired bootstrap. Same inputs as pf_reverdict.sh. The verdict, failing clauses, reads and clause 3 must equal
# the base verdict's (reverdict_ac90ce1e if present, else <THRDIR>/VERDICT_<ARM>.json), compared by d3_cmp.py
# (the only reads allowed to leave are <ARM>.beside.lane/.edge, NOT-COMPUTABLE before; the rows are clause3=False: they never decide). Out: <THRDIR>/reverdict_d3_4e49ebb5/. Ends ZZD3REV-DONE / ZZD3REV-ABORT-<why>.
export PATH=/usr/bin:/bin:$PATH; set -u
ARM="$1"; REF="$2"; XN="$3"
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; K=$R/kit_d3_4e49ebb5; PY=/home/nvidia/venvs/tanitad-train/bin/python
LV=$RD/LIFT_VALID_M1.npz; FD=$F/floor_thr_pswpedgeF; AD=$RD/arm_$ARM; VS=$K/stack/scripts/r9_e1_verdict.py; ST=$F/D3_REVERDICT_STATUS.txt
PPC="$K/stack:$K/stack/scripts:$K/taniteval:$K/stack/ops/r9e1_read/shim_scipy"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZD3REV-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "$ARM ${*:2}"; exit 3; }
grep -q -- '--map-beside-d3' "$VS" || die OLD-KIT "$VS has no --map-beside-d3"
XD=$AD/$XN; BASE=$XD/reverdict_ac90ce1e/VERDICT_$ARM.json; [ -s "$BASE" ] || BASE=$XD/VERDICT_$ARM.json
[ -s "$F/d3_cmp.py" ] || die NO-CMP "$F/d3_cmp.py"
[ -s "$XD/VERDICT_$ARM.json" ] && [ -s "$BASE" ] || die NO-XD "$XD (needs its VERDICT)"
DR=$RD/dump_$REF; [ "$REF" = C0 ] || DR=$RD/arm_$REF/dump_$REF
OD=$XD/reverdict_d3_4e49ebb5; [ -e "$OD" ] && die EXISTS "$OD"; mkdir -p $OD
ev START "$ARM vs $REF $XD"
( cd "$AD" && env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN PYTHONPATH="$PPC" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= \
    $PY -X utf8 "$VS" --floor "$FD/FLOOR.json" --floor-eval-c0 "$FD/EVAL_C0.json" --paired "$AD/EVAL_${ARM}_vs_${REF}.json" \
    --eval-arm "$XD/EVAL_$ARM.json" --eval-ref "$XD/EVAL_$REF.json" --arm "$ARM" --ref "$REF" --rast-arm "$AD/dump_$ARM/rast" --rast-ref "$DR/rast" \
    --lift-valid "$LV" --map-thresholds-arm "$F/thr_${ARM}_pswpedgeF.json" --map-thresholds-ref "$F/thr_${REF}_pswpedgeF.json" \
    --map-beside-d3 --boot 2000 --seed 0 --out-dir "$OD" > "$OD/verdict.out" 2> "$OD/verdict.err" < /dev/null ) || die VERDICT-RC "(read $OD/verdict.err)"
CMP=$($PY -I "$F/d3_cmp.py" "$BASE" "$OD/VERDICT_$ARM.json" "$ARM")
[ "${CMP%% *}" = SAME ] || die DIFFER "$CMP"
ev DONE "$ARM vs $REF $OD: $CMP"
