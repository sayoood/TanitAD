#!/bin/bash
# P7-WIRE-INTEGRATE: the pin procedure's LITERAL commands (PROCEDURE_P7_PINS.md steps A-D), run on the synthetic inputs of build_synth.py.
# usage: bash rehearse.sh <refe dir> <work dir>      (CPU only; the python is the one on PATH or $PY)
set -eu
REFE="$1"; W="$2"; PY="${PY:-python}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
rm -rf "$W"; mkdir -p "$W"
"$PY" "$(dirname "$0")/build_synth.py" "$REFE" "$W"
RES="$W/reserve_sessions.json"
echo "== A. the filter on the bank (this is what measure_rank1_3c.sh runs after its download; --token-table is added there)"
"$PY" "$REFE/session_reserve.py" filter --reserve "$RES" --bank "$W/train_grow_cmd_3c" --out "$W/train_grow_cmd_3c_p7"
echo "== B. the training label store: copy, split the reserve's shards out, RE-FINALISE both (block 5a of p6_pod.sh, verbatim)"
cp -r "$W/label_store" "$W/label_store_p7"
"$PY" "$REFE/p6_prep.py" reserve-labelstore --reserve "$RES" --store "$W/label_store_p7" --reserve-store "$W/label_store_p7_reserve"
"$PY" "$REFE/aux_label_store.py" finalize --out "$W/label_store_p7"
"$PY" "$REFE/aux_label_store.py" finalize --out "$W/label_store_p7_reserve"
echo "== C. the calibration table without the reserve's logs (optional cross-check)"
"$PY" "$REFE/p6_prep.py" reserve-calib --reserve "$RES" --calib "$W/calib_table.json" --out "$W/calib_p7.json"
echo "== D. READ the three pins off the outputs (P7_BANK_TUPLES, P7_TARGETS_SHA256, P7_AUX_STORE_PINS) (scenes: the synthetic bank has 6, the real one 92,896 = launch_gate.P7_SCENES)"
"$PY" "$REFE/p7_pins.py" --reserve "$RES" --bank "$W/train_grow_cmd_3c_p7" --store "$W/label_store_p7" --calib "$W/calib_p7.json" --expect-scenes 6 | tee "$W/pins.out"
echo "== E. the REFUSALS (each must exit 1 and print no registration)"
set +e
"$PY" "$REFE/p7_pins.py" --reserve "$RES" --bank "$W/train_grow_cmd_3c" --store "$W/label_store_p7" --expect-scenes 8; echo "unfiltered bank: rc=$?"
"$PY" "$REFE/p7_pins.py" --reserve "$RES" --bank "$W/train_grow_cmd_3c_p7" --store "$W/label_store" --expect-scenes 6; echo "FULL store: rc=$?"
"$PY" "$REFE/p7_pins.py" --reserve "$RES" --bank "$W/train_grow_cmd_3c_p7" --store "$W/label_store_p7"; echo "default epoch_len (92,896) vs the synthetic bank's 6 scenes: rc=$?"
echo REHEARSAL_DONE
