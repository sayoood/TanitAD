#!/usr/bin/env bash
# refcv7 launch gate -- the THOR stage (SPEC_REFCV7 sec. 2 + 4): G-LIVE + G-CKPT on the REAL data,
# G-CLOCK on the REAL caches, G-HYG + G-DVB on Thor's own build (FIX-5 on Thor's cached ImageNet
# weights), [G-MAP-OVERFIT when --map-hires on], then the token over the dev-box evidence.
# Written by the gate agent; RUN BY THE MASTER MIND. Never on a box that is training.
#
#   CODE=/home/nvidia/refcv7_run/<commit10>        # git archive of COMMIT, md5-verified ship
#   COMMIT=<40-char sha>  ARGV=<refcv7 launch argv .json>  GATE_DIR=<fresh dir>
#   DEVBOX_EVIDENCE=<the dev-box stage's evidence/ dir, copied here>
#   CLOCK_REF=<q4c_grid_vs_egolog_ALLTRAIN.json, copied here>
#   TAU_RECORD=<nav_compliance_tau_train.json>     # required: refcv7 runs --graft-nav-compliance
#   OVERFIT_RECORD=<G-MAP-OVERFIT PASS record>      # required: refcv7 runs --map-hires on
#   [PI_COST_APPROVAL=<json>]                       # only after a NEEDS-PI and the PI's word
#   ./run_gate_thor.sh
#
# ⛔ The verdict is the TOKEN FILE: PASS_<commit12>.json in GATE_DIR, verified below by
#    `launch_gate.py verify` (HMAC + tree + argv + data re-measured). An exit code is not a verdict.
# ⛔ No pipes around the gate: `$?` after a pipeline is the LAST element's status.
set -u
PY="${PY:-/home/nvidia/venvs/tanitad-train/bin/python}"
: "${CODE:?}" "${COMMIT:?}" "${ARGV:?}" "${GATE_DIR:?}" "${DEVBOX_EVIDENCE:?}" "${CLOCK_REF:?}"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
case "$COMMIT" in ????????????????????????????????????????) : ;;
  *) echo "ZZGATETHOR-BADCOMMITZZ"; exit 2 ;; esac
C12="${COMMIT:0:12}"
EXTRA=()
if [ -n "${TAU_RECORD:-}" ]; then EXTRA+=(--nav-tau-record "$TAU_RECORD"); fi
if [ -n "${OVERFIT_RECORD:-}" ]; then EXTRA+=(--map-overfit-record "$OVERFIT_RECORD"); fi
if [ -n "${PI_COST_APPROVAL:-}" ]; then EXTRA+=(--pi-cost-approval "$PI_COST_APPROVAL"); fi
mkdir -p "$GATE_DIR"
"$PY" "$CODE/stack/scripts/launch_gate_refcv7.py" run --stage thor --tree "$CODE" \
  --commit "$COMMIT" --argv-file "$ARGV" --out-dir "$GATE_DIR" \
  --import-evidence "$DEVBOX_EVIDENCE" --clock-reference "$CLOCK_REF" \
  ${EXTRA[@]+"${EXTRA[@]}"} >> "$GATE_DIR/gate_thor.log" 2>&1
echo "gate rc=$? (NOT the verdict)" >> "$GATE_DIR/gate_thor.log"
if [ ! -f "$GATE_DIR/PASS_${C12}.json" ]; then
  for v in FAIL INCOMPLETE NEEDS-PI; do
    [ -f "$GATE_DIR/${v}_${C12}.json" ] && echo "ZZGATETHOR-${v}-${C12}ZZ"
  done
  exit 1
fi
# the token must verify HERE, exactly as the supervisor will verify it
"$PY" "$CODE/stack/scripts/launch_gate.py" verify --token "$GATE_DIR/PASS_${C12}.json" \
  --argv-file "$ARGV" --tree "$CODE" --commit "$COMMIT" --json "$GATE_DIR/verify_after_pass.json" \
  >> "$GATE_DIR/gate_thor.log" 2>&1
"$PY" - "$GATE_DIR/verify_after_pass.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("ZZGATETHOR-" + ("PASS-VERIFIED" if d.get("verdict") == "MATCH" else "PASS-BUT-REFUSED") + "ZZ")
PYEOF
