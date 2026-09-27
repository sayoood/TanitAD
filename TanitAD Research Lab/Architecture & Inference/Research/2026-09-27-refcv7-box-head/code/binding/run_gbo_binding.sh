#!/bin/bash
# THE BINDING G-BOX-OVERFIT RUN (SPEC_REFCV7 A9/A10 prereg + A13 launch optimiser + A17 lr decay + A11 closure).
# ⛔ PREPARED, NOT RUN: it waits for the FINAL canonical argv (the map's A16 weights verdict decides the class-weights
# path, which is in the closure's data) -- the Master Mind starts it.
#
# usage: run_gbo_binding.sh <launch sha> <launch tree> <run dir>
#   <launch tree> = `git archive <launch sha> -- stack taniteval tools | tar -x` on Thor (the launch tree, A11)
#   the audit's two DATA files are staged under <run dir>/audit/raw/ by this script (prereg + frame set; md5-checked)
#   PREFLIGHT_ONLY=1 runs every check and stops before the run (exit 0 = ready; 3 = an input; 4 = the GPU is busy)
set -u
SHA="$1"; T="$2"; D="$3"
PY=/home/nvidia/venvs/tanitad-train/bin/python
ARGV="$T/stack/ops/runs.d/refcv7-r101-s0.argv.json"
SIDECAR=/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz
SIDECAR_SHA=278443b3356bca054c15753e7c08d465d71b0327a2ce564e143d091261349dd4
AUDIT_SRC="${AUDIT_SRC:-/home/nvidia/bx_0412/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit}"
mkdir -p "$D/audit/raw/visibility" || exit 2
# ---- pre-flight: every input is the registered one, or nothing starts ----------------------------------------- #
[ ${#SHA} -eq 40 ] || { echo "PREFLIGHT: the launch sha must be 40 hex"; exit 3; }
[ -f "$T/stack/scripts/g_box_overfit.py" ] && [ -f "$T/stack/scripts/closure_run.py" ] || { echo "PREFLIGHT: not a launch tree: $T"; exit 3; }
cp "$AUDIT_SRC/raw/PREREG_G_BOX_OVERFIT.md" "$D/audit/raw/" && cp "$AUDIT_SRC/raw/visibility/gbo_frameset.json" "$D/audit/raw/visibility/" || exit 3
echo "594c71196cc5bbd527fee40b2cb0e3f1  $D/audit/raw/PREREG_G_BOX_OVERFIT.md" | md5sum -c - || { echo "PREFLIGHT: prereg md5"; exit 3; }
echo "b291404c36f83c3e397e3b90367e8e7b  $D/audit/raw/visibility/gbo_frameset.json" | md5sum -c - || { echo "PREFLIGHT: frame set md5"; exit 3; }
echo "$SIDECAR_SHA  $SIDECAR" | sha256sum -c - || { echo "PREFLIGHT: the VIS-1 sidecar sha256"; exit 3; }
$PY - "$ARGV" <<'PYEOF' || { echo "PREFLIGHT: the canonical argv does not carry the box head"; exit 3; }
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))["argv"]
need = {"--slot-presence-loss": "focal", "--slot-presence-prior": "0.01", "--slot-query-select": "learned_ref",
        "--vis1-sidecar": "/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz"}
bad = [f for f, v in need.items() if f not in a or a[a.index(f) + 1] != v]
bad += [f for f in ("--slot-deep-supervision", "--slot-vis1") if f not in a]
print("canonical argv box tokens:", "OK" if not bad else f"MISSING/WRONG {bad}")
sys.exit(1 if bad else 0)
PYEOF
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || { echo "PREFLIGHT: nvidia-smi failed -- cannot tell whether the GPU is free"; exit 4; }
others=$(printf '%s' "$apps" | tr '\n' ' ')
[ -z "${others// /}" ] || { echo "PREFLIGHT: the GPU is busy ($others) -- one job at a time"; exit 4; }
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then echo "PREFLIGHT OK -- inputs verified, GPU idle; NOT started (PREFLIGHT_ONLY=1)"; exit 0; fi
# ---- the environment: offline, NO HF token anywhere (the offline error text dumps request headers) --------------- #
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd "$T" || exit 2
# ---- the run: the UNMODIFIED harness under the closure wrapper (A11); 3 arms x 2,000 steps (~4.5 h on Thor alone) -- #
exec nice -n 10 $PY stack/scripts/closure_run.py \
  --out "$D/gbo_closure.json" --result "$D/gbo_binding.json" \
  --data-root /home/nvidia/data --data-root "$D/audit" \
  --binding --commit "$SHA" --tree "$T" -- \
  stack/scripts/g_box_overfit.py --launch-argv "$ARGV" --audit-dir "$D/audit" \
    --out "$D/gbo_binding.json" --device cuda --binding --commit "$SHA" \
    --candidate "BINDING: launch commit $SHA (A13 launch optimiser, A17 lr decay, learned_ref)" \
    --arms main,memory_zeros,presence_w0
