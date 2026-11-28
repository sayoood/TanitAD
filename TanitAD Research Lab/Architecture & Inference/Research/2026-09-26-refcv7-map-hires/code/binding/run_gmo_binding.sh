#!/bin/bash
# THE BINDING G-MAP-OVERFIT RUN (SPEC_REFCV7 §23 A18 protocol + A11 closure) -- the analogue of the
# box builder's code/binding/run_gbo_binding.sh.
# ⛔ PREPARED, NOT RUN: the Master Mind's chain starts it after the box binding (~22:30 Berlin), and
# ONLY if the early A18 MAIN passed (a FAIL leaves the argv NOT final: refine blocks 2 goes to the PI).
#
# usage: run_gmo_binding.sh <launch sha> <launch tree> <run dir>
#   <launch tree> = the binding tree (the Master Mind): tip + box A17 + R5's two files + the FINAL
#                   154-token argv -- `stack/scripts/map_hires_overfit.py` MUST be blob 9bec9e88
#   <run dir>     = a FRESH directory; the A18 spec and the audit's frame set are staged under
#                   <run dir>/inputs/ (md5-checked) so the closure records them as data
#   PREFLIGHT_ONLY=1 runs every check and stops before the run (exit 0 = ready; 3 = an input;
#                   4 = the GPU is busy or unreadable)
#
# The protocol (A18): the REGISTERED spec raw/gmo_spec_A18.json (md5 4eda0636) -- 3,000 steps, lr 1e-3
# held to step 2,700 then cosine to 0, batch 4, seed 0; A15's map path (near lift 20 m + ONE near
# refine block) with the TRAIN sqrt_mf launch weights; ALL FOUR arms: healthy (MAIN), s8_zeros,
# near_block_zeros, lane_w0 (~2 h 15 min alone on Thor; MAIN 0.711 s/step at b4).
set -u
SHA="$1"; T="$2"; D="$3"
PY=/home/nvidia/venvs/tanitad-train/bin/python
ARGV="$T/stack/ops/runs.d/refcv7-r101-s0.argv.json"
ARGV_SHA=6402d33de75b7f1c6dbdeb9aeedd46a00a82e7325eec420fa179f366213bd5cd
W=/home/nvidia/data/refcv7/map_hires_class_weights_train_100x30.json
W_SHA=d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67
HARNESS_BLOB=9bec9e8844d522550eee5e5b4d1769fb09147d76
RES="$T/TanitAD Research Lab/Architecture & Inference/Research"
SPEC_SRC="$RES/2026-09-26-refcv7-map-hires/raw/gmo_spec_A18.json"
FS_SRC="$RES/2026-09-26-map-signal-audit/raw/gmo_frameset.json"
V2=/home/nvidia/data/refcv6-b1-416x1024-train
GT=/home/nvidia/data/sam3_gt_v3
EXT=/home/nvidia/data/refcv6_train_eval139_extrinsics.json
CAMTS=/home/nvidia/data/_b1stage416/r0/camera_front_wide
# ---- pre-flight: every input is the registered one, or nothing starts ------------------------- #
[ ${#SHA} -eq 40 ] || { echo "PREFLIGHT: the launch sha must be 40 hex"; exit 3; }
[ -f "$T/stack/scripts/map_hires_overfit.py" ] && [ -f "$T/stack/scripts/closure_run.py" ] \
  || { echo "PREFLIGHT: not a launch tree: $T"; exit 3; }
[ -e "$D/out/g_map_overfit.json" ] || [ -e "$D/gmo_closure.json" ] \
  && { echo "PREFLIGHT: $D already holds a record -- use a FRESH run dir"; exit 3; }
mkdir -p "$D/inputs" "$D/out" || exit 2
$PY - "$T/stack/scripts/map_hires_overfit.py" "$HARNESS_BLOB" <<'PYEOF' || { echo "PREFLIGHT: the map harness is not blob $HARNESS_BLOB"; exit 3; }
import hashlib, sys
b = open(sys.argv[1], "rb").read()
got = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
print("map harness blob:", got)
sys.exit(0 if got == sys.argv[2] else 1)
PYEOF
cp "$SPEC_SRC" "$D/inputs/gmo_spec_A18.json" && cp "$FS_SRC" "$D/inputs/gmo_frameset.json" \
  || { echo "PREFLIGHT: the A18 spec / frame set is not in the tree"; exit 3; }
echo "4eda0636f1a41a9b60b59ccb6a4afca3  $D/inputs/gmo_spec_A18.json" | md5sum -c - \
  || { echo "PREFLIGHT: the A18 spec md5"; exit 3; }
echo "4eafa03c2b6a6e6d6336be1d78acb91d  $D/inputs/gmo_frameset.json" | md5sum -c - \
  || { echo "PREFLIGHT: the frame set md5"; exit 3; }
echo "$W_SHA  $W" | sha256sum -c - || { echo "PREFLIGHT: the launch class weights sha256"; exit 3; }
for p in "$V2" "$GT" "$EXT" "$CAMTS"; do
  [ -e "$p" ] || { echo "PREFLIGHT: data input absent: $p"; exit 3; }
done
$PY - "$ARGV" "$ARGV_SHA" "$W" <<'PYEOF' || { echo "PREFLIGHT: the canonical argv is not the FINAL list"; exit 3; }
import hashlib, json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))["argv"]
sha = hashlib.sha256(json.dumps(a, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
need = {"--map-hires-near-lift-m": "20", "--map-hires-near-refine-blocks": "1",
        "--map-hires-class-weights": sys.argv[3], "--map-hires-decision-rule": "prior_corrected",
        "--map-hires-x-max-m": "100", "--map-hires-y-half-m": "30"}
bad = [f for f, v in need.items() if f not in a or a[a.index(f) + 1] != v]
print(f"canonical argv: {len(a)} tokens, gate sha {sha[:16]}...,",
      "map tokens OK" if not bad else f"MISSING/WRONG {bad}")
sys.exit(0 if sha == sys.argv[2] and len(a) == 154 and not bad else 1)
PYEOF
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) \
  || { echo "PREFLIGHT: nvidia-smi failed -- cannot tell whether the GPU is free"; exit 4; }
others=$(printf '%s' "$apps" | tr '\n' ' ')
[ -z "${others// /}" ] || { echo "PREFLIGHT: the GPU is busy ($others) -- one job at a time"; exit 4; }
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "PREFLIGHT OK -- inputs verified, GPU idle; NOT started (PREFLIGHT_ONLY=1)"; exit 0
fi
# ---- the environment: offline, NO HF token anywhere ------------------------------------------ #
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd "$T" || exit 2
# ---- the run: the UNMODIFIED harness under the closure wrapper (A11); 4 arms x 3,000 steps ---- #
exec nice -n 10 $PY stack/scripts/closure_run.py \
  --out "$D/gmo_closure.json" --result "$D/out/g_map_overfit.json" \
  --data-root /home/nvidia/data --data-root "$D/inputs" \
  --binding --commit "$SHA" --tree "$T" -- \
  stack/scripts/map_hires_overfit.py --spec "$D/inputs/gmo_spec_A18.json" \
    --frameset "$D/inputs/gmo_frameset.json" --class-weights "$W" --decision-rule prior_corrected \
    --near-lift-m 20 --near-refine-blocks 1 \
    --v2-cache "$V2" --gt-root "$GT" --extrinsics "$EXT" --cam-ts-dir "$CAMTS" \
    --launch-commit "$SHA" --launch-argv-sha256 "$ARGV_SHA" \
    --arms healthy,s8_zeros,near_block_zeros,lane_w0 --out "$D/out" --device cuda
