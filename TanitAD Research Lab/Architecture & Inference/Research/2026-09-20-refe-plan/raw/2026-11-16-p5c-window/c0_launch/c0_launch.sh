#!/bin/bash
# MM 2026-11-16: hand launch of P5-C arm C0 = p5c_pod.sh's `train_C0` stage (token_ok + arm_train), with its two guards:
# (1) refuse an existing ckpt_last.pt in the arm's --out, (2) a FRESH `launch_gate.py verify` of the PASS token (re-measures argv, tree, data, ckpt, HMAC).
# Extra guard: the GPU must be idle. train.py starts ONLY if all pass. Prints ZZC0_LAUNCHED <pid> or ZZC0_REFUSED <why>.
set -u
ROOT=/workspace/data/refe_p5c
PKG=$ROOT/code_33824420
PY=/workspace/venv-teacher/bin/python
OUT=/workspace/data/refe_runs/p5c_c0
TOK=$ROOT/lg_C0/PASS_33824420cc5e.json
export TANITAD_GATE_KEY_FILE=$ROOT/launch_gate.key
ARGS="$(cat $ROOT/ARGS_C0.txt)"
case "$ARGS" in *"--out $OUT"*|*"--out=$OUT"*) ;; *) echo "ZZC0_REFUSED argv does not carry --out $OUT"; exit 3 ;; esac
[ -e "$OUT/ckpt_last.pt" ] && { echo "ZZC0_REFUSED $OUT/ckpt_last.pt exists"; exit 3; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$n" = 0 ] || { echo "ZZC0_REFUSED GPU busy ($n compute processes)"; exit 3; }
cd "$PKG/refe" || exit 3
env -u HF_TOKEN PYTHONPATH=$PKG/refe "$PY" "$PKG/refe/launch_gate.py" verify --profile refe_p5c_c0 --token "$TOK" --args "$ARGS" --pkg "$PKG" --json "$ROOT/token_verify_C0_launch.json" > "$ROOT/token_verify_C0_launch.log" 2>&1
rc=$?
[ "$rc" = 0 ] && grep -q "ZZGATE-MATCH" "$ROOT/token_verify_C0_launch.log" || { echo "ZZC0_REFUSED verify rc $rc: $(tail -n 2 $ROOT/token_verify_C0_launch.log | tr '\n' ' ' | cut -c1-300)"; exit 3; }
cd "$ROOT" || exit 3
setsid nohup env -u HF_TOKEN PYTHONUNBUFFERED=1 PYTHONPATH=$PKG/refe REFE_BACKBONE_ROOT=/workspace/data/backbones PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 NUPLAN_MAPS_ROOT=/workspace/data/nuplan-maps/nuplan-maps-v1.0 \
  "$PY" "$PKG/refe/train.py" $ARGS > "$ROOT/train_C0.log" 2>&1 < /dev/null &
pid=$!
echo "$pid $(date -u +%FT%TZ) token $(sha256sum $TOK | cut -c1-16)" > "$ROOT/train_C0.pid"
echo "ZZC0_LAUNCHED $pid"
