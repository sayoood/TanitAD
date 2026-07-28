#!/bin/bash
# PURE-IMAGINATION DECAY CURVE. Pre-registered: PREREG_imagination_horizon.md (eb27a36).
#
# Reuses the harness's OWN --canary-only path rather than re-wiring the dataset, so the rollout,
# grounding and SE(2) mechanics are byte-identical to the ones v1's 0.4271 is anchored against.
# The only thing that varies per run is the imagination horizon K.
#
# The canary encodes an 8-frame context and then rolls the predictor forward in LATENT SPACE with
# NO further frames — that IS "the camera is not fed". Actions stay expert_future, so this measures
# WORLD-MODEL fidelity, not action selection.
set -u
S=/workspace/TanitAD/stack
TEV=/workspace/tev/taniteval
CK=/workspace/v4run/hfcache/hub/models--Sayood--tanitad-flagship-4b-speedjerk/snapshots/17902296dee666e27ab60bccafcf889fef0ab0a8/ckpt.pt
VAL=/workspace/val40cache
OUT=/workspace/imag
mkdir -p "$OUT"; cd "$S" || exit 1
for K in 10 20 40 80; do
  S_S=$(python3 -c "print(f'{$K/10:g}')")
  echo "=== horizon ${S_S}s  (K=$K imagined steps, no camera) ==="
  # WP_STEPS drives K_MAX = max(WP_STEPS); overriding it in a sitecustomize-style shim keeps
  # eval_flagship_v4 itself untouched.
  PYTHONPATH="$S:$TEV:$S/scripts" OMP_NUM_THREADS=6 python3 - "$K" "$CK" "$VAL" "$OUT" <<'PY' 2>&1 | tail -6
import sys, json, runpy
K, CK, VAL, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
import eval_flagship_v4 as E
steps = sorted({max(1, K//4), max(1, K//2), max(1, 3*K//4), K})
E.WP_STEPS = steps
E.K_MAX = K
sys.argv = ["eval_flagship_v4.py", "--ckpt", CK, "--val-cache", VAL, "--canary-only",
            "--key", f"imag-K{K}", "--out", f"{OUT}/imag_K{K}.json",
            "--results-dir", OUT, "--device", "cuda"]
try:
    E.main()
except SystemExit as e:
    print(f"[imag] K={K} exit={e.code}")
PY
  echo "  -> $(python3 -c "
import json;d=json.load(open('$OUT/imag_K$K.json'));print({k:v for k,v in d.items() if 'canary' in k or k in ('n_windows','n')})" 2>/dev/null || echo 'no result')"
done
echo "=== IMAGINATION SWEEP DONE ==="
