#!/bin/sh
# A0b_replicate — A0's flags and A0's seed, run a SECOND time.
#
# ⭐ WHY THIS ARM EXISTS (added mid-panel, 2026-09-05, and it is not cosmetic).
# The panel's paired episode-cluster bootstrap resamples EPISODES at eval time with the
# trained models HELD FIXED. It answers "would this delta survive on other windows?" --
# NOT "would it survive on another training run?". With one seed per arm those are
# different questions and only the second licenses "the lever caused it".
#
# MEASURED from the panel's own logs (noise_floor.py): over the 450-step stretch where
# A1 and A0 are the SAME configuration, 66/72 logged differences are non-zero and
# |d withheld_speed_mae| reaches 0.701 m/s. Training is not deterministic, and the
# divergence amplifies. So a "separated" A1-vs-A0 delta may be nothing but that.
#
# A0b is the control that settles it AT THE LEVEL THE PANEL SCORES: identical flags,
# identical seed, scored through the identical instrument. Its paired delta vs A0 is the
# eval-level noise floor. Any arm's delta that is not clearly larger than A0b's is not
# attributable to its lever.
#
# ⛔ Run this ONLY after panel.sh reports PANEL-COMPLETE: one GPU, one arm at a time
# (torch spawns ~113 threads per process and concurrent arms make no progress).
set -u
W=/c/Users/Admin/run_wbank
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
DR="C:\\Users\\Admin\\tanitad-data\\physicalai\\_epcache"
ANC="C:\\Users\\Admin\\run_wbank\\refc_anchors_6s_v0cond_alat_117.pt"
export PYTHONPATH="C:\\Users\\Admin\\refcv4b_suite\\stack;C:\\Users\\Admin\\tanitad-wt\\taniteval"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export OMP_NUM_THREADS=6
cd /c/Users/Admin/refcv4b_suite/stack || exit 1

# byte-identical to panel.sh's COMMON + A0's two flags; ONLY --out differs
COMMON="--arm hier --size tiny --data-root $DR --episodes 48 --steps 2000 \
--batch 12 --lr 1e-4 --warmup 250 --seed 0 --log-every 50 --save-every 2000 \
--device cuda \
--anchors $ANC --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
--sel-accel-max 2.0 --ego-state-inject"

name=A0b_replicate
if [ -f "$W/arms/$name/summary.json" ]; then echo "ZZSKIP-$name-ZZ"; exit 0; fi
echo "ZZSTART-$name-$(date +%T)-ZZ"
# shellcheck disable=SC2086
$PY scripts/refc_v3_train.py $COMMON --out "$W/arms/$name" \
    --ego-dropout 0.5 --withheld-bank fixed \
    > "$W/arms/$name.log" 2>&1
rc=$?
echo "ZZDONE-$name-rc$rc-$(date +%T)-ZZ"
if [ ! -s "$W/arms/$name/ckpt.pt" ] || [ ! -s "$W/arms/$name/metrics.jsonl" ]; then
  echo "ZZBAD-$name-NO-ARTIFACT-ZZ"
fi
