#!/bin/sh
# REPLACES chain_after_arms.sh (killed by explicit PID 3008 at 14:04Z, confirmed gone).
#
# WHAT CHANGED AND WHY, stated rather than silently dropped: the original queued a
# `ctrl_null` arm at 2,000 steps as a DOSE-MATCHED zero-information floor for `veto2k`.
# `veto2k_s0` then landed with NO separated improvement on any feasibility metric
# (fan_peak_g_mean -0.00777 ns, top32_infeasible -0.00475 ns) -- and a floor is a HURDLE FOR
# A POSITIVE CLAIM. Adding a hurdle cannot make an unseparated delta quotable, so the
# 36-minute control cannot change the 2k verdict and the GPU is better spent on the T1
# four-family read, which is a committed deliverable.
# CONSEQUENCE, recorded: the 2k arm's WORSENINGS (sel_peak_g +0.01046 sep, R3 +0.01214 sep)
# are NOT attributable between the veto and the optimizer's own drift without that control.
# They are reported as unattributed rather than as veto effects.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
W="/c/Users/Admin/rl_refcv3_min"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO" LAUNCH_APPROVED=1

while ! grep -q "ZZARMS-COMPLETE" /c/Users/Admin/veto_run/arms_run.log 2>/dev/null; do
  grep -q "ZZBAD-" /c/Users/Admin/veto_run/arms_run.log 2>/dev/null && { echo "ZZCHAIN-ABORT-arms-failed-ZZ"; exit 1; }
  sleep 20
done
echo "ZZCHAIN-ARMS-DONE-$(date -u +%H:%M:%S)Z-ZZ"
echo "ZZCHAIN-SKIP-n2k-ctrl_null-no-positive-2k-claim-to-floor-ZZ"

if [ ! -s /c/Users/Admin/veto_run/raw/fan_rerank.json ]; then
  echo "ZZSTART-rerank-$(date -u +%H:%M:%S)Z-ZZ"
  "$PY" -u "$REPO/stack/scripts/rl_fan_rerank_probe.py" \
    --ckpt "$W/base/ckpt_step40284_frozen.pt" --config "$W/base/config.json" --expect-step 40284 \
    --episodes /c/Users/Admin/run_refcv3_ol/data/eval \
    --labels /c/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz \
    --lead-block /c/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz \
    --windows 480 --batch 4 --lru 6 --device cuda --n-boot 4000 --seed 11 \
    --out /c/Users/Admin/veto_run/raw/fan_rerank.json \
    > /c/Users/Admin/veto_run/raw/fan_rerank.log 2>&1
  echo "ZZDONE-rerank-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
fi
echo "ZZCHAIN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
