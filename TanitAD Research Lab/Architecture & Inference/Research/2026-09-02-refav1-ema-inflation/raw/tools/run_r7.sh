#!/usr/bin/env bash
# E-ARCH-TSC-2 R7 — resume tolerance: copy B' (an OFF checkpoint, no ema.* keys) and resume it
# under --ema-targets for 10 more steps. Do NOT fix anything; the outcome is the measurement.
set -u
cd /c/Users/Admin/tsc2 || exit 9
export PYTHONIOENCODING=utf-8 PYTHONPATH='C:\Users\Admin\tsc2\stack;C:\Users\Admin\tanitad-wt\colab'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
SRC='C:\Users\Admin\tsc2\Bp'; DST='C:\Users\Admin\tsc2\R7_resume'
rm -rf /c/Users/Admin/tsc2/R7_resume
MSYS_NO_PATHCONV=1 robocopy "$SRC" "$DST" /E /NFL /NDL /NJH /NP /R:3 /W:2 > /c/Users/Admin/tsc2/R7_robocopy.log 2>&1
echo "robocopy exit $? ; md5 of the two ckpts:"
md5sum /c/Users/Admin/tsc2/Bp/ckpt.pt /c/Users/Admin/tsc2/R7_resume/ckpt.pt
# keep B''s own launcher record aside (the launcher overwrites config.json in the run dir)
cp /c/Users/Admin/tsc2/R7_resume/config.json /c/Users/Admin/tsc2/R7_resume/config_Bprime_source.json
echo "R7 start $(date -u +%H:%M:%SZ)"
"$PY" tools/run_arm.py --arm R7 --out "$DST" --note "R7 resume tolerance: B' OFF ckpt (no ema.* keys) resumed under --ema-targets, +10 steps" -- \
  --cache 'C:\Users\Admin\tsc2\cache' --episodes 'C:\Users\Admin\refav1_eval_slice\eps' \
  --labels 'C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz' \
  --nav 'C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz' \
  --steps 260 --bs 1 --lru 16 --log-every 10 --save-every 250 --seed 0 \
  --target-space frozen --detach-aux-targets --bptt-truncate 15 --min-participation 0 --device cuda \
  --resume --ema-targets > R7_resume.log 2>&1
echo "R7 exit rc=$? $(date -u +%H:%M:%SZ)"
grep -a -n -i -E "resumed from step|Missing key|Unexpected key|Error|Traceback|run_arm\]" R7_resume.log | cut -c1-400 | head -20
tail -3 /c/Users/Admin/tsc2/R7_resume/train_log.jsonl | cut -c1-300
