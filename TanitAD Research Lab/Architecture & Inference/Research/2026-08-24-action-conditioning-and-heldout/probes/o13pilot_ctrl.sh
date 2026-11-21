#!/usr/bin/env bash
# ⛔ THE MATCHED CONTROL, WITHOUT WHICH THE PILOT SAYS NOTHING ABOUT o5.
#
# The pilot shows o13_excess +0.17..+0.21 with its shuffled control at the floor.
# That is encouraging and it is NOT sufficient: **O11 ALSO showed a positive
# excess** (pick_acc 1.000) while degrading o5 by +18.7 % — separation without
# accuracy, the degenerate solution. The ONLY way to tell the two apart is o5 on
# a matched arm, same corpus, same seed, same steps, differing ONLY in the o13
# weight. Comparing the pilot's o5 to the 12-step smoke's would be a scope error.
#
# ⚠️ NON-PARITY, 24 episodes, TRAINING-SET ONLY: this pair answers
# "does o13 cost prediction accuracy?", NOT "does it generalise?". The
# generalisation read is the pre-registered held-out one on the Thor arm.
set -u
SPD="$(cd "$(dirname "$0")" && pwd)"
M="C:/Users/Admin/tanitad-mirror/stack"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
PS="/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# wait for the pilot to finish -- one GPU, and concurrent torch arms make NO
# progress while looking exactly like a hang (MEASURED: 7 arms, sm 0-6 %, 50 min).
while true; do
  n=$($PS -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'o13pilot' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
  [ "${n:-0}" = "0" ] && break
  sleep 60
done
echo "pilot done; launching the matched w=0 control"
rm -rf "$SPD/o13ctrl"
cd "$M" || exit 1
PYTHONPATH="$M" PYTHONIOENCODING=utf-8 "$PY" scripts/train_v6_staged.py \
  --out "$SPD/o13ctrl" --stage S-W \
  --v2-cache "$SPD/sp2/cache/v7tiny-heldout24-w120-256x640cyl" \
  --frame-h 256 --frame-w 640 --patch 16 --enc-dim 128 --enc-depth 3 --enc-heads 4 \
  --pred-dim 256 --pred-depth 3 --pred-heads 4 --readout-grid 4 --readout-grid-w 8 \
  --readout-dim 64 --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 8 --d-tac 128 \
  --d-str 64 --steps 3000 --batch 4 --v2-lru 4 --log-every 100 --save-every 100000 \
  --seed 0 --spectrum-accum 43 --sigreg-slices 512 --no-require-parity \
  --w-o13-ego 0 --o13-k 4 > "$SPD/o13ctrl.log" 2>&1
echo "ZZCTRL-DONE ZZ"
