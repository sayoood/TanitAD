#!/usr/bin/env bash
# PREREG_V8F_TAC Amendment 1: finish the chain that stopped at its PULL step (2026-10-21 18:16) WITHOUT re-running a single eval.
# The 32 group-oracle evals completed (v8ftacgrp_run rc=0, 32 fresh files on box 2). The chain named its list GROUPS, a bash
# builtin that silently ignores assignment and expands to the user's gid (197121), so it tried to pull eval_*-tacorc-197121.npz.
# This runs steps 3 onward exactly as the chain would have, with the list renamed TAC_GROUPS: pull, read, telemetry stop, DONE.
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_tacgrp; mkdir -p $E/eval
TAC_GROUPS="goal latdec londec latcons lonspeed stoplead decisions constraints"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
EVR=D:/tanitad/runs/v8f_mini_full/eval
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
X=${X:?the chain s reader extract, e.g. X=$B/v8ftacgrp_tip_20261021180700}
T=${T:?the tip the chain refreshed box 2 to}
stop() { echo "ZZV8FTAC-STOP $*"; exit 1; }
grep -q '^ZZV8FTAC-STOP pull ' $L/chain_v8f_tacgrp.out || stop "the chain did not stop at its pull step: this finisher does not apply"
grep -q 'v8ftacgrp_run rc=0' $L/chain_v8f_tacgrp.out || stop "the run step did not end rc=0"
RD="$(cygpath -m "$X/$PK/code")"; [ -f "$RD/v8f_tac_groups_read.py" ] || stop "the extract has no v8f_tac_groups_read.py"
n=0
for c in M1_s0 M1_s1 M2b_s0 M2b_s1; do
  for s in tacpred tacorc; do $SCP "tanitad-box2:$EVR/eval_$c-$s.npz" $E/eval/ || stop "pull eval_$c-$s.npz"; done
  for g in $TAC_GROUPS; do s=tacorc-$g
    for x in npz json; do $SCP "tanitad-box2:$EVR/eval_$c-$s.$x" $E/eval/ || stop "pull eval_$c-$s.$x"; done; n=$((n+1))
  done
done
[ $n = 32 ] || stop "pulled $n group evals, expected 32"
echo "$(date '+%m-%d %H:%M') pulled 32 group evals + 8 tacpred / tacorc"
PYTHONIOENCODING=utf-8 $PY "$RD/v8f_tac_groups_read.py" --eval-dir "$(cygpath -m $E/eval)" --tac-read "$(cygpath -m $L/v8f_tac/tac_read.json)" \
  --out "$(cygpath -m $E/tac_groups_read.json)" || stop "tac groups read"
$SSH "powershell -NoProfile -Command \"New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null\"" 2>/dev/null
$SCP tanitad-box2:D:/tanitad/logs/telemetry_full.csv $E/telemetry_v8ftacgrp.csv 2>/dev/null
echo "ZZV8FTAC-DONE $(date '+%m-%d %H:%M') tip $T: $(PYTHONIOENCODING=utf-8 $PY -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/tac_groups_read.json)")"
