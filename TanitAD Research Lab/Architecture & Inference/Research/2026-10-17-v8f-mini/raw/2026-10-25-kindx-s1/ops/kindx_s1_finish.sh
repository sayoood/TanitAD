#!/usr/bin/env bash
# kindx_s1_finish.sh (MM, 2026-10-25 21:40): the relaunched KINDX seed-1 chain finished every box-2 stage (train records PASS, 7 MIXO evals and
# the 10 KINDX evals rc=0, X sha 68a0f335... before and after the evals), then stopped at its step 7 on a dev-box pull of
# `eval_M2b-mixo_s1-kindx-own.log`: the 10 KINDX evals run in ONE process (`v8f_mini_eval_kindx.py evals`) and write no per-eval .log, so the
# chain's pull list (chain_v8f_kindx_s1.sh:322, `json npz log`) asked for files that never exist. This runs step 7 EXACTLY as the chain does,
# with the KINDX evals' .log replaced by that one process's job log, at the same tip export the chain extracted. Nothing on box 2 runs.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_kindx_s1; O=$L/kindx_s1_finish.out
X=$B/v8fkindx_tip_20261025212536; PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"; KX="$X/$PK/raw/2026-10-25-kindx"
R=D:/tanitad/runs/v8f_mini_full; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $O; }
stop() { echo "ZZV8FMINI-STOP $* (kindx_s1_finish.sh)" | tee -a $O; exit 1; }
[ "$(cat $X/EXPORT_COMMIT.txt 2>/dev/null || git -C /c/Users/Admin/tanitad-push -c safe.directory=* ls-remote origin agent/arch-inf-20260803 | cut -f1)" ] || true
for n in own none all union only-speed only-angle only-tte only-timegap only-geom only-leadgap; do
  for x in json npz; do f="eval_M2b-mixo_s1-kindx-$n.$x"; [ -s "$E/eval/$f" ] || $SCP "tanitad-box2:$R/eval/$f" "$E/eval/" || stop "pull $R/eval/$f"; done
done
$SCP "tanitad-box2:$R/eval/eval_M2b-mixo_s1-kindx-own_proj.npz" "$E/eval/" 2>/dev/null && say "pulled the optional own_proj side file" || say "no own_proj side file (optional)"
$SCP "tanitad-box2:D:/tanitad/logs/v8fkindx10252120_k_M2b_s1_mixo.log" "$E/eval/kindx_evals_job.log" || stop "pull the KINDX evals' job log"
say "pulled the 10 KINDX evals (json + npz) and their one job log"
grep ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null && stop "guard stop"
export PYTHONIOENCODING=utf-8 KINDX_OUT="$(cygpath -m $E)" KINDX_CODE="$(cygpath -m $X)" KINDX_EV="$(cygpath -m $E/eval)" KINDX_BE="$(cygpath -m $E/b_kind/eval)" KINDX_SEED=1
export CUDA_VISIBLE_DEVICES=-1
"$PY" -I "$(cygpath -m "$KX")/kindx_controls.py" --seed 1 1 >> $O 2>&1 || stop "kindx controls 1"
"$PY" -I "$(cygpath -m "$KX")/kindx_controls.py" --seed 1 2 >> $O 2>&1 || stop "kindx controls 2"
"$PY" -I "$(cygpath -m "$KX")/kindx_read.py" --seed 1 >> $O 2>&1 || stop "kindx read s1"
r=$("$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('reading'), '| next:', d.get('committed_next'))" "$(cygpath -m $E/READ_s1.json)")
echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') KINDX seed 1, finished by kindx_s1_finish.sh after the chain's pull-list stop: $r" | tee -a $O
