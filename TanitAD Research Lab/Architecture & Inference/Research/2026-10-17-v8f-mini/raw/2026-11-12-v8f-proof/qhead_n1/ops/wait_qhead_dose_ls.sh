#!/bin/bash
# wait_qhead_dose_ls.sh -- the LS checkpoints (note PREREG_V8F_PROOF_ADD42_N3), made by make_ls_dose.py from wait_qhead_dose.sh. First form:
# wait_qhead_dose.sh -- dev box, note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step, LSD type). Written by the lead. CPU only on both machines.
# 1. polls the pod every 120 s for the end marker of pod_qhead_dose_inputs.sh (bound 4 h); 2. pulls the row layout; 3. builds the label pack of all 4,572 training
# clips on the dev box (build_dose_pack.py) and runs control d1 against the banked pack of the 1,200 probe clips; 4. pushes the pack and the d1 record to the pod;
# 5. runs control d2 on the pod for both checkpoints (GPU hidden; only with 20 GB of the container memory limit free); 6. only if d1 and d2 pass: starts the two
# fits on the pod CPU (pod_qhead_dose_fits.sh) and waits for their end marker; 7. pulls the head records and heads. Starts no GPU job. Deletes nothing.
# Last line: ZZ QHEAD-DOSE-WAIT DONE | ZZ QHEAD-DOSE-WAIT STOP: <why> | ZZ QHEAD-DOSE-WAIT TIMEOUT.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
cd $H || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
O=/workspace/v8proof/qhead_dose_ls
T=/workspace/v8proof/qhead_dose/tools
DD=$H/qhead_n1/dose; L=$DD/pulled_ls; LW=$W/qhead_n1/dose/pulled_ls
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
stop() { echo "ZZ QHEAD-DOSE-WAIT STOP: $* ($(ts))"; exit 0; }
pull1() {  # remote local
  local a b
  a=$(pod 60 "sha256sum $1 2>/dev/null | cut -d' ' -f1")
  [ -n "$a" ] || { echo "[$(ts)] not on the pod: $1"; return 1; }
  mkdir -p "$(dirname "$2")"
  bash "$C" "$1" "$2" < /dev/null > /dev/null 2>&1
  b=$(sha256sum "$2" 2>/dev/null | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "[$(ts)] pulled $(basename "$1") sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $1"; return 1; fi
}
push1() {  # local remote
  local a b
  a=$(sha256sum "$1" | cut -d' ' -f1)
  timeout 600 bash "$P" "cat > $2" < "$1" 2>/dev/null | sed '/hf_/d'
  b=$(pod 120 "sha256sum $2 | cut -d' ' -f1" | tr -d '\n')
  if [ "$a" = "$b" ]; then echo "[$(ts)] pushed $(basename "$2") sha256 ${a:0:16}"; else echo "[$(ts)] PUSH-MISMATCH $2"; return 1; fi
}
memwait() {
  local kb i
  for i in $(seq 1 30); do
    kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory" | tr -d '\r\n ')
    [ "${kb:-0}" -ge 5767168 ] && return 0
    sleep 60
  done
  stop "free commit memory on the dev box stayed under 5.5 GB"
}
waitmark() {  # <done marker> <stop marker> <max polls of 120 s> ; sets st
  local n=0 s
  st=""
  while true; do
    s=$(pod 60 "[ -f $1 ] && echo DONE; [ -f $2 ] && echo STOP; echo MARK")
    case "$s" in DONE*) st=DONE; return 0 ;; STOP*) st=STOP; return 0 ;; esac
    n=$((n+1)); [ "$n" -gt "$3" ] && { st=TIMEOUT; return 0; }
    sleep 120
  done
}
waitmark $O/run/QHEAD_DOSE_INPUTS_DONE.txt $O/run/QHEAD_DOSE_INPUTS_STOP.txt 120
echo "[$(ts)] input job state $st"
pod 60 "grep -a 'DONE\|STOP\|rc \|seconds\|start;' $O/run/dose_inputs.log | tail -n 14 | cut -c1-260; ls -la --time-style=+%H:%M $O/states | cut -c24-120"
[ "$st" = TIMEOUT ] && { echo "ZZ QHEAD-DOSE-WAIT TIMEOUT $(ts)"; exit 8; }
[ "$st" = DONE ] || stop "the input job stopped"
[ "$(pod 60 "cat $O/run/QHEAD_DOSE_INPUTS_DONE.txt" | tr -d ' \n')" = 2 ] || stop "the input job did not complete both checkpoints"
pull1 $O/layout/train_layout_0_4572.npz $L/layout/train_layout_0_4572.npz || stop "layout pull"
pull1 $O/layout/train_layout_0_4572.npz.json $L/layout/train_layout_0_4572.npz.json || echo "[$(ts)] no layout record file"
mkdir -p $L/pack $L/controls $L/heads
memwait
"$PY" $DD/build_dose_pack.py --layout "$LW/layout/train_layout_0_4572.npz" --join C:/Users/Admin/lorapod/ship2/join_train.npz --out "$LW/pack/train_pack_all.npz" > $L/pack/build.log 2>&1; rc=$?
echo "[$(ts)] build pack rc $rc : $(tail -n 1 $L/pack/build.log | cut -c1-300)"
[ $rc = 0 ] && [ -f $L/pack/train_pack_all.npz ] || stop "the pack build failed"
memwait
"$PY" $DD/dose_d1_pack_check.py --new-pack "$LW/pack/train_pack_all.npz" --banked-pack "$W/spatial/leadsup/read/train_pack.npz" --out "$LW/controls/d1_pack.json" > $L/controls/d1.log 2>&1; rc=$?
echo "[$(ts)] control d1 rc $rc : $(tail -n 1 $L/controls/d1.log | cut -c1-300)"
[ $rc = 0 ] || stop "control d1 failed: the new pack does not equal the banked pack on positions 600 to 1799"
push1 $L/pack/train_pack_all.npz $O/pack/train_pack_all.npz || stop "pack push"
push1 $L/controls/d1_pack.json $O/controls/d1_pack.json || stop "d1 record push"
d2ok=1
for ARM in LS_s0 LS_s1; do
  out=$(pod 1500 "a=\$(awk '\$1==\"anon\"{print \$2}' /sys/fs/cgroup/memory.stat); m=\$(cat /sys/fs/cgroup/memory.max); g=\$(( (m - a) / 1073741824 )); if [ \$g -lt 20 ]; then echo D2-NOT-RUN free \$g GB; else CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 nice -n 19 /workspace/venv/bin/python $T/dose_d2_state_check.py --new $O/states/${ARM}_dose_train.npz --banked /workspace/v8proof/probe/states/${ARM}_train.npz --layout $O/layout/train_layout_0_4572.npz --out $O/controls/d2_${ARM}.json 2>&1 | tail -n 2 | cut -c1-300; echo D2-RC \${PIPESTATUS[0]}; fi")
  echo "[$(ts)] control d2 $ARM: $(echo "$out" | tr '\n' ' ' | cut -c1-420)"
  echo "$out" | grep -q "D2-RC 0" || d2ok=0
  pull1 $O/controls/d2_${ARM}.json $L/controls/d2_${ARM}.json
done
[ $d2ok = 1 ] || stop "control d2 failed or was not run for a checkpoint: no fit is started"
pod 30 "cd $O && QD_OUT=$O QD_TOOLS=$T nohup setsid bash $T/pod_qhead_dose_fits_ls.sh > $O/start_fits.out 2>&1 < /dev/null &" > /dev/null
echo "[$(ts)] fits started on the pod CPU"
sleep 30
waitmark $O/run/QHEAD_DOSE_FITS_DONE.txt $O/run/QHEAD_DOSE_FITS_STOP.txt 90
echo "[$(ts)] fit job state $st"
pod 60 "tail -n 8 $O/run/dose_fits.log | cut -c1-330; tail -n 3 $O/start_fits.out | cut -c1-200"
[ "$st" = TIMEOUT ] && { echo "ZZ QHEAD-DOSE-WAIT TIMEOUT $(ts)"; exit 8; }
for b in LS_s0 LS_s1; do
  pull1 $O/heads/$b.qhead.json $L/heads/$b.qhead.json
  pull1 $O/heads/$b.qhead.npz $L/heads/$b.qhead.npz
done
pull1 $O/run/dose_fits.log $L/dose_fits.log
pull1 $O/run/dose_inputs.log $L/dose_inputs.log
[ "$st" = DONE ] || stop "the fit job stopped"
echo "ZZ QHEAD-DOSE-WAIT DONE $(ts)"
