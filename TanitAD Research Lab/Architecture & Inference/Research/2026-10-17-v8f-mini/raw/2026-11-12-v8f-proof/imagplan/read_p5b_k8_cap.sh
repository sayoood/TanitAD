#!/usr/bin/env bash
# read_p5b_k8.sh : dev box, card PREREG_V8F_PROOF_ADD40 note N6 section 3, control p5b (plumb_read3.py, unchanged; source cand-imag, family decel, bars unchanged:
# at least 0.5 for an arm the reader names IMAG-RESPONDS, below 0.25 for one it names IMAG-DEAF). Arms: RLFT and RLAm at bases 2 and 3, the dose arms RLA3 and RLA2
# at bases 0 and 1. The responds flag of each arm is the name the reader of card ADD36 gave it (RLAm, RLA3, RLA2: IMAG-RESPONDS; RLFT: IMAG-DEAF).
# One Python job at a time, each only with 2.5 GB of commit memory free (60 s steps, at most 60 min per wait). Starts nothing on the pod, deletes nothing.
# Last line: ZZ READ-P5B-K8 DONE | STOP.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
H=/f/TanitAD-scratch/v8proof
cd $H/imagplan || exit 1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
ts() { date -u +%H:%M:%SZ; }
memwait() {
  local kb i
  for i in $(seq 1 60); do
    kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory" | tr -d '\r\n ')
    [ "${kb:-0}" -ge 2621440 ] && return 0
    sleep 60
  done
  echo "ZZ READ-P5B-K8 STOP: free commit memory stayed under 2.5 GB for 60 min ($(ts))"; exit 3
}
for spec in RLAm_s2:yes RLAm_s3:yes RLFT_s2:no RLFT_s3:no RLA3_s0:yes RLA3_s1:yes RLA2_s0:yes RLA2_s1:yes; do
  B=${spec%%:*}; R=${spec##*:}
  [ -f pull/IPix_$B/p45_IPix_$B.npz ] && [ -f pull/IPix_$B/imag_log_IPix_$B.npz ] || { echo "[$(ts)] $B: the logs of its cand-imag run are not pulled"; continue; }
  cv=()
  [ -f pull/IPcx_$B/imag_log_IPcx_$B.npz ] && cv=(--cv-log pull/IPcx_$B/imag_log_IPcx_$B.npz)
  memwait
  "$PY" plumb_read3.py --p45 pull/IPix_$B/p45_IPix_$B.npz --imag-log pull/IPix_$B/imag_log_IPix_$B.npz ${cv[@]+"${cv[@]}"} --responds $R --out read_plumb/PLUMB3_$B.json > read_plumb/plumb3_$B.log 2>&1
  echo "[$(ts)] $B responds $R rc $?"
done
memwait
"$PY" - <<'PYEOF'
import json, os
for b in ("RLAm_s0", "RLAm_s1", "RLAm_s2", "RLAm_s3", "RLFT_s0", "RLFT_s1", "RLFT_s2", "RLFT_s3", "RLA3_s0", "RLA3_s1", "RLA2_s0", "RLA2_s1"):
    p = "read_plumb/PLUMB3_%s.json" % b
    if not os.path.isfile(p):
        print(b, "no record"); continue
    d = json.load(open(p)); q = d["p5b"]
    print(b, "p5b slope %.4f ci [%.4f, %.4f] pairs %d responds %s PASS %s | p4b %s | p7 %s" % (q["slope_T2.0"], q["ci_episodes"][0], q["ci_episodes"][1], q["n_pairs"], q["responds"], q["PASS"],
          (d.get("p4b") or {}).get("PASS"), (d.get("p7") or {}).get("PASS")))
PYEOF
echo "ZZ READ-P5B-K8 DONE $(ts)"
