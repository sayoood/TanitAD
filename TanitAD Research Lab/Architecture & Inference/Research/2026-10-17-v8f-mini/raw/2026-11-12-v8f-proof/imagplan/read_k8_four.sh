#!/usr/bin/env bash
# read_k8_four.sh : dev box, card PREREG_V8F_PROOF_ADD40 note N6 section 3, "beside, naming nothing": the four-base read (bases 0 to 3 in one two-level bootstrap)
# of the same differences as read_k8.sh, for RLFT and RLAm, with the headway reader r2. It names nothing. It waits for the final line of imagplan/read_p5b_k8.out
# (one Python job at a time), then runs each read only with 5.5 GB of commit memory free. Starts nothing on the pod, deletes nothing. Last line: ZZ READ-K8-FOUR DONE | STOP.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1"
export PULL_ROOT=$H/imagplan/pull_t1 RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
j=0; while ! grep -q '^ZZ READ-P5B-K8' imagplan/read_p5b_k8.out 2>/dev/null; do j=$((j+1)); [ $j -gt 240 ] && { echo "ZZ READ-K8-FOUR STOP: the p5b read has not ended in 4 h ($(ts))"; exit 3; }; sleep 60; done
memwait() {
  local kb i
  for i in $(seq 1 60); do
    kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory" | tr -d '\r\n ')
    [ "${kb:-0}" -ge 5767168 ] && return 0
    sleep 60
  done
  echo "ZZ READ-K8-FOUR STOP: free commit memory stayed under 5.5 GB for 60 min ($(ts))"; exit 3
}
for A in RLFT RLAm; do
  n() { echo "${1}x_$A"; }
  PAIRS="$(n IPl):$(n IPo),$(n IPk):$(n IPo),$(n IPk):$(n IPc),$(n IPk):$(n IPl),$(n IPc):$(n IPo),$(n IPk):$(n IPi),$(n IPi):$(n IPo)"
  RW=$W/imagplan/read_k8/four/$A; mkdir -p "$H/imagplan/read_k8/four/$A"
  memwait
  ( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1,2,3 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$H/imagplan/read_k8/four/$A/hw.log" 2>&1 ); echo "[$(ts)] headway four bases $A rc $? : $(tail -n 1 "$H/imagplan/read_k8/four/$A/hw.log" | cut -c1-160)"
done
memwait
"$PY" - <<'PYEOF'
import json
for A in ("RLFT", "RLAm"):
    d = json.load(open("imagplan/read_k8/four/%s/hw/RESULT_HEADWAY.json" % A))
    print(A, "controls ALL_PASS", d["controls"].get("ALL_PASS"))
    for k, v in d["pairs"].items():
        c = v["cells"]["U"]
        print("  ", k, "U per base", [round(x, 4) for x in c["d_s"]], "pooled %.4f [%.4f, %.4f] excludes 0 %s" % (c["D"], c["A_lo"], c["A_hi"], c["A_excludes_0"]), "seeds", v.get("seeds_read"))
PYEOF
echo "ZZ READ-K8-FOUR DONE $(ts)"
