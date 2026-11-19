#!/bin/bash
# navrbd_chain.sh -- ONE detached dev-box chain: every remaining input of the NAVRBD read, then the read (card PREREG_V8F_NAVRBD.md, FROZEN sha256 881d7f3d, sections 0a and 5).  CPU, 2 threads, zero training.
# Launched by launch_wmi.ps1 -> job_navrbd_chain.cmd -> this script (WMI, CreateFlags 0x08000200, stdin NUL, Git usr\bin\bash.exe).  Re-runnable: a finished step leaves out/steps/<name>.ok and is skipped.
#   0  gate: the UM read log ends with its DONE or FAIL sentinel (it is live and F: has < 1 GiB margin: NOTHING over a few MB is written to F: before), F: >= 5 GB free, commit headroom >= 5.5 GB
#   1  pull the N1D_s0 / N1D_s1 checkpoints from the pod (sha256 against the pod's own sha256sum)
#   2  export the minimal tree of 0c3b374f (blob-hash verified); eval-141 T1 dumps (navrbd_dumps.py, 4 tags); eval-141 T0 passes real / donor / zero x 2 arms (navrbd_t0.py)
#   3  STRATEGIC in full: the view (N1D presented as N1), frozen strat1_strategic.py and strat1_ctx_read.py, the no-fog ctx read, navrbd_strategic.py (the N1D - N1R contrast)
#   4  (optional) held-out Z-EVAL mask: NOT-COMPUTABLE from the B2 tables (see HANDOFF_RBDSTRAT.md); the read prints that cell NOT-YET
#   5  wait for the pod's FINAL line `ZZ RBDPOD ALL END ZZ main: ...; zero dumps DONE|FAIL` (poll 5 min, cap 6 h; the main script's DONE / FAIL line is written BEFORE the zero dumps and is NOT the end); the line is logged;
#      then pull every dump_NAVRBD-HO-* dir (sha256-verified) into read/dumps/
#   6  pytest -q test_navrbd_read.py (whole file) and test_navrbd_chain.py, then navrbd_read.py
# Last line of the log: "ZZ NAVRBD READ DONE ZZ" or "ZZ NAVRBD READ FAIL ZZ <step and why>".  Waiters match that sentinel.
# DRY RUN: NAVRBD_STANDIN_ROOT=<dir> (holds t0/, d141/ stand-in passes) skips steps 0-2 and 5 and reads the stand-ins; the read uses --skip-141 and the N1R held-out dumps under the N1D names (a code check; files named *_standin).
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
B=F:/TanitAD-scratch/strat1/navrbd
R=$B/read
SR=F:/TanitAD-scratch/strat1/read
UMLOG=${NAVRBD_UMLOG:-F:/TanitAD-scratch/datascale/read_mix/out/dscale_mix.log}
PY=${NAVRBD_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
COMMIT=0c3b374f
SUF=""
if [ -n "${NAVRBD_STANDIN_ROOT:-}" ]; then
  SUF="_standin"; export NAVRBD_T0="$NAVRBD_STANDIN_ROOT/t0" NAVRBD_D_141="$NAVRBD_STANDIN_ROOT/d141"
  O=${NAVRBD_STANDIN_OUT:-$NAVRBD_STANDIN_ROOT/out}
else
  export NAVRBD_T0=$R/out/t0 NAVRBD_D_141=$R/dumps141; O=$R/out
fi
export NAVRBD_TREE=${NAVRBD_TREE:-$B/tree}
LOG=$O/navrbd_read_chain${SUF}.log
STEPS=$O/steps${SUF}
VIEW=$O/view${SUF}
cd "$R" || exit 1
mkdir -p "$O" "$STEPS"
fail() { echo "ZZ NAVRBD READ FAIL ZZ $1" | tee -a "$LOG"; echo "FAIL $1" > "$O/NAVRBD_READ${SUF}.done"; exit 1; }
gate() { $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"; }
# step NAME DESCRIPTION cmd...: skipped when out/steps/NAME.ok exists; on failure the chain stops with the step named
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n (done)" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[navrbd chain] start $(date) commit $COMMIT standin=${NAVRBD_STANDIN_ROOT:-no}" >> "$LOG"

# ---- 0 gate
if [ -z "${NAVRBD_STANDIN_ROOT:-}" ]; then
  echo "[step] gate0: waiting for the UM read sentinel in $UMLOG $(date)" >> "$LOG"
  OKUM=0
  for i in $(seq 1 1440); do
    L=$(tail -n 1 "$UMLOG" 2>/dev/null | tr -d '\r')
    case "$L" in "ZZ DSCALE MIX READ DONE ZZ"*|"ZZ DSCALE MIX READ FAIL ZZ"*) OKUM=1; echo "UM read ended: $L" >> "$LOG"; break;; esac
    sleep 60
  done
  [ "$OKUM" = 1 ] || fail "gate0: the UM read log never ended with its sentinel within 24 h"
  FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
  [ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "gate0: F: has under 5 GB free ($FREE GB)"
  # ---- 1 pull the checkpoints
  step pull_ckpts "pull N1D_s0 / N1D_s1 from the pod, sha256-verified" $PY -u navrbd_pod.py pull-ckpts
  # ---- 2 export, dumps, T0
  step export_tree "export $COMMIT minimal tree, blob-hash verified" $PY -u navrbd_export.py $COMMIT "$NAVRBD_TREE"
  step dumps141 "eval-141 T1 dumps (4 tags)" $PY -u navrbd_dumps.py NAVRBD-N1D_s0 NAVRBD-N1D_s1 NAVRBD-N1D_s0Z NAVRBD-N1D_s1Z
  step t0 "eval-141 T0 passes real / donor / zero" $PY -u navrbd_t0.py --out "$NAVRBD_T0"
fi
# ---- 3 STRATEGIC in full
gate view
rm -rf "${VIEW:?}"
$PY -u navrbd_view.py "$VIEW" >> "$LOG" 2>&1 || fail "navrbd_view.py (a pass is missing)"
$PY -u $SR/strat1_strategic.py --t0 "$VIEW/t0" --t1 "$VIEW/t1" --out "$O/NAVRBD_STRATEGIC_view${SUF}.json" >> "$LOG" 2>&1 || fail "frozen strat1_strategic.py on the view"
$PY -u $SR/strat1_ctx_read.py --t0 "$VIEW/t0" --out "$O/NAVRBD_CTX_view${SUF}.json" >> "$LOG" 2>&1 || fail "frozen strat1_ctx_read.py on the view"
$PY -u F:/TanitAD-scratch/strat1/navrb2/read/navrb2_ctx_nofog.py --t0 "$VIEW/t0" --out "$O/NAVRBD_CTX_nofog${SUF}.json" >> "$LOG" 2>&1 || fail "navrb2_ctx_nofog.py (no-fog ctx read)"
$PY -u navrbd_strategic.py --t0 "$NAVRBD_T0" --out "$O/NAVRBD_STRATEGIC_contrast${SUF}.json" >> "$LOG" 2>&1 || fail "navrbd_strategic.py (N1D - N1R contrast)"
rm -rf "${VIEW:?}"
# ---- 4 held-out Z-EVAL mask: not computable from the B2 tables; the read prints NOT-YET for that cell
echo "[step] ho_mask: NOT-COMPUTABLE (the B2 tables hold only the 4,572 train clips and the 141 eval clips; the held-out episodes are not in them). Held-out Z-EVAL reads NOT-YET." >> "$LOG"
# ---- 5 the pod
READARGS=""; READENV="NAVRBD_STANDIN_DUMMY=1"
if [ -z "${NAVRBD_STANDIN_ROOT:-}" ]; then
  echo "[step] pod_wait: polling the pod log for ZZ RBDPOD ALL END ZZ $(date)" >> "$LOG"
  NAVRBD_POLL_S=300 NAVRBD_CAP_H=6 $PY -u navrbd_pod.py wait-end >> "$LOG" 2>&1 || fail "pod_wait: no 'ZZ RBDPOD ALL END ZZ' line within 6 h"
  step pull_dumps "pull every dump_NAVRBD-HO-* dir, sha256-verified" $PY -u navrbd_pod.py pull-dumps
else
  READENV="NAVRBD_D_HO=$B/step0/t1"
  READARGS="--skip-141 --alias N1D_s0=N1R_s0 --alias N1D_s1=N1R_s1 --alias N1Dx_s0=N1R_s0"
fi
# ---- 6 the read
gate read
env -u NAVRBD_T0 -u NAVRBD_D_141 -u NAVRBD_D_HO $PY -m pytest -q -p no:cacheprovider --basetemp="$O/bt${SUF}" test_navrbd_read.py >> "$LOG" 2>&1 || fail "pytest test_navrbd_read.py is not green"
env -u NAVRBD_T0 -u NAVRBD_D_141 -u NAVRBD_D_HO $PY -m pytest -q -p no:cacheprovider --basetemp="$O/bt${SUF}" test_navrbd_chain.py >> "$LOG" 2>&1 || fail "pytest test_navrbd_chain.py is not green"
env $READENV $PY -u navrbd_read.py --out "$O/RESULT_NAVRBD${SUF}.json" --contrast "$O/NAVRBD_STRATEGIC_contrast${SUF}.json" $READARGS >> "$LOG" 2>&1 || fail "navrbd_read.py (a control problem or a missing input: see the PROBLEM lines above)"
rm -rf "${O:?}/bt${SUF}"
echo "DONE" > "$O/NAVRBD_READ${SUF}.done"
echo "ZZ NAVRBD READ DONE ZZ" | tee -a "$LOG"
