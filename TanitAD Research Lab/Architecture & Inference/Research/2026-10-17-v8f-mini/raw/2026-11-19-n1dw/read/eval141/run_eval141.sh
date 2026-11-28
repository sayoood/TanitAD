#!/bin/bash
# run_eval141.sh -- STAGED, NOT STARTED.  ONE detached dev-box chain: the eval-141 guard inputs of the N1DW read (card PREREG_V8F_N1DW.md sha256 59f2b7f9, section 5.2) and the guard read itself.
# CPU only, 2 threads, ONE process at a time, every stage gated on dev-box free commit >= 5.5 GB (wait_commit.py).  Re-runnable: a finished step leaves steps/<name>.ok and is skipped.
# Launch (when the PI / MM says go, not before): WMI CreateFlags 0x08000200, stdin NUL, via run_eval141.cmd (Git usr\bin\bash.exe, never Git\bin\bash.exe).
#   0  wait for E:/Projects/TanitAD-artifacts/p5c_read/markers/chain.DONE (another session's long read; any chain.FAILED* file there is IGNORED), cap 24 h
#   1  verify the four checkpoints' sha256 against ckpts/SHA256_PODS.txt (the pod's own sha256sum)
#   2  eval-141 T1 dumps, one process per tag: 4 follow + 4 zero twins (n1dw_dumps.py)
#   3  eval-141 T0 passes real / donor / zero x 4 arms, one process each (n1dw_t0.py)
#   4  STRATEGIC per weight: view, frozen strat1_strategic.py and strat1_ctx_read.py, no-fog ctx read; then n1dw_strategic.py (both weights; the ROUTE-COST contrast)
#   5  n1dw_read141.py (the four families, guard, Z-EVAL, Z-USE, K-RE)
# Last line of the log: "ZZ N1DW141 DONE ZZ" or "ZZ N1DW141 FAIL ZZ <reason>".  Waiters match that sentinel.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
E=F:/TanitAD-scratch/n1dw/read/eval141
SR=F:/TanitAD-scratch/strat1/read
N2R=F:/TanitAD-scratch/strat1/navrb2/read
PY=${N1DW_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
MARK=${N1DW_P5C_MARK:-E:/Projects/TanitAD-artifacts/p5c_read/markers/chain.DONE}
O=$E/out
STEPS=$O/steps
LOG=$O/n1dw141_chain.log
export N1DW_TREE=$E/tree N1DW_CKPTS=F:/TanitAD-scratch/n1dw/read/ckpts N1DW_D_141=$E/dumps141 N1DW_T0=$O/t0
cd "$E" || exit 1
mkdir -p "$O" "$STEPS"
fail() { echo "ZZ N1DW141 FAIL ZZ $1" | tee -a "$LOG"; echo "FAIL $1" > "$O/N1DW141.done"; exit 1; }
gate() { $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"; }
# step NAME DESCRIPTION cmd...: skipped when steps/NAME.ok exists; every step is gated; on failure the chain stops with the step named
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n (done)" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[n1dw141 chain] start $(date) commit 134f5da1 tree $N1DW_TREE" >> "$LOG"

# ---- 0 wait for the other session's read (chain.DONE only; chain.FAILED* is ignored)
OK0=0
for i in $(seq 1 1440); do
  if [ -f "$MARK" ]; then OK0=1; echo "p5c_read chain.DONE present: $(date)" >> "$LOG"; break; fi
  sleep 60
done
[ "$OK0" = 1 ] || fail "gate0: $MARK never appeared within 24 h"
# ---- 1 checkpoints
step ckpt_sha "checkpoint sha256 equal to the pod's" bash -c "cd F:/TanitAD-scratch/n1dw/read/ckpts && for a in N1DW4_s0 N1DW4_s1 N1DW2_s0 N1DW2_s1; do got=\$(sha256sum \$a/ckpt_last.pt | cut -c1-64); want=\$(grep \" \$a/\" SHA256_PODS.txt | cut -c1-64); [ -n \"\$want\" ] && [ \"\$got\" = \"\$want\" ] || { echo \"sha mismatch \$a\"; exit 1; }; done"
# ---- 2 T1 dumps, one process per tag
for A in N1DW4_s0 N1DW4_s1 N1DW2_s0 N1DW2_s1; do
  step "dump141_$A" "eval-141 T1 dump $A (follow)" $PY -u n1dw_dumps.py N1DW-$A
  step "dump141_${A}Z" "eval-141 T1 dump ${A}Z (zero twin)" $PY -u n1dw_dumps.py N1DW-${A}Z
done
# ---- 3 T0 passes
for A in N1DW4_s0 N1DW4_s1 N1DW2_s0 N1DW2_s1; do
  for V in real donor zero; do
    step "t0_${A}_$V" "eval-141 T0 pass $A $V" $PY -u n1dw_t0.py --arms $A --variants $V --out "$N1DW_T0"
  done
done
# ---- 4 STRATEGIC in full, per weight
for W in 4 2; do
  VIEW=$O/view_w$W
  step "view_w$W" "STRATEGIC view + frozen readers for weight $W" bash -c "
    rm -rf \"$VIEW\" && $PY -u n1dw_view.py \"$VIEW\" $W &&
    $PY -u $SR/strat1_strategic.py --t0 \"$VIEW/t0\" --t1 \"$VIEW/t1\" --out \"$O/N1DW${W}_STRATEGIC_view.json\" &&
    $PY -u $SR/strat1_ctx_read.py --t0 \"$VIEW/t0\" --out \"$O/N1DW${W}_CTX_view.json\" &&
    $PY -u $N2R/navrb2_ctx_nofog.py --t0 \"$VIEW/t0\" --out \"$O/N1DW${W}_CTX_nofog.json\" &&
    rm -rf \"$VIEW\""
done
step contrast "ROUTE-COST contrast N1DW{2,4} - N1R" $PY -u n1dw_strategic.py --t0 "$N1DW_T0" --out "$O/N1DW_STRATEGIC_contrast.json"
# ---- 5 the guard read
step read141 "the four families on eval-141 (guard)" $PY -u n1dw_read141.py --out "$O/RESULT_N1DW_EVAL141.json" --contrast "$O/N1DW_STRATEGIC_contrast.json"
echo "DONE" > "$O/N1DW141.done"
echo "ZZ N1DW141 DONE ZZ" | tee -a "$LOG"
