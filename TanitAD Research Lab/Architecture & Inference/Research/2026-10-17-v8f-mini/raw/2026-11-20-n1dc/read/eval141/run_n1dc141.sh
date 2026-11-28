#!/bin/bash
# run_n1dc141.sh -- STAGED, NOT STARTED.  ONE detached dev-box chain: the eval-141 guard of the N1DC card (PREREG_V8F_N1DC.md sha256 2eae7964, section 3) and the STRATEGIC reads.
# Adapted from the N1DW chain run_eval141.sh (ran 2026-11-20).  CPU only, 2 threads, ONE process at a time, every step gated on dev-box free commit >= 5.5 GB (wait_commit.py).
# Re-runnable: a finished step leaves steps/<name>.ok and is skipped.  Launch only through run_n1dc141.cmd by WMI (CreateFlags 0x08000200, stdin NUL, Git usr\bin\bash.exe), when the MM says go.
#   0  optional wait for a marker file (env N1DC_WAIT_MARK; unset = no wait), cap 24 h.  Refuses unless K-HB0-141.json says pass (the wrapper smoke).
#   1  checkpoints' sha256 equal the pod's own sha256sum (ckpts/SHA256_PODS.txt); banked N1D inputs present (banked_n1d/x, from the HF tar sha256 143f3b2a)
#   2  eval-141 T1 dumps under the executor of record hb0.2: N1DC_s{0,1} and N1D_s{0,1}, follow + zero twin (n1dc_dumps.py, N1DC_KAPPA=0.2)  -> dumps141_hb02
#   3  eval-141 T1 dumps under the old executor: N1DC_s{0,1} follow (N1DC_KAPPA=none)                                                        -> dumps141_old
#   4  T0 passes real / donor / zero for N1DC_s{0,1} (n1dc_t0.py; the N1D T0 passes are banked)                                            -> out/t0
#   5  STRATEGIC: N1-slot view + frozen strat1_strategic / strat1_ctx_read / navrb2_ctx_nofog; S1-slot views (n1dw_view_s1slot.py, landed 94636bf7) for N1DC and the N1D reference; contrast N1DC-N1D / N1DC-N1R
#   6  n1dc_read141.py: the four families, guard, Z-EVAL, K1, K-RE, K-HB0-141
# Last line of the log: "ZZ N1DC141 DONE ZZ" or "ZZ N1DC141 FAIL ZZ <reason>".  Waiters match that sentinel.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
E=F:/TanitAD-scratch/n1dc/read/eval141
SR=F:/TanitAD-scratch/strat1/read
N2R=F:/TanitAD-scratch/strat1/navrb2/read
PY=${N1DC_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
MARK=${N1DC_WAIT_MARK:-}
O=$E/out
STEPS=$O/steps
LOG=$O/n1dc141_chain.log
export N1DC_TREE=$E/tree N1DC_CKPTS=$E/ckpts N1DC_T0=$O/t0 TMP=$E/tmp TEMP=$E/tmp
HB=$E/dumps141_hb02
BN=$E/banked_n1d/x
DEL=F:/TanitAD-scratch/n1dc/DELETIONS.log
cd "$E" || exit 1
mkdir -p "$O" "$STEPS" "$E/tmp"
fail() { echo "ZZ N1DC141 FAIL ZZ $1" >> "$LOG"; echo "FAIL $1" > "$O/N1DC141.done"; exit 1; }
gate() { $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"; }
# step NAME DESCRIPTION cmd...: skipped when steps/NAME.ok exists; every step is gated; on failure the chain stops with the step named
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n (done)" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[n1dc141 chain] start $(date) tree $N1DC_TREE (bcf3ba3f + hb wrapper bcf267ce)" >> "$LOG"

# ---- 0 optional wait, then the smoke must have passed
if [ -n "$MARK" ]; then
  OK0=0
  for i in $(seq 1 1440); do
    if [ -f "$MARK" ]; then OK0=1; echo "wait mark present: $(date)" >> "$LOG"; break; fi
    sleep 60
  done
  [ "$OK0" = 1 ] || fail "gate0: $MARK never appeared within 24 h"
fi
$PY -c "import json,sys; d=json.load(open('$E/K-HB0-141.json')); sys.exit(0 if d.get('pass') is True else 1)" >> "$LOG" 2>&1 || fail "gate0: K-HB0-141.json missing or not pass (run the smoke first)"
# ---- 1 checkpoints and banked inputs
step ckpt_sha "checkpoint sha256 equal to the pod's" bash -c "cd $E/ckpts && sha256sum -c SHA256_PODS.txt"
step banked_n1d "banked N1D eval-141 dumps and T0 passes present" $PY -u check_banked_n1d.py "$BN"
# ---- 2 hb0.2 T1 dumps, one process per tag
for A in N1DC_s0 N1DC_s1 N1D_s0 N1D_s1; do
  step "hb_dump_$A" "eval-141 T1 hb0.2 dump $A (follow)" env N1DC_KAPPA=0.2 $PY -u n1dc_dumps.py N1DC-$A
  step "hb_dump_${A}Z" "eval-141 T1 hb0.2 dump ${A}Z (zero twin)" env N1DC_KAPPA=0.2 $PY -u n1dc_dumps.py N1DC-${A}Z
done
# ---- 3 old-executor T1 dumps (N1DC only; the banked old N1D dumps are the comparator)
for A in N1DC_s0 N1DC_s1; do
  step "old_dump_$A" "eval-141 T1 old-executor dump $A" env N1DC_KAPPA=none $PY -u n1dc_dumps.py N1DC-$A
done
# ---- 4 T0 passes
for A in N1DC_s0 N1DC_s1; do
  for V in real donor zero; do
    step "t0_${A}_$V" "eval-141 T0 pass $A $V" $PY -u n1dc_t0.py --arms $A --variants $V --out "$N1DC_T0"
  done
done
# ---- 5 STRATEGIC
VIEW=$O/view_n1
step view_n1 "STRATEGIC N1-slot view + frozen readers for N1DC" bash -c "
  rm -rf \"$VIEW\" && $PY -u n1dc_view.py \"$VIEW\" &&
  $PY -u $SR/strat1_strategic.py --t0 \"$VIEW/t0\" --t1 \"$VIEW/t1\" --out \"$O/N1DC_STRATEGIC_view.json\" &&
  $PY -u $SR/strat1_ctx_read.py --t0 \"$VIEW/t0\" --out \"$O/N1DC_CTX_view.json\" &&
  $PY -u $N2R/navrb2_ctx_nofog.py --t0 \"$VIEW/t0\" --out \"$O/N1DC_CTX_nofog.json\" &&
  rm -rf \"$VIEW\" && echo \"\$(date -u +%FT%TZ) deleted $VIEW (copies; sources in out/t0, dumps141_hb02/ticks, strat1/read)\" >> $DEL"
s1slot() {  # s1slot NAME PREFIX T0DIR TICKSDIR : the arm in the S1 slot of the frozen strat1_strategic.py (n1dw_view_s1slot.py = unedited copy of the landed 94636bf7 file)
  local n=$1 p=$2 t=$3 k=$4
  step "s1slot_$n" "STRATEGIC S1-slot view $n" bash -c "
    rm -rf \"$O/view_s1_$n\" && $PY -u n1dw_view_s1slot.py \"$O/view_s1_$n\" \"$p\" \"$t\" \"$k\" &&
    $PY -u $SR/strat1_strategic.py --t0 \"$O/view_s1_$n/t0\" --t1 \"$O/view_s1_$n/t1\" --out \"$O/${n}_STRATEGIC_S1slot.json\" &&
    rm -rf \"$O/view_s1_$n\" && echo \"\$(date -u +%FT%TZ) deleted $O/view_s1_$n (copies)\" >> $DEL"
}
s1slot N1DC N1DC-N1DC "$N1DC_T0" "$HB/ticks"
s1slot N1D NAVRBD-N1D "$BN/out/t0" "$BN/dumps141/ticks"
step contrast "STRATEGIC contrasts N1DC-N1D (deciding), N1DC-N1R (beside)" $PY -u n1dc_strategic.py --t0 "$N1DC_T0" --n1d-t0 "$BN/out/t0" --out "$O/N1DC_STRATEGIC_contrast.json"
# ---- 6 the guard read
step read141 "the four families on eval-141 (guard)" $PY -u n1dc_read141.py --out "$O/RESULT_N1DC_EVAL141.json" --contrast "$O/N1DC_STRATEGIC_contrast.json"
echo "DONE" > "$O/N1DC141.done"
echo "ZZ N1DC141 DONE ZZ" >> "$LOG"
