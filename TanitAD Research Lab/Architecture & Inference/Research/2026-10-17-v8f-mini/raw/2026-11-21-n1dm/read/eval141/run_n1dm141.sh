#!/bin/bash
# run_n1dm141.sh -- STAGED, NOT STARTED.  ONE detached dev-box chain: the eval-141 guard of the N1DM card (PREREG_V8F_N1DM.md sha256 N1DM-DRAFT, section 3) and the STRATEGIC reads.
# Adapted from the N1DW chain run_eval141.sh (ran 2026-11-20).  CPU only, 2 threads, ONE process at a time, every step gated on dev-box free commit >= 5.5 GB (wait_commit.py).
# Re-runnable: a finished step leaves steps/<name>.ok and is skipped.  Launch only through run_n1dm141.cmd by WMI (CreateFlags 0x08000200, stdin NUL, Git usr\bin\bash.exe), when the MM says go.
#   0  optional wait for a marker file (env N1DM_WAIT_MARK; unset = no wait), cap 24 h.  Then step khb0 re-runs the wrapper smoke in THIS tree (N1D_s0, --kappa-max none == the banked old-executor dump, array for array) -> K-HB0-141.json.
#   1  checkpoints' sha256 equal the pod's own sha256sum (ckpts/SHA256_PODS.txt); banked N1D inputs present (banked_n1d/x, from the HF tar sha256 143f3b2a)
#   2  eval-141 T1 dumps under the executor of record hb0.2: N1DM_s{0,1} and N1D_s{0,1}, follow + zero twin (n1dm_dumps.py, N1DM_KAPPA=0.2)  -> dumps141_hb02
#   3  eval-141 T1 dumps under the old executor: N1DM_s{0,1} follow (N1DM_KAPPA=none)                                                        -> dumps141_old
#   4  T0 passes real / donor / zero for N1DM_s{0,1} (n1dm_t0.py; the N1D T0 passes are banked)                                            -> out/t0
#   5  STRATEGIC: N1-slot view + frozen strat1_strategic / strat1_ctx_read / navrb2_ctx_nofog; S1-slot views (n1dw_view_s1slot.py, landed 94636bf7) for N1DM and the N1D reference; contrast N1DM-N1D / N1DM-N1R
#   6  n1dm_read141.py: the four families, guard, Z-EVAL, K1, K-RE, K-HB0-141
# Last line of the log: "ZZ N1DM141 DONE ZZ" or "ZZ N1DM141 FAIL ZZ <reason>".  Waiters match that sentinel.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
E=F:/TanitAD-scratch/n1dc/read_n1dm/eval141
SR=F:/TanitAD-scratch/strat1/read
N2R=F:/TanitAD-scratch/strat1/navrb2/read
PY=${N1DM_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
MARK=${N1DM_WAIT_MARK:-}
O=$E/out
STEPS=$O/steps
LOG=$O/n1dm141_chain.log
export N1DM_TREE=F:/TanitAD-scratch/n1dc/read_n1dm/tree N1DM_CKPTS=$E/ckpts N1DM_T0=$O/t0 TMP=$E/tmp TEMP=$E/tmp
CKN1D=${N1DM_CKPTS_N1D:-F:/TanitAD-scratch/n1dc/read/eval141/ckpts}      # N1D_s0/s1 (the N1DH chain pulled them; sha256 = the pod's own)
export N1DM_CKPTS_N1D=$CKN1D
HB=$E/dumps141_hb02
BN=${N1DM_BN:-F:/TanitAD-scratch/n1dc/read/eval141/banked_n1d/x}          # banked N1D old-executor eval-141 dumps + T0 passes (HF navrbd_read_outputs.tar sha256 143f3b2a)
PRIOR_N1D=${N1DM_PRIOR_N1D_HB:-F:/TanitAD-scratch/n1dc/read_n1dh/eval141/dumps141_hb02}   # the N1DH chain's hb0.2 N1D dumps (bcf3ba3f tree): K-N1D-TREE control, optional
DEL=F:/TanitAD-scratch/n1dc/DELETIONS.log
cd "$E" || exit 1
mkdir -p "$O" "$STEPS" "$E/tmp"
fail() { echo "ZZ N1DM141 FAIL ZZ $1" >> "$LOG"; echo "FAIL $1" > "$O/N1DM141.done"; exit 1; }
gate() { $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"; }
# step NAME DESCRIPTION cmd...: skipped when steps/NAME.ok exists; every step is gated; on failure the chain stops with the step named
step() { local n=$1 d=$2; shift 2; if [ -f "$STEPS/$n.ok" ]; then echo "[skip] $n (done)" >> "$LOG"; return 0; fi
  echo "[step] $n: $d $(date)" >> "$LOG"; gate "$n"; "$@" >> "$LOG" 2>&1 || fail "$n ($d): see the log above"; echo ok > "$STEPS/$n.ok"; }
echo "[n1dm141 chain] start $(date) tree $N1DM_TREE (the-N1DM-commit + hb wrapper bcf267ce)" >> "$LOG"

# ---- 0 optional wait, then the smoke must have passed
if [ -n "$MARK" ]; then
  OK0=0
  for i in $(seq 1 1440); do
    if [ -f "$MARK" ]; then OK0=1; echo "wait mark present: $(date)" >> "$LOG"; break; fi
    sleep 60
  done
  [ "$OK0" = 1 ] || fail "gate0: $MARK never appeared within 24 h"
fi
# ---- 1 checkpoints and banked inputs
step ckpt_sha "N1DM checkpoint sha256 equal to the pod's" bash -c "cd $E/ckpts && sha256sum -c SHA256_PODS.txt"
step ckpt_sha_n1d "N1D checkpoint sha256 equal to the pod's" bash -c "cd $CKN1D && sha256sum -c SHA256_PODS.txt"
step banked_n1d "banked N1D eval-141 dumps and T0 passes present" $PY -u check_banked_n1d.py "$BN"
step khb0 "K-HB0-141 in this tree: N1D_s0 through the hb wrapper, --kappa-max none == the banked old-executor dump" bash -c "
  env N1DM_KAPPA=none N1DM_D_141=$E/smoke_khb0 $PY -u n1dm_dumps.py N1DM-N1D_s0 &&
  $PY -u k_hb0_141.py $E/smoke_khb0/dump_N1DM-N1D_s0_m10_causal_heads $BN/dumps141/dump_NAVRBD-N1D_s0_m10_causal_heads $E/K-HB0-141.json &&
  $PY -c \"import json,sys; d=json.load(open('$E/K-HB0-141.json')); sys.exit(0 if d.get('pass') is True else 1)\" &&
  rm -rf $E/smoke_khb0 && echo \"\$(date -u +%FT%TZ) deleted $E/smoke_khb0 (K-HB0 smoke dump; re-creatable by this step)\" >> $DEL"
# ---- 2 hb0.2 T1 dumps, one process per tag
for A in N1DM_s0 N1DM_s1 N1D_s0 N1D_s1; do
  step "hb_dump_$A" "eval-141 T1 hb0.2 dump $A (follow)" env N1DM_KAPPA=0.2 $PY -u n1dm_dumps.py N1DM-$A
  step "hb_dump_${A}Z" "eval-141 T1 hb0.2 dump ${A}Z (zero twin)" env N1DM_KAPPA=0.2 $PY -u n1dm_dumps.py N1DM-${A}Z
done
# ---- 3 old-executor T1 dumps (N1DM only; the banked old N1D dumps are the comparator)
for A in N1DM_s0 N1DM_s1; do
  step "old_dump_$A" "eval-141 T1 old-executor dump $A" env N1DM_KAPPA=none $PY -u n1dm_dumps.py N1DM-$A
done
# ---- 4 T0 passes
for A in N1DM_s0 N1DM_s1; do
  for V in real donor zero; do
    step "t0_${A}_$V" "eval-141 T0 pass $A $V" $PY -u n1dm_t0.py --arms $A --variants $V --out "$N1DM_T0"
  done
done
# ---- 5 STRATEGIC
VIEW=$O/view_n1
step view_n1 "STRATEGIC N1-slot view + frozen readers for N1DM" bash -c "
  rm -rf \"$VIEW\" && $PY -u n1dm_view.py \"$VIEW\" &&
  $PY -u $SR/strat1_strategic.py --t0 \"$VIEW/t0\" --t1 \"$VIEW/t1\" --out \"$O/N1DM_STRATEGIC_view.json\" &&
  $PY -u $SR/strat1_ctx_read.py --t0 \"$VIEW/t0\" --out \"$O/N1DM_CTX_view.json\" &&
  $PY -u $N2R/navrb2_ctx_nofog.py --t0 \"$VIEW/t0\" --out \"$O/N1DM_CTX_nofog.json\" &&
  rm -rf \"$VIEW\" && echo \"\$(date -u +%FT%TZ) deleted $VIEW (copies; sources in out/t0, dumps141_hb02/ticks, strat1/read)\" >> $DEL"
s1slot() {  # s1slot NAME PREFIX T0DIR TICKSDIR : the arm in the S1 slot of the frozen strat1_strategic.py (n1dw_view_s1slot.py = unedited copy of the landed 94636bf7 file)
  local n=$1 p=$2 t=$3 k=$4
  step "s1slot_$n" "STRATEGIC S1-slot view $n" bash -c "
    rm -rf \"$O/view_s1_$n\" && $PY -u n1dw_view_s1slot.py \"$O/view_s1_$n\" \"$p\" \"$t\" \"$k\" &&
    $PY -u $SR/strat1_strategic.py --t0 \"$O/view_s1_$n/t0\" --t1 \"$O/view_s1_$n/t1\" --out \"$O/${n}_STRATEGIC_S1slot.json\" &&
    rm -rf \"$O/view_s1_$n\" && echo \"\$(date -u +%FT%TZ) deleted $O/view_s1_$n (copies)\" >> $DEL"
}
s1slot N1DM N1DM-N1DM "$N1DM_T0" "$HB/ticks"
s1slot N1D NAVRBD-N1D "$BN/out/t0" "$BN/dumps141/ticks"
step contrast "STRATEGIC contrasts N1DM-N1D (deciding), N1DM-N1R (beside)" $PY -u n1dm_strategic.py --t0 "$N1DM_T0" --n1d-t0 "$BN/out/t0" --out "$O/N1DM_STRATEGIC_contrast.json"
# ---- 6 the guard read
step read141 "the four families on eval-141 (guard)" $PY -u n1dm_read141.py --out "$O/RESULT_N1DM_EVAL141.json" --contrast "$O/N1DM_STRATEGIC_contrast.json" --n1d-old "$BN/dumps141" --n1d-prior "$PRIOR_N1D"
echo "DONE" > "$O/N1DM141.done"
echo "ZZ N1DM141 DONE ZZ" >> "$LOG"
