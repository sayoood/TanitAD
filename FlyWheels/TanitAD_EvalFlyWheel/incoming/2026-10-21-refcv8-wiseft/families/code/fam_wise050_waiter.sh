#!/bin/bash
# Four-family PAIRED reads for the theta_0.5 navtest roll (R8-WISEFT stage 2), BESIDE the registered PDMS rule -- post-hoc, never a bar.
# Waits for the stage-2 waiter's ROLLED marker (roll + post-gate done), pulls the theta_0.5 seams (md5 both ends), runs families7 per arm
# and paired_families_navsim.py for: theta05 - EMA48852 (s0, s1), theta05 - PRIOR (s0), theta05 s1 - s0 (inference-seed floor).  ZZFW<STAGE>ZZ.
set -u
F=/c/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest; LOG=$F/fam_wise050.log; S2=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/wiseft_stage2.log
B=/home/nvidia/r8ns/navtest_read/wiseft_l0p50/bridge_navtest; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export OMP_NUM_THREADS=4 PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"
mark() { echo "ZZFW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
until grep -q 'ZZWS2ROLLEDZZ' "$S2" 2>/dev/null; do
  grep -qE 'ZZWS2(REFUSED|TIMEOUT|S1FAIL|SHIPBAD|NOSTAGE2)ZZ' "$S2" 2>/dev/null && { mark UPSTREAM "$(tail -1 $S2)"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT; exit 1; }
  sleep 60
done
mark ROLLED
mkdir -p $F/seams_w050 $F/out
ssh -n tanitad-thor "cd $B && tar -cf - seam_R8_LEGAL.npz seam_R8_LEGAL_s1.npz seam_PRIOR_ha0p.npz seam_R8_LEGAL.manifest.json seam_R8_LEGAL_s1.manifest.json" | tar -xf - -C $F/seams_w050
r=$(ssh -n tanitad-thor "cd $B && md5sum seam_R8_LEGAL.npz seam_R8_LEGAL_s1.npz seam_PRIOR_ha0p.npz" | awk '{print $1}' | tr '\n' ' ')
l=$(cd $F/seams_w050 && md5sum seam_R8_LEGAL.npz seam_R8_LEGAL_s1.npz seam_PRIOR_ha0p.npz | awk '{print $1}' | tr '\n' ' ')
[ ${#r} -ge 99 ] && [ "$r" = "$l" ] || { mark MD5BAD "remote=$r local=$l"; exit 2; }
mark PULLED "$l"
cd $F
for A in R8_LEGAL R8_LEGAL_s1; do
  $PY -X utf8 code/families7.py --seam seams_w050/seam_$A.npz --inputs inputs/navtest_inputs.json.gz --stage 1 \
      --label "refcv8 WiSE theta_0.5 navtest $A (R8-WISEFT stage 2; families beside the registered PDMS rule, never a bar)" \
      --out out/families_navtest_w050_$A.json > out/families_w050_$A.log 2>&1
  [ -s out/families_navtest_w050_$A.json ] && mark FAM_OK "$A" || mark FAM_BAD "$A"
done
pf() {  # $1 seam-a $2 name-a $3 seam-b $4 name-b $5 out-stem
  $PY -X utf8 code/paired_families_navsim.py --inputs inputs/navtest_inputs.json.gz --seam-a $1 --name-a $2 --seam-b $3 --name-b $4 \
      --label "navtest paired families: $4 minus $2 (R8-WISEFT stage 2 beside; post-hoc, never a bar)" --out out/$5.json > out/$5.log 2>&1
  [ -s out/$5.json ] && mark PF_OK "$5 $(grep -o 'QQPF.*' out/$5.log | cut -c1-200)" || mark PF_BAD "$5"
}
pf seams/seam_R8_LEGAL.npz EMA48852_s0 seams_w050/seam_R8_LEGAL.npz W050_s0 paired_navtest_W050_minus_EMA48852_s0
pf seams/seam_R8_LEGAL_s1.npz EMA48852_s1 seams_w050/seam_R8_LEGAL_s1.npz W050_s1 paired_navtest_W050_minus_EMA48852_s1
pf seams/seam_PRIOR_ha0p.npz PRIOR_ha0p seams_w050/seam_R8_LEGAL.npz W050_s0 paired_navtest_W050_minus_PRIOR_s0
pf seams_w050/seam_R8_LEGAL.npz W050_s0 seams_w050/seam_R8_LEGAL_s1.npz W050_s1 paired_navtest_W050_s1_minus_s0
mark END
