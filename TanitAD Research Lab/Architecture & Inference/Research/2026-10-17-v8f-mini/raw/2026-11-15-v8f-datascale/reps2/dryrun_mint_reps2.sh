#!/bin/bash
# dryrun_mint_reps2.sh -- DRY mint + verify of the patched checker for UE_s2 and UE_s3 (pod, CPU only, one python process at a time, nice 19). Writes ONLY under /root/dscale10k/dry_reps2/mintroot/.
# It never starts a trainer (no `run`: the runner would start training if its token were accepted; verify_token_reps2.py applies the same token_problems() with the runner's exact arguments instead),
# never opens the GPU (CUDA_VISIBLE_DEVICES=-1), never takes a lock or a marker, never modifies a file outside mintroot. Read-only inputs: the real merged/, in/, study/ (symlinked INTO mintroot), the real tree,
# the real argv B_s0 / UE_s0 (copied, with --out of the UE_s0 copy rewritten under mintroot so the checker's --out derivation is exercised against a root of its own), U_s0's dry record (symlink).
# STAND-IN EVIDENCE (labelled DRY, void as a token): UE_s1's dry record and UE_s1's evidence file are copied as UE_s2's / UE_s3's, because no seed-2 / seed-3 dry build, init record or reach probe exists
# (they need the GPU lock and ~20 CPU minutes: they are the chain's prep). So this run exercises: C-ARGV with the reference, the binding of the reference file, the mint, token_problems and 8 tampered copies.
# Output: mintroot/gate/tokens/<arm>.A/{checks.json,gate_token.json}, mintroot/gate/verify_<arm>.A.json, mintroot/mint_<arm>.log; the last line `ZZ DRY-MINT-REPS2 DONE ZZ` or `... FAIL ZZ <why>`.
R=/root/dscale10k; X=$R/dry_reps2; M=$X/mintroot; B=$X/blockreps2; PY=/root/venv/bin/python
fail() { echo "ZZ DRY-MINT-REPS2 FAIL ZZ $1"; exit 1; }
[ -d $B ] || fail "no $B"
[ -e $M ] && fail "$M exists (never reused)"
mkdir -p $M/argv $M/dry $M/gate/tokens $M/tmp $M/runs $M/blockreps2 || fail mkdir
ln -s $R/merged $M/merged; ln -s $R/in $M/in; ln -s $R/study $M/study
cp $R/argv/B_s0.A.json $M/argv/B_s0.A.json
sed "s#/root/dscale10k/runs/UE_s0/A#$M/runs/UE_s0/A#" $R/argv/UE_s0.A.json > $M/argv/UE_s0.A.json
grep -q "$M/runs/UE_s0/A" $M/argv/UE_s0.A.json || fail "reference --out not rewritten"
ln -s $R/dry/U_s0.A.dry.json $M/dry/U_s0.A.dry.json
for a in UE_s2 UE_s3; do cp $R/dry/UE_s1.A.dry.json $M/dry/$a.A.dry.json; cp $R/gate/evidence_UE_s1.json $M/gate/evidence_$a.json; done
echo "DRY: stand-in evidence copied from UE_s1 (dry record and evidence); tokens minted here are VOID" > $M/DRY_STANDIN.txt
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
DS_ROOT=$M DS_REF=$M/argv/UE_s0.A.json nice -n 19 $PY $B/gen_argv_reps2.py > $M/gen_argv.log 2>&1 || fail "gen_argv: $(tail -c 300 $M/gen_argv.log)"
tail -n 1 $M/gen_argv.log
for arm in UE_s2 UE_s3; do
  DS_ROOT=$M DS_RP=$B DS_PY=$PY nice -n 19 bash $B/chk_arm_reps2.sh mint $arm A > $M/mint_$arm.log 2>&1; rc=$?
  echo "mint $arm rc=$rc: $(tail -n 1 $M/mint_$arm.log | cut -c1-300)"
  [ $rc -eq 0 ] || fail "mint $arm refused (see $M/mint_$arm.log and $M/gate/tokens/$arm.A/checks.json)"
  DS_ROOT=$M DS_RP=$B DS_GD=$M/gate/tokens/$arm.A DS_VERIFY_OUT=$M/gate/verify_$arm.A.json nice -n 19 $PY $B/verify_token_reps2.py $arm A > $M/verify_$arm.log 2>&1; rc=$?
  echo "verify $arm rc=$rc: $(tail -n 1 $M/verify_$arm.log | cut -c1-300)"
  [ $rc -eq 0 ] || fail "verify $arm refused (see $M/verify_$arm.log)"
done
echo "ZZ DRY-MINT-REPS2 DONE ZZ"
