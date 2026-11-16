#!/bin/bash
# gate_rerun_mm.sh -- MM 2026-11-17: re-run the items of the DSCALE10K block-1 gate that failed on 2026-11-16 (FINISH-GATE FAIL at assemble_evidence), after two repairs:
#  (1) both stage-B selfchecks were REFUSED at K-B3: the canonical eval-141 grid is not in the study tree (only stack/ and taniteval/ were exported).  It is copied from
#      /root/strat1/new4 and checked against sha256 659c3b65 (the a4357e79 blob; the file last changed in 4658d5c3, 2026-10-17).  It lies outside TREE_DIRS, so the tree
#      manifest the tokens bind does not move.
#  (2) resume_smoke.py: the bitwise bar replaced by a floor-relative bar with an optimizer-state defect arm (docstring; the original stays as resume_smoke.py.pre_mm).
# Then finish_gate.sh as written (B_rep selfcheck, stageb items, evidence, the stage-A mints of B_s0 / B_rep / U_s0, the dry stage-B mint) and summary_report.py.
# One GPU job at a time: every GPU step waits for < 3,500 MiB used.  Sentinel (last line of gate/gate_rerun_mm.log): ZZ GATE-RERUN DONE|FAIL ZZ.
set -u
D=/root/dscale10k; G=$D/gate; TREE=$D/tree; PY=/root/venv/bin/python
GRID_REL="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
fail() { echo "ZZ GATE-RERUN FAIL ZZ $1"; exit 1; }
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${u:-99999}" -lt 3500 ]; }
wait_gpu() { n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before $1"; sleep 10; done; }
echo "$(date -u +%FT%TZ) gate re-run start"
# repairs
mkdir -p "$TREE/$(dirname "$GRID_REL")" || fail "grid dir"
cp "/root/strat1/new4/$GRID_REL" "$TREE/$GRID_REL" || fail "copy the grid"
echo "659c3b65e364ea5de87a4e17d076eba5645df95cdcf39d8abc08caab1b2d594c  $TREE/$GRID_REL" | sha256sum -c || fail "grid sha256"
[ -f $G/resume_smoke.py.pre_mm ] || cp $G/resume_smoke.py $G/resume_smoke.py.pre_mm || fail "keep the original resume_smoke.py"
echo "87300cc77104238fa97b4206b6c7ec7161b85dc99786c07a903f206a700cca2f  $G/resume_smoke.py.pre_mm" | sha256sum -c || fail "pre_mm is not the original"
cp $G/mmfix/resume_smoke.py $G/resume_smoke.py || fail "install resume_smoke.py"
echo "b2d76454ff064daf4859c93b328aa598f2119e508458b84255c0d502b1a93d18  $G/resume_smoke.py" | sha256sum -c || fail "resume_smoke.py sha256"
for f in finish_gate.log resume.json resume.log stageb.json sb/B_s0.B.result.json sb/B_s0.B.mut.json sb/U_s0.B.result.json GATE_RESULTS_BLOCK1.md; do
  [ -f $G/$f ] && cp $G/$f $G/$f.fail1
done
# stage-B selfchecks (gpu_gate.sh steps 2, 3 and the U half of 4; the banks are the ones built on 2026-11-16)
wait_gpu "stage-B selfcheck B_s0"
$PY $G/sb_selfcheck.py $G/sb/B_s0.B.argv.json $G/sb/B_s0.B $G/sb/B_s0.B.result.json > $G/sb/B_s0.B.log 2>&1; echo "$(date -u +%T) B_s0.B selfcheck: $(tail -n1 $G/sb/B_s0.B.log | cut -c1-300)"
wait_gpu "mutated selfcheck"
$PY $G/sb_selfcheck.py $G/sb/B_s0.B.argv.json $G/sb/B_s0.B.mut $G/sb/B_s0.B.mut.json --mutate-apply > $G/sb/B_s0.B.mut.log 2>&1; echo "$(date -u +%T) mutated selfcheck: $(tail -n1 $G/sb/B_s0.B.mut.log | cut -c1-300)"
wait_gpu "U stage-B selfcheck"
$PY $G/sb_selfcheck.py $G/sb/U_s0.B.argv.json $G/sb/U_s0.B $G/sb/U_s0.B.result.json --max-clips 200 > $G/sb/U_s0.B.log 2>&1; echo "$(date -u +%T) U_s0.B smoke selfcheck: $(tail -n1 $G/sb/U_s0.B.log | cut -c1-300)"
# resume smoke (gpu_gate.sh step 5)
wait_gpu "resume smoke"
$PY $G/resume_smoke.py > $G/resume.log 2>&1 || fail "resume smoke: $(tail -c 300 $G/resume.log)"
echo "$(date -u +%T) $(tail -n1 $G/resume.log | cut -c1-400)"
# finish gate, as written
bash $G/finish_gate.sh > $G/finish_gate.log 2>&1
echo "$(date -u +%T) finish gate: $(tail -n 1 $G/finish_gate.log | cut -c1-400)"
$PY $G/summary_report.py > $G/summary_report.log 2>&1; echo "summary rc=$?"
tail -n 1 $G/finish_gate.log | grep -q "ZZ FINISH-GATE DONE ZZ" || fail "finish gate: $(tail -n 1 $G/finish_gate.log | cut -c1-300)"
echo "ZZ GATE-RERUN DONE ZZ"
