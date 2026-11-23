#!/bin/bash
# s3_dry_mint.sh -- the BUILD AGENT's CPU dry mint of B0_s2 (SEED3), run ON THE POD from /root/rollhist/kit_s3/. Writes ONLY under /root/rollhist/kit_s3/dry/ (results; the scratch root $ROOT = kit_s3/dry/root, deleted at the end by literal path).
# Touches no GPU, no lock (/root/gpu_queue.lock), no marker (/root/N1DW_READY), no process it did not start. nice 19, OMP/MKL 2 threads, CUDA hidden, ONE selfcheck. Guard: container total_rss (cgroup, never free) < 40 GB before the start, 90 min cap.
# Steps (REAL tree_cdb 97fe0a56, REAL UE_s2.A, REAL s0 argv, scratch root for everything written):
#   A  s3_argv.py render --arms B0_s2 into the scratch root (B0_s2 argv + the RH_s2 pair skeleton); the CPU argv = the launch argv with `--device cpu` the ONLY difference (cpu-argv); the trainer's --selfcheck-only on it (rh_selfcheck.py)
#   B  the stock pair check B0_s2 vs the skeleton; rh_gate_record_s3.py mint + verify (scratch root: the real G-SUITE result at 97fe0a56, the real pilot, a PLANTED tree marker); the study-side DRY mint (chk_arm_s3.sh with DRY_GD / DRY_SC): checks.json
# Last line of $DR/dry.log: ZZ S3 DRY MINT DONE ZZ or ZZ S3 DRY MINT STOP ZZ <why>.
set -u
K=$(cd "$(dirname "$0")" && pwd); DR=$K/dry; ROOT=$DR/root
[ "$K" = /root/rollhist/kit_s3 ] || { echo "run from /root/rollhist/kit_s3 (got $K)"; exit 1; }
PY=/root/venv/bin/python
LOG=$DR/dry.log
rss() { awk '/^total_rss /{printf "%d", $2}' /sys/fs/cgroup/memory/memory.stat; }
[ ! -e $ROOT ] || { echo "$ROOT exists; refusing (delete it by literal path first)"; exit 1; }
mkdir -p $DR
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
stop() { say "ZZ S3 DRY MINT STOP ZZ $1"; exit 1; }
[ "$(rss)" -lt 42949672960 ] || stop "container total_rss >= 40 GB"
[ -d /root/rollhist/tree_cdb ] || stop "no tree_cdb"
say "start; container total_rss $(( $(rss) / 1048576 )) MB"
mkdir -p $ROOT/runs/B0_s0 $ROOT/runs/RH_s0 $ROOT/study $ROOT/pilot $ROOT/done_s3 $ROOT/logs
for f in s3_env.sh s3_argv.py chk_arm_s3.sh rh_gate_record_s3.py dscale10k_check_s3.py U_s2.B.tmpl.json; do cp $K/$f $ROOT/$f || stop "kit file $f missing"; done
for f in rh_env.sh rh_evidence.py rh_selfcheck.py blobcheck.py; do cp /root/rollhist/$f $ROOT/$f || stop "no /root/rollhist/$f"; done
cp /root/rollhist/study/dscale10k_merge.py $ROOT/study/; cp /root/rollhist/pilot/PILOT_RH.json $ROOT/pilot/; cp /root/rollhist/gsuite_result_rh.json $ROOT/; cp /root/rollhist/RH_READOUT.chosen $ROOT/
ln -s /root/rollhist/tree_cdb $ROOT/tree_cdb; ln -s /root/rollhist/tree_ki $ROOT/tree_ki
cp /root/rollhist/runs/B0_s0/argv.json $ROOT/runs/B0_s0/argv.json; cp /root/rollhist/runs/RH_s0/argv.json $ROOT/runs/RH_s0/argv.json      # unmodified: the pins hash them
date -u +%FT%TZ > $ROOT/done_s3/tree_rh.done          # PLANTED (the scratch root's own; the chain writes the real one under the lock)
export RH_ROOT=$ROOT RH_READOUT=causal_g1x HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= S3_OMP=2
. $ROOT/s3_env.sh
s3_arm_env B0_s2 || stop "env"
[ "$TREE" = $ROOT/tree_cdb ] || stop "TREE is $TREE"
$PY $ROOT/s3_argv.py render --arms B0_s2 > $DR/render.log 2>&1 || stop "render: $(tail -c 300 $DR/render.log | tr '\n' ' ')"
say "render: $(tail -n 2 $DR/render.log | tr '\n' ' ' | cut -c1-300)"
D=$ROOT/runs/B0_s2
$PY $ROOT/s3_argv.py verify --arms B0_s2 > $D/argv_verify.log 2>&1 || stop "argv verify: $(tail -c 300 $D/argv_verify.log | tr '\n' ' ')"
$PY $ROOT/s3_argv.py cpu-argv --arm B0_s2 --out-dir $DR/cpu > $DR/cpuargv.log 2>&1 || stop "cpu-argv: $(tail -c 300 $DR/cpuargv.log | tr '\n' ' ')"
say "cpu argv: $(head -n 1 $DR/cpuargv.log | cut -c1-300)"
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE"
$PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/pair_rh_skeleton.json --b0 $D/argv.json > $D/pair_B0_s2.log 2>&1 || stop "pair check: $(tail -c 300 $D/pair_B0_s2.log | tr '\n' ' ')"
say "pair check: $(tail -n 1 $D/pair_B0_s2.log)"
say "selfcheck start (CPU, nice 19, cap 90 min); container total_rss $(( $(rss) / 1048576 )) MB"
t0=$(date +%s)
RH_TREE=$TREE timeout 5400 nice -n 19 $PY $ROOT/rh_selfcheck.py $DR/cpu/B0_s2_argv_cpu.json $D/selfcheck $D/selfcheck.result.json > $DR/selfcheck.log 2>&1; rc=$?
say "selfcheck rc=$rc after $(( $(date +%s) - t0 )) s: $(tail -n 1 $DR/selfcheck.log | cut -c1-300)"
[ -s $D/selfcheck/selfcheck.json ] || stop "selfcheck wrote no selfcheck.json"
$PY $ROOT/rh_gate_record_s3.py mint --tag B0_s2 --readout causal_g1x > $D/mint.log 2>&1; mrc=$?
$PY $ROOT/rh_gate_record_s3.py verify --tag B0_s2 > $D/verify.log 2>&1; vrc=$?
say "gate record mint rc=$mrc verify rc=$vrc: $(grep -E 'ZZ|PROBLEM' $D/mint.log $D/verify.log | tr '\n' ' ' | cut -c1-500)"
DRY_GD=$ROOT/dry_gate_s2 DRY_SC=$D/selfcheck/selfcheck.json nice -n 19 bash $ROOT/chk_arm_s3.sh mint B0_s2 > $DR/mint_token_dry.log 2>&1; crc=$?
say "study-side dry mint rc=$crc: $(tail -n 2 $DR/mint_token_dry.log | cut -c1-300 | tr '\n' ' ')"
mkdir -p $DR/B0_s2
cp $D/selfcheck.log $D/selfcheck.result.json $D/pair_B0_s2.log $D/mint.log $D/verify.log $D/argv_verify.log $D/argv.json $D/pair_rh_skeleton.json $DR/B0_s2/ 2>/dev/null
cp $D/selfcheck/selfcheck.json $DR/B0_s2/selfcheck_cpu.json 2>/dev/null
cp $D/gate/RH_GATE_RECORD.json $D/gate/RH_GATE_REFUSED.json $DR/B0_s2/ 2>/dev/null
cp $ROOT/dry_gate_s2/checks.json $ROOT/dry_gate_s2/evidence.log $DR/B0_s2/ 2>/dev/null
rm -f $ROOT/tree_cdb $ROOT/tree_ki; rm -rf "${ROOT:?}"; mkdir -p $K/prep_logs; echo "$(date -u +%FT%TZ) $ROOT (dry-mint scratch root: argv copies, selfcheck dir, gate dir) deleted by literal path; results kept in $DR" >> $K/prep_logs/deletions.log
[ $mrc = 0 ] && [ $vrc = 0 ] && [ $crc = 0 ] || stop "a step failed (mint $mrc verify $vrc study-side $crc)"
say "ZZ S3 DRY MINT DONE ZZ"
