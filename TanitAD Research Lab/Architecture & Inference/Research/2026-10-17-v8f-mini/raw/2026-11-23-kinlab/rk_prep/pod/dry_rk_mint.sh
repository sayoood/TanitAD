#!/bin/bash
# dry_rk_mint.sh -- the BUILD AGENT's CPU dry mint of the RK kit (KINLAB), run ON THE POD from /root/rollhist/kit_rk/. Writes ONLY under /root/rollhist/dry_rk/ (the scratch root: $ROOT = /root/rollhist/dry_rk/root) and the results folder
# /root/rollhist/kit_rk/dry_results/. Touches no GPU, no lock (/root/gpu_queue.lock), no marker (/root/N1DW_READY), no process it did not start. Everything runs at nice 19, OMP/MKL 2 threads, CUDA hidden.
# Guards: the container's total_rss (cgroup, never free) must be < 40 GB before it starts and stays under that; the selfcheck child is killed BY PID (never by pattern) if its RSS passes DRY_MAX_RSS_GB (default 12), or it runs longer
# than DRY_MAX_MIN minutes (default 30), or the container total_rss reaches 40 GB; the run is then reported, not retried.
# What it does (the scratch root is a mini /root/rollhist: kit copies, copies of rh_env.sh / rh_evidence.py / rh_selfcheck.py / blobcheck.py / PILOT_RH.json / the checker's merge module, and READ-ONLY LINKS to tree_ra, all_blobs_ra.txt and bank/):
#   1 tree prep       rk_tree_prep.sh with RH_ROOT=$ROOT: tree_rk = tree_ra + rk_delta.tar, blob by blob against all_blobs_rk.txt (the REAL tree delta)
#   2 render          rk_argv.py render with RH_ROOT=$ROOT on RL's argv with /root/rollhist rewritten to $ROOT (bank and --out only): RK_s0 / RK_s1 argv, the real banks (links), the real init checkpoints
#   3 per seed        verify, the stock pair check (argv minus the label tokens vs B0), the trainer's --selfcheck-only of the RK argv on CPU (the ONLY difference from the launch argv: --device cpu; the dry argv is argv_cpu.json),
#                     rk_gate_record.py mint + verify, and the study-side mint in its DRY form (chk_arm_rk.sh with DRY_GD / DRY_SC): checks.json
#   The G-SUITE has NOT run at 0ca6f49d. For the evidence step only, the scratch root holds a PLANTED gsuite_result_rk.json (RA's result with the commit set to 0ca6f49d and "PLANTED_FOR_DRY_MINT_ONLY": true); it lives in the scratch root,
#   is deleted with it, and its G-SUITE check stands for the real G-SUITE that the chain runs before the lock. The scratch tree marker done_rk/tree.done is likewise planted. Everything else is real.
# Last line of $DR/dry.log: ZZ RK DRY MINT DONE ZZ or ZZ RK DRY MINT STOP ZZ <why>.
set -u
K=/root/rollhist/kit_rk; DR=/root/rollhist/dry_rk; ROOT=$DR/root; RES=$K/dry_results
PY=/root/venv/bin/python
LOG=$DR/dry.log
MAXMIN=${DRY_MAX_MIN:-30}; MAXRSS_KB=$(( ${DRY_MAX_RSS_GB:-12} * 1048576 ))
rss() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{printf "%d", $2}'; }
[ ! -e $DR ] || { echo "$DR exists; refusing (delete it by literal path first)"; exit 1; }
mkdir -p $DR $RES
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
stop() { say "ZZ RK DRY MINT STOP ZZ $1"; exit 1; }
R0=$(rss); [ "${R0:-99999999999}" -lt 42949672960 ] || stop "container total_rss $R0 B >= 40 GB before the start"
say "start; container total_rss $((R0/1048576)) MB; caps: ${MAXMIN} min, $((MAXRSS_KB/1048576)) GB per selfcheck"
mkdir -p $ROOT/runs $ROOT/study $ROOT/pilot $ROOT/done_rk $ROOT/logs
# ---- stage the scratch root (copies of kit files, copies of three pod files, links to read-only inputs)
for f in rk_env.sh rk_tree_prep.sh rk_gsuite.sh rk_gate_record.py rh_rk_chain.sh rk_arm.sh chk_arm_rk.sh rk_argv.py rh_rk_marker_watch.sh launch_rk.sh all_blobs_rk.txt rk_delta.tar; do cp $K/$f $ROOT/$f || stop "kit file $f missing in $K"; done
cp $K/dscale10k_check_rk.py $ROOT/study/
for f in rh_env.sh rh_evidence.py rh_selfcheck.py blobcheck.py; do cp /root/rollhist/$f $ROOT/$f || stop "no /root/rollhist/$f"; done
cp /root/rollhist/study/dscale10k_merge.py $ROOT/study/; cp /root/rollhist/pilot/PILOT_RH.json $ROOT/pilot/
ln -s /root/rollhist/tree_ra $ROOT/tree_ra; ln -s /root/rollhist/all_blobs_ra.txt $ROOT/all_blobs_ra.txt; ln -s /root/rollhist/bank $ROOT/bank
for s in 0 1; do
  mkdir -p $ROOT/runs/RL_s$s $ROOT/runs/B0_s$s
  sed "s#/root/rollhist#$ROOT#g" /root/rollhist/runs/RL_s$s/argv.json > $ROOT/runs/RL_s$s/argv.json
  cp /root/rollhist/runs/B0_s$s/argv.json $ROOT/runs/B0_s$s/argv.json
done
$PY - <<'PYE'
import json
g = json.load(open("/root/rollhist/gsuite_result_ra.json"))
g["new"] = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"
if "new_commit" in g:
    g["new_commit"] = g["new"]
g["PLANTED_FOR_DRY_MINT_ONLY"] = True
json.dump(g, open("/root/rollhist/dry_rk/root/gsuite_result_rk.json", "w"))
PYE
date -u +%FT%TZ > $ROOT/done_rk/tree.done
export RH_ROOT=$ROOT RH_READOUT=causal_g1x HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
# ---- 1 tree prep
say "step tree prep"; nice -n 19 bash $ROOT/rk_tree_prep.sh > $DR/tree_prep.log 2>&1; tail -n 1 $DR/tree_prep.log | grep -q "ZZ RK TREE OK ZZ" || stop "tree prep: $(tail -c 300 $DR/tree_prep.log | tr '\n' ' ')"
say "tree prep OK: $(grep -c . $DR/tree_prep.log) lines: $(head -n 2 $DR/tree_prep.log | tr '\n' ' ')"
. $ROOT/rk_env.sh
# ---- 2 render
say "step render"; $PY $ROOT/rk_argv.py render > $DR/render.log 2>&1 || stop "render: $(tail -c 300 $DR/render.log | tr '\n' ' ')"
say "render: $(tail -n 3 $DR/render.log | tr '\n' ' ')"
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE"
for s in 0 1; do
  tag=RK_s$s; D=$ROOT/runs/$tag
  $PY $ROOT/rk_argv.py verify --seeds $s > $D/argv_verify.log 2>&1 || stop "argv verify $tag: $(tail -c 300 $D/argv_verify.log | tr '\n' ' ')"
  sha=$($PY -c "import hashlib;print(hashlib.sha256(open('$ROOT/bank/s$s/bank.json','rb').read()).hexdigest())")
  $PY - $D/argv.json $D/argv_nolabel.json $D/argv_cpu.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
i = t.index("--dagger-label")
assert t[i + 1] == "kin_v1" and i + 2 == len(t)
json.dump({"argv": t[:i]}, open(sys.argv[2], "w"), indent=1)
c = list(t)
j = c.index("--device")
assert c[j + 1] == "cuda"
c[j + 1] = "cpu"                                       # the dry form's ONLY difference from the launch argv
json.dump({"argv": c}, open(sys.argv[3], "w"), indent=1)
PYE
  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv_nolabel.json --b0 $ROOT/runs/B0_s$s/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || stop "pair check $tag: $(tail -c 300 $D/pair_$tag.log | tr '\n' ' ')"
  say "$tag pair check: $(tail -n 1 $D/pair_$tag.log)"
  # ---- the selfcheck-only of the RK argv on CPU, guarded
  say "$tag selfcheck start (CPU, nice 19)"
  ( RH_TREE=$TREE exec nice -n 19 $PY $ROOT/rh_selfcheck.py $D/argv_cpu.json $D/selfcheck $D/selfcheck.result.json > $D/selfcheck.log 2>&1 ) &
  P=$!; t0=$(date +%s); last=0
  while kill -0 $P 2>/dev/null; do
    sleep 10
    el=$(( $(date +%s) - t0 )); pr=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); cr=$(rss)
    if [ $((el - last)) -ge 120 ]; then last=$el; say "$tag selfcheck ${el}s rss $((${pr:-0}/1024)) MB container $((cr/1048576)) MB: $(tail -n 1 $D/selfcheck.log | cut -c1-160)"; fi
    if [ "${pr:-0}" -gt $MAXRSS_KB ] || [ $el -gt $((MAXMIN*60)) ] || [ "${cr:-0}" -ge 42949672960 ]; then
      kill -TERM $P 2>/dev/null; sleep 5; kill -KILL $P 2>/dev/null
      cp $D/selfcheck.log $RES/selfcheck_${tag}_STOPPED.log 2>/dev/null
      stop "$tag selfcheck stopped by the guard at ${el}s (rss ${pr:-0} KB, container ${cr} B); last log line: $(tail -n 1 $D/selfcheck.log | cut -c1-200)"
    fi
  done
  wait $P; rc=$?
  say "$tag selfcheck rc=$rc after $(( $(date +%s) - t0 )) s: $(tail -n 1 $D/selfcheck.log | cut -c1-300)"
  [ -s $D/selfcheck/selfcheck.json ] || { cp $D/selfcheck.log $RES/selfcheck_${tag}_FAILED.log; stop "$tag selfcheck wrote no selfcheck.json"; }
  # ---- gate record (real selfcheck, planted G-SUITE and tree marker) and the study-side DRY mint
  $PY $ROOT/rk_gate_record.py mint --tag $tag --readout causal_g1x > $D/mint.log 2>&1; mrc=$?
  $PY $ROOT/rk_gate_record.py verify --tag $tag > $D/verify.log 2>&1; vrc=$?
  say "$tag gate record mint rc=$mrc verify rc=$vrc: $(grep -E 'census|ZZ' $D/mint.log $D/verify.log | tr '\n' ' ')"
  DRY_GD=$ROOT/dry_gate_s$s DRY_SC=$D/selfcheck/selfcheck.json nice -n 19 bash $ROOT/chk_arm_rk.sh mint $tag > $DR/mint_s$s.log 2>&1; crc=$?
  say "$tag study-side dry mint rc=$crc: $(tail -n 2 $DR/mint_s$s.log | cut -c1-300 | tr '\n' ' ')"
  # ---- results out (small files only)
  mkdir -p $RES/$tag
  cp $D/selfcheck/selfcheck.json $D/selfcheck.result.json $D/selfcheck.log $D/pair_$tag.log $D/mint.log $D/verify.log $D/argv.json $D/argv_cpu.json $D/argv_verify.log $RES/$tag/ 2>/dev/null
  cp $D/gate/RH_GATE_RECORD.json $D/gate/RH_GATE_REFUSED.json $RES/$tag/ 2>/dev/null
  cp $ROOT/dry_gate_s$s/checks.json $ROOT/dry_gate_s$s/evidence.log $RES/$tag/ 2>/dev/null
  cp $DR/mint_s$s.log $RES/$tag/mint_token_dry.log
done
cp $DR/tree_prep.log $DR/render.log $DR/dry.log $RES/ 2>/dev/null
say "ZZ RK DRY MINT DONE ZZ"
cp $DR/dry.log $RES/dry.log
