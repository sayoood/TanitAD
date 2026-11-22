#!/bin/bash
# dryrun_reps2.sh -- control-flow dry run of blockreps2.sh and reps2_marker_watch.sh (fake leaf operations, the REAL chainlib.py and dscale_banker.py, a directory as the hub, a FAKE lock file and a FAKE marker file).
# Pod, CPU only: no GPU, no trainer, no real data, NO REAL LOCK, NO REAL MARKER (/root/gpu_queue.lock and /root/N1DW_READY are never opened, created or removed: every path is under DRY below).
# Reads (read-only): /root/dscale10k/chainlib.py, /root/dscale10k/dscale_banker.py. Writes ONLY under /root/dscale10k/dry_reps2/ (fake roots fake_root, fake_w) and deletes the fake roots at the end.
# One PASS / FAIL line per scenario, then `scenarios PASS n FAIL m` and `ZZ DRYRUN-REPS2 DONE ZZ`.
R=/root/dscale10k; DRY=$R/dry_reps2; B=$DRY/blockreps2; F=$DRY/fake_root; PY=/root/venv/bin/python
LK=$F/fake_gpu.lock; MK=$F/fake_marker
ok=0; bad=0
chk() { if [ "$2" = "$3" ]; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: got [$2] want [$3]"; bad=$((bad+1)); fi; }
has() { if echo "$2" | grep -q -- "$3"; then echo "PASS $1"; ok=$((ok+1)); else echo "FAIL $1: [$3] not in [$(echo "$2" | tail -c 300)]"; bad=$((bad+1)); fi; }
mk() {
  rm -rf "$F"; mkdir -p $F/argv $F/dry $F/gate/tokens $F/chains
  cp $R/chainlib.py $F/chainlib.py; cp $R/dscale_banker.py $F/dscale_banker.py; cp $B/fake_leaf_reps2.sh $F/fake_leaf.sh
  for a in UE_s2 UE_s3; do echo '{"argv": ["--seed", "2", "--steps", "61568"]}' > $F/argv/$a.A.json; done
}
run() { DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 DS_RP=$B DS_LOCK=$LK DS_MARKER=$MK bash $B/blockreps2.sh > /dev/null 2>&1; tail -n 1 $F/blockreps2.log; }
lg() { cat $F/blockreps2.log; }
lock_free() { ( flock -n 8 && echo free || echo held ) 8>$LK; }
# 1 full run
mk; last=$(run)
chk "1 sentinel DONE" "$last" "ZZ DSCALE10K REPS2 DONE ZZ"
order=$(grep -E "fake (prep|trainer UE)" $F/fake_order.log | sed -E 's/^[0-9:]+ fake //' | tr '\n' ',')
chk "1 order: prep, UE_s2, UE_s3" "$order" "prep,trainer UE_s2,trainer UE_s3,"
chk "1 pair + verify before each arm" "$(grep -cE 'fake (pair|verify) ' $F/fake_order.log)" "4"
chk "1 two markers written by the banker" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s2.A.sha256 UE_s3.A.sha256 "
chk "1 hub holds the two checkpoints" "$(ls $F/fakehub/*/*/ckpt_last.pt | wc -l)" "2"
chk "1 ARM_A_DONE and ARM_DONE of both" "$(ls $F/runs/UE_s2/ARM_DONE $F/runs/UE_s3/ARM_DONE $F/runs/UE_s2/A/ARM_A_DONE $F/runs/UE_s3/A/ARM_A_DONE | wc -l)" "4"
chk "1 checkpoints at step 61568" "$($PY -c "import torch,glob;print(sorted({torch.load(p,weights_only=False)['step'] for p in glob.glob('$F/runs/*/A/ckpt_last.pt')}))")" "[61568]"
chk "1 post.json read the 252 / 252 zero-grad gate" "$(grep -c '"registered": 252, "with_adam_state": 252' $F/runs/UE_s2/A/post.json)" "1"
chk "1 the lock line is logged" "$(lg | grep -c 'GPU queue lock held to the end of the chain')" "1"
chk "1 lock free after the chain" "$(lock_free)" "free"
chk "1 the chain created no marker" "$([ -e $MK ] && echo yes || echo no)" "no"
# 2 rerun: skip
out=$(run); chk "2 rerun DONE" "$out" "ZZ DSCALE10K REPS2 DONE ZZ"; has "2 skip UE_s2" "$(lg)" "skip UE_s2 (ARM_DONE)"; has "2 skip UE_s3" "$(lg)" "skip UE_s3 (ARM_DONE)"; chk "2 no second trainer run" "$(grep -c 'fake trainer UE' $F/fake_order.log)" "2"
# 3 prep fail
mk; DS_FAKE_FAIL="prep:-" run > /dev/null; has "3 prep fail" "$(tail -n 1 $F/blockreps2.log)" "REPS2 prep"; chk "3 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c 'fake trainer UE')" "0"
chk "3 lock handed back at once" "$(lock_free)" "free"
# 4 tokens exist: no prep
mk; for a in UE_s2 UE_s3; do mkdir -p $F/gate/tokens/$a.A; echo '{}' > $F/gate/tokens/$a.A/gate_token.json; done; out=$(run); chk "4 DONE without prep" "$out" "ZZ DSCALE10K REPS2 DONE ZZ"; chk "4 no prep call" "$(grep -c 'fake prep' $F/fake_order.log)" "0"
# 5 disk guard BEFORE the lock
mk; out=$(DS_FREE_GB=10 run); has "5 disk guard" "$out" "free disk 10 GB < 60 GB"; chk "5 refused before the lock line" "$(lg | grep -c 'GPU queue lock held')" "0"
# 6 STOP before the lock
mk; touch $F/chains/STOP; out=$(run); has "6 STOP before anything" "$out" "STOP file before the lock"; chk "6 refused before the lock line" "$(lg | grep -c 'GPU queue lock held')" "0"
# 7 UE_s2 run fails
mk; out=$(DS_FAKE_FAIL="run:UE_s2" run); has "7 failed UE_s2" "$out" "UE_s2 stage A rc=7"; chk "7 UE_s3 never ran" "$(grep -c 'trainer UE_s3' $F/fake_order.log)" "0"; chk "7 lock handed back at once" "$(lock_free)" "free"
# 8 UE_s3 run fails after UE_s2 is done and banked
mk; out=$(DS_FAKE_FAIL="run:UE_s3" run); has "8 failed UE_s3" "$out" "UE_s3 stage A rc=7"; chk "8 UE_s2 banked before" "$(ls $F/bank_ok | tr '\n' ' ')" "UE_s2.A.sha256 "
# 9 a KILLED run is resumed from ckpt_last.pt (not refused, not restarted)
mk; out=$(DS_FAKE_KILL=UE_s2 run); has "9 killed run ends the chain" "$out" "UE_s2 stage A rc=9"
chk "9 partial checkpoint at step 500, no summary" "$($PY -c "import torch;print(torch.load('$F/runs/UE_s2/A/ckpt_last.pt',weights_only=False)['step'])") $([ -e $F/runs/UE_s2/A/summary.json ] && echo summary || echo nosummary)" "500 nosummary"
out=$(run); chk "9 relaunch DONE" "$out" "ZZ DSCALE10K REPS2 DONE ZZ"; has "9 the relaunch says RESUME at step 500" "$(lg)" "UE_s2: RESUME from $F/runs/UE_s2/A/ckpt_last.pt (step 500)"
chk "9 the trainer started at step 500 on the relaunch" "$(grep -c 'fake trainer start step 500' $F/fake_order.log)" "1"
chk "9 UE_s2 trained twice (kill + resume), UE_s3 once" "$(grep -c 'trainer UE_s2' $F/fake_order.log) $(grep -c 'trainer UE_s3' $F/fake_order.log)" "2 1"
chk "9 no prep on the relaunch" "$(grep -c 'fake prep' $F/fake_order.log)" "1"
# 10 a finished run that has no ARM_A_DONE: gates only, no training
rm -f $F/runs/UE_s2/ARM_DONE $F/runs/UE_s2/A/ARM_A_DONE; n0=$(grep -c 'fake trainer UE' $F/fake_order.log); out=$(run)
chk "10 DONE" "$out" "ZZ DSCALE10K REPS2 DONE ZZ"; has "10 post gates only" "$(lg)" "UE_s2: finished run found"; chk "10 no trainer call" "$(grep -c 'fake trainer UE' $F/fake_order.log)" "$n0"; chk "10 ARM_A_DONE rewritten" "$([ -e $F/runs/UE_s2/A/ARM_A_DONE ] && echo yes || echo no)" "yes"
# 11 pair refusal before training
mk; out=$(DS_FAKE_FAIL="pair:UE_s2" run); has "11 pair refusal" "$out" "UE_s2 pair check refused"; chk "11 nothing trained" "$(cat $F/fake_order.log 2>/dev/null | grep -c 'fake trainer UE')" "0"
# 12 token verify refusal of UE_s3 after UE_s2
mk; out=$(DS_FAKE_FAIL="verify:UE_s3" run); has "12 verify refusal" "$out" "UE_s3 token verify refused"; chk "12 UE_s3 never trained" "$(grep -c 'trainer UE_s3' $F/fake_order.log)" "0"; chk "12 UE_s2 done" "$([ -e $F/runs/UE_s2/ARM_DONE ] && echo yes || echo no)" "yes"
# 13 a held lock blocks the chain until released
mk; ( flock $LK sleep 7 ) & sleep 1
( DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 DS_RP=$B DS_LOCK=$LK DS_MARKER=$MK bash $B/blockreps2.sh > /dev/null 2>&1 ) & CP=$!
sleep 3; chk "13 while the lock is held: no lock line, nothing trained" "$(lg | grep -c 'GPU queue lock held') $(cat $F/fake_order.log 2>/dev/null | grep -c 'fake trainer')" "0 0"
wait $CP; wait; has "13 chain finished after the release" "$(tail -n 1 $F/blockreps2.log)" "ZZ DSCALE10K REPS2 DONE ZZ"
# 14 marker watcher: set while waiting, removed when the lock line is logged
mk; ( flock $LK sleep 8 ) & sleep 1
W() { DS_ROOT=$F DS_MARKER=$MK DS_CHAINLOG=$F/blockreps2.log DS_WLOG=$F/watch.log DS_PATTERN='dry_reps2/blockreps2/blockreps2[.]sh' DS_GRACE=4 DS_POLL=1 DS_ITER=40 bash $B/reps2_marker_watch.sh; }
W & WP=$!
sleep 1; chk "14 marker set by the watcher" "$([ -e $MK ] && echo yes || echo no) $([ -e $F/chains/reps2_marker_mine ] && echo mine || echo notmine)" "yes mine"
( DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_GPU_TRIES=2 DS_RP=$B DS_LOCK=$LK DS_MARKER=$MK bash $B/blockreps2.sh > /dev/null 2>&1 ) & CP=$!
sleep 3; chk "14 marker still set while the chain waits for the lock" "$([ -e $MK ] && echo yes || echo no)" "yes"
wait $CP; wait $WP; wrc=$?
chk "14 watcher exit 0 after the lock line" "$wrc" "0"; chk "14 marker removed" "$([ -e $MK ] && echo yes || echo no)" "no"; has "14 watcher log" "$(cat $F/watch.log)" "holds the GPU lock; marker removed"
# 15 a refusal before the lock clears the marker at once (watcher)
mk; touch $F/chains/STOP; W & WP=$!; sleep 1
( DS_FAKE=1 DS_ROOT=$F DS_PY=$PY DS_POLL=0 DS_RP=$B DS_LOCK=$LK DS_MARKER=$MK bash $B/blockreps2.sh > /dev/null 2>&1 ); sleep 2; wait $WP; wrc=$?
chk "15 watcher exit 1 (chain ended before the lock line)" "$wrc" "1"; chk "15 marker removed" "$([ -e $MK ] && echo yes || echo no)" "no"
# 16 the chain itself clears a marker it was given when it fails after the lock (watcher gone)
mk; touch $MK $F/chains/reps2_marker_mine; out=$(DS_FAKE_FAIL="run:UE_s2" run); chk "16 chain cleared its own marker on the refusal" "$([ -e $MK ] && echo yes || echo no)" "no"
# 17 the chain never removes a marker it did not set (no reps2_marker_mine)
mk; touch $MK; out=$(DS_FAKE_FAIL="run:UE_s2" run); chk "17 foreign marker kept" "$([ -e $MK ] && echo yes || echo no)" "yes"
# 18 static
for f in blockreps2.sh blockreps2_prep.sh chk_arm_reps2.sh reps2_marker_watch.sh fake_leaf_reps2.sh dryrun_reps2.sh; do bash -n $B/$f 2>/dev/null && r=ok || r=BAD; chk "18 bash -n $f" "$r" "ok"; done
chk "18 the chain takes fd 9 on the lock" "$(grep -c 'exec 9>\$LOCKF' $B/blockreps2.sh)" "1"
chk "18 the chain sets no marker" "$(grep -cE 'touch \$MARKER|> *\$MARKER' $B/blockreps2.sh)" "0"
chk "18 the real lock and marker are only defaults" "$(grep -c '/root/gpu_queue.lock\|/root/N1DW_READY' $B/blockreps2.sh | awk '{print ($1>=2)?"ok":"few"}')" "ok"
chk "18 the runner is the reps2 wrapper" "$(grep -c 'chk_arm_reps2.sh run' $B/blockreps2.sh)" "1"
rm -rf "$F"
echo "scenarios PASS $ok FAIL $bad"
echo "ZZ DRYRUN-REPS2 DONE ZZ"
