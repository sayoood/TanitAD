#!/bin/bash
# P6 relaunch after the L40 G-LIVE-LAB refusal (PREREG_P6 Amendment (e), MM 2026-11-06): --label-workers 6 -> 13 (the chain's own CORES-2); everything else exactly
# as restage_v2.sh s_final (guard on the first labeller events, the chain with the start env + token from file).  The chain re-keys lab (w13-c2), mem and egr (new ARGS sha)
# and re-runs the launch gate by its own rules: no stage marker is touched here.  Run DETACHED.
set -u
W=/workspace; O=$W/ops/relaunch_e; P6=$W/data/refe_p6; RUN=$W/data/refe_runs/p6_paper_faithful; PKG=$W/refe-plan-p6; ENVF=$W/ops/p6_start.env; HERE=/root/tad/restage_v2
mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/relaunch_e.log; }
fail() { log "ZZRE_FAIL $*"; echo "$*" > $O/FAILED; exit 1; }
rm -f $O/FAILED $O/DONE
log "start pid $$"
ps -eo args= | grep -qE "[p]6_pod.sh|[t]rain.py|[r]estage_v2.sh" && fail "a chain, trainer or restage driver is running"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || fail "GPU busy"
[ "$(md5sum < $PKG/refe/p6_pod.sh | cut -c1-12)" = 65f90c76f3d7 ] || fail "p6_pod.sh is not the chain of the first L40 attempt"
python3 -I $HERE/restage_helpers.py verify --run $RUN --expect-step 6266 > $O/verify.out 2>&1 || fail "verify: $(tail -n 1 $O/verify.out)"
CORES=$(( $(cut -d' ' -f1 /sys/fs/cgroup/cpu.max) / $(cut -d' ' -f2 /sys/fs/cgroup/cpu.max) ))
[ $((CORES - 2)) = 13 ] || fail "the chain's own derivation is $((CORES - 2)), not 13: the quota changed (new amendment needed)"
[ -e /root/.hf_auth_hdr ] && fail "auth header file present"
cp -p $ENVF $ENVF.pre_amend_e
sed -i 's/^P6_LABEL_WORKERS=6$/P6_LABEL_WORKERS=13/' $ENVF
for p in P6_LABEL_WORKERS=13 P6_D7_FALLBACK=0 P6_CBANK=$W/data/refe_navtrain/train_grow_cmd_3c P6_YAW_WRAP=off; do grep -qx "$p" $ENVF || fail "start env lacks $p"; done
off=$(stat -c %s $RUN/metrics.jsonl); coff=$(stat -c %s $P6/chain.log)
log "ZZRE_LAUNCH P6_LABEL_WORKERS=13 (was 6; backup $ENVF.pre_amend_e), resume step 6266, metrics offset $off"
echo "$(date -u +%FT%TZ) MM relaunch_e: PREREG_P6 Amendment (e) -- G-LIVE-LAB refused at 6 label workers on the L40 (GPU stall 5.4 % > 5 % in one window); --label-workers 13 (the chain's CORES-2); lab/mem/egr/launch gate re-run, then resume from 6266" >> $P6/p6_stdout.log
python3 -I $HERE/restage_helpers.py guard --metrics $RUN/metrics.jsonl --offset "$off" --wait-s 14400 --lines 5 --alert $O/ALERT_COVERAGE.txt >> $O/guard.log 2>&1 &
GPID=$!
( for i in $(seq 1 720); do l="$(tail -c +$((coff + 1)) $P6/chain.log 2> /dev/null | grep -a -m1 'ARGS sha')"; [ -n "$l" ] && break; sleep 10; done; log "ZZRE_ARGS $l" ) &
( set -a; . $ENVF; set +a
  HF_TOKEN="$(cat <TOKEN_FILE>)" bash $PKG/refe/p6_pod.sh ) >> $P6/p6_stdout.log 2>&1 < /dev/null
rc=$?; kill $GPID 2> /dev/null
log "ZZRE_CHAIN_EXIT rc $rc last: $(tail -n 1 $P6/chain.log | cut -c1-200)"
date -u +%FT%T > $O/DONE
