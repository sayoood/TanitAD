#!/bin/bash
# P6 rebuild driver, new pod (2026-10-22, MM). PI: "Stay on this pod" + "Use the token from our keys.txt".
# DBs on local disk (move_dbs.sh) -> prebuild (bcc04b5 chain, STOP_AFTER=cache, no P6_CBANK) || pixels 19-32 (authenticated, curl -f)
#  -> verify train_grow_cmd shas -> 3-C twin parts -> re-roll (720 shards) -> merge -> verify 3-C shas
#  -> final chain (tip tree, p6_start.env with P6_CBANK) = resume from HF ckpt_last.pt.
# Resumable: re-run it; finished steps leave $O/<step>.done. Any mismatch STOPS (ZZRB_FAIL), never relaxes.
# The token is read from <TOKEN_FILE> into the environment of the HF-using children only; never an argument, never printed.
set -u
W=/workspace; O=$W/ops/rebuild; mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/driver.log; }
fail() { log "ZZRB_FAIL $*"; echo "$*" > $O/FAILED; exit 1; }
done_() { [ -f $O/$1.done ]; }
mark() { date -u +%FT%T > $O/$1.done; log "ZZRB_OK $1 $2"; }
R="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
T="$R/raw/2026-10-17-twin-bank-3c"
BCC=bcc04b52233e28ef8eaafc346472c080e027e4a9
TIP=f2f604f5d8067352287e113d90bc925f76c9641e
G="git -C $W/repo"
NB=$W/data/refe_navtrain
F=$W/ops/bank3c/full
S0=daa23b962e9513673347694e097fb714a53b646c4d402802d6d7b96354c0bc5e
S1IN=2f3e461391d7bbc6c92a4153ef1837943c458d9635aad13fdb27b74a46035898
S1OUT=2a63acc2f63404818939b21d45cfa27efb5412cbefdf38e14aeb8673598cf409
NP=8
sha() { sha256sum < "$1" | cut -c1-64; }
rm -f $O/FAILED
[ -s <TOKEN_FILE> ] || fail "no <TOKEN_FILE>"
log "driver start pid $$"
# the DBs live on the local container disk (move_dbs.sh, 2026-10-22): the network volume's quota left ~16 GiB
until grep -q "^.* ZZMOVE_OK" $O/move_dbs.log 2>/dev/null; do grep -q "ZZMOVE_FAIL" $O/move_dbs.log 2>/dev/null && fail "DB move failed: $(tail -n 1 $O/move_dbs.log)"; sleep 60; done
[ "$(ls $W/data/navtrain_dbs/*.db | wc -l)" -ge 1192 ] || fail "DBs not reachable through the symlink"

# ---- pixels 19-32 (the 2026-10-21 anonymous run lost 14 of 32: no -f, so an HTTP error body was saved as the tgz)
pix_one() {
  n=$1; PIX=$W/data/navtrain_pixels
  HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim
  [ -f $PIX/.done_$n ] && return 0
  for a in 1 2 3 4 5; do
    mkdir -p /root/tad/tgz; tgz=/root/tad/tgz/navtrain_current_$n.tgz; rm -f "$tgz"
    if curl -fsSL --retry 5 --retry-all-errors -H @/root/.hf_auth_hdr -o "$tgz" "$HF/navtrain_current_$n.tgz" 2>> $O/pix_err.log \
       && tar -xzf "$tgz" --no-same-owner -C $PIX --wildcards '*/CAM_F0/*' '*/CAM_B0/*' '*/CAM_L0/*' '*/CAM_R0/*' 2>> $O/pix_err.log; then
      touch $PIX/.done_$n; rm -f "$tgz"; echo "ZZPIX_OK $n a$a $(date -u +%T)" >> $O/pix.log; return 0
    fi
    echo "ZZPIX_RETRY $n a$a $(date -u +%T) $(stat -c %s "$tgz" 2>/dev/null)" >> $O/pix.log; rm -f "$tgz"; sleep $((60*a))
  done
  echo "ZZPIX_FAIL $n" >> $O/pix.log; return 1
}
if ! done_ pix; then
  ( umask 077; printf 'Authorization: Bearer %s\n' "$(cat <TOKEN_FILE>)" > /root/.hf_auth_hdr )
  ( for n in $(seq 1 32); do [ -f $W/data/navtrain_pixels/.done_$n ] || echo $n; done > $O/pix_todo.txt
    export -f pix_one; export W O
    xargs -a $O/pix_todo.txt -P 2 -I{} bash -c 'pix_one {}'
    c=$(ls $W/data/navtrain_pixels/.done_* 2>/dev/null | wc -l)
    if [ "$c" -ge 32 ]; then mark pix "32/32 shards"; else log "ZZRB_PIX_INCOMPLETE $c/32 (pix.log, pix_err.log)"; fi
    rm -f /root/.hf_auth_hdr ) &
  PIXPID=$!
fi

# ---- prebuild: bcc04b5's chain (identical to the tip up to `cache` except the resume-bank guard), no P6_CBANK
if ! done_ prebuild; then
  PB=$W/refe-plan-p6-prebuild
  if [ ! -f $PB/SHIP_MANIFEST.json ]; then
    rm -rf $PB.tmp; mkdir -p $PB.tmp
    $G archive $BCC:"$R" code refe splits | tar -x -C $PB.tmp || fail "archive bcc04b5"
    ( cd $PB.tmp/refe && python3 launch_gate.py ship-manifest --pkg $PB.tmp --commit $BCC --git-dir $W/repo/.git --repo-prefix "$R/" --out $PB.tmp/SHIP_MANIFEST.json ) > $O/ship_bcc.out 2>&1 || fail "ship-manifest bcc04b5 (ship_bcc.out)"
    rm -rf $PB; mv $PB.tmp $PB
  fi
  log "prebuild: chain at $BCC, P6_STOP_AFTER=cache, P6_EXPECT_PIX_SHARDS=0"
  ( set -a; . $W/ops/p6_prebuild.env; set +a
    export P6_PKG=$PB P6_STOP_AFTER=cache P6_EXPECT_PIX_SHARDS=0
    HF_TOKEN="$(cat <TOKEN_FILE>)" bash $PB/refe/p6_pod.sh ) >> $W/data/refe_p6/p6_stdout.log 2>&1 < /dev/null
  tail -n 20 $W/data/refe_p6/chain.log | grep -q "stopping after cache (P6_STOP_AFTER)" || fail "prebuild did not stop after cache: $(tail -n 1 $W/data/refe_p6/chain.log | cut -c1-250)"
  [ "$(sha $NB/train_grow_cmd/targets_rank0.jsonl)" = $S0 ] || fail "train_grow_cmd rank0 sha != daa23b96"
  [ "$(sha $NB/train_grow_cmd/targets_rank1.jsonl)" = $S1IN ] || fail "train_grow_cmd rank1 sha != 2f3e4613"
  mark prebuild "train_grow_cmd rank0 daa23b96 rank1 2f3e4613 (bit-identical to the old pod)"
fi

# ---- 3-C identity controls: the banked I1-I4 result (old pod, 2026-10-17) + this pod's 74-twin cross-pod control (bit-equal to it)
CJ=$W/ops/bank3c/controls_banked/controls.json
mkdir -p $(dirname $CJ)
[ -s $CJ ] || $G show $TIP:"$T/results/controls/controls.json" > $CJ || fail "cannot read the banked controls.json"
grep -q '"all_pass": true' $CJ || fail "banked controls.json not all_pass"
grep -q '^ZZCTL74_PASS' $W/ops/bank3c/controls74/ctl74_verdict.txt || fail "the 74-twin cross-pod control has not passed"
export CONTROLS_JSON=$CJ

# ---- twin parts (44,409) on the frozen D7 tree
if [ ! -f $F/PARTS_DONE ]; then
  log "twin parts: start (WORKERS=$NP)"
  WORKERS=$NP bash $W/ops/bank3c/code/build_twin_parts_3c.sh > $O/parts.out 2>&1
  [ -f $F/PARTS_DONE ] || fail "twin parts: $(tail -n 1 $O/parts.out | cut -c1-250)"
  mark parts "44,409 twin parts, D7 distinct OK K=1..3"
fi

# ---- the re-roll (720 shards, chunk 32)
if [ ! -f $F/RUN_DONE ]; then
  for t in 1 2 3; do
    rm -f $F/RUN_INCOMPLETE
    NPROC=$NP bash $W/ops/bank3c/run_full.sh launch > $O/reroll_launch_$t.out 2>&1 || fail "run_full launch refused: $(tail -n 1 $O/reroll_launch_$t.out | cut -c1-250)"
    log "reroll: launched try $t"
    until [ -f $F/RUN_DONE ] || [ -f $F/RUN_INCOMPLETE ]; do sleep 120; done
    [ -f $F/RUN_DONE ] && break
    log "reroll: $(cat $F/RUN_INCOMPLETE)"
  done
  [ -f $F/RUN_DONE ] || fail "reroll incomplete after 3 tries"
  mark reroll "$(cat $F/RUN_DONE)"
fi

# ---- merge + the three recorded shas
OUT=$NB/train_grow_cmd_3c
if ! done_ merge; then
  if [ ! -f $OUT/targets_rank1.jsonl ]; then
    ( source $W/ops/bank3c/code/env_3c.sh
      run3c $PY $CODE3C/reroll_bank_3c.py merge --bank $NB/train_grow_cmd --shards "$F/shards/s*" --out-dir $OUT ) > $O/merge.out 2>&1
    rc=$?; [ $rc = 0 ] || fail "merge rc $rc: $(tail -n 1 $O/merge.out | cut -c1-250)"
  fi
  [ "$(sha $OUT/targets_rank1.jsonl)" = $S1OUT ] || fail "3-C rank1 sha $(sha $OUT/targets_rank1.jsonl | cut -c1-16) != 2a63acc2 -- NOT the bank P6 trained on; no resume"
  [ "$(sha $OUT/targets_rank0.jsonl)" = $S0 ] || fail "3-C rank0 sha != daa23b96"
  mark merge "train_grow_cmd_3c rank1 2a63acc2 rank0 daa23b96 (byte-identical to the bank P6 trained on)"
fi

# ---- the final chain needs all 32 pixel shards
if [ -n "${PIXPID:-}" ]; then wait $PIXPID; fi
done_ pix || fail "pixels incomplete: $(ls $W/data/navtrain_pixels/.done_* 2>/dev/null | wc -l)/32 (pix.log)"

# ---- final chain: the tip tree with P6_CBANK = the 3-C bank; recover.ok exists, the guard reads config.json -> 3-C = CBANK
grep -q '^P6_CBANK=/workspace/data/refe_navtrain/train_grow_cmd_3c$' $W/ops/p6_start.env || fail "p6_start.env does not pin the 3-C bank"
log "ZZRB_FINAL_CHAIN launching (tip $TIP, p6_start.env)"
echo "$(date -u +%FT%TZ) MM rebuild driver: final chain on the new pod, env /workspace/ops/p6_start.env (P6_CBANK=3-C), resume from the HF checkpoint" >> $W/data/refe_p6/p6_stdout.log
( set -a; . $W/ops/p6_start.env; set +a
  HF_TOKEN="$(cat <TOKEN_FILE>)" bash $W/refe-plan-p6/refe/p6_pod.sh ) >> $W/data/refe_p6/p6_stdout.log 2>&1 < /dev/null
log "ZZRB_FINAL_CHAIN_EXIT rc $? last: $(tail -n 1 $W/data/refe_p6/chain.log | cut -c1-200)"
