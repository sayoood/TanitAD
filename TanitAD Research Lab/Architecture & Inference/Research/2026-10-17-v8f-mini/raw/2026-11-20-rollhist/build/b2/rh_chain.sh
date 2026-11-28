#!/bin/bash
# ROLLHIST chain (PREREG_V8F_ROLLHIST, code of record 2bda9e1a, PI go 2026-11-20: both seeds, about 5.2 LoRA pod-h). STAGED, NOT LAUNCHED. One process, one log. Adapted from n1dc_chain.sh / block2.sh.
# The readout of the training history is ONE parameter, RH_READOUT=causal_g1x (C*, recommended) | causal_g1 (ADD63 causal). It has NO default: an unset value is refused (the PI's decision is not defaulted).
# Order: (1) flock fd 9 on /root/gpu_queue.lock, HELD TO THE END  (2) tree + inputs + disk checks  (3) render argv + pair check (CPU)  (4) G-SUITE (CPU)  (5) pilot, refuses on any failed check (a)-(d)
#        (6) bank s0 (4 shards, merge) + full-bank re-check (b,c,d,e) + fill argv  (7) RH_s0, then B0_s0  (8) bank s1 + re-check + fill, then RH_s1, then B0_s1.
# Every step leaves $R/done/<name>.done; a relaunch skips finished steps (a half-finished arm is moved aside, an incomplete bank is deleted: own temp, logged). STOP file: $R/chains/STOP, read between steps.
# Log: /root/rollhist/rh_chain.log ; its LAST line is `ZZ ROLLHIST DONE ZZ` or `ZZ ROLLHIST FAIL ZZ <why>`.
# MARKER PROTOCOL (the MM's, not done here): the MM sets /root/N1DW_READY BEFORE the launch so the other session's (TanitADRuled) next slice skips; rh_marker_watch.sh removes it when this chain holds the lock.
# LAUNCH (MM):  RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_chain.sh < /dev/null > /dev/null 2>&1 &     (use the same RH_READOUT for every relaunch; it is recorded in $R/RH_READOUT.chosen)
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rh_env.sh
D=$R/done; mkdir -p $D $R/logs $R/runs $R/bank $R/chains
LOG=$R/rh_chain.log
exec >> $LOG 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { echo "ZZ ROLLHIST FAIL ZZ $1"; exit 1; }
rh_readout_flags
export_py
step() { local n=$1; shift; [ -e $D/$n.done ] && { say "skip $n (done)"; return 0; }; [ -e $R/chains/STOP ] && fail "STOP file before step $n"; say "step $n"; "$@" || fail "step $n"; date -u +%FT%TZ > $D/$n.done; }
say "chain start commit=${COMMIT:0:8} readout=$RH_READOUT (decl token $DECL_TOKEN; bank flags: $BANK_RO_FLAGS)"
# 1. the GPU queue lock, held to the end
exec 9>/root/gpu_queue.lock
say "waiting for the GPU queue lock"
flock 9 || fail "flock"
say "GPU queue lock held to the end of the chain"
[ -e $R/chains/STOP ] && fail "STOP file at $R/chains/STOP"
# 2. tree, inputs, disk
chk_tree() {
  $PY $R/blobcheck.py $TREE $R/all_blobs.txt > $R/logs/blobcheck.log 2>&1 || return 1
  [ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs.txt)" ] || return 1
}
step tree chk_tree
chk_inputs() {
  echo "2fd4b747532f55b24523c578948cb713fb80fa0e698586c329dd23ba6e6df914  $DS/runs/UE_s0/A/ckpt_last.pt" >  $R/inputs.sha256
  echo "f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc  $DS/runs/UE_s1/A/ckpt_last.pt" >> $R/inputs.sha256
  sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify.log 2>&1 || return 1
  for f in $DS/argv/U_s0.B.tmpl.json $DS/argv/U_s1.B.tmpl.json $DS/argv/UE_s0.A.json $DS/argv/UE_s1.A.json $DS/dry/UE_s0.A.dry.json $DS/dry/UE_s1.A.dry.json $DS/chainlib.py \
           $DS/merged/v9_labels_union.npz $DS/merged/lead_union.npz $DS/merged/speed_union.jsonl $DS/merged/join_union.npz $DS/merged/native_tracks_union.pt $S1/inputs/p10/p10_prior.pt \
           $S1/eval_grid_windows.json $S1/overlay_strat1_check.py $S1/shards_like_base.py $S1/n1dc/kclip.py; do [ -s "$f" ] || { echo "missing input $f" >> $R/logs/inputs_verify.log; return 1; }; done
  [ -d /root/data/cache_union_dscale/train ] || return 1
  echo "$DS/merged/v9_labels_union.npz md5 9881b170a889f743c5b0894f9ad13f1d" >> $R/logs/inputs_verify.log
  [ "$(md5sum $DS/merged/v9_labels_union.npz | cut -d' ' -f1)" = "9881b170a889f743c5b0894f9ad13f1d" ] || return 1
}
step inputs chk_inputs
free_gb() { df -BG /root | tail -1 | awk '{gsub("G","",$4); print $4}'; }
disk_ok() { f=$(free_gb); say "free disk ${f} GB"; [ "${f:-0}" -ge 60 ]; }
step disk0 disk_ok
# 3. render + pair (CPU); 4. G-SUITE (CPU)
rn() { $PY $HERE/render_argv_rh.py render > $R/logs/render.log 2>&1; tail -n 1 $R/logs/render.log | grep -q "ZZ RENDER-RH PASS"; }
step render rn
gs() { bash $HERE/rh_gsuite.sh > $R/logs/rh_gsuite.driver.log 2>&1; tail -n 1 $R/logs/rh_gsuite.log | grep -q "ZZ RH GSUITE DONE ZZ"; }
step gsuite gs
# 5. the pilot: any failed check refuses the chain
pl() { bash $HERE/rh_pilot.sh > $R/logs/rh_pilot.log 2>&1; tail -n 1 $R/logs/rh_pilot.log | grep -q "ZZ RH PILOT PASS ZZ"; }
step pilot pl
# 6/8. a bank: all shards, merge, full re-check, then the argv with the bank filled in and the pair check
mk_bank() {   # mk_bank SEED
  local s=$1 B=$R/bank/s$1 ba ec ew A
  disk_ok || { say "free disk below 60 GB before the s$s bank"; return 1; }
  if [ -d "$B" ] && [ ! -s "$B/BANK_CHECK.json" ]; then       # an incomplete bank of an earlier launch: own temp, re-creatable, deleted by literal path
    say "incomplete bank dir $B deleted ($(du -sk "$B" | cut -f1) KiB)"; echo "$(date -u +%FT%TZ) $B $(du -sk "$B" | cut -f1)KiB (incomplete bank of a failed launch)" >> $R/deletions.log
    rm -rf "${B:?}"
  fi
  $PY $DS/chainlib.py m2bargv $DS/argv/UE_s$s.A.json $R/bank/m2b_argv_s$s.json || return 1
  ba=$($PY $DS/chainlib.py bankargs $DS/dry/UE_s$s.A.dry.json) || return 1; ec=${ba% *}; ew=${ba#* }
  A=$($PY - $DS/argv/UE_s$s.A.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
g = lambda f: t[t.index(f) + 1]
print(g("--cache-dir"), g("--tracks"), g("--join-train"))
PYE
) || return 1
  set -- $A
  say "bank s$s: n_shards 4 parallel, $ec clips, $ew windows, flags [$BANK_RO_FLAGS]"
  bash $HERE/rh_bank.sh $TREE $DS/runs/UE_s$s/A/ckpt_last.pt $R/bank/m2b_argv_s$s.json $1 $2 $3 $B $ec $ew 0 "$BANK_RO_FLAGS" > $R/bank/bank_s$s.log 2>&1 || { tail -n 3 $R/bank/bank_s$s.log; return 1; }
  $PY $HERE/rh_pilot_check.py bank --bank $B --readout $RH_READOUT --expect-windows $ew --out $B/BANK_CHECK.json > $R/bank/bank_check_s$s.log 2>&1 || { tail -c 600 $R/bank/bank_check_s$s.log; return 1; }
  $PY $HERE/render_argv_rh.py fill $s > $R/logs/fill_s$s.log 2>&1 || { tail -n 5 $R/logs/fill_s$s.log; return 1; }
}
arm() {       # arm TAG : skip a finished arm, move a partial one aside (never delete a checkpoint), run it
  local tag=$1
  [ -e $R/runs/$tag/ARM_DONE ] && [ -s $R/runs/$tag/train/ckpt_last.pt ] && { say "skip $tag (ARM_DONE, ckpt present)"; return 0; }
  if [ -d $R/runs/$tag/train ] || [ -d $R/runs/$tag/selfcheck ]; then
    local A=$R/runs/aside_${tag}_$(date -u +%H%M%S); mkdir -p $A
    for d in train selfcheck; do [ -e $R/runs/$tag/$d ] && mv $R/runs/$tag/$d $A/; done
    rm -f $R/runs/$tag/ARM_DONE; say "partial $tag moved aside to $A"
  fi
  say "arm $tag"
  bash $HERE/rh_arm.sh $tag
}
for s in 0 1; do
  step bank_s$s mk_bank $s
  step RH_s$s arm RH_s$s
  step B0_s$s arm B0_s$s
done
say "free disk $(free_gb) GB at the end; banks kept: $(du -sh $R/bank/s0 $R/bank/s1 2>/dev/null | tr '\n' ' ')"
echo "ZZ ROLLHIST DONE ZZ"
