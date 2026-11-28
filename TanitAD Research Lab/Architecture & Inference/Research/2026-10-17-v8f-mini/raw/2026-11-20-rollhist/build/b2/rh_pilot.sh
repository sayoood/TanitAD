#!/bin/bash
# rh_pilot.sh -- HANDOFF_B1 step 3: the bank pilot (the part CPU cannot show). Called by rh_chain.sh, which HOLDS /root/gpu_queue.lock on fd 9: this script takes no lock.
# usage: RH_READOUT=causal_g1x|causal_g1 bash rh_pilot.sh        Writes $R/pilot/PILOT_RH.json ; last line `ZZ RH PILOT PASS ZZ` or `ZZ RH PILOT FAIL ZZ <why>` (exit 0 / 1).
# Pilots (UE_s0.A as the generator, union TRAIN windows, --limit-clips 200, 4 shards at the same time as bank4.sh does, then merge; about 111 s each, measured on DSCALE10K):
#   rh        = the RH generator flags (from RH_READOUT)            tree = $R/tree
#   default   = no new flags (the identity config)                  tree = $R/tree
#   unpatched = no new flags                                        tree = /root/dscale10k/tree (cdbff971: v8f_dagger_bank.py + v8f_dagger.py are byte-identical to the build base 7b26a8ac)
#   unpatched2 only if check (a) reads a nonzero diff (the GPU's own run-to-run floor)
# Checks (a)-(d): rh_pilot_check.py. The pilot shard dirs' npz files are deleted at the end (own temp, re-creatable; logged in $R/deletions.log); the json records stay.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rh_env.sh
rh_readout_flags
export_py
P=$R/pilot; mkdir -p $P
say() { echo "$(date -u +%T) [pilot] $*"; }
pfail() { echo "ZZ RH PILOT FAIL ZZ $1"; exit 1; }
CK=$DS/runs/UE_s0/A/ckpt_last.pt
$PY $DS/chainlib.py m2bargv $DS/argv/UE_s0.A.json $P/bank_m2b_argv.json || pfail "m2b argv"
read CACHE TRACKS JOIN <<< "$($PY - $DS/argv/UE_s0.A.json <<'PYE'
import json, sys
t = json.load(open(sys.argv[1]))["argv"]
g = lambda f: t[t.index(f) + 1]
print(g("--cache-dir"), g("--tracks"), g("--join-train"))
PYE
)"
[ -n "$JOIN" ] || pfail "bank paths"
run_pilot() {   # run_pilot NAME TREE "EXTRA"
  [ -s $P/$1/bank.json ] && { say "$1 pilot already has bank.json (kept)"; return 0; }
  rm -rf "${P:?}/$1"
  bash $HERE/rh_bank.sh "$2" $CK $P/bank_m2b_argv.json $CACHE $TRACKS $JOIN $P/$1 0 0 200 "$3" > $P/$1.log 2>&1 || pfail "$1 pilot: $(tail -n 3 $P/$1.log | tr '\n' ' ' | cut -c1-300)"
  say "$1: $(grep -E 'gen wall_s|total_wall_s' $P/$1.log | tr '\n' ' ')"
}
run_pilot rh $TREE "$BANK_RO_FLAGS"
run_pilot default $TREE ""
run_pilot unpatched $DS/tree ""
chk() { $PY $HERE/rh_pilot_check.py pilot --rh $P/rh --default $P/default --unpatched $P/unpatched ${1:+--unpatched2 $P/unpatched2} --readout $RH_READOUT --out $P/PILOT_RH.json > $P/check.log 2>&1; }
if ! chk; then
  if grep -q '"max_abs_diff_new_default_vs_unpatched": 0.0' $P/check.log || [ -d $P/unpatched2 ]; then :; else
    say "check (a) read a nonzero diff or another check failed; running a second unpatched pilot to read the device floor"
    run_pilot unpatched2 $DS/tree ""
    chk 2
  fi
fi
tail -n 1 $P/check.log | grep -q "ZZ RH-PILOT-CHECK PASS" || pfail "checks (a)-(d): $(tail -c 500 $P/check.log | tr '\n' ' ')"
# own temp: the pilot shard npz files (the jsons and the bank.json stay)
for d in rh default unpatched unpatched2; do
  for f in $P/$d/shard_*.npz; do [ -e "$f" ] || continue; echo "$(date -u +%FT%TZ) $f $(stat -c %s "$f") (pilot shard, re-creatable; PILOT_RH.json passed)" >> $R/deletions.log; rm -f "$f"; done
done
echo "ZZ RH PILOT PASS ZZ"
