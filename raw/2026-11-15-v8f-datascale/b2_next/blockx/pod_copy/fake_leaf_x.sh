#!/bin/bash
# FORK of block3/fake_leaf3.sh for the conditional arm X_s0 (the 10,000 data-scale clips alone; card 2.3 row X_s0, section 6 DATA-HURTS). Changes: blockx/FORK_NOTES.md. NOT launched: the MM asks the PI first.
# fake_leaf3.sh (block 3: UE_s0 / BL_s0, stage A only, 61,568 steps; + prep; the trainer's summary also carries ARM_*_DONE via the chain) -- the leaf operations of block1.sh / block2.sh for a DRY RUN (DS_FAKE=1). Emulates the trainer runner (refuses rc 2 without a token file), the bank build, the stage-B selfcheck and the mint.
# usage: fake_leaf.sh <run|mint|bank|selfcheck> ARM STAGE   (DS_ROOT = the fake root; DS_FAKE_FAIL="run:U_s0.A" makes that leaf exit 7)
D=$DS_ROOT; K=$1; ARM=$2; ST=$3
[ "${DS_FAKE_FAIL:-}" = "$K:$ARM.$ST" ] && { echo "fake failure of $K $ARM $ST"; exit 7; }
PY=${DS_PY:-/root/venv/bin/python}
case $K in
  run)
    [ -s $D/gate/tokens/$ARM.$ST/gate_token.json ] || { echo "REFUSED $ARM: no gate token"; exit 2; }
    n=19164; [ "$ST" = B ] && n=3000
    mkdir -p $D/runs/$ARM/$ST
    echo "$(date -u +%T) fake trainer $ARM $ST" >> $D/fake_order.log
    $PY - $D/runs/$ARM/$ST $n <<'E'
import json, sys, torch
d, n = sys.argv[1], int(sys.argv[2])
torch.save({"step": n, "model": {f"t{i}": torch.full((2,), float(n + i)) for i in range(252)},
            "opt": {"param_groups": [{"params": list(range(252))}], "state": {i: {"a": 1} for i in range(252)}}}, d + "/ckpt_last.pt")
json.dump({"done": True, "steps": n}, open(d + "/summary.json", "w"))
E
    ;;
  prep)
    for x in X_s0; do mkdir -p $D/gate/tokens/$x.A; echo '{"fake": true}' > $D/gate/tokens/$x.A/gate_token.json; done
    echo "$(date -u +%T) fake prep" >> $D/fake_order.log; echo "ZZ BLOCKX-PREP DONE ZZ";;
  mint)
    mkdir -p $D/gate/tokens/$ARM.$ST; echo '{"fake": true}' > $D/gate/tokens/$ARM.$ST/gate_token.json
    echo "$(date -u +%T) fake mint $ARM $ST" >> $D/fake_order.log;;
  bank)
    o=$D/runs/$ARM/bank; mkdir -p $o
    for k in 0 1 2 3; do head -c 1000 /dev/zero > $o/shard_00${k}of004.npz; done
    echo '{"n_shards": 4}' > $o/bank.json; echo '{}' > $o/PROVENANCE.json
    echo "$(date -u +%T) fake bank $ARM (n_shards 4 in parallel)" >> $D/fake_order.log;;
  selfcheck)
    mkdir -p $D/runs/$ARM/B_selfcheck; echo '{}' > $D/runs/$ARM/B_selfcheck/selfcheck.json; echo "$(date -u +%T) fake selfcheck $ARM" >> $D/fake_order.log;;
esac
exit 0
