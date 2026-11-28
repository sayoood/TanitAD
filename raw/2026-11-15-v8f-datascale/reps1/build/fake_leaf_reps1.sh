#!/bin/bash
# fake_leaf_reps1.sh -- the leaf operations of blockreps1.sh for a DRY RUN (DS_FAKE=1). Emulates the trainer runner (refuses rc 2 without a token file), the prep and the UM checks.
# usage: fake_leaf_reps1.sh <run|prep|um> ARG2 ARG3   (DS_ROOT = the fake root; DS_FAKE_FAIL="run:UE_s1.A" / "um:post.-" makes that leaf exit 7)
D=$DS_ROOT; K=$1; ARM=$2; ST=$3
[ "${DS_FAKE_FAIL:-}" = "$K:$ARM.$ST" ] && { echo "fake failure of $K $ARM $ST"; exit 7; }
PY=${DS_PY:-/root/venv/bin/python}
case $K in
  run)
    [ -s $D/gate/tokens/$ARM.$ST/gate_token.json ] || { echo "REFUSED $ARM: no gate token"; exit 2; }
    n=61568
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
    for x in UE_s1 UM_s1; do mkdir -p $D/gate/tokens/$x.A; echo '{"fake": true}' > $D/gate/tokens/$x.A/gate_token.json; done
    echo "$(date -u +%T) fake prep" >> $D/fake_order.log; echo "ZZ REPS1-PREP DONE ZZ";;
  um)
    echo "$(date -u +%T) fake um_check $ARM" >> $D/fake_order.log;;
esac
exit 0
