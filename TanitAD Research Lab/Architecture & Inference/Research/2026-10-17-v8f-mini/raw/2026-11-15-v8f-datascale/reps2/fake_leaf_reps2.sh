#!/bin/bash
# fake_leaf_reps2.sh -- the leaf operations of blockreps2.sh for a DRY RUN (DS_FAKE=1). Emulates the token-refusing trainer runner (rc 2 without a token file), a KILLED run (DS_FAKE_KILL=<ARM>: writes a step-500
# ckpt_last.pt and no summary.json, rc 9) and the RESUME of such a directory (the fake trainer reads the existing ckpt step, as the real trainer does at v8f_mini_train.py:3499), the prep, the token verify and the pair check.
# usage: fake_leaf_reps2.sh <run|prep|verify|pair> ARM A   (DS_ROOT = the fake root; DS_FAKE_FAIL="run:UE_s2" / "verify:UE_s3" / "pair:UE_s2" / "prep:-" makes that leaf exit 7)
D=$DS_ROOT; K=$1; ARM=$2
[ "${DS_FAKE_FAIL:-}" = "$K:$ARM" ] && { echo "fake failure of $K $ARM"; exit 7; }
PY=${DS_PY:-/root/venv/bin/python}
case $K in
  run)
    [ -s $D/gate/tokens/$ARM.A/gate_token.json ] || { echo "REFUSED $ARM: no gate token"; exit 2; }
    n=61568
    mkdir -p $D/runs/$ARM/A
    echo "$(date -u +%T) fake trainer $ARM" >> $D/fake_order.log
    DS_KILL=$([ "${DS_FAKE_KILL:-}" = "$ARM" ] && echo 1 || echo 0) $PY - $D/runs/$ARM/A $n <<'E'
import json, os, sys, torch
d, n = sys.argv[1], int(sys.argv[2])
ck = d + "/ckpt_last.pt"
start = torch.load(ck, weights_only=False)["step"] if os.path.exists(ck) else 0
print(f"[resume] step {start}" if start else "[fresh] step 0", flush=True)
open(os.path.dirname(os.path.dirname(os.path.dirname(d))) + "/fake_order.log", "a").write(f"fake trainer start step {start}\n")
end = 500 if os.environ.get("DS_KILL") == "1" else n
torch.save({"step": end, "model": {f"t{i}": torch.full((2,), float(end + i)) for i in range(252)},
            "opt": {"param_groups": [{"params": list(range(252))}], "state": {i: {"a": 1} for i in range(252)}}}, ck + ".tmp")
os.replace(ck + ".tmp", ck)
json.dump({"declared_vs_built": "PASS"}, open(d + "/selfcheck.json", "w"))
if end == n:
    json.dump({"done": True, "steps": n}, open(d + "/summary.json", "w"))
sys.exit(0 if end == n else 9)
E
    rc=$?; [ $rc -eq 0 ] || exit $rc
    ;;
  prep)
    for x in UE_s2 UE_s3; do mkdir -p $D/gate/tokens/$x.A; echo '{"fake": true}' > $D/gate/tokens/$x.A/gate_token.json; done
    echo "$(date -u +%T) fake prep" >> $D/fake_order.log; echo "ZZ REPS2-PREP DONE ZZ";;
  verify) echo "$(date -u +%T) fake verify $ARM" >> $D/fake_order.log;;
  pair)   echo "$(date -u +%T) fake pair $ARM" >> $D/fake_order.log;;
esac
exit 0
