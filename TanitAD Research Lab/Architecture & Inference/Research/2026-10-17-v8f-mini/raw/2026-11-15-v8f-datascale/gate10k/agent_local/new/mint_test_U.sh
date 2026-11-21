#!/bin/bash
# mint_test_U.sh (pod, CPU): exercises `dscale10k_check.py mint --arm U_s0 --stage A` end to end with a TEST evidence file (G-CKPT stubbed: the GPU resume smoke has not finished) into a scratch gate dir.
# The token it may write binds a test evidence file and is never a launch token (gate dir = /root/dscale10k/tmp/mint_test_U, deleted at the end).
D=/root/dscale10k; G=$D/gate; PY=/root/venv/bin/python; TREE=$D/tree; T=$D/tmp/mint_test_U
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE:$D/study" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 DS10K_TMP=$D/tmp CUDA_VISIBLE_DEVICES=
mkdir -p $T
$PY - <<'E'
import json
G = "/root/dscale10k/gate/"
L = lambda n: json.load(open(G + n + ".json"))
init, clock = L("init"), L("clock")
ev = {"hyg": {"arm": init["record"]["U_s0"], "base": init["record"]["B_s0"]}, "reach": {k: v for k, v in json.load(open("/root/dscale10k/probe/U_s0.A.json")).items() if k in ("corpus", "tenk", "mixed", "control")},
      "clock": {"tenk": clock["record"]["tenk"], "corpus": clock["record"]["corpus"]}, "G-EVAL": {"pass": L("eval")["pass"]}, "G-CKPT": {"pass": True, "TEST_STUB": "resume smoke not finished"}, "G-SUITE": {"pass": L("suite")["pass"]},
      "c_init": init["record"]["c_init_report"]}
json.dump(ev, open("/root/dscale10k/tmp/mint_test_U/evidence_test.json", "w"))
E
$PY $D/study/dscale10k_check.py mint --arm U_s0 --stage A --argv $D/argv/U_s0.A.json --root $TREE --commit cdbff971fa6dad4d20b937fe8abda6cb9f622adb --gate-dir $T/gate --py $PY --gpu 0 \
  --base-argv $D/argv/B_s0.A.json --evidence $T/evidence_test.json --dry $D/dry/U_s0.A.dry.json --merge-record $D/merged/merge_record.json --union-release $D/merged/v9_labels_union.npz \
  --tenk-v9 $D/in/labels/v9_labels_datascale10k.npz --corpus-join /root/strat1/inputs/join_train.npz --union-join $D/merged/join_union.npz --eval-ids /root/ops/clips141.txt \
  --holdout-manifest /root/data/cache_pod/train/manifest.jsonl --digest-json $(ls $TREE/stack/tanitad/data/*_clip_digests.json | tr '\n' ' ') > $T/mint.log 2>&1
echo "mint rc $?" >> $T/mint.log
tail -c 1500 $T/mint.log
echo "ZZ MINT-TEST DONE ZZ"
