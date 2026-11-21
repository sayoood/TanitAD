#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 4): n1dh_arm.sh line 27 read `--control 'route_head\.weight'--json-out <path>` (no space), so argparse saw the control
# regex `route_head\.weight--json-out` and the path as an unrecognised argument: the zero-grad gate exited 2 after N1DH_s0 had trained (3,000 steps,
# post check PASS). The fix inserts the one space. Then the arm's three remaining lines for N1DH_s0 (zero-grad gate, K-CLIP, ARM_DONE) run by hand,
# under the GPU queue lock, exactly as the fixed script writes them. The relaunched chain then skips N1DH_s0 (ARM_DONE + ckpt) and runs N1DH_s1.
set -euo pipefail
R=/root/strat1; N=$R/n1dh; TREE=$N/tree; RUNS=$N/runs; PY=/root/venv/bin/python; ZG="$TREE/stack/scripts/zerograd_gate.py"
cd $N
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
[ -e n1dh_arm.sh.bzg ] && { echo "n1dh_arm.sh.bzg exists"; exit 1; }
OLD="--control 'route_head\\.weight'--json-out"
NEW="--control 'route_head\\.weight' --json-out"
[ "$(grep -cF -- "$OLD" n1dh_arm.sh)" = 1 ] || { echo "typo not found exactly once"; exit 1; }
cp -p n1dh_arm.sh n1dh_arm.sh.bzg
$PY -I - "$OLD" "$NEW" <<'EOF'
import sys
o, n = sys.argv[1], sys.argv[2]
p = "/root/strat1/n1dh/n1dh_arm.sh"
s = open(p, encoding="utf-8").read()
assert s.count(o) == 1
open(p, "w", encoding="utf-8").write(s.replace(o, n))
EOF
bash -n n1dh_arm.sh
[ "$(diff n1dh_arm.sh n1dh_arm.sh.bzg | grep -c '^[<>]')" = "2" ] || { echo "not exactly one line changed"; exit 1; }
grep -cF -- "$NEW" n1dh_arm.sh
mv n1dh_chain.log n1dh_chain_1f217bf0_zgtypo.log
mv runs/zerograd_N1DH_s0.log runs/zerograd_N1DH_s0_typo.log
echo "$(date -u +%FT%TZ) N1DH Amendment 4: n1dh_arm.sh.bzg copy; one space inserted before --json-out (line 27); chain log -> n1dh_chain_1f217bf0_zgtypo.log; runs/zerograd_N1DH_s0.log -> zerograd_N1DH_s0_typo.log" >> $N/deletions.log
# the arm's remaining lines for N1DH_s0, as the fixed script writes them (lines 27-29), under the lock
tag=N1DH_s0
[ -s $RUNS/$tag/train/ckpt_last.pt ] && [ ! -e $RUNS/$tag/ARM_DONE ] || { echo "s0 is not in the trained-not-done state"; exit 1; }
grep -q "^PASS" $RUNS/post_$tag.log || { echo "post check of s0 did not PASS"; exit 1; }
exec 9>/root/gpu_queue.lock
flock -w 60 9 || { echo "GPU queue lock busy"; exit 1; }
unset CUDA_VISIBLE_DEVICES; export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=$TREE/stack:$TREE $PY $ZG --ckpt $RUNS/$tag/train/ckpt_last.pt --allow '.*' --control 'route_emb\.weight' --control 'route_head\.weight' --json-out $RUNS/$tag/zerograd_n1dh.json > $RUNS/zerograd_$tag.log 2>&1 || { echo "ZZ ZG FAIL ZZ $(tail -c 300 $RUNS/zerograd_$tag.log)"; exit 1; }
tail -n 3 $RUNS/zerograd_$tag.log
$PY $N/kclip.py $RUNS/$tag/train/train_log.jsonl --json > $RUNS/$tag/kclip_$tag.json 2> $RUNS/kclip_$tag.err || { echo "ZZ KCLIP FAIL ZZ $(tail -c 300 $RUNS/kclip_$tag.err)"; exit 1; }
head -c 400 $RUNS/$tag/kclip_$tag.json; echo
echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
echo "ZZ ZG FIX OK ZZ"
