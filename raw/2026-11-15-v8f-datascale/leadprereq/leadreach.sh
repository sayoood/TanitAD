#!/bin/bash
# leadreach.sh -- lead-head reach proof on the 10k union files (G-LIVE lead_head_probe via the trainer's --selfcheck-only), UE_s1's data argv, tree = $D/tree (cdbff971).
# 0 training: --selfcheck-only exits before the loop. Takes /root/gpu_queue.lock (blocking) -> starts only after REPS1 and K-TREE released it.
# W = 3.362 : the frozen LLR weight (GOALS_AND_CLAIMS.md:17139 @cdbff971: "--lead-head on --w-lead 3.362 --aux-cond token"; rule PREREG_LEVELHEAD_A2.md:16 = lead term a tenth of the total at step 0). Two wirings: add (trainer default) and token (LLR's).
# Sentinels: ZZ LEADREACH DONE ZZ / ZZ LEADREACH FAIL ZZ <reason>
D=/root/dscale10k; O=$D/leadreach; PY=/root/venv/bin/python; T=$D/tree
say() { echo "$(date -u +%T) $*"; }
fail() { echo "ZZ LEADREACH FAIL ZZ $1"; exit 1; }
mkdir -p $O || fail "mkdir"
say "LEADREACH queued (waiting on /root/gpu_queue.lock)"
exec 9>/root/gpu_queue.lock || fail "cannot open lock"
flock 9 || fail "flock"
say "got the lock"
[ -e $D/chains/STOP ] && fail "STOP file"
n=0; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; do n=$((n+1)); [ $n -gt 180 ] && fail "GPU not idle"; sleep 10; done
[ -s $D/merged/lead_union.npz ] && [ -s $D/merged/join_union.npz ] && [ -s $D/argv/UE_s1.A.json ] || fail "missing inputs"
for mode in add token; do
  rm -f $O/$mode.done
  $PY - $D/argv/UE_s1.A.json $O/argv_$mode.json $O/$mode $mode <<'E' || fail "argv build $mode"
import json, sys
t = [x for x in json.load(open(sys.argv[1]))["argv"] if x != "--fit-join"]      # join_union.npz exists; this run must not rewrite it
assert t[t.index("--join-train") + 1] == "/root/dscale10k/merged/join_union.npz" and t[t.index("--lead-train") + 1] == "/root/dscale10k/merged/lead_union.npz"
t[t.index("--out") + 1] = sys.argv[3]
t += ["--lead-head", "on", "--w-lead", "3.362"]
if sys.argv[4] == "token": t += ["--aux-cond", "token"]
t += ["--selfcheck-only"]
json.dump({"argv": t}, open(sys.argv[2], "w"))
E
  say "selfcheck $mode"
  ( cd $T && PYTHONPATH=$T/stack:$T/stack/scripts:$T $PY $T/stack/scripts/v8f_mini_train.py $($PY -c "import json;print(' '.join(json.load(open('$O/argv_$mode.json'))['argv']))") > $O/$mode.log 2>&1 ); rc=$?
  [ $rc -eq 0 ] || fail "selfcheck $mode rc=$rc: $(tail -c 500 $O/$mode.log | tr '\n' ' ')"
  [ -s $O/$mode/selfcheck.json ] || fail "$mode wrote no selfcheck.json"
  touch $O/$mode.done
  say "$mode: $(tail -n 1 $O/$mode.log | cut -c1-300)"
done
echo "ZZ LEADREACH DONE ZZ"
