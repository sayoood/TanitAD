#!/usr/bin/env bash
# stage_ip10k.sh : dev-box side staging for card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE). Started by the lead. Runs no model and no GPU job.
#   0. the pod folder of this card (no probe file; a full volume shows as a failed copy in step 3)
#   1. code tree of commit cdbff971 (stack/tanitad and stack/scripts, bytes as in git) -> $W/tree, four files checked by sha256
#   2. the two frozen 10k checkpoints from the private dataset (pull_ue.py on the pod; sha256 checked there)
#   3. the evaluation cache of the other session's reads (141 clips, 2.8 GB) as a tar stream -> $W/eval_cache/eval, checked against its manifest
# The pod address is never printed; pod text goes through the token filter. Last line: ZZ STAGE-IP10K DONE or ZZ STAGE-IP10K STOP <step>.
set -u
P=/c/Users/Admin/v8pod/pssh.sh
M=/c/Users/Admin/tanitad-push
C=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
W=/workspace/v8proof/integ10k
SRC=/e/Projects/TanitAD-artifacts/sfresh_inputs/cache
H=/f/TanitAD-scratch/v8proof/integ10k
MAN=78631f832d73afdb2d68794bdad1d3649f694d528565486fcb2ef60afaff202d
pod() { timeout "$1" bash $P "$2" 2>/dev/null | sed '/hf_/d'; }
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ STAGE-IP10K STOP $1"; exit 4; }
echo "[$(ts)] stage start"
# 0
r=$(pod 120 "mkdir -p $W && echo MKDIR-OK")
echo "$r" | grep -q 'MKDIR-OK' || stop "0 pod folder"
echo "[$(ts)] no probe file is written: a full volume shows as a failed copy, and step 3 checks every clip against the manifest"
# 1
git -C $M -c core.autocrlf=false archive $C stack/tanitad stack/scripts | timeout 1800 bash $P "mkdir -p $W/tree && tar -x -C $W/tree" 2>/dev/null | sed '/hf_/d'
for f in stack/scripts/v8f_t1_eval.py stack/scripts/v8f_mini_train.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py; do
  want=$(git -C $M -c core.autocrlf=false show $C:$f | sha256sum | cut -c1-64)
  got=$(pod 60 "sha256sum $W/tree/$f | cut -c1-64" | tr -d '\r\n ')
  [ "$want" = "$got" ] || stop "1 tree file $f"
done
echo "[$(ts)] tree: four files equal to commit $(echo $C | cut -c1-8); files on the pod: $(pod 60 "find $W/tree -type f | wc -l" | tr -d '\r\n ')"
# 2
for f in pull_ue.py verify_cache.py; do
  timeout 120 bash $P "cat > $W/$f" < $H/$f 2>/dev/null
  [ "$(sha256sum $H/$f | cut -c1-64)" = "$(pod 60 "sha256sum $W/$f | cut -c1-64" | tr -d '\r\n ')" ] || stop "2 push $f"
done
r=$(pod 1800 "cd $W && HF_HUB_DISABLE_XET=1 /workspace/venv/bin/python $W/pull_ue.py $W/ckpt 2>&1 | grep -a 'pulled\|PULL-FAILED\|ZZ PULL-UE'")
echo "$r" | cut -c1-160
echo "$r" | grep -q 'ZZ PULL-UE DONE' || stop "2 checkpoints"
# 3
if pod 60 "[ -e $W/eval_cache/eval/manifest.jsonl ] && echo HAVE" | grep -q HAVE; then echo "[$(ts)] evaluation cache folder exists on the pod: copy skipped, verify only"; else
  echo "[$(ts)] evaluation cache copy start"
  tar -C $SRC -cf - eval | timeout 14000 bash $P "mkdir -p $W/eval_cache && tar -x -C $W/eval_cache" 2>/dev/null | sed '/hf_/d'
  echo "[$(ts)] evaluation cache copy end"
fi
r=$(pod 1800 "/workspace/venv/bin/python $W/verify_cache.py $W/eval_cache/eval $MAN 2>&1 | tail -n 3")
echo "$r" | cut -c1-200
echo "$r" | grep -q 'ZZ VERIFY-CACHE DONE' || stop "3 evaluation cache"
echo "[$(ts)] on the pod: $(pod 60 "du -sm $W | cut -f1" | tr -d '\r\n ') MB under the stage folder"
echo "ZZ STAGE-IP10K DONE $(ts)"
