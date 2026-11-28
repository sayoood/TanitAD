#!/bin/bash
# box 2 -> pod, piped through the dev box (no local staging). usage: xfer.sh <stream 0|1|2>
source C:/Users/Admin/lorapod/podenv.sh
S=$1
ROOT_B='D:\tanitad'
declare -a LIST
LIST=(
"0|runs\v8f_mini_full\M2b_s0\ckpt_last.pt|M2b_s0/ckpt_last.pt"
"0|runs\v8f_mini_full\M2b_s1\ckpt_last.pt|M2b_s1/ckpt_last.pt"
"0|data\v9\lead_train4572.npz|data/v9/lead_train4572.npz"
"0|data\v9\v9_labels_train.npz|data/v9/v9_labels_train.npz"
"1|runs\v8f_mini_full\dagger_bank\s0\shard_000of004.npz|dagger_bank/s0/shard_000of004.npz"
"1|runs\v8f_mini_full\dagger_bank\s0\shard_001of004.npz|dagger_bank/s0/shard_001of004.npz"
"1|runs\v8f_mini_full\dagger_bank\s0\shard_002of004.npz|dagger_bank/s0/shard_002of004.npz"
"1|runs\v8f_mini_full\dagger_bank\s0\shard_003of004.npz|dagger_bank/s0/shard_003of004.npz"
"2|runs\v8f_mini_full\dagger_bank\s1\shard_000of004.npz|dagger_bank/s1/shard_000of004.npz"
"2|runs\v8f_mini_full\dagger_bank\s1\shard_001of004.npz|dagger_bank/s1/shard_001of004.npz"
"2|runs\v8f_mini_full\dagger_bank\s1\shard_002of004.npz|dagger_bank/s1/shard_002of004.npz"
"2|runs\v8f_mini_full\dagger_bank\s1\shard_003of004.npz|dagger_bank/s1/shard_003of004.npz"
)
SMALL=(
"runs\v8f_mini_full\dagger_bank\s0\bank.json|dagger_bank/s0/bank.json"
"runs\v8f_mini_full\dagger_bank\s0\PROVENANCE.json|dagger_bank/s0/PROVENANCE.json"
"runs\v8f_mini_full\dagger_bank\s1\bank.json|dagger_bank/s1/bank.json"
"runs\v8f_mini_full\dagger_bank\s1\PROVENANCE.json|dagger_bank/s1/PROVENANCE.json"
)
for e in "${LIST[@]}"; do
  IFS='|' read -r st src dst <<< "$e"
  [ "$st" = "$S" ] || continue
  echo "$(date +%T) START $dst"
  pssh "mkdir -p /root/strat1/inputs/$(dirname $dst)"
  ssh -q tanitad-box2 "type $ROOT_B\\$src" 2>/dev/null | pssh "cat > /root/strat1/inputs/$dst.part && mv /root/strat1/inputs/$dst.part /root/strat1/inputs/$dst"
  echo "$(date +%T) END $dst rc=$?"
done
if [ "$S" = "0" ]; then
  for e in "${SMALL[@]}"; do
    IFS='|' read -r src dst <<< "$e"
    pssh "mkdir -p /root/strat1/inputs/$(dirname $dst)"
    ssh -q tanitad-box2 "type $ROOT_B\\$src" 2>/dev/null | pssh "cat > /root/strat1/inputs/$dst"
  done
  pssh 'mkdir -p /root/strat1/inputs/p10; cp -n /root/runs/bprime/join_train.npz /root/strat1/inputs/join_train.npz; cp -n /root/runs/bprime/p10/p10_prior.pt /root/strat1/inputs/p10/p10_prior.pt; cp -n /root/data/cache_pod/native_tracks.pt /root/strat1/inputs/native_tracks.pt'
fi
echo "ZZ XFER $S DONE ZZ"
