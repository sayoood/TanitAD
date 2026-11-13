#!/bin/bash
# dev-box: copy stage-A files to the pod and verify each by sha256 on both sides. Prints no address.
S=/f/TanitAD-scratch/v8proof/pod/stageA
L=/c/Users/Admin/lorapod/ship2
P="bash /c/Users/Admin/v8pod/pssh.sh"
$P 'mkdir -p /workspace/v8proof/stageA/code /workspace/v8proof/stageA/ops /workspace/v8proof/data/inputs /workspace/v8proof/data/v9 /workspace/v8proof/hierrep' >/dev/null
bad=0
put() { # put <local> <remote>
  $P "cat > $2" < "$1" >/dev/null
  a=$(sha256sum "$1" | cut -d' ' -f1)
  b=$($P "sha256sum $2" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "OK   $2 ${a:0:12}"; else echo "BAD  $2 local ${a:0:12} pod ${b:0:12}"; bad=1; fi
}
for f in v2ep_pod.py cache_pod.py pull_encode_pooled.py; do put $S/code/$f /workspace/v8proof/stageA/code/$f; done
for f in stageA_fetch.py stageA_final.py; do put $S/$f /workspace/v8proof/stageA/code/$f; done
for f in train4572.txt clips141.txt banked_hashes.json intrinsics_dev.csv; do put $S/ops/$f /workspace/v8proof/stageA/ops/$f; done
put $S/stageA.sh /workspace/v8proof/stageA/stageA.sh
for f in join_train.npz native_tracks.pt p10_prior.pt; do put $L/$f /workspace/v8proof/data/inputs/$f; done
$P 'chmod +x /workspace/v8proof/stageA/stageA.sh'
exit $bad
