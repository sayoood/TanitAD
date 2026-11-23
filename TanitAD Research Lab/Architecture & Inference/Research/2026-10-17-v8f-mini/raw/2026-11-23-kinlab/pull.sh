#!/usr/bin/env bash
source C:/Users/Admin/lorapod/podenv.sh >/dev/null 2>&1
cd /f/TanitAD-scratch/kinlab/tmp
pscp "$POD_TARGET:/root/rollhist/bank/s0/shard_000of004.npz" . && pscp "$POD_TARGET:/root/dscale10k/merged/native_tracks_union.pt" . ; echo "PULL_RC $?"
sha256sum shard_000of004.npz native_tracks_union.pt
