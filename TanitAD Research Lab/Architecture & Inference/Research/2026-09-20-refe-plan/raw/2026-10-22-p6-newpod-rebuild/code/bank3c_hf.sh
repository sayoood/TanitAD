#!/bin/bash
# Sidecar: once the rebuild driver has merged AND sha-verified the 3-C bank (merge.done), bank it to the private HF dataset. The 3-C bank was never
# banked off the old pod, which is why P6 needed a 23 h rebuild; this closes that. Token read from the token file inside python; never printed.
set -u
O=/workspace/ops/rebuild; TF=<TOKEN_FILE>; REPO=Sayood/tanitad-refe-navtrain-banks; NAME=refe_train_grow_cmd_3c.tar.zst; L=/root/tad
log() { echo "$(date -u +%FT%TZ) $*" >> $O/bank3c_hf.log; }
until [ -f $O/merge.done ]; do [ -f $O/FAILED ] && { log "ZZB3_SKIP driver FAILED"; exit 1; }; sleep 300; done
mkdir -p $L; cd /workspace/data/refe_navtrain || exit 1
tar -cf - train_grow_cmd_3c | zstd -q -T2 -10 -o $L/$NAME -f || { log "ZZB3_FAIL tar"; exit 1; }
SHA=$(sha256sum $L/$NAME | cut -c1-64); SZ=$(stat -c %s $L/$NAME); log "packed $SZ B sha256 $SHA"
source /workspace/teacher_env.sh
HF_HUB_DISABLE_XET=1 TF=$TF REPO=$REPO NAME=$NAME F=$L/$NAME SHA=$SHA SZ=$SZ $DRIVERL_EVAL_PYTHON - >> $O/bank3c_hf.log 2>&1 <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=open(os.environ["TF"]).read().strip())
r, n = os.environ["REPO"], os.environ["NAME"]
api.upload_file(path_or_fileobj=os.environ["F"], path_in_repo=n, repo_id=r, repo_type="dataset",
                commit_message="3-C twin bank train_grow_cmd_3c (rebuilt on the new pod 2026-10-22, rank1 2a63acc2, rank0 daa23b96)")
info = [x for x in api.get_paths_info(r, [n], repo_type="dataset", expand=True)][0]
ok = info.size == int(os.environ["SZ"]) and info.lfs is not None and info.lfs.sha256 == os.environ["SHA"]
print(("ZZB3_OK" if ok else "ZZB3_FAIL remote mismatch"), n, info.size, (info.lfs.sha256[:16] if info.lfs else None))
PY
