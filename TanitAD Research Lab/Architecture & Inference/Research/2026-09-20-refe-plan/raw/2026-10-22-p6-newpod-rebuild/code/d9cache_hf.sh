#!/bin/bash
# Sidecar: bank the D9 metric cache (refe_navtrain/metric_cache, 107,860 files, 36.5 GB, rebuilt on the new pod 2026-10-22 in 5.2 CPU-h)
# to the private HF dataset, so a pod stop does not cost the rebuild again (MM ruling Q5, SPEC_P7). Plain tar: the parts are already .pkl.xz.
# Idle I/O priority; token read from the token file inside python, never printed.
set -u
O=/workspace/ops/rebuild; TF=<TOKEN_FILE>; REPO=Sayood/tanitad-refe-navtrain-banks; NAME=refe_d9_metric_cache_newpod_2026-10-22.tar; L=/root/tad
log() { echo "$(date -u +%FT%TZ) $*" >> $O/d9cache_hf.log; }
exec 9> $O/d9cache_hf.lock; flock -n 9 || { log "ZZD9_SKIP another instance holds the lock"; exit 1; }
mkdir -p $L; cd /workspace/data/refe_navtrain || exit 1
# wait until the chain has stopped writing into the cache: no build_metric_cache.py process and a stable file count over 60 s, twice
s=0; while [ $s -lt 2 ]; do
  if [ "$(ps -eo args | grep -c '[b]uild_metric_cache\.py')" = 0 ]; then a=$(find metric_cache -type f | wc -l); sleep 60; b=$(find metric_cache -type f | wc -l); [ "$a" = "$b" ] && s=$((s+1)) || s=0; else s=0; sleep 120; fi
done; log "cache stable (no build_metric_cache.py, file count steady twice over 60 s)"
N=$(find metric_cache -type f | wc -l); log "packing metric_cache files=$N"
tar -cf $L/$NAME metric_cache || { log "ZZD9_FAIL tar"; exit 1; }
M=$(tar -tf $L/$NAME | grep -vc '/$'); [ "$M" = "$N" ] || { log "ZZD9_FAIL member count $M != files $N"; exit 1; }
SHA=$(sha256sum $L/$NAME | cut -c1-64); SZ=$(stat -c %s $L/$NAME); log "packed $SZ B members $M sha256 $SHA"
source /workspace/teacher_env.sh
HF_HUB_DISABLE_XET=1 TF=$TF REPO=$REPO NAME=$NAME F=$L/$NAME SHA=$SHA SZ=$SZ N=$N $DRIVERL_EVAL_PYTHON - >> $O/d9cache_hf.log 2>&1 <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=open(os.environ["TF"]).read().strip())
r, n = os.environ["REPO"], os.environ["NAME"]
api.upload_file(path_or_fileobj=os.environ["F"], path_in_repo=n, repo_id=r, repo_type="dataset",
                commit_message=f"D9 metric cache, {os.environ['N']} parts (rebuilt on the new pod 2026-10-22; P6 3-C resume)")
info = [x for x in api.get_paths_info(r, [n], repo_type="dataset", expand=True)][0]
ok = info.size == int(os.environ["SZ"]) and info.lfs is not None and info.lfs.sha256 == os.environ["SHA"]
print(("ZZD9_OK" if ok else "ZZD9_FAIL remote mismatch"), n, info.size, (info.lfs.sha256[:16] if info.lfs else None))
PY
grep -q ZZD9_OK $O/d9cache_hf.log && rm -f $L/$NAME
