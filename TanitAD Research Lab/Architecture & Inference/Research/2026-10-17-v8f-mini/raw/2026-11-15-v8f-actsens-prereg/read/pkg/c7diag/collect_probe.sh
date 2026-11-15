#!/bin/bash
# Collector thread probe: the driver's collect_latents_arm.py (unchanged) on the FIRST 4 of the 141 TRAIN clips of R35_s0 (68 windows; control 4 needs 64),
# once at OMP_NUM_THREADS=4 (the driver's step-3 setting; the collector calls torch.set_num_threads(OMP_NUM_THREADS)) and once at 1. Read-only sftp from box 2, as the driver.
# usage: collect_probe.sh <omp>      output: c7diag/tmp/col<omp>/latents_train_R35_s0.pt
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /f/TanitAD-scratch/actsens/read/env.sh
N=$1
O=F:/TanitAD-scratch/actsens/read/c7diag/tmp/col$N
mkdir -p $O/ts/cache/train/clips
$PY - $O/ts/clips141.json <<'EOF'
import json, hashlib, sys
m = json.load(open("F:/TanitAD-scratch/actsens/read/ts/clips141.json"))
c = m["clips"][:4]
m["clips"] = c; m["clips_sha256"] = hashlib.sha256("\n".join(c).encode()).hexdigest()
for k in ("n_frames", "bytes", "md5"):
    m[k] = {x: m[k][x] for x in c}
json.dump(m, open(sys.argv[1], "w"))
EOF
COL_OUT_DIR=$O COL_TAG=R35_s0 COL_TS=$O/ts COL_CKPT=$PULLS/ladder_R35_s0/ckpt_last.pt OMP_NUM_THREADS=$N $PY -u $RD/collect_latents_arm.py </dev/null > $O/collect.log 2>&1
echo "rc=$? $(date -Iseconds)" > $O/done.txt
