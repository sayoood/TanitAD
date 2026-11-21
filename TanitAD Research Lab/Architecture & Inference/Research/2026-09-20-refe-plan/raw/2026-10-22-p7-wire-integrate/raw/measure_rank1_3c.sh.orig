#!/bin/bash
# PENDING STEP -- NOT RUN by P7-WIRE-RESERVE (the 3-C bank was still being rebuilt on the pod; the brief forbids fetching it).
# It measures the rank-1 exclusion and the derived tuple count of `train_grow_cmd_3c_p7`, the number the MM pins in launch_gate (a P6_BANK_TUPLES-style entry) BEFORE step 0.
#
# Precondition: bank3c_hf.sh has printed ZZB3_OK, i.e. HF dataset Sayood/tanitad-refe-navtrain-banks holds refe_train_grow_cmd_3c.tar.zst (tar of train_grow_cmd_3c/).
# Where: any box with python + huggingface_hub + zstd + tar and the landed refe tree; CPU only; ~1 min after the download.  Needs HF_TOKEN_FILE = a file that holds the token; it is read IN PLACE by
#        the python child, never echoed, never an argument.  HF_HUB_DISABLE_XET=1 (uploads / downloads hang in CloseWait under hf_xet).
# Run:   HF_TOKEN_FILE=<path to Keys.txt> PKG=<tree holding refe/ and raw/2026-10-22-p7-session-reserve/> bash measure_rank1_3c.sh
#
# What success looks like (the numbers are the MEASUREMENT; the first two are checks of the INPUT, not results):
#   ZZR1_DOWNLOAD_OK <size> <sha256[:16]>         remote listing (size + LFS sha256) == the downloaded bytes
#   rank-0 / rank-1 file sha256 prefixes daa23b96 / 2a63acc2 and 167,300 rows   (PREREG_3C.addendum6; if they differ, STOP: it is another bank)
#   filter JSON: n_in 167300, by_rank_in {"0": 103037, "1": 64263}, by_rank_excluded {"0": 10141, "1": <THE RANK-1 NUMBER>}, n_out = 92896 + (64263 - rank-1 excluded)
#   token_log_checked true: every row's log_name equals its token's log in the token table (a twin carries its parent's log)
# Then: pin n_out as the tuple count of the bank directory name `train_grow_cmd_3c_p7`, and targets_sha256 as its data pin, in the P7 launch-gate profile.
set -eu
: "${HF_TOKEN_FILE:?HF_TOKEN_FILE: a file holding the HF token (read in place, never printed)}"
: "${PKG:?PKG: the refe-plan tree}"
W="${W:-$PWD/rank1_measure}"; mkdir -p "$W"
NAME=refe_train_grow_cmd_3c.tar.zst; REPO=Sayood/tanitad-refe-navtrain-banks
export HF_HUB_DISABLE_XET=1 HF_TOKEN_FILE NAME REPO W
python - <<'PY'
import hashlib, os, re
from huggingface_hub import HfApi, hf_hub_download
tok = re.search(r"hf_[A-Za-z0-9]+", open(os.environ["HF_TOKEN_FILE"], encoding="utf-8").read()).group(0)
api = HfApi(token=tok)
info = list(api.get_paths_info(os.environ["REPO"], [os.environ["NAME"]], repo_type="dataset", expand=True))[0]
p = hf_hub_download(os.environ["REPO"], os.environ["NAME"], repo_type="dataset", token=tok, local_dir=os.environ["W"])
h, n = hashlib.sha256(), 0
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b); n += len(b)
assert info.lfs is not None and n == info.size and h.hexdigest() == info.lfs.sha256, "remote listing != downloaded bytes: do not use this file"
print("ZZR1_DOWNLOAD_OK", n, h.hexdigest()[:16])
PY
zstd -d -c "$W/$NAME" | tar -x -C "$W"
B="$W/train_grow_cmd_3c"
sha256sum "$B"/targets_rank*.jsonl | cut -c1-8,65-      # expect daa23b96 ... targets_rank0.jsonl, 2a63acc2 ... targets_rank1.jsonl (prefixes only)
cat "$B"/targets_rank*.jsonl | wc -l                      # expect 167300
python "$PKG/refe/session_reserve.py" filter \
  --reserve "$PKG/refe/fixtures/p7_reserve_sessions.json" \
  --bank "$B" --out "$W/train_grow_cmd_3c_p7" \
  --token-table "$PKG/raw/2026-10-22-p7-session-reserve/raw/src/navtrain_token_table.tsv.gz" | tee "$W/rank1_filter.json"
echo "report: $W/train_grow_cmd_3c_p7/reserve_filter_report.json  (per-file counts, per-session counts, source and output targets_sha256)"
