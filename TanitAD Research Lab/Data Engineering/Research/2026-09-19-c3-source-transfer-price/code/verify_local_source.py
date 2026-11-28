"""E17 / C3 — are the 4,713 train clips' SOURCE FRAMES already on the dev box?

PREREG_REFCV6_DEVBOX_PREPARATION.md constant 13 (MEASURED 2026-09-18, du/df) states
"There is no train corpus and no source-frame corpus on this box" — measured over
`D:\\Projects\\TanitAD-artifacts` ONLY. CLAUDE.md operating standard 2: absence found
at ONE location is not absence. A second-location probe (find over C:\\Users\\Admin
and D:\\) found 4,719 mp4s in C:\\Users\\Admin\\tanitad-data\\physicalai\\camera\\
camera_front_wide_120fov — exactly the v7 corpus count.

This script decides whether those files ARE the corpus, bit for bit, against an
INDEPENDENT source: the per-file size and LFS sha256 that the private HF dataset
`Sayood/tanitad-v7-training-corpus` publishes in its own file metadata. Nothing is
downloaded — the comparison uses repo METADATA only.

It also derives the 6 clips the parity ingest gate drops (the corpus's intersection
with the deployed val40, via `deployed_val40_clip_digests.json`) and checks the
remaining 4,713 against the B1 membership digest pinned in `parity_manifest.json`
(`physicalai-b1-w120-256x640cyl`). That digest is the positive control: a wrong
exclusion set cannot reproduce it.

🔒 Clip ids are gated-confidential. Output carries sha12 = sha256(id)[:12] only.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("D:/Projects/TanitAD")
LOCAL = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
EGO = Path("C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo")
HF_REPO = "Sayood/tanitad-v7-training-corpus"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verify_local_source.json")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def s12(cid):
    return sha256_hex(cid)[:12]


# ── 1. the local set ─────────────────────────────────────────────────────────
local = {}
for p in glob.glob(str(LOCAL / "*.mp4")):
    cid = os.path.basename(p)[:-4]
    assert UUID.match(cid), "unexpected mp4 name shape"
    local[cid] = p
ts_local = {os.path.basename(p)[:36] for p in glob.glob(str(LOCAL / "*.timestamps.parquet"))}
ego_local = {os.path.basename(p)[:36] for p in glob.glob(str(EGO / "*.parquet"))}
print("local mp4: %d   timestamps: %d   egomotion parquets: %d"
      % (len(local), len(ts_local), len(ego_local)), flush=True)

# ── 2. the 6 the parity gate drops, and the B1 membership control ────────────
v40 = json.load(open(REPO_ROOT / "stack/tanitad/data/deployed_val40_clip_digests.json",
                     encoding="utf-8"))
assert v40["digest_algorithm"] == "sha256(clip_id.encode('utf-8')).hexdigest()"
v40set = set(v40["clip_id_digests"])
drop = sorted(c for c in local if sha256_hex(c) in v40set)
train = sorted(c for c in local if sha256_hex(c) not in v40set)
pm = json.load(open(REPO_ROOT / "stack/tanitad/data/parity_manifest.json", encoding="utf-8"))
b1 = pm["corpora"]["physicalai-b1-w120-256x640cyl"]
b1_digest = b1["clip_membership"]["clip_id_sha256_sorted"]
got_digest = sha256_hex("\n".join(train))
b1_ok = (got_digest == b1_digest) and len(train) == b1["episode_count"]
print("val40 digests: %d   corpus clips inside val40: %d   train set: %d"
      % (len(v40set), len(drop), len(train)))
print("B1 membership digest reproduced: %s" % b1_ok, flush=True)

# ── 3. HF metadata: size + LFS sha256 per camera mp4 (no content downloaded) ──
import truststore  # noqa: E402
truststore.inject_into_ssl()
tok = re.search(r"hf_[A-Za-z0-9]+",
                (REPO_ROOT / "Keys.txt").read_text(encoding="utf-8", errors="replace")).group(0)
from huggingface_hub import HfApi  # noqa: E402
api = HfApi(token=tok)
info = api.repo_info(HF_REPO, repo_type="dataset", files_metadata=True)
hf = {}
for s in info.siblings:
    m = re.match(r"^camera/([0-9a-f-]{36})\.mp4$", s.rfilename)
    if not m:
        continue
    lfs = s.lfs
    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    hf[m.group(1)] = (s.size, sha)
print("HF revision %s   camera mp4s in metadata: %d   with sha256: %d"
      % (info.sha[:12], len(hf), sum(1 for v in hf.values() if v[1])), flush=True)

only_local = sorted(set(local) - set(hf))
only_hf = sorted(set(hf) - set(local))
both = sorted(set(local) & set(hf))
size_bad = [c for c in both if os.path.getsize(local[c]) != hf[c][0]]

# ── 4. full content hash of every local mp4, timed (a MEASURED local read rate) ─
sha_bad, nbytes, t0 = [], 0, time.time()
for i, c in enumerate(both):
    h = hashlib.sha256()
    with open(local[c], "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
            nbytes += len(chunk)
    if hf[c][1] is None or h.hexdigest() != hf[c][1]:
        sha_bad.append(c)
    if (i + 1) % 500 == 0:
        print("  hashed %d/%d  %.1f GB  %.0f MB/s"
              % (i + 1, len(both), nbytes / 1e9, nbytes / 1e6 / (time.time() - t0)), flush=True)
hash_s = time.time() - t0

sz = {c: os.path.getsize(local[c]) for c in local}
res = {
    "local_dir": str(LOCAL),
    "hf_repo": HF_REPO,
    "hf_revision": info.sha,
    "n_local_mp4": len(local),
    "n_local_timestamps": len(ts_local),
    "n_local_egomotion_parquets": len(ego_local),
    "timestamps_cover_train": all(c in ts_local for c in train),
    "egomotion_covers_train": all(c in ego_local for c in train),
    "n_hf_mp4": len(hf),
    "n_hf_with_sha256": sum(1 for v in hf.values() if v[1]),
    "n_both": len(both),
    "only_local_sha12": [s12(c) for c in only_local],
    "only_hf_sha12": [s12(c) for c in only_hf],
    "size_mismatch_sha12": [s12(c) for c in size_bad],
    "sha256_mismatch_sha12": [s12(c) for c in sha_bad],
    "val40_dropped_sha12": [s12(c) for c in drop],
    "n_train": len(train),
    "b1_membership_digest_expected": b1_digest,
    "b1_membership_digest_reproduced": got_digest,
    "b1_membership_ok": b1_ok,
    "bytes_all_4719": sum(sz.values()),
    "bytes_train_4713": sum(sz[c] for c in train),
    "bytes_dropped_6": sum(sz[c] for c in drop),
    "bytes_hashed": nbytes,
    "hash_seconds": round(hash_s, 1),
    "local_read_plus_sha256_mb_s": round(nbytes / 1e6 / hash_s, 1),
    "verdict": ("IDENTICAL" if (not only_local and not only_hf and not size_bad
                                and not sha_bad and b1_ok) else "NOT IDENTICAL"),
}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))
print("mismatch lists:", {k: len(v) for k, v in res.items() if isinstance(v, list)})
