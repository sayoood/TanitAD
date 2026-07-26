"""Pod-side gated HF backup of flagship-v4-fromscratch's FINAL ckpt.pt.

WHY: metrics.json milestone_archives = [5000,10000,15000,20000]. There is NO 25k
and NO 30k milestone. The final state of a 59.0 h from-scratch run exists ONLY in
/workspace/experiments/flagship-v4-fromscratch/ckpt.pt on ONE disk.

INVARIANTS (do not remove):
  1. gated="manual" is VERIFIED via repo_info BEFORE a single weight byte is sent.
     Repo is created EMPTY, gated, re-read, and only then uploaded to.
  2. Token read from a tmpfs file (RAM, mode 600), deleted on first read.
     Never printed, never in argv, never on a real disk.
  3. ckpt.pt is opened READ-ONLY. Never copied, moved, renamed or truncated.
  4. md5 + sha256 computed locally BEFORE upload; after upload the HF-side LFS
     sha256 is read back from the repo tree and compared -> end-to-end byte proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from io import BytesIO

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi  # noqa: E402

REPO = "Sayood/flagship-v4-fromscratch"
BASE = "/workspace/experiments/flagship-v4-fromscratch"

# (src, dst_in_repo, is_weight)
FILES = [
    (f"{BASE}/ckpt.pt", "ckpt.pt", True),
    (f"{BASE}/ckpt_step20000.pt", "ckpt_step20000.pt", True),
    (f"{BASE}/config.json", "config.json", False),
    (f"{BASE}/metrics.json", "metrics.json", False),
    (f"{BASE}/train_log.jsonl", "train_log.jsonl", False),
]

CARD = b"""---
license: other
extra_gated_prompt: >-
  TanitAD research checkpoint. Access is granted manually by the repo owner.
tags: [tanitad, flagship, world-model, autonomous-driving, from-scratch]
---

# flagship-v4-fromscratch -- FINAL 30k checkpoint (step 29999)

The full 59.0 h (212,544.6 s wallclock) from-scratch v4 run: joint planner +
world-model training with the lambda_plan seam ramped to `lam_mult_final = 1.0`.

`ckpt.pt` is the FINAL state at `final_step 29999`. It is archived here because
the trainer's `milestone_archives` are [5000, 10000, 15000, 20000] ONLY -- there
is no 25k and no 30k milestone, so before this push the finished run existed on a
single pod disk.

## MEASURED (metrics.json, in-loop, n=881 val windows)

| quantity | value |
|---|---|
| `final_step` | 29999 |
| `canary_ade@2s` (WM integrity) | 1.1409 |
| `canary_baseline` | 15.6742 |
| in-loop `val ade@2s` (DENSE-20) | 0.5063 |
| in-loop `val oracle_ade@2s` | 0.1892 |
| in-loop `val miss@2m` | 0.2145 |
| `lam_mult_final` | 1.0 |

The from-scratch WM canary descended to 1.1409 against a 15.6742 baseline
THROUGH full planner coupling -- that descent is the v4 thesis.

## READ THIS BEFORE QUOTING A NUMBER

The in-loop `ade@2s` above is the trainer's **DENSE-20** statistic (mean over 20
dense steps 0.1-2.0 s). It is **NOT** comparable to the historical
`ade_0_2s`/`g_op_fwd_ade_m` convention, which is the **4-waypoint** mean over
steps 5/10/15/20. On the same forward pass at 15k this arm read dense-20 0.4596
vs 4-waypoint 0.5839. A metric NAME is not a metric DEFINITION
(TanitAD RETRACTION_LOG C1). Quote `eval_flagship_v4.py` output only.

`ckpt_step20000.pt` is the last archived milestone before the finish.

Pushed from tanitad-pod2, gated (manual approval), 2026-07-26.
"""


def digests(path: str) -> tuple[str, str, int]:
    """md5 + sha256 in ONE read-only streaming pass. Never mutates the file."""
    m, s, n = hashlib.md5(), hashlib.sha256(), 0
    with open(path, "rb") as f:          # 'rb' -- read only, no truncation
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            m.update(b); s.update(b); n += len(b)
    return m.hexdigest(), s.hexdigest(), n


def main() -> int:
    tokp = os.environ.get("HF_TOKEN_FILE", "/dev/shm/hf_tok")
    try:
        with open(tokp) as f:
            tok = f.read().strip()
    finally:
        try:
            os.remove(tokp)
        except OSError:
            pass
    if not tok.startswith("hf_"):
        print("FATAL: no hf_ token in token file", flush=True)
        return 3

    api = HfApi(token=tok)

    # ---- STEP 1: create EMPTY, then gate, then VERIFY. No bytes yet. --------
    api.create_repo(repo_id=REPO, repo_type="model", private=False, exist_ok=True)
    print(f"REPO_CREATED_OR_EXISTS {REPO}", flush=True)

    api.update_repo_settings(repo_id=REPO, repo_type="model", gated="manual")
    info = api.repo_info(repo_id=REPO, repo_type="model", files_metadata=False)
    gated = getattr(info, "gated", None)
    print(f"GATED_READBACK gated={gated!r} private={getattr(info,'private',None)!r}", flush=True)
    if str(gated).lower() != "manual":
        print("FATAL: gated is not 'manual' -- REFUSING to upload any weight byte", flush=True)
        return 4
    print("GATE_VERIFIED_BEFORE_ANY_WEIGHT_BYTE", flush=True)

    # ---- STEP 2: local digests (read-only) ---------------------------------
    local: dict[str, dict] = {}
    for src, dst, is_w in FILES:
        if not os.path.exists(src):
            print("SKIP_MISSING", src, flush=True)
            continue
        md5, sha, n = digests(src)
        local[dst] = {"md5": md5, "sha256": sha, "bytes": n, "weight": is_w}
        print(f"LOCAL {dst} bytes={n} md5={md5} sha256={sha}", flush=True)

    # ---- STEP 3: upload ----------------------------------------------------
    for src, dst, _ in FILES:
        if dst not in local:
            continue
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=REPO, repo_type="model")
        print(f"UPLOADED {dst}", flush=True)
    api.upload_file(path_or_fileobj=BytesIO(CARD), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print("UPLOADED README.md", flush=True)

    # ---- STEP 4: end-to-end verification from the HF side -------------------
    paths = list(local.keys())
    infos = api.get_paths_info(repo_id=REPO, repo_type="model", paths=paths)
    remote: dict[str, dict] = {}
    for pi in infos:
        lfs = getattr(pi, "lfs", None)
        sha = None
        if lfs is not None:
            sha = getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
        remote[pi.path] = {"size": getattr(pi, "size", None), "sha256": sha,
                           "blob_id": getattr(pi, "blob_id", None)}

    ok = True
    verdicts = {}
    for dst, L in local.items():
        R = remote.get(dst, {})
        size_ok = (R.get("size") == L["bytes"])
        if R.get("sha256"):
            sha_ok = (R["sha256"] == L["sha256"])
            how = "lfs_sha256"
        else:
            sha_ok = size_ok            # small non-LFS files: size + local md5 recorded
            how = "size_only(non-LFS)"
        verdicts[dst] = {"size_match": size_ok, "content_match": sha_ok, "method": how,
                         "local_md5": L["md5"], "local_sha256": L["sha256"],
                         "remote_sha256": R.get("sha256"), "bytes": L["bytes"],
                         "remote_size": R.get("size")}
        print(f"VERIFY {dst} size_match={size_ok} content_match={sha_ok} via={how} "
              f"remote_sha256={R.get('sha256')}", flush=True)
        ok = ok and size_ok and sha_ok

    out = {"repo": REPO, "url": f"https://huggingface.co/{REPO}", "gated": str(gated),
           "all_verified": ok, "files": verdicts}
    with open("/workspace/tmp/v4_hf_push_receipt.json", "w") as f:
        json.dump(out, f, indent=2)
    print("RECEIPT /workspace/tmp/v4_hf_push_receipt.json", flush=True)
    print("PUSH_RESULT " + ("ALL_VERIFIED" if ok else "MISMATCH"), flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
