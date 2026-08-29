"""Push the validated 61.6 GB camera bank into the private B1 dataset.

PI instruction (2026-08-29): "You should push the resulting data set (the
augmented one with alpamayo, tact/strategic labels and ego data + 120deg front
camera) as private data set to my hf account."

⛔ HF QUOTA IS A HARD CEILING (binding). The API exposes no readable
storage-usage figure for this account, so headroom is NOT measured -- it is
inferred (HF Pro allowance vs ~64 GB for this repo after the push). Therefore:
**any quota/403/payment signal ABORTS the run and reports.** It is never retried,
never worked around, and spend is never escalated autonomously.

4,719 files: `upload_large_folder` batches commits and resumes across restarts.
One commit per file would be 4,719 commits and is not an option.
"""
import truststore; truststore.inject_into_ssl()

import re
import sys
import time
from pathlib import Path

import huggingface_hub as H

SRC = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
REPO = "Sayood/tanitad-v7-training-corpus"
# ⚠️ These must be PHRASES, never bare status digits. MEASURED 2026-08-30: the
# first version matched "403" and fired "QUOTA/BILLING SIGNAL" on a **401
# Unauthorized** — because the error text carries a Request ID whose hex happens
# to contain "403". That is the CLAUDE.md monitor-matches-its-own-echo trap, and
# class C83 (an instrument that invents the defect it reports) committed by the
# very guard meant to protect the quota rule. Status codes are read from the
# exception's response, not grepped out of prose.
QUOTA_PHRASES = ("quota", "storage limit", "payment required", "billing",
                 "storage quota", "exceeded your", "insufficient storage")
QUOTA_CODES = (402, 413)


def out(*a):
    print(*a)
    sys.stdout.flush()


def keys(tries=150):
    for i in range(tries):
        try:
            return open("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt",
                        encoding="utf-8", errors="ignore").read()
        except OSError:
            time.sleep(min(60, 4 * (i + 1)))
    raise RuntimeError("Keys.txt unreadable")


tok = re.search(r"hf_[A-Za-z0-9]+", keys()).group(0)
api = H.HfApi(token=tok)

info = api.repo_info(REPO, repo_type="dataset")
assert info.private, "REFUSING to push: repo is not private"
mp4s = sorted(SRC.glob("*.mp4"))
tot = sum(p.stat().st_size for p in mp4s)
assert len(mp4s) == 4719, f"expected 4719 mp4, found {len(mp4s)}"
out(f"repo PRIVATE ok | {len(mp4s)} files, {tot/1e9:.2f} GB")

t0 = time.time()
try:
    # ⛔ api.upload_folder, NOT the module-level H.upload_*: the token lives on
    # the HfApi instance. MEASURED 2026-08-30: calling the module function sent
    # the request UNAUTHENTICATED and died 401 on api/repos/create, seconds in,
    # while repo_info on the same token had just succeeded two lines above.
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(SRC),
                      path_in_repo="camera", allow_patterns=["*.mp4"])
except Exception as e:                                      # noqa: BLE001
    msg = f"{type(e).__name__}: {e}"
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in QUOTA_CODES or any(s in msg.lower() for s in QUOTA_PHRASES):
        out(f"\n⛔ QUOTA/BILLING SIGNAL (status {code}) — ABORTING, NOT RETRYING: "
            f"{msg[:300]}")
        out("HF quota is a hard ceiling; spend is the PI's decision. Reporting.")
        sys.exit(3)
    out(f"upload error (status {code}, NOT quota): {msg[:400]}")
    sys.exit(1)

# --- verify by CONTENT: the remote must hold every file at the right size ----
remote = {s.rfilename: (getattr(s, "size", 0) or 0)
          for s in api.repo_info(REPO, repo_type="dataset",
                                 files_metadata=True).siblings}
missing = [p.name for p in mp4s if f"camera/{p.name}" not in remote]
wrong = [p.name for p in mp4s
         if f"camera/{p.name}" in remote
         and remote[f"camera/{p.name}"] != p.stat().st_size]
out(f"\nremote listing: {len(remote)} files | missing {len(missing)} | "
    f"size-mismatch {len(wrong)}")
if missing or wrong:
    out(f"  e.g. missing {missing[:3]} wrong {wrong[:3]}")
    out("CAMERA PUSH INCOMPLETE")
    sys.exit(1)
out(f"CAMERA PUSH VERIFIED — {len(mp4s)} files, {tot/1e9:.2f} GB, "
    f"{(time.time()-t0)/60:.0f} min")
out(f"RELEASE: https://huggingface.co/datasets/{REPO} (private)")
