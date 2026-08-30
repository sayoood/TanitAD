"""Push the 61.6 GB camera bank in RESUMABLE BATCHES, under a stall watchdog.

Why not `upload_folder` on the whole bank: MEASURED 2026-08-30 it stalled at
**0.0 MB of process IO over 25 s** with 61.6 GB still to move, having staged
~1.2 GB into the xet cache — the same silent 0 B/s hang that hit the 1.97 GB tar
twice. And because it commits ONCE at the end, the remote showed 0 mp4 files the
whole time: **there is no way to tell a working run from a hung one**, and a kill
throws away everything.

⇒ batches of BATCH files, one commit each (~24 commits, not 4,719). Each batch is
skipped when the remote already holds every file at the right size, so a restart
resumes at batch granularity and previous work is never redone.

Run under `push_camera_watchdog.py`, which kills a stalled batch by explicit PID
(never `pkill -f`, which matches its own command line) and relaunches.
"""
import truststore; truststore.inject_into_ssl()

import re
import sys
import time
from pathlib import Path

import huggingface_hub as H

SRC = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
REPO = "Sayood/tanitad-v7-training-corpus"
BATCH = 200
# ⚠️ PHRASES, never bare status digits: the first version matched "403" and fired
# a QUOTA alarm on a 401, because the Request ID hex contained "403". That is the
# monitor-matches-its-own-echo trap, committed by the guard itself (class C83).
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


api = H.HfApi(token=re.search(r"hf_[A-Za-z0-9]+", keys()).group(0))
assert api.repo_info(REPO, repo_type="dataset").private, "REFUSING: repo not private"

mp4s = sorted(SRC.glob("*.mp4"))
assert len(mp4s) == 4719, f"expected 4719, found {len(mp4s)}"
remote = {s.rfilename: (getattr(s, "size", 0) or 0)
          for s in api.repo_info(REPO, repo_type="dataset",
                                 files_metadata=True).siblings}
todo = [p for p in mp4s if remote.get(f"camera/{p.name}") != p.stat().st_size]
out(f"repo PRIVATE ok | {len(mp4s)} files | already on remote "
    f"{len(mp4s)-len(todo)} | to push {len(todo)} "
    f"({sum(p.stat().st_size for p in todo)/1e9:.2f} GB)")
if not todo:
    out("CAMERA PUSH VERIFIED — every file already on the remote at the right size")
    sys.exit(0)

batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
t0 = time.time()
for bi, batch in enumerate(batches, 1):
    gb = sum(p.stat().st_size for p in batch) / 1e9
    out(f"batch {bi}/{len(batches)}: {len(batch)} files, {gb:.2f} GB ...")
    try:
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(SRC),
                          path_in_repo="camera",
                          allow_patterns=[p.name for p in batch],
                          commit_message=f"camera batch {bi}/{len(batches)}")
    except Exception as e:                                  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in QUOTA_CODES or any(s in msg.lower() for s in QUOTA_PHRASES):
            out(f"\n⛔ QUOTA/BILLING SIGNAL (status {code}) — ABORTING, NOT RETRYING: "
                f"{msg[:250]}")
            out("HF quota is a hard ceiling; spend is the PI's decision.")
            sys.exit(3)
        out(f"batch {bi} FAILED (status {code}): {msg[:200]}")
        sys.exit(1)
    el = time.time() - t0
    done_gb = sum(sum(p.stat().st_size for p in b) for b in batches[:bi]) / 1e9
    out(f"  ok — {done_gb:.1f} GB in {el/60:.0f} min ({done_gb*1000/max(el,1):.1f} MB/s)")

# --- verify by CONTENT -------------------------------------------------------
remote = {s.rfilename: (getattr(s, "size", 0) or 0)
          for s in api.repo_info(REPO, repo_type="dataset",
                                 files_metadata=True).siblings}
bad = [p.name for p in mp4s if remote.get(f"camera/{p.name}") != p.stat().st_size]
if bad:
    out(f"INCOMPLETE — {len(bad)} files missing/size-mismatched, e.g. {bad[:3]}")
    sys.exit(1)
out(f"camera bytes verified — 4719 files, "
    f"{sum(p.stat().st_size for p in mp4s)/1e9:.2f} GB, {(time.time()-t0)/60:.0f} min")

# ⛔ THE BUNDLE FILES TOO. MEASURED 2026-08-30: this script's allow_patterns are
# mp4 NAMES, so it pushed 61.6 GB of frames and left the RE-ISSUED MANIFEST.json
# and the new camera/camera_sha256.json behind — the remote kept the SUPERSEDED
# manifest (camera described as "not shipped, retrieve with a tool"; exclusions
# with no parity split) sitting next to the very data it misdescribes. The push
# said VERIFIED because it verified THE FILES IT CHOSE TO SEND.
# ⇒ a completeness claim must be checked against the WHOLE bundle, not against
# the subset the uploader was given. Same class as C83b: the check could not
# report the failure because it never looked there.
BUNDLE = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
drift = []
for p in sorted(BUNDLE.rglob("*")):
    if not p.is_file():
        continue
    rel = str(p.relative_to(BUNDLE)).replace("\\", "/")
    if remote.get(rel) != p.stat().st_size:
        drift.append(rel)
        out(f"  re-uploading drifted bundle file: {rel}")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=REPO,
                        repo_type="dataset", commit_message=f"sync {rel}")
out(f"bundle files checked: {'all in sync' if not drift else f'{len(drift)} re-uploaded'}")
out("CAMERA PUSH VERIFIED — frames AND bundle files match the remote")
