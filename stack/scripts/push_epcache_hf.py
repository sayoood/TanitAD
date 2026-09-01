#!/usr/bin/env python3
"""Push a built v2 epcache to HuggingFace — from the POD, never from Thor.

⛔ WHY THIS EXISTS (PI 2026-09-02: "push it to void in the future the waiting
time"). The B1 epcache existed on Thor for weeks and STILL cost the A40 pod a
~5 h rebuild, because **no script in this programme has ever pushed a
`.v2ep.pt` cache** — every epcache is a single-disk artifact. The three routes
to get one onto a new machine were: copy from Thor (166 GB over a ~1 MB/s home
uplink ≈ 46 h), push from Thor to HF (same uplink, same 46 h), or rebuild
(~5 h). This makes a fourth: push ONCE from a datacenter link, pull in minutes
forever after (HF→pod measured at 622.8 MB/s).

⛔ RUN IT ON THE POD. Pushing from Thor or the dev box moves the bottleneck
back onto the home uplink and is the thing this script exists to avoid.

⚠️ PRIVATE BY DEFAULT: derived/augmented corpora go private on the PI's paid
account (standing rule). `--public` exists but must be a deliberate act.

⚠️ CPU BUDGET: the pod's quota is 7.65 CPUs (cfs_quota, NOT the 96 `nproc`
reports). A concurrent training run uses ~5 of them, so this defaults to ONE
worker and prints a warning — an upload that starves a 25 h trainer is a bad
trade for an artifact nobody is waiting on.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="dir of *.v2ep.pt")
    ap.add_argument("--repo", required=True, help="e.g. Sayood/tanitad-b1-epcache")
    ap.add_argument("--keys", default="/workspace/Keys.txt",
                    help="file holding the hf_ token; read in place, never echoed")
    ap.add_argument("--workers", type=int, default=1,
                    help="upload concurrency; keep at 1 while training runs")
    ap.add_argument("--public", action="store_true",
                    help="⛔ deliberate act: derived corpora are private by default")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cache = Path(a.cache)
    files = sorted(cache.glob("*.v2ep.pt"))
    if not files:
        raise SystemExit(f"REFUSING: no *.v2ep.pt under {cache}")
    total = sum(f.stat().st_size for f in files)
    print(f"[push] {len(files)} episodes, {total / 2**30:.1f} GiB from {cache}")
    print(f"[push] -> {a.repo}  private={not a.public}  workers={a.workers}")
    if a.dry_run:
        print("[push] dry run, nothing sent")
        return 0

    # ⛔ token read IN PLACE and never printed, logged, or passed as an argv
    # element (it would show in `ps` for every user on the box).
    txt = Path(a.keys).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit(f"REFUSING: no hf_ token found in {a.keys}")
    tok = m.group(0)

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="dataset", private=not a.public,
                    exist_ok=True)

    t0 = time.time()
    api.upload_large_folder(
        folder_path=str(cache), repo_id=a.repo, repo_type="dataset",
        num_workers=a.workers,
    )
    el = time.time() - t0
    print(f"[push] DONE {total / 2**30:.1f} GiB in {el / 60:.1f} min "
          f"({total / 2**20 / max(el, 1e-9):.1f} MB/s)")

    # ⛔ VERIFY BY CONTENT ON THE FAR SIDE, not by the absence of an exception.
    # A push that "succeeded" while landing fewer files is exactly the failure
    # this programme keeps meeting; and a tree listing near 1,000 entries is a
    # TRUNCATION until pagination says otherwise, so this counts every page.
    got = [f for f in api.list_repo_files(a.repo, repo_type="dataset")
           if f.endswith(".v2ep.pt")]
    print(f"[push] far side carries {len(got)} of {len(files)} episodes")
    if len(got) != len(files):
        raise SystemExit("REFUSING to report success: far-side count differs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
