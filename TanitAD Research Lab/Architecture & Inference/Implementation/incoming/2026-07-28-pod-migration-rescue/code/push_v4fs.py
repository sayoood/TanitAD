"""Back up the flagship-v4 FROM-SCRATCH 30k final checkpoint to HF.

WHY: it is the completed 30k arm (step 29,999) and it exists on EXACTLY ONE DISK
(pod3:/workspace/v4instr/v4fs_ckpt.pt). LOOP_STATE carries "CKPT-BACKUP STANDING RISK"
for precisely this, and today that risk looked real for an hour: pod2's experiments dir
is empty and I briefly concluded the run was lost. It was not — it had been copied to
pod3 under a different NAME (v4fs_*), which a *flagship-v4* glob cannot match.

Repo is created PRIVATE: this is a backup, not a publication. Private is strictly more
restrictive than the program's existing gated-public pattern, so it cannot over-expose.

Token is read in place from Keys.txt and never printed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

from huggingface_hub import HfApi

CKPT = pathlib.Path("/workspace/v4instr/v4fs_ckpt.pt")
CFG = pathlib.Path("/workspace/v4instr/v4fs_config.json")
REPO = "Sayood/tanitad-flagship-v4-fromscratch-30k"


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    tok = re.search(r"hf_[A-Za-z0-9]+", pathlib.Path("/workspace/TanitAD/Keys.txt").read_text(errors="ignore")).group(0)
    api = HfApi(token=tok)

    if not CKPT.exists():
        print(f"FATAL: {CKPT} absent")
        return 1
    size = CKPT.stat().st_size
    print(f"ckpt {CKPT}  {size:,} B", flush=True)
    digest = md5(CKPT)
    print(f"md5(local) = {digest}", flush=True)

    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    print(f"repo ready (PRIVATE): {REPO}", flush=True)

    card = f"""---
tags: [tanitad, flagship, world-model, autonomous-driving]
---

# TanitAD flagship-v4 FROM-SCRATCH — 30k final

**Backup of the completed v4 from-scratch arm.** Random-init trunk (no v1 warm-start);
world model and anchored-diffusion planner co-evolve.

| field | value |
|---|---|
| step | **29,999 / 30,000** (report as step-29999) |
| parity train corpus | `physicalai-train-e438721ae894` |
| skip-hash | `f09e44db` |
| bytes | {size:,} |
| md5 | `{digest}` |
| source | `pod3:/workspace/v4instr/v4fs_ckpt.pt` |

⚠️ **The formal 8-metric gate has NOT been run on this checkpoint yet.** Its card is
`Project Steering/Gates/flagship-v4-30k.card.json`, registered at step 29,650 — i.e.
*before* this checkpoint existed, so no threshold was chosen after seeing a number.

⚠️ **`ade_0_2s` on this line is `wm_fidelity_ade_2s`** — the world model is handed the
expert's TRUE future actions. v1's **0.4271** is therefore a fidelity reference, **not a
planning bar**, and must not be used as one.

Private backup, not a publication.
"""
    (pathlib.Path("/tmp/README.md")).write_text(card, encoding="utf-8")

    print("uploading config + card ...", flush=True)
    if CFG.exists():
        api.upload_file(path_or_fileobj=str(CFG), path_in_repo="config.json", repo_id=REPO, repo_type="model")
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id=REPO, repo_type="model")

    print("uploading ckpt (this is the big one) ...", flush=True)
    api.upload_file(path_or_fileobj=str(CKPT), path_in_repo="ckpt.pt", repo_id=REPO, repo_type="model")

    files = api.list_repo_files(REPO, repo_type="model")
    print("repo files:", files, flush=True)
    print(json.dumps({"repo": REPO, "bytes": size, "md5": digest, "step": 29999}, indent=1))
    print("UPLOAD_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
