"""v5.8f RELEASE BUNDLE — collect the artifact set, verify it, push it to HF.

The release is defined by a MANIFEST that names every file with its md5 and the
registry section that quotes it. Nothing is uploaded that the manifest does not
name, and the manifest is uploaded LAST (the same protocol the pod↔pod relay
uses) so a consumer never sees a half-published release.

Refuses to publish when a REQUIRED artifact is missing — a partial release that
looks complete is worse than none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

REPO_DEFAULT = "Sayood/tanitad-flagship-v5f-w120"
PREFIX_DEFAULT = "release/v58f"

# (local path, repo path, required?, what quotes it)
ARTIFACTS = [
    ("experiments/stage-a-predictor/ckpt_stage_a.pt",
     "ckpt/ckpt_stage_a.pt", True, "registry §1.13c — the repaired trunk"),
    ("experiments/stage-a-predictor/stage_a_gate.json",
     "gates/stage_a_gate.json", True, "registry §1.13c"),
    ("experiments/w4r-unicycle-head-stagea/unicycle_emission.pt",
     "ckpt/w4r_unicycle_emission.pt", True, "registry §1.14 — W4r head"),
    ("experiments/w4r-unicycle-head-stagea/w4_gate.json",
     "gates/w4r_gate.json", True, "registry §1.14 — W4r PASS"),
    ("experiments/w4-unicycle-head-c/unicycle_emission.pt",
     "ckpt/w4_unicycle_emission_frozen_trunk.pt", False, "registry §1.13 — W4"),
    ("experiments/p8-occupancy-c/p8_gate.json",
     "gates/p8_gate_attempt2.json", True, "registry §1.14 — P8 retention 0.932"),
    ("experiments/w7-full-roll/w7_gate.json",
     "gates/w7_full_gate.json", False, "registry §1.14 — selector-free W7"),
    ("experiments/w7-repaired-w4r-k32/w7_gate.json",
     "gates/w7_w4r_k32_gate.json", False, "registry §1.14"),
    ("experiments/i4a/flagship-v5f-w120-30k-i4a-none.json",
     "gates/i4a_none.json", False, "registry §1.14 — imagination intact"),
    ("experiments/i4a/flagship-v5f-w120-30k-i4a-zero.json",
     "gates/i4a_zero.json", False, "registry §1.14 — imagination zeroed"),
    ("experiments/i4a/flagship-v5f-w120-30k-i4a-shuffle.json",
     "gates/i4a_shuffle.json", False, "registry §1.14 — imagination shuffled"),
    ("experiments/t1-v58f/t1_summary.json",
     "gates/t1_summary.json", False, "registry §1.14 — T1 pseudo-closed-loop"),
    ("experiments/p8-occupancy-c/reel/p8_belief_reel.mp4",
     "media/p8_belief_reel.mp4", False, "the WM's believed scene (I1c)"),
    ("experiments/p8-occupancy-c/reel/p8_belief_still.png",
     "media/p8_belief_still.png", False, "I1c still"),
]

CARD = """---
license: other
tags: [autonomous-driving, world-model, jepa, tanitad]
---

# TanitAD v5.8f — release bundle

Self-supervised hierarchical latent world model for driving (sub-300 M params,
no perception labels, no maps, no reward in any trunk loss).

**This bundle is evidence, not a leaderboard entry.** Every gate JSON here is the
raw artifact behind a row in `Project Steering/MODEL_REGISTRY.md` §1.13–§1.14 of
the source repository, including the gates that FAILED — the failures are part of
the record and are what the design decisions were made from.

## What is in it

| path | what |
|---|---|
| `ckpt/ckpt_stage_a.pt` | the repaired trunk (predictor-only post-training; action-response gain 0.27 → 0.97) |
| `ckpt/w4r_unicycle_emission.pt` | the unicycle emission head refit on that trunk (fan oracle 0.1273, violations 0) |
| `gates/*.json` | every pre-registered gate with its measured verdict |
| `media/*` | the decoded-BEV belief reel: camera │ what the world model believes │ belief ∩ truth |

## Reading the numbers

Tier stamps are binding: **T0** is a world-model diagnostic and is never "driving
performance"; **T1** (action-closed) is the primary capability tier. Intervals are
episode-cluster bootstraps over the 40 validation episodes. See `MANIFEST.json`
for the registry section that quotes each file.
"""


def md5(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("release_v58f")
    ap.add_argument("--root", default="/workspace",
                    help="prefix for the artifact paths")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--prefix", default=PREFIX_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    present, missing_required, missing_optional = [], [], []
    for rel, dst, required, why in ARTIFACTS:
        p = os.path.join(a.root, rel)
        if os.path.exists(p):
            present.append((p, dst, why))
        elif required:
            missing_required.append(rel)
        else:
            missing_optional.append(rel)

    print(f"[release] present {len(present)} · missing-optional "
          f"{len(missing_optional)} · missing-REQUIRED {len(missing_required)}",
          flush=True)
    for m in missing_optional:
        print(f"[release]   optional absent: {m}", flush=True)
    if missing_required:
        for m in missing_required:
            print(f"[release]   REQUIRED ABSENT: {m}", flush=True)
        raise SystemExit("[release] refusing to publish a partial release")

    manifest = {
        "release": "tanitad-v5.8f",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_of_truth": "Project Steering/MODEL_REGISTRY.md §1.13–§1.14",
        "tier_note": "T0 = world-model diagnostic, never driving performance; "
                     "T1 = action-closed, the primary capability tier",
        "estimator_note": "decision-grade intervals are episode-cluster "
                          "bootstraps over the 40 val episodes",
        "files": {},
    }
    for p, dst, why in present:
        manifest["files"][dst] = {"md5": md5(p), "bytes": os.path.getsize(p),
                                  "quoted_by": why}
        print(f"[release]   {dst}  {manifest['files'][dst]['bytes']>>10} KiB",
              flush=True)

    if a.dry_run:
        print(json.dumps(manifest, indent=1)[:1500], flush=True)
        print("RELEASE_DRYRUN_OK", flush=True)
        return 0

    from huggingface_hub import HfApi
    tok = open("/root/.cache/huggingface/token").read().strip()
    api = HfApi(token=tok)
    for p, dst, _why in present:
        print(f"[release] upload {dst}", flush=True)
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=f"{a.prefix}/{dst}", repo_id=a.repo)
    card = os.path.join("/tmp", "v58f_release_card.md")
    open(card, "w").write(CARD)
    api.upload_file(path_or_fileobj=card,
                    path_in_repo=f"{a.prefix}/README.md", repo_id=a.repo)
    mpath = os.path.join("/tmp", "v58f_release_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    api.upload_file(path_or_fileobj=mpath,                 # LAST, by protocol
                    path_in_repo=f"{a.prefix}/MANIFEST.json", repo_id=a.repo)
    print("RELEASE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
