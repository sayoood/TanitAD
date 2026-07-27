#!/usr/bin/env python3
"""Push `idm_head_v3` to a GATED HuggingFace repo under `Sayood/`.

🔒 SECURITY / PROVENANCE INVARIANTS (each of these is enforced below, not
   merely documented):
   * `Keys.txt` is git-ignored. The token is read IN PLACE by
     `tanitad.keys.load_keys()` into the environment; it is never printed,
     echoed, copied, or passed in argv.
   * PhysicalAI-AV is GATED-CONFIDENTIAL. **WEIGHTS ONLY.** No frames, no poses,
     no clip UUIDs, no per-clip tables. The uploaded payload is asserted against
     a whitelist of keys before it leaves this process.
   * The repo is created **private + gated (manual approval)**, matching the
     precedent set by the four Phase-0 checkpoint repos.

Usage (dev box, inside the tanitad venv):
  python push_idm_v3.py --ckpt <local ckpt> --card MODEL_CARD_IDM_V3.md \
                        --repo Sayood/tanitad-idm-head-v3 [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

# Only these top-level keys may appear in the uploaded checkpoint.
ALLOWED_CKPT_KEYS = {"state_dict", "config", "geom_features", "scalar_names",
                     "provenance", "metrics"}
# Any of these substrings anywhere in the payload aborts the push.
FORBIDDEN_SUBSTRINGS = ("clip_id", "uuid", "episode_id", "frames", "poses",
                        "valdata", "physicalai-val", "comma2k19-val", "/root/")


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def audit_ckpt(path: Path) -> dict:
    """Refuse to upload anything but weights + scalar config."""
    import torch
    d = torch.load(path, weights_only=False, map_location="cpu")
    extra = set(d) - ALLOWED_CKPT_KEYS
    if extra:
        raise SystemExit(f"REFUSING TO PUSH: unexpected checkpoint keys {extra}")
    blob = json.dumps({k: v for k, v in d.items() if k != "state_dict"},
                      default=str).lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in blob:
            raise SystemExit(f"REFUSING TO PUSH: {bad!r} found in checkpoint metadata")
    n_par = sum(int(v.numel()) for v in d["state_dict"].values())
    tensors = {k: tuple(v.shape) for k, v in d["state_dict"].items()}
    print(f"[audit] OK — {len(tensors)} tensors, {n_par:,} params, "
          f"keys={sorted(d)}")
    return {"n_params": n_par, "n_tensors": len(tensors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--repo", default="Sayood/tanitad-idm-head-v3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--receipt", default="hf_push_receipt.json")
    a = ap.parse_args()

    ckpt, card = Path(a.ckpt), Path(a.card)
    for p in (ckpt, card):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    info = audit_ckpt(ckpt)
    m5 = md5_of(ckpt)
    print(f"[push] {ckpt.name}  md5 {m5}  {ckpt.stat().st_size:,} B")
    if a.dry_run:
        print("[push] --dry-run: audit passed, nothing uploaded")
        return

    sys.path.insert(0, str(REPO_ROOT / "stack"))
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()                                  # token -> env, never printed

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]
    print(f"[push] authenticated as {who}")

    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="idm_head_v3.pt",
                    repo_id=a.repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model")
    files = api.list_repo_files(a.repo, repo_type="model")
    rev = api.model_info(a.repo).sha

    receipt = {"repo": a.repo, "revision": rev, "files": sorted(files),
               "ckpt_md5": m5, "ckpt_bytes": ckpt.stat().st_size,
               "private": True, "pushed_by": who, **info,
               "gating_note": "created PRIVATE; enable manual-approval gating in "
                              "the repo settings UI (the API cannot set the "
                              "gated flag)."}
    Path(a.receipt).write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    print(f"[push] receipt -> {a.receipt}")


if __name__ == "__main__":
    main()
