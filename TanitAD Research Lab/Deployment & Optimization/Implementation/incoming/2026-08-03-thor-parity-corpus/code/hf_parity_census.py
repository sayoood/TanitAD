#!/usr/bin/env python3
"""Prove — from the HF listing alone, BEFORE moving a single byte — that the
raw parity epcache mirrored to HuggingFace is the canonical corpus.

WHY THIS EXISTS. The transfer plan for Thor is "pull 278.78 GB from HF, then
check parity". That is backwards: a 50-minute download is the *expensive* half,
and ``parity.check_uids`` runs on the SET OF FILENAMES, which the HF tree API
already gives us for free. So the parity verdict is available in ~2 s and the
transfer only starts if it PASSES.

⚠️ WHAT THIS PROVES AND WHAT IT DOES NOT
  proves     — the remote directory holds EXACTLY the manifest's episode uid set
               (count + sha256 over the sorted ``ep_%05d.pt`` basenames), and the
               skip markers match the manifest's 24 recorded decode failures.
  proves     — the total byte size is consistent with the committed frame count
               (472,627 frames x 9x256x256 uint8), an INDEPENDENT cross-check of
               the same corpus through a different quantity.
  does NOT   — prove the tensor CONTENTS. ``parity.py``'s own docstring says the
               digest "does NOT hash episode CONTENT (tensor bytes)". A file of
               the right name and the right size can still be corrupt (the
               programme has a live example: Thor's val ``ep_00028.pt`` at
               92,299,264 B against a true 117,383,256 B). Byte integrity is the
               DESTINATION-side check (``verify_thor_epcache.py``), by size AND
               by ``torch.load``, never by exit code.

Usage:
    python hf_parity_census.py --repo Sayood/tanitad-physicalai-w120-256x640cyl \
        --path epcache-256px-phase0/physicalai-train-e438721ae894 \
        --corpus physicalai-train-e438721ae894 --out census_train.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT / "stack"))

from tanitad.data import parity  # noqa: E402

PER_FRAME_BYTES = 9 * 256 * 256          # uint8 [T, 9, 256, 256] epcache frame


def hf_token() -> str:
    """Read the token IN PLACE from the git-ignored Keys.txt. Never printed,
    never written into an argv, never copied to another file."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    txt = (REPO_ROOT / "Keys.txt").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m:
        raise SystemExit("no hf_ token found in Keys.txt")
    return m.group(0)


def census(repo: str, path: str, token: str) -> dict:
    try:
        import truststore
        truststore.inject_into_ssl()      # certifi fails behind the dev-box proxy
    except Exception:                     # noqa: BLE001
        pass
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files: dict[str, int] = {}
    for it in api.list_repo_tree(repo, repo_type="dataset", recursive=True,
                                 path_in_repo=path):
        sz = getattr(it, "size", None)
        if sz is None:                    # a directory entry
            continue
        files[os.path.basename(it.path)] = int(sz)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Sayood/tanitad-physicalai-w120-256x640cyl")
    ap.add_argument("--path", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--expect-frames", type=int, default=0,
                    help="committed total frame count for the byte cross-check "
                         "(train: 472627 from parity_manifest cross_checks)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    files = census(a.repo, a.path, hf_token())
    eps = {k: v for k, v in files.items() if parity.episode_index(k) is not None}
    skips = sorted(int(m.group(1)) for k in files
                   if (m := re.match(r"^skip_(\d+)$", k)))
    other = sorted(set(files) - set(eps) - {f"skip_{i:05d}" for i in skips})

    rec = {
        "repo": a.repo, "path_in_repo": a.path, "corpus_key": a.corpus,
        "n_files": len(files), "n_episodes": len(eps),
        "n_skip_markers": len(skips), "other_files": other,
        "total_bytes": sum(files.values()),
        "episode_bytes": sum(eps.values()),
        "min_episode_bytes": min(eps.values()) if eps else None,
        "max_episode_bytes": max(eps.values()) if eps else None,
    }

    # -- the parity content check, on the remote uid set ---------------------- #
    try:
        rec["parity"] = parity.check_uids(
            sorted(eps), corpus_key=a.corpus, label=f"HF {a.repo}:{a.path}",
            cache_dir=f"hf://{a.repo}/{a.path}", mode="strict")
        rec["parity_verdict"] = "PASS"
    except SystemExit as e:               # ParityViolation subclasses SystemExit
        rec["parity_verdict"] = "REFUSED"
        rec["parity_refusal"] = str(e)

    # -- skip-marker check (the manifest records the 24 decode failures) ------ #
    ent = parity.manifest_entry(a.corpus) or {}
    exp_skips = sorted(int(i) for i in ent.get("skip_indices", []))
    rec["skip_indices_observed"] = skips
    rec["skip_indices_expected"] = exp_skips
    rec["skip_match"] = (skips == exp_skips)

    # -- INDEPENDENT byte cross-check ---------------------------------------- #
    if a.expect_frames:
        exp = a.expect_frames * PER_FRAME_BYTES
        rec["bytes_expected_from_frame_count"] = exp
        rec["bytes_observed_minus_expected"] = rec["episode_bytes"] - exp
        rec["bytes_overhead_per_episode"] = round(
            (rec["episode_bytes"] - exp) / max(len(eps), 1), 1)
        rec["byte_crosscheck_within_1pct"] = abs(
            rec["episode_bytes"] - exp) / exp < 0.01

    print(json.dumps({k: v for k, v in rec.items() if k != "parity_refusal"},
                     indent=2, default=str))
    if rec.get("parity_refusal"):
        print(rec["parity_refusal"], file=sys.stderr)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=2, default=str),
                               encoding="utf-8")
    return 0 if rec["parity_verdict"] == "PASS" and rec["skip_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
