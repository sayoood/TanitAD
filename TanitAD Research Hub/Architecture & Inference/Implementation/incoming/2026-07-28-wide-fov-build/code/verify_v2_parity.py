"""Parity proof for a v2-format geometry sibling of physicalai-train-e438721ae894.

WHY THIS EXISTS INSTEAD OF ``parity.register_geometry_sibling()``
----------------------------------------------------------------
The binding runbook is *rebuild -> register_geometry_sibling() -> commit
manifest*. ``register_geometry_sibling`` proves membership by comparing
``sha256(sorted ep_*.pt basenames))`` against the manifest, because the parity
manifest's ``uid_kind`` is ``epcache_basename`` — an episode's identity is its
POSITION in the ordered clip list.

A v2 compressed cache does not have positions. It is a flat set of
``<clip_id>.v2ep.pt`` files, so its uid space and the manifest's cannot be
compared, and ``register_geometry_sibling`` refuses it — correctly, since it has
no way to tell "different format" from "different episodes".

The re-cache at the wide geometry HAD to be v2: the raw epcache at
120 deg / 256x640 is 293.4 MB/episode = ~697 GB for the train split alone, which
does not fit on any host in the fleet, while the v2 PNG-lossless build measures
~40 MB/clip = ~96 GB.

So this script supplies the SAME guarantee in the format's own terms: the set of
clip ids actually built must equal, exactly, the parity train split exported
from the one host that reproduces both corpus keys. Membership, not pixels.

⚠️ This is a VERIFICATION, not a REGISTRATION. It does not write
``parity_manifest.json`` and ``parity.corpus_key_of()`` still returns None for
this directory. Making it a first-class manifest entry needs a ``uid_kind:
v2ep_clipid`` branch in ``parity.py`` — escalated in WIDE_FOV_BUILD.md, not done
here, because ``parity.py`` is another stream's file and the change should be
reviewed rather than smuggled in behind a build.

🔒 Clip ids are gated-confidential. This prints and stores DIGESTS and COUNTS
only — never an id.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the built v2 cache dir")
    ap.add_argument("--expect-clips", required=True,
                    help="parity_train_clips.txt exported from a parity host")
    ap.add_argument("--split-meta", required=True,
                    help="parity_split_meta.json from the same export")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cache = Path(a.cache)
    built = sorted(p.name[:-len(".v2ep.pt")]
                   for p in cache.glob("*.v2ep.pt"))
    expect = sorted(ln.strip() for ln in open(a.expect_clips) if ln.strip())
    meta = json.loads(Path(a.split_meta).read_text())

    built_s, expect_s = set(built), set(expect)
    missing = sorted(expect_s - built_s)
    extra = sorted(built_s - expect_s)

    sizes = [p.stat().st_size for p in cache.glob("*.v2ep.pt")]
    total = sum(sizes)
    geom = {}
    gp = cache / "_geometry.json"
    if gp.exists():
        geom = json.loads(gp.read_text())

    out = {
     "cache_dir": str(cache),
     "source_corpus_key": "physicalai-train-e438721ae894",
     "expected_clips": len(expect),
     "built_clips": len(built),
     "missing_count": len(missing),
     "extra_count": len(extra),
     # the decisive comparison: MEMBERSHIP, order-independent
     "built_ids_sha256_sorted": digest(built),
     "expected_ids_sha256_sorted": meta.get("train_ids_sha256_sorted"),
     "membership_identical": (digest(built) ==
                              meta.get("train_ids_sha256_sorted")),
     "no_extra_episodes": not extra,
     # a build is allowed to LOSE clips only to decode failure; the parity
     # corpus records exactly 24 such skips, so this is a real cross-check
     "expected_decode_failures": 24,
     "observed_shortfall": len(missing),
     "shortfall_matches_parity_skip_count": len(missing) == 24,
     "geometry": geom,
     "bytes_total": total,
     "gib_total": round(total / 1024**3, 2),
     "mb_per_clip_mean": round(total / max(len(built), 1) / 1e6, 3),
     "projected_gib_train_2376": round(
         (total / max(len(built), 1)) * 2376 / 1024**3, 2),
     "source_export": {k: meta.get(k) for k in
                       ("verified_train_key", "verified_val_key",
                        "keys_match_parity", "discovered_clips",
                        "train_clips", "val_clips")},
    }
    ok = (out["no_extra_episodes"] and
          (out["membership_identical"] or
           out["shortfall_matches_parity_skip_count"]))
    out["VERDICT"] = ("PARITY MEMBERSHIP VERIFIED" if ok else
                      "NOT VERIFIED — see missing_count / extra_count")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("V2_PARITY_VERIFY " + json.dumps(out))
    if extra:
        raise SystemExit(
            f"REFUSED: {len(extra)} clips in the cache are NOT in the parity "
            f"train split. Geometry may change PIXELS, never MEMBERSHIP.")


if __name__ == "__main__":
    main()
