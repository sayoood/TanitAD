"""Export the canonical train/val CLIP-ID assignment + its digests.

The parity key ``physicalai-train-e438721ae894`` is a hash of the ORDERED
discovered clip ids, so the assignment is only derivable on a host holding the
complete raw corpus. This exports it once, verifying both keys before it writes
anything, so a host that CANNOT reproduce the corpus can still be given the
exact membership as data.

Emits digests as well as the lists. 🔒 The clip ids are gated-confidential and
must stay on pods; only the DIGESTS are quotable in a repo artifact.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

PARITY_TRAIN_KEY = "e438721ae894"
PARITY_VAL_KEY = "0c5f7dac3b11"


def _source_id(s):
    return "clip:" + s["clip_id"]


def cache_key(sources, params):
    ids = [_source_id(s) for s in sources]
    return hashlib.sha1(json.dumps({"ids": ids, "params": params},
                                   sort_keys=True,
                                   default=str).encode()).hexdigest()[:12]


def main() -> None:
    root = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, sys.argv[3])
    from tanitad.data.physicalai import discover_r0_clips, split_clips

    clips = discover_r0_clips(root)
    params = {"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2"}
    tr, va = split_clips(clips, val_frac=0.2, seed=0)
    tk, vk = cache_key(tr, params), cache_key(va, params)
    ok = (tk == PARITY_TRAIN_KEY and vk == PARITY_VAL_KEY)
    if not ok:
        raise SystemExit(
            f"REFUSING to export: this host does not reproduce the canonical "
            f"corpus (train {tk} != {PARITY_TRAIN_KEY} or val {vk} != "
            f"{PARITY_VAL_KEY}). {len(clips)} clips discovered.")

    def digest(ids):
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()

    tr_ids = [c["clip_id"] for c in tr]
    va_ids = [c["clip_id"] for c in va]
    all_ids = [c["clip_id"] for c in clips]
    (outdir / "parity_train_clips.txt").write_text("\n".join(tr_ids) + "\n")
    (outdir / "parity_val_clips.txt").write_text("\n".join(va_ids) + "\n")
    (outdir / "parity_all_clips.txt").write_text("\n".join(all_ids) + "\n")
    meta = {
        "verified_train_key": tk, "verified_val_key": vk,
        "keys_match_parity": ok,
        "discovered_clips": len(clips), "train_clips": len(tr_ids),
        "val_clips": len(va_ids),
        # ordered digest = the split ORDER (what the cache key hashes)
        "train_ids_sha256_ordered": digest(tr_ids),
        "val_ids_sha256_ordered": digest(va_ids),
        "all_ids_sha256_ordered": digest(all_ids),
        # sorted digest = MEMBERSHIP only, order-independent — the right check
        # for a v2 cache, whose files are a set, not an ordered list
        "train_ids_sha256_sorted": digest(sorted(tr_ids)),
        "val_ids_sha256_sorted": digest(sorted(va_ids)),
        "all_ids_sha256_sorted": digest(sorted(all_ids)),
        "build_params": params,
        "note": "clip ids are gated-confidential; only these digests are "
                "quotable outside a pod",
    }
    (outdir / "parity_split_meta.json").write_text(json.dumps(meta, indent=1))
    print("PARITY_SPLIT_EXPORT " + json.dumps(meta))


if __name__ == "__main__":
    main()
