"""Verify, from the artifacts that OWN the facts, the two premises the v2
membership check rests on. Neither is quoted from prose.

Premise 1 — **the parity TRAIN split is 2 400 CLIPS, not 2 376.**
    Source of truth: ``…/2026-07-28-wide-fov-build/raw/parity_split_meta_2026-07-27.json``,
    written by ``parity_split_export.py`` on pod1, which REFUSES to write unless
    the host reproduces both canonical corpus keys first.

Premise 2 — **the skip indices live in the SAME positional space as the episode
    uids, and together they tile the split exactly.**
    Source of truth: ``stack/tanitad/data/parity_manifest.json``.
    ⭐ This is the load-bearing one and it is NOT recorded anywhere else: the
    identity-of-the-24 check in ``parity.verify_v2_membership`` indexes the
    ordered clip list at the skip positions, which is only valid if those
    positions index that same 2 400-long list. ``epcache.build_episodes_cached``
    writes ``ep_%05d.pt`` / ``skip_%05d`` from ``enumerate(sources)``
    (``epcache.py:104-106``), so the tiling below is the check that the manifest
    agrees with that loop over the real corpus.

🔒 Reads digests and indices only. No clip ids.

    python verify_clip_membership_facts.py --out raw/membership_facts_2026-07-27.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data import parity                                  # noqa: E402

EXPORT = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
          "Implementation" / "incoming" / "2026-07-28-wide-fov-build" / "raw" /
          "parity_split_meta_2026-07-27.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = json.loads(EXPORT.read_text(encoding="utf-8"))
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    cm = parity.clip_membership_of(parity.PARITY_TRAIN_KEY)
    uids = list(ent["episode_uids"])
    skips = sorted(int(i) for i in ent["skip_indices"])
    idx = sorted(int(re.match(r"ep_(\d+)\.pt", u).group(1)) for u in uids)
    tiles = sorted(idx + skips) == list(range(len(idx) + len(skips)))

    out = {
        "what": "the two premises behind parity.py §9's v2 membership check, "
                "each read from the artifact that owns it",
        "when": date.today().isoformat(),
        "evidence_class": "MEASURED (committed artifacts, read directly; no "
                          "prose, no summary)",
        "premise_1_clips_are_not_episodes": {
            "source": str(EXPORT.relative_to(REPO)).replace("\\", "/"),
            "discovered_clips": meta["discovered_clips"],
            "train_clips": meta["train_clips"],
            "val_clips": meta["val_clips"],
            "verified_train_key": meta["verified_train_key"],
            "verified_val_key": meta["verified_val_key"],
            "keys_match_parity": meta["keys_match_parity"],
            "train_key_equals_PARITY_TRAIN_KEY": (
                parity.PARITY_TRAIN_KEY.endswith(meta["verified_train_key"])),
            "val_key_equals_PARITY_VAL_KEY": (
                parity.PARITY_VAL_KEY.endswith(meta["verified_val_key"])),
            "val_clips_equals_PARITY_VAL_EPISODES": (
                meta["val_clips"] == parity.PARITY_VAL_EPISODES),
            "reading": "3000 discovered -> split_clips(val_frac=0.2) -> 2400 "
                       "train / 600 val. A check written against 2400-as-total "
                       "or 2376-as-clips is wrong in opposite directions.",
        },
        "premise_2_skip_indices_tile_the_split": {
            "source": "stack/tanitad/data/parity_manifest.json",
            "episode_count": ent["episode_count"],
            "n_episode_uids": len(uids),
            "uid_index_min": idx[0], "uid_index_max": idx[-1],
            "n_skip_indices": len(skips),
            "skip_index_first": skips[0], "skip_index_last": skips[-1],
            "union_is_exactly_0_to_N_minus_1": tiles,
            "N": len(idx) + len(skips),
            "N_equals_train_clips": (len(idx) + len(skips)
                                     == meta["train_clips"]),
            "reading": "⭐ the 2376 episode positions and the 24 skip positions "
                       "tile 0..2399 with no gap and no overlap. That is the "
                       "independent confirmation that (a) the split really is "
                       "2400 clips and (b) skip_%05d indexes the SAME ordered "
                       "list as ep_%05d.pt — which is exactly what the "
                       "identity-of-the-24 shortfall check assumes.",
        },
        "premise_3_order_equals_sort": {
            "train_ordered_digest": meta["train_ids_sha256_ordered"],
            "train_sorted_digest": meta["train_ids_sha256_sorted"],
            "identical": (meta["train_ids_sha256_ordered"]
                          == meta["train_ids_sha256_sorted"]),
            "reading": "the discovered clip order already IS clip-id order, so "
                       "a SET membership proof (all the v2 format can offer) is "
                       "exactly as strong here as an ordered one. ⚠️ that is a "
                       "property of THIS corpus, not a general guarantee.",
        },
        "manifest_agrees_with_the_export": {
            "manifest_clip_id_sha256_sorted": cm["clip_id_sha256_sorted"],
            "export_train_ids_sha256_sorted": meta["train_ids_sha256_sorted"],
            "identical": (cm["clip_id_sha256_sorted"]
                          == meta["train_ids_sha256_sorted"]),
            "manifest_n_clips": cm["n_clips"],
            "export_train_clips": meta["train_clips"],
        },
    }
    checks = {
        "premise_1": out["premise_1_clips_are_not_episodes"]["keys_match_parity"]
        and out["premise_1_clips_are_not_episodes"]["val_clips_equals_PARITY_VAL_EPISODES"],
        "premise_2": tiles and out["premise_2_skip_indices_tile_the_split"]["N_equals_train_clips"],
        "premise_3": out["premise_3_order_equals_sort"]["identical"],
        "manifest_matches_export": out["manifest_agrees_with_the_export"]["identical"],
    }
    out["ALL_PASS"] = all(checks.values())
    out["checks"] = checks
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
