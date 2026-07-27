"""Prove — and then register — that a V2 COMPRESSED cache holds exactly the
parity clip split. This is step 2 of the v5 runbook (*rebuild -> register ->
commit manifest -> train*) for the case the runbook's own text could not cover.

WHY A SECOND SCRIPT AND NOT ``make_parity_manifest.py --record``
---------------------------------------------------------------
``--record`` scans ``ep_*.pt`` and writes ``uid_kind: epcache_basename``. A v2
cache has no ``ep_*.pt`` — its episode identity is the CLIP ID — so ``--record``
finds zero episodes and exits. The two uid spaces need two recorders; what they
share is ``tanitad/data/parity_manifest.json`` and the rule that **the
registration IS the proof**: nothing here can mint an entry for a cache whose
membership was not demonstrated first (``parity.register_v2_geometry_sibling``).

THE THREE MODES
---------------
1. ``--verify-only``  prove membership, write the raw JSON, change nothing.
   Safe on any host, including one that is training. This is the command owed
   when a wide build finishes.

2. (default)          prove membership and PRINT the manifest entry. Nothing is
   written to the manifest — so the entry can be reviewed before it becomes a
   fact the trainers act on.

3. ``--write-manifest`` prove, mint, write ``parity_manifest.json`` in place.
   ⚠️ **STAGE THE DIFF** (`git add stack/tanitad/data/parity_manifest.json`) —
   an entry that exists only on a pod is exactly the stranding failure the
   Agent Operating Standard exists to prevent, and here it is worse than
   stranded work: the trainer on another host will read the cache as
   NON-PARITY and, under ``--require-parity``, refuse to start.

   ``--write-manifest`` REFUSES to overwrite an existing entry unless
   ``--force`` is given, for the same reason ``make_parity_manifest.py
   --record`` does: a truncated cache must never be able to quietly re-record
   itself into a passing manifest.

🔒 CONFIDENTIALITY. Clip ids are gated-confidential PhysicalAI-AV content. This
script reads them (from the cache filenames and from ``--expect-clips``) and
**never prints or stores one** — the JSON it writes and the manifest entry it
mints carry counts and sha256 digests only. That is why the exported clip list
is a pod-side path and not a repo artifact.

EXAMPLES
--------
    # 1. when the wide build finishes on pod2 — prove it, write nothing
    PYTHONPATH=/workspace/TanitAD/stack python3 scripts/register_v2_sibling.py \\
        --verify-only \\
        --cache /workspace/data/pai_wide120_v2png_train \\
        --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \\
        --out /workspace/wfov/v2_parity_verify.json

    # 2. register it (the cache dir must already contain the new key)
    PYTHONPATH=/workspace/TanitAD/stack python3 scripts/register_v2_sibling.py \\
        --cache /workspace/data/physicalai-train-w120x256cyl-<hex> \\
        --new-key physicalai-train-w120x256cyl-<hex> \\
        --expect-clips /workspace/wfov/paritysplit/parity_train_clips.txt \\
        --out /workspace/wfov/v2_sibling_entry.json --write-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.data import parity                                  # noqa: E402


def _geometry(cache_dirs, explicit: str | None) -> dict:
    """The build's own geometry record. ``v2_compressed.py build`` writes
    ``_geometry.json`` before the first decode precisely so a wide cache is
    distinguishable on disk from the deployed square one.

    ⚠️ It is RECORDED here, not verified: membership proves WHICH clips, never
    WHICH PIXELS. A cache built at the wrong FOV with the right clips passes
    this script — the pre-decode assert in the builder
    (``_assert_geometry_deliverable``) is what covers that, and it runs hours
    earlier."""
    if explicit:
        return json.loads(Path(explicit).read_text(encoding="utf-8"))
    for cd in cache_dirs:
        p = Path(cd) / "_geometry.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"_": "NO _geometry.json sidecar found in any --cache dir; this "
                 "cache does not record the frame it was built at"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", required=True,
                    help="the v2 cache dir(s) of *.v2ep.pt. Several dirs are "
                         "treated as ONE corpus (their union), because that is "
                         "how train_flagship4b --v2-cache consumes them.")
    ap.add_argument("--new-key",
                    help="the corpus key to register. MUST appear in the cache "
                         "directory name — parity.corpus_key_of() resolves by "
                         "path substring, so a key that appears nowhere in the "
                         "path is an INERT registration nothing will ever find.")
    ap.add_argument("--expect-clips",
                    help="🔒 pod-side ordered clip-id list exported by "
                         "parity_split_export.py. WITHOUT it this degrades to "
                         "DIGEST-ONLY: a COMPLETE build still verifies exactly, "
                         "but an incomplete one can only be refused, never "
                         "diagnosed, and the 24 legitimate decode failures "
                         "cannot be accepted.")
    ap.add_argument("--source-key", default=parity.PARITY_TRAIN_KEY,
                    help=f"the corpus this is a re-cache OF "
                         f"(default {parity.PARITY_TRAIN_KEY})")
    ap.add_argument("--geometry-json",
                    help="override the _geometry.json read from the cache dir")
    ap.add_argument("--out", help="write the raw proof / entry JSON here")
    ap.add_argument("--verify-only", action="store_true",
                    help="prove membership and stop; mint nothing")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write the entry into parity_manifest.json in place "
                         "(then STAGE the diff)")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default the committed one)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest entry for --new-key")
    a = ap.parse_args(argv)

    if a.verify_only:
        rec = parity.verify_v2_membership(
            a.cache, label="verify-v2", source_key=a.source_key,
            expect_clips=a.expect_clips, manifest_path=a.manifest)
        rec["geometry"] = _geometry(a.cache, a.geometry_json)
        rec["verified_on"] = date.today().isoformat()
        out = rec
    else:
        if not a.new_key:
            ap.error("--new-key is required unless --verify-only is passed")
        ent = parity.register_v2_geometry_sibling(
            a.cache, new_key=a.new_key,
            geometry=_geometry(a.cache, a.geometry_json),
            source_key=a.source_key, expect_clips=a.expect_clips,
            manifest_path=a.manifest)
        ent["provenance"]["registered_on"] = date.today().isoformat()
        ent["provenance"]["evidence_class"] = (
            "MEASURED (live scan of the built cache + set-diff against the "
            "exported parity clip split)" if a.expect_clips else
            "MEASURED (live scan of the built cache; DIGEST-ONLY — no clip "
            "list was supplied, so this can only have passed on a COMPLETE "
            "build)")
        out = ent
        if a.write_manifest:
            mp = Path(a.manifest) if a.manifest else parity.MANIFEST_PATH
            man = json.loads(mp.read_text(encoding="utf-8"))
            prev = man.get("corpora", {}).get(a.new_key)
            if prev and not a.force:
                raise SystemExit(
                    f"[v2-sibling] {a.new_key} is ALREADY registered "
                    f"({prev.get('episode_count')} episodes, digest "
                    f"{str(prev.get('episode_uid_sha256'))[:12]}…). "
                    f"Re-registering would let a truncated cache overwrite a "
                    f"good manifest. Pass --force only if you have verified "
                    f"the cache is the one that should own this key.")
            man.setdefault("corpora", {})[a.new_key] = ent
            mp.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
            print(f"[v2-sibling] WROTE {mp}\n"
                  f"[v2-sibling] ⚠️ NOW STAGE IT:  git add "
                  f"stack/tanitad/data/parity_manifest.json\n"
                  f"[v2-sibling] An entry that lives only on this host makes "
                  f"the cache read NON-PARITY everywhere else — and under "
                  f"--require-parity the trainer will refuse to start.")
        else:
            print("[v2-sibling] entry NOT written (no --write-manifest). "
                  "Review it, then re-run with --write-manifest.")

    txt = json.dumps(out, indent=1, ensure_ascii=False, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"[v2-sibling] proof -> {a.out}")
    print("V2_SIBLING " + txt)
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
