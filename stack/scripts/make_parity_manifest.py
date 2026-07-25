"""Generate / verify ``tanitad/data/parity_manifest.json`` — the committed
episode manifest every trainer asserts against (``tanitad.data.parity``).

Three modes, in order of authority:

1. ``--from-profile-csv <path>``  (the path used to mint the CURRENT committed
   train entry, on the dev box, with no pod access)

   Derives the train uid set from a committed pod-side scan of the canonical
   cache. The scan used is
   ``TanitAD Research Hub/Data Engineering/Implementation/incoming/
     2026-07-25-v2-corpus-qa/parity_profile.csv``
   — 2 376 rows, one per ``ep_*.pt``, produced by ``parity_profile.py`` reading
   ``…/_epcache/physicalai-train-e438721ae894`` (``parity_profile.json``
   ``cache_dir`` / ``episodes: 2376`` / ``ok: 2376`` / ``bad: []``).

   The derived set is CROSS-CHECKED against three independent committed
   artifacts before it is written (``--from-profile-csv`` refuses otherwise):
     * the 24 absent indices must be exactly the 24 skips, and their first/last
       must be ``1798`` / ``1941`` — the values written independently into
       ``scripts/rebuild_pai_rolling.py`` (``--skip-idx 1798,1835,…,1941``);
     * ``sum(T_out) == 472627`` must equal ``total_frames`` in
       ``…/2026-07-24-parity-corpus-profile/corpus_profile.json``, which was
       scanned on a DIFFERENT pod and a DIFFERENT path
       (``tanitad-pod3:/workspace/pai_epcache/…``);
     * the count 2 376 must equal ``n_episodes`` in
       ``…/2026-07-22-v4-labels/labels_train_v4_provenance.json``.

2. ``--record --cache-dir <dir> --split train|val``  (pod-side, authoritative)

   Records the manifest entry from a cache that has just been VERIFIED GOOD by
   ``scripts/pod_ops/compute_skipset.py`` (VERDICT MATCH). This is how the val
   entry gets its uid digest — the val uid set is not enumerated anywhere in
   this repo, so the committed val entry is COUNT-ONLY (600, MEASURED from
   ``labels_val_v4_provenance.json``) until someone runs:

       # on a pod, after compute_skipset.py prints VERDICT MATCH
       PYTHONPATH=/workspace/TanitAD/stack python3 \\
         scripts/make_parity_manifest.py --record --split val \\
         --cache-dir /workspace/.../physicalai-val-0c5f7dac3b11
       # then STAGE the changed tanitad/data/parity_manifest.json into the repo

   ``--record`` REFUSES to overwrite an entry that already carries a uid digest
   unless ``--force`` is given, so a truncated cache can never quietly
   "re-record" itself into a passing manifest.

3. ``--verify --cache-dir <dir>``  — check a live cache against the manifest and
   exit non-zero on a violation. No writes. Safe to run anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.data import parity                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HUB = REPO / "TanitAD Research Hub"
DEFAULT_PROFILE_CSV = (HUB / "Data Engineering/Implementation/incoming/"
                       "2026-07-25-v2-corpus-qa/parity_profile.csv")
CORPUS_PROFILE_JSON = (HUB / "Data Engineering/Implementation/incoming/"
                       "2026-07-24-parity-corpus-profile/corpus_profile.json")
V4_TRAIN_PROV = (HUB / "Benchmarks & Eval/Implementation/incoming/"
                 "2026-07-22-v4-labels/labels_train_v4_provenance.json")

# Cross-check constants — each is MEASURED and lives in a committed artifact.
XCHK_TOTAL_FRAMES = 472627          # corpus_profile.json:size.total_frames
XCHK_SKIP_FIRST, XCHK_SKIP_LAST = 1798, 1941   # rebuild_pai_rolling.py:32
XCHK_SKIPS = 24
XCHK_TRAIN_EPISODES = 2376
XCHK_N_SOURCES = 2400               # parity_skipset.sh: len(train) == 2400


# --------------------------------------------------------------------------- #
def _read_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": parity.MANIFEST_SCHEMA, "corpora": {}, "notes": {}}


def _write(path: Path, man: dict) -> None:
    path.write_text(json.dumps(man, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")
    print(f"[manifest] wrote {path}")


# --------------------------------------------------------------------------- #
def from_profile_csv(csv_path: Path, man: dict, *, strict_xcheck: bool = True
                     ) -> dict:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    uids = sorted(r["file"] for r in rows)
    idx = sorted(parity.episode_index(u) for u in uids)
    if any(i is None for i in idx):
        raise SystemExit(f"[manifest] non-epcache filenames in {csv_path}")
    missing = sorted(set(range(XCHK_N_SOURCES)) - set(idx))

    problems = []
    if len(uids) != XCHK_TRAIN_EPISODES:
        problems.append(f"episode count {len(uids)} != {XCHK_TRAIN_EPISODES}")
    if len(missing) != XCHK_SKIPS:
        problems.append(f"{len(missing)} absent indices != {XCHK_SKIPS} skips")
    elif (missing[0], missing[-1]) != (XCHK_SKIP_FIRST, XCHK_SKIP_LAST):
        problems.append(f"skip range {missing[0]}..{missing[-1]} != "
                        f"{XCHK_SKIP_FIRST}..{XCHK_SKIP_LAST} "
                        f"(rebuild_pai_rolling.py --skip-idx)")
    frames = sum(int(r["T_out"]) for r in rows)
    if frames != XCHK_TOTAL_FRAMES:
        problems.append(f"sum(T_out) {frames} != corpus_profile total_frames "
                        f"{XCHK_TOTAL_FRAMES}")
    if CORPUS_PROFILE_JSON.exists():
        cp = json.loads(CORPUS_PROFILE_JSON.read_text(encoding="utf-8"))
        if cp["size"]["usable_clips"] != XCHK_TRAIN_EPISODES:
            problems.append("corpus_profile.json usable_clips disagrees")
        if cp["size"]["total_frames"] != frames:
            problems.append("corpus_profile.json total_frames disagrees")
    if V4_TRAIN_PROV.exists():
        vp = json.loads(V4_TRAIN_PROV.read_text(encoding="utf-8"))
        if vp["n_episodes"] != XCHK_TRAIN_EPISODES:
            problems.append("labels_train_v4_provenance n_episodes disagrees")
    if problems and strict_xcheck:
        raise SystemExit("[manifest] REFUSING to write — cross-checks failed:\n  "
                         + "\n  ".join(problems))

    ent = parity.build_entry(
        uids, corpus_key=parity.PARITY_TRAIN_KEY, split="train",
        skip_indices=missing, uid_source="measured-enumeration",
        provenance={
            "derived_from": str(csv_path.relative_to(REPO)).replace("\\", "/"),
            "derived_on": date.today().isoformat(),
            "scanned_cache_dir":
                "/workspace/data/physicalai_phase0/_epcache/"
                "physicalai-train-e438721ae894",
            "scanner": "parity_profile.py (READ-ONLY pod scan; "
                       "parity_profile.json: episodes 2376, ok 2376, bad [])",
            "evidence_class": "MEASURED (committed pod-side scan of the "
                              "canonical cache dir)",
            "cross_checks": {
                "skip_indices_first_last_vs_rebuild_pai_rolling":
                    f"{XCHK_SKIP_FIRST}..{XCHK_SKIP_LAST} MATCH",
                "sum_T_out_vs_corpus_profile_total_frames":
                    f"{frames} == {XCHK_TOTAL_FRAMES} MATCH (different pod, "
                    f"different path)",
                "count_vs_labels_train_v4_provenance":
                    f"{XCHK_TRAIN_EPISODES} MATCH",
            },
        })
    ent["episode_uids"] = uids
    man.setdefault("corpora", {})[parity.PARITY_TRAIN_KEY] = ent
    print(f"[manifest] train entry: {len(uids)} episodes, "
          f"sha256 {ent['episode_uid_sha256']}, {len(missing)} skips "
          f"{missing[0]}..{missing[-1]}")
    return man


def val_count_only_entry(man: dict) -> dict:
    """The val entry. COUNT-ONLY on purpose: 600 is MEASURED
    (``labels_val_v4_provenance.json`` ``n_episodes``; ``parity_skipset.sh``
    asserts ``len(val) == 600``), but NO committed artifact enumerates the val
    uid set, and inventing one would be exactly the failure mode CLAUDE.md
    forbids. The count alone already refuses a truncated val cache — which is
    the failure this workstream closes; the uid digest additionally refuses a
    SUBSTITUTED set of the right size and is filled in by ``--record``."""
    ent = {
        "corpus_key": parity.PARITY_VAL_KEY,
        "split": "val",
        "episode_count": parity.PARITY_VAL_EPISODES,
        "uid_kind": "epcache_basename",
        "uid_source": "count-only-unrecorded",
        "episode_uid_sha256": None,
        "skip_indices": [],
        "skip_count": 0,
        "provenance": {
            "evidence_class": "MEASURED (count) / UNRECORDED (uid set)",
            "count_sources": [
                "labels_val_v4_provenance.json: n_episodes = 600",
                "scripts/parity_skipset.sh: assert len(val) == 600",
            ],
            "todo": "run --record --split val on a pod against a cache that "
                    "compute_skipset.py has just verified, then stage the diff",
        },
    }
    man.setdefault("corpora", {})[parity.PARITY_VAL_KEY] = ent
    print(f"[manifest] val entry: COUNT-ONLY, {ent['episode_count']} episodes")
    return man


def record(man: dict, cache_dir: Path, split: str, key: str | None,
           force: bool) -> dict:
    key = key or parity.corpus_key_of(cache_dir)
    if key is None:
        raise SystemExit(f"[manifest] {cache_dir} references no registered "
                         f"parity key; pass --key explicitly to register a new "
                         f"corpus.")
    uids = parity.scan_cache_dir(cache_dir)
    if not uids:
        raise SystemExit(f"[manifest] no {parity.EPISODE_GLOB} under {cache_dir}")
    prev = man.get("corpora", {}).get(key)
    if prev and prev.get("episode_uid_sha256") and not force:
        raise SystemExit(
            f"[manifest] {key} ALREADY carries a uid digest "
            f"({prev['episode_uid_sha256'][:12]}…, {prev['episode_count']} "
            f"episodes). Re-recording would let a truncated cache overwrite a "
            f"good manifest. Verify the cache with "
            f"scripts/pod_ops/compute_skipset.py first, then pass --force.")
    ent = parity.build_entry(
        uids, corpus_key=key, split=split,
        skip_indices=parity.scan_skip_markers(cache_dir),
        uid_source="recorded-from-verified-cache",
        provenance={
            "recorded_from": str(cache_dir),
            "recorded_on": date.today().isoformat(),
            "evidence_class": "MEASURED (live cache scan)",
            "precondition": "scripts/pod_ops/compute_skipset.py printed "
                            "VERDICT MATCH for this cache",
        })
    ent["episode_uids"] = uids
    man.setdefault("corpora", {})[key] = ent
    print(f"[manifest] recorded {key} ({split}): {len(uids)} episodes, "
          f"sha256 {ent['episode_uid_sha256']}, "
          f"{ent['skip_count']} skip markers")
    return man


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(parity.MANIFEST_PATH))
    ap.add_argument("--from-profile-csv", nargs="?", const=str(DEFAULT_PROFILE_CSV))
    ap.add_argument("--with-val-count-entry", action="store_true",
                    help="(re)write the COUNT-ONLY val entry")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--cache-dir")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--key", help="corpus key, when the dir name does not carry one")
    ap.add_argument("--mode", choices=["strict", "subset"], default="strict")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-xcheck", action="store_true",
                    help="skip the committed-artifact cross-checks (DEBUG ONLY)")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.verify:
        if not a.cache_dir:
            raise SystemExit("--verify needs --cache-dir")
        rec = parity.assert_parity_corpus(a.cache_dir, label="verify",
                                          mode=a.mode, manifest_path=out)
        print(json.dumps(rec, indent=2))
        return 0

    if not (a.from_profile_csv or a.record or a.with_val_count_entry):
        ap.error("nothing to do: pass --from-profile-csv, --record, "
                 "--with-val-count-entry or --verify")

    man = _read_manifest(out)
    man["schema"] = parity.MANIFEST_SCHEMA
    man.setdefault("notes", {}).update({
        "what": "Episode manifest for the SACRED parity corpus (CLAUDE.md "
                "§Invariants). tanitad.data.parity asserts every trainer's "
                "loaded episode set against this file at startup.",
        "uid": "episode uid = the ep_%05d.pt basename; the index is the "
               "position in the ordered source list (tanitad/data/epcache.py), "
               "so it is the stable identity WITHIN a build key.",
        "digest": "episode_uid_sha256 = sha256('\\n'.join(sorted(uids)))",
        "limitations": "This pins WHICH episode slots are present. It does not "
                       "hash episode CONTENT, so a same-name file with "
                       "different tensor bytes is not detected here (the build "
                       "key e438721ae894 and compute_skipset.py cover the "
                       "build side).",
        "regenerate": "scripts/make_parity_manifest.py --from-profile-csv "
                      "(dev box) or --record --cache-dir <verified cache> (pod)",
    })

    if a.from_profile_csv:
        man = from_profile_csv(Path(a.from_profile_csv), man,
                               strict_xcheck=not a.no_xcheck)
    if a.with_val_count_entry:
        man = val_count_only_entry(man)
    if a.record:
        if not a.cache_dir:
            raise SystemExit("--record needs --cache-dir")
        man = record(man, Path(a.cache_dir), a.split, a.key, a.force)

    _write(out, man)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
