"""Build TanitDataSet-C / -R from the license-clean episode caches on this box.

The run-once builder that turns the DESIGN (two tiers, one schema, a per-record
`tier` stamp) into actual records:

    discover cached ep_*.pt  ->  CachedEpisodeIngestor  ->  schema LakeRecords
      ->  SA-segregated tar shards + Parquet catalog  ->  two-pass dedup report
      ->  tier / license breakdown.

Design (`TanitDataSet Research Hub/Data Engineering/TANITDATASET_V1_STRATEGY.md`
rev-4 + `TANITDATASET_TIER_INTEGRATION_2026-07-21.md`):

- **TanitDataSet-C** (commercial, tier ``ship``) = PERMISSIVE sources only, so it
  is HF-publishable and commercially deployable. Today: **comma2k19** (MIT, in
  the lake). L2D (Apache-2.0), PandaSet, Udacity, WorldModel-Synth join here once
  their adapters land — all already registered ``owned-safe`` in SOURCE_REGISTRY.
- **TanitDataSet-R** (research superset) = C ∪ NC, same schema, ``nc`` tier stamp;
  never redistributed. NC sources (nuScenes/AV2/…) join when downloaded.

Two things this builder does NOT do, by construction:
  * it can NEVER admit the PhysicalAI-AV parity corpus — ``assemble_lake_record``
    raises ``PermissionError`` on ``gated-confidential`` (and on ``refuse``);
  * it NEVER pushes. Publishing is ``tanitad.lake.hf_export`` behind a human gate.

Parity is untouched: this is a NEW corpus, entirely separate from
``physicalai-train-e438721ae894``.

Usage (defaults to the real comma2k19 256px cache on this box)::

    python scripts/build_tanitdataset.py \
        --lake-root C:/Users/Admin/tanitad-data/tanitdataset/lake \
        --report-json build_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from tanitad.data.mixing import load_episode
from tanitad.lake.catalog import catalog_summary
from tanitad.lake.dedup import clip_phash, two_pass_dedup
from tanitad.lake.filtering import tier_of
from tanitad.lake.ingest import CachedEpisodeIngestor, ingest_source
from tanitad.lake.schema import SOURCE_REGISTRY
from tanitad.lake.shards import iter_shard_samples

# The license-clean 256px caches actually present on this dev box (all MIT
# comma2k19 — the canonical [T,9,256,256] contract). Override with --cache.
DEFAULT_COMMA_CACHES = [
    "C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f",
]


def discover_unique_episodes(cache_dirs: list[str]) -> dict[int, str]:
    """Map episode_id -> ep_*.pt path across caches, keeping the first occurrence.

    episode_id is content-stable (the same comma segment gets the same id across
    builds), so this de-dups re-hydrated copies of the same episode up front."""
    by_id: dict[int, str] = {}
    for d in cache_dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep_*.pt"))):
            ep = load_episode(f, mmap=True)
            eid = int(ep.episode_id)
            by_id.setdefault(eid, f)
    return by_id


def split_map_by_id(by_id: dict[int, str], val_every: int = 5) -> dict[str, list[str]]:
    """Deterministic episode-level split: every ``val_every``-th episode (sorted by
    id) is val. NOTE: comma2k19 route ids are not preserved in a bare ep cache, so
    this is an EPISODE-level split, not route-disjoint — documented in the card."""
    ordered = [by_id[i] for i in sorted(by_id)]
    train, val = [], []
    for k, path in enumerate(ordered):
        (val if k % val_every == 0 else train).append(path)
    return {"train": train, "val": val}


def build_source(source: str, cache_dirs: list[str], lake_root: str,
                 val_every: int, max_units: int | None) -> dict:
    """Ingest one source's cache into the lake. Returns the ingest summary."""
    by_id = discover_unique_episodes(cache_dirs)
    if max_units is not None:
        by_id = {i: by_id[i] for i in sorted(by_id)[:max_units]}
    split_paths = split_map_by_id(by_id, val_every=val_every)

    ing = CachedEpisodeIngestor(source=source, size=256, stride=2, n_stack=3)
    # discover() consumes root={split: [paths]}; ingest_source drives the rest.
    summary = ingest_source(ing, split_paths, lake_root, run_id=source,
                            verbose=True)
    summary["unique_input_episodes"] = len(by_id)
    return summary


def dedup_report(lake_root: str, source: str) -> dict:
    """Two-pass dedup over the ingested shards (authoritative; verifies sha256).

    comma2k19 has no lat/lon in a bare cache, so pass-2 (GPS/time) sees no geo and
    every clip is its own cross-cluster; pass-1 (pHash within-source) collapses
    near-identical clips. Reports the exemplar count = the de-duplicated corpus."""
    items = []
    for shard in sorted((Path(lake_root) / "shards").rglob("*.tar")):
        for s in iter_shard_samples(shard, verify_sha256=True):
            if s["meta"].get("source") != source:
                continue
            items.append({"id": int(s["episode_id"]), "source": source,
                          "phash": clip_phash(s["frames"])})
    if not items:
        return {"n": 0, "exemplars": 0}
    verdicts = two_pass_dedup(items)
    exemplars = [i for i, v in verdicts.items() if v.is_exemplar]
    clusters = {v.dedup_cluster_id for v in verdicts.values()}
    return {"n": len(items), "exemplars": len(exemplars),
            "phash_clusters": len(clusters),
            "near_dup_collapsed": len(items) - len(exemplars)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake-root", required=True)
    ap.add_argument("--source", default="comma2k19")
    ap.add_argument("--cache", action="append", default=None,
                    help="cache dir(s) of ep_*.pt; repeatable")
    ap.add_argument("--val-every", type=int, default=5, help="1/N episodes -> val")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    caches = args.cache or DEFAULT_COMMA_CACHES
    lic = SOURCE_REGISTRY[args.source]
    tier = tier_of(lic.license_class, lic.share_alike, lic.commercial_ok)
    print(f"[build] source={args.source} license={lic.license_name} "
          f"class={lic.license_class} tier={tier} commercial_ok={lic.commercial_ok}")
    print(f"[build] caches: {caches}")

    summary = build_source(args.source, caches, args.lake_root,
                           args.val_every, args.max_units)
    dd = dedup_report(args.lake_root, args.source)
    cat = catalog_summary(args.lake_root)

    report = {
        "source": args.source,
        "tier": tier,
        "license_name": lic.license_name,
        "license_class": lic.license_class,
        "commercial_ok": lic.commercial_ok,
        "share_alike": lic.share_alike,
        "ingest": {k: summary[k] for k in
                   ("per_split", "catalog_rows", "unique_input_episodes",
                    "already_present", "params_hash")},
        "n_skipped": len(summary.get("skipped", [])),
        "shards": summary.get("shards", []),
        "dedup": dd,
        "catalog_summary": cat,
    }
    print("\n=== BUILD REPORT ===")
    print(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"\n[build] wrote {args.report_json}")


if __name__ == "__main__":
    main()
