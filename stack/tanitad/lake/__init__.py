"""TanitAD Data Lake (Phase A) — the license-clean, write-once episode lake.

This package operationalizes ``Data Engineering/DATA_LAKE_ARCHITECTURE.md`` for
Phase A: stand up the PERMISSIVE lake offline on the dev box from license-clean
sources (comma2k19 MIT first; Cosmos-Drive-Dreams CC-BY scaffolded), validated as
a BYTE-IDENTICAL drop-in for the current trainers.

Design invariants (from the architecture spec, enforced here):

- **I-A. Strict superset of the current contract.** The core projection
  ``{frames[T,C,S,S] u8, actions[T,2], poses[T,4], episode_id}`` is recoverable
  byte-for-byte, so ``assert_contract`` and the I7 ``CORPUS_META`` identity still
  pass and existing trainers run unchanged.
- **I-C. License is a first-class, structural axis** — physical partitioning of
  shards/catalog by ``license_class`` + an export guard, not just a filter.
- **I-D. The recipe never dies.** Each shard/episode carries source ids + a
  build-params hash + a per-episode ``sha256`` (verify-without-rebuild).

The lake is ADDITIVE: it does not modify ``stack/tanitad/data/_contract.py`` or
any ``build_episode`` adapter — the ingestors *wrap* them. Rollback is repointing
``--data-root`` back at the origin epcache.

Two tiers (``schema`` / ``shards`` / ``catalog``):

- **catalog** — a Parquet table, one row per episode (all scalars + modality
  flags + native intrinsics + ``sha256`` + ``license_class`` + shard pointer),
  Hive-partitioned by ``license_class / source / split``. Predicate-pushdown +
  column projection resolves a *view* to a shard/member list with no frame I/O.
- **shards** — WebDataset-format tar shards (written with the stdlib ``tarfile``
  — no new dependency) holding the canonical ``uint8`` frame blobs plus a
  ``motion.npz`` (actions/poses) and a ``meta.json`` per episode.

The read layer (``view.LakeWindowDataset``) resolves a view, hydrates the shards
ONCE into local ``ep_*.pt`` (the exact ``epcache`` layout — download replaces
decode), and yields the byte-identical ``EpisodeWindowDataset`` window contract,
a drop-in via the trainer's existing ``data="cached"`` path.
"""

from tanitad.lake.schema import (  # noqa: F401
    LakeRecord,
    SOURCE_REGISTRY,
    LICENSE_CLASSES,
    assemble_lake_record,
    validate_superset,
    catalog_arrow_schema,
)

__all__ = [
    "LakeRecord",
    "SOURCE_REGISTRY",
    "LICENSE_CLASSES",
    "assemble_lake_record",
    "validate_superset",
    "catalog_arrow_schema",
]
