"""Parquet catalog — the lake's metadata tier + view planner (spec §3.1, §5).

One Parquet row per episode carrying everything EXCEPT the frame blob (all
scalars + modality flags + native intrinsics + sha256 + license_class + the
``shard_key``/``member_key`` pointer). Hive-partitioned by
``license_class / source / split`` so a view is a predicate over the catalog
that prunes whole partitions before any I/O — and NO frame is touched to plan a
view (spec §5: "a view is a Parquet query returning a filtered shard/member
list").

The catalog is a DERIVED, rebuildable index over the shard ``meta.json`` sidecars
(the single source of truth); ``rebuild_catalog`` regenerates it from the tars if
it is ever lost or drifts (spec §8.3 risk 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from tanitad.lake.schema import (LakeRecord, catalog_arrow_schema,
                                 record_to_catalog_row)

CATALOG_SUBDIR = "catalog"
PARTITION_COLS = ["license_class", "source", "split"]


def catalog_dir(lake_root: str | Path) -> Path:
    return Path(lake_root) / CATALOG_SUBDIR


class CatalogWriter:
    """Accumulates catalog rows and flushes them to the Hive-partitioned store.

    Appends across ingest runs (``existing_data_behavior='overwrite_or_ignore'``);
    a per-run ``run_id`` in the file basename keeps concurrent/rerun writes from
    clobbering each other. Idempotent re-ingest overwrites the same run's files.
    """

    def __init__(self, lake_root: str | Path, run_id: str = "0"):
        self.dir = catalog_dir(lake_root)
        self.run_id = str(run_id)
        self._rows: list[dict[str, Any]] = []

    def append(self, rec: LakeRecord) -> None:
        self._rows.append(record_to_catalog_row(rec))

    def extend(self, recs: Iterable[LakeRecord]) -> None:
        for r in recs:
            self.append(r)

    def flush(self) -> int:
        """Write accumulated rows to partitioned Parquet; return the row count."""
        if not self._rows:
            return 0
        schema = catalog_arrow_schema()
        table = pa.Table.from_pylist(self._rows, schema=schema)
        self.dir.mkdir(parents=True, exist_ok=True)
        pads.write_dataset(
            table, self.dir, format="parquet",
            partitioning=PARTITION_COLS, partitioning_flavor="hive",
            basename_template=f"part-{self.run_id}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        n = len(self._rows)
        self._rows.clear()
        return n


# --------------------------------------------------------------------------- #
# View resolution — predicate pushdown + column projection                     #
# --------------------------------------------------------------------------- #
def open_catalog(lake_root: str | Path) -> pads.Dataset:
    """Open the catalog as a partitioned pyarrow Dataset (partition cols restored
    from the Hive path). Empty/missing catalog raises a clear error."""
    d = catalog_dir(lake_root)
    if not d.exists() or not any(d.rglob("*.parquet")):
        raise FileNotFoundError(f"no catalog under {d} — ingest a source first")
    return pads.dataset(d, format="parquet", partitioning="hive")


def resolve_view(lake_root: str | Path,
                 filter_expr: pads.Expression | None = None,
                 columns: list[str] | None = None,
                 sort_by: str | None = "episode_id") -> pa.Table:
    """Return the catalog rows for a view (predicate pushdown + projection).

    ``filter_expr`` is a ``pyarrow.dataset`` expression, e.g.
    ``(pads.field('license_class') == 'owned-safe') & pads.field('commercial_ok')``
    — partition-column predicates prune files before read. ``columns`` projects
    (pass ``None`` for all). Deterministically sorted so a hydrated cache is
    stable across runs.
    """
    ds = open_catalog(lake_root)
    table = ds.to_table(filter=filter_expr, columns=columns)
    if sort_by and sort_by in table.column_names:
        table = table.sort_by([(sort_by, "ascending")])
    return table


def resolve_members(lake_root: str | Path,
                    filter_expr: pads.Expression | None = None
                    ) -> list[dict[str, Any]]:
    """A view resolved to the minimal shard/member list the loader needs:
    ``[{episode_id, shard_key, member_key, sha256, license_class, split, source}]``.
    """
    cols = ["episode_id", "shard_key", "member_key", "sha256",
            "license_class", "commercial_ok", "share_alike", "split", "source",
            "split_unit_id"]
    table = resolve_view(lake_root, filter_expr, columns=cols)
    return table.to_pylist()


def rebuild_catalog(lake_root: str | Path, run_id: str = "rebuilt") -> int:
    """Regenerate the catalog from every shard's ``meta.json`` (the source of
    truth). The anti-drift fallback (spec §8.3 risk 7); returns rows written."""
    import io
    import json
    import tarfile

    lake_root = Path(lake_root)
    schema = catalog_arrow_schema()
    rows: list[dict[str, Any]] = []
    for shard in sorted((lake_root / "shards").rglob("*.tar")):
        with tarfile.open(shard, "r") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".meta.json"):
                    rows.append(json.loads(tar.extractfile(m).read().decode()))
    if not rows:
        return 0
    d = catalog_dir(lake_root)
    # fresh rebuild: clear then rewrite
    table = pa.Table.from_pylist(rows, schema=schema)
    d.mkdir(parents=True, exist_ok=True)
    pads.write_dataset(
        table, d, format="parquet",
        partitioning=PARTITION_COLS, partitioning_flavor="hive",
        basename_template=f"part-{run_id}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore")
    return len(rows)


def catalog_summary(lake_root: str | Path) -> dict[str, Any]:
    """Small human summary of the catalog (counts per partition) for reports."""
    ds = open_catalog(lake_root)
    t = ds.to_table(columns=["license_class", "source", "split", "episode_id",
                             "commercial_ok", "is_synthetic"])
    import collections
    per = collections.Counter(
        (r["license_class"], r["source"], r["split"])
        for r in t.select(["license_class", "source", "split"]).to_pylist())
    return {
        "episodes": t.num_rows,
        "partitions": {f"{lc}/{src}/{sp}": n for (lc, src, sp), n in
                       sorted(per.items())},
    }
