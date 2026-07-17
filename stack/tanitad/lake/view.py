"""Multi-view read layer (spec §5): a view -> shard/member list -> hydrate-once
-> the byte-identical ``EpisodeWindowDataset`` window contract.

A **view** = a column projection x a metadata predicate over the Parquet catalog.
``LakeView.resolve()`` returns the filtered ``(shard_key, member_key)`` list with
no frame I/O. ``hydrate_cached`` then streams only those shards ONCE into local
``ep_*.pt`` — the EXACT ``epcache`` on-disk layout, so ``hydrate`` *replaces
``build_episodes_cached``'s decode with a download* and the result is a drop-in
for the trainer's ``data="cached"`` path. ``LakeWindowDataset`` wraps the
reused, unchanged :class:`EpisodeWindowDataset`, so its windows are byte-identical
to the current trainers' (the acceptance gate proves it).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tanitad.data._contract import EpisodeWindowDataset
from tanitad.data.mixing import load_episode, save_episode
from tanitad.data.toy_driving import ToyEpisode
from tanitad.lake.catalog import resolve_members
from tanitad.lake.shards import iter_shard_samples

SERVING_SUBDIR = "_serving"


# --------------------------------------------------------------------------- #
# View                                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class LakeView:
    """A named view: a pyarrow predicate over the catalog + a license scope.

    ``resolve()`` -> the minimal member list. ``scope`` is the set of
    ``license_class`` values this view is ALLOWED to contain; the export guard
    (``license_guard.verify_license_scope``) asserts it before any egress.
    """

    lake_root: str | Path
    name: str = "view"
    filter_expr: Any = None                 # pyarrow.dataset.Expression | None
    scope: frozenset[str] = frozenset({"owned-safe"})

    def resolve(self) -> list[dict[str, Any]]:
        members = resolve_members(self.lake_root, self.filter_expr)
        if not members:
            raise ValueError(f"view {self.name!r} resolved to 0 episodes "
                             f"(predicate too strict, or lake empty)")
        return members

    def signature(self) -> str:
        """Stable hash of the resolved member set (episode ids + sha256) — keys
        the hydrate cache so a changed view rehydrates, an unchanged one reuses."""
        members = self.resolve()
        payload = sorted((m["episode_id"], m["sha256"]) for m in members)
        return hashlib.sha1(json.dumps(payload, default=str).encode()
                            ).hexdigest()[:12]


# Convenience constructors for the standard views (spec §5.2) ---------------- #
def world_model_view(lake_root, scope=("owned-safe",),
                     include_synthetic: bool = True,
                     source: str | None = None) -> LakeView:
    """frames+actions+poses over license scope — the flagship trainer view."""
    import pyarrow.dataset as pads
    expr = (pads.field("modality_flags", "has_actions") &
            pads.field("modality_flags", "has_poses"))
    lc = None
    for cls in scope:
        lc = (pads.field("license_class") == cls) if lc is None else \
            (lc | (pads.field("license_class") == cls))
    expr = expr & lc
    if not include_synthetic:
        expr = expr & (pads.field("is_synthetic") == False)   # noqa: E712
    if source is not None:
        expr = expr & (pads.field("source") == source)
    return LakeView(lake_root, name="world_model", filter_expr=expr,
                    scope=frozenset(scope))


def owned_safe_commercial_view(lake_root) -> LakeView:
    """The commercial-clean export view: owned-safe AND commercial_ok."""
    import pyarrow.dataset as pads
    expr = ((pads.field("license_class") == "owned-safe") &
            pads.field("commercial_ok"))
    return LakeView(lake_root, name="owned_safe_commercial", filter_expr=expr,
                    scope=frozenset({"owned-safe"}))


# --------------------------------------------------------------------------- #
# hydrate_cached — download(copy)-once -> ep_*.pt (epcache layout)             #
# --------------------------------------------------------------------------- #
def _reconstruct_episode(sample: dict) -> ToyEpisode:
    return ToyEpisode(frames=sample["frames"], actions=sample["actions"],
                      poses=sample["poses"], episode_id=sample["episode_id"])


def hydrate_cached(lake_root: str | Path, members: list[dict],
                   cache_dir: str | Path, tag: str,
                   verify_sha256: bool = True) -> Path:
    """Materialize ``members`` into ``<cache_dir>/<tag>/ep_%05d.pt`` (epcache
    layout, episode-id-sorted). Reads each shard once; idempotent per ep file.

    This is the crux of the drop-in: the on-disk result is indistinguishable from
    an ``epcache`` dir built by ``build_episodes_cached`` — same ``ep_%05d.pt``,
    same ``mmap`` load path — but produced by a copy, not a decode.
    """
    lake_root = Path(lake_root)
    out = Path(cache_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    ordered = sorted(members, key=lambda m: int(m["episode_id"]))
    idx_of = {int(m["episode_id"]): i for i, m in enumerate(ordered)}

    # which episodes still need materializing?
    need_by_shard: dict[str, set[int]] = defaultdict(set)
    n_have = 0
    for m in ordered:
        i = idx_of[int(m["episode_id"])]
        if (out / f"ep_{i:05d}.pt").exists():
            n_have += 1
            continue
        need_by_shard[m["shard_key"]].add(int(m["episode_id"]))

    n_built = 0
    for shard_key, wanted in need_by_shard.items():
        shard_path = lake_root / shard_key
        for sample in iter_shard_samples(shard_path, verify_sha256=verify_sha256):
            eid = int(sample["episode_id"])
            if eid not in wanted:
                continue
            ep = _reconstruct_episode(sample)
            save_episode(ep, str(out / f"ep_{idx_of[eid]:05d}.pt"))
            n_built += 1

    (out / "DONE").write_text(json.dumps(
        {"episodes": len(ordered), "hydrated": n_built, "reused": n_have}))
    return out


# --------------------------------------------------------------------------- #
# LakeWindowDataset — the drop-in                                              #
# --------------------------------------------------------------------------- #
class LakeWindowDataset(torch.utils.data.Dataset):
    """Drop-in for :class:`EpisodeWindowDataset` reading from the lake.

    Resolves the view, hydrates the shards once into ``ep_*.pt``, loads them
    (``mmap=True``, F-7) and delegates ``__getitem__`` to the reused
    ``EpisodeWindowDataset`` — so the window contract is byte-identical.
    """

    def __init__(self, view: LakeView, window: int = 8, max_horizon: int = 16,
                 cache_dir: str | Path | None = None, split: str | None = None,
                 verify_sha256: bool = True):
        members = view.resolve()
        if split is not None:
            members = [m for m in members if m.get("split") == split]
            if not members:
                raise ValueError(f"view {view.name!r} has no '{split}' episodes")
        cache_dir = Path(cache_dir or (Path(view.lake_root) / "_cache"))
        tag = f"{view.name}-{split or 'all'}-{view.signature()}"
        self.cache_path = hydrate_cached(view.lake_root, members, cache_dir, tag,
                                         verify_sha256=verify_sha256)
        files = sorted(self.cache_path.glob("ep_*.pt"))
        eps = [load_episode(str(p), mmap=True) for p in files]
        self._inner = EpisodeWindowDataset(eps, window=window,
                                           max_horizon=max_horizon)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, i: int):
        return self._inner[i]


def hydrate_view_for_trainer(view: LakeView, cache_dir: str | Path | None = None,
                             verify_sha256: bool = True) -> Path:
    """Materialize a view's train+val into a single serving root whose subdirs
    match the trainer's ``data="cached"`` globs (``*train*`` / ``*val*``), and
    return that root. The exact command is then::

        --data cached --data-root <returned path>

    (points at ``ep_*.pt`` the trainer loads with no code change).
    """
    members = view.resolve()
    root = Path(cache_dir or (Path(view.lake_root) / SERVING_SUBDIR)) / view.name
    root.mkdir(parents=True, exist_ok=True)
    sig = view.signature()
    for split in ("train", "val"):
        sm = [m for m in members if m.get("split") == split]
        if sm:
            hydrate_cached(view.lake_root, sm, root,
                           f"{view.name}-{split}-{sig}",
                           verify_sha256=verify_sha256)
    return root
