"""WebDataset-format tar shards for the lake's frame/heavy tier (spec §3.1).

A shard is a plain tar file whose members follow the WebDataset convention
``{key}.{ext}`` (the ``webdataset`` package is NOT required — a shard is just a
tar, and we read/write it with the stdlib ``tarfile``, so the running dev box
needs no new dependency). One tar sample per episode:

    {episode_id}.frames.npy   uint8 [T, C, S, S]         (the canonical blob)
    {episode_id}.motion.npz   actions/poses/timestamps   (npz; absent -> omitted)
    {episode_id}.meta.json    the full catalog row + provenance (source of truth)

The frame blob is stored with ``numpy.save`` (deterministic bytes) and its
``sha256`` is recorded in ``meta.json``; the reader re-verifies it on load
(I-D verify-without-rebuild). Shards are SA-segregated: ``share_alike`` data is
written under a ``sharealike/`` prefix and NEVER shares a tar with non-SA data
(spec §6 layer 2).

Shard path layout under the lake root::

    shards/<license_class>[/sharealike]/<source>/<split>/shard-XXXXX.tar
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from tanitad.lake.schema import LakeRecord, frames_sha256, record_to_catalog_row

DEFAULT_MAX_SHARD_BYTES = 1 << 30       # ~1 GiB target shard (spec §3.1: 1-2 GB)


def shard_prefix(license_class: str, source: str, split: str,
                 share_alike: bool) -> str:
    """Relative shard directory (POSIX) under ``<lake>/shards`` for a partition."""
    parts = ["shards", license_class]
    if share_alike:
        parts.append("sharealike")     # SA-segregation (spec §6 layer 2)
    parts += [source, split]
    return "/".join(parts)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return buf.getvalue()


class ShardWriter:
    """Rolling tar-shard writer for one (license_class, source, split) partition.

    Use as a context manager. ``write(rec)`` appends one episode as three tar
    members and stamps ``rec.shard_key`` / ``rec.member_key``. A new shard rolls
    when the current one exceeds ``max_bytes``. Deterministic member mtime keeps
    shards reproducible.
    """

    def __init__(self, lake_root: str | Path, license_class: str, source: str,
                 split: str, share_alike: bool,
                 max_bytes: int = DEFAULT_MAX_SHARD_BYTES,
                 shard_start: int = 0):
        self.lake_root = Path(lake_root)
        self.rel_dir = shard_prefix(license_class, source, split, share_alike)
        self.dir = self.lake_root / self.rel_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        # append-new-shard: continue numbering AFTER any existing shards in the
        # partition so a second ingest never overwrites the first's tars.
        existing = sorted(self.dir.glob("shard-*.tar"))
        if existing and shard_start == 0:
            self._idx = max(int(p.stem.split("-")[-1]) for p in existing) + 1
        else:
            self._idx = int(shard_start)
        self._tar: tarfile.TarFile | None = None
        self._bytes = 0
        self.written_shards: list[str] = []
        self.n_episodes = 0

    # -- shard lifecycle --
    def _shard_rel(self, idx: int) -> str:
        return f"{self.rel_dir}/shard-{idx:05d}.tar"

    def _open_new(self) -> None:
        self._close_current()
        rel = self._shard_rel(self._idx)
        self._tar = tarfile.open(self.lake_root / rel, "w")
        self._bytes = 0
        self.written_shards.append(rel)

    def _close_current(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def _add(self, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mtime = 0                    # deterministic / reproducible shards
        self._tar.addfile(info, io.BytesIO(payload))
        self._bytes += len(payload)

    # -- API --
    def write(self, rec: LakeRecord) -> None:
        if self._tar is None or self._bytes >= self.max_bytes:
            if self._tar is not None:
                self._idx += 1
            self._open_new()

        stem = str(int(rec.episode_id))
        rec.member_key = stem
        rec.shard_key = self.written_shards[-1]

        frames_np = np.ascontiguousarray(rec.frames.cpu().numpy())
        self._add(f"{stem}.frames.npy", _npy_bytes(frames_np))

        motion = {}
        if rec.actions is not None:
            motion["actions"] = rec.actions.cpu().numpy()
        if rec.poses is not None:
            motion["poses"] = rec.poses.cpu().numpy()
        if motion:
            self._add(f"{stem}.motion.npz", _npz_bytes(**motion))

        meta = record_to_catalog_row(rec)
        if rec.language is not None:
            self._add(f"{stem}.lang.json",
                      json.dumps(rec.language, sort_keys=True).encode())
        self._add(f"{stem}.meta.json",
                  json.dumps(meta, sort_keys=True).encode())
        self.n_episodes += 1

    def close(self) -> None:
        self._close_current()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #
def _load_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def iter_shard_samples(shard_path: str | Path, verify_sha256: bool = True
                       ) -> Iterator[dict]:
    """Yield ``{episode_id, frames, actions, poses, meta}`` per episode in a shard.

    Frames come back as a ``uint8`` torch tensor byte-identical to what was
    written; ``actions``/``poses`` are ``float32`` (or ``None``). With
    ``verify_sha256`` the frame digest is re-checked against ``meta.json`` — a
    corrupt/rotted shard fails loudly rather than silently training on garbage.
    """
    shard_path = Path(shard_path)
    with tarfile.open(shard_path, "r") as tar:
        members: dict[str, dict[str, bytes]] = {}
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = m.name
            stem, _, ext = name.partition(".")
            payload = tar.extractfile(m).read()
            members.setdefault(stem, {})[ext] = payload

    for stem, parts in members.items():
        meta = json.loads(parts["meta.json"].decode())
        frames_np = _load_npy(parts["frames.npy"])
        frames = torch.from_numpy(np.ascontiguousarray(frames_np))
        if verify_sha256:
            got = frames_sha256(frames)
            if got != meta.get("sha256"):
                raise ValueError(
                    f"{shard_path.name}:{stem} sha256 mismatch "
                    f"(shard {got[:12]} != meta {str(meta.get('sha256'))[:12]}) "
                    f"— corrupt shard, refusing (I-D)")
        actions = poses = None
        if "motion.npz" in parts:
            with np.load(io.BytesIO(parts["motion.npz"])) as mz:
                if "actions" in mz:
                    actions = torch.from_numpy(np.ascontiguousarray(mz["actions"]))
                if "poses" in mz:
                    poses = torch.from_numpy(np.ascontiguousarray(mz["poses"]))
        yield {
            "episode_id": int(meta["episode_id"]),
            "frames": frames,
            "actions": actions,
            "poses": poses,
            "meta": meta,
        }
