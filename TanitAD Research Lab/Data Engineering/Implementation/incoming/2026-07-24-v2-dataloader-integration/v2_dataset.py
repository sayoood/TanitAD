"""Lazy, memory-bounded provider list for the v2 compressed episode cache.

The v2 corpus (``physicalai-v2bal-...``) stores each ~20 s clip as one
``*.v2ep.pt`` of JPEG-encoded f-theta-cropped 256 px frames -- see
``scripts/v2_compressed.py`` (:func:`build_compressed` / :func:`load_compressed`).
At ~50 h / ~9 000 clips the *decoded* corpus is ~1 TB, so it CANNOT be held in
RAM the way the raw epcache path (``tanitad.data.mixing.load_episode`` -> a list
of in-memory ``ToyEpisode``) is.

This module is a drop-in, contract-identical replacement for that episode list:
a list of :class:`LazyV2Episode` providers exposing the SAME attribute surface as
:class:`tanitad.data.toy_driving.ToyEpisode` (``.frames``, ``.actions``,
``.poses``, ``.episode_id``) but loading lazily:

* ``.poses`` / ``.actions`` are the small float32 ``[T, *]`` tensors, read ONCE
  per clip at index-build time (a metadata-only ``mmap`` scan that never pages in
  the JPEG buffer) and kept resident -- the whole 9 000-clip corpus of
  poses+actions is only ~45 MB.
* ``.frames`` is a :class:`_V2FramesProxy`: ``.shape`` is O(1) (from the index),
  and a SLICE decodes ONLY the JPEGs that slice needs (window+horizon frames,
  ~30 of ~200), stacking them EXACTLY as :func:`load_compressed` does. The
  compressed clip payloads live in a bounded LRU, so RAM stays flat regardless of
  corpus size.

Because the providers are fed to the UNCHANGED ``FlagshipWindowDataset``
(``scripts/train_flagship4b.py``, via ``_wrap``), every emitted window -- its
keys, shapes, dtypes, nav/maneuver labels and ``pose_prev`` -- is produced by
byte-for-byte the same code as the raw path. Passing the trainer ``--v2-cache
<dir>`` therefore changes ONLY the frame *source*; with no flag the trainer is
byte-identical to today.

Faithfulness to :func:`load_compressed` (pinned in ``tests/test_v2_dataset.py``
and MEASURED against real ``*.v2ep.pt`` on the eval pod, 2026-07-24)::

    frames  = stack_frames(decode_jpeg(...), n_stack)   [T_out, 3*n_stack, S, S] u8
    poses   = payload["poses"][n_stack-1:]              [T_out, 4] f32
    actions = payload["actions"][n_stack-1:]            [T_out, 2] f32
    T_out   = len(payload["poses"]) - (n_stack-1)

It reuses ``tanitad.data.comma2k19.stack_frames`` (the identical D-015 stack) and
``torchvision.io.decode_jpeg``; it deliberately does NOT import
``scripts/v2_compressed.py`` (whose *build* path pulls pandas / pyav), so the
training + CI import surface stays minimal.
"""

from __future__ import annotations

import glob
import os
import time
from collections import OrderedDict

import torch
import torchvision.io as tvio

from tanitad.data.comma2k19 import stack_frames

MANIFEST_NAME = "_v2manifest.pt"
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- #
# Frame decode helpers (mirror load_compressed exactly, minus the build deps)  #
# --------------------------------------------------------------------------- #
def _jpeg_offsets(jpeg_len: torch.Tensor) -> torch.Tensor:
    """Prefix-sum byte offsets into the concatenated JPEG buffer (== the
    offsets :func:`load_compressed` derives)."""
    return torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(jpeg_len.to(torch.int64), 0)])


def _decode_stacked(jpeg_buf: torch.Tensor, offs: torch.Tensor, n_stack: int,
                    a: int, b: int) -> torch.Tensor:
    """Decode + D-015 channel-stack ONLY stacked-frame rows ``[a:b]``.

    ``stack_frames`` output row ``j`` = channel-concat of raw frames
    ``j, j+1, ..., j+n_stack-1``, so rows ``[a:b]`` need raw frames
    ``[a : b + n_stack - 1]``. Stacking is per-frame-independent, so decoding
    just that raw sub-block and stacking it is BIT-IDENTICAL to decoding the
    whole clip and slicing ``[a:b]`` (validated in tests + on real data)."""
    k = n_stack - 1
    raw = [tvio.decode_jpeg(jpeg_buf[int(offs[i]):int(offs[i + 1])],
                            mode=tvio.ImageReadMode.RGB)
           for i in range(a, b + k)]                       # [3, S, S] u8 each
    return stack_frames(torch.stack(raw), n_stack)         # [b-a, 3*n_stack, S, S] u8


# --------------------------------------------------------------------------- #
# Frames proxy — the ONLY surface of ep.frames that the window datasets touch  #
# --------------------------------------------------------------------------- #
class _V2FramesProxy:
    """Stand-in for the ``[T, C, H, W]`` uint8 frames tensor.

    Supports exactly what ``EpisodeWindowDataset`` / ``FailLoudWindowDataset`` /
    ``FlagshipWindowDataset`` touch: ``.shape`` / ``.ndim`` / ``.dtype`` /
    ``len()`` and dim-0 slice (or int) indexing. A contiguous slice triggers a
    partial JPEG decode; the result is a real owned uint8 tensor, so the
    downstream ``to_float_frames`` (``.float()/255``) works unchanged."""

    __slots__ = ("_cache", "_clip", "_shape")

    def __init__(self, cache: "V2CompressedCache", clip_idx: int,
                 shape: torch.Size):
        self._cache = cache
        self._clip = clip_idx
        self._shape = shape

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def ndim(self) -> int:
        return 4

    @property
    def dtype(self) -> torch.dtype:
        return torch.uint8

    def __len__(self) -> int:
        return int(self._shape[0])

    def __getitem__(self, idx):
        T = int(self._shape[0])
        if isinstance(idx, slice):
            a, b, step = idx.indices(T)
            if step != 1:
                raise ValueError("v2 frames proxy supports contiguous slices only")
            if b <= a:
                return torch.empty((0, *self._shape[1:]), dtype=torch.uint8)
            return self._cache.decode_stacked_range(self._clip, a, b)
        i = int(idx)
        if i < 0:
            i += T
        return self._cache.decode_stacked_range(self._clip, i, i + 1)[0]


class LazyV2Episode:
    """``ToyEpisode``-shaped lazy provider for one v2 clip.

    ``.frames`` proxies partial JPEG decode; ``.poses`` / ``.actions`` are
    resident float32 tensors; ``.maneuvers`` is ``None`` (the window path
    recomputes maneuver labels from poses -- see ``FlagshipWindowDataset``, which
    never reads ``ep.maneuvers``)."""

    __slots__ = ("frames", "poses", "actions", "episode_id", "maneuvers")

    def __init__(self, cache: "V2CompressedCache", clip_idx: int,
                 poses: torch.Tensor, actions: torch.Tensor, episode_id: int,
                 shape: torch.Size):
        self.frames = _V2FramesProxy(cache, clip_idx, shape)
        self.poses = poses
        self.actions = actions
        self.episode_id = int(episode_id)
        self.maneuvers = None


# --------------------------------------------------------------------------- #
# Per-cache-dir payload LRU + partial decode                                   #
# --------------------------------------------------------------------------- #
class V2CompressedCache:
    """Owns one v2 cache dir: the clip filename list and a bounded LRU of loaded
    compressed payloads ``(jpeg_buf, offsets, n_stack)``. One instance is shared
    by all :class:`LazyV2Episode` of that dir.

    The LRU is per-PROCESS and never crosses the DataLoader-worker boundary (see
    ``__getstate__``): every worker fills its own, so total RAM is
    ``num_workers * lru_size * mean_payload`` (~2-4 MB/clip)."""

    def __init__(self, cache_dir, lru_size: int = 64):
        self.cache_dir = str(cache_dir)
        self.lru_size = max(1, int(lru_size))
        self.files: list[str] = []
        self._lru: "OrderedDict[int, tuple]" | None = None

    # Pickling: drop the live LRU so a populated cache is never serialised into
    # each worker (they rebuild their own, starting empty).
    def __getstate__(self) -> dict:
        return {"cache_dir": self.cache_dir, "lru_size": self.lru_size,
                "files": self.files}

    def __setstate__(self, s: dict) -> None:
        self.cache_dir = s["cache_dir"]
        self.lru_size = s["lru_size"]
        self.files = s["files"]
        self._lru = None

    def _payload(self, clip_idx: int) -> tuple:
        if self._lru is None:
            self._lru = OrderedDict()
        hit = self._lru.get(clip_idx)
        if hit is not None:
            self._lru.move_to_end(clip_idx)
            return hit
        path = os.path.join(self.cache_dir, self.files[clip_idx])
        d = torch.load(path, map_location="cpu", weights_only=False)
        payload = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]), int(d["n_stack"]))
        self._lru[clip_idx] = payload
        while len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        return payload

    def decode_stacked_range(self, clip_idx: int, a: int, b: int) -> torch.Tensor:
        jpeg_buf, offs, n_stack = self._payload(clip_idx)
        return _decode_stacked(jpeg_buf, offs, n_stack, a, b)


# --------------------------------------------------------------------------- #
# Manifest (cheap, cached) + provider construction                            #
# --------------------------------------------------------------------------- #
def _scan_meta(path: str) -> tuple:
    """Metadata-only read of one clip: ``(poses[k:], actions[k:], episode_id,
    n_stack, image_size)``. ``mmap=True`` pages in ONLY the small pose/action
    storages -- the multi-MB ``jpeg_buf`` is never touched -- and ``.clone()``
    copies them off the mmap into owned resident tensors."""
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    k = int(d["n_stack"]) - 1
    poses = d["poses"][k:].clone().contiguous().float()
    actions = d["actions"][k:].clone().contiguous().float()
    return poses, actions, int(d["episode_id"]), int(d["n_stack"]), int(d["image_size"])


def _list_clips(cache_dir: str) -> list[str]:
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(cache_dir, "*.v2ep.pt")))


def load_or_build_manifest(cache_dir, rebuild: bool = False,
                           verbose: bool = True) -> dict:
    """Return the per-clip metadata manifest for ``cache_dir``, building + caching
    it (sidecar ``_v2manifest.pt``) on first use. Rebuilds automatically if the
    ``*.v2ep.pt`` set changed. Resident size ~= poses+actions of the corpus
    (~45 MB for 9 000 clips); the sidecar makes subsequent starts instant."""
    cache_dir = str(cache_dir)
    files = _list_clips(cache_dir)
    if not files:
        raise FileNotFoundError(f"no *.v2ep.pt under {cache_dir}")
    mp = os.path.join(cache_dir, MANIFEST_NAME)
    if (not rebuild) and os.path.exists(mp):
        try:
            man = torch.load(mp, map_location="cpu", weights_only=False)
            if man.get("version") == MANIFEST_VERSION and man.get("files") == files:
                return man
            if verbose:
                print(f"[v2] manifest {mp} stale (file set changed) -> rebuild",
                      flush=True)
        except Exception as e:                     # noqa: BLE001 (corrupt sidecar)
            if verbose:
                print(f"[v2] manifest {mp} unreadable ({e!r}) -> rebuild",
                      flush=True)
    poses_l, act_l, eid_l, ns_l, sz_l, tout_l = [], [], [], [], [], []
    t0 = time.time()
    for j, fn in enumerate(files):
        poses, actions, eid, n_stack, S = _scan_meta(os.path.join(cache_dir, fn))
        poses_l.append(poses)
        act_l.append(actions)
        eid_l.append(eid)
        ns_l.append(n_stack)
        sz_l.append(S)
        tout_l.append(int(poses.shape[0]))
        if verbose and (j + 1) % 500 == 0:
            print(f"[v2] manifest {cache_dir}: {j + 1}/{len(files)} clips "
                  f"({time.time() - t0:.0f}s)", flush=True)
    man = {"version": MANIFEST_VERSION, "files": files, "poses": poses_l,
           "actions": act_l, "episode_id": eid_l, "n_stack": ns_l,
           "image_size": sz_l, "T_out": tout_l}
    try:
        tmp = mp + ".tmp"
        torch.save(man, tmp)
        os.replace(tmp, mp)
        if verbose:
            print(f"[v2] manifest cached -> {mp} ({len(files)} clips, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    except OSError as e:
        if verbose:
            print(f"[v2] manifest NOT cached ({e}); held in RAM this run",
                  flush=True)
    return man


def build_v2_providers(cache_dirs, lru_size: int = 64, rebuild: bool = False,
                       verbose: bool = True) -> list[LazyV2Episode]:
    """Build the lazy provider list for one or more v2 cache dirs.

    The returned list is a drop-in replacement for the raw episode list fed to
    ``FlagshipWindowDataset`` (via ``_wrap``): every element quacks like a
    ``ToyEpisode``. Providers from multiple dirs are concatenated (the
    consolidated-cache case -- e.g. pod1 bottom-half + pod3 top-half). Each dir
    keeps its OWN :class:`V2CompressedCache` (and LRU)."""
    if isinstance(cache_dirs, (str, os.PathLike)):
        cache_dirs = [cache_dirs]
    providers: list[LazyV2Episode] = []
    for cd in cache_dirs:
        cd = str(cd)
        man = load_or_build_manifest(cd, rebuild=rebuild, verbose=verbose)
        cache = V2CompressedCache(cd, lru_size=lru_size)
        cache.files = man["files"]
        for i in range(len(man["files"])):
            n_stack = int(man["n_stack"][i])
            S = int(man["image_size"][i])
            shape = torch.Size((int(man["T_out"][i]), 3 * n_stack, S, S))
            providers.append(LazyV2Episode(
                cache, i, man["poses"][i], man["actions"][i],
                int(man["episode_id"][i]), shape))
        if verbose:
            print(f"[v2] {cd}: {len(man['files'])} providers "
                  f"(channels {3 * int(man['n_stack'][0])}, "
                  f"{int(man['image_size'][0])} px)", flush=True)
    return providers


def decode_full_episode(path: str):
    """Non-lazy convenience: decode a clip to a full ``ToyEpisode`` using ONLY
    this module's deps (torchvision.io + stack_frames). Byte-identical to
    ``scripts/v2_compressed.load_compressed`` on frames/poses/actions/episode_id
    (the ``.maneuvers`` field is left ``None`` -- deriving it needs
    ``physicalai.maneuvers_for_poses`` and no training window path reads it).
    Used by the tests as an import-light reference."""
    from tanitad.data.toy_driving import ToyEpisode
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_stack = int(d["n_stack"])
    k = n_stack - 1
    offs = _jpeg_offsets(d["jpeg_len"])
    frames = _decode_stacked(d["jpeg_buf"], offs, n_stack, 0, len(d["jpeg_len"]) - k)
    return ToyEpisode(frames=frames, actions=d["actions"][k:].float(),
                      poses=d["poses"][k:].float(), episode_id=int(d["episode_id"]),
                      maneuvers=None)
