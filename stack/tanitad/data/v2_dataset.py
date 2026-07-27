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
import hashlib
import os
import time
from collections import OrderedDict

import torch
import torchvision.io as tvio

from tanitad.data.calib import CanonicalFrame, subframe_slice
from tanitad.data.comma2k19 import stack_frames

MANIFEST_NAME = "_v2manifest.pt"
MANIFEST_VERSION = 3        # v2: + clip_id/episode_uid; v3: + per-clip image_h/
#                             image_w (NON-SQUARE frames, 2026-07-27). A v2
#                             sidecar is simply rebuilt — it is a derived cache.


# --------------------------------------------------------------------------- #
# Collision-free episode identity                                             #
# --------------------------------------------------------------------------- #
def stable_episode_id(clip_id: str) -> int:
    """A collision-free 63-bit episode id derived from the FULL ``clip_id``.

    ``v2_compressed.build_compressed`` stores
    ``episode_id = int.from_bytes(clip_id.encode()[:4], "big")`` -- the first
    **4 characters** of the UUID, i.e. 16 bits of entropy. MEASURED
    (``V2_CORPUS_QA.md`` P3): that collides for **609 of the 9 000** v2 clips
    (6.8 %, max multiplicity 4); the parity corpus has the same defect at 1.4 %.

    Training is unaffected -- the trainer emits ``episode_id`` in every window
    but never consumes it. It matters for anything that groups *by episode*:
    episode-disjoint splitting, and the decision-grade **episode-cluster
    bootstrap** (``taniteval.ci.episode_cluster_bootstrap``, which clusters on
    the unique values of its ``eid`` argument). Under the 16-bit id, 609 pairs
    of genuinely different clips are silently merged into one cluster, which
    **narrows the interval** -- the exact failure mode CLAUDE.md's "never quote
    an interval without its estimator" rule exists to prevent.

    63 bits (not 64) keeps the value inside torch's signed-int64 default
    collate. Collision probability over 9 000 clips is ~9000^2 / 2^64 ~= 4e-12.

    This is derived at **load** time rather than fixed in ``build_compressed``
    on purpose: it repairs the 9 000 clips already on disk with **no rebuild**,
    and it leaves every existing ``*.v2ep.pt`` byte-for-byte untouched, so the
    QA's build/load byte-identity proof still stands.
    """
    return int.from_bytes(
        hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(), "big") >> 1


# --------------------------------------------------------------------------- #
# Frame decode helpers (mirror load_compressed exactly, minus the build deps)  #
# --------------------------------------------------------------------------- #
def _jpeg_offsets(jpeg_len: torch.Tensor) -> torch.Tensor:
    """Prefix-sum byte offsets into the concatenated JPEG buffer (== the
    offsets :func:`load_compressed` derives)."""
    return torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(jpeg_len.to(torch.int64), 0)])


def _decode_stacked(jpeg_buf: torch.Tensor, offs: torch.Tensor, n_stack: int,
                    a: int, b: int, codec: str = "jpeg",
                    sl: "tuple[slice, slice] | None" = None) -> torch.Tensor:
    """Decode + D-015 channel-stack ONLY stacked-frame rows ``[a:b]``.

    ``stack_frames`` output row ``j`` = channel-concat of raw frames
    ``j, j+1, ..., j+n_stack-1``, so rows ``[a:b]`` need raw frames
    ``[a : b + n_stack - 1]``. Stacking is per-frame-independent, so decoding
    just that raw sub-block and stacking it is BIT-IDENTICAL to decoding the
    whole clip and slicing ``[a:b]`` (validated in tests + on real data).

    ``sl`` is the ``(rows, cols)`` CENTRED SUB-FRAME slice (see
    :class:`V2CompressedCache`). It is applied to each RAW frame before
    stacking, which is the order ``scripts/v2_compressed.load_compressed``
    uses; slicing spatial dims and concatenating channels commute, so the two
    orders are bit-identical and the cheaper one is used here."""
    k = n_stack - 1
    dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
    raw = [dec(jpeg_buf[int(offs[i]):int(offs[i + 1])],
               mode=tvio.ImageReadMode.RGB)
           for i in range(a, b + k)]                       # [3, H, W] u8 each
    if sl is not None:
        rs, cs = sl
        raw = [f[:, rs, cs] for f in raw]                  # [3, h, w] u8 each
    return stack_frames(torch.stack(raw), n_stack)         # [b-a, 3*n_stack, h, w] u8


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
def stored_frame_of(payload: dict) -> CanonicalFrame:
    """The :class:`CanonicalFrame` a ``*.v2ep.pt`` payload was BUILT at.

    Payloads written after the 2026-07-27 wide-FOV enablement carry the full
    ``frame`` dict; older ones carry only ``image_size`` (square, f_ref 266,
    pinhole by construction), so the fallback is exact rather than a guess."""
    fr = payload.get("frame")
    if isinstance(fr, dict):
        return CanonicalFrame.from_dict(fr)
    s = int(payload["image_size"])
    from tanitad.data.calib import F_REF
    return CanonicalFrame(height=int(payload.get("image_h", s)),
                          width=int(payload.get("image_w", s)), f_ref=F_REF)


class V2CompressedCache:
    """Owns one v2 cache dir: the clip filename list and a bounded LRU of loaded
    compressed payloads ``(jpeg_buf, offsets, n_stack, codec, slice)``. One instance
    is shared by all :class:`LazyV2Episode` of that dir.

    The LRU is per-PROCESS and never crosses the DataLoader-worker boundary (see
    ``__getstate__``): every worker fills its own, so total RAM is
    ``num_workers * lru_size * mean_payload`` (~2-4 MB/clip).

    ⭐ ``frame`` (opt-in, 2026-07-28) is THE RIG-CLEAN FIX ON THE TRAINING PATH.
    A CENTRED sub-frame of the geometry a clip was built at is a pure pixel
    slice of it, so passing e.g. ``PHYSICALAI_RIG_CLEAN_176x624`` against a
    ``256x640`` cache feeds the trainer EXACTLY the frames a rebuild at that
    geometry would produce — MEASURED bit-identical on 6 clips x 201 real frames
    of this very cache (``max_abs_diff 0``;
    ``…/incoming/2026-07-28-rig-clean-fix/raw/verify_val.json``). No rebuild, no
    re-emit, no second copy; the slice happens after the decode that already
    runs, so it is strictly cheaper than not slicing.

    ``None`` (the default) is byte-identical to the pre-2026-07-28 loader, which
    is what keeps pod1's running ``--v2-cache`` arm untouched.

    ⚠️ LOAD-BEARING PRECONDITION: the identity holds only for a LOSSLESS cache.
    Re-encoding a JPEG at a different crop offset moves the 8x8 blocks, so a
    slice of a lossy cache is NOT what a rebuild would produce. A lossy source
    is therefore REFUSED unless ``allow_lossy=True`` says the caller knows."""

    def __init__(self, cache_dir, lru_size: int = 64,
                 frame: CanonicalFrame | None = None,
                 allow_lossy: bool = False):
        self.cache_dir = str(cache_dir)
        self.lru_size = max(1, int(lru_size))
        self.files: list[str] = []
        self.frame = frame
        self.allow_lossy = bool(allow_lossy)
        self._lru: "OrderedDict[int, tuple]" | None = None

    # Pickling: drop the live LRU so a populated cache is never serialised into
    # each worker (they rebuild their own, starting empty).
    def __getstate__(self) -> dict:
        return {"cache_dir": self.cache_dir, "lru_size": self.lru_size,
                "files": self.files,
                "frame": None if self.frame is None else self.frame.to_dict(),
                "allow_lossy": self.allow_lossy}

    def __setstate__(self, s: dict) -> None:
        self.cache_dir = s["cache_dir"]
        self.lru_size = s["lru_size"]
        self.files = s["files"]
        fr = s.get("frame")
        self.frame = None if fr is None else CanonicalFrame.from_dict(fr)
        self.allow_lossy = bool(s.get("allow_lossy", False))
        self._lru = None

    def _slice_for(self, d: dict, path: str) -> "tuple[slice, slice] | None":
        """The ``(rows, cols)`` of THIS clip that are the requested sub-frame.

        Resolved per clip against the clip's OWN stored geometry, so a cache
        that mixes rasters cannot silently take one clip's slice on another."""
        if self.frame is None:
            return None
        stored = stored_frame_of(d)
        if self.frame == stored:
            return None
        if str(d.get("codec", "jpeg")) != "png" and not self.allow_lossy:
            raise ValueError(
                f"refusing to sub-frame a LOSSY cache: {path} has codec "
                f"{d.get('codec', 'jpeg')!r}. A centred slice equals a rebuild "
                f"at that geometry only for a LOSSLESS cache — re-encoding a "
                f"JPEG at a new crop offset moves the 8x8 blocks. Rebuild at "
                f"the sub-frame, or pass allow_lossy=True to state that an "
                f"approximate crop is what you want.")
        return subframe_slice(stored, self.frame)   # refuses a non-slice

    def _payload(self, clip_idx: int) -> tuple:
        if self._lru is None:
            self._lru = OrderedDict()
        hit = self._lru.get(clip_idx)
        if hit is not None:
            self._lru.move_to_end(clip_idx)
            return hit
        path = os.path.join(self.cache_dir, self.files[clip_idx])
        d = torch.load(path, map_location="cpu", weights_only=False)
        payload = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]),
                   int(d["n_stack"]), str(d.get("codec", "jpeg")),
                   self._slice_for(d, path))
        self._lru[clip_idx] = payload
        while len(self._lru) > self.lru_size:
            self._lru.popitem(last=False)
        return payload

    def decode_stacked_range(self, clip_idx: int, a: int, b: int) -> torch.Tensor:
        jpeg_buf, offs, n_stack, codec, sl = self._payload(clip_idx)
        return _decode_stacked(jpeg_buf, offs, n_stack, a, b, codec, sl)


# --------------------------------------------------------------------------- #
# Manifest (cheap, cached) + provider construction                            #
# --------------------------------------------------------------------------- #
def _scan_meta(path: str) -> tuple:
    """Metadata-only read of one clip: ``(poses[k:], actions[k:], episode_id,
    n_stack, image_size, clip_id)``. ``mmap=True`` pages in ONLY the small
    pose/action storages -- the multi-MB ``jpeg_buf`` is never touched -- and
    ``.clone()`` copies them off the mmap into owned resident tensors.

    ``clip_id`` is the full UUID string ``build_compressed`` stores alongside
    the 16-bit ``episode_id``; it is what :func:`stable_episode_id` hashes.
    Falls back to the filename stem for any payload predating that field.

    Geometry (2026-07-27): payloads written after the wide-FOV enablement carry
    ``image_h`` / ``image_w``; older ones carry only the scalar ``image_size``
    and are SQUARE by construction, so the fallback is exact, not a guess."""
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    k = int(d["n_stack"]) - 1
    poses = d["poses"][k:].clone().contiguous().float()
    actions = d["actions"][k:].clone().contiguous().float()
    clip_id = str(d.get("clip_id") or os.path.basename(path).split(".v2ep")[0])
    s = int(d["image_size"])
    h = int(d.get("image_h", s))
    w = int(d.get("image_w", s))
    return (poses, actions, int(d["episode_id"]), int(d["n_stack"]),
            s, clip_id, h, w)


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
    cid_l, uid_l, h_l, w_l = [], [], [], []
    t0 = time.time()
    for j, fn in enumerate(files):
        poses, actions, eid, n_stack, S, cid, H, W = _scan_meta(
            os.path.join(cache_dir, fn))
        poses_l.append(poses)
        act_l.append(actions)
        eid_l.append(eid)
        cid_l.append(cid)
        uid_l.append(stable_episode_id(cid))
        ns_l.append(n_stack)
        sz_l.append(S)
        h_l.append(H)
        w_l.append(W)
        tout_l.append(int(poses.shape[0]))
        if verbose and (j + 1) % 500 == 0:
            print(f"[v2] manifest {cache_dir}: {j + 1}/{len(files)} clips "
                  f"({time.time() - t0:.0f}s)", flush=True)
    man = {"version": MANIFEST_VERSION, "files": files, "poses": poses_l,
           "actions": act_l, "episode_id": eid_l, "n_stack": ns_l,
           "image_size": sz_l, "image_h": h_l, "image_w": w_l, "T_out": tout_l,
           "clip_id": cid_l, "episode_uid": uid_l}
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


def _assert_subframe_deliverable(cache_dir: str, files: list[str],
                                 h: int, w: int, frame: CanonicalFrame,
                                 allow_lossy: bool) -> dict:
    """Prove AT LAUNCH that ``frame`` is a real centred slice of this cache.

    ``_slice_for`` would catch a bad sub-frame too — but on the first window
    fetch, i.e. inside a DataLoader worker, minutes into a run. This raises in
    the launching process, before any GPU work, and it reads the FIRST CLIP'S
    OWN payload rather than the manifest, so the focal and projection (which
    the manifest does not carry) are checked against real bytes on disk."""
    probe = os.path.join(cache_dir, files[0])
    d = torch.load(probe, map_location="cpu", weights_only=False, mmap=True)
    stored = stored_frame_of(d)
    if (int(stored.height), int(stored.width)) != (int(h), int(w)):
        raise ValueError(
            f"{cache_dir}: the manifest says {h}x{w} but the payload declares "
            f"{stored.height}x{stored.width} — the cache disagrees with its own "
            f"sidecar; rebuild the manifest (rebuild=True).")
    codec = str(d.get("codec", "jpeg"))
    if codec != "png" and not allow_lossy:
        raise ValueError(
            f"refusing to sub-frame a LOSSY cache: {cache_dir} has codec "
            f"{codec!r}. A centred slice equals a rebuild at that geometry only "
            f"for a LOSSLESS cache. Pass allow_lossy=True to override.")
    rs, cs = subframe_slice(stored, frame)              # refuses a non-slice
    return {"parent": stored.to_dict(), "parent_tag": stored.tag(),
            "sub": frame.to_dict(), "sub_tag": frame.tag(),
            "rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop],
            "codec": codec, "bit_exact_slice": codec == "png"}


def build_v2_providers(cache_dirs, lru_size: int = 64, rebuild: bool = False,
                       verbose: bool = True,
                       stable_ids: bool = True,
                       frame: CanonicalFrame | None = None,
                       allow_lossy: bool = False) -> list[LazyV2Episode]:
    """Build the lazy provider list for one or more v2 cache dirs.

    ⭐ ``frame`` (2026-07-28) requests a CENTRED SUB-FRAME of whatever geometry
    the cache was built at — the zero-cost rig-clean fix. See
    :class:`V2CompressedCache`. Every provider's ``.frames.shape`` then reports
    the SLICED raster, which is what makes the trainer's existing geometry
    binding (``parity.assert_v2_geometry_matches``) able to prove the slice
    actually happened instead of merely being configured. ``None`` (default)
    is byte-identical to the pre-2026-07-28 behaviour.

    The returned list is a drop-in replacement for the raw episode list fed to
    ``FlagshipWindowDataset`` (via ``_wrap``): every element quacks like a
    ``ToyEpisode``. Providers from multiple dirs are concatenated (the
    consolidated-cache case -- e.g. pod1 bottom-half + pod3 top-half). Each dir
    keeps its OWN :class:`V2CompressedCache` (and LRU).

    ``stable_ids=True`` (default) gives each provider the collision-free
    :func:`stable_episode_id` of its full ``clip_id`` instead of the 16-bit
    ``episode_id`` baked into the payload -- see that function for why. This
    changes nothing about training (``episode_id`` is emitted but never
    consumed) and makes episode-clustered inference correct. It also matters
    for the **multi-dir** case specifically: the raw 16-bit ids collide ACROSS
    dirs too, so concatenating two shards under the old scheme would fuse
    unrelated clips from opposite shards into one bootstrap cluster.

    Pass ``stable_ids=False`` only to reproduce the exact ids stored at build
    time (e.g. when diffing against ``load_compressed``)."""
    if isinstance(cache_dirs, (str, os.PathLike)):
        cache_dirs = [cache_dirs]
    providers: list[LazyV2Episode] = []
    for cd in cache_dirs:
        cd = str(cd)
        man = load_or_build_manifest(cd, rebuild=rebuild, verbose=verbose)
        cache = V2CompressedCache(cd, lru_size=lru_size, frame=frame,
                                  allow_lossy=allow_lossy)
        cache.files = man["files"]
        key = "episode_uid" if (stable_ids and "episode_uid" in man) else "episode_id"
        _sl_rep = None
        for i in range(len(man["files"])):
            n_stack = int(man["n_stack"][i])
            S = int(man["image_size"][i])
            # NON-SQUARE support (2026-07-27): a manifest predating v3 carries no
            # image_h/image_w, and every such cache is square, so S is exact.
            H = int(man.get("image_h", [S] * len(man["files"]))[i])
            W = int(man.get("image_w", [S] * len(man["files"]))[i])
            if frame is not None and (H, W) != frame.hw:
                # Validate ONCE per dir against real bytes, then trust the
                # manifest's per-clip raster for the remaining shapes.
                if _sl_rep is None:
                    _sl_rep = _assert_subframe_deliverable(
                        cd, man["files"], H, W, frame, allow_lossy)
                H, W = frame.hw
            shape = torch.Size((int(man["T_out"][i]), 3 * n_stack, H, W))
            providers.append(LazyV2Episode(
                cache, i, man["poses"][i], man["actions"][i],
                int(man[key][i]), shape))
        if verbose:
            n_raw = len(set(int(x) for x in man["episode_id"]))
            _h0 = int(man.get("image_h", man["image_size"])[0])
            _w0 = int(man.get("image_w", man["image_size"])[0])
            if _sl_rep is not None:
                print(f"[v2] {cd}: SUB-FRAME {_sl_rep['sub_tag']} sliced from "
                      f"{_sl_rep['parent_tag']} at rows {_sl_rep['rows']} cols "
                      f"{_sl_rep['cols']} (codec {_sl_rep['codec']}, bit-exact "
                      f"slice {_sl_rep['bit_exact_slice']}) — the trainer sees "
                      f"{frame.height}x{frame.width}, not {_h0}x{_w0}",
                      flush=True)
                _h0, _w0 = frame.hw
            msg = (f"[v2] {cd}: {len(man['files'])} providers "
                   f"(channels {3 * int(man['n_stack'][0])}, "
                   f"{_h0}x{_w0} px), episode ids from '{key}'")
            if key == "episode_uid":
                msg += (f" [collision-free; the as-built 16-bit ids would give "
                        f"only {n_raw} distinct for {len(man['files'])} clips]")
            print(msg, flush=True)
    if verbose and len(providers):
        n_uniq = len(set(p.episode_id for p in providers))
        print(f"[v2] {len(providers)} providers, {n_uniq} distinct episode ids"
              + ("" if n_uniq == len(providers)
                 else f"  <-- WARNING: {len(providers) - n_uniq} COLLISIONS; "
                      "episode-clustered CIs over this set would be too narrow"),
              flush=True)
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
    frames = _decode_stacked(d["jpeg_buf"], offs, n_stack, 0,
                             len(d["jpeg_len"]) - k, str(d.get("codec", "jpeg")))
    return ToyEpisode(frames=frames, actions=d["actions"][k:].float(),
                      poses=d["poses"][k:].float(), episode_id=int(d["episode_id"]),
                      maneuvers=None)
