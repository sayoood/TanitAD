"""Disk-backed episode cache (F-6) — production-hardened cache key.

Compliance review #1 (Production & Optimization, 2026-07-08). PROPOSED replacement
for `stack/tanitad/data/epcache.py`. Only two things change vs the current file:

1. **Cache-key derivation is now collision-safe (the headline fix).** The current
   key builds each source's identity as
   ``getattr(s, "name", None) or (s.get("clip_id") if dict else str(s))``. For a
   ``Path`` source ``s.name`` is the BASENAME, so two clips with the same filename
   in different directories (``chunk_0/scene_000.hevc`` vs
   ``chunk_1/scene_000.hevc`` — exactly the cosmos chunk-pairing failure class,
   PROJECT_STATE 2026-07-08) produce the SAME cache key and a relaunch silently
   loads the WRONG episodes. A dict source missing ``clip_id`` yields ``None`` for
   every id, so unrelated dict-source sets also collide. Both are proven in
   ``tests/test_epcache_key.py`` against a faithful copy of the legacy logic.
   Fix: :func:`_source_id` keys path-like sources by their FULL path, keys dict
   sources by ``clip_id`` and RAISES if it is absent (fail-fast, never a silent
   ``None``), and falls back to a stable ``repr`` for anything else.

2. **DONE docstring corrected to match behavior.** DONE is written but never read;
   resume is per-file (``ep_%05d.pt`` present -> load). The old wording implied
   DONE gates rebuilds — it does not. Marked informational.

Behavior is otherwise byte-identical: same layout, same per-source fault tolerance,
same mmap disk-backed loading (F-7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from tanitad.data.mixing import load_episode, save_episode
from tanitad.data.toy_driving import ToyEpisode


def _source_id(s: object) -> str:
    """Unambiguous, collision-safe identity string for one build source.

    - dict  -> ``clip:<clip_id>``; RAISES ValueError if ``clip_id`` is absent
      (a silent ``None`` id is what let unrelated dict sets share a key).
    - Path  -> ``path:<full-posix-path>`` (FULL path, not the basename, so clips
      that share a filename across chunk directories stay distinct).
    - str   -> ``str:<value>`` (already unambiguous).
    - other -> ``name:<obj.name>`` when non-empty, else ``repr:<repr>``.

    Note: keying Path sources by full path makes the cache machine-local (as it
    already is — it lives under ``cache_root``); correctness beats cross-machine
    key portability for a local build cache.
    """
    if isinstance(s, dict):
        cid = s.get("clip_id")
        if cid is None:
            raise ValueError(
                f"epcache source is a dict without 'clip_id'; cannot build a "
                f"collision-safe cache key from {s!r}")
        return f"clip:{cid}"
    if isinstance(s, Path):
        return f"path:{s.as_posix()}"
    if isinstance(s, str):
        return f"str:{s}"
    name = getattr(s, "name", None)
    if name:
        return f"name:{name}"
    return f"repr:{s!r}"


def cache_key(sources: list, params: dict) -> str:
    """12-hex cache key over the ordered source identities + build params."""
    ids = [_source_id(s) for s in sources]
    return hashlib.sha1(
        json.dumps({"ids": ids, "params": params}, sort_keys=True,
                   default=str).encode()).hexdigest()[:12]


def build_episodes_cached(sources: list, build_one: Callable[[object], ToyEpisode],
                          cache_root: str | Path, tag: str,
                          params: dict) -> list[ToyEpisode]:
    """Build (or load) episodes for `sources`, cached under cache_root.

    Layout: ``<cache_root>/<tag>-<key>/ep_00000.pt ...``. Resume is per-source-file:
    a killed build restarts exactly where it stopped (existing ``ep_%05d.pt`` ->
    load; ``skip_%05d`` -> a previously-corrupt item, not retried). ``DONE`` is an
    informational summary written last; it is NOT read on resume.
    """
    key = cache_key(sources, params)
    d = Path(cache_root) / f"{tag}-{key}"
    d.mkdir(parents=True, exist_ok=True)

    ids = [_source_id(s) for s in sources]
    eps: list[ToyEpisode] = []
    n_loaded = n_built = 0
    for i, src in enumerate(sources):
        f = d / f"ep_{i:05d}.pt"
        skip = d / f"skip_{i:05d}"
        if f.exists():
            eps.append(load_episode(str(f), mmap=True))          # F-7: disk-backed
            n_loaded += 1
            continue
        if skip.exists():
            continue
        if i % 20 == 0:
            print(f"[{tag}] episodes {i}/{len(sources)} "
                  f"({n_loaded} from cache, {n_built} built) -> {d.name}",
                  flush=True)
        try:
            ep = build_one(src)
            save_episode(ep, str(f))
            # F-7: never keep the built tensor resident — reload disk-backed so
            # RAM stays ~one episode regardless of corpus size (62 GB cgroup).
            del ep
            eps.append(load_episode(str(f), mmap=True))
            n_built += 1
        except Exception as e:      # one bad clip must never kill the run (F-6)
            skip.write_text(f"{type(e).__name__}: {e}")
            print(f"[{tag}] skipping item {i} ({ids[i]}): "
                  f"{type(e).__name__}: {e}", flush=True)
    (d / "DONE").write_text(json.dumps(
        {"episodes": len(eps), "skipped": len(sources) - len(eps)}))
    print(f"[epcache] {tag}: {len(eps)} episodes ready "
          f"({n_loaded} cached, {n_built} built, "
          f"{len(sources) - len(eps)} skipped) at {d}", flush=True)
    return eps
