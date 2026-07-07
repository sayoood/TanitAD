"""Disk-backed episode cache (F-6).

Dataset builds decode ~1000 videos (~40 min); before this cache, any crash
after (or during) the build lost everything, and one corrupt clip could kill
the run. Episodes are built fault-tolerantly (per-item try/except) and
persisted per corpus/split; a relaunch loads them in seconds. The cache key
hashes the source list + build params, so changing the selection or the
preprocessing invalidates it automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from tanitad.data.mixing import load_episode, save_episode
from tanitad.data.toy_driving import ToyEpisode


def build_episodes_cached(sources: list, build_one: Callable[[object], ToyEpisode],
                          cache_root: str | Path, tag: str,
                          params: dict) -> list[ToyEpisode]:
    """Build (or load) episodes for `sources`, cached under cache_root.

    Layout: <cache_root>/<tag>-<key>/ep_00000.pt ... + DONE marker (written
    last — a crash mid-build leaves no DONE and the next run rebuilds).
    """
    ids = [getattr(s, "name", None) or (s.get("clip_id") if isinstance(s, dict)
                                        else str(s)) for s in sources]
    key = hashlib.sha1(json.dumps({"ids": ids, "params": params},
                                  sort_keys=True, default=str).encode()
                       ).hexdigest()[:12]
    d = Path(cache_root) / f"{tag}-{key}"
    d.mkdir(parents=True, exist_ok=True)

    # Per-SOURCE-INDEX files: a killed build resumes exactly where it stopped
    # (the overnight pod interruption lesson). skip_* markers remember corrupt
    # items so resumes do not retry them.
    eps: list[ToyEpisode] = []
    n_loaded = n_built = 0
    for i, src in enumerate(sources):
        f = d / f"ep_{i:05d}.pt"
        skip = d / f"skip_{i:05d}"
        if f.exists():
            eps.append(load_episode(str(f)))
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
            eps.append(ep)
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
