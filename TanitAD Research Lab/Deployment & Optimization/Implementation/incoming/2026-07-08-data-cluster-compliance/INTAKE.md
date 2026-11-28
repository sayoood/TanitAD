# INTAKE — data-cluster compliance review #1: collision-safe epcache key + fail-fast save_episode

- **Package:** `Production & Optimization/Implementation/incoming/2026-07-08-data-cluster-compliance/`
- **Author agent / date:** production-optimization-agent, 2026-07-08
- **Proposed target:** `stack/tanitad/data/epcache.py` (replace) + `stack/tanitad/data/mixing.py` (`save_episode` only)
- **Hypothesis / WP served:** Production stream (D-020 §3); data-integrity hardening for the D-010 real+sim mix and the F-6/F-7 cache path
- **Research note:** `Production & Optimization/Research/2026-07-08-onnx-export-and-data-compliance.md`

## What & why (≤10 lines)

Compliance review #1 of the `tanitad/data/` cluster found two real, test-proven defects:

1. **Cache-key collision (`epcache.py:30-34`) — headline.** The per-source identity is
   `getattr(s,"name",None) or (s.get("clip_id") if dict else str(s))`. For a `Path`, `.name` is
   the **basename**, so `chunk_0/scene_000.hevc` and `chunk_1/scene_000.hevc` hash to the **same**
   cache dir — a relaunch silently loads the WRONG episodes. This is exactly the cosmos
   chunk-pairing failure class fixed in the loader on 2026-07-08 (PROJECT_STATE), still latent in
   the cache key. A dict source missing `clip_id` yields `None` for every id, so unrelated dict
   sets collide too. Proven live: both legacy sets hash to `37215f6f5632` / `6f33b4d44cec`.
   Fix: `_source_id` keys paths by **full** path, keys dicts by `clip_id` and **raises** when
   absent (no silent `None`), stable `repr` fallback otherwise.
2. **Silent persistence of mis-shaped episodes (`mixing.save_episode`).** A build item whose
   actions/poses length ≠ frames `T` (or non-4D frames) is written without complaint and detonates
   later inside a training window. Adds the cheap shape check at the write boundary — the same
   "fail here, not deep in training" doctrine `_contract.assert_contract` already encodes.

Both are minimal drop-ins; no behavior change on well-formed inputs; no new deps.

## Evidence & tests

- `tests/test_epcache_key.py` (7) + `tests/test_save_episode_validate.py` (5) — **12 passed in 1.62 s**
  on the RTX-4060 dev machine (py3.13, torch 2.11).
- Failing-then-passing is explicit: `*_legacy_collides` tests embed a faithful copy of the current
  stack logic and assert the collision **exists today**; the paired `*_fixed_*` tests assert the fix
  resolves it. Same pattern for the save-path (mis-shaped episode raises `ValueError`).
- Measured collision keys (legacy logic, reproduced live): chunk dirs → `37215f6f5632` (identical);
  dicts w/o clip_id → `6f33b4d44cec` (identical).

## Risk & rollback

- **Blast radius:** `epcache.build_episodes_cached` cache-dir naming changes for path/dict sources,
  so existing on-disk caches keyed by the OLD scheme are not re-used → **one rebuild** on first run
  after integration (cost: the ~40-min decode, one time; the cache then persists). No training-loop
  or contract change. `save_episode` gains 3 shape guards; well-formed callers unaffected.
- **Rollback:** revert the two files; old caches were never deleted.
- **Note for orchestrator:** if cross-machine cache-key portability is desired (pod vs local), key
  paths by a repo-relative or corpus-relative path instead of absolute — flagged, not decided here.

## Lower-priority findings (NOT in this package — logged for a later review, kept small)

- `epcache.py` `DONE` marker is written but **never read**; resume is per-file. Docstring corrected
  in this package's copy; behavior unchanged.
- `EpisodeWindowDataset.__init__` (`_contract.py:120`): an episode shorter than `window+max_horizon`
  contributes **0 windows silently** (no log/counter) — observability gap, candidate for review #2.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** **integrate-with-changes**
- **Date / by:** 2026-07-08 night / MVP orchestrator (loop)
- **Reason & notes:** Both defects real and well-evidenced; the cache-key collision is exactly the
  cosmos chunk-pairing class. ONE integration change: added a **read-only legacy-dir fallback**
  (`_legacy_cache_key`) so the ~138 GB of pre-existing pod caches are reused instead of rebuilt —
  new-keyed dir empty + legacy-keyed dir has episodes ⇒ use legacy with a warning; new builds
  always use collision-safe keys (+1 test). Cross-machine portability flag noted: pods share the
  /workspace layout so keys match pod-to-pod; local differs but never builds big caches.
  Lower-prio findings (DONE marker, silent short-episode drop) stay on your review-#3 backlog.
- **Integrated as:** `stack/tanitad/data/epcache.py` (replaced) + `mixing.save_episode` guards +
  `stack/tests/{test_epcache_key,test_save_episode_validate}.py` (imports rewired to stack paths);
  suite 178 green.
