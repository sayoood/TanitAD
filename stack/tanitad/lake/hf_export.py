"""License-scope-guarded HF exporter for the owned-safe view (spec §6.2).

SCAFFOLD ONLY — Phase A does NOT push. ``export_hf`` resolves the owned-safe
view, runs the export guard (a HARD assertion that every row is in scope +
commercial-clean), and STAGES the HF dataset bundle locally (shards + a filtered
catalog + DATA_CARD/MANIFEST/NOTICE). Uploading is gated behind an explicit
``push=True`` AND a ``confirm`` token; the default path stops before any network
call and returns the staged directory for review.

The exporter cannot select gated-confidential (not in the lake) or nc-research
(the guard blocks it) — the AV firewall is an invariant of the store topology,
not a discipline to remember.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tanitad.lake.license_guard import verify_license_scope
from tanitad.lake.view import LakeView, owned_safe_commercial_view


def _data_card(repo_id: str, members: list[dict], sources: dict) -> str:
    import collections
    n = len(members)
    per_source = collections.Counter(m.get("source") for m in members)
    per_split = collections.Counter(m.get("split") for m in members)
    # Per-source license line, counts drawn from the actual staged members.
    src_lines = "\n".join(
        f"| `{s}` | {sources.get(s, {}).get('license_name', '?')} "
        f"| `{sources.get(s, {}).get('license_class', 'owned-safe')}` "
        f"| {per_source[s]} |"
        for s in sorted(per_source))
    split_lines = ", ".join(f"{k}: {v}" for k, v in sorted(per_split.items()))
    return f"""---
license: other
task_categories: [robotics]
tags: [autonomous-driving, world-model, tanitad, ego-driving, camera]
---

# {repo_id} — TanitDataSet-C (commercial-clean tier)

The **commercially-clean, HF-publishable** tier of TanitDataSet: a camera-first
autonomous-driving corpus under one schema. Every record is `owned-safe` and
`commercial_ok` (permissive license, **no** share-alike, **no** gated/firewalled
or `refuse`-class source) — the tier is a per-record stamp derived structurally
from a per-source license CONSTANT, never inferred, and a hard export guard
refuses egress if a single row falls outside that scope.

**Episodes:** {n}  ({split_lines})

## Sources & licenses

| source | license | class | episodes |
|---|---|---|---|
{src_lines}

## Record schema (the world-model contract, D-015/D-016)

Each episode is the byte-identical contract every TanitAD adapter emits:

- `frames`  — `uint8 [T, 9, 256, 256]` — a 3-frame RGB stack (9 = 3×RGB),
  canonicalized to `f_eff ≈ 266 px` (D-016 geometry canon).
- `actions` — `f32 [T, 2]` — `(steer, accel)`, the action applied between t, t+1.
- `poses`   — `f32 [T, 4]` — `(x, y, yaw, v)` ego trajectory.
- per-episode metadata: `source`, `license_*`, `commercial_ok`, `sha256` of the
  frame blob, `build_params_hash`, native intrinsics, modality flags.

Stored as WebDataset tar shards (`{{id}}.frames.npy` / `.motion.npz` /
`.meta.json`); the reader re-verifies each frame blob's `sha256` on load
(verify-without-rebuild), so a rotted shard fails loudly instead of training on
garbage.

## Provenance & reproducibility

Every episode carries a `sha256` of its exact frame bytes and a
`build_params_hash` — a consumer can verify a shard member without rebuilding it,
and the build recipe travels with the data (`MANIFEST.json` / `NOTICE`).

## Notes for consumers

- **Split granularity.** Where a source's route/clip id is preserved the split is
  route-disjoint; for caches where only the episode survives, the split is
  **episode-level** — rebuild from origin if you need strictly route-disjoint
  train/val.
- **Near-duplicates & multi-traversal.** Records are NOT dropped for perceptual
  similarity. Homogeneous highway sources (e.g. comma2k19 — one commute route
  re-driven) contain wanted **multi-traversal** repeats; use the catalog's
  curation weights to control sampling frequency rather than deleting records.
- **Anonymization.** Real camera footage may contain faces/plates. Sources here
  are already publicly distributed under their stated licenses; still, run a
  face/plate check appropriate to your jurisdiction before any re-distribution.

_Staged by the TanitAD Phase-A HF exporter. Attribution/NOTICE ships alongside._
"""


def export_hf(lake_root: str | Path, repo_id: str, out_dir: str | Path,
              view: LakeView | None = None, require_commercial_ok: bool = True,
              push: bool = False, confirm: str | None = None,
              stage_shards: bool = True) -> dict[str, Any]:
    """Stage (and, only if explicitly confirmed, push) the owned-safe view to HF.

    Returns a summary of what was staged. With ``push`` falsy (the default) NO
    network call is made — this is the Phase-A scaffold. ``stage_shards=False``
    stages only the metadata bundle (DATA_CARD/MANIFEST/NOTICE) + runs the guard,
    skipping the (potentially large) shard copy — a fast guard/dry-run.
    """
    lake_root = Path(lake_root)
    view = view or owned_safe_commercial_view(lake_root)
    members = view.resolve()

    # --- layer-3 export guard: HARD scope assertion (spec §6) ---
    verify_license_scope(members, allowed_classes=set(view.scope),
                         require_commercial_ok=require_commercial_ok,
                         forbid_share_alike=require_commercial_ok,
                         context=f"hf_export:{repo_id}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_keys = sorted({m["shard_key"] for m in members})
    if stage_shards:
        # MIRROR the lake's partition layout (shards/<class>/<source>/<split>/
        # shard-NNNNN.tar). A flat basename copy is WRONG: train and val each
        # number from shard-00000, so basenames COLLIDE and silently drop/mix
        # shards across splits. Mirroring keeps every shard and its split.
        for sk in shard_keys:
            src = lake_root / sk
            dst = out / sk
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copyfile(src, dst)

    manifest = {}
    man = lake_root / "MANIFEST.json"
    if man.exists():
        manifest = json.loads(man.read_text())
    sources = manifest.get("sources", {})

    # UTF-8 explicitly: HF cards are UTF-8 and Windows' default cp1252 codec
    # cannot encode the card's math glyphs (≈, ×) — a silent-on-POSIX crash.
    (out / "DATA_CARD.md").write_text(_data_card(repo_id, members, sources),
                                      encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps(
        {"repo_id": repo_id, "episodes": len(members),
         "shards": shard_keys,               # full relative keys (unique)
         "sources": sources}, indent=2, sort_keys=True), encoding="utf-8")
    notice = lake_root / "NOTICE"
    if notice.exists():
        shutil.copyfile(notice, out / "NOTICE")

    summary = {
        "repo_id": repo_id, "staged_dir": str(out), "episodes": len(members),
        "shards": shard_keys, "n_shards": len(shard_keys),
        "shards_staged": bool(stage_shards),
        "guard": "passed (owned-safe, commercial_ok, SA-free)",
        "pushed": False,
    }

    if push:
        if confirm != f"PUSH {repo_id}":
            raise PermissionError(
                "HF push requires confirm == f'PUSH {repo_id}' AND explicit "
                "human authorization. Phase A does NOT push — refusing.")
        raise NotImplementedError(
            "Upload path intentionally unimplemented in the Phase-A scaffold. "
            "Wire huggingface_hub.HfApi().upload_folder here ONLY after Sayed's "
            "go + a private-repo create; keep the guard above upstream of it.")

    return summary
