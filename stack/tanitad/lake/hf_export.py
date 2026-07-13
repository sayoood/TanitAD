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
    n = len(members)
    src_lines = "\n".join(
        f"- **{s}**: {info.get('license_name')} "
        f"(`{info.get('license_class')}`)"
        for s, info in sorted(sources.items()))
    return f"""---
license: other
tags: [autonomous-driving, world-model, tanitad]
---

# {repo_id}

TanitAD owned-safe (commercial-clean) episode view — canonical
`[T, 9, 256, 256]` uint8 frame stacks + `actions[T,2]` + `poses[T,4]`, the
byte-identical world-model contract (D-015/D-016).

**Episodes:** {n}

**Sources & licenses:**
{src_lines}

Every episode carries a `sha256` of its frame blob and a `build_params_hash`
(verify-without-rebuild / rebuild-from-origin — the recipe never dies).

_Staged by the Phase-A HF exporter scaffold. Not an official release._
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
        (out / "shards").mkdir(parents=True, exist_ok=True)
        for sk in shard_keys:
            src = lake_root / sk
            dst = out / "shards" / Path(sk).name
            if src.exists():
                shutil.copyfile(src, dst)

    manifest = {}
    man = lake_root / "MANIFEST.json"
    if man.exists():
        manifest = json.loads(man.read_text())
    sources = manifest.get("sources", {})

    (out / "DATA_CARD.md").write_text(_data_card(repo_id, members, sources))
    (out / "MANIFEST.json").write_text(json.dumps(
        {"repo_id": repo_id, "episodes": len(members),
         "shards": [Path(s).name for s in shard_keys],
         "sources": sources}, indent=2, sort_keys=True))
    notice = lake_root / "NOTICE"
    if notice.exists():
        shutil.copyfile(notice, out / "NOTICE")

    summary = {
        "repo_id": repo_id, "staged_dir": str(out), "episodes": len(members),
        "shards": [Path(s).name for s in shard_keys],
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
