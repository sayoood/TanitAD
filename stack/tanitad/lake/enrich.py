"""Stage 1 — the SCHEMA EXTENSION + the per-episode enrichment compose
(TanitDataSet rev-3 §7.1).

The core tensors (``frames``/``actions``/``poses``) and the base ``LakeRecord`` /
catalog stay BYTE-FOR-BYTE unchanged — this module is purely additive:

- a per-episode **JSON sidecar** (``<lake>/sidecars/<id>.goal.json``) carrying the
  language/goal labels + the VLM-pending stubs, and
- an **enrichment catalog** (Parquet, one row per episode, partitioned by ``tier``)
  carrying the flat, queryable rev-3 columns (``tier``, ``rig_id``,
  ``crop_center_cy``, the quality bands, dedup, safety, holdout, curation weight,
  the kinematic goal tokens) — so "why is this clip in / weighted how much" is one
  column away and a training run is a *query*, not a bespoke pull.

``enrich_episode`` composes Stage-2 filtering + Stage-4 kinematic goal minting for
one episode; ``enrich_corpus`` adds the Stage-3 corpus-level curation. The VLM
augmentation is the ONLY stubbed step: ``scene_tags`` / ``lead_state`` /
``sign_reads`` / ``language.coc_trace`` are shape-fixed ``unknown`` skeletons the
deferred Cosmos-Reason2 pass fills in place (§8), with the goal's vlm/map slots
already carrying provenance ``unknown``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tanitad.lake import curation as CU
from tanitad.lake import filtering as FL
from tanitad.lake import goal_labels as GL
from tanitad.lake import vocab as V

SIDECAR_SUBDIR = "sidecars"
ENRICHMENT_SUBDIR = "enrichment"

# The kinematic labeler's stamp (the provenance card for every slot it mints).
KINEMATIC_LABELER = "scripts/refb_labels.py"
KINEMATIC_LABELER_VERSION = "v2-2026-07-18"     # curvature-relative labels-v2


# =========================================================================== #
# VLM-PENDING STUBS — the ONLY deferred step (shape-fixed for the Cosmos pass)  #
# =========================================================================== #
def vlm_pending_scene_tags() -> dict:
    """Scene-tag skeleton (prompt A) with every field ``unknown``/prov ``vlm`` and
    ``_pending=True``. Shape-fixed so the deferred pass fills it in place; the
    curation scene axis reads ``unknown`` from it today."""
    axes = ("weather", "time_of_day", "road_type", "surface", "traffic_density")
    st = {a: {"token": V.UNKNOWN, "prov": "vlm"} for a in axes}
    st["vru_present"] = {"token": V.UNKNOWN, "prov": "vlm"}
    st["notable_events"] = []
    st["_pending"] = True
    return st


def vlm_pending_lead_state() -> dict:
    """Lead-state skeleton (detector+monodepth / Cosmos-Reason2): feeds LONMODE lead
    modes / HEADWAY / VSOURCE=lead_constrained once filled. All ``None`` + pending."""
    return {"present": None, "gap_m": None, "closing_speed_ms": None,
            "ttc_s": None, "prov": "vlm", "_pending": True}


def vlm_pending_sign_reads() -> list:
    """Sign-read list (prompt E OCR): feeds VSOURCE=sign_limit + the VTARGET cap.
    Empty + a pending marker element the deferred pass replaces with real reads."""
    return [{"_pending": True, "prov": "vlm"}]


def vlm_pending_language() -> dict:
    """Reasoning skeleton: caption + Chain-of-Causation trace (prompt B) + QA, all
    pending. Rides the existing ``language`` field; the CoC ``physics_flag`` feeds
    the ``anomaly`` safety class once the deferred pass runs."""
    return {"caption": None,
            "coc_trace": {"observation": None, "critical_agents": [],
                          "justification": None, "decision": None,
                          "action": None, "physics_flag": None},
            "qa": [], "prov": "vlm", "_pending": True}


def kinematic_label_stamp(source_license: str) -> dict:
    """The provenance stamp for the kinematic pass (§5 governance format). The VLM
    pass appends its own stamp (model=cosmos-reason2, prompt vA/B/C/E) when it runs."""
    return {"kinematic": {"model": KINEMATIC_LABELER,
                          "model_version": KINEMATIC_LABELER_VERSION,
                          "prompt_version": None, "provenance": "kinematic",
                          "source_license": source_license},
            "vlm": {"_pending": True}}


# =========================================================================== #
# The enrichment record (the rev-3 schema extension)                           #
# =========================================================================== #
@dataclass
class EpisodeEnrichment:
    episode_id: int
    source: str = ""
    split_unit_id: str = ""
    # license/tier (derived from existing fields, §1)
    tier: str = "ship"
    # rig (per-clip, the two-rig fix)
    rig_id: str = ""
    crop_center_cy: float | None = None
    # quality (Stage-2 banded verdicts)
    quality: dict = field(default_factory=dict)
    # dedup (Stage-2 two-pass verdict)
    dedup: dict = field(default_factory=dict)
    # VLM-pending stubs
    scene_tags: dict = field(default_factory=vlm_pending_scene_tags)
    lead_state: dict = field(default_factory=vlm_pending_lead_state)
    sign_reads: list = field(default_factory=vlm_pending_sign_reads)
    language: dict = field(default_factory=vlm_pending_language)
    # the v3 goal tuple (kinematic slots minted; vlm/map slots unknown)
    goal: dict = field(default_factory=V.empty_goal)
    # curation (Stage-3; filled by the corpus pass)
    curation: dict = field(default_factory=dict)
    # governance
    label_stamp: dict = field(default_factory=dict)

    # -- serialization --
    def to_sidecar_dict(self) -> dict:
        """The full nested per-episode sidecar JSON (language/goal + all verdicts)."""
        return {
            "episode_id": int(self.episode_id),
            "source": self.source,
            "split_unit_id": self.split_unit_id,
            "tier": self.tier,
            "rig_id": self.rig_id,
            "crop_center_cy": self.crop_center_cy,
            "quality": self.quality,
            "dedup": self.dedup,
            "scene_tags": self.scene_tags,
            "lead_state": self.lead_state,
            "sign_reads": self.sign_reads,
            "language": self.language,
            "goal": self.goal,
            "curation": self.curation,
            "label_stamp": self.label_stamp,
            "goal_provenance": V.goal_provenance_summary(self.goal),
        }

    def to_catalog_row(self) -> dict:
        """The flat, queryable enrichment-catalog row (frame blob + nested labels
        excluded — those live in the sidecar)."""
        q, d, c = self.quality, self.dedup, self.curation

        def tok(slot):
            return self.goal.get(slot, V.unknown_slot())["token"]
        return {
            "episode_id": int(self.episode_id),
            "source": self.source,
            "tier": self.tier,
            "split_unit_id": self.split_unit_id,
            "rig_id": self.rig_id,
            "crop_center_cy": (float(self.crop_center_cy)
                               if self.crop_center_cy is not None else None),
            "corrupt": q.get("corrupt"),
            "blur_band": q.get("blur_band", ""),
            "exposure_band": q.get("exposure_band", ""),
            "truncation_band": q.get("truncation_band", ""),
            "truncation_frac": float(q.get("truncation_frac", 0.0)),
            "egomotion_sane": bool(q.get("egomotion_sane", True)),
            "phash": int(d.get("phash", 0)),
            "dedup_cluster_id": d.get("dedup_cluster_id", ""),
            "is_exemplar": bool(d.get("is_exemplar", True)),
            "multi_traversal": bool(d.get("multi_traversal", False)),
            "safety_event": c.get("safety_event"),
            "is_eval_holdout": bool(c.get("is_eval_holdout", False)),
            "curation_weight": float(c.get("weight", 1.0)),
            "goal_route": tok("ROUTE"),
            "goal_vtarget": tok("VTARGET"),
            "goal_vsource": tok("VSOURCE"),
            "goal_lonmode": tok("LONMODE"),
            "goal_latmaneuver": tok("LATMANEUVER"),
            "goal_dyn": tok("DYN"),
            "goal_kinematic_slots": V.goal_provenance_summary(self.goal)["kinematic"],
        }


def enrichment_arrow_schema():
    """Arrow schema for the enrichment catalog (mirrors ``catalog.py`` style;
    partitioned by ``tier``). A DERIVED index over the sidecars — rebuildable."""
    import pyarrow as pa
    return pa.schema([
        ("episode_id", pa.int64()),
        ("source", pa.string()),
        ("tier", pa.string()),
        ("split_unit_id", pa.string()),
        ("rig_id", pa.string()),
        ("crop_center_cy", pa.float32()),
        ("corrupt", pa.string()),
        ("blur_band", pa.string()),
        ("exposure_band", pa.string()),
        ("truncation_band", pa.string()),
        ("truncation_frac", pa.float32()),
        ("egomotion_sane", pa.bool_()),
        ("phash", pa.uint64()),                       # 64-bit unsigned aHash
        ("dedup_cluster_id", pa.string()),
        ("is_exemplar", pa.bool_()),
        ("multi_traversal", pa.bool_()),
        ("safety_event", pa.string()),
        ("is_eval_holdout", pa.bool_()),
        ("curation_weight", pa.float32()),
        ("goal_route", pa.string()),
        ("goal_vtarget", pa.string()),
        ("goal_vsource", pa.string()),
        ("goal_lonmode", pa.string()),
        ("goal_latmaneuver", pa.string()),
        ("goal_dyn", pa.string()),
        ("goal_kinematic_slots", pa.int32()),
    ])


# =========================================================================== #
# Per-episode compose (Stage 2 filtering + Stage 4 goal minting)               #
# =========================================================================== #
def enrich_episode(*, episode_id: int, source: str, frames: Tensor,
                   poses: Tensor, split_unit_id: str = "",
                   license_class: str = "owned-safe", share_alike: bool = False,
                   commercial_ok: bool = True, license_name: str = "",
                   cy: float | None = None, hz: float = 10.0,
                   has_can: bool = False, geo: tuple | None = None,
                   t: float | None = None) -> EpisodeEnrichment:
    """One episode → its :class:`EpisodeEnrichment` (everything but the corpus-level
    curation, which needs all records — see :func:`enrich_corpus`).

    Runs: tier derivation → quality/rig/egomotion assessment (Stage 2) → kinematic
    goal summary (Stage 4) → VLM-pending stubs. ``geo`` = ``(lat, lon)`` and ``t``
    (unix s) enable the cross-source GPS/time dedup at the corpus step."""
    from tanitad.lake import dedup as DD

    tier = FL.tier_of(license_class, share_alike, commercial_ok)
    qv = FL.assess_quality(frames, poses, source=source, cy=cy, hz=hz)
    goal = GL.episode_goal_summary(poses, has_can=has_can)
    enr = EpisodeEnrichment(
        episode_id=episode_id, source=source, split_unit_id=split_unit_id,
        tier=tier, rig_id=qv.rig_id, crop_center_cy=qv.crop_center_cy,
        quality=qv.to_dict(), goal=goal,
        label_stamp=kinematic_label_stamp(license_name or license_class))
    # per-episode perceptual hash + geo/time/poses stashed for the corpus passes
    # (Stage-2 dedup + Stage-3 curation need all clips; not serialized as columns)
    enr._phash = DD.clip_phash(frames)                            # type: ignore[attr-defined]
    enr._geo = geo                                                # type: ignore[attr-defined]
    enr._t = t                                                    # type: ignore[attr-defined]
    enr._poses = poses                                           # type: ignore[attr-defined]
    return enr


# =========================================================================== #
# Corpus compose (adds Stage 2 dedup + Stage 3 curation)                        #
# =========================================================================== #
def enrich_corpus(episodes: list[EpisodeEnrichment]) -> list[EpisodeEnrichment]:
    """Add the corpus-level passes to a list of per-episode enrichments (in place):
    the two-pass dedup (Stage 2) and the stratified curation + safety + holdout
    (Stage 3). Returns the same list, now fully enriched."""
    from tanitad.lake import dedup as DD

    # -- Stage 2 dedup (needs all clips) --
    dd_items = []
    for e in episodes:
        phash = int(getattr(e, "_phash", 0) or e.dedup.get("phash", 0))
        item = {"id": e.episode_id, "source": e.source, "phash": phash}
        geo = getattr(e, "_geo", None)
        if geo is not None:
            item["geo_cell"] = DD.geo_cell(geo[0], geo[1])
            item["t"] = getattr(e, "_t", 0.0) or 0.0
        dd_items.append(item)
    dd = DD.two_pass_dedup(dd_items)
    for e in episodes:
        e.dedup = dd[e.episode_id].to_dict()

    # -- Stage 3 curation (needs the stratum histogram) --
    cur_records = [{
        "id": e.episode_id, "split_unit_id": e.split_unit_id, "tier": e.tier,
        "goal": e.goal, "scene_tags": e.scene_tags, "lead_state": e.lead_state,
        "poses": getattr(e, "_poses", None),
    } for e in episodes]
    cur = CU.curate_corpus(cur_records)
    for e in episodes:
        e.curation = cur[e.episode_id].to_dict()
    return episodes


# =========================================================================== #
# Sidecar + enrichment-catalog I/O                                             #
# =========================================================================== #
def sidecar_dir(lake_root: str | Path) -> Path:
    return Path(lake_root) / SIDECAR_SUBDIR


def write_sidecar(lake_root: str | Path, enr: EpisodeEnrichment) -> Path:
    """Write ``<lake>/sidecars/<episode_id>.goal.json`` (the per-episode label
    sidecar). Deterministic (sorted keys) so a rebuild is reproducible."""
    d = sidecar_dir(lake_root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{int(enr.episode_id)}.goal.json"
    p.write_text(json.dumps(enr.to_sidecar_dict(), indent=2, sort_keys=True,
                            default=_json_default), encoding="utf-8")
    return p


def read_sidecar(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def write_enrichment_catalog(lake_root: str | Path,
                             episodes: list[EpisodeEnrichment],
                             run_id: str = "0") -> int:
    """Write the flat enrichment catalog (Parquet, Hive-partitioned by ``tier``);
    returns rows written. Mirrors ``catalog.CatalogWriter`` so a training run is a
    predicate over ``tier``/quality/curation columns."""
    import pyarrow as pa
    import pyarrow.dataset as pads

    rows = [e.to_catalog_row() for e in episodes]
    if not rows:
        return 0
    table = pa.Table.from_pylist(rows, schema=enrichment_arrow_schema())
    d = Path(lake_root) / ENRICHMENT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    pads.write_dataset(table, d, format="parquet", partitioning=["tier"],
                       partitioning_flavor="hive",
                       basename_template=f"enr-{run_id}-{{i}}.parquet",
                       existing_data_behavior="overwrite_or_ignore")
    return len(rows)


def run_enrichment(lake_root: str | Path, episodes: list[EpisodeEnrichment],
                   run_id: str = "0", write_sidecars: bool = True) -> dict:
    """Finalize a batch: corpus enrichment → sidecars + enrichment catalog. The CPU
    driver end of the pipeline (no GPU, no VLM). Returns a small summary."""
    enrich_corpus(episodes)
    if write_sidecars:
        for e in episodes:
            write_sidecar(lake_root, e)
    n = write_enrichment_catalog(lake_root, episodes, run_id=run_id)
    tiers: dict[str, int] = {}
    holdout = evt = 0
    for e in episodes:
        tiers[e.tier] = tiers.get(e.tier, 0) + 1
        holdout += int(e.curation.get("is_eval_holdout", False))
        evt += int(bool(e.curation.get("safety_event")))
    return {"episodes": len(episodes), "catalog_rows": n, "tiers": tiers,
            "eval_holdout": holdout, "safety_events": evt,
            "sidecars_dir": str(sidecar_dir(lake_root))}


# =========================================================================== #
# The real-lake driver — reads the existing catalog + shards (no decode)        #
# =========================================================================== #
def enrich_lake(lake_root: str | Path, filter_expr=None, run_id: str = "0",
                verify_sha256: bool = True) -> dict:
    """Run the whole NON-VLM pipeline over an already-ingested lake: resolve the
    catalog view → stream each shard ONCE (reusing ``iter_shard_samples``, no
    decode) → per-episode enrich (Stage 2 + 4) → corpus enrich (Stage 2 dedup +
    Stage 3 curation) → write sidecars + the enrichment catalog. Additive: the base
    catalog + shard blobs are never touched.

    ``filter_expr`` is a ``pyarrow.dataset`` predicate (e.g. tier/source scope);
    ``None`` enriches the whole lake. GPS/time cross-source dedup stays inactive
    until lat/lon are surfaced onto the record (poses are lake-local ENU) — the
    perceptual within-source pass is fully active regardless."""
    from collections import defaultdict

    from tanitad.lake.catalog import resolve_members
    from tanitad.lake.shards import iter_shard_samples

    members = resolve_members(lake_root, filter_expr)
    wanted_by_shard: dict[str, set[int]] = defaultdict(set)
    for m in members:
        wanted_by_shard[m["shard_key"]].add(int(m["episode_id"]))

    enrichments: list[EpisodeEnrichment] = []
    for shard_key, wanted in wanted_by_shard.items():
        for sample in iter_shard_samples(Path(lake_root) / shard_key,
                                         verify_sha256=verify_sha256):
            eid = int(sample["episode_id"])
            if eid not in wanted or sample["poses"] is None:
                continue
            meta = sample["meta"]
            cy = (meta.get("intrinsics_native") or {}).get("cy") \
                if meta.get("intrinsics_native") else None
            enrichments.append(enrich_episode(
                episode_id=eid, source=meta["source"], frames=sample["frames"],
                poses=sample["poses"], split_unit_id=meta.get("split_unit_id", ""),
                license_class=meta["license_class"],
                share_alike=bool(meta.get("share_alike", False)),
                commercial_ok=bool(meta.get("commercial_ok", False)),
                license_name=meta.get("license_name", ""),
                cy=cy, hz=float(meta.get("hz", 10.0)),
                has_can=bool(meta.get("has_can", False))))
    if not enrichments:
        raise ValueError(f"enrich_lake: view resolved to 0 enrichable episodes "
                         f"under {lake_root}")
    return run_enrichment(lake_root, enrichments, run_id=run_id)
