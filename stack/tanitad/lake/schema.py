"""Canonical SUPERSET schema for the TanitAD Data Lake (architecture spec §2).

One record == one episode (an independent recording unit — a comma2k19 segment,
a Cosmos/PhysicalAI clip). Frames + per-frame signals are ``T``-indexed inside
the record. A source that lacks a modality sets the field NULL *and* flips its
``modality_flags.*`` bit — consumers filter on the cheap flag, never touch null.

Two physical homes for a record's fields:

- **Core tensors** (``frames``/``actions``/``poses``) live in the WebDataset tar
  shard blob (``shards.py``). These are the exact contract emitted by every
  adapter today (``_contract.assemble_episode`` / ``comma2k19.build_episode`` /
  ``physicalai.build_episode``).
- **Everything else** (scalars, modality flags, native intrinsics, sha256, the
  shard pointer) is denormalized into the Parquet catalog (``catalog.py``) — one
  queryable row per episode.

``license_class`` is set by the ingestor from a per-source CONSTANT
(``SOURCE_REGISTRY``), never inferred, so it cannot drift (spec §6).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tanitad.data._contract import assert_contract
from tanitad.data.toy_driving import ToyEpisode

# --------------------------------------------------------------------------- #
# License axis (spec §2.2, §6) — first-class + structural                      #
# --------------------------------------------------------------------------- #
# ``refuse`` (added 2026-07-21, TANITDATASET_TIER_INTEGRATION §2): a class WORSE
# than gated-confidential. gated/nc constrain the DATA; ``refuse`` sources carry
# terms that follow the TRAINED WEIGHTS into the model/product (Waymax §2.e bars
# training any foundation model; Waymo Open / WOD-E2E terms reach vehicle
# operation), so the contamination survives training and NO tier — not even an
# internal-only one — can contain them. Its ingest handler raises exactly as
# gated-confidential does; it may never enter the lake on any tier.
LICENSE_CLASSES = ("owned-safe", "nc-research", "gated-confidential", "refuse")


@dataclass(frozen=True)
class SourceLicense:
    """Per-source license CONSTANT — the single source of truth for the license
    axis. Set once here; the ingestor copies it onto every record so a view can
    never mis-tag a source (spec §6: "never inferred, so it cannot drift")."""

    license_class: str          # one of LICENSE_CLASSES
    license_name: str           # human/NOTICE string, e.g. "MIT", "CC-BY-4.0"
    share_alike: bool           # copyleft virality (ZOD -> True); segregates shards
    is_synthetic: bool          # rendered vs real camera (D-010 role split)

    @property
    def commercial_ok(self) -> bool:
        """The commercial-clean gate: owned-safe AND not share-alike (spec §2.2)."""
        return self.license_class == "owned-safe" and not self.share_alike


# The permissive Phase-A sources + the AV firewall marker. comma2k19 is the real
# anchor (D-009); Cosmos-Drive-Dreams is the publicly-claimable synthetic arm
# (D-014). PhysicalAI-AV is listed ONLY so an ingestor that tries to admit it
# fails loudly — it must NEVER enter the lake (recipe-only; spec §3.3).
SOURCE_REGISTRY: dict[str, SourceLicense] = {
    "comma2k19": SourceLicense("owned-safe", "MIT", share_alike=False,
                               is_synthetic=False),
    "cosmos_dd": SourceLicense("owned-safe", "CC-BY-4.0", share_alike=False,
                               is_synthetic=True),
    # --- rev-3 permissive (TanitDataSet-C shippable core, §7.1) ---
    # tier `ship`: owned-safe, not share-alike -> commercial_ok, HF-publishable.
    "l2d": SourceLicense("owned-safe", "Apache-2.0", share_alike=False,
                         is_synthetic=False),          # yaak-ai L2D surround anchor
    "pandaset": SourceLicense("owned-safe", "CC-BY-4.0", share_alike=False,
                              is_synthetic=False),
    "udacity": SourceLicense("owned-safe", "MIT", share_alike=False,
                             is_synthetic=False),
    "worldmodel_synth": SourceLicense("owned-safe", "OpenMDW-1.1",
                                      share_alike=False, is_synthetic=True),
    # tier `ship-sa`: owned-safe AND share-alike -> segregated copyleft shard.
    "zod": SourceLicense("owned-safe", "CC-BY-SA-4.0", share_alike=True,
                         is_synthetic=False),
    # --- rev-3 NC (TanitDataSet-R research-only, tier `nc`, §3) ---
    "nuscenes": SourceLicense("nc-research", "CC-BY-NC-4.0", share_alike=False,
                              is_synthetic=False),
    "bdd100k": SourceLicense("nc-research", "BDD-NC", share_alike=False,
                             is_synthetic=False),
    "a2d2": SourceLicense("nc-research", "CC-BY-ND-4.0", share_alike=False,
                          is_synthetic=False),         # no-derivatives -> NC
    "argoverse2": SourceLicense("nc-research", "CC-BY-NC-SA-4.0",
                                share_alike=True, is_synthetic=False),
    "kitti360": SourceLicense("nc-research", "CC-BY-NC-SA-3.0",
                              share_alike=True, is_synthetic=False),
    "once": SourceLicense("nc-research", "ONCE-NC", share_alike=False,
                          is_synthetic=False),
    "drama": SourceLicense("nc-research", "Honda-HRI-NC", share_alike=False,
                           is_synthetic=False),
    "rank2tell": SourceLicense("nc-research", "Honda-HRI-NC", share_alike=False,
                               is_synthetic=False),
    # --- refuse: terms follow the trained WEIGHTS/model, not just the data;
    #     no tier can contain them (TANITDATASET_TIER_INTEGRATION §2). Present
    #     ONLY so an ingestor that tries to admit them fails loudly. ---
    "waymo": SourceLicense("refuse", "Waymo-Dataset-License",     # WOD / WOD-E2E
                           share_alike=False, is_synthetic=False),
    "waymax": SourceLicense("refuse", "Waymax-NC-License-2.e",    # bars FM training
                            share_alike=False, is_synthetic=False),
    # --- firewalled: present as a guard, never ingestible in Phase A ---
    "physicalai_av": SourceLicense("gated-confidential", "NVIDIA-AV-internal",
                                   share_alike=False, is_synthetic=False),
}

# Sources that may physically enter the Phase-A lake (permissive only).
# Excludes BOTH hard-fail classes: gated-confidential (firewalled, recipe-only)
# and refuse (weight-contaminating; never on any tier).
PERMISSIVE_SOURCES = tuple(
    s for s, lic in SOURCE_REGISTRY.items()
    if lic.license_class not in ("gated-confidential", "refuse"))


# --------------------------------------------------------------------------- #
# The record                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class LakeRecord:
    """One canonical episode record (superset of the current contract).

    ``frames``/``actions``/``poses``/``episode_id`` are the I-A core (byte-for-byte
    the current contract). All other fields are catalog metadata. Absent
    modalities are ``None`` with the matching ``modality_flags`` bit ``False``.
    """

    # --- I-A core (goes to the shard blob) ---
    episode_id: int
    frames: torch.Tensor                 # uint8 [T, C, S, S]
    actions: torch.Tensor | None         # f32 [T, 2] or None
    poses: torch.Tensor | None           # f32 [T, 4] or None

    # --- core metadata (spec §2.2) ---
    source: str = ""
    license_class: str = ""
    license_name: str = ""
    share_alike: bool = False
    commercial_ok: bool = False
    is_synthetic: bool = False
    T: int = 0
    channels: int = 9
    image_size: int = 256
    hz: float = 10.0
    f_eff_px: float = 266.0
    split_unit_id: str = ""              # I3 route/clip id (never split its windows)
    split: str = "unassigned"           # train | val | test | unassigned
    build_params_hash: str = ""
    sha256: str = ""                     # digest of the frames blob (verify-no-rebuild)
    attribution_id: str = ""             # key into the shard NOTICE

    # --- motion provenance (spec §2.3) ---
    action_source: str = "none"         # can | pose_derived | idm | vo | none
    has_can: bool = False

    # --- geometry / native intrinsics (spec §2.5; enables H17 re-derive) ---
    camera_model: str = "pinhole"       # pinhole | ftheta | kannala_brandt
    intrinsics_native: dict[str, Any] | None = None

    # --- language / VLA (spec §2.4; None for a pure world-model source) ---
    language: dict[str, Any] | None = None
    language_source: str = "none"

    # --- the availability bitmap — the primary view filter (spec §2.5) ---
    modality_flags: dict[str, bool] = field(default_factory=dict)

    # --- shard pointer (filled by the shard writer) ---
    shard_key: str = ""                 # relative shard path under the lake root
    member_key: str = ""                # tar member stem, == str(episode_id)

    def core_episode(self) -> ToyEpisode:
        """The I-A core as a ``ToyEpisode`` (for ``assert_contract`` / windows)."""
        return ToyEpisode(frames=self.frames, actions=self.actions,
                          poses=self.poses, episode_id=self.episode_id)


# --------------------------------------------------------------------------- #
# sha256 over the exact frame bytes (I-D verify-without-rebuild)               #
# --------------------------------------------------------------------------- #
def frames_sha256(frames: torch.Tensor) -> str:
    """Digest of the frame tensor's raw bytes (C-contiguous uint8).

    Deterministic: identical frames -> identical digest across machines, so a
    consumer can verify a shard member without rebuilding it from origin."""
    arr = np.ascontiguousarray(frames.cpu().numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Record assembly + validation                                                 #
# --------------------------------------------------------------------------- #
def assemble_lake_record(ep: ToyEpisode, source: str, split: str,
                         build_params_hash: str,
                         meta: dict[str, Any] | None = None,
                         language: dict[str, Any] | None = None,
                         split_unit_id: str | None = None,
                         action_source: str = "can",
                         has_can: bool | None = None) -> LakeRecord:
    """Assemble a validated :class:`LakeRecord` from a core episode + metadata.

    NULL-fills absent modalities and sets ``modality_flags`` consistently
    (spec §4.1 ``assemble_lake_record``). ``license_class``/``share_alike``/
    ``is_synthetic`` come from the per-source CONSTANT — never inferred.
    """
    lic = SOURCE_REGISTRY.get(source)
    if lic is None:
        raise ValueError(f"unknown source {source!r}; register it in "
                         f"SOURCE_REGISTRY with an explicit license class")
    if lic.license_class == "gated-confidential":
        raise PermissionError(
            f"source {source!r} is gated-confidential (license {lic.license_name}) "
            f"and MUST NOT enter the lake — recipe-only per spec §3.3. Refusing.")
    if lic.license_class == "refuse":
        raise PermissionError(
            f"source {source!r} is license_class 'refuse' (license "
            f"{lic.license_name}): its terms follow the TRAINED WEIGHTS into the "
            f"model/product (Waymax bars foundation-model training; Waymo Open / "
            f"WOD-E2E reaches vehicle operation), so contamination survives "
            f"training and NO tier — not even internal-only — can contain it. "
            f"MUST NEVER enter the lake. Refusing.")

    meta = dict(meta or {})
    T = int(ep.frames.shape[0])
    channels = int(ep.frames.shape[1])
    image_size = int(ep.frames.shape[-1])

    has_actions = ep.actions is not None
    has_poses = ep.poses is not None
    if has_can is None:
        has_can = (action_source == "can")

    flags = {
        "has_actions": bool(has_actions),
        "has_poses": bool(has_poses),
        "has_can": bool(has_can),
        "has_language": language is not None,
        "has_qa": bool(language and language.get("qa_pairs")),
        "has_maneuver": bool(language and language.get("maneuver_labels")),
        "has_intrinsics": meta.get("intrinsics_native") is not None,
    }

    rec = LakeRecord(
        episode_id=int(ep.episode_id),
        frames=ep.frames,
        actions=ep.actions,
        poses=ep.poses,
        source=source,
        license_class=lic.license_class,
        license_name=lic.license_name,
        share_alike=lic.share_alike,
        commercial_ok=lic.commercial_ok,
        is_synthetic=lic.is_synthetic,
        T=T,
        channels=channels,
        image_size=image_size,
        hz=float(meta.get("hz", 10.0)),
        f_eff_px=float(meta.get("f_eff_px", 266.0)),
        split_unit_id=str(split_unit_id if split_unit_id is not None
                          else ep.episode_id),
        split=split,
        build_params_hash=build_params_hash,
        sha256=frames_sha256(ep.frames),
        attribution_id=meta.get("attribution_id", lic.license_name),
        action_source=action_source,
        has_can=bool(has_can),
        camera_model=meta.get("camera_model", "pinhole"),
        intrinsics_native=meta.get("intrinsics_native"),
        language=language,
        language_source=(language.get("language_source", "none")
                         if language else "none"),
        modality_flags=flags,
        member_key=str(int(ep.episode_id)),
    )
    validate_superset(rec)
    return rec


def validate_superset(rec: LakeRecord, channels: int | None = None) -> None:
    """Validate a record: I-A core contract + flag<->field consistency (raises).

    ``channels=None`` accepts any channel count (comma/physicalai are 9); the
    world-model corpora are 9-channel D-015 stacks. The flag/field cross-check
    catches an ingestor that sets a modality flag without the data (or vice
    versa) at the WRITE boundary, not deep in a training window.
    """
    assert_contract(rec.core_episode(), channels=channels)

    f = rec.modality_flags
    if f.get("has_actions") != (rec.actions is not None):
        raise ValueError(f"ep {rec.episode_id}: has_actions flag "
                         f"{f.get('has_actions')} != (actions is not None)")
    if f.get("has_poses") != (rec.poses is not None):
        raise ValueError(f"ep {rec.episode_id}: has_poses flag "
                         f"{f.get('has_poses')} != (poses is not None)")
    if f.get("has_language") != (rec.language is not None):
        raise ValueError(f"ep {rec.episode_id}: has_language flag mismatch")
    if rec.license_class not in LICENSE_CLASSES:
        raise ValueError(f"ep {rec.episode_id}: bad license_class "
                         f"{rec.license_class!r}")
    if rec.has_can and rec.action_source != "can":
        raise ValueError(f"ep {rec.episode_id}: has_can but action_source="
                         f"{rec.action_source!r}")
    if not rec.sha256:
        raise ValueError(f"ep {rec.episode_id}: missing frames sha256 (I-D)")


# --------------------------------------------------------------------------- #
# Parquet catalog schema (pyarrow) — one row per episode (spec §3.1)           #
# --------------------------------------------------------------------------- #
def catalog_arrow_schema():
    """The Arrow schema for the catalog table (everything but the frame blob).

    Kept explicit + stable so the catalog is a DERIVED, rebuildable index over
    the shard ``meta.json`` (spec §8.3 risk 7: ``rebuild_catalog`` regenerates
    it from the tar sidecars, the single source of truth)."""
    import pyarrow as pa

    intr = pa.struct([
        ("model", pa.string()),
        ("params", pa.list_(pa.float32())),
        ("cx", pa.float32()), ("cy", pa.float32()),
        ("width", pa.int32()), ("height", pa.int32()),
    ])
    flags = pa.struct([
        ("has_actions", pa.bool_()), ("has_poses", pa.bool_()),
        ("has_can", pa.bool_()), ("has_language", pa.bool_()),
        ("has_qa", pa.bool_()), ("has_maneuver", pa.bool_()),
        ("has_intrinsics", pa.bool_()),
    ])
    return pa.schema([
        ("episode_id", pa.int64()),
        ("source", pa.string()),
        ("license_class", pa.string()),
        ("license_name", pa.string()),
        ("share_alike", pa.bool_()),
        ("commercial_ok", pa.bool_()),
        ("is_synthetic", pa.bool_()),
        ("T", pa.int32()),
        ("channels", pa.int32()),
        ("image_size", pa.int32()),
        ("hz", pa.float32()),
        ("f_eff_px", pa.float32()),
        ("split_unit_id", pa.string()),
        ("split", pa.string()),
        ("build_params_hash", pa.string()),
        ("sha256", pa.string()),
        ("attribution_id", pa.string()),
        ("action_source", pa.string()),
        ("has_can", pa.bool_()),
        ("camera_model", pa.string()),
        ("intrinsics_native", intr),
        ("language_source", pa.string()),
        ("modality_flags", flags),
        ("shard_key", pa.string()),
        ("member_key", pa.string()),
    ])


def record_to_catalog_row(rec: LakeRecord) -> dict[str, Any]:
    """Flatten a record into a catalog row dict (frame blob excluded)."""
    intr = rec.intrinsics_native
    intr_row = None
    if intr is not None:
        intr_row = {
            "model": intr.get("model", rec.camera_model),
            "params": [float(x) for x in intr.get("params", [])],
            "cx": float(intr.get("cx", 0.0)), "cy": float(intr.get("cy", 0.0)),
            "width": int(intr.get("width", 0)),
            "height": int(intr.get("height", 0)),
        }
    return {
        "episode_id": int(rec.episode_id),
        "source": rec.source,
        "license_class": rec.license_class,
        "license_name": rec.license_name,
        "share_alike": bool(rec.share_alike),
        "commercial_ok": bool(rec.commercial_ok),
        "is_synthetic": bool(rec.is_synthetic),
        "T": int(rec.T),
        "channels": int(rec.channels),
        "image_size": int(rec.image_size),
        "hz": float(rec.hz),
        "f_eff_px": float(rec.f_eff_px),
        "split_unit_id": rec.split_unit_id,
        "split": rec.split,
        "build_params_hash": rec.build_params_hash,
        "sha256": rec.sha256,
        "attribution_id": rec.attribution_id,
        "action_source": rec.action_source,
        "has_can": bool(rec.has_can),
        "camera_model": rec.camera_model,
        "intrinsics_native": intr_row,
        "language_source": rec.language_source,
        "modality_flags": dict(rec.modality_flags),
        "shard_key": rec.shard_key,
        "member_key": rec.member_key,
    }
