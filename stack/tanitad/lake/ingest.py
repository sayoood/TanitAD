"""Write-once ingestion (spec §4): per-source ingestor + the run-once driver.

An ingestor WRAPS the existing ``build_episode`` adapter and adds metadata +
record assembly + shard write + license tag. The heavy lifting (decode ->
geometry-canon -> action/pose derivation) stays the UNCHANGED
``comma2k19.build_episode`` / ``cosmos_drive.build_episode`` — the lake is
additive, never a fork of the contract.

``ingest_source`` is idempotent (re-running skips episodes already in the
catalog) and fault-tolerant (one bad unit is skip-logged, never kills the run —
the ``epcache`` F-6 guarantee).

Two Phase-A intake paths, both producing IDENTICAL records:

- :class:`Comma2k19Ingestor` — ``build_core`` = ``comma2k19.build_episode`` on a
  raw segment (real PyAV decode). The canonical "wrap the adapter" ingestor.
- :class:`CachedEpisodeIngestor` — ingest already-built ``ep_*.pt`` (the memoized
  output of ``build_episode``). This is the spec's Phase-A "copy/normalize of
  *already-built* episodes; no decode on any pod" path — same records, no video.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tanitad.data.toy_driving import ToyEpisode
from tanitad.lake.catalog import CatalogWriter, catalog_dir
from tanitad.lake.schema import (SOURCE_REGISTRY, assemble_lake_record)
from tanitad.lake.shards import ShardWriter


def build_params_hash(params: dict) -> str:
    return hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Ingestor base                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class SourceIngestor:
    """Base ingestor. Subclasses set ``source`` and implement the 4 hooks
    (``discover`` / ``split_units`` / ``build_core`` / ``unit_meta``). The license
    class is looked up from the per-source CONSTANT in ``SOURCE_REGISTRY`` —
    never passed in, never inferred (spec §6)."""

    source: str = ""
    build_params: dict = field(default_factory=dict)
    action_source: str = "can"
    has_can: bool = True

    # -- hooks --
    def discover(self, root: str | Path) -> list[Any]:
        raise NotImplementedError

    def split_units(self, units: list[Any], seed: int = 0
                    ) -> dict[str, list[Any]]:
        raise NotImplementedError

    def build_core(self, unit: Any) -> ToyEpisode:
        raise NotImplementedError

    def unit_meta(self, unit: Any) -> dict[str, Any]:
        return {}

    def split_unit_id(self, unit: Any) -> str:
        return ""

    # -- derived --
    @property
    def license(self):
        lic = SOURCE_REGISTRY.get(self.source)
        if lic is None:
            raise ValueError(f"source {self.source!r} not in SOURCE_REGISTRY")
        return lic

    @property
    def params_hash(self) -> str:
        return build_params_hash({**self.build_params, "source": self.source})


# --------------------------------------------------------------------------- #
# The run-once driver                                                          #
# --------------------------------------------------------------------------- #
def _existing_episode_ids(lake_root: Path, source: str) -> set[int]:
    """Episode ids already in the catalog for this source (idempotent resume)."""
    if not (catalog_dir(lake_root)).exists():
        return set()
    try:
        import pyarrow.dataset as pads
        from tanitad.lake.catalog import open_catalog
        ds = open_catalog(lake_root)
        t = ds.to_table(filter=(pads.field("source") == source),
                        columns=["episode_id"])
        return set(t.column("episode_id").to_pylist())
    except Exception:
        return set()


def ingest_source(ing: SourceIngestor, root: str | Path, lake_root: str | Path,
                  seed: int = 0, max_units: int | None = None,
                  run_id: str = "0", verbose: bool = True) -> dict[str, Any]:
    """Run an ingestor ONCE: discover -> I3 split -> per-unit build/enrich/assemble
    -> SA-segregated shard write + catalog row. Idempotent + fault-tolerant.

    Returns a summary ``{source, license_class, per_split counts, shards,
    skipped, catalog_rows}``.
    """
    lake_root = Path(lake_root)
    lic = ing.license
    if lic.license_class == "gated-confidential":
        raise PermissionError(
            f"{ing.source!r} is gated-confidential — it must NEVER enter the "
            f"lake (recipe-only, spec §3.3). Refusing to ingest.")
    if lic.license_class == "refuse":
        raise PermissionError(
            f"{ing.source!r} is license_class 'refuse' — its terms follow the "
            f"trained weights into the model/product, so no tier can contain it "
            f"(TANITDATASET_TIER_INTEGRATION §2). Refusing to ingest.")

    units = ing.discover(root)
    if max_units is not None:
        units = units[:max_units]
    if not units:
        raise ValueError(f"no units discovered for {ing.source!r} under {root}")
    split_map = ing.split_units(units, seed=seed)

    already = _existing_episode_ids(lake_root, ing.source)
    catalog = CatalogWriter(lake_root, run_id=run_id)
    summary: dict[str, Any] = {
        "source": ing.source, "license_class": lic.license_class,
        "license_name": lic.license_name, "share_alike": lic.share_alike,
        "params_hash": ing.params_hash, "per_split": {}, "shards": [],
        "skipped": [], "catalog_rows": 0, "already_present": len(already),
    }

    for split, sunits in split_map.items():
        n_built = n_skip = n_dup = 0
        with ShardWriter(lake_root, lic.license_class, ing.source, split,
                         lic.share_alike) as w:
            for i, unit in enumerate(sunits):
                try:
                    ep = ing.build_core(unit)
                    if int(ep.episode_id) in already:
                        n_dup += 1
                        continue
                    meta = ing.unit_meta(unit)
                    rec = assemble_lake_record(
                        ep, source=ing.source, split=split,
                        build_params_hash=ing.params_hash, meta=meta,
                        split_unit_id=ing.split_unit_id(unit) or None,
                        action_source=ing.action_source, has_can=ing.has_can)
                    w.write(rec)                    # stamps shard_key/member_key
                    catalog.append(rec)
                    already.add(int(ep.episode_id))
                    n_built += 1
                except Exception as e:              # F-6: one bad unit never kills
                    summary["skipped"].append(
                        {"split": split, "i": i, "err": f"{type(e).__name__}: {e}"})
                    n_skip += 1
                    if verbose:
                        print(f"[lake:{ing.source}:{split}] skip {i}: "
                              f"{type(e).__name__}: {e}", flush=True)
                if verbose and n_built and n_built % 20 == 0:
                    print(f"[lake:{ing.source}:{split}] {n_built} episodes -> "
                          f"{w.written_shards[-1] if w.written_shards else '?'}",
                          flush=True)
            summary["shards"].extend(w.written_shards)
        summary["per_split"][split] = {
            "built": n_built, "skipped": n_skip, "duplicate": n_dup}
        if verbose:
            print(f"[lake:{ing.source}:{split}] done: {n_built} built, "
                  f"{n_dup} already-present, {n_skip} skipped", flush=True)

    summary["catalog_rows"] = catalog.flush()
    write_sidecars(lake_root, ing, summary)
    return summary


def write_sidecars(lake_root: Path, ing: SourceIngestor, summary: dict) -> None:
    """MANIFEST.json + NOTICE (I-D: the recipe travels with the data)."""
    lic = ing.license
    man = lake_root / "MANIFEST.json"
    manifest = {}
    if man.exists():
        try:
            manifest = json.loads(man.read_text())
        except Exception:
            manifest = {}
    manifest.setdefault("lake", "tanitad-lake")
    manifest.setdefault("phase", "A")
    manifest.setdefault("sources", {})
    manifest["sources"][ing.source] = {
        "license_class": lic.license_class, "license_name": lic.license_name,
        "share_alike": lic.share_alike, "is_synthetic": lic.is_synthetic,
        "build_params": ing.build_params, "build_params_hash": ing.params_hash,
        "per_split": summary.get("per_split", {}),
        "shards": summary.get("shards", []),
    }
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                   encoding="utf-8")

    notice = lake_root / "NOTICE"
    lines = ["TanitAD Data Lake - attribution / license NOTICE",
             "=" * 52, ""]
    for src, info in sorted(manifest["sources"].items()):
        lines.append(f"- {src}: {info['license_name']} "
                     f"(class={info['license_class']}, "
                     f"share_alike={info['share_alike']})")
    notice.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# comma2k19 ingestor — wraps comma2k19.build_episode (spec §4.1)               #
# --------------------------------------------------------------------------- #
@dataclass
class Comma2k19Ingestor(SourceIngestor):
    """Wraps ``comma2k19.build_episode`` (real PyAV decode + D-016 canon + CAN
    actions). License class comes from the constant (owned-safe / MIT)."""

    source: str = "comma2k19"
    size: int = 256
    stride: int = 2
    max_steps: int | None = 300
    n_stack: int = 3
    val_frac: float = 0.2
    decode_fn: Callable | None = None       # injectable for CI (no real video)

    def __post_init__(self):
        self.build_params = {"size": self.size, "n_stack": self.n_stack,
                             "stride": self.stride, "max_steps": self.max_steps,
                             "adapter": "comma2k19.build_episode"}
        self.action_source = "can"
        self.has_can = True

    def discover(self, root):
        from tanitad.data.comma2k19 import discover_segments
        return discover_segments(root)

    def split_units(self, units, seed=0):
        from tanitad.data.comma2k19 import split_by_route
        train, val = split_by_route(units, val_frac=self.val_frac, seed=seed)
        return {"train": train, "val": val}

    def build_core(self, unit) -> ToyEpisode:
        from tanitad.data.comma2k19 import build_episode
        kw = {"size": self.size, "stride": self.stride,
              "max_steps": self.max_steps, "n_stack": self.n_stack}
        if self.decode_fn is not None:
            kw["decode_fn"] = self.decode_fn
        return build_episode(unit, **kw)

    def split_unit_id(self, unit) -> str:
        from tanitad.data.comma2k19 import route_of
        return route_of(Path(unit))

    def unit_meta(self, unit) -> dict:
        from tanitad.data.calib import COMMA2K19_FOCAL_PX, F_REF
        # comma2k19 EON road camera: pinhole, 1164x874, f~910px (calib.py).
        return {
            "hz": 20.0 / self.stride,
            "f_eff_px": F_REF,
            "camera_model": "pinhole",
            "attribution_id": "comma2k19-MIT",
            "intrinsics_native": {
                "model": "pinhole",
                "params": [COMMA2K19_FOCAL_PX, COMMA2K19_FOCAL_PX],
                "cx": 1164 / 2.0, "cy": 874 / 2.0,
                "width": 1164, "height": 874,
            },
        }


# --------------------------------------------------------------------------- #
# Cached-episode ingestor — the no-decode "copy/normalize" Phase-A path        #
# --------------------------------------------------------------------------- #
@dataclass
class CachedEpisodeIngestor(SourceIngestor):
    """Ingest already-built ``ep_*.pt`` (the memoized output of ``build_episode``).

    ``discover`` takes a mapping ``{split: [ep_*.pt paths]}`` (passed as ``root``)
    so pre-split caches keep their split. Records are byte-identical to what the
    raw-video ingestor would produce — this is the spec's "copy/normalize of
    already-built episodes; no decode" path, used to stand the lake up fast from
    the existing real caches.
    """

    source: str = "comma2k19"
    size: int = 256
    n_stack: int = 3
    stride: int = 2
    intrinsics: dict | None = None

    def __post_init__(self):
        self.build_params = {"size": self.size, "n_stack": self.n_stack,
                             "stride": self.stride,
                             "adapter": "build_episode(cached ep_*.pt)"}

    def discover(self, root):
        # root is a dict {split: [paths]}; flatten while remembering the split.
        self._split_of: dict[str, str] = {}
        units: list[str] = []
        for split, paths in dict(root).items():
            for p in paths:
                sp = str(p)
                self._split_of[sp] = split
                units.append(sp)
        return units

    def split_units(self, units, seed=0):
        out: dict[str, list[str]] = {}
        for u in units:
            out.setdefault(self._split_of[u], []).append(u)
        return out

    def build_core(self, unit) -> ToyEpisode:
        from tanitad.data.mixing import load_episode
        return load_episode(str(unit), mmap=False)

    def split_unit_id(self, unit) -> str:
        # no route info in a bare cache; let assemble default the split unit to
        # the episode id (avoids a redundant per-unit reload — the driver already
        # loaded the episode in build_core).
        return ""

    def unit_meta(self, unit) -> dict:
        from tanitad.data.calib import COMMA2K19_FOCAL_PX, F_REF
        m = {"hz": 20.0 / self.stride, "f_eff_px": F_REF,
             "camera_model": "pinhole", "attribution_id": "comma2k19-MIT"}
        if self.intrinsics is not None:
            m["intrinsics_native"] = self.intrinsics
        else:
            m["intrinsics_native"] = {
                "model": "pinhole",
                "params": [COMMA2K19_FOCAL_PX, COMMA2K19_FOCAL_PX],
                "cx": 1164 / 2.0, "cy": 874 / 2.0, "width": 1164, "height": 874}
        return m


# --------------------------------------------------------------------------- #
# Cosmos-Drive-Dreams ingestor (SCAFFOLD) — wraps cosmos_drive.build_episode   #
# --------------------------------------------------------------------------- #
@dataclass
class CosmosDriveIngestor(SourceIngestor):
    """Wraps ``cosmos_drive.build_episode`` (CC-BY-4.0, synthetic; D-014). Ready
    to run; scaling to Cosmos-DD is: download the RDS-HQ shards from HF to the
    Drive, then ``ingest_source(CosmosDriveIngestor(), cosmos_root, lake)``.
    Actions are POSE-DERIVED (no CAN), so ``action_source='pose_derived'``."""

    source: str = "cosmos_dd"
    size: int = 256
    n_stack: int = 3
    val_frac: float = 0.2

    def __post_init__(self):
        self.build_params = {"size": self.size, "n_stack": self.n_stack,
                             "adapter": "cosmos_drive.build_episode"}
        self.action_source = "pose_derived"
        self.has_can = False

    def discover(self, root):
        from tanitad.data.cosmos_drive import discover_clips
        return discover_clips(root)

    def split_units(self, units, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)          # I3 clip-level split
        perm = torch.randperm(len(units), generator=g).tolist()
        n_val = max(1, int(len(units) * self.val_frac))
        val_idx = set(perm[:n_val])
        train = [u for i, u in enumerate(units) if i not in val_idx]
        val = [u for i, u in enumerate(units) if i in val_idx]
        return {"train": train, "val": val}

    def build_core(self, unit) -> ToyEpisode:
        from tanitad.data.cosmos_drive import build_episode
        return build_episode(unit, size=self.size, n_stack=self.n_stack)

    def split_unit_id(self, unit) -> str:
        return str(unit.get("clip_id", ""))

    def unit_meta(self, unit) -> dict:
        from tanitad.data.calib import (PHYSICALAI_FRONT_WIDE_HFOV_DEG, F_REF,
                                        nominal_focal_from_hfov)
        f = nominal_focal_from_hfov(1920, PHYSICALAI_FRONT_WIDE_HFOV_DEG)
        m = {"hz": 10.0, "f_eff_px": F_REF, "camera_model": "pinhole",
             "attribution_id": "cosmos-drive-dreams-CC-BY-4.0",
             "intrinsics_native": {
                 "model": "pinhole", "params": [f, f],
                 "cx": 1920 / 2.0, "cy": 1080 / 2.0, "width": 1920,
                 "height": 1080}}
        if unit.get("weather"):
            m["weather"] = unit["weather"]
        return m


# --------------------------------------------------------------------------- #
# L2D ingestor (yaak-ai/L2D) — wraps l2d.build_episode (Apache-2.0, tier ship)  #
# --------------------------------------------------------------------------- #
@dataclass
class L2DIngestor(SourceIngestor):
    """Wraps ``l2d.build_episode`` (LeRobot v3, Apache-2.0). The map + horizon +
    ego-indicator source. Records are drive-level de-duplicated
    (``groupby(session_id)`` + non-overlapping unix-time tiling) and split
    drive-disjoint (``split_unit_id = session_id``) — see ``data/l2d.py``.

    Paths follow the LeRobot layout under ``lerobot_root`` (a local mirror dir);
    ``discover`` takes the meta-episodes parquet as ``root``. Actions are
    POSE-DERIVED (L2D's CAN steering is normalized/unscaled) -> ``action_source=
    'pose_derived'``, ``has_can=False`` (the CAN turn_signal rides the SIGNAL vocab
    slot, not the action). ``frame_source`` = ``none`` (state-only), ``front_camera``
    (real pixels, ESTIMATED focal -> flagged intrinsics), or ``bev_map``
    (intrinsics-free)."""

    source: str = "l2d"
    size: int = 256
    n_stack: int = 3
    val_frac: float = 0.2
    lerobot_root: str = ""              # local mirror: has data/... and videos/...
    frame_source: str = "none"         # none | front_camera | bev_map
    episode_filter: Callable | None = None   # keep only locally-available episodes

    def __post_init__(self):
        self.build_params = {"size": self.size, "n_stack": self.n_stack,
                             "frame_source": self.frame_source,
                             "adapter": "l2d.build_episode",
                             "dedup": "drive_level(session_id)+nonoverlap_tiling"}
        self.action_source = "pose_derived"
        self.has_can = False

    def discover(self, root):
        from tanitad.data import l2d
        idx = l2d.read_episode_index(root)          # root = meta episodes parquet
        idx = l2d.dedup_index(idx)                  # drive-level de-dup (trap #2)
        if self.episode_filter is not None:
            idx = [e for e in idx if self.episode_filter(e)]
        return idx

    def split_units(self, units, seed=0):
        from tanitad.data import l2d
        return l2d.split_by_drive(units, val_frac=self.val_frac, seed=seed)

    def build_core(self, unit):
        from tanitad.data import l2d
        root = Path(self.lerobot_root)
        data_pq = root / l2d.data_rel_path(unit)
        vpath = None
        if self.frame_source == "front_camera":
            vpath = root / l2d.video_rel_path(unit, "front_left")
        elif self.frame_source == "bev_map":
            vpath = root / l2d.video_rel_path(unit, "map")
        return l2d.build_episode(unit, data_pq, size=self.size,
                                 n_stack=self.n_stack,
                                 frame_source=self.frame_source, video_path=vpath)

    def split_unit_id(self, unit) -> str:
        return str(unit["session_id"])              # drive-disjoint (I3)

    def unit_meta(self, unit) -> dict:
        from tanitad.data import l2d
        from tanitad.data.calib import F_REF
        instr = unit["tasks"][0] if unit.get("tasks") else ""
        route = l2d.parse_route_token(instr)
        dist_m = l2d.parse_distance_m(instr)
        m: dict = {"hz": 10.0, "attribution_id": "yaak-ai-L2D-Apache-2.0",
                   "canonical_name": unit.get("canonical_name"),
                   "session_id": unit.get("session_id"),
                   "instruction": instr, "route_from_instruction": route,
                   "routedist_m_from_instruction": dist_m}
        if self.frame_source == "front_camera":
            # L2D ships NO intrinsics; this focal is ESTIMATED (flagged) — never
            # asserted as the real f_eff. camera-scale UNVERIFIED per the arm card.
            m["camera_model"] = "pinhole"
            m["f_eff_px"] = F_REF
            m["intrinsics_native"] = {
                "model": "pinhole", "estimated": True,
                "hfov_deg_assumed": l2d.L2D_FRONT_HFOV_DEG_ASSUMED,
                "params": [l2d.estimated_front_focal_px(),
                           l2d.estimated_front_focal_px()],
                "cx": 1920 / 2.0, "cy": 1080 / 2.0, "width": 1920, "height": 1080,
                "note": "L2D ships no intrinsics; focal ESTIMATED from assumed HFOV"}
        elif self.frame_source == "bev_map":
            m["camera_model"] = "bev_render"
        return m
