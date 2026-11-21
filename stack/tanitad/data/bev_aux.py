"""WP-D — the BEV auxiliary TARGET: a polar agent-occupancy map with THREE states.

⭐ **What this is for.** WP-A (`E-READOUT-CEILING-1`) measured a dissociation: under
a PERFECT front-end the token grid carries a lot of BEV agent localisation
(AP 0.4713 at 16x40), but on the REAL planning-trained trunk **none of it
transfers across episodes** (test AP 0.027-0.034 against a 0.0325 marginal
control). ⇒ *the map does not yet contain agents*, and a waypoint-indexed
attention (DiffusionDrive coupling (1)) built on it would index into an empty
room. This module builds the supervision that would put agents INTO the map.

⛔ **BINDING — labels may use ego/agents/maps/future poses; INFERENCE IS
VISION-ONLY** (PI, 2026-08-03). Everything here is a TRAIN-TIME TARGET built from
``obstacle.offline`` cuboids. Nothing in this module is ever read at inference,
and the head that consumes it (``tanitad.refs.refc_bev_aux``) is removable with
**bit-identical** planner output — pinned by ``tests/test_bev_aux.py``. The
published precedent is PhyLatent's Physical State Grounding, *"used only during
training and not required by the planner"* (`LIT-2`, banked `2608.05720`).

WHY POLAR, AND NOT THE CARTESIAN [120, 64] RASTER
==================================================
``tanitad.data.bev_raster`` already builds a Cartesian ego raster and it stays the
programme's Cartesian target. This module is a **different address space**, chosen
because of one MEASURED fact and one geometric one:

* **MEASURED (WP-A §6.1):** mapping the Cartesian 0.5 m grid into the encoder's
  token grid puts a **median of 4 BEV cells (max 313) into one token cell**, and
  only **242 of 640** token cells receive any ground-plane BEV cell at all. A
  dense Cartesian BEV built by indexing an image-token grid therefore inherits a
  quantisation floor no training removes.
* **The corpus is CYLINDRICAL** (256x640, ``f_ref`` 305.577, HFOV **120°**), so
  the image column axis is **LINEAR IN AZIMUTH**. A target indexed by
  (range bin, azimuth column) is registered to the encoder's own columns with
  **zero resampling**: one target column per feature-map column, exactly.
  ⚠️ ⛔ That property is scoped to THIS corpus. On a pinhole projection the same
  code mis-registers every agent, and the pinhole formula
  ``2*atan((W/2)/f)`` gives **92.641°** here and looks entirely plausible
  (the retracted FOV error, and :data:`PINHOLE_TRAP_DEG` keeps the number where a
  reader can see it).

⇒ refcv5's ResNet map is **8x20** at ``--image-hw 256 640`` (stride 32,
``refc.py:294-296,333-337``) ⇒ **20 azimuth columns = 6.0°/column**. The default
spec below has ``n_az = 20`` for exactly that reason. ⛔ The "4 readout columns /
30° per bin" figure is a **v6/v7** fact (``SpatialGridReadout``) and REF-C has no
``SpatialGridReadout`` at all — do not quote it here (WP-A §1.2).

THE THIRD STATE — the whole point of this module
=================================================
⛔ A two-state target merges *seen-and-empty* with *agent-occluded*, and
supervising "empty" on an occupied-but-occluded cell teaches the trunk that
occluded space is free. That is the failure mode that matters for driving.

⛔⛔ **AND THE PROGRAMME'S EXISTING "occluded" FLAG IS NOT AN OCCLUSION FLAG.**
The join's per-agent ``occ`` column IS the camera FIELD-OF-VIEW mask: MEASURED
bit-identical to ``bev_raster.fov_mask``'s predicate, **0 of 7,680 cells
disagreeing at every half-angle tried** (``build_obstacle_join.py``
``P4_PREDICATE_IDENTITY`` / ``assert_occ_matches_fov_mask``,
``tests/test_p4_fov_predicate.py``). ``obstacle.offline`` itself carries **no
visibility column at all** (``bev_raster.agents_at_time``: *"occ = -1 always"*).
⇒ **object-object occlusion does not exist anywhere in what we hold and must be
DERIVED.** It is derived here, in BEV, from the cuboid footprints the join does
carry — see :func:`shadow_mask`.

Three states, and they are never collapsed:

===================  =====================================================
``OCC_FREE``   0.0   seen and empty
``OCC_AGENT``  1.0   an agent footprint covers this cell
``OCC_IGNORE`` mask  unobservable — not supervised, and NOT "free"
===================  =====================================================

``OCC_IGNORE`` is carried in a separate boolean ``mask`` (True = supervised), not
as a third value in the occupancy array, so a consumer that forgets the mask gets
a *wrong* number rather than a silently plausible one — and every metric in
:mod:`tanitad.refs.refc_bev_aux` takes the mask as a required argument.

WHAT REACHES ``OCC_IGNORE``, AND WHAT DOES NOT
-----------------------------------------------
* **NO_LABEL** — an absent ``(clip, frame)`` line in the join. The whole frame is
  IGNORE (``mask`` all-False). ⛔ NEVER "road clear": the join's labels span
  ~20 s while egomotion runs 48-140 s, so most frames of a long clip have no
  label at all (``build_obstacle_join.py`` docstring; join doc §4). An empty
  ``agents`` list IS a label — *labelled clear* — and is fully supervised.
* **AGENT-OCCLUDED** — :func:`shadow_mask`, derived by BEV ray-casting.
* **OUT OF FIELD** — ⭐ does not arise: every column of this grid is inside the
  120° field **by construction**, which is one reason the polar space is the
  honest one here. (The Cartesian grid loses 590 of 7,680 cells to it, all at
  x < 9.09 m — ``bev_raster`` §CAMERA-FIELD GEOMETRY.)
* ⚠️ **BELOW THE VERTICAL FIELD / under the ego's own hood** — a KNOWN
  unmodelled state, NOT silently handled. The join notes the eval frame's ~45°
  VFOV excludes ground-level agent centres only at **< ~2 m** range, i.e. less
  than one 2.5 m range bin. :attr:`PolarBEVSpec.r_min_m` exposes it as a named
  flag defaulting to **0.0** so the geometry is never altered without a record.

Pure numpy. No torch, no corpus, no pandas — unit-testable from literals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np

__all__ = [
    "OCC_FREE", "OCC_AGENT", "PINHOLE_TRAP_DEG", "PolarBEVSpec",
    "POLAR_DEFAULT", "azimuth_column", "range_bin", "sample_lattice",
    "polar_occupancy", "shadow_mask", "build_target", "target_census",
    "OCCLUSION_MODES",
]

OCC_FREE = 0.0
OCC_AGENT = 1.0

#: ⚠️ The number a pinhole formula gives on THIS corpus. It is here so a reader
#: who reaches for ``2*atan((W/2)/f)`` sees the value it would produce
#: (2*deg(atan(320/305.5774907364391))) beside the true **120.0°**. MEASURED
#: 2026-08-21; the rig's own name is ``camera_front_wide_120fov``.
PINHOLE_TRAP_DEG = 92.641

#: The occlusion policies. ``"mask"`` is the default and the pre-registered arm;
#: ``"none"`` is the DELIBERATE REGRESSION (it re-introduces the two-state merge
#: this module exists to remove) and must be asked for by name.
OCCLUSION_MODES = ("mask", "none")


@dataclass(frozen=True)
class PolarBEVSpec:
    """Polar (range x azimuth) target geometry, registered to the token columns.

    ``n_az`` MUST equal the encoder feature map's column count for the
    registration to be exact — refcv5 at ``--image-hw 256 640`` is **20**.
    ``hfov_deg`` is the rig's true field (**120.0**, cylindrical), never a
    pinhole derivation (:data:`PINHOLE_TRAP_DEG`).
    """

    n_az: int = 20                # = refcv5's 8x20 ResNet map column count
    n_rng: int = 24               # 24 x 2.5 m = 60 m, = bev_raster's x_fwd_m
    r_max_m: float = 60.0
    hfov_deg: float = 120.0
    r_min_m: float = 0.0          # near-band IGNORE (vertical FOV) — see module doc
    sub_r: int = 2                # sub-samples per cell, range axis
    sub_az: int = 3               # sub-samples per cell, azimuth axis

    def __post_init__(self):
        if self.n_az < 1 or self.n_rng < 1:
            raise ValueError("n_az and n_rng must be >= 1")
        if not (0.0 < self.hfov_deg <= 360.0):
            raise ValueError(f"hfov_deg out of range: {self.hfov_deg}")
        if self.sub_r < 1 or self.sub_az < 1:
            raise ValueError("sub_r and sub_az must be >= 1")
        if not (0.0 <= self.r_min_m < self.r_max_m):
            raise ValueError("need 0 <= r_min_m < r_max_m")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rng, self.n_az)

    @property
    def dr_m(self) -> float:
        return self.r_max_m / self.n_rng

    @property
    def daz_deg(self) -> float:
        return self.hfov_deg / self.n_az

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(shape=list(self.shape), dr_m=self.dr_m, daz_deg=self.daz_deg,
                 n_cells=self.n_rng * self.n_az)
        return d


POLAR_DEFAULT = PolarBEVSpec()


def azimuth_column(az_deg, spec: PolarBEVSpec = POLAR_DEFAULT):
    """Ego azimuth in DEGREES -> image column index, or ``-1`` outside the field.

    ⭐ Column 0 is the LEFTMOST image column, which is azimuth ``+hfov/2``,
    because the ego frame is **+x forward, +y LEFT** (MEASURED by the parked-car
    experiment, ``bev_raster.py`` / ``build_obstacle_join.py:15`` — NOT assumed;
    a sign error here does not crash and does not show in a loss curve, it
    teaches a MIRRORED world, which is exactly how E-DEC-18's build justified
    measuring it).

    This is the vectorised twin of ``tanitad.data.psg_targets.azimuth_column``
    with two deliberate differences: it takes an array, and it returns ``-1``
    rather than ``None`` outside the field. ⚠️ Agreement with that function is a
    CONSISTENCY check (two implementations of one formula), not a correctness
    check — the correctness check is analytic and lives in the tests: at
    ``hfov 120 / n_az 20``, ``+60 -> 0``, ``0 -> 10``, ``+30 -> 5``,
    ``-59.999 -> 19``.
    """
    a = np.asarray(az_deg, dtype=np.float64)
    half = spec.hfov_deg / 2.0
    col = np.floor((half - a) / spec.daz_deg).astype(np.int64)
    out = np.where((a > half) | (a < -half), -1,
                   np.clip(col, 0, spec.n_az - 1))
    return out if out.ndim else int(out)


def range_bin(r_m, spec: PolarBEVSpec = POLAR_DEFAULT):
    """Range in METRES -> range-bin index, or ``-1`` outside ``[r_min, r_max)``."""
    r = np.asarray(r_m, dtype=np.float64)
    b = np.floor(r / spec.dr_m).astype(np.int64)
    out = np.where((r < spec.r_min_m) | (r >= spec.r_max_m) | (r < 0.0), -1,
                   np.clip(b, 0, spec.n_rng - 1))
    return out if out.ndim else int(out)


@lru_cache(maxsize=8)
def sample_lattice(spec: PolarBEVSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian ego coords of every sub-sample: ``(x, y)`` each
    ``[n_rng, n_az, sub_r * sub_az]``.

    Sub-sampling exists because these cells are COARSE (6.0° x 2.5 m at the
    default): a car 2 m wide at 40 m subtends 2.86°, less than half a column, so
    a single cell-CENTRE test would miss it in most columns and the target would
    be sparse and jittery in a way that is an artefact of the lattice rather than
    of the world. A cell is occupied iff ANY sub-sample falls inside a footprint.
    """
    kr = (np.arange(spec.sub_r, dtype=np.float64) + 0.5) / spec.sub_r
    ka = (np.arange(spec.sub_az, dtype=np.float64) + 0.5) / spec.sub_az
    i = np.arange(spec.n_rng, dtype=np.float64)[:, None, None, None]
    j = np.arange(spec.n_az, dtype=np.float64)[None, :, None, None]
    r = (i + kr[None, None, :, None]) * spec.dr_m                  # [R,A,sr,1]
    half = spec.hfov_deg / 2.0
    az = np.deg2rad(half - (j + ka[None, None, None, :]) * spec.daz_deg)
    r, az = np.broadcast_arrays(r, az)
    x = (r * np.cos(az)).reshape(spec.n_rng, spec.n_az, -1)
    y = (r * np.sin(az)).reshape(spec.n_rng, spec.n_az, -1)
    return x, y


def polar_occupancy(agents_frame, spec: PolarBEVSpec = POLAR_DEFAULT
                    ) -> np.ndarray:
    """Ego-frame agent footprints -> ``[n_rng, n_az] float32`` in {0, 1}.

    ``agents_frame`` is anything ``bev_raster.agents_to_array`` accepts (the
    join's ``{"cx","cy","yaw","l","w"[,"occ"]}`` dicts, or an ``[A, 5|6]``
    array), ALREADY in the ego frame (+x fwd, +y LEFT) — which is natively what
    ``obstacle.offline`` rig-frame rows are at their own timestamp.

    ⭐ **The footprint predicate is EXACTLY ``bev_raster.rasterize``'s** — the
    same rotate-into-the-agent-frame lines, so the two rasterisers cannot drift
    apart. Only the sample lattice differs (polar here, Cartesian there), and
    ``tests/test_bev_aux.py`` pins agreement on a shared point set.

    ⭐ An agent whose CENTRE is outside the 120° field but whose FOOTPRINT
    reaches inside is handled correctly and automatically, because the test is
    on the lattice samples, not on the centre. That is strictly better than the
    join's per-agent ``occ`` centre flag, which grades the centre only.
    """
    from tanitad.data.bev_raster import agents_to_array

    ag = agents_to_array(agents_frame)
    out = np.zeros(spec.shape, dtype=np.float32)
    if ag.shape[0] == 0:
        return out
    px, py = sample_lattice(spec)                 # [R, A, S] each
    for cx, cy, yaw, length, width, _occ in ag:
        r_circ = 0.5 * math.hypot(length, width)
        # cheap reject: circumscribed circle entirely beyond the grid
        if math.hypot(cx, cy) - r_circ > spec.r_max_m:
            continue
        dx = px - cx
        dy = py - cy
        c, s = math.cos(-yaw), math.sin(-yaw)     # bev_raster.rasterize:188-190
        lon = dx * c - dy * s
        lat = dx * s + dy * c
        hit = ((np.abs(lon) <= 0.5 * length)
               & (np.abs(lat) <= 0.5 * width)).any(axis=-1)
        out[hit] = 1.0
    return out


def shadow_mask(occ: np.ndarray, spec: PolarBEVSpec = POLAR_DEFAULT
                ) -> np.ndarray:
    """``[n_rng, n_az] bool`` — True where a cell is AGENT-OCCLUDED.

    THE RULE, stated because it is a choice and not a fact: along each azimuth
    column, find the nearest occupied cell and the end of its contiguous
    occupied run; **every cell beyond that run is occluded**, whatever its own
    ground truth.

    Two sub-decisions, both deliberate:

    1. **An occluder's own footprint is NOT self-occluded.** A vehicle is ~4.5 m
       long = ~2 range bins; marking its far half unobservable would delete most
       of the positive signal to model an effect a 2.5 m bin cannot resolve.
    2. **A cell in shadow is IGNORE even when the GT says an agent is there.**
       Supervising a hidden agent as OCCUPIED asks a monocular head to
       hallucinate; supervising it as FREE is the defect this module exists to
       remove. Neither is admissible, so it is not supervised at all.

    ⚠️ **What this approximation gets wrong, named rather than hidden:**
    (a) it is 2-D — the join drops ``z`` (``build_obstacle_join.py`` line
    schema), so a low ``stroller`` shadows a ``bus`` behind it; (b) the shadow
    starts at a CELL boundary, not at the true ray-exit range, so it is
    quantised to ``dr_m``; (c) only *labelled agents* occlude — buildings,
    walls and vegetation are not in ``obstacle.offline`` (10 classes, **all
    dynamic agents**), so a real urban scene is under-shadowed here.
    ⇒ this mask is a LOWER BOUND on true occlusion. Every number computed
    through it must be read that way.
    """
    r, a = occ.shape
    shadow = np.zeros((r, a), dtype=bool)
    for j in range(a):
        col = occ[:, j]
        nz = np.flatnonzero(col > 0.0)
        if nz.size == 0:
            continue
        i0 = int(nz[0])
        i1 = i0
        while i1 + 1 < r and col[i1 + 1] > 0.0:
            i1 += 1
        if i1 + 1 < r:
            shadow[i1 + 1:, j] = True
    return shadow


def build_target(agents_frame, *, labelled: bool = True,
                 spec: PolarBEVSpec = POLAR_DEFAULT,
                 occlusion: str = "mask") -> tuple[np.ndarray, np.ndarray]:
    """One frame -> ``(occ [n_rng, n_az] float32, mask [n_rng, n_az] bool)``.

    ``labelled`` is the join's own NO_LABEL/LABELLED distinction (the trainer's
    ``agent_label`` field): ``False`` -> **mask all-False**, i.e. the frame
    contributes nothing. ⛔ It must never be collapsed into an empty agent list;
    that is the *"an unlabelled frame is an empty road"* defect the join
    documentation calls out by name.

    ``occlusion``:
      * ``"mask"`` (default, the pre-registered arm) — shadowed cells are IGNORE.
      * ``"none"`` — ⛔ **DELIBERATE REGRESSION.** Two-state target: shadowed
        cells are supervised as FREE. This re-introduces exactly the merge that
        WP-A flagged and is the arm the WP-D gate must catch. It is shipped as a
        named arm rather than as a mutation, so the regression that must go RED
        is a thing anyone can run.
    """
    if occlusion not in OCCLUSION_MODES:
        raise ValueError(f"occlusion must be one of {OCCLUSION_MODES}, "
                         f"got {occlusion!r}")
    occ = polar_occupancy(agents_frame, spec)
    if not labelled:
        return np.zeros(spec.shape, dtype=np.float32), \
            np.zeros(spec.shape, dtype=bool)
    mask = np.ones(spec.shape, dtype=bool)
    if spec.r_min_m > 0.0:
        n_near = int(math.ceil(spec.r_min_m / spec.dr_m))
        mask[:n_near, :] = False
    if occlusion == "mask":
        mask &= ~shadow_mask(occ, spec)
    return occ, mask


def target_census(occ: np.ndarray, mask: np.ndarray) -> dict:
    """Per-frame counts. ⛔ **No accuracy anywhere**, by construction.

    Occupancy is ~1-2 % of cells, so an ALL-ZERO predictor scores ~98 %
    accuracy; every number here is a COUNT or a base RATE, and the metrics that
    consume them (``refc_bev_aux.bev_aux_metrics``) are AP / IoU / F1.
    """
    n = int(occ.size)
    n_sup = int(mask.sum())
    n_pos = int((occ[mask] > 0.0).sum()) if n_sup else 0
    n_pos_all = int((occ > 0.0).sum())
    return {
        "n_cells": n,
        "n_supervised": n_sup,
        "n_ignored": n - n_sup,
        "n_pos_supervised": n_pos,
        "n_pos_all": n_pos_all,
        "n_pos_hidden_by_mask": n_pos_all - n_pos,
        "base_rate_supervised": (n_pos / n_sup) if n_sup else float("nan"),
    }
