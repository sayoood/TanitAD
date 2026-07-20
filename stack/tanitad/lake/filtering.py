"""Stage 2 — FILTERING (TanitDataSet rev-3 §7.2): cheap→expensive, banded, and
verdicts written back onto the record (never a silent drop of hard-but-valid OOD).

Ordered exactly as the strategy: license/tier stamp (structural, first) →
corrupt-clip skip (extends the ``f09e44db`` parity skipset) → quality gates
(blur / exposure / truncation → BANDS, downweight not drop) → ego-motion & rig
sanity (the two-rig cy≈543/755 lesson + kinematic plausibility). Perceptual/GPS
dedup is the last, heavier pass and lives in ``dedup.py``.

Every function is pure over tensors + plain metadata (no I/O). Quality gates are
CPU-only: variance-of-Laplacian and exposure/occlusion fractions, no cv2 — the
9-channel D-015 frame stack's latest RGB frame is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

# =========================================================================== #
# 1. LICENSE → TIER (structural; derived from the existing schema fields, §1)  #
# =========================================================================== #
TIERS = ("ship", "ship-sa", "nc", "firewalled")


def tier_of(license_class: str, share_alike: bool, commercial_ok: bool) -> str:
    """Derive the record ``tier`` from the existing license fields (§1 table).

    ``ship`` = commercial_ok (owned-safe, not SA); ``ship-sa`` = owned-safe AND
    share_alike (ZOD — segregated copyleft shard); ``nc`` = nc-research;
    ``firewalled`` = gated-confidential (PhysicalAI-AV — never enters the lake,
    recipe-only). Derived, never inferred: it cannot drift from the license axis."""
    if license_class == "gated-confidential":
        return "firewalled"
    if license_class == "nc-research":
        return "nc"
    if license_class == "owned-safe":
        return "ship-sa" if share_alike else ("ship" if commercial_ok else "ship-sa")
    raise ValueError(f"unknown license_class {license_class!r}")


def tier_of_record(rec) -> str:
    """``tier_of`` on a :class:`~tanitad.lake.schema.LakeRecord`-like object."""
    return tier_of(rec.license_class, bool(rec.share_alike), bool(rec.commercial_ok))


# =========================================================================== #
# 2. CORRUPT-CLIP SKIP — the committed, deterministic skipset (§7.2 step 2)     #
# =========================================================================== #
# The parity skipset that the strict-rebuild discipline keys on. The known PhysicalAI
# skip (24 corrupt front-wide clips, parity key ``f09e44db``; strict-parity build key
# ``e438721ae894``) is RECORDED here as a per-source, committed artifact so a rebuild
# is deterministic. The concrete clip ids live pod-side with the gated dataset (not
# committed to this public repo); ``register_corrupt`` seeds them at build time and
# ``detect_corrupt`` MINTS new skips from content checks — both feed the same set.
PARITY_SKIP_KEY = "f09e44db"          # PhysicalAI 24-corrupt-clip parity marker
STRICT_PARITY_BUILD_KEY = "e438721ae894"

CORRUPT_SKIPSET: dict[str, set[str]] = {"physicalai_av": set()}


def register_corrupt(source: str, clip_id: str) -> None:
    """Add a known-corrupt clip id to the committed per-source skipset."""
    CORRUPT_SKIPSET.setdefault(source, set()).add(str(clip_id))


def is_skipped(source: str, clip_id: str) -> bool:
    return str(clip_id) in CORRUPT_SKIPSET.get(source, set())


def detect_corrupt(frames: Tensor | None, poses: Tensor | None = None,
                   *, black_mean: float = 1.0, frozen_std: float = 1e-3
                   ) -> str | None:
    """Content corruption check → the reason string (skip), or ``None`` (keep).

    Catches: zero-length, frames/poses length mismatch, NaN/Inf poses, all-black
    frames (mean luma ~0), all-frozen frames (no temporal variation — a stuck
    decode). These extend the parity skipset with newly-found corruption so the
    skip artifact stays complete + deterministic."""
    if frames is None or frames.shape[0] == 0:
        return "zero_length"
    if poses is not None:
        if poses.shape[0] != frames.shape[0]:
            return f"frame_count_mismatch(frames={frames.shape[0]},poses={poses.shape[0]})"
        if not torch.isfinite(poses).all():
            return "nonfinite_poses"
    fv = frames.float()
    if float(fv.mean()) <= black_mean:
        return "all_black"
    if frames.shape[0] >= 2:
        # temporal std per pixel, averaged — ~0 means every frame is identical
        if float(fv.std(dim=0).mean()) <= frozen_std:
            return "all_frozen"
    return None


# =========================================================================== #
# 3. QUALITY GATES — banded, not binary (§7.2 step 3)                          #
# =========================================================================== #
BLUR_BANDS = ("sharp", "soft", "blurred")
EXPOSURE_BANDS = ("ok", "dim", "bright", "extreme")
TRUNCATION_BANDS = ("clear", "partial", "heavy")

# Documented thresholds on the uint8 luma scale (0..255). Swept-tunable; chosen so
# motion blur at speed stays "soft" (SIGNAL, kept) and only a genuinely mushy frame
# is "blurred". A clip is a drop candidate only if >``blur_drop_frac`` of frames are
# below the hard floor — never a single-frame drop.
BLUR_SHARP_VAR = 120.0        # var-of-Laplacian >= this -> sharp
BLUR_SOFT_VAR = 20.0          # >= this -> soft; below -> blurred (hard floor)
EXPO_CLIP_LO, EXPO_CLIP_HI = 8.0, 247.0       # under/over-exposed pixel cutoffs
EXPO_DIM_FRAC, EXPO_BRIGHT_FRAC = 0.25, 0.25  # >this fraction clipped -> dim/bright
EXPO_EXTREME_FRAC = 0.5
TRUNC_PARTIAL, TRUNC_HEAVY = 0.08, 0.20       # static-occlusion area fractions

_LAPLACIAN = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def _latest_luma(frames: Tensor) -> Tensor:
    """The latest RGB frame of the D-015 stack → luma ``[T,1,H,W]`` float 0..255.

    Frames are ``[T, C, H, W]`` uint8 with ``C`` a multiple of 3 (3 stacked RGB
    frames = 9ch); the newest frame is the last 3 channels. A 1-channel (toy) or
    non-multiple-of-3 stack falls back to a channel mean."""
    f = frames.float()
    C = f.shape[1]
    rgb = f[:, C - 3:, :, :] if C >= 3 and C % 3 == 0 else f
    if rgb.shape[1] == 3:                              # Rec.601 luma
        w = torch.tensor([0.299, 0.587, 0.114], device=f.device).view(1, 3, 1, 1)
        return (rgb * w).sum(dim=1, keepdim=True)
    return rgb.mean(dim=1, keepdim=True)


def variance_of_laplacian(luma: Tensor) -> Tensor:
    """Per-frame focus measure: Var(∇²luma) over ``[T,1,H,W]`` → ``[T]``. Higher =
    sharper. The classic no-reference blur metric (no cv2 — a fixed 3×3 conv).

    VALID convolution (no padding): a zero-padded border would inject huge
    artificial edge responses that read a genuinely flat frame as 'sharp'. The
    1-px border is dropped instead, so a smooth/flat frame correctly measures ~0."""
    if luma.shape[-1] < 3 or luma.shape[-2] < 3:
        return luma.new_zeros(luma.shape[0])
    k = _LAPLACIAN.to(luma.dtype).to(luma.device)
    lap = F.conv2d(luma, k, padding=0)
    return lap.var(dim=(1, 2, 3))


def blur_band(frames: Tensor) -> tuple[str, float]:
    """(band, frac_below_hard_floor) for a clip. Band = the MEDIAN frame's focus
    bucket; the fraction of frames under the hard floor is what curation gates a
    drop on (kept here as a downweight signal, not an auto-drop)."""
    v = variance_of_laplacian(_latest_luma(frames))
    med = float(v.median())
    band = ("sharp" if med >= BLUR_SHARP_VAR else
            "soft" if med >= BLUR_SOFT_VAR else "blurred")
    frac_below = float((v < BLUR_SOFT_VAR).float().mean())
    return band, frac_below


def exposure_band(frames: Tensor) -> tuple[str, float, float]:
    """(band, under_frac, over_frac): fraction of clipped-dark / clipped-bright
    pixels over the clip. Night-blown / sun-washed clips are BANDED (downweighted),
    never dropped unless outside every night/glare stratum (curation's call)."""
    luma = _latest_luma(frames)
    under = float((luma <= EXPO_CLIP_LO).float().mean())
    over = float((luma >= EXPO_CLIP_HI).float().mean())
    if under >= EXPO_EXTREME_FRAC or over >= EXPO_EXTREME_FRAC:
        band = "extreme"
    elif over >= EXPO_BRIGHT_FRAC:
        band = "bright"
    elif under >= EXPO_DIM_FRAC:
        band = "dim"
    else:
        band = "ok"
    return band, under, over


def truncation_frac(frames: Tensor) -> tuple[str, float]:
    """(band, fraction) of the frame occluded by a STATIC obstruction — the cheap,
    honest kinematic-free proxy for ego-hood / wiper / rain-on-lens: pixels whose
    temporal std across the clip is ~0 (a fixed obstruction moves with the car, so
    it never changes) in the bottom band where the hood/wiper sits.

    NOTE: precise ego-hood / wiper / rain-on-lens segmentation is a DETECTOR/VLM
    task (deferred). This proxy gives the banded ``truncation_frac`` the schema +
    curation want now; the deferred pass can refine it in place."""
    if frames.shape[0] < 2:
        return "clear", 0.0
    luma = _latest_luma(frames)                        # [T,1,H,W]
    H = luma.shape[-2]
    bottom = luma[:, :, int(H * 0.75):, :]             # lower quarter (hood zone)
    static = (bottom.std(dim=0) <= 1.0).float().mean()  # near-constant pixels
    frac = float(static) * 0.25                        # scale to whole-frame area
    band = ("heavy" if frac >= TRUNC_HEAVY else
            "partial" if frac >= TRUNC_PARTIAL else "clear")
    return band, frac


# =========================================================================== #
# 4. EGO-MOTION & RIG SANITY (§7.2 step 4 — the rig/cy lesson that burned us)   #
# =========================================================================== #
# Per-source principal-point CLUSTERS for multi-rig detection. The PhysicalAI
# front-wide (and the Cosmos-DD synthetic sharing its rig geometry) is BIMODAL:
# cy≈543 (rig A, ~23% of clips) vs cy≈755 (rig B, ~77%) — a geometric-center crop
# is ~215px wrong for rig B. Split at the midpoint; set rig_id + crop_center_cy
# PER CLIP. Sources not listed are single-rig (rig_id = the source name).
RIG_CLUSTERS: dict[str, dict] = {
    "physicalai_av": {"a": 543.0, "b": 755.0, "split": 650.0},
    "cosmos_dd": {"a": 543.0, "b": 755.0, "split": 650.0},
}


def assign_rig(source: str, cy: float | None) -> tuple[str, float | None]:
    """(rig_id, crop_center_cy) for one clip. For a multi-rig source the per-clip
    principal point ``cy`` picks rig A vs B (the two-rig vertical fix); a single-rig
    source returns ``'{source}:mono'``. ``cy`` absent → ``'{source}:unknown'`` (the
    crop must then revert to geometric-center — flagged, not silently wrong)."""
    if cy is None:
        return f"{source}:unknown", None
    clust = RIG_CLUSTERS.get(source)
    if clust is None:
        return f"{source}:mono", float(cy)
    rig = "a" if cy < clust["split"] else "b"
    return f"{source}:rig_{rig}", float(cy)


# kinematic-plausibility gates (robust percentile checks, not single-sample —
# the "verify against the metric definition + multiple samples" false-alarm lesson)
MAX_ACCEL_MS2 = 8.0           # |accel| above this at the 99th pct -> implausible
MAX_JERK_MS3 = 50.0           # |jerk| 99th pct ceiling
NEG_SPEED_MS = -0.5           # speed below this = a sign/derivation error
TELEPORT_SLACK_M = 5.0        # |step displacement - v*dt| 99th pct ceiling


def _pctl(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.abs(), q)) if x.numel() else 0.0


def egomotion_sane(poses: Tensor, hz: float = 10.0) -> tuple[bool, list[str], dict]:
    """(sane, reasons, stats) kinematic-plausibility gate over poses ``[T,4]``.

    Rejects (via robust 99th-pct stats, so one noisy sample never trips it):
    ``|accel| > 8 m/s²``, implausible jerk, negative speed, and pose teleports
    (step displacement far from ``v·dt``). Verified against the metric DEFINITION
    (accel = Δv/Δt over the 10 Hz contract), not a single step."""
    reasons: list[str] = []
    T = poses.shape[0]
    if T < 3:
        return True, [], {"T": T, "note": "too short to judge; passed"}
    dt = 1.0 / float(hz)
    v = poses[:, 3]
    dv = (v[1:] - v[:-1]) / dt
    jerk = (dv[1:] - dv[:-1]) / dt
    disp = (poses[1:, :2] - poses[:-1, :2]).norm(dim=-1)
    expected = v[:-1] * dt
    teleport = (disp - expected).abs()

    a99 = _pctl(dv, 0.99)
    j99 = _pctl(jerk, 0.99)
    vmin = float(v.min())
    tp99 = _pctl(teleport, 0.99)
    if a99 > MAX_ACCEL_MS2:
        reasons.append(f"accel_p99={a99:.1f}>{MAX_ACCEL_MS2}")
    if j99 > MAX_JERK_MS3:
        reasons.append(f"jerk_p99={j99:.1f}>{MAX_JERK_MS3}")
    if vmin < NEG_SPEED_MS:
        reasons.append(f"neg_speed_min={vmin:.2f}")
    if tp99 > TELEPORT_SLACK_M:
        reasons.append(f"teleport_p99={tp99:.1f}>{TELEPORT_SLACK_M}")
    stats = {"accel_p99": round(a99, 3), "jerk_p99": round(j99, 3),
             "v_min": round(vmin, 3), "teleport_p99": round(tp99, 3), "T": T}
    return (not reasons), reasons, stats


# =========================================================================== #
# The bundled per-episode quality verdict                                      #
# =========================================================================== #
@dataclass
class QualityVerdict:
    corrupt: str | None                       # skip reason, or None (keep)
    blur_band: str
    blur_frac_below_floor: float
    exposure_band: str
    exposure_under_frac: float
    exposure_over_frac: float
    truncation_band: str
    truncation_frac: float
    egomotion_sane: bool
    egomotion_reasons: list[str] = field(default_factory=list)
    egomotion_stats: dict = field(default_factory=dict)
    rig_id: str = ""
    crop_center_cy: float | None = None

    def to_dict(self) -> dict:
        return {
            "corrupt": self.corrupt,
            "blur_band": self.blur_band,
            "blur_frac_below_floor": round(self.blur_frac_below_floor, 4),
            "exposure_band": self.exposure_band,
            "exposure_under_frac": round(self.exposure_under_frac, 4),
            "exposure_over_frac": round(self.exposure_over_frac, 4),
            "truncation_band": self.truncation_band,
            "truncation_frac": round(self.truncation_frac, 4),
            "egomotion_sane": self.egomotion_sane,
            "egomotion_reasons": list(self.egomotion_reasons),
            "egomotion_stats": dict(self.egomotion_stats),
            "rig_id": self.rig_id,
            "crop_center_cy": self.crop_center_cy,
        }


def assess_quality(frames: Tensor, poses: Tensor, *, source: str,
                   cy: float | None = None, hz: float = 10.0) -> QualityVerdict:
    """Run the full §7.2 step-2→4 gate for one episode → a banded verdict. Corrupt
    clips short-circuit (the quality bands are then irrelevant)."""
    corrupt = detect_corrupt(frames, poses)
    rig_id, crop_cy = assign_rig(source, cy)
    if corrupt is not None:
        sane, reasons, stats = egomotion_sane(poses, hz) if poses is not None \
            else (True, [], {})
        return QualityVerdict(corrupt, "blurred", 1.0, "extreme", 0.0, 0.0,
                              "heavy", 0.0, sane, reasons, stats, rig_id, crop_cy)
    bband, bfrac = blur_band(frames)
    eband, uf, of = exposure_band(frames)
    tband, tfrac = truncation_frac(frames)
    sane, reasons, stats = egomotion_sane(poses, hz)
    return QualityVerdict(None, bband, bfrac, eband, uf, of, tband, tfrac,
                          sane, reasons, stats, rig_id, crop_cy)
