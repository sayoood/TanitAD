"""Camera intrinsic canonicalization (D-016; the VLM3/H7 principle applied now).

Problem: comma2k19's road camera (~50 deg HFOV, f~910 px @ 1164x874) and
PhysicalAI-AV's front-wide (120 deg HFOV) have wildly different pixel<->metric
scales. A naive center-crop+resize feeds the world model inconsistent
action->pixel-motion geometry across corpora, corrupting exactly the dynamics
it must learn (and any metric probe on top).

Fix: crop each camera so the EFFECTIVE focal length at the model input size is
one shared constant, then resize. After cropping a centered square of side c
from an image with focal f_px and resizing to `size`:
    f_eff = f_px * size / c        =>       c = f_px * size / F_REF
F_REF is chosen so the reference camera (comma2k19) is (nearly) uncropped:
910 * 256 / 874 ~= 266.5 -> F_REF = 266. PhysicalAI front-wide then gets a
tighter central crop (~51 deg retained) — angularly consistent with comma;
the sacrificed wide periphery is precisely what H2 modality steering
re-introduces later as dedicated side views.

Extrinsics (mount height/pitch/roll) are NOT yet fully normalized (full pitch/
height homography is the R1 follow-up, Deep Think 8) — but the two-rig VERTICAL
principal-point split is fixed here: `ftheta_crop_resize(center="principal")`
centers the crop on each clip's per-clip (cx, cy), so the horizon lands at the
same output row for rig A (cy~543) and rig B (cy~755). Per-clip intrinsics from
PhysicalAI `calibration/` (loaded in data/physicalai.py) drive both the focal
canonicalization and this centering.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

F_REF = 266.0                     # effective focal [px] at the 256-px input
COMMA2K19_FOCAL_PX = 910.0        # EON road camera, 1164x874
PHYSICALAI_FRONT_WIDE_HFOV_DEG = 120.0

# comma2k19's ENTIRE horizontal field, MEASURED from its published intrinsics
# (1164x874, f=910): 2*atan(582/910) = 65.2027 deg. It is a HARD CEILING — no
# choice of F_REF, output size or projection can extract more field than the
# sensor recorded. Any canonical frame wider than this necessarily letterboxes
# (or drops) comma2k19. Stated here so the number is quotable from code.
COMMA2K19_MAX_HFOV_DEG = math.degrees(2.0 * math.atan(582.0 / COMMA2K19_FOCAL_PX))


def nominal_focal_from_hfov(width_px: int, hfov_deg: float) -> float:
    """Pinhole focal from horizontal FOV: f = W / (2 tan(HFOV/2))."""
    return width_px / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


# =========================================================================== #
# THE CANONICAL FRAME — one explicit object instead of ~10 (size, f_ref) pairs #
# =========================================================================== #
# WHY THIS EXISTS. Until 2026-07-27 the canonical input geometry was two module
# constants (``F_REF``) threaded as DEFAULT ARGUMENTS through ~10 functions
# (``size: int = 256, f_ref: float = F_REF``) and re-declared as a literal in
# every corpus module's ``CORPUS_META`` and every cache-build ``params`` dict.
# Changing the geometry therefore meant editing every call site, and MISSING ONE
# was silent — the exact failure class that produced this program's
# unreproducible v4 numbers (a changed ``vision_rank`` default made every
# committed number fail a STRICT load).
#
# ``CanonicalFrame`` is the single object that flows config -> cache build ->
# trainer -> encoder. ``CANONICAL_256`` is byte-for-byte today's geometry, and
# every function below keeps its old signature with the frame as an OPTIONAL
# keyword — so nothing changes unless a frame is explicitly passed.
#
# ⛔ NOTHING IN THIS FILE CHOOSES A GEOMETRY. Which frame to train on is the
# `2026-07-27-fov-crop-audit` measurement's decision.

PROJECTIONS = ("pinhole", "cylindrical")


@dataclass(frozen=True)
class CanonicalFrame:
    """The model's input geometry: output size, canonical focal, projection.

    ``height`` / ``width`` are the OUTPUT pixel dims (H, W) — independent, so a
    wide non-square frame (e.g. 256 x 640) is expressible. ``f_ref`` is the
    canonical focal in output px. ``projection`` fixes how an output pixel
    offset maps to a ray angle, and therefore how (width, f_ref) determine the
    field:

    - ``"pinhole"``   x = f_ref * tan(phi)   ->  HFOV = 2*atan((W/2)/f_ref)
      The legacy convention. Every number this program has published is this.
    - ``"cylindrical"`` x = f_ref * phi      ->  HFOV = 2*(W/2)/f_ref  (linear)
      Equidistant in azimuth; vertical stays pinhole per column. Standard for
      wide driving cameras — at a 50 deg half-angle ``tan 50 = 1.19``, so a
      pinhole rectification spends 19 % of its new pixels stretching the
      periphery, while equidistant azimuth spends them uniformly.

    ``CANONICAL_256 == CanonicalFrame()`` is the deployed frame. ``is_canonical``
    is what every parity-preserving branch keys on.
    """

    height: int = 256
    width: int = 256
    f_ref: float = F_REF
    projection: str = "pinhole"

    def __post_init__(self):
        if self.projection not in PROJECTIONS:
            raise ValueError(f"projection must be one of {PROJECTIONS}, "
                             f"got {self.projection!r}")
        if self.height < 8 or self.width < 8:
            raise ValueError(f"degenerate frame {self.height}x{self.width}")
        if not (self.f_ref > 0):
            raise ValueError(f"f_ref must be positive, got {self.f_ref}")

    # -- identity ---------------------------------------------------------- #
    @property
    def is_square(self) -> bool:
        return self.height == self.width

    @property
    def is_canonical(self) -> bool:
        """True iff this is EXACTLY the deployed 256/266/pinhole/square frame.

        Every parity-preserving branch in the codebase keys on this and on
        nothing else, so "canonical" can never drift to mean something looser.
        """
        return (self.height == 256 and self.width == 256
                and float(self.f_ref) == 266.0 and self.projection == "pinhole")

    @property
    def size(self) -> int:
        """Back-compat scalar for square frames only — raises otherwise.

        Deliberately FAILS LOUD on a non-square frame instead of silently
        returning one dimension: a caller that still thinks in one scalar is a
        call site that has not been converted, and that is what must surface.
        """
        if not self.is_square:
            raise ValueError(
                f"CanonicalFrame({self.height}x{self.width}) is not square — "
                f"this call site still assumes one scalar `size`. Use "
                f"`frame.height` / `frame.width` (or `frame.hw`).")
        return self.height

    @property
    def hw(self) -> tuple[int, int]:
        return (self.height, self.width)

    # -- angles ------------------------------------------------------------ #
    def half_angle_x_rad(self) -> float:
        """Retained HORIZONTAL half-angle of this frame."""
        r = (self.width / 2.0) / self.f_ref
        return math.atan(r) if self.projection == "pinhole" else r

    def half_angle_y_rad(self) -> float:
        """Retained VERTICAL half-angle. Pinhole in both projections — a
        cylinder is only equidistant in azimuth."""
        return math.atan((self.height / 2.0) / self.f_ref)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * self.half_angle_x_rad())

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * self.half_angle_y_rad())

    def focal_for_halfangle_x(self, theta: float) -> float:
        """The f_ref that would put ray ``theta`` exactly at the frame edge."""
        return (self.width / 2.0) / (math.tan(theta)
                                     if self.projection == "pinhole" else theta)

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def from_hfov(cls, hfov_deg: float, height: int, width: int,
                  projection: str = "pinhole") -> "CanonicalFrame":
        """Solve f_ref so this (H, W) frame retains exactly ``hfov_deg``.

        This is the constructor the FOV decision will use: state the field and
        the pixel budget, get the focal. It does NOT pick either.
        """
        theta = math.radians(hfov_deg) / 2.0
        f = (width / 2.0) / (math.tan(theta) if projection == "pinhole" else theta)
        return cls(height=height, width=width, f_ref=f, projection=projection)

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"height": int(self.height), "width": int(self.width),
                "f_ref": float(self.f_ref), "projection": self.projection}

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalFrame":
        return cls(height=int(d["height"]), width=int(d["width"]),
                   f_ref=float(d["f_ref"]),
                   projection=str(d.get("projection", "pinhole")))

    def tag(self) -> str:
        """Compact, stable, filesystem-safe identity — the cache-key fragment."""
        f = f"{float(self.f_ref):.4f}".rstrip("0").rstrip(".")
        return f"{self.height}x{self.width}f{f}{self.projection[:3]}"

    def report(self) -> dict:
        """Data-card row: what field this frame actually retains."""
        return {
            **self.to_dict(),
            "tag": self.tag(),
            "hfov_deg": round(self.hfov_deg, 3),
            "vfov_deg": round(self.vfov_deg, 3),
            "n_tokens_at_patch16": (self.height // 16) * (self.width // 16),
            "is_canonical": self.is_canonical,
            "comma2k19_hfov_ceiling_deg": round(COMMA2K19_MAX_HFOV_DEG, 3),
            "exceeds_comma2k19_field": self.hfov_deg > COMMA2K19_MAX_HFOV_DEG,
        }


#: The deployed frame. Identical to the pre-2026-07-27 constants in every path.
CANONICAL_256 = CanonicalFrame()


def as_frame(frame: "CanonicalFrame | None", size: int, f_ref: float
             ) -> "CanonicalFrame":
    """Resolve the (frame | legacy scalars) pair every geometry function takes.

    ``frame=None`` reconstructs the SQUARE frame the legacy ``(size, f_ref)``
    arguments describe, so an unconverted call site keeps producing exactly what
    it produced before. Passing BOTH a frame and non-default scalars is a
    programming error and is refused rather than silently resolved — that
    ambiguity is precisely how a stale default survives a refactor.
    """
    if frame is None:
        return CanonicalFrame(height=size, width=size, f_ref=f_ref)
    if size != 256 or float(f_ref) != float(F_REF):
        raise ValueError(
            f"both a CanonicalFrame ({frame.tag()}) and non-default legacy "
            f"scalars (size={size}, f_ref={f_ref}) were passed. Pass the frame "
            f"alone — the scalars are the pre-2026-07-27 spelling of the same "
            f"thing and having two sources of truth is the bug this object "
            f"exists to remove.")
    return frame


def geometry_params(frame: "CanonicalFrame | None" = None) -> dict:
    """Build-param fragment that separates geometries by CACHE KEY.

    ⚠️ PARITY-CRITICAL, and the exact discipline of
    :func:`physicalai.label_params`: the CANONICAL frame contributes an EMPTY
    dict, so ``epcache.cache_key`` hashes exactly what it hashes today and
    ``physicalai-train-e438721ae894`` keeps meaning precisely what it means.
    Only a NON-canonical frame adds a key — which is what makes a re-cropped
    cache structurally unable to collide with the parity cache.

    ⚠️ NOTE THE PRE-EXISTING HOLE THIS CLOSES: the build params carried
    ``{"size": ...}`` but NEVER ``f_ref``. Changing ``F_REF`` alone therefore
    produced DIFFERENT PIXELS UNDER THE SAME CACHE KEY — a silent collision.
    Never make this unconditional.
    """
    f = frame or CANONICAL_256
    return {} if f.is_canonical else {"geom": f.tag()}


def focal_crop_size(f_px: float, h: int, w: int, size: int,
                    f_ref: float = F_REF) -> int:
    """Centered-square crop side that yields f_eff == f_ref (clamped)."""
    c = int(round(f_px * size / f_ref))
    return max(32, min(c, min(h, w)))


def focal_crop_resize(vid: Tensor, f_px: float, size: int,
                      f_ref: float = F_REF,
                      frame: "CanonicalFrame | None" = None) -> Tensor:
    """[T, 3, H, W] (uint8 or float) -> [T, 3, H_out, W_out] uint8, canonical focal.

    Center crop of side focal_crop_size(...), then bilinear resize. Returns the
    achieved effective focal in `focal_crop_resize.last_f_eff` for data cards.

    ``frame`` (default None == the square legacy frame) generalises the crop to
    a RECTANGLE: ``c_w = f_px*W/f_ref``, ``c_h = f_px*H/f_ref``. The crop CLAMPS
    to the native frame, so asking a narrow camera for a wide field silently
    ZOOMS instead of widening — ``.last_clamped`` / ``.last_f_eff`` expose that,
    and :func:`pinhole_rectify` is the honest alternative (explicit unobserved
    mask). ⚠️ comma2k19's entire field is 65.2027 deg (``COMMA2K19_MAX_HFOV_DEG``);
    no crop can widen it.
    """
    t, _, h, w = vid.shape
    fr = as_frame(frame, size, f_ref)
    c_w = focal_crop_size(f_px, h, w, fr.width, fr.f_ref)
    c_h = focal_crop_size(f_px, h, w, fr.height, fr.f_ref)
    if fr.is_square:
        c_h = c_w = min(c_h, c_w)                 # legacy expression, exactly
    top, left = (h - c_h) // 2, (w - c_w) // 2
    out = vid[..., top:top + c_h, left:left + c_w].float()
    out = F.interpolate(out, size=(fr.height, fr.width), mode="bilinear",
                        align_corners=False)
    focal_crop_resize.last_f_eff = f_px * fr.width / c_w
    focal_crop_resize.last_f_eff_y = f_px * fr.height / c_h
    focal_crop_resize.last_clamped = bool(
        int(round(f_px * fr.width / fr.f_ref)) > min(h, w)
        or int(round(f_px * fr.height / fr.f_ref)) > min(h, w))
    focal_crop_resize.last_frame = fr
    return out.clamp(0, 255).to(torch.uint8)


# --------------------------------------------------------------------------- #
# f-theta (fisheye) canonicalization — D-016 fix (GEOMETRY_INTEGRITY_AUDIT.md) #
# --------------------------------------------------------------------------- #
# The PhysicalAI front-wide is an f-theta FISHEYE, not a rectilinear pinhole.
# The old path fed its NOMINAL 120-deg pinhole focal (554 px) into the pinhole
# `focal_crop_resize`, cropping a 533-px square that — under the REAL fisheye
# radial map r(theta) — retains only ~16.4 deg half-angle -> canonical f_eff
# ~434 px, i.e. 1.63x more zoomed than the f_eff=266 the pipeline claims (and
# than comma2k19). Fix: crop against the REAL radial map so the retained
# half-angle equals the shared canonical one (comma's), making f_eff == F_REF.
#
# The canonical FOV is defined as the half-angle a pinhole of focal F_REF
# subtends over size/2 px (exactly what comma2k19 achieves). Cropping the
# fisheye to that SAME half-angle equalizes the action->pixel scale across
# corpora — the whole point of D-016.


def canonical_halfangle_rad(size: int = 256, f_ref: float = F_REF) -> float:
    """Retained half-angle of the shared canonical frame: atan((size/2)/f_ref).

    A pinhole of focal ``f_ref`` maps a ray at this angle to the frame edge
    (size/2). comma2k19 achieves exactly this (f_eff ~= 266); every other corpus
    is cropped to retain the same half-angle so f_eff and angular field match.
    """
    return math.atan((size / 2.0) / f_ref)


@dataclass(frozen=True)
class FThetaIntrinsics:
    """f-theta fisheye model: radius ``r(theta) = sum poly[i]*theta**i`` [native
    px] from the principal point (cx, cy), on a native ``width x height`` sensor.

    PhysicalAI-AV ships this per clip in ``calibration/camera_intrinsics``
    (columns ``fw_poly_0..4``, ``cx``, ``cy``, ``width``, ``height``);
    ``fw_poly_1`` is the paraxial focal dr/dtheta|_0. Poly is evaluated with
    Horner so it works on floats and on tensors (undistort grid) alike.

    ``per_clip`` records whether (cx, cy) are this clip's REAL measured principal
    point (True) or the corpus-median fallback (False). The principal-point-
    centered crop (`ftheta_crop_resize(center="principal")`) REQUIRES per_clip=
    True: the front-wide has two rigs with cy ~543 (A) and ~755 (B), so a single
    global cy is wrong for one of them and must never drive a centered crop.
    """

    poly: tuple[float, ...]
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080
    per_clip: bool = False

    def r_of_theta(self, theta):
        """Fisheye radius [native px] for incidence angle ``theta`` [rad]."""
        r = 0.0 if isinstance(theta, float) else torch.zeros_like(theta)
        for c in reversed(self.poly):
            r = r * theta + c
        return r

    def theta_of_r(self, r_target: float, hi: float = 1.6) -> float:
        """Inverse map (scalar): incidence angle whose radius is ``r_target``.

        The map is monotone over a fisheye's field, so plain bisection is exact
        and dependency-free.
        """
        lo, high = 0.0, hi
        for _ in range(80):
            mid = 0.5 * (lo + high)
            if float(self.r_of_theta(mid)) < r_target:
                lo = mid
            else:
                high = mid
        return 0.5 * (lo + high)

    @property
    def paraxial_focal(self) -> float:
        return float(self.poly[1])


# Corpus-median fallback (MEASURED 2026-07-12 over the 500 R0-selected clips /
# 30 calibration chunks; per-clip focal sigma 0.47%). USED ONLY when a per-clip
# entry is unavailable. NOTE: the vertical principal point is BIMODAL across two
# rigs (cy~543 for 23% of clips, cy~755 for 77%) — so principal-point-dependent
# processing MUST use per-clip cy, never this median. cy=753.18 here is a RIG-B
# value; there is NO single correct global cy. Hence ``per_clip=False``: it pins
# f_eff via the near-constant focal (robust to the rig split) and feeds the
# undistort helper, but it must NEVER drive the (cx, cy)-centered crop — that
# path refuses this fallback and reverts to geometric-center with a warning.
PHYSICALAI_FRONT_WIDE_FTHETA = FThetaIntrinsics(
    poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067),
    cx=958.0, cy=753.18, width=1920, height=1080, per_clip=False)


def ftheta_crop_size(intr: FThetaIntrinsics, size: int = 256,
                     f_ref: float = F_REF) -> int:
    """Centered-square crop side that retains the shared canonical half-angle
    under the REAL f-theta radial map (-> edge f_eff == f_ref).

    Unlike `focal_crop_size` (which assumes a pinhole, ``c = f*size/f_ref``), a
    fisheye's radius is sub-linear in angle, so we invert the real polynomial at
    the canonical half-angle. Plugging the paraxial focal into the pinhole
    formula would over-crop ~8% and land f_eff ~245, not 266.
    """
    r = float(intr.r_of_theta(canonical_halfangle_rad(size, f_ref)))
    c = int(round(2.0 * r))
    return max(32, min(c, min(intr.height, intr.width)))


def ftheta_crop_size_hw(intr: FThetaIntrinsics, frame: CanonicalFrame
                        ) -> tuple[int, int]:
    """Native ``(c_h, c_w)`` crop RECTANGLE retaining ``frame``'s half-angles.

    The fisheye radial map is isotropic, so a rectangle of half-width
    ``r(theta_x)`` and half-height ``r(theta_y)`` retains exactly ``theta_x``
    horizontally and ``theta_y`` vertically — which is what makes a WIDE
    (non-square) canonical frame expressible on this sensor at all.

    ⚠️ For a SQUARE frame this is bit-identical to :func:`ftheta_crop_size`,
    including the clamp: ``theta_x == theta_y`` so both sides round to the same
    integer, and the square branch re-applies the legacy ``min(H, W)`` bound
    rather than a per-axis one. That equality is asserted in
    ``tests/test_geometry_configurable.py``.
    """
    rx = float(intr.r_of_theta(frame.half_angle_x_rad()))
    ry = float(intr.r_of_theta(frame.half_angle_y_rad()))
    c_w = max(32, min(int(round(2.0 * rx)), intr.width))
    c_h = max(32, min(int(round(2.0 * ry)), intr.height))
    if frame.is_square:                       # legacy clamp, exactly
        c_h = c_w = min(c_h, c_w)
    return c_h, c_w


_warned_geometric = [False]


def ftheta_crop_box_hw(intr: FThetaIntrinsics, h: int, w: int,
                       size: int = 256, f_ref: float = F_REF, *,
                       center: str = "principal",
                       frame: CanonicalFrame | None = None
                       ) -> tuple[int, int, int, int]:
    """Rectangular crop box ``(c_h, c_w, top, left)`` in DECODED pixels.

    The general form of :func:`ftheta_crop_box` (which is this function with
    ``c_h == c_w``). Position rules are unchanged: ``center="principal"``
    centers on the per-clip ``(cx, cy)`` — the two-rig fix — and
    ``center="geometric"`` uses the frame center.
    """
    f = as_frame(frame, size, f_ref)
    sx = w / float(intr.width)
    sy = h / float(intr.height)
    c_h_native, c_w_native = ftheta_crop_size_hw(intr, f)
    s = min(sx, sy)                       # isotropic: preserves aspect on rescale
    c_w = max(32, min(int(round(c_w_native * s)), w))
    c_h = max(32, min(int(round(c_h_native * s)), h))
    if f.is_square:                       # legacy clamp, exactly
        c_h = c_w = min(c_h, c_w)
    if center == "principal":
        top = int(round(intr.cy * sy - c_h / 2.0))
        left = int(round(intr.cx * sx - c_w / 2.0))
    elif center == "geometric":
        top, left = (h - c_h) // 2, (w - c_w) // 2
    else:
        raise ValueError(f"center must be 'principal' or 'geometric', got {center!r}")
    return c_h, c_w, top, left


def ftheta_crop_box(intr: FThetaIntrinsics, h: int, w: int, size: int = 256,
                    f_ref: float = F_REF, *, center: str = "principal"
                    ) -> tuple[int, int, int]:
    """Square-crop box ``(c, top, left)`` in DECODED pixels for a ``h x w`` frame.

    ``c`` is the f-theta-correct side (`ftheta_crop_size`, scaled to the decoded
    resolution) — the SAME for both centerings, so the achieved f_eff is
    identical either way. Only the box POSITION differs:

    - ``center="geometric"``: top-left at ((h-c)//2, (w-c)//2) (legacy; matches
      comma2k19's convention). Robust to the rig split but leaves the horizon at
      DIFFERENT output rows for rig A (cy~543) vs rig B (cy~755).
    - ``center="principal"``: crop centered on the per-clip principal point
      (cx, cy), so the optical axis (θ=0, the straight-ahead horizon of a level
      mount) lands at the OUTPUT CENTER for every clip regardless of rig. The
      box may extend past the frame edge (rig B's cy is ~215 px below the
      geometric center, so a centered crop overflows the bottom by ~90 px);
      `ftheta_crop_resize` pads that genuinely-unobserved region rather than
      shifting the box (which would reintroduce the per-rig offset).

    Square-only by signature (it returns ONE side). :func:`ftheta_crop_box_hw`
    is the rectangular form; this is a thin unpack of it.
    """
    c_h, c_w, top, left = ftheta_crop_box_hw(intr, h, w, size, f_ref,
                                             center=center)
    assert c_h == c_w                          # square by construction here
    return c_w, top, left


def ftheta_crop_resize(vid: Tensor, intr: FThetaIntrinsics, size: int = 256,
                       f_ref: float = F_REF, *, center: str = "principal",
                       frame: CanonicalFrame | None = None) -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,size,size] uint8, f-theta-correct canonical.

    ``frame`` (opt-in, default None == today's square 256/266 frame) generalises
    the crop to a RECTANGLE, so a wide canonical frame is expressible. It only
    changes WHICH FIELD is retained — the output pixels remain the sensor's own
    f-theta mapping, rescaled. For a projection-faithful resample see
    :func:`cylindrical_rectify`.

    Square crop of side `ftheta_crop_size` (retaining the shared canonical
    half-angle), then bilinear resize. ``center`` selects where the square sits:

    - ``"principal"`` (default, D-016 R1 fix): centered on the clip's per-clip
      (cx, cy). This puts the horizon/optical-axis at the SAME output row for
      BOTH camera rigs (cy~543 and cy~755) — the two-rig vertical inconsistency
      the geometric crop produced. REQUIRES ``intr.per_clip`` (a real measured
      principal point); with the corpus-median fallback it warns once and
      reverts to geometric-center (the fallback cy is a rig-B value — wrong for
      rig A). Where a rig-B crop overflows the native bottom edge (near-field
      road the sensor never captured), the missing rows are replicate-padded so
      cy stays at the true crop center.
    - ``"geometric"``: legacy center ((h-c)//2, (w-c)//2); comma2k19's
      convention, robust to the rig split but horizon-inconsistent across rigs.

    Achieved edge-referenced f_eff (independent of centering — the crop SIDE is
    unchanged) is stored in `.last_f_eff`, measured by round-tripping the integer
    crop through the real poly so a data card / build check reports the TRUE
    value.
    """
    t, _, h, w = vid.shape
    f = as_frame(frame, size, f_ref)
    eff_center = center
    if center == "principal" and not intr.per_clip:
        if not _warned_geometric[0]:
            _warned_geometric[0] = True
            warnings.warn(
                "ftheta_crop_resize(center='principal') needs a per-clip principal "
                "point but got the corpus-median fallback (per_clip=False; cy is a "
                "rig-B value). Reverting to geometric-center — the horizon will be "
                "rig-inconsistent. Provide per-clip calibration to enable the fix.",
                RuntimeWarning, stacklevel=2)
        eff_center = "geometric"

    c_h, c_w, top, left = ftheta_crop_box_hw(intr, h, w, center=eff_center,
                                             frame=f)
    # Clip the box to the frame, then replicate-pad the shortfall back to
    # c_h x c_w so the principal point stays at the exact crop center even when
    # the box spills past a native edge (rig B). Float only the crop, never the
    # clip.
    y0, y1 = max(0, top), min(h, top + c_h)
    x0, x1 = max(0, left), min(w, left + c_w)
    out = vid[..., y0:y1, x0:x1].float()
    pt, pb, pl, pr = y0 - top, (top + c_h) - y1, x0 - left, (left + c_w) - x1
    if pt or pb or pl or pr:
        out = F.pad(out, (pl, pr, pt, pb), mode="replicate")
    out = F.interpolate(out, size=(f.height, f.width), mode="bilinear",
                        align_corners=False)
    sx, sy = w / float(intr.width), h / float(intr.height)
    theta_x = intr.theta_of_r((c_w / min(sx, sy)) / 2.0)
    theta_y = intr.theta_of_r((c_h / min(sx, sy)) / 2.0)
    ftheta_crop_resize.last_f_eff = f.focal_for_halfangle_x(theta_x)
    ftheta_crop_resize.last_f_eff_y = (f.height / 2.0) / math.tan(theta_y)
    ftheta_crop_resize.last_frame = f
    return out.clamp(0, 255).to(torch.uint8)


def ftheta_project_ray(intr: FThetaIntrinsics,
                       d_cam: tuple[float, float, float]) -> tuple[float, float]:
    """Forward f-theta projection of a camera-frame ray -> native pixel (u, v).

    Camera convention (matches `ftheta_undistort_grid`): +x right, +y DOWN, +z
    the optical axis / boresight. A ray at incidence angle θ = atan2(‖x,y‖, z)
    maps to radius r(θ) from the principal point along its azimuth:
    ``u = cx + r·x/ρ``, ``v = cy + r·y/ρ`` (ρ = ‖x,y‖). The boresight (0,0,1)
    projects to exactly (cx, cy). Used to locate the horizon (the vehicle-forward
    horizontal ray, transformed into the camera frame via the clip's extrinsics)
    so the rig-consistency of the crop can be verified in pixels.
    """
    x, y, z = float(d_cam[0]), float(d_cam[1]), float(d_cam[2])
    rho = math.hypot(x, y)
    if rho < 1e-9:
        return intr.cx, intr.cy
    r = float(intr.r_of_theta(math.atan2(rho, z)))
    return intr.cx + r * x / rho, intr.cy + r * y / rho


def ftheta_horizon_row(intr: FThetaIntrinsics,
                       d_cam: tuple[float, float, float] = (0.0, 0.0, 1.0),
                       h: int = 1080, w: int = 1920, size: int = 256,
                       f_ref: float = F_REF, *, center: str = "principal",
                       frame: CanonicalFrame | None = None) -> float:
    """Output ROW that the horizon ray ``d_cam`` (vehicle-forward, in the camera
    frame — from the clip's extrinsics) lands on after the crop+resize.

    This is the metric that proves the two-rig fix: for ``center="principal"`` it
    is ~size/2 minus a small per-clip pitch term for EVERY clip (rig A and rig B
    alike, since the crop is centered on each clip's cy); for the legacy
    ``center="geometric"`` it is offset by (cy - h/2)·size/c, i.e. ~66 rows lower
    for rig B (cy~755) than rig A (cy~543). ``d_cam`` defaults to the optical axis
    (0,0,1), i.e. a perfectly level mount whose horizon is exactly the principal
    point.
    """
    _u, v = ftheta_project_ray(intr, d_cam)
    sy = h / float(intr.height)
    f = as_frame(frame, size, f_ref)
    c_h, _c_w, top, _left = ftheta_crop_box_hw(intr, h, w, center=center,
                                               frame=f)
    return (v * sy - top) * f.height / c_h


def ftheta_feff_report(intr: FThetaIntrinsics, size: int = 256,
                       f_ref: float = F_REF) -> dict:
    """Achieved edge f_eff of the corrected crop vs the f_eff the OLD nominal-
    pinhole path silently produced. Regression guard + data-card provenance."""
    c_after = ftheta_crop_size(intr, size, f_ref)
    th_after = intr.theta_of_r(c_after / 2.0)
    f_after = (size / 2.0) / math.tan(th_after)

    f_nom = nominal_focal_from_hfov(intr.width, PHYSICALAI_FRONT_WIDE_HFOV_DEG)
    c_before = focal_crop_size(f_nom, intr.height, intr.width, size, f_ref)
    th_before = intr.theta_of_r(c_before / 2.0)
    f_before = (size / 2.0) / math.tan(th_before)
    return {
        "f_eff_after": round(f_after, 2),
        "f_eff_before_nominal": round(f_before, 2),
        "crop_side_after": c_after, "crop_side_before": c_before,
        "retained_hfov_after_deg": round(math.degrees(2 * th_after), 2),
        "retained_hfov_before_deg": round(math.degrees(2 * th_before), 2),
        "paraxial_focal_px": round(intr.paraxial_focal, 2),
        "nominal_pinhole_focal_px": round(f_nom, 2),
    }


def ftheta_undistort_grid(intr: FThetaIntrinsics, size: int = 256,
                          f_ref: float = F_REF,
                          device: str | torch.device = "cpu") -> Tensor:
    """`grid_sample` grid [1,size,size,2] mapping an ideal pinhole (focal f_ref,
    centered on the optical axis) back onto the f-theta native frame.

    This is the fully-rectilinear option for the deferred D-016 R1 step. It is
    NOT the default cache build: 77% of PhysicalAI clips (rig B) have the
    principal point low in the frame (cy~755), so a full pinhole ray fan centered
    on the optical axis samples FAR below the native bottom edge (near-field road
    the sensor never captured). The default `ftheta_crop_resize(center=
    "principal")` also centers on (cx, cy) but retains only the +/-c/2 canonical
    patch, so it spills past the edge by at most ~90 px (replicate-padded) rather
    than fanning a full undistorted field. Kept + tested so the f-theta forward
    map is executable and R1 can adopt it with per-rig extrinsics.
    """
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32), indexing="ij")
    x = xs - (size - 1) / 2.0
    y = ys - (size - 1) / 2.0
    rho = torch.sqrt(x * x + y * y)                       # rectilinear radius px
    theta = torch.atan2(rho, torch.full_like(rho, float(f_ref)))
    r = intr.r_of_theta(theta)                            # native fisheye radius
    scale = torch.where(rho > 1e-6, r / rho, torch.zeros_like(rho))
    u = intr.cx + x * scale
    v = intr.cy + y * scale
    gx = u / (intr.width - 1) * 2.0 - 1.0
    gy = v / (intr.height - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).to(device)


def ftheta_undistort(vid: Tensor, intr: FThetaIntrinsics, size: int = 256,
                     f_ref: float = F_REF) -> Tensor:
    """[T,3,H,W] -> [T,3,size,size] uint8 true rectilinear pinhole (f_eff=f_ref),
    fisheye distortion removed. See `ftheta_undistort_grid` for why this is the
    R1 option, not the default. f_eff == f_ref exactly by construction."""
    grid = ftheta_undistort_grid(intr, size, f_ref, device=vid.device)
    out = F.grid_sample(vid.float(), grid.expand(vid.shape[0], -1, -1, -1),
                        mode="bilinear", padding_mode="border",
                        align_corners=False)
    ftheta_undistort.last_f_eff = float(f_ref)
    return out.clamp(0, 255).to(torch.uint8)


# =========================================================================== #
# D-016 R1 — PINHOLE rectify-to-canvas (undistort + pad)                      #
# =========================================================================== #
# FOLDED IN 2026-07-26 from the 2026-07-17 R1 bundle
# (`TanitAD Research Hub/Data Engineering/Implementation/incoming/
#   2026-07-17-d016-r1-pinhole-rectify/calib_r1.py`, 9/9 tests green), where it
# sat UNMERGED for 9 days. It is the standing prerequisite for the whole
# owned-real-urban tier and it is what unblocks nuScenes.
#
# THE WALL IT REMOVES. `focal_crop_resize` canonicalizes by cropping a centered
# SQUARE of side c = fx*size/F_REF. On a wide frame that square is bounded by the
# HEIGHT, so any narrow-FOV camera clamps and lands a WRONG, silently-zoomed
# f_eff. MEASURED anchors:
#   PandaSet front  fx=1970.01 / 1080-tall -> needs c=1896 >> 1080 -> f_eff ~467
#   nuScenes CAM_FRONT fx~1266 /  900-tall -> needs c=1219 >>  900 -> f_eff ~360
# against the canonical 266. General rule (proved 2026-07-15): fx > 1122 px on a
# 1080-tall frame is not square-croppable to 266; scaled to 900-tall the
# threshold is ~935 px.
#
# THE FIX. Build an ideal PINHOLE canvas of focal F_REF at the output size,
# forward-map each ideal ray through Brown-Conrady onto the native sensor, and
# `grid_sample` it back. f_eff == F_REF holds BY CONSTRUCTION, lens distortion is
# removed, and rays landing outside the native frame become an explicit MEASURED
# unobserved mask instead of a silent zoom-in — "height-bound -> masked
# periphery (honest)" replacing "height-bound -> zoom (wrong)".
#
# Fisheye corpora (PhysicalAI/Cosmos f-theta, ZOD Kannala-Brandt) keep the
# `ftheta_*` path above; this is the rectilinear-lens counterpart.


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Brown-Conrady pinhole model on a native ``width x height`` sensor.

    ``dist = (k1, k2, p1, p2, k3)`` are the OpenCV-order radial/tangential coeffs
    (radial k1,k2,k3; tangential p1,p2). Defaults to zero -> a pure pinhole, in
    which case :func:`pinhole_rectify` is an undistort-free pad-crop.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))


def brown_conrady_distort(x: Tensor, y: Tensor,
                          dist: tuple[float, float, float, float, float]
                          ) -> tuple[Tensor, Tensor]:
    """Forward Brown-Conrady: ideal normalized ray ``(x, y)=(X/Z, Y/Z)`` ->
    DISTORTED normalized coords (still focal-normalized, pre-``fx`` scaling).

    ``x_d = x*(1+k1 r^2+k2 r^4+k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)`` and the
    symmetric ``y_d``. This is the map a real lens applies; rectification samples
    the native frame at ``fx*x_d+cx`` so the OUTPUT is the ideal rectilinear ray.
    """
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))          # Horner in r^2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def pinhole_rectify_grid(intr: PinholeIntrinsics, h: int, w: int,
                         size: int = 256, f_ref: float = F_REF,
                         device: str | torch.device = "cpu",
                         frame: CanonicalFrame | None = None
                         ) -> tuple[Tensor, Tensor]:
    """``grid_sample`` grid + observed mask mapping an ideal pinhole (focal
    ``f_ref``, centered on the optical axis) onto the native ``h x w`` frame.

    Intrinsics are defined on ``intr.width x intr.height``; they are scaled to the
    DECODED ``(h, w)`` so a frame decoded at a non-native resolution still maps
    correctly. Returns ``(grid [1,size,size,2] in [-1,1], mask [size,size] bool)``
    where ``mask`` is True for output pixels whose ideal ray lands INSIDE the
    native frame (the observed region). ``f_eff == f_ref`` holds by construction:
    output pixel at ``d`` px from center is the ray ``atan(d / f_ref)``.
    """
    fr = as_frame(frame, size, f_ref)
    sx = w / float(intr.width)
    sy = h / float(intr.height)
    fx, fy = intr.fx * sx, intr.fy * sy
    cx, cy = intr.cx * sx, intr.cy * sy

    ys, xs = torch.meshgrid(
        torch.arange(fr.height, dtype=torch.float32, device=device),
        torch.arange(fr.width, dtype=torch.float32, device=device), indexing="ij")
    if fr.projection == "cylindrical":
        # equidistant azimuth: x_px = f_ref * phi; vertical stays pinhole per
        # column. Ray (sin phi, y_n, cos phi) -> ideal normalized (x/z, y/z).
        phi = (xs - (fr.width - 1) / 2.0) / float(fr.f_ref)
        y_n = (ys - (fr.height - 1) / 2.0) / float(fr.f_ref)
        cosp = torch.cos(phi).clamp_min(1e-6)
        x = torch.tan(phi)
        y = y_n / cosp
    else:
        x = (xs - (fr.width - 1) / 2.0) / float(fr.f_ref)   # ideal normalized ray
        y = (ys - (fr.height - 1) / 2.0) / float(fr.f_ref)
    x_d, y_d = brown_conrady_distort(x, y, intr.dist)
    u = fx * x_d + cx                                      # native px
    v = fy * y_d + cy
    mask = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    # normalization convention matches ftheta_undistort_grid (codebase-consistent)
    gx = u / (w - 1) * 2.0 - 1.0
    gy = v / (h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return grid, mask


def pinhole_rectify(vid: Tensor, intr: PinholeIntrinsics, size: int = 256,
                    f_ref: float = F_REF, padding_mode: str = "zeros",
                    frame: CanonicalFrame | None = None) -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,size,size] uint8 rectilinear pinhole at
    ``f_eff == f_ref``, barrel distortion removed, out-of-frame periphery masked.

    Unobserved pixels are filled per ``padding_mode`` (``"zeros"`` = honest black
    periphery, the default; ``"border"`` = edge-extended). The achieved focal and
    the observed fraction are exposed for the data card:
        ``pinhole_rectify.last_f_eff``        == f_ref (exact by construction)
        ``pinhole_rectify.last_observed_frac`` fraction of pixels inside native
        ``pinhole_rectify.last_mask``          [size,size] bool observed mask
    """
    t, _, h, w = vid.shape
    fr = as_frame(frame, size, f_ref)
    grid, mask = pinhole_rectify_grid(intr, h, w, device=vid.device, frame=fr)
    out = F.grid_sample(vid.float(), grid.expand(t, -1, -1, -1),
                        mode="bilinear", padding_mode=padding_mode,
                        align_corners=False)
    if padding_mode == "zeros":
        out = out * mask.to(out.dtype)                     # crisp unobserved band
    pinhole_rectify.last_f_eff = float(fr.f_ref)
    pinhole_rectify.last_mask = mask
    pinhole_rectify.last_observed_frac = float(mask.float().mean())
    pinhole_rectify.last_frame = fr
    return out.clamp(0, 255).to(torch.uint8)


def square_crop_feff(fx: float, h: int, w: int, size: int = 256,
                     f_ref: float = F_REF) -> dict:
    """What the OLD centered-square crop (``focal_crop_size``) actually lands for
    this pinhole — reproduces the 0715 blocker number as a regression anchor."""
    c_ideal = int(round(fx * size / f_ref))
    c_used = max(32, min(c_ideal, min(h, w)))
    achieved = fx * size / c_used
    return {"ideal_crop_px": c_ideal, "used_crop_px": c_used,
            "achieved_feff_px": round(achieved, 2),
            "height_clamped": c_ideal > min(h, w),
            "drop_in": abs(achieved - f_ref) / f_ref <= 0.05}


def pinhole_geometry_report(intr: PinholeIntrinsics, size: int = 256,
                            f_ref: float = F_REF) -> dict:
    """Naive square-crop vs pinhole-rectify for one camera. Data-card + regression
    provenance: proves the rectify path lands f_eff==f_ref where the crop can't,
    and quantifies the unobserved cost (``observed_frac``) and the distortion it
    corrects (``max_distort_px`` at the canonical corner)."""
    naive = square_crop_feff(intr.fx, intr.height, intr.width, size, f_ref)
    # rectify on a native-resolution dummy frame -> real observed mask
    dummy = torch.zeros(1, 3, intr.height, intr.width)
    pinhole_rectify(dummy, intr, size, f_ref)
    obs = pinhole_rectify.last_observed_frac

    # distortion magnitude at the canonical corner ray (px on the native sensor)
    corner = canonical_halfangle_rad(size, f_ref)          # edge half-angle
    xr = math.tan(corner)                                   # ideal normalized @ edge
    xd, _ = brown_conrady_distort(torch.tensor(xr), torch.tensor(0.0), intr.dist)
    max_distort_px = abs(float(xd) - xr) * intr.fx

    return {
        "camera_hfov_deg": round(intr.hfov_deg, 2),
        "camera_vfov_deg": round(intr.vfov_deg, 2),
        "canonical_hfov_deg": round(math.degrees(2 * corner), 2),
        "naive_square_crop": naive,
        "rectify_feff_px": float(f_ref),
        "rectify_observed_frac": round(obs, 4),
        "rectify_drop_in": True,
        "max_distort_px_at_edge": round(max_distort_px, 2),
        "k1": intr.dist[0],
    }


# =========================================================================== #
# CYLINDRICAL / equidistant-azimuth rectification (2026-07-27)                 #
# =========================================================================== #
# WHY A THIRD PROJECTION EXISTS. `ftheta_crop_resize` equalizes the retained
# FIELD but leaves the sensor's own fisheye warp in the pixels;
# `pinhole_rectify` / `ftheta_undistort` produce a true rectilinear image, whose
# cost grows as tan(theta). COMPUTED from the definitions (no data, no model);
# reproduced by `projection_density_report` and pinned in
# tests/test_geometry_configurable.py:
#
#   half-angle    cumulative radius     LOCAL px-per-radian at the edge
#                 tan(t)/t              1/cos^2(t)           cylindrical
#     25.70 deg   1.0730                1.2316               1.0  <- today
#     50.00 deg   1.3656                2.4203               1.0
#     60.00 deg   1.6540                4.0000               1.0
#
# i.e. a pinhole canvas at a 50-deg half-angle packs 2.42x more pixels per radian
# at the edge than at the center: the NEW pixels the PI is asking for would land
# disproportionately on smeared periphery. A cylinder is equidistant in azimuth
# (x = f*phi), so angular density is UNIFORM across the width, and it keeps
# straight verticals (a pole stays a column) which a raw fisheye does not.
# This is the standard choice for wide driving cameras.
#
# ⚠️ IT IS NOT A CLAIM THAT CYLINDRICAL WINS. Uniform angular density is a
# geometric fact; whether it produces a better world model is an EXPERIMENT and
# belongs to `2026-07-27-fov-crop-audit`. This code makes that experiment a
# config change instead of a refactor.
#
# ⚠️ THE RIG FIX IS PRESERVED BY CONSTRUCTION, not by care: the output center
# ray is (0, 0, 1) = the boresight, and `ftheta_project_ray` maps the boresight
# to EXACTLY (cx, cy) — the clip's own principal point. So rig A (cy~543) and
# rig B (cy~755) both land their optical axis at the output center, which is the
# same guarantee `ftheta_crop_resize(center="principal")` gives. A fallback
# (per_clip=False) intrinsic is REFUSED here rather than silently downgraded:
# unlike a crop there is no "geometric center" fallback that is even meaningful
# for a full ray-fan resample.


def cylindrical_rays(frame: CanonicalFrame, device: str | torch.device = "cpu"
                     ) -> tuple[Tensor, Tensor, Tensor]:
    """Per-output-pixel camera-frame ray ``(x, y, z)`` for ``frame``.

    Camera convention matches :func:`ftheta_project_ray`: +x right, +y DOWN, +z
    boresight. For ``projection="cylindrical"`` the ray is
    ``(sin phi, y_n, cos phi)`` with ``phi = (u - (W-1)/2)/f_ref`` and
    ``y_n = (v - (H-1)/2)/f_ref``; for ``"pinhole"`` it is ``(x_n, y_n, 1)``.
    Returned unnormalized (only the direction matters downstream).
    """
    ys, xs = torch.meshgrid(
        torch.arange(frame.height, dtype=torch.float32, device=device),
        torch.arange(frame.width, dtype=torch.float32, device=device),
        indexing="ij")
    u = (xs - (frame.width - 1) / 2.0) / float(frame.f_ref)
    v = (ys - (frame.height - 1) / 2.0) / float(frame.f_ref)
    if frame.projection == "cylindrical":
        return torch.sin(u), v, torch.cos(u)
    return u, v, torch.ones_like(u)


def ftheta_project_rays(intr: FThetaIntrinsics, x: Tensor, y: Tensor, z: Tensor
                        ) -> tuple[Tensor, Tensor]:
    """Tensor form of :func:`ftheta_project_ray` — rays -> native ``(u, v)`` px."""
    rho = torch.sqrt(x * x + y * y)
    theta = torch.atan2(rho, z)
    r = intr.r_of_theta(theta)
    scale = torch.where(rho > 1e-9, r / rho.clamp_min(1e-9), torch.zeros_like(rho))
    return intr.cx + x * scale, intr.cy + y * scale


def cylindrical_grid(intr: FThetaIntrinsics, h: int, w: int,
                     frame: CanonicalFrame,
                     device: str | torch.device = "cpu"
                     ) -> tuple[Tensor, Tensor]:
    """``grid_sample`` grid + observed mask for a cylindrical/pinhole resample of
    an f-theta fisheye onto ``frame``.

    Intrinsics are defined on ``intr.width x intr.height`` and are scaled to the
    DECODED ``(h, w)``. Returns ``(grid [1,H,W,2] in [-1,1], mask [H,W] bool)``;
    ``mask`` is True where the ideal ray lands INSIDE the native frame, i.e. the
    genuinely observed region (a wide cylinder over a 120-deg sensor is mostly
    observed horizontally and unobserved in the corners).
    """
    x, y, z = cylindrical_rays(frame, device=device)
    u_n, v_n = ftheta_project_rays(intr, x, y, z)          # native-sensor px
    sx, sy = w / float(intr.width), h / float(intr.height)
    u, v = u_n * sx, v_n * sy
    mask = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (z > 0)
    gx = u / (w - 1) * 2.0 - 1.0
    gy = v / (h - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0), mask


def cylindrical_rectify(vid: Tensor, intr: FThetaIntrinsics,
                        frame: CanonicalFrame,
                        padding_mode: str = "zeros",
                        *, require_per_clip: bool = True) -> Tensor:
    """[T,3,H,W] uint8/float -> [T,3,frame.height,frame.width] uint8, equidistant
    -azimuth (or pinhole) resample of an f-theta fisheye.

    ``f_eff`` is ``frame.f_ref`` exactly by construction. Unobserved pixels
    follow ``padding_mode`` ("zeros" = honest black, the default). Provenance for
    the data card:
        ``cylindrical_rectify.last_f_eff``         == frame.f_ref
        ``cylindrical_rectify.last_observed_frac`` fraction inside the native frame
        ``cylindrical_rectify.last_mask``          [H,W] bool observed mask
        ``cylindrical_rectify.last_frame``         the frame it was run with

    ``require_per_clip=True`` (default) REFUSES a corpus-median intrinsic: its
    ``cy`` is a rig-B value, and centering a full ray fan on the wrong principal
    point reintroduces the ~215 px two-rig error this program already paid for.
    """
    if require_per_clip and not intr.per_clip:
        raise ValueError(
            "cylindrical_rectify needs a PER-CLIP principal point but got the "
            "corpus-median fallback (per_clip=False; its cy is a rig-B value). "
            "The front-wide has two rigs (cy~543 / cy~755) and a ray fan "
            "centered on the wrong one is ~215 px off — the D-016 R1 error. "
            "Provide per-clip calibration, or pass require_per_clip=False and "
            "state in the artifact that the rig fix is DISABLED.")
    t = vid.shape[0]
    h, w = int(vid.shape[-2]), int(vid.shape[-1])
    grid, mask = cylindrical_grid(intr, h, w, frame, device=vid.device)
    out = F.grid_sample(vid.float(), grid.expand(t, -1, -1, -1), mode="bilinear",
                        padding_mode=padding_mode, align_corners=False)
    if padding_mode == "zeros":
        out = out * mask.to(out.dtype)
    cylindrical_rectify.last_f_eff = float(frame.f_ref)
    cylindrical_rectify.last_mask = mask
    cylindrical_rectify.last_observed_frac = float(mask.float().mean())
    cylindrical_rectify.last_frame = frame
    return out.clamp(0, 255).to(torch.uint8)


def projection_density_report(frame: CanonicalFrame) -> dict:
    """Angular-density cost of ``frame``'s projection — the number that motivates
    the cylindrical option, computed from the definitions (no data needed).

    Two distinct quantities, reported separately because conflating them is easy:

    - ``edge_local_density_vs_center`` — LOCAL px per radian at the edge relative
      to the center. Pinhole ``d/dphi (f tan phi)/f = 1/cos^2(phi)``;
      cylindrical ``d/dphi (f phi)/f = 1`` exactly.
    - ``cumulative_radius_vs_equidistant`` — how much image RADIUS the projection
      spends to reach the same field: pinhole ``tan(t)/t``, cylindrical 1.
    """
    th = frame.half_angle_x_rad()
    cyl = frame.projection == "cylindrical"
    return {
        "projection": frame.projection,
        "half_angle_x_deg": round(math.degrees(th), 4),
        "hfov_deg": round(frame.hfov_deg, 4),
        "edge_local_density_vs_center":
            1.0 if cyl else round(1.0 / (math.cos(th) ** 2), 4),
        "cumulative_radius_vs_equidistant":
            1.0 if cyl else round(math.tan(th) / th, 4),
        "pinhole_edge_local_density_at_this_field":
            round(1.0 / (math.cos(th) ** 2), 4),
    }


# =========================================================================== #
# THE RIG-CLEAN FIELD — a frame BOTH rigs fully observe (2026-07-27)           #
# =========================================================================== #
# WHY. Retraction class C26: the deployed crop replicate-PADS rows that fall
# outside the sensor — 0.0017 % on rig A, 8.897 % on rig B (n = 3,000) — so
# ~73 % of training frames carry fabricated rows the rest do not, in a pattern
# that identifies the rig. `cylindrical_rectify` removes the FABRICATION (an
# explicit mask, not replicate-pad) but NOT the asymmetry: a rig-correlated
# BLACK region is still a rig-correlated signal, and this model eats shortcuts.
#
# The clean fix is a field BOTH rigs fully observe, i.e. one whose observed mask
# is EXACTLY zero for every clip of both rigs.
#
# ⭐ AND IT IS A PURE SLICE OF THE PARENT'S PIXELS. `cylindrical_rays` puts the
# boresight at ((W-1)/2, (H-1)/2), so for a CENTRED sub-rectangle of even
# margin the child's ray coordinates are *identically equal* to the parent's:
#
#     u_parent(c0 + j) = (c0 + j - (W-1)/2)/f
#                      = (j - (w-1)/2)/f  =  u_child(j)   when c0 == (W-w)/2
#
# — an exact float identity, not an approximation (the (W-1)/2 and (w-1)/2 halves
# cancel against the integer margin). The same holds for rows. So the ray map,
# the native (u, v), the sampling grid, the observed mask and therefore the
# `grid_sample` output of a centred sub-frame are the corresponding sub-block of
# the parent's. **A rebuild at a centred sub-frame is a ROW/COLUMN SLICE of the
# frames already built**, and (with a LOSSLESS codec) the cached pixels can be
# re-emitted at the smaller geometry with no decode of the source video at all.
#
# ⚠️ THE LOSSLESS CAVEAT IS LOAD-BEARING. A slice is bit-exact only if the cache
# stores the pixels losslessly (`codec="png"`). Slicing a JPEG cache re-encodes
# 8x8 blocks at different offsets and is NOT bit-exact — check `codec` first.


def centred_subframe(frame: "CanonicalFrame", height: int,
                     width: int | None = None) -> "CanonicalFrame":
    """A CENTRED sub-rectangle of ``frame``: same ``f_ref``, same projection.

    This is the ONLY sub-frame shape that is simultaneously (a) expressible as a
    :class:`CanonicalFrame` — which pins the boresight at the output centre — and
    (b) a pure pixel slice of ``frame``'s output. Both margins must be even, so
    ``frame.height - height`` and ``frame.width - width`` must be even; an odd
    margin is REFUSED rather than rounded, because a half-pixel boresight shift
    is exactly the silent geometry drift this module exists to prevent.
    """
    width = frame.width if width is None else width
    if height > frame.height or width > frame.width:
        raise ValueError(
            f"centred_subframe({height}x{width}) is not INSIDE "
            f"{frame.height}x{frame.width} — a sub-frame can only shrink. To "
            f"widen the field you must rebuild from the source video.")
    if (frame.height - height) % 2 or (frame.width - width) % 2:
        raise ValueError(
            f"centred_subframe needs EVEN margins: {frame.height}-{height} and "
            f"{frame.width}-{width} must both be even, else the boresight lands "
            f"half a pixel off centre and the sub-frame is no longer the frame "
            f"it declares.")
    return CanonicalFrame(height=height, width=width, f_ref=frame.f_ref,
                          projection=frame.projection)


def subframe_slice(parent: "CanonicalFrame", sub: "CanonicalFrame"
                   ) -> tuple[slice, slice]:
    """``(rows, cols)`` of ``parent``'s output that are exactly ``sub``'s output.

    Refuses any pair that is not a centred sub-rectangle with the SAME focal and
    projection — those are the conditions under which the slice is bit-exact.
    """
    if sub.projection != parent.projection or \
            float(sub.f_ref) != float(parent.f_ref):
        raise ValueError(
            f"{sub.tag()} is not a slice of {parent.tag()}: a slice preserves "
            f"f_ref and projection exactly (got f_ref "
            f"{sub.f_ref} vs {parent.f_ref}, projection {sub.projection} vs "
            f"{parent.projection}). Changing either is a RESAMPLE, i.e. a "
            f"rebuild from the source video.")
    centred_subframe(parent, sub.height, sub.width)          # validates margins
    r0 = (parent.height - sub.height) // 2
    c0 = (parent.width - sub.width) // 2
    return slice(r0, r0 + sub.height), slice(c0, c0 + sub.width)


def observed_report(intr: FThetaIntrinsics, frame: "CanonicalFrame") -> dict:
    """What ``intr``'s sensor actually delivers for ``frame`` — from the RAY MAP.

    ``observed_frac`` is a property of the per-clip intrinsics and the requested
    frame, not of the pixels, so this needs no decode and is exact. Also reports
    the widest field the sensor could deliver on each axis, which is what turns
    "it is masked" into "by how much, and in which direction".
    """
    _, mask = cylindrical_grid(intr, int(intr.height), int(intr.width), frame)
    row_ok = mask.all(dim=1)
    col_ok = mask.all(dim=0)
    half_x_px = min(float(intr.cx), float(intr.width) - 1 - float(intr.cx))
    return {
        "frame": frame.tag(),
        "observed_frac": float(mask.float().mean()),
        "masked_frac": float(1.0 - mask.float().mean()),
        "fully_observed": bool(mask.all()),
        "n_rows_fully_observed": int(row_ok.sum()),
        "n_cols_fully_observed": int(col_ok.sum()),
        # widest SYMMETRIC field this sensor can deliver, per axis
        "max_hfov_deg": math.degrees(2.0 * intr.theta_of_r(half_x_px)),
        "up_halffield_deg": math.degrees(intr.theta_of_r(float(intr.cy))),
        "down_halffield_deg": math.degrees(
            intr.theta_of_r(float(intr.height) - 1 - float(intr.cy))),
        "requested_hfov_deg": float(frame.hfov_deg),
        "requested_half_vfov_deg": float(frame.vfov_deg) / 2.0,
        "cy": float(intr.cy), "cx": float(intr.cx),
        "per_clip": bool(intr.per_clip),
    }


def ftheta_crop_pad_report(intr: FThetaIntrinsics, h: int, w: int,
                           frame: "CanonicalFrame | None" = None,
                           *, center: str = "principal") -> dict:
    """FABRICATED-pixel fractions of the DEPLOYED crop path — the C26 defect.

    :func:`ftheta_crop_resize` replicate-pads whatever part of the principal-
    point-centred box falls outside the native frame. This reports that padding
    as a fraction of the box, without decoding anything, so the deployed path and
    the cylindrical path can be compared on ONE instrument.
    """
    f = as_frame(frame, 256, F_REF)
    c_h, c_w, top, left = ftheta_crop_box_hw(intr, h, w, center=center, frame=f)
    pad_rows = max(0, -top) + max(0, top + c_h - h)
    pad_cols = max(0, -left) + max(0, left + c_w - w)
    kept_h, kept_w = c_h - pad_rows, c_w - pad_cols
    return {
        "pad_frac_rows": pad_rows / float(c_h),
        "pad_frac_cols": pad_cols / float(c_w),
        "fabricated_frac": 1.0 - max(kept_h, 0) * max(kept_w, 0) /
                           float(c_h * c_w),
        "box_hw": [int(c_h), int(c_w)], "top": int(top), "left": int(left),
        "fabricates": bool(pad_rows or pad_cols),
    }


class RigAsymmetry(ValueError):
    """A frame whose observed mask is non-zero — i.e. rig-correlated content."""


def assert_fully_observed(intrs, frame: "CanonicalFrame", *,
                          label: str = "frame") -> dict:
    """⭐ THE C26 GUARD. REFUSE a frame any clip does not fully observe.

    ``intrs`` is one :class:`FThetaIntrinsics` or an iterable of them. Raises
    :class:`RigAsymmetry` naming the worst offender when ANY clip has a single
    masked pixel — deliberately zero-tolerance, because the defect this guards is
    not "a lot of mask" but "a mask that correlates with the rig".

    ⚠️ It CAN fail, and must: at ``256x640`` / 120 deg cylindrical it fails on rig B
    (~8.9 % masked) and, less obviously, on rig A too (~0.06 % — the 120 deg
    request over-runs some clips' horizontal field). That second failure is why
    this takes a POPULATION, not one clip.
    """
    seq = [intrs] if isinstance(intrs, FThetaIntrinsics) else list(intrs)
    worst, worst_rep = None, None
    for i in seq:
        rep = observed_report(i, frame)
        if worst_rep is None or rep["masked_frac"] > worst_rep["masked_frac"]:
            worst, worst_rep = i, rep
    if worst_rep is None:
        raise RigAsymmetry(f"[{label}] no intrinsics supplied — a guard with an "
                           f"empty population cannot fail, so it is not a guard.")
    if worst_rep["masked_frac"] > 0.0:
        raise RigAsymmetry(
            f"[{label}] {frame.tag()} is NOT fully observed by every clip.\n"
            f"  worst clip: masked_frac {worst_rep['masked_frac']:.6f} "
            f"(cx {worst.cx:.2f}, cy {worst.cy:.2f}, "
            f"rig {'B' if worst.cy >= 650 else 'A'})\n"
            f"  requested   HFOV {frame.hfov_deg:.3f} deg, half-VFOV "
            f"{frame.vfov_deg / 2:.3f} deg\n"
            f"  sensor can  HFOV {worst_rep['max_hfov_deg']:.3f} deg, down "
            f"{worst_rep['down_halffield_deg']:.3f} deg / up "
            f"{worst_rep['up_halffield_deg']:.3f} deg\n"
            f"  A masked region that correlates with the rig is a free rig label "
            f"(class C26). Shrink the frame — see largest_fully_observed_"
            f"subframe() — or state explicitly that the asymmetry is accepted.")
    return {"frame": frame.tag(), "n_clips": len(seq), "masked_frac_max": 0.0,
            "fully_observed_by_all": True}


def largest_fully_observed_subframe(intrs, parent: "CanonicalFrame", *,
                                    heights=None, widths=None,
                                    patch: int = 16, tile: int | None = None
                                    ) -> tuple["CanonicalFrame | None", list]:
    """Largest CENTRED sub-frame of ``parent`` that EVERY clip fully observes.

    Returns ``(frame_or_None, table)``; ``table`` carries the worst-case masked
    fraction of every candidate so the trade is visible, not just the winner.
    ``tile`` (e.g. ``patch * readout.grid == 64``) restricts the search to
    candidates whose token grid tiles the readout grid exactly.
    """
    step = tile or patch
    heights = heights or [h for h in range(parent.height, patch - 1, -step)
                          if (parent.height - h) % 2 == 0]
    widths = widths or [w for w in range(parent.width, patch - 1, -step)
                        if (parent.width - w) % 2 == 0]
    seq = [intrs] if isinstance(intrs, FThetaIntrinsics) else list(intrs)
    masks = [cylindrical_grid(i, int(i.height), int(i.width), parent)[1]
             for i in seq]
    table, best = [], None
    for h in heights:
        r0 = (parent.height - h) // 2
        for w in widths:
            c0 = (parent.width - w) // 2
            worst = max(float(1.0 - m[r0:r0 + h, c0:c0 + w].float().mean())
                        for m in masks)
            ok = worst == 0.0
            table.append({"height": h, "width": w, "masked_frac_max": worst,
                          "fully_observed": ok,
                          "n_tokens": (h // patch) * (w // patch)})
            if ok and (best is None or h * w > best.height * best.width):
                best = centred_subframe(parent, h, w)
    return best, table


#: The wide-FOV frame the v5 corpus was BUILT at (`_geometry.json`, pod2
#: 2026-07-27). Its observed mask is 8.897 % on rig B and 0.0017 % on rig A.
PHYSICALAI_WIDE120_256x640 = CanonicalFrame(
    height=256, width=640, f_ref=305.5774907364391, projection="cylindrical")

#: ⭐ THE RIG-CLEAN FRAME. MEASURED 2026-07-27 over all 3,000 clips of the
#: canonical selection (812 rig A / 2,188 rig B, 121 distinct sensor geometries,
#: 0 intrinsics failures): the LARGEST centred sub-rectangle of
#: `PHYSICALAI_WIDE120_256x640` whose observed mask is EXACTLY 0.000000 for every
#: clip of BOTH rigs. 117.000 deg x 32.131 deg, a pure [40:216, 8:632] slice of the
#: built frames. Raw: `…/incoming/2026-07-28-rig-clean-fix/band_full_3000.json`.
PHYSICALAI_RIG_CLEAN_176x624 = CanonicalFrame(
    height=176, width=624, f_ref=305.5774907364391, projection="cylindrical")

#: The same guarantee under a STRICT readout-tiling constraint (token grid
#: divisible by the 4x4 readout grid, i.e. both dims a multiple of 64): 108 deg x
#: 23.658 deg. Buys exact pooling bins for 33 % fewer tokens than the 176x624 frame.
PHYSICALAI_RIG_CLEAN_128x576 = CanonicalFrame(
    height=128, width=576, f_ref=305.5774907364391, projection="cylindrical")


# The REAL PandaSet front calibration (arXiv 2112.12610), grounded 2026-07-15.
PANDASET_FRONT_INTR = PinholeIntrinsics(
    fx=1970.0131, fy=1970.0091, cx=970.0002, cy=483.2988,
    width=1920, height=1080,
    dist=(-0.5894, 0.66, 0.0011, -0.001, -1.0088))

# comma2k19 EON road camera (near-pinhole reference; f~910 px on 1164x874, ~0 dist)
COMMA2K19_INTR = PinholeIntrinsics(
    fx=COMMA2K19_FOCAL_PX, fy=COMMA2K19_FOCAL_PX, cx=582.0, cy=437.0,
    width=1164, height=874)

# ⚠️ nuScenes CAM_FRONT — NOMINAL REFERENCE ONLY, for the pre-ingest geometry
# report. nuScenes intrinsics are PER-SCENE (`calibrated_sensor.camera_intrinsic`)
# and the corpus spans more than one sensor build, so the adapter MUST read the
# real per-sample matrix (`nuscenes_meta.camera_intrinsics_of`) and must never
# assert these numbers on a record. This is the PhysicalAI two-rig lesson
# (cy~543 vs cy~755) applied pre-emptively: a corpus-level constant that drives
# per-clip geometry is exactly how the rig-B crop error happened.
# Class: PUBLISHED-nominal (the widely-quoted v1.0 CAM_FRONT matrix), UNVERIFIED
# against real bytes because the corpus is not yet acquired (terms gate).
NUSCENES_CAM_FRONT_INTR_NOMINAL = PinholeIntrinsics(
    fx=1266.417203046554, fy=1266.417203046554,
    cx=816.2670197447984, cy=491.50706579294757,
    width=1600, height=900)
