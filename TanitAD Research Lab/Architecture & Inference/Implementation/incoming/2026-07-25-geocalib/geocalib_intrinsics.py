"""Per-video camera-intrinsics estimation for the YouTube-IDM pipeline (GeoCalib).

WHY: the YouTube-IDM pilot canonicalizes every clip with a FIXED nominal HFOV
(`yt_pilot_common.decode_canonical` -> `nominal_focal_px(W, hfov_deg=100)` ->
`tanitad.data.calib.focal_crop_resize`). YouTube dashcams have UNKNOWN, VARYING
intrinsics, so a single assumed HFOV biases the crop -> the apparent-motion scale
-> the pseudo speed/trajectory the IDM head labels. This module estimates each
clip's intrinsics from its own frames with GeoCalib (ECCV 2024, cvg/GeoCalib) and
produces the SAME crop/resize the pilot uses, but at the per-video focal.

DROP-IN CONTRACT (what the pseudo-label geometry step needs):
    est = GeoCalibEstimator().estimate_from_video(mp4_path)      # robust, N frames
    f_px = est.focal_px(width=W, height=H)                       # focal at decode res
    canon = focal_crop_resize(frame_tchw, f_px, size=256)        # -> f_eff ~= 266
  or, as a single call mirroring yt_pilot_common.decode_canonical:
    frames_u8, meta = decode_canonical_geocalib(mp4, anonymizer, estimator=est)

The ONLY change vs the pilot is the FOCAL SOURCE: GeoCalib per-video instead of a
fixed HFOV. Everything downstream (`focal_crop_resize`, the 9-ch stack, the encoder
contract) is byte-identical, so this is a pure geometry-front-end swap.

VALIDATED (MEASURED 2026-07-25, eval pod A40; see geocalib_validation_results.json
+ VALIDATION_REPORT.md): on comma2k19 (GT focal 910 px) the `distorted` weights
recover focal to median |err| ~7%, vFoV err ~+3.6 deg, and are resolution-robust
(native == ~480p). GeoCalib has per-FRAME outliers (up to +/-30 %) and regresses
toward a ~50-55 deg vFoV prior for narrow fields, so SINGLE-FRAME use is unsafe:
this module robust-aggregates over N frames (MAD outlier rejection) and FALLS BACK
to the fixed HFOV when GeoCalib disagrees with itself (low confidence). On true
wide fisheye (PhysicalAI 120 deg f-theta) GeoCalib's pinhole fit is a poor match
(reports ~25-55 deg) -> `confidence="low"` / fallback is expected and correct.

Evidence class: MEASURED (the estimator runs; recovery numbers in the report).
No hard dependency on the tanitad stack for estimation; `focal_crop_resize` and
`av` are imported lazily only where used, so `import geocalib_intrinsics` is cheap
and safe even where GeoCalib is not installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

# GeoCalib defaults (chosen from the 2026-07-25 known-GT validation)
DEFAULT_WEIGHTS = "distorted"       # beat "pinhole": comma |f_err| 9.7% vs 14.4%
DEFAULT_CAMERA_MODEL = "pinhole"    # LM fits a pinhole -> f/vfov directly usable
DEFAULT_HFOV_FALLBACK_DEG = 100.0   # the pilot's fixed assumption (fallback only)
DEFAULT_N_FRAMES = 16
# Decode SINGLE-THREADED. A threaded PyAV decoder ("AUTO") that is torn down while
# a CUDA context is live can DEADLOCK on container close — and this module decodes
# right after GeoCalib (CUDA) runs, per clip, thousands of times. Single-threaded
# decode of small frames is not the bottleneck and removes that deadlock surface.
# (MEASURED 2026-07-25: "AUTO" hung intermittently on close post-CUDA; "NONE" clean.)
DECODE_THREAD_TYPE = "NONE"
# confidence / fallback thresholds on the per-frame vFoV spread (MAD, degrees)
MAD_HIGH_DEG = 3.0
MAD_MEDIUM_DEG = 6.0
MAD_FALLBACK_DEG = 9.0              # above this we do not trust GeoCalib -> fallback
MIN_VALID_FRAMES = 4


def focal_from_vfov(vfov_deg: float, height: int) -> float:
    """Pinhole focal [px] that yields vertical FoV ``vfov_deg`` over ``height`` px."""
    return (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def vfov_from_focal(focal_px: float, height: int) -> float:
    return math.degrees(2.0 * math.atan((height / 2.0) / focal_px))


def hfov_from_focal(focal_px: float, width: int) -> float:
    return math.degrees(2.0 * math.atan((width / 2.0) / focal_px))


@dataclass
class EstimatedIntrinsics:
    """Robust per-video intrinsics estimate + provenance.

    ``vfov_deg`` is the load-bearing quantity: it is resolution-independent, so
    ``focal_px(W, H)`` reconstructs the focal for ANY decode resolution. When
    ``fallback_used`` is True the numbers are the fixed-HFOV assumption, NOT a
    GeoCalib estimate (GeoCalib was unavailable or low-confidence).
    """
    vfov_deg: float
    hfov_deg: float
    focal_px_at_est: float          # median focal at the estimation resolution
    est_width: int
    est_height: int
    n_frames_total: int
    n_frames_used: int
    vfov_mad_deg: float             # per-frame spread (robust) -> confidence
    focal_uncertainty_med: float    # GeoCalib's own median focal uncertainty [px]
    confidence: str                 # "high" | "medium" | "low"
    fallback_used: bool
    weights: str = DEFAULT_WEIGHTS
    camera_model: str = DEFAULT_CAMERA_MODEL
    fallback_hfov_deg: float = DEFAULT_HFOV_FALLBACK_DEG
    per_frame_vfov_deg: list = field(default_factory=list)

    def focal_px(self, width: int, height: int) -> float:
        """Focal [px] for a frame of ``height`` px (via the estimated vFoV).

        This is what feeds ``tanitad.data.calib.focal_crop_resize(vid, f_px, size)``
        so the crop lands at the canonical ``f_eff ~= F_REF (266)``. ``width`` is
        accepted for symmetry / logging; the crop uses the vertical field.
        """
        return focal_from_vfov(self.vfov_deg, height)

    def as_dict(self) -> dict:
        return asdict(self)


def _fixed_hfov_estimate(width: int, height: int,
                         hfov_deg: float = DEFAULT_HFOV_FALLBACK_DEG,
                         n_total: int = 0) -> EstimatedIntrinsics:
    """The pilot's fixed-HFOV geometry, packaged as an EstimatedIntrinsics so the
    caller has ONE code path (GeoCalib when confident, this otherwise)."""
    f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return EstimatedIntrinsics(
        vfov_deg=vfov_from_focal(f, height), hfov_deg=hfov_deg,
        focal_px_at_est=f, est_width=width, est_height=height,
        n_frames_total=n_total, n_frames_used=0, vfov_mad_deg=float("nan"),
        focal_uncertainty_med=float("nan"), confidence="low", fallback_used=True,
        fallback_hfov_deg=hfov_deg)


class GeoCalibEstimator:
    """Wraps GeoCalib for robust, per-video intrinsics estimation.

    GeoCalib is imported and the network is built lazily on first use, so
    constructing this object is cheap and importing the module never requires
    GeoCalib to be installed.
    """

    def __init__(self, weights: str = DEFAULT_WEIGHTS,
                 camera_model: str = DEFAULT_CAMERA_MODEL,
                 device: Optional[str] = None,
                 hfov_fallback_deg: float = DEFAULT_HFOV_FALLBACK_DEG):
        self.weights = weights
        self.camera_model = camera_model
        self.hfov_fallback_deg = hfov_fallback_deg
        self._device = device
        self._model = None

    # -- lazy model ------------------------------------------------------- #
    def _ensure(self):
        if self._model is None:
            import torch
            from geocalib import GeoCalib
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = GeoCalib(weights=self.weights).to(self._device)
            self._model.eval()
        return self._model

    # -- one frame -------------------------------------------------------- #
    def _calibrate_frame(self, rgb_u8: np.ndarray) -> Optional[dict]:
        """rgb_u8 [H,W,3] uint8 -> {focal_px, vfov_deg, hfov_deg, focal_unc} or None."""
        import torch
        model = self._ensure()
        t = (torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1)
             .float().div(255.0).to(self._device))
        try:
            res = model.calibrate(t, camera_model=self.camera_model)
        except Exception:
            return None
        cam = res["camera"]
        f = float(cam.f.detach().cpu().numpy().ravel()[0])
        unc = res.get("focal_uncertainty")
        unc = float(unc.detach().cpu().numpy().ravel()[0]) if unc is not None else float("nan")
        if not math.isfinite(f) or f <= 1.0:
            return None
        return {"focal_px": f, "vfov_deg": math.degrees(float(cam.vfov)),
                "hfov_deg": math.degrees(float(cam.hfov)), "focal_unc": unc}

    # -- robust aggregate over frames ------------------------------------- #
    def estimate_from_frames(self, frames: Sequence[np.ndarray]
                             ) -> EstimatedIntrinsics:
        """Robust intrinsics over a list of [H,W,3] uint8 frames.

        Per-frame GeoCalib -> median vFoV with MAD outlier rejection. Falls back to
        the fixed HFOV when too few frames survive or the spread is too large
        (GeoCalib disagreeing with itself == untrustworthy clip)."""
        frames = list(frames)
        if not frames:
            raise ValueError("no frames provided")
        H, W = frames[0].shape[:2]
        rows = [r for r in (self._calibrate_frame(f) for f in frames) if r is not None]
        n_total = len(frames)
        if len(rows) < MIN_VALID_FRAMES:
            return _fixed_hfov_estimate(W, H, self.hfov_fallback_deg, n_total)

        vfovs = np.array([r["vfov_deg"] for r in rows], float)
        med = float(np.median(vfovs))
        mad = float(np.median(np.abs(vfovs - med))) * 1.4826
        # reject frames > 3*MAD from the median (keep all if MAD ~ 0)
        keep = np.abs(vfovs - med) <= (3.0 * mad + 1e-6) if mad > 1e-6 else np.ones_like(vfovs, bool)
        surv = [r for r, k in zip(rows, keep) if k]
        if len(surv) < MIN_VALID_FRAMES:
            surv = rows
        sv = np.array([r["vfov_deg"] for r in surv], float)
        vfov_med = float(np.median(sv))
        vfov_mad = float(np.median(np.abs(sv - vfov_med))) * 1.4826
        focal_med = float(np.median([r["focal_px"] for r in surv]))
        unc_med = float(np.nanmedian([r["focal_unc"] for r in surv]))

        if vfov_mad <= MAD_HIGH_DEG:
            conf = "high"
        elif vfov_mad <= MAD_MEDIUM_DEG:
            conf = "medium"
        else:
            conf = "low"
        if vfov_mad > MAD_FALLBACK_DEG or not math.isfinite(vfov_med):
            return _fixed_hfov_estimate(W, H, self.hfov_fallback_deg, n_total)

        return EstimatedIntrinsics(
            vfov_deg=round(vfov_med, 4),
            hfov_deg=round(hfov_from_focal(focal_from_vfov(vfov_med, H), W), 4),
            focal_px_at_est=round(focal_med, 3), est_width=W, est_height=H,
            n_frames_total=n_total, n_frames_used=len(surv),
            vfov_mad_deg=round(vfov_mad, 4),
            focal_uncertainty_med=round(unc_med, 4) if math.isfinite(unc_med) else float("nan"),
            confidence=conf, fallback_used=False, weights=self.weights,
            camera_model=self.camera_model, fallback_hfov_deg=self.hfov_fallback_deg,
            per_frame_vfov_deg=[round(v, 3) for v in vfovs.tolist()])

    # -- decode N frames from a video and estimate ------------------------ #
    def estimate_from_video(self, mp4_path: str, *, n_frames: int = DEFAULT_N_FRAMES,
                            skip_head: int = 30, anonymizer=None,
                            max_side: Optional[int] = None) -> EstimatedIntrinsics:
        """Decode ~``n_frames`` evenly-spaced frames from a video and estimate.

        ``anonymizer`` (optional, e.g. ``yt_pilot_common.Anonymizer``) is applied to
        each full-res RGB frame before it touches GeoCalib, matching the pilot's
        privacy design (no imagery persists). ``max_side`` optionally downscales for
        speed; GeoCalib's vFoV is resolution-independent so this does not bias the
        estimate."""
        import av
        # first pass: count decodable frames cheaply via the stream if possible
        frames: list = []
        with av.open(str(mp4_path)) as c:
            st = c.streams.video[0]
            st.thread_type = DECODE_THREAD_TYPE
            total = st.frames or 0
            # choose a stride to sample n_frames after skip_head
            if total and total > skip_head + n_frames:
                idxs = set(np.linspace(skip_head, total - 1, n_frames).astype(int).tolist())
            else:
                idxs = None  # take the first n_frames after skip_head
            i = 0
            for fr in c.decode(st):
                take = (i in idxs) if idxs is not None else (i >= skip_head)
                if take:
                    rgb = fr.to_ndarray(format="rgb24")
                    if anonymizer is not None:
                        rgb = anonymizer(np.ascontiguousarray(rgb))
                    if max_side:
                        import torch, torch.nn.functional as F
                        H, W = rgb.shape[:2]
                        s = min(1.0, max_side / float(max(H, W)))
                        if s < 1.0:
                            t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
                            t = F.interpolate(t, size=(int(H * s), int(W * s)),
                                              mode="bilinear", align_corners=False)
                            rgb = t[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
                    frames.append(np.ascontiguousarray(rgb))
                    if idxs is None and len(frames) >= n_frames:
                        break
                i += 1
        if not frames:
            raise RuntimeError(f"no frames decoded from {mp4_path}")
        return self.estimate_from_frames(frames)


# --------------------------------------------------------------------------- #
# Drop-in for yt_pilot_common.decode_canonical (fixed-HFOV -> GeoCalib focal)  #
# --------------------------------------------------------------------------- #
def decode_canonical_geocalib(mp4_path: str, anonymizer, *,
                              estimator: Optional[GeoCalibEstimator] = None,
                              estimated: Optional[EstimatedIntrinsics] = None,
                              size: int = 256, target_hz: float = 10.0,
                              max_frames: Optional[int] = None,
                              hfov_fallback_deg: float = DEFAULT_HFOV_FALLBACK_DEG,
                              n_estimate_frames: int = DEFAULT_N_FRAMES):
    """Mirror of ``yt_pilot_common.decode_canonical`` but with a per-video GeoCalib
    focal instead of the fixed ``nominal_focal_px(W, hfov_deg)``.

    Returns ``(frames_u8 [T,3,size,size], meta)``. ``meta`` carries the full
    EstimatedIntrinsics so the pseudo-label record is self-describing. If GeoCalib
    is low-confidence/unavailable the estimate's ``fallback_used`` is True and the
    geometry is exactly the pilot's fixed-HFOV crop — i.e. this is a safe swap that
    never does WORSE than the pilot.
    """
    import av
    import torch
    from tanitad.data.calib import focal_crop_resize   # lazy: only here

    if estimated is None:
        est = estimator or GeoCalibEstimator(hfov_fallback_deg=hfov_fallback_deg)
        estimated = est.estimate_from_video(
            mp4_path, n_frames=n_estimate_frames, anonymizer=anonymizer)

    frames = []
    src_fps = None
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = DECODE_THREAD_TYPE
        tb = stream.time_base
        try:
            src_fps = float(stream.average_rate) if stream.average_rate else None
        except Exception:
            src_fps = None
        dt = 1.0 / target_hz
        next_t = 0.0
        # index-based fallback stride for containers without usable pts (raw
        # elementary streams). YouTube mp4s carry pts -> the timestamp path is used.
        stride = max(1, int(round((src_fps or target_hz) / target_hz))) if src_fps else 1
        fi = -1
        for frame in container.decode(stream):
            fi += 1
            if frame.pts is not None and tb is not None:
                t = float(frame.pts * tb)
                if t + 1e-6 < next_t:
                    continue
            else:                              # no pts: resample by frame index
                if fi % stride != 0:
                    continue
            rgb = frame.to_ndarray(format="rgb24")
            rgb = anonymizer(np.ascontiguousarray(rgb))
            H, W = rgb.shape[:2]
            f_px = estimated.focal_px(width=W, height=H)     # <-- GeoCalib per-video
            vt = torch.from_numpy(rgb).permute(2, 0, 1)[None]
            canon = focal_crop_resize(vt, f_px, size)
            frames.append(canon[0])
            next_t += dt
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError("no frames decoded")
    vid = torch.stack(frames)
    # achieved canonical focal. It equals F_REF (~266) UNLESS the canonical crop
    # exceeded the frame (narrow FoV + low res) and clamped to full frame -> then
    # f_eff > 266 (under-canonicalized). Surfaced so the pseudo-label record can
    # flag clips the pipeline could NOT fully canonicalize at this resolution.
    achieved_f_eff = float(getattr(focal_crop_resize, "last_f_eff", float("nan")))
    meta = {"src_fps": src_fps, "n_frames_10hz": int(vid.shape[0]), "size": size,
            "geometry": "geocalib_per_video", "intrinsics": estimated.as_dict(),
            "hfov_used_deg": estimated.hfov_deg, "achieved_f_eff": round(achieved_f_eff, 2),
            "fully_canonical": abs(achieved_f_eff - 266.0) / 266.0 < 0.02,
            "geocalib_fallback_used": estimated.fallback_used,
            "anon": dict(anonymizer.stats) if hasattr(anonymizer, "stats") else None}
    return vid, meta
