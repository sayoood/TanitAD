"""Video <-> sensor time synchronisation.

Three independent estimates of the video start time are produced and reconciled:

1.  **Container-anchor** -- ``Camera/<epoch_ms>.mp4``.  The stem equals
    ``Metadata.csv:recording epoch time``, i.e. the moment *logging* started.
    The camera pipeline needs a few hundred ms to deliver its first frame, so
    this is a lower bound, not the start.

2.  **creation_time minus duration**.  Android's MP4 muxer writes
    ``format.tags.creation_time`` when it *finalises* the file, i.e. at the END
    of the recording -- verified on this data set: 15:16:21.000Z minus the
    77.713 s duration lands 0.537 s after the logging epoch, whereas treating it
    as the start would place the video's first frame after the sensor log had
    already ended.  This is the accurate anchor.

3.  **Motion cross-correlation** -- the ground truth.  The angular-rate
    magnitude recovered from frame-to-frame image motion is correlated against
    the gyroscope's angular-rate magnitude.  Both are invariant to the unknown
    phone->vehicle rotation, so this works before any extrinsic calibration and
    directly measures the residual offset left by (2).

Frame timestamps always come from real container PTS, never ``index / fps``:
this recording is nominally 30 fps but actually averages 29.943 fps with
per-frame jitter, which accumulates to ~1.7 s of drift over 78 s if ignored.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


def _ff(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"{name} not found on PATH - required for video timing")
    return exe


# --------------------------------------------------------------------------- #
# Container metadata
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    duration: float
    nb_frames: int
    avg_fps: float
    creation_time_utc: float | None   # epoch seconds, as written in the container
    pts: np.ndarray                   # (n,) presentation timestamps, seconds from stream start

    @property
    def stem_epoch_ms(self) -> int | None:
        stem = os.path.splitext(os.path.basename(self.path))[0]
        return int(stem) if stem.isdigit() and len(stem) == 13 else None


def probe_video(path: str, read_pts: bool = True) -> VideoInfo:
    """Read stream metadata and (optionally) every frame's PTS."""
    out = subprocess.run(
        [_ff("ffprobe"), "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    vs = next(s for s in meta["streams"] if s["codec_type"] == "video")

    ct = vs.get("tags", {}).get("creation_time") or meta["format"].get("tags", {}).get("creation_time")
    ct_epoch = None
    if ct:
        s = ct.replace("Z", "+00:00")
        ct_epoch = datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() \
            if "+" not in s else datetime.fromisoformat(s).timestamp()

    pts = np.array([], dtype=float)
    if read_pts:
        raw = subprocess.run(
            [_ff("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True).stdout
        # ffprobe's csv writer appends a field separator on some builds and for
        # some containers -- the iPhone HEVC files here emit "0.000000," -- and a
        # missing timestamp comes through as "N/A".  Parse defensively: a strict
        # float() over whitespace-split tokens dies on the first comma.
        vals = []
        for tok in raw.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue                      # N/A and friends
        pts = np.array(vals, dtype=float)
        if len(pts) and not np.all(np.diff(pts) >= 0):
            pts = np.sort(pts)                # B-frames can emit out of order

    num, den = (vs.get("avg_frame_rate") or "0/1").split("/")
    avg_fps = float(num) / float(den) if float(den) else 0.0

    return VideoInfo(
        path=path,
        width=int(vs["width"]),
        height=int(vs["height"]),
        duration=float(vs.get("duration") or meta["format"]["duration"]),
        nb_frames=int(vs.get("nb_frames") or len(pts)),
        avg_fps=avg_fps,
        creation_time_utc=ct_epoch,
        pts=pts,
    )


# --------------------------------------------------------------------------- #
# Coarse anchors
# --------------------------------------------------------------------------- #
@dataclass
class SyncResult:
    t_video_start: float          # video PTS 0 expressed in session `seconds_elapsed`
    source: str                   # which estimate won
    anchor_container: float | None
    anchor_creation_end: float | None
    xcorr_shift: float | None     # residual applied on top of the coarse anchor
    xcorr_peak: float | None      # normalised correlation at the peak, in [-1, 1]
    anchor_creation_start: float | None = None
    xcorr_curve: tuple | None = None   # (lags, correlation) for plotting

    def frame_times(self, pts: np.ndarray) -> np.ndarray:
        """Session `seconds_elapsed` for each frame."""
        return self.t_video_start + np.asarray(pts, dtype=float)


def coarse_anchors(video: VideoInfo, epoch_ms: int) -> dict:
    """Container-derived estimates of the video start, in session seconds.

    ``creation_time`` means different things on the two platforms, measured on
    real exports (value = implied video start, in session seconds):

        recording                    container   ct as START   ct as END
        Pacific Coast Hwy (iOS)          0.105         0.076    -264.639
        Rose Ave (iOS)                   0.093        -0.316    -631.813
        Liebstoeckelweg (Android)        0.000        78.250       0.537

    Android writes it when the muxer *finalises* the file; iOS writes it at the
    *start*.  Picking one interpretation would be wrong on half the corpus, so
    both are offered and the caller keeps whichever is physically possible --
    the camera cannot start before logging does, nor minutes afterwards.
    """
    res: dict[str, float | None] = {"container": None,
                                    "creation_start": None, "creation_end": None}
    if video.stem_epoch_ms is not None:
        res["container"] = (video.stem_epoch_ms - epoch_ms) / 1000.0
    if video.creation_time_utc is not None:
        res["creation_start"] = video.creation_time_utc - epoch_ms / 1000.0
        res["creation_end"] = res["creation_start"] - video.duration
    return res


def pick_anchor(anchors: dict, lo: float = -3.0, hi: float = 15.0):
    """Choose the coarse anchor that a camera could actually have produced.

    The container stem is preferred when plausible: it is the logging epoch the
    app itself wrote, and it agreed to within 0.11 s on every recording tested.
    """
    order = ("container", "creation_start", "creation_end")
    for k in order:
        v = anchors.get(k)
        if v is not None and lo <= v <= hi:
            return v, k
    cand = [(abs(v), v, k) for k, v in anchors.items() if v is not None]
    if cand:
        _, v, k = min(cand)
        return v, f"{k} (implausible {v:.1f}s; nothing better available)"
    return 0.0, "no container timing available"


# --------------------------------------------------------------------------- #
# Image-derived angular rate
# --------------------------------------------------------------------------- #
def _iter_gray_frames(path: str, width: int, height: int, first: int = 0,
                      count: int | None = None):
    """Yield down-scaled grayscale frames straight out of ffmpeg (fast path).

    Decoding always starts at frame 0 and unwanted frames are dropped here.
    Input seeking (``-ss`` before ``-i``) is deliberately avoided: it snaps to
    the preceding keyframe, which on this ~1 s-GOP phone footage silently
    shifted every timestamp by up to a second and poisoned the lag estimate.
    """
    cmd = [_ff("ffmpeg"), "-v", "error", "-i", path,
           "-vf", f"scale={width}:{height}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    n = width * height
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 8)
    emitted = 0
    try:
        k = 0
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            if k >= first:
                # copy: frombuffer is read-only and OpenCV needs writable input
                yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width).copy()
                emitted += 1
                if count is not None and emitted >= count:
                    break
            k += 1
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()


def image_angular_rate(video: VideoInfo, t_start: float = 0.0, t_dur: float | None = None,
                       proc_width: int = 480, max_features: int = 600,
                       roi: tuple = (0.06, 0.86), fov_deg: float = 66.0):
    """Camera angular rates recovered from frame-to-frame image motion.

    Returns ``(t_mid, omega_mag, comps)``: PTS midpoints between consecutive
    frames, the angular-rate magnitude, and an (n, 3) array of
    (yaw, pitch, roll) rates -- all in rad/s.

    LK-tracked corners are fitted with a partial-affine (similarity) transform;
    ``tx/f`` and ``ty/f`` are then the yaw and pitch increments and the in-plane
    angle is the roll increment.

    ``roi`` clips the analysis band vertically as a fraction of image height.
    The default drops the top 6 % (featureless sky) and the bottom 14 %: the
    car's hood and dashboard are *static* in the image, and RANSAC will happily
    lock onto that stationary majority and report zero motion.

    Forward translation still leaks into ``tx`` whenever the tracked features
    sit asymmetrically about the focus of expansion, so the yaw amplitude here
    is only approximate.  That is fine -- only the *timing* of the signal is
    used, and the leak is broadband rather than delayed.
    """
    import cv2

    scale = proc_width / video.width
    pw, ph = proc_width, int(round(video.height * scale / 2) * 2)
    f_px = (pw / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)

    pts = video.pts
    if t_dur is None:
        first, last = 0, len(pts)
    else:
        idxs = np.nonzero((pts >= t_start) & (pts < t_start + t_dur))[0]
        if len(idxs) < 3:
            raise ValueError("not enough frames in the requested range")
        first, last = int(idxs[0]), int(idxs[-1]) + 1
    sel = pts[first:last]

    y0, y1 = int(roi[0] * ph), int(roi[1] * ph)
    mask = np.zeros((ph, pw), dtype=np.uint8)
    mask[y0:y1, :] = 255

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    feat = dict(maxCorners=max_features, qualityLevel=0.01, minDistance=6, blockSize=7)
    cx, cy = pw / 2.0, ph / 2.0
    c_h = np.array([cx, cy, 1.0])
    c_2 = np.array([cx, cy])

    src = (video.iter_gray(pw, ph, first, last - first) if hasattr(video, "iter_gray")
           else _iter_gray_frames(video.path, pw, ph, first, last - first))

    prev = None
    rows = []
    for k, frame in enumerate(src):
        if prev is None:
            prev = frame.copy()
            continue
        p0 = cv2.goodFeaturesToTrack(prev, mask=mask, **feat)
        yaw = pitch = roll = np.nan
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, frame, p0, None, **lk)
            if p1 is not None and st is not None and st.sum() >= 12:
                a, b = p0[st.ravel() == 1], p1[st.ravel() == 1]
                M, inl = cv2.estimateAffinePartial2D(
                    a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                    maxIters=3000, confidence=0.995)
                if M is not None and inl is not None and inl.sum() >= 10:
                    # The affine's translation column is referenced to the image
                    # ORIGIN, so an in-plane rotation dumps a large spurious
                    # offset into it.  Evaluate the displacement at the principal
                    # point instead -- that is the true boresight motion.
                    d = M @ c_h - c_2
                    yaw = -d[0] / f_px             # image shifts +x  <->  camera yaws -x
                    pitch = -d[1] / f_px
                    roll = np.arctan2(M[1, 0], M[0, 0])
        rows.append((k, yaw, pitch, roll))
        prev = frame.copy()

    rows = np.array(rows, dtype=float)
    idx = rows[:, 0].astype(int)
    dt = np.diff(sel)[idx - 1]
    comps = rows[:, 1:] / dt[:, None]
    t_mid = 0.5 * (sel[idx - 1] + sel[idx])
    omega = np.sqrt(np.nansum(comps ** 2, axis=1))
    omega[np.all(np.isnan(comps), axis=1)] = np.nan
    return t_mid, omega, comps


def gyro_yaw_rate(session) -> tuple:
    """Yaw rate about the gravity direction -- independent of how the phone sits.

    Android reports the gravity vector pointing *up* (a phone lying flat and
    face-up reads +9.81 on z), so the yaw rate about the downward vertical is
    the negated projection of the angular rate onto it.
    """
    gy = session["gyro"]
    gr = session["gravity"]
    t = gy["seconds_elapsed"].to_numpy(dtype=float)
    w = gy[["x", "y", "z"]].to_numpy(dtype=float)
    tg = gr["seconds_elapsed"].to_numpy(dtype=float)
    g = np.stack([np.interp(t, tg, gr[c].to_numpy(dtype=float)) for c in "xyz"], axis=1)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    return t, -np.sum(w * g, axis=1)


# --------------------------------------------------------------------------- #
# Cross-correlation of the two angular-rate magnitudes
# --------------------------------------------------------------------------- #
def _bandpass(x: np.ndarray, fs: float, lo: float = 0.15, hi: float = 4.0) -> np.ndarray:
    """Zero-phase band-pass: kills DC/bias drift and above-Nyquist gyro noise."""
    from scipy.signal import butter, sosfiltfilt
    hi = min(hi, 0.45 * fs)
    sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def estimate_time_shift(t_cam: np.ndarray, sig_cam: np.ndarray,
                        t_imu: np.ndarray, sig_imu: np.ndarray,
                        max_shift: float = 3.0, dt: float = 0.02,
                        band=(0.15, 4.0)):
    """Lag ``tau`` such that ``sig_cam(t)`` best matches ``sig_imu(t + tau)``.

    Positive ``tau`` means the camera timeline is *behind* the IMU timeline, i.e.
    the true video start is ``t_video_start + tau``.

    Returns ``(tau, peak_corr, lags, corr)``.
    """
    ok = np.isfinite(sig_cam)
    t_cam, sig_cam = t_cam[ok], sig_cam[ok]
    if len(t_cam) < 20:
        raise ValueError("too few valid camera samples for cross-correlation")

    lo = max(t_cam[0], t_imu[0] + max_shift)
    hi = min(t_cam[-1], t_imu[-1] - max_shift)
    if hi - lo < 5.0:
        raise ValueError("overlap between camera and IMU too short to correlate")

    grid = np.arange(lo, hi, dt)
    fs = 1.0 / dt
    a = _bandpass(np.interp(grid, t_cam, sig_cam), fs, *band)
    a = (a - a.mean()) / (a.std() + 1e-12)

    lags = np.arange(-max_shift, max_shift + dt / 2, dt)
    corr = np.empty(len(lags))
    for i, tau in enumerate(lags):
        b = np.interp(grid + tau, t_imu, sig_imu)
        b = _bandpass(b, fs, *band)
        b = (b - b.mean()) / (b.std() + 1e-12)
        corr[i] = float(np.mean(a * b))

    k = int(np.argmax(corr))
    tau = float(lags[k])
    # sub-sample refinement by fitting a parabola through the peak
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            tau += dt * 0.5 * (y0 - y2) / denom
    return tau, float(corr[k]), lags, corr


def estimate_shift_3axis(t_cam: np.ndarray, w_cam: np.ndarray,
                         t_imu: np.ndarray, w_imu: np.ndarray,
                         max_shift: float = 3.0, dt: float = 0.02,
                         band=(0.02, 1.0)):
    """Joint lag + camera<-phone axis-map fit on the full 3-axis angular rate.

    For every candidate lag ``tau`` the gyro triad is resampled onto the camera
    times and a free 3x3 map ``M`` is fitted with ``w_cam ~ w_imu @ M``.  The
    score is the fraction of camera-signal variance that map explains.

    Letting ``M`` be a general matrix rather than a rotation is deliberate: the
    image-derived rates carry an unknown gain (forward translation past nearby
    buildings leaks into the apparent yaw and inflates it several-fold on this
    footage), and a free matrix absorbs that gain per axis instead of fighting
    it.  Using all three axes at once is what makes the peak sharp -- any single
    axis alone is far more ambiguous.

    Returns ``(tau, score, lags, scores, M)``.
    """
    ok = np.all(np.isfinite(w_cam), axis=1)
    t_cam, w_cam = t_cam[ok], w_cam[ok]
    if len(t_cam) < 40:
        raise ValueError("too few valid camera samples for 3-axis correlation")

    lo = max(t_cam[0], t_imu[0] + max_shift)
    hi = min(t_cam[-1], t_imu[-1] - max_shift)
    if hi - lo < 5.0:
        raise ValueError("overlap between camera and IMU too short to correlate")

    grid = np.arange(lo, hi, dt)
    fs = 1.0 / dt
    A = np.stack([_bandpass(np.interp(grid, t_cam, w_cam[:, j]), fs, *band) for j in range(3)], axis=1)
    A -= A.mean(axis=0)
    denom = float(np.sum(A ** 2)) + 1e-30

    lags = np.arange(-max_shift, max_shift + dt / 2, dt)
    scores = np.empty(len(lags))
    best = (None, -np.inf)
    for i, tau in enumerate(lags):
        B = np.stack([_bandpass(np.interp(grid + tau, t_imu, w_imu[:, j]), fs, *band)
                      for j in range(3)], axis=1)
        B -= B.mean(axis=0)
        M, *_ = np.linalg.lstsq(B, A, rcond=None)
        resid = A - B @ M
        s = 1.0 - float(np.sum(resid ** 2)) / denom
        scores[i] = s
        if s > best[1]:
            best = (M, s)

    k = int(np.argmax(scores))
    tau = float(lags[k])
    if 0 < k < len(scores) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-12:
            tau += dt * 0.5 * (y0 - y2) / d
    return tau, float(scores[k]), lags, scores, best[0]


_BANDS = [(0.02, 1.0), (0.05, 2.0), (0.10, 3.0)]


def synchronise(session, video: VideoInfo, refine: bool = True,
                max_shift: float = 3.0, min_score: float = 0.10,
                max_band_spread: float = 0.10, cam_flow=None) -> SyncResult:
    """Full sync: take the container anchor, then refine against the gyroscope.

    The refinement is run in three frequency bands.  Agreement between them is
    the acceptance test: a genuine lag is band-independent, whereas a spurious
    correlation peak moves around.  If the bands disagree by more than
    ``max_band_spread`` seconds the refinement is rejected and the coarse
    container anchor is kept, with ``source`` recording that fact.
    """
    anchors = coarse_anchors(video, session.epoch_ms)
    t0, source = pick_anchor(anchors)

    shift = score = curve = None
    if refine and session.has("gyro"):
        gy = session["gyro"]
        t_g = gy["seconds_elapsed"].to_numpy(dtype=float)
        w_g = gy[["x", "y", "z"]].to_numpy(dtype=float)
        if cam_flow is None:
            cam_flow = image_angular_rate(video)
        t_cam, _, comps = cam_flow

        taus, scores_at_peak = [], []
        for band in _BANDS:
            tau, sc, lags, scores, _ = estimate_shift_3axis(
                t_cam + t0, comps, t_g, w_g, max_shift=max_shift, band=band)
            taus.append(tau)
            scores_at_peak.append(sc)
            if band == _BANDS[0]:
                curve = (lags, scores)

        spread = float(np.max(taus) - np.min(taus))
        best = float(np.median(taus))
        score = float(np.max(scores_at_peak))
        if spread <= max_band_spread and score >= min_score:
            shift = best
            t0 = t0 + best
            source = f"gyro-xcorr (bands agree to {spread * 1e3:.0f} ms)"
        else:
            source += f" (refinement rejected: spread={spread:.3f}s score={score:.2f})"

    return SyncResult(t_video_start=float(t0), source=source,
                      anchor_container=anchors["container"],
                      anchor_creation_end=anchors["creation_end"],
                      anchor_creation_start=anchors["creation_start"],
                      xcorr_shift=shift, xcorr_peak=score, xcorr_curve=curve)
