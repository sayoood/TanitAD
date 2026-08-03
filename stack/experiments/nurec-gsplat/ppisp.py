"""ppisp.py -- the NuRec scene's own trained post-processing ISP, applied.

A NuRec ``volume.nurec`` ships a trained PPISP (Physically-Plausible Image Signal
Processor) alongside the gaussians. This module reads those parameters out of the
msgpack and applies them, as a faithful numpy port of the published reference
implementation.

PROVENANCE
  PUBLISHED  nv-tlabs/ppisp v1.0.0, ``tests/torch_reference.py`` (Apache-2.0), and
             arXiv:2601.18336 "PPISP: Physically-Plausible Compensation and Control of
             Photometric Variations in Radiance Field Reconstruction".
  MEASURED   parameter locations, dtypes, shapes and VALUES read from
             00040136-e651-4abd-991d-0655ccda9430 : volume.nurec (nre version 26.4.96).

WHERE THE PARAMETERS LIVE (MEASURED -- all fp16, all under ``state_dict``)

    .post_processings.0.ppisp.exposure_params    [n_views]      per VIEW, log2 gain
    .post_processings.0.ppisp.vignetting_params  [n_cams, 3, 5] per camera per channel
    .post_processings.0.ppisp.color_params       [n_views, 8]   per VIEW, latent chroma
    .post_processings.0.ppisp.crf_params         [n_cams, 3, K] per camera per channel

  ``n_views = n_cams * n_frames``, CAMERA-MAJOR (a contiguous block of n_frames per
  camera). For this scene 3594 = 6 x 599.

⛔ VERSION MISMATCH, AND IT IS LOAD-BEARING
  The published v1.0.0 header declares ``PPISP_CRF_PARAMS_PER_CHANNEL = 4``
  (toe, shoulder, gamma, center). **This 26.04 scene stores K = 7.** The extra three
  slots are undocumented and the ordering of the four known quantities inside the seven
  is NOT determined by the public repo. ``apply()`` therefore takes an explicit
  ``crf_perm``; there is no default that can be trusted. Applying the public first-4
  reading to this file makes the render WORSE (see FINDINGS.md). Treat any CRF result
  from this file as UNIDENTIFIED until the 26.04 layout is published or recovered.

  The other three stages have no such ambiguity and are applied exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

PREFIX = ".post_processings.0.ppisp."
PUBLISHED_CRF_PARAMS_PER_CHANNEL = 4  # nv-tlabs/ppisp v1.0.0 ppisp_constants.h

# ZCA pinv blocks mapping latent colour params -> real chromaticity offsets,
# order [Blue, Red, Green, Neutral]. PUBLISHED: tests/torch_reference.py.
_ZCA_BLOCKS = (
    (0.0480542, -0.0043631, 0.0481283),
    (0.0580570, -0.0179872, 0.0431061),
    (0.0433336, -0.0180537, 0.0580500),
    (0.0128369, -0.0034654, 0.0128158),
)


def _block_diag() -> np.ndarray:
    m = np.zeros((8, 8), np.float64)
    for i, (a, b, c) in enumerate(_ZCA_BLOCKS):
        m[2 * i:2 * i + 2, 2 * i:2 * i + 2] = ((a, b), (b, c))
    return m


_BLK = _block_diag()

_softplus = lambda z: np.logaddexp(0.0, z)
_sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------------------
@dataclass
class PPISPParams:
    """The four trained tensors, straight out of the file. No activations applied."""

    exposure: np.ndarray      # [n_views]
    vignetting: np.ndarray    # [n_cams, 3, 5]
    color: np.ndarray         # [n_views, 8]
    crf: np.ndarray           # [n_cams, 3, K]
    n_cams: int
    n_frames: int
    per_frame_enabled: bool

    @property
    def crf_params_per_channel(self) -> int:
        return int(self.crf.shape[2])

    def view_index(self, cam_idx: int, frame: int) -> int:
        """View ordering is CAMERA-MAJOR (MEASURED: a block of n_frames per camera)."""
        if not 0 <= cam_idx < self.n_cams:
            raise IndexError(f"cam_idx {cam_idx} outside [0,{self.n_cams})")
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} outside [0,{self.n_frames})")
        return cam_idx * self.n_frames + frame

    def describe(self, cam_idx: int) -> dict:
        """What this camera's ISP actually does, before any image is touched.

        Cheap sanity report: several stages are frequently at their identity value
        (``per_frame_ppisp_enabled: false`` freezes the two per-view stages at init),
        and knowing that BEFORE rendering stops the ISP being blamed for a residual it
        cannot produce.
        """
        exp = self.exposure
        vig = self.vignetting[cam_idx]
        return {
            "exposure_all_zero": bool(np.all(exp == 0.0)),
            "exposure_gain_range": (float(2.0 ** exp.min()), float(2.0 ** exp.max())),
            "color_constant_across_views": bool(np.all(self.color.std(axis=0) == 0.0)),
            "color_row0": self.color[0].tolist(),
            "vignetting_center": vig[:, :2].tolist(),
            "vignetting_alpha_absmax": float(np.abs(vig[:, 2:]).max()),
            "crf_params_per_channel": self.crf_params_per_channel,
            "crf_matches_published_layout":
                self.crf_params_per_channel == PUBLISHED_CRF_PARAMS_PER_CHANNEL,
            "per_frame_ppisp_enabled": self.per_frame_enabled,
        }


def read_ppisp(nre_data: dict, n_cams: Optional[int] = None) -> PPISPParams:
    """Pull the PPISP tensors out of a decoded ``volume.nurec``.

    Every array carries an explicit ``<key>.shape`` sibling; we read it with ``[]`` (not
    ``.get``) so a missing stamp is a loud failure, and cross-check it against the raw
    byte count so the fp16 assumption is proven rather than assumed.
    """
    sd = nre_data["state_dict"]

    def arr(name: str) -> np.ndarray:
        key = PREFIX + name
        shape = [int(x) for x in sd[key + ".shape"]]
        raw = sd[key]
        n_elem = int(np.prod(shape)) if shape else 1
        if n_elem * 2 != len(raw):
            raise ValueError(
                f"{key}: shape {shape} = {n_elem} elems but {len(raw)} bytes "
                f"({len(raw) / max(n_elem, 1):.3f} B/elem) -- dtype is NOT float16"
            )
        return np.frombuffer(raw, np.float16).astype(np.float64).reshape(shape)

    exposure = arr("exposure_params")
    vignetting = arr("vignetting_params")
    color = arr("color_params")
    crf = arr("crf_params")

    if exposure.shape[0] != color.shape[0]:
        raise ValueError(f"exposure {exposure.shape} vs color {color.shape} view count")
    n_cams = int(vignetting.shape[0]) if n_cams is None else n_cams
    if crf.shape[0] != n_cams:
        raise ValueError(f"crf n_cams {crf.shape[0]} != vignetting {n_cams}")
    n_views = int(exposure.shape[0])
    if n_views % n_cams:
        raise ValueError(f"{n_views} views is not a multiple of {n_cams} cameras")

    pp = nre_data["config"]["post_processing"]
    blk = pp[next(iter(pp))]
    return PPISPParams(
        exposure=exposure, vignetting=vignetting, color=color, crf=crf,
        n_cams=n_cams, n_frames=n_views // n_cams,
        per_frame_enabled=bool(blk["per_frame_ppisp_enabled"]),
    )


# --------------------------------------------------------------------------------------
def color_homography(latent8: np.ndarray) -> np.ndarray:
    """Latent colour params [8] -> 3x3 chromaticity homography. PUBLISHED port."""
    off = np.asarray(latent8, np.float64) @ _BLK
    t_b = np.array([0.0 + off[0], 0.0 + off[1], 1.0])
    t_r = np.array([1.0 + off[2], 0.0 + off[3], 1.0])
    t_g = np.array([0.0 + off[4], 1.0 + off[5], 1.0])
    t_k = np.array([1 / 3 + off[6], 1 / 3 + off[7], 1.0])
    T = np.stack([t_b, t_r, t_g], axis=1)
    skew = np.array([[0.0, -t_k[2], t_k[1]],
                     [t_k[2], 0.0, -t_k[0]],
                     [-t_k[1], t_k[0], 0.0]])
    M = skew @ T
    cands = (np.cross(M[0], M[1]), np.cross(M[0], M[2]), np.cross(M[1], M[2]))
    lam = cands[int(np.argmax([c @ c for c in cands]))]
    s_inv = np.array([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    h = T @ np.diag(lam) @ s_inv
    return h / (h[2, 2] + 1e-10)


def crf_curve(x: np.ndarray, toe: float, shoulder: float, gamma: float,
              center: float) -> np.ndarray:
    """PUBLISHED toe-shoulder CRF: G(x) = f0(x; toe, shoulder, center) ** gamma.

    f0 is the C1-continuous piecewise power curve; ``a``/``b`` follow from continuity.
    """
    x = np.clip(x, 0.0, 1.0)
    eps = 1e-6
    lerp = toe + center * (shoulder - toe)
    a = (shoulder * center) / lerp
    b = 1.0 - a
    lo = a * np.power(np.clip(x / center, eps, None), toe)
    hi = 1.0 - b * np.power(np.clip((1.0 - x) / (1.0 - center), eps, None), shoulder)
    return np.power(np.clip(np.where(x <= center, lo, hi), eps, None), gamma)


def decode_crf(raw: Sequence[float], perm: Sequence[int]) -> Tuple[float, float, float, float]:
    """Raw stored params + a slot assignment -> (toe, shoulder, gamma, center).

    Activations are PUBLISHED (ppisp_math.cuh): softplus with a floor for the three
    positive quantities, sigmoid for the inflection point.
    """
    r = np.asarray(raw, np.float64)
    t, s, g, c = (int(i) for i in perm)
    return (0.3 + _softplus(r[t]), 0.3 + _softplus(r[s]),
            0.1 + _softplus(r[g]), _sigmoid(r[c]))


def vignetting_falloff(vig5: np.ndarray, height: int, width: int) -> np.ndarray:
    """[H,W] multiplicative falloff for one channel.

    UV normalisation is PUBLISHED: coordinates are centred then divided by the LARGER
    image dimension, so uv spans about [-0.5, 0.5] on the long axis.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    mx = float(max(width, height))
    du = (xs - width * 0.5) / mx - vig5[0]
    dv = (ys - height * 0.5) / mx - vig5[1]
    r2 = du * du + dv * dv
    fall = np.ones_like(r2)
    r2p = r2.copy()
    for alpha in vig5[2:]:
        fall = fall + alpha * r2p
        r2p = r2p * r2
    return np.clip(fall, 0.0, 1.0)


def apply(img: np.ndarray, params: PPISPParams, cam_idx: int, frame: int,
          crf_perm: Optional[Sequence[int]] = None,
          stages: Sequence[str] = ("exposure", "vignetting", "color", "crf")) -> np.ndarray:
    """Run the PPISP over an [H,W,3] linear render. Returns [H,W,3] in [0,1].

    ``crf_perm`` selects which of the stored per-channel slots are
    (toe, shoulder, gamma, center). It is REQUIRED whenever "crf" is in ``stages`` --
    see the version-mismatch note in the module docstring. There is deliberately no
    default: a wrong guess silently produces a plausible but incorrect image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected [H,W,3], got {img.shape}")
    h, w = img.shape[:2]
    view = params.view_index(cam_idx, frame)
    rgb = np.asarray(img, np.float64).copy()

    if "exposure" in stages:
        rgb = rgb * (2.0 ** params.exposure[view])

    if "vignetting" in stages:
        out = np.empty_like(rgb)
        for ch in range(3):
            out[..., ch] = rgb[..., ch] * vignetting_falloff(
                params.vignetting[cam_idx, ch], h, w)
        rgb = out

    if "color" in stages:
        hom = color_homography(params.color[view])
        intensity = rgb.sum(-1, keepdims=True)
        rgi = np.concatenate([rgb[..., 0:1], rgb[..., 1:2], intensity], -1) @ hom.T
        rgi = rgi * (intensity / (rgi[..., 2:3] + 1e-5))
        r_out, g_out = rgi[..., 0], rgi[..., 1]
        rgb = np.stack([r_out, g_out, rgi[..., 2] - r_out - g_out], -1)

    if "crf" in stages:
        if crf_perm is None:
            raise ValueError(
                "crf_perm is required: this scene stores "
                f"{params.crf_params_per_channel} CRF params per channel but the "
                f"published layout has {PUBLISHED_CRF_PARAMS_PER_CHANNEL}, so the slot "
                "assignment is not known. Pass one explicitly or drop 'crf' from stages."
            )
        out = np.empty_like(rgb)
        for ch in range(3):
            toe, shoulder, gamma, center = decode_crf(params.crf[cam_idx, ch], crf_perm)
            out[..., ch] = crf_curve(rgb[..., ch], toe, shoulder, gamma, center)
        rgb = out

    return np.clip(rgb, 0.0, 1.0)
