"""IDM v3 — the camera-geometry substrate the IDM has never been told about.

MEASURED 2026-07-27 from the PhysicalAI-AV dataset's OWN gated calibration
features (`calibration/camera_intrinsics`, `calibration/sensor_extrinsics`),
joined to our 40 eval-pod val episodes through the build's ordered clip list.
The join is VERIFIED, not assumed: ep_id = int.from_bytes(clip_id[:4],"big")
reproduced **40/40** eval-pod `episode_id`s from `val_clip_order.tsv[:40]`.

GATED-CONFIDENTIAL: PhysicalAI-AV clip UUIDs are NOT in this file. The table is
keyed by episode INDEX only, and carries per-clip camera geometry — six scalars
per clip, from which no frame, pose or clip identity can be reconstructed.

--------------------------------------------------------------------------- #
WHY THIS FILE EXISTS — the physics
--------------------------------------------------------------------------- #
For a camera of effective focal `f` at height `h` above the road plane, a ground
point at forward distance X projects `dv = f*h/X` px below the horizon. Ego
motion at speed v gives d(dv)/dt = dv^2 * v / (f*h), so

        v = (f * h) * PHI(image motion)                                  (1)

where PHI is a purely image-domain quantity. **Metric speed is LINEAR in the
product f*h.** A network shown only pixels, never told f or h, cannot resolve
that factor and must emit one blended mapping — which is exactly the measured
failure (56.6 % of speed MSE is a per-clip level bias, gain 0.830).

Rotation behaves DIFFERENTLY. A yaw rate w displaces the image by du ~ f*w*dt,
so

        w = (du/dt) / f                                                  (2)

**yaw_rate depends on f ONLY and NOT on h.** Our pipeline already canonicalises
f_eff to ~266 px on every corpus (rig-A 266.13 / rig-B 266.10 / comma 266.50,
`…/2026-07-22-idm-proof/results_regate.json`), so the yaw channel is ALREADY
geometry-matched while the speed channel is NOT. That asymmetry is the
pre-registered discriminator in `PRE_REGISTRATION_IDMV3.md`: geometry
conditioning must help SPEED and must NOT help YAW. A method that helps both
equally is memorising the corpus, not using the geometry.

--------------------------------------------------------------------------- #
WHAT WAS MEASURED (n = 40 PhysicalAI val clips)
--------------------------------------------------------------------------- #
  cam height   1.2450 .. 1.6066 m   (mean 1.3417, std 0.0991, CV 7.4 %,
                                     full range 29.0 %, 37 distinct values/40)
  cy           535.0 .. 764.5 px    (two clusters -> the two rigs)
  rig split    14 rig-A (cy~542) / 26 rig-B (cy~754)
  f_paraxial   920.1 .. 938.5 px @ native 1920x1080  (2.0 % spread)
  pitch_down   -1.15 .. +2.34 deg
  cam fwd x    1.797 .. 2.140 m

>> THE THREE CIRCULATING `cam_h` VALUES ARE ALL WRONG AS A CONSTANT. <<
   1.22 m is BELOW the observed minimum (1.245); 1.43 m sits at the 93rd
   percentile; 1.5 m is exceeded by only 5 of 40 clips. The measured median is
   **1.306 m** and the quantity is **per-clip**, not a constant.

>> RIG IDENTITY IS NOT A PROXY FOR CAMERA HEIGHT. <<
   rig-A median 1.3127 m, rig-B median 1.2931 m — a 1.5 % difference, while the
   WITHIN-rig spread is 29 %. Conditioning on a rig label therefore cannot
   substitute for conditioning on the height, and the prior sibling result that
   tested a row-profile rig PROXY does not bear on this.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# comma2k19 — a single fixed camera model (comma EON, 1164x874, focal 910 px)  #
# --------------------------------------------------------------------------- #
# f_eff: MEASURED (results_regate.json, 266.50).
# cam_h: **INHERITED and UNVERIFIED** — 1.22 m is the repo's standing constant
#        (`tanitad/replay/rr_log.py:93`). comma2k19 ships no per-segment mount
#        height, so this is the one geometry number in this work that is not
#        measured from data. Arm C_shuf (shuffled geometry) exists partly to
#        show the result does not hinge on its exact value.
COMMA_GEOM = {
    "cam_h_m": 1.22,
    "pitch_down_rad": 0.0,
    "cy_native": 437.0,          # 874/2, geometric centre (comma crop is centred)
    "cx_native": 582.0,
    "f_paraxial_native_px": 910.0,
    "f_eff_px": 266.5,
    "cam_fwd_m": 1.5,
    "rig": "cm",
    "domain": "cm",
}

_HERE = Path(__file__).resolve().parent
_TABLE = _HERE / "pai_geom_table.json"

# Order of the geometry feature vector handed to the head. Keep it stable —
# checkpoints carry it in their config.
GEOM_FEATURES = (
    "f_eff_px",            # (2): sets the yaw scale.  ~266 everywhere by design
    "cam_h_m",             # (1): sets the metric scale.  1.245-1.607 m, per clip
    "metric_gain",         # f_eff * cam_h  — the physical scale factor of eq. (1)
    "pitch_down_rad",      # locates the horizon row
    "cy_norm",             # (cy - H/2)/H — the rig signature
    "cam_fwd_m",           # camera longitudinal offset from the vehicle origin
)


def load_table() -> dict:
    """ep tag -> raw geometry dict, for BOTH corpora, keyed by idm2 tag."""
    pai = json.loads(_TABLE.read_text())
    out = {}
    for ep, g in pai.items():                    # ep = "ep_00000" -> tag "pai_00000"
        out["pai_" + ep.split("_")[1]] = dict(g)
    return out


def geom_for_tag(tag: str, table: dict) -> dict:
    if tag.startswith("cm_"):
        return dict(COMMA_GEOM)
    g = table.get(tag)
    if g is None:
        raise KeyError(f"no geometry for {tag!r}")
    return g


def feature_vector(g: dict) -> np.ndarray:
    """Raw geometry dict -> the GEOM_FEATURES vector (unnormalised)."""
    H = 1080.0 if g["domain"] == "pai" else 874.0
    return np.array([
        g["f_eff_px"],
        g["cam_h_m"],
        g["f_eff_px"] * g["cam_h_m"],
        g["pitch_down_rad"],
        (g["cy_native"] - H / 2.0) / H,
        g["cam_fwd_m"],
    ], dtype=np.float64)


def rig_onehot(g: dict) -> np.ndarray:
    """3-dim CONTROL: pai-rigA / pai-rigB / comma. Discrete identity only."""
    r = g["rig"]
    return np.array([float(r == "a"), float(r == "b"), float(r == "cm")])


def corpus_onehot(g: dict) -> np.ndarray:
    """2-dim CONTROL: which dataset. The 'knows the corpus' null hypothesis."""
    return np.array([float(g["domain"] == "pai"), float(g["domain"] == "cm")])


def build_matrices(tags):
    """tags -> (G [n,6] geometry, R [n,3] rig one-hot, C [n,2] corpus one-hot)."""
    tab = load_table()
    gs = [geom_for_tag(t, tab) for t in tags]
    return (np.stack([feature_vector(g) for g in gs]),
            np.stack([rig_onehot(g) for g in gs]),
            np.stack([corpus_onehot(g) for g in gs]))
