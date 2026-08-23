#!/usr/bin/env python3
"""bench_geometry_check.py -- is a community benchmark's camera rig ABLE to feed our models?

The question this exists to answer, exactly
-------------------------------------------
Every closed-loop benchmark hands a policy a *rig*: N cameras, each with a projection
model, a resolution and a mounting yaw.  Our two candidate arms want something else:

  * ``v5f``        -- a **120 deg cylindrical** raster, 256x640, of which the model sees
                      the 176x624 sub-frame.   (MODEL_REGISTRY-adjacent source:
                      ``Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md`` line 22, MEASURED
                      from the run's own config.json.)
  * ``v1/REF-C``   -- a 256x256 **square** crop canonicalised to a reference focal
                      ``F_REF = 266.0`` px (MEASURED live in the AlpaSim driver adapter:
                      ``f_eff = 265.6..266.014 == F_REF``, ALPASIM_STATE.md row 6 and
                      ``Videos/alpasim-openloop-thor-2026-08-03/README.md``).

"Can we run on benchmark X without retraining?" then decomposes into exactly three
geometric tests, and this module computes all three:

  T1 AZIMUTH COVERAGE   does the union of the rig's camera frusta contain the target's
                        azimuth span at all?  If not: INFEASIBLE without a retrain --
                        no amount of resampling invents pixels that were never captured.

  T2 ANGULAR SAMPLING   at the target's *worst* azimuth, does the source supply at least
                        as many pixels per radian as the target consumes?  For a PINHOLE
                        camera d(x)/d(theta) = f*sec^2(theta) >= f, so the *minimum* is at
                        the optical axis and the whole test collapses to
                            f_src_px  >=  W_target / HFOV_target_rad .
                        Above that ratio the resample is a DOWN-sample (safe, band-limit
                        then decimate); below it we are inventing detail (blur, and the
                        model sees a distribution it never trained on).

  T3 PARALLAX / SEAM    a cylinder is a single-viewpoint projection.  Stitching it from
                        k >= 2 cameras with DIFFERENT optical centres is exact only at
                        infinity; at a range r the seam error is
                            eps_px  ~=  (b / r) * (W_target / HFOV_target_rad)
                        for a baseline b.  This is the term that decides whether a
                        "just re-project the front trio" plan is a resample or a hack.
                        k == 1, or k co-located cameras (b == 0), makes it vanish.

Nothing here is a claim about *domain* (exposure, ISP, weather, city, sensor noise).
Geometry feasible != distribution matched.  That is stated in the accompanying note and
is deliberately NOT modelled here -- this file answers the geometry question only.

Run
---
    python bench_geometry_check.py                 # table + writes bench_geometry.json
    python bench_geometry_check.py --selftest      # closed-form checks

Every rig entry carries an ``evidence`` field.  PUBLISHED entries cite where the number
came from; ESTIMATED entries are flagged, and the verdicts they drive are marked in the
output so no reader can mistake one for a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------------------
# targets -- what OUR models consume
# --------------------------------------------------------------------------------------


@dataclass
class Target:
    name: str
    kind: str            # "cylindrical" | "pinhole_square"
    width_px: int
    height_px: int
    hfov_deg: float
    note: str
    evidence: str

    @property
    def px_per_rad(self) -> float:
        """Pixels the target consumes per radian of azimuth, at its worst azimuth.

        A cylindrical raster is uniform in azimuth, so this is exact everywhere.
        A pinhole target is *densest* at its edge and sparsest on axis, so the on-axis
        value (= the focal length) is again the binding one.
        """
        if self.kind == "cylindrical":
            return self.width_px / math.radians(self.hfov_deg)
        return (self.width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)


TARGETS = [
    Target(
        name="v5f",
        kind="cylindrical",
        width_px=640,
        height_px=256,
        hfov_deg=120.0,
        note="model sees the 176x624 sub-frame of this raster",
        evidence="MEASURED -- Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md:22 (from the run's config.json)",
    ),
    Target(
        name="v1_refc_256sq",
        kind="pinhole_square",
        width_px=256,
        height_px=256,
        hfov_deg=2.0 * math.degrees(math.atan2(128.0, 266.0)),  # from F_REF, not asserted
        note="square canon at F_REF=266.0 px; hfov is DERIVED from F_REF, not quoted",
        evidence="MEASURED -- live f_eff 265.6-266.014 == F_REF in refc_driver.py (ALPASIM_STATE.md row 6)",
    ),
]


# --------------------------------------------------------------------------------------
# sources -- what a benchmark hands you
# --------------------------------------------------------------------------------------


@dataclass
class Cam:
    name: str
    width_px: int
    height_px: int
    hfov_deg: float
    yaw_deg: float = 0.0
    model: str = "pinhole"          # "pinhole" | "ftheta"
    baseline_m: float = 0.0         # lateral offset of this camera's optical centre

    @property
    def f_px(self) -> float:
        if self.model == "ftheta":
            # equidistant: r = f * theta  ->  uniform px/rad = f = (W/2)/(hfov/2)
            return (self.width_px / 2.0) / (math.radians(self.hfov_deg) / 2.0)
        return (self.width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def min_px_per_rad(self) -> float:
        """Worst-case angular sampling density inside this camera's frustum."""
        # pinhole: min at the optical axis, = f.  ftheta/equidistant: uniform, = f.
        return self.f_px

    def span_deg(self) -> tuple[float, float]:
        return (self.yaw_deg - self.hfov_deg / 2.0, self.yaw_deg + self.hfov_deg / 2.0)


@dataclass
class Rig:
    name: str
    cams: list[Cam]
    colocated: bool
    evidence: str
    note: str = ""
    extra: dict = field(default_factory=dict)


def _rigs() -> list[Rig]:
    return [
        # ---- our own training rig, and the AlpaSim/NuRec reconstruction rig -------------
        Rig(
            name="physicalai_av_nurec",
            cams=[Cam("front_wide_120fov", 1920, 1216, 120.0, 0.0, model="ftheta")],
            colocated=True,
            evidence=(
                "MEASURED (ours) -- the rig we render today on tanitad-thor; 1920-px raster cited in "
                "stack/experiments/alpasim-gsplat/results/2026-08-03-rolling-shutter/ROLLING_SHUTTER.md. "
                "PUBLISHED -- NuRec scenes are reconstructed from 6 views incl. front-wide 120 deg "
                "(huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)."
            ),
            note="the ONLY rig in this table that is our own training rig; renderer under our control",
            extra={"other_training_views": ["front_tele_30", "cross_left_120", "cross_right_120",
                                            "rear_left_70", "rear_right_70"]},
        ),
        # ---- NAVSIM / nuPlan (OpenScene) ----------------------------------------------
        Rig(
            name="navsim_nuplan_8cam",
            cams=[
                Cam("CAM_F0", 1920, 1080, 64.0, 0.0, baseline_m=0.0),
                Cam("CAM_L0", 1920, 1080, 64.0, -55.0, baseline_m=0.35),
                Cam("CAM_R0", 1920, 1080, 64.0, +55.0, baseline_m=0.35),
            ],
            colocated=False,
            evidence=(
                "PUBLISHED (count/resolution) -- NAVSIM agent input is 8 cameras at 1920x1080 "
                "(arxiv 2406.15349). ESTIMATED (per-camera HFOV ~64 deg and the +-55 deg yaws, and the "
                "0.35 m baseline) -- NOT read from a calibration file. Any verdict that turns on these "
                "is marked estimated_inputs=True."
            ),
            note="log replay; optical centres are physically apart -> the cylinder stitch has parallax",
        ),
        # ---- nuScenes-style rig (NeuroNCAP / HUGSIM host data) -------------------------
        Rig(
            name="nuscenes_front_trio",
            cams=[
                Cam("CAM_FRONT", 1600, 900, 70.0, 0.0, baseline_m=0.0),
                Cam("CAM_FRONT_LEFT", 1600, 900, 70.0, -55.0, baseline_m=0.30),
                Cam("CAM_FRONT_RIGHT", 1600, 900, 70.0, +55.0, baseline_m=0.30),
            ],
            colocated=False,
            evidence="PUBLISHED -- nuScenes 6x 1600x900, front 70 deg HFOV. ESTIMATED -- yaws/baselines.",
            note="the rig NeuroNCAP and the nuScenes half of HUGSIM inherit",
        ),
        # ---- CARLA (Leaderboard 2.0 / Bench2Drive) -------------------------------------
        Rig(
            name="carla_declared_trio_colocated",
            cams=[
                Cam("cam_l", 800, 600, 60.0, -60.0, baseline_m=0.0),
                Cam("cam_c", 800, 600, 60.0, 0.0, baseline_m=0.0),
                Cam("cam_r", 800, 600, 60.0, +60.0, baseline_m=0.0),
            ],
            colocated=True,
            evidence=(
                "PUBLISHED -- the agent DECLARES its sensors (width/height/fov/xyz/rpy); Leaderboard 2.0 "
                "permits up to 8 RGB cameras (leaderboard.carla.org/get_started_v2_0). Co-location at one "
                "(x,y,z) is legal, which is what makes the stitch exact."
            ),
            note="we choose the rig -> baseline can be driven to zero -> exact cylindrical stitch",
        ),
    ]


# --------------------------------------------------------------------------------------
# the three tests
# --------------------------------------------------------------------------------------


def coverage(rig: Rig, target: Target) -> dict:
    """T1 -- union azimuth coverage of the target span, and how many cams it takes."""
    lo, hi = -target.hfov_deg / 2.0, target.hfov_deg / 2.0
    step = 0.25
    n = int(round((hi - lo) / step)) + 1
    covered, used = 0, set()
    worst_pxrad, worst_az = float("inf"), None
    for i in range(n):
        az = lo + i * step
        best = None
        for c in rig.cams:
            a, b = c.span_deg()
            if a <= az <= b and (best is None or c.min_px_per_rad > best.min_px_per_rad):
                best = c
        if best is not None:
            covered += 1
            used.add(best.name)
            if best.min_px_per_rad < worst_pxrad:
                worst_pxrad, worst_az = best.min_px_per_rad, az
    return {
        "target_span_deg": [lo, hi],
        "covered_frac": covered / n,
        "n_cams_needed": len(used),
        "cams_used": sorted(used),
        "worst_px_per_rad": None if worst_pxrad == float("inf") else worst_pxrad,
        "worst_az_deg": worst_az,
    }


def seam_error_px(rig: Rig, target: Target, ranges_m=(5.0, 15.0, 50.0)) -> dict:
    """T3 -- cylindrical stitch parallax, in TARGET pixels, at a few ranges."""
    b = max((c.baseline_m for c in rig.cams), default=0.0)
    if rig.colocated:
        b = 0.0
    ppr = target.px_per_rad
    return {"baseline_m": b, "px_err": {f"{r:g}m": round(b / r * ppr, 2) for r in ranges_m}}


def verdict(cov: dict, ratio: float | None, seam: dict) -> tuple[str, str]:
    if cov["covered_frac"] < 0.999:
        return ("INFEASIBLE_COVERAGE",
                f"only {cov['covered_frac']*100:.1f}% of the target azimuth span is captured at all")
    if ratio is None:
        return ("INFEASIBLE_COVERAGE", "no camera covers the span")
    worst_seam = max(seam["px_err"].values())
    if ratio < 1.0:
        return ("UNDERSAMPLED",
                f"source supplies {ratio:.2f}x the needed px/rad on axis -- the resample invents detail")
    if worst_seam > 8.0:
        return ("FEASIBLE_WITH_SEAM",
                f"resample is a down-sample ({ratio:.2f}x) but the stitch is off by {worst_seam:.1f} px "
                f"at 5 m -- a near-field seam the model never saw in training")
    return ("FEASIBLE_RESAMPLE",
            f"down-sample by {ratio:.2f}x, stitch error <= {worst_seam:.1f} px at 5 m")


def evaluate() -> dict:
    out = {"targets": [], "rigs": [], "matrix": []}
    for t in TARGETS:
        d = asdict(t)
        d["px_per_rad"] = round(t.px_per_rad, 2)
        d["deg_per_px"] = round(1.0 / (t.px_per_rad * math.pi / 180.0), 5)
        out["targets"].append(d)
    for rig in _rigs():
        out["rigs"].append({
            "name": rig.name, "colocated": rig.colocated, "evidence": rig.evidence,
            "note": rig.note, "extra": rig.extra,
            "cams": [{**asdict(c), "f_px": round(c.f_px, 1),
                      "px_per_rad_min": round(c.min_px_per_rad, 1)} for c in rig.cams],
        })
        for t in TARGETS:
            cov = coverage(rig, t)
            ratio = (cov["worst_px_per_rad"] / t.px_per_rad) if cov["worst_px_per_rad"] else None
            seam = seam_error_px(rig, t)
            v, why = verdict(cov, ratio, seam)
            out["matrix"].append({
                "rig": rig.name, "target": t.name, "verdict": v, "why": why,
                "sampling_ratio_src_over_tgt": None if ratio is None else round(ratio, 3),
                "n_cams_needed": cov["n_cams_needed"], "cams_used": cov["cams_used"],
                "covered_frac": round(cov["covered_frac"], 4),
                "stitch_px_err": seam["px_err"],
                "estimated_inputs": ("ESTIMATED" in rig.evidence),
            })
    return out


# --------------------------------------------------------------------------------------


def selftest() -> None:
    # 1. cylinder px/rad is exactly W / hfov_rad
    t = TARGETS[0]
    assert abs(t.px_per_rad - 640 / math.radians(120)) < 1e-9
    assert abs(t.px_per_rad - 305.577) < 1e-2, t.px_per_rad
    # 2. the square target's derived HFOV round-trips through F_REF
    sq = TARGETS[1]
    assert abs(sq.px_per_rad - 266.0) < 1e-6, sq.px_per_rad
    # 3. a pinhole's angular density is minimal ON AXIS (the claim T2 rests on)
    c = Cam("t", 1000, 1000, 90.0)
    f = c.f_px
    on_axis = f
    at_edge = f * (1 + math.tan(math.radians(45.0)) ** 2)
    assert at_edge > on_axis and abs(at_edge - 2 * f) < 1e-6
    # 4. an equidistant f-theta camera is uniform, and 120 deg over 1920 px gives
    #    exactly 1920/(2*pi/3) px/rad
    ft = Cam("ft", 1920, 1216, 120.0, model="ftheta")
    assert abs(ft.f_px - 1920 / math.radians(120)) < 1e-9
    # 5. a co-located rig has zero seam error by construction
    rig = [r for r in _rigs() if r.name == "carla_declared_trio_colocated"][0]
    assert max(seam_error_px(rig, t)["px_err"].values()) == 0.0
    # 6. coverage is a strict gate: a single 70 deg camera cannot cover a 120 deg target
    solo = Rig("solo", [Cam("f", 1600, 900, 70.0)], True, "selftest")
    assert coverage(solo, t)["covered_frac"] < 1.0
    print("selftest OK -- 6/6")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("bench_geometry.json")))
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = evaluate()
    print(f"{'rig':32s} {'target':16s} {'verdict':22s} {'srat':>6s} {'cams':>5s} {'seam@5m':>8s}")
    print("-" * 96)
    for m in res["matrix"]:
        srat = "-" if m["sampling_ratio_src_over_tgt"] is None else f"{m['sampling_ratio_src_over_tgt']:.2f}"
        print(f"{m['rig']:32s} {m['target']:16s} {m['verdict']:22s} {srat:>6s} "
              f"{m['n_cams_needed']:>5d} {m['stitch_px_err']['5m']:>8.1f}"
              + ("   [estimated inputs]" if m["estimated_inputs"] else ""))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
