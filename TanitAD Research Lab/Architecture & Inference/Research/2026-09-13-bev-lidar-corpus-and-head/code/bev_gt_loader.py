#!/usr/bin/env python3
"""P3 - the loader for `<sha12>.bevgt.npz` (P2), and the checks that make it trustworthy.

⛔ LABEL ONLY. LiDAR-derived targets; inference is VISION-ONLY (PI, 2026-08-03). Nothing in
an inference path may import this module.

Three layers of verification, each catching a different failure:
  1. `verify_meta`  -- the artifact's own `meta_json` against the builder's specs
     (`p2_build_corpus.SPECS`, `lidar_bev.ZBand`) AND against LITERALS. The literals exist
     because a check derived from the builder shares the builder's defects: if someone
     edits the spec, the literal side goes red on purpose.
  2. `verify_arrays` -- shapes/dtypes against the meta, content non-degenerate.
  3. `registration_check` -- ORIENTATION, from an INDEPENDENT source: the `obstacle.offline`
     boxes of the B1 EVAL join must land on `cart_occ` far more often than their MIRROR
     image does. A mirrored artifact has a perfectly valid `meta_json`; only content can
     catch it (`R-2026-09-08-wpa-mirror`, and the mirrored 09-11 BEV panel).

State encoding (one definition, used by the head and the figures):
  0 OUT_OF_FIELD   cam_vis < 128  -- the camera cannot see this cell; never scored
  1 OBSERVED_EMPTY observed, not occupied
  2 OCCUPIED       occ (default rule)
  3 OCCLUDED       not observed under the chosen rule (B default: beyond the first obstacle
                   return in that bearing AND no return in the cell)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_OF_FIELD, OBSERVED_EMPTY, OCCUPIED, OCCLUDED = 0, 1, 2, 3
CAM_VIS_IN_FIELD = 128
SCHEMA = "tanitad.lidar_bev_gt/2"

#: LITERAL expectations (never expressions over the builder).
LITERAL_SPEC = {
    "cartesian": {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5, "shape": [120, 64]},
    "polar": {"n_rng": 24, "n_az": 20, "r_max_m": 60.0, "hfov_deg": 120.0,
              "cell_deg": 6.0, "cell_rng_m": 2.5, "shape": [24, 20]},
    "polar48": {"n_rng": 48, "n_az": 40, "r_max_m": 60.0, "hfov_deg": 120.0,
                "cell_deg": 3.0, "cell_rng_m": 1.25, "shape": [48, 40]},
    "z_band_m": {"ground_max_m": 0.30, "obstacle_min_m": 0.30, "obstacle_max_m": 3.00,
                 "spread_min_m": 0.30, "spread_min_pts": 2},
    "no_label_rule_ms": 50.0,
    "frame": "rig",
}


class ArtifactSpecError(ValueError):
    pass


def verify_meta(meta: dict, builder_specs: dict | None = None, zband=None) -> None:
    """Raise ArtifactSpecError on ANY disagreement with the literals or the builder."""
    errs = []

    def eq(path, got, want, tol=1e-9):
        if isinstance(want, float) or isinstance(got, float):
            if got is None or not math.isfinite(float(got)) or abs(float(got) - float(want)) > tol:
                errs.append(f"{path}: {got!r} != {want!r}")
        elif got != want:
            errs.append(f"{path}: {got!r} != {want!r}")

    eq("schema", meta.get("schema"), SCHEMA)
    eq("frame", meta.get("frame"), LITERAL_SPEC["frame"])
    if "+y LEFT" not in str(meta.get("frame_convention", "")):
        errs.append("frame_convention does not state +y LEFT")
    if "+yaw_rate*dt" not in str(meta.get("deskew_rotation_sign", "")):
        errs.append("deskew_rotation_sign is not the P0-measured +yaw_rate*dt")
    for grid in ("cartesian", "polar", "polar48"):
        g = meta.get(grid) or {}
        for k, v in LITERAL_SPEC[grid].items():
            eq(f"{grid}.{k}", g.get(k), v)
    if "LEFT" not in str((meta.get("polar") or {}).get("col0", "")):
        errs.append("polar.col0 does not state LEFT")
    if "LEFT" not in str((meta.get("polar48") or {}).get("col0", "")):
        errs.append("polar48.col0 does not state LEFT")
    for k, v in LITERAL_SPEC["z_band_m"].items():
        eq(f"z_band_m.{k}", (meta.get("z_band_m") or {}).get(k), v)
    tg = meta.get("time_grid") or {}
    eq("time_grid.no_label_rule_ms", tg.get("no_label_rule_ms"), LITERAL_SPEC["no_label_rule_ms"])
    if "i = j + 2" not in str(tg.get("consumer_row", "")):
        errs.append("time_grid.consumer_row does not state i = j + 2")

    # builder-spec side (catches a builder/loader drift the literals would also catch,
    # but names the builder field that moved)
    if builder_specs is not None:
        c = builder_specs["cart"]
        eq("builder.cart.cell_m", (meta.get("cartesian") or {}).get("cell_m"), c.cell_m)
        eq("builder.cart.shape", (meta.get("cartesian") or {}).get("shape"), [c.n_x, c.n_y])
        for name, key in (("polar24", "polar"), ("polar48", "polar48")):
            p = builder_specs[name]
            eq(f"builder.{name}.shape", (meta.get(key) or {}).get("shape"), [p.n_rng, p.n_az])
            eq(f"builder.{name}.cell_deg", (meta.get(key) or {}).get("cell_deg"), p.cell_deg)
            eq(f"builder.{name}.cell_rng_m", (meta.get(key) or {}).get("cell_rng_m"), p.cell_rng_m)
    if zband is not None:
        from dataclasses import asdict
        for k, v in asdict(zband).items():
            eq(f"builder.z_band_m.{k}", (meta.get("z_band_m") or {}).get(k), v)
    if errs:
        raise ArtifactSpecError("; ".join(errs))


@dataclass
class BevClip:
    meta: dict
    grid: str
    occ: np.ndarray            # [T, A, B] bool
    observed: np.ndarray       # [T, A, B] bool -- the mask chosen at load (default rule B)
    cam_vis: np.ndarray        # [A, B] uint8
    label_valid: np.ndarray    # [T] bool
    t_img_us: np.ndarray       # [T] int64
    raw_frame: np.ndarray      # [T] int16

    @property
    def in_field(self) -> np.ndarray:
        return self.cam_vis >= CAM_VIS_IN_FIELD

    def state(self) -> np.ndarray:
        """[T, A, B] uint8 in {OUT_OF_FIELD, OBSERVED_EMPTY, OCCUPIED, OCCLUDED}."""
        # observed FIRST: under rule C an occupied cell behind the first hit is OCCLUDED.
        # Under rules A/B `occ` implies `observed`, so the order is a no-op there.
        s = np.where(~self.observed, OCCLUDED, np.where(self.occ, OCCUPIED, OBSERVED_EMPTY))
        s = np.where(self.in_field[None], s, OUT_OF_FIELD)
        return s.astype(np.uint8)

    def scored_mask(self) -> np.ndarray:
        """[T, A, B] bool: cells a vision head is scored on -- valid instant, in the camera
        field, and observed (occupied or observed-empty). OCCLUDED/OUT_OF_FIELD excluded."""
        # NOT `observed | occ`: that would re-admit occupied-behind-first-hit cells under
        # rule C and silently turn it back into the MEASURED-biased rule A.
        return self.label_valid[:, None, None] & self.in_field[None] & self.observed

    def rows_to_frames(self, stacked_rows: np.ndarray, n_stack: int = 3) -> np.ndarray:
        """Trunk STACKED ROW j -> raw frame i = j + (n_stack - 1): the current frame of the
        D-015 stack (`comma2k19.stack_frames`: newest frame in the LAST 3 channels)."""
        return np.asarray(stacked_rows) + (n_stack - 1)


OBSERVED_RULES = {
    "B": "observed",                 # consumer default: ~shadow | occ | n_pts>0 (lidar_bev cart rule)
    "A": "observed_shipped",         # lidar_bev.rasterise_polar as shipped -- MEASURED biased
    "C": "visible_first_hit",        # ~shadow only
}


def load_clip(path: str | Path, grid: str = "polar48", verify: bool = True,
              builder_specs: dict | None = None, zband=None, rule: str = "B") -> BevClip:
    with np.load(path) as z:
        meta = json.loads(str(z["meta_json"]))
        if verify:
            verify_meta(meta, builder_specs, zband)
        occ = z[f"{grid}_occ"]
        if grid == "cart" and rule != "B":
            raise ValueError("the Cartesian grid carries rule B only")
        observed = z[f"{grid}_{OBSERVED_RULES[rule]}"]
        cam_vis = z[f"{grid}_cam_vis"]
        lv = z["label_valid"]
        clip = BevClip(meta=meta, grid=grid, occ=occ.astype(bool), observed=observed.astype(bool),
                       cam_vis=cam_vis, label_valid=lv.astype(bool), t_img_us=z["t_img_us"],
                       raw_frame=z["raw_frame"])
    if verify:
        verify_arrays(clip)
    return clip


def verify_arrays(clip: BevClip) -> None:
    key = {"polar48": "polar48", "polar24": "polar", "cart": "cartesian"}[clip.grid]
    shp = tuple(clip.meta[key]["shape"])
    T = clip.label_valid.shape[0]
    errs = []
    if clip.occ.shape != (T, *shp):
        errs.append(f"occ shape {clip.occ.shape} != {(T, *shp)}")
    if clip.observed.shape != (T, *shp):
        errs.append(f"observed shape {clip.observed.shape} != {(T, *shp)}")
    if clip.cam_vis.shape != shp or clip.cam_vis.dtype != np.uint8:
        errs.append(f"cam_vis {clip.cam_vis.shape}/{clip.cam_vis.dtype}")
    if not np.array_equal(clip.raw_frame, np.arange(T)):
        errs.append("raw_frame is not 0..T-1")
    if not clip.occ[clip.label_valid].any():
        errs.append("occ ALL ZERO on valid frames")
    if not clip.observed[clip.label_valid].any():
        errs.append("observed ALL ZERO on valid frames")
    if clip.occ[~clip.label_valid].any():
        errs.append("occ non-zero on a NO_LABEL frame")
    if errs:
        raise ArtifactSpecError("; ".join(errs))


# ---------------------------------------------------------------------------
# orientation, from an independent source
# ---------------------------------------------------------------------------
def box_cells_cart(agents, x_max=60.0, y_half=16.0, cell=0.5, mirror=False) -> np.ndarray:
    """[120, 64] bool: cell CENTRES inside an oriented footprint (bev_raster's predicate).
    Written independently of the rasteriser: only the grid literals are shared."""
    nx, ny = int(round(x_max / cell)), int(round(2 * y_half / cell))
    gx = (np.arange(nx) + 0.5) * cell
    gy = (np.arange(ny) + 0.5) * cell - y_half
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    out = np.zeros((nx, ny), dtype=bool)
    for a in agents:
        cx, cy, yaw = float(a["cx"]), float(a["cy"]), float(a["yaw"])
        if mirror:
            cy, yaw = -cy, -yaw
        l, w = float(a["l"]), float(a["w"])
        c, s = math.cos(-yaw), math.sin(-yaw)
        ex, ey = GX - cx, GY - cy
        u = ex * c - ey * s
        v = ex * s + ey * c
        out |= (np.abs(u) <= l / 2.0) & (np.abs(v) <= w / 2.0)
    return out


def registration_check(npz_path: str | Path, join_rows: list[tuple[float, list]],
                       max_join_dt_ms: float = 60.0) -> dict:
    """Pooled hit rates of real vs mirrored `obstacle.offline` boxes on `cart_occ`,
    observed cells only. `join_rows` = [(t_s, agents)] for this clip."""
    with np.load(npz_path) as z:
        occ = z["cart_occ"].astype(bool)
        obs = z["cart_observed"].astype(bool)
        lv = z["label_valid"].astype(bool)
        t_img = z["t_img_us"].astype(np.float64) * 1e-6
    jt = np.asarray([r[0] for r in join_rows], dtype=np.float64)
    hit = {"real": [0, 0], "mirror": [0, 0]}
    base = [0, 0]
    for i in np.nonzero(lv)[0]:
        if jt.size == 0:
            break
        j = int(np.abs(jt - t_img[i]).argmin())
        if abs(jt[j] - t_img[i]) * 1e3 > max_join_dt_ms:
            continue
        o, ob = occ[i], obs[i]
        base[0] += int(o[ob].sum())
        base[1] += int(ob.sum())
        for name, mir in (("real", False), ("mirror", True)):
            bc = box_cells_cart(join_rows[j][1], mirror=mir) & ob
            hit[name][0] += int(o[bc].sum())
            hit[name][1] += int(bc.sum())
    marg = base[0] / base[1] if base[1] else float("nan")
    res = {"marginal": marg, "n_observed_cells": base[1]}
    for k, (h, n) in hit.items():
        res[f"{k}_hit_rate"] = h / n if n else float("nan")
        res[f"{k}_n_cells"] = n
    res["real_over_mirror"] = (res["real_hit_rate"] / res["mirror_hit_rate"]
                               if res["mirror_hit_rate"] and res["mirror_hit_rate"] > 0 else float("inf"))
    res["real_over_marginal"] = res["real_hit_rate"] / marg if marg and marg > 0 else float("nan")
    return res
