# -*- coding: utf-8 -*-
"""WP-A -- the READOUT AZIMUTH GEOMETRY, derived TWICE by different routes.

⭐ A cross-check must be derived INDEPENDENTLY of the value it checks
(CLAUDE.md).  Route 1 is `bev_raster.fov_census`, the programme's own
instrument.  Route 2 is an ANALYTIC derivation from the cylindrical projection
(column linear in azimuth at 256x640, f_ref 305.577, true FOV 120 deg) and the
BEV grid spec -- written as literals, not as an expression over route 1.
"""
import json
import math
import sys

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")

import numpy as np
from tanitad.data.bev_raster import GRID_DEFAULT, fov_census, fov_mask

HFOV_DEG = 120.0            # the rig's own name: camera_front_wide_120fov
HALF_RAD = math.radians(HFOV_DEG / 2.0)   # ⚠ fov_mask/fov_census take the HALF angle
W_PX, F_REF = 640, 305.5774907364391
# route-2 independent check of the FOV itself: cylindrical column is LINEAR in
# azimuth, az_max = (W/2)/f_ref  (the pinhole 2*atan((W/2)/f) = 92.6 deg is WRONG here)
az_cyl = 2.0 * math.degrees((W_PX / 2.0) / F_REF)
az_pin = 2.0 * math.degrees(math.atan((W_PX / 2.0) / F_REF))
print(f"[projection] cylindrical FOV {az_cyl:.3f} deg   "
      f"(pinhole formula would say {az_pin:.3f} deg -- WRONG on this corpus)")
assert abs(az_cyl - HFOV_DEG) < 0.2, az_cyl

out = {"_evidence_class": "MEASURED (ours; artifact = this file)",
       "projection": {"cyl_fov_deg": az_cyl, "pinhole_would_say_deg": az_pin,
                      "w_px": W_PX, "f_ref": F_REF},
       "grid": {"shape": list(GRID_DEFAULT.shape), "cell_m": GRID_DEFAULT.cell_m,
                "x_fwd_m": GRID_DEFAULT.x_fwd_m, "y_half_m": GRID_DEFAULT.y_half_m},
       "ladder": {}}

m = fov_mask(GRID_DEFAULT, HALF_RAD)
print(f"[fov_mask] in-field {int(m.sum())}/{m.size} ({100*m.mean():.2f} %) "
      f"-- TWO-STATE: in-field / out-of-field, NEVER seen-and-empty")
out["fov_mask"] = {"in_field": int(m.sum()), "cells": int(m.size),
                   "in_field_frac": float(m.mean()),
                   "states": "TWO (in-field / out-of-field). NOT three: it "
                             "cannot say seen-and-empty."}

for ncol in (40, 20, 8, 4, 2, 1):
    cen = fov_census(GRID_DEFAULT, HALF_RAD, ncol)
    cols = cen["columns"] if isinstance(cen, dict) and "columns" in cen else cen
    # route 2: analytic azimuth wedge -> lateral extent at range R
    dpc = HFOV_DEG / ncol
    lat = {f"{R:g}m": 2.0 * R * math.tan(math.radians(dpc / 2.0))
           for R in (10, 20, 30, 50)}
    out["ladder"][f"{ncol}col"] = {"deg_per_col": dpc,
                                   "lat_full_width_m": lat,
                                   "lat_half_width_at_30m": lat["30m"] / 2.0,
                                   "fov_census": cols}
    print(f"[{ncol:2d} cols] {dpc:5.2f} deg/col  lateral width @30 m "
          f"{lat['30m']:6.3f} m (half {lat['30m']/2:5.3f} m)  @50 m "
          f"{lat['50m']:6.3f} m")

json.dump(out, open(r"C:\Users\Admin\wpa-readout\raw\geometry.json", "w",
                    encoding="utf-8"), indent=1, default=str)
print("WROTE geometry.json")
print(json.dumps(out["ladder"]["4col"]["fov_census"], indent=1, default=str)[:900])
