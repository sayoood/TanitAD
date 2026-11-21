"""P1 step 4 - THE DISCRIMINATING TEST between two explanations of the
ratio-1.01 coupling found in step 3.

  H_echo : the "limit" is the EGO SPEEDOMETER read off the dashboard at the
           anchor frame (C87's dashboard-roundel failure mode).
  H_real : the "limit" is a genuine posted sign, and it tracks ego speed only
           because drivers obey limits.

DISCRIMINATOR - TEMPORAL SPECIFICITY. A speedometer echo is a reading of ONE
FRAME: |limit - ego(t)| must be MINIMISED at t = the Alpamayo anchor. A real
posted limit applies to the whole road stretch and has no reason to be
sharpest at the anchor.

Also reports the QUANTISATION check: real limits are legal round numbers.
ASCII-only output.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data import egomotion_source as ES  # noqa: E402

KMH, MPH = 1 / 3.6, 0.44704
ALPA_T0 = 5.1
rows = json.load(open(sys.argv[1], encoding="utf-8"))["rows"]

# sweep the anchor over the whole clip
grid = np.arange(0.0, 19.01, 0.5)
err = np.full((len(rows), len(grid)), np.nan)

for i, r in enumerate(rows):
    tr = ES.load(r["clip"])
    f = MPH if r["unit"] == "mph" else KMH
    lim = r["value"] * f
    v = tr.poses[:, 3]
    for j, t in enumerate(grid):
        k = int(round(t * 10.0))
        if k < v.shape[0]:
            err[i, j] = abs(lim - float(v[k]))

med = np.nanmedian(err, axis=0)
print("TEMPORAL SPECIFICITY - median |limit - ego(t)| over %d clips" % len(rows))
print("  (H_echo predicts a MINIMUM at t = %.1f s, the Alpamayo anchor)" % ALPA_T0)
print()
print("     t_s   median|limit-ego|  n")
for j, t in enumerate(grid):
    n = int(np.isfinite(err[:, j]).sum())
    mark = "  <== ALPAMAYO ANCHOR" if abs(t - ALPA_T0) < 0.26 else ""
    bar = "#" * int(round(med[j] * 3))
    print("   %5.1f   %8.2f  %4d  %-28s%s" % (t, med[j], n, bar, mark))

jbest = int(np.nanargmin(med))
print()
print("ARGMIN of the median error is at t = %.1f s (value %.2f m/s)"
      % (grid[jbest], med[jbest]))
janch = int(np.argmin(np.abs(grid - ALPA_T0)))
print("Median error AT the anchor  t=%.1f s : %.2f m/s" % (grid[janch], med[janch]))
print("Median error at the clip minimum      : %.2f m/s" % med[jbest])
print()
if abs(grid[jbest] - ALPA_T0) <= 1.0:
    print(">> The minimum COINCIDES with the anchor -> consistent with H_echo.")
else:
    print(">> The minimum does NOT coincide with the anchor (%.1f s away)."
          % abs(grid[jbest] - ALPA_T0))
    print(">> H_echo predicted a sharp minimum at the anchor and does not get one.")
    print(">> The flat profile is what H_real predicts: a limit that applies to")
    print(">> the whole stretch, with ego speed near it throughout.")
print()

# spread of the profile - an echo is SHARP, a road-wide limit is FLAT
rng = float(np.nanmax(med) - np.nanmin(med))
print("Profile range over the clip (max-min of the median): %.2f m/s" % rng)
print("Profile value at t=0.0 / anchor / t=15.0 : %.2f / %.2f / %.2f"
      % (med[0], med[janch], med[int(np.argmin(np.abs(grid - 15.0)))]))
print()

# quantisation
vals = [r["value"] for r in rows]
LEGAL = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 130}
n_legal = sum(1 for v in vals if v in LEGAL)
print("QUANTISATION: %d/%d read values are legal posted-limit round numbers"
      % (n_legal, len(vals)))
print("  distinct values: %s" % sorted(set(vals)))

json.dump({"grid_s": grid.tolist(), "median_abs_err": med.tolist(),
           "argmin_t": float(grid[jbest]), "anchor_t": ALPA_T0,
           "err_at_anchor": float(med[janch]), "err_at_argmin": float(med[jbest]),
           "profile_range": rng, "n_legal_values": n_legal, "n": len(vals)},
          open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print()
print("WROTE %s" % sys.argv[2])
