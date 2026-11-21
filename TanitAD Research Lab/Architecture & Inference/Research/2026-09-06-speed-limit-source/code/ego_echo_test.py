"""P1 step 3 - THE ADMISSIBILITY TEST.

C87's two highest-scoring sign false positives were a DASHBOARD `30` ROUNDEL
(the ego speedometer) and a hoarding. `30` is the modal value in Alpamayo's
speed-limit readings. So: is the claimed "limit" an ECHO OF THE EGO'S OWN SPEED?

Controls (a probe with no control manufactures a result):
  * CONSTANT control - a fixed 50 km/h "limit" for every clip. If the real
    readings are no closer to ego speed than this, there is no echo.
  * SHUFFLE control  - the read values permuted across clips. Breaks any
    clip-specific link while keeping the value distribution.
ASCII-only output.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

from tanitad.data import egomotion_source as ES  # noqa: E402

ALPA_T0 = 5.1
d = json.load(open(sys.argv[1], encoding="utf-8"))
read = d["read"]

rows, missing = [], 0
for cid, v in sorted(read.items()):
    try:
        tr = ES.load(cid)
    except Exception:
        missing += 1
        continue
    i = int(round(ALPA_T0 * 10.0))
    if i >= tr.poses.shape[0]:
        missing += 1
        continue
    ego = float(tr.poses[i, 3])                  # m/s at the Alpamayo anchor
    ego_max = float(tr.poses[:, 3].max())
    rows.append({"clip": cid, "value": v["value"], "unit": v["unit"],
                 "task": v["task"], "ego_ms": ego, "ego_max_ms": ego_max,
                 "text": v["text"]})

print("READ clips with ego track: %d  (missing %d)" % (len(rows), missing))
print()

KMH, MPH = 1 / 3.6, 0.44704


def stats(vals_ms, egos, tag):
    vals_ms = np.asarray(vals_ms, float)
    egos = np.asarray(egos, float)
    diff = vals_ms - egos
    ratio = np.where(egos > 1.0, vals_ms / np.maximum(egos, 1e-6), np.nan)
    print("  %-26s n=%3d  median|limit-ego| = %6.2f m/s   "
          "median ratio limit/ego = %s   frac(limit<ego) = %.3f"
          % (tag, len(vals_ms), float(np.median(np.abs(diff))),
             ("%5.2f" % np.nanmedian(ratio)) if np.isfinite(ratio).any() else "  n/a",
             float(np.mean(vals_ms < egos))))
    return float(np.median(np.abs(diff)))


egos = [r["ego_ms"] for r in rows]

print("=" * 72)
print("EGO-ECHO TEST at the Alpamayo anchor t0 = %.1f s" % ALPA_T0)
print("  (if the 'limit' were the speedometer, |limit-ego| would be ~0)")
print()
# stated-unit reading, defaulting a missing unit to km/h
v_kmh = [r["value"] * KMH for r in rows]
v_mph = [r["value"] * MPH for r in rows]
m_kmh = stats(v_kmh, egos, "ALL read as km/h")
m_mph = stats(v_mph, egos, "ALL read as mph")

# controls
rng = np.random.default_rng(0)
const = [50 * KMH] * len(rows)
m_const = stats(const, egos, "CONTROL const 50 km/h")
perm = rng.permutation([r["value"] for r in rows])
m_shuf = stats([p * KMH for p in perm], egos, "CONTROL shuffled (km/h)")
print()

# split by whether a unit was STATED
for want in ("km/h", "mph", None):
    sub = [r for r in rows if r["unit"] == want]
    if not sub:
        continue
    lab = want if want else "NO UNIT STATED"
    e = [r["ego_ms"] for r in sub]
    print("  --- unit stated = %s  (n=%d) ---" % (lab, len(sub)))
    stats([r["value"] * KMH for r in sub], e, "     as km/h")
    stats([r["value"] * MPH for r in sub], e, "     as mph")
print()

print("=" * 72)
print("PER-CLIP TABLE (limit under stated unit vs ego speed at anchor)")
print("  %-9s %-14s %5s %-5s %8s %8s %8s" %
      ("clip", "task", "val", "unit", "lim_ms", "ego_ms", "ego_max"))
n_below = 0
for r in rows:
    u = r["unit"] or "?"
    f = MPH if u == "mph" else KMH
    lim = r["value"] * f
    if lim < r["ego_ms"]:
        n_below += 1
    print("  %-9s %-14s %5d %-5s %8.2f %8.2f %8.2f %s" %
          (r["clip"][:8], r["task"], r["value"], u, lim, r["ego_ms"],
           r["ego_max_ms"], "<-- LIMIT BELOW EGO" if lim < r["ego_ms"] else ""))
print()
print("clips where the claimed limit is BELOW the ego's speed at the anchor: "
      "%d / %d" % (n_below, len(rows)))

# the 30-specific check (C87's dashboard roundel)
r30 = [r for r in rows if r["value"] == 30]
if r30:
    e30 = np.array([r["ego_ms"] for r in r30])
    print()
    print("VALUE==30 subset (C87's dashboard-roundel failure mode): n=%d" % len(r30))
    print("   ego speed at anchor: median %.2f m/s  min %.2f  max %.2f"
          % (float(np.median(e30)), float(e30.min()), float(e30.max())))
    print("   30 km/h = %.2f m/s ; 30 mph = %.2f m/s" % (30 * KMH, 30 * MPH))

json.dump({"rows": rows, "n_with_ego": len(rows), "n_missing": missing,
           "median_abs_diff": {"as_kmh": m_kmh, "as_mph": m_mph,
                               "control_const50": m_const,
                               "control_shuffled": m_shuf},
           "n_limit_below_ego": n_below},
          open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print()
print("WROTE %s" % sys.argv[2])
