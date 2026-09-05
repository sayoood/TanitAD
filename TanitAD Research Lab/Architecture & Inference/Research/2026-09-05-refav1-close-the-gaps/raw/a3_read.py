#!/usr/bin/env python3
"""READ THE GOAL-CONDITIONED COST ARMS -- and their WITHIN-ARM known-value control.

⭐⭐ THE CONTROL THIS FILE EXISTS FOR. `G_gkappa` charges the SAME W_KAPPA
(15.11245) on LANE_KEEP-goal windows that `T_wk15` charged there, and 0.0 on
TURN-goal windows. So, on the same rig and the same window grid:

    LANE_KEEP-goal windows : G_gkappa's controls MUST be bit-identical to T_wk15's
    TURN-goal windows      : they MUST differ

That is a stronger statement than a separate flag-off arm, and it costs no GPU:
it proves the lever is inert exactly where it should be AND active exactly where
it should be, on the trained checkpoint, inside one arm.
⛔ If BOTH columns differ, or NEITHER does, the A3 reading is VOID.

⚠️ A caution this instrument must not fall into: on a window whose canonical goal
seed is already the argmin, changing the weight changes the OBJECTIVE and not the
PLAN -- measured while writing the library test. So "identical controls" on a
TURN window is not automatically a failure; what must move is the population of
TURN windows AS A WHOLE, and the med|k|max is the readable statistic.

ASCII output only.
"""
import glob
import json
import os
import sys

import numpy as np

from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)

LAT = list(TACTICAL_LAT_ACTIONS_V7)
LON = list(TACTICAL_LON_ACTIONS_V7)
TURN_TOK = {LAT.index(t) for t in LAT if t.startswith("TURN_")}
LK_TOK = {LAT.index("LANE_KEEP")}


def load(dumpdir, arm="cl"):
    """(controls [N,H,2], goal_lat [N], ws [N], w_kappa_eff [N] or None)."""
    ctl, lat, ws, wke, wkc = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        key = "%s_controls" % arm
        if key not in z.files:
            return None
        ctl.append(z[key])
        lat.append(z["goal_lat_%s" % arm])
        ws.append(z["ws"])
        wke.append(z["w_kappa_eff_%s" % arm] if "w_kappa_eff_%s" % arm in z.files
                   else np.full(z["ws"].shape, np.nan))
        wkc.append(z["w_kappa_class_%s" % arm] if "w_kappa_class_%s" % arm in z.files
                   else np.full(z["ws"].shape, -1))
    if not ctl:
        return None
    return (np.concatenate(ctl, 0), np.concatenate(lat), np.concatenate(ws),
            np.concatenate(wke), np.concatenate(wkc))


def by_goal(name, ctl, lat):
    k = ctl[..., 1]
    absk = np.abs(k).max(-1)
    print("  %-14s %5s %11s %11s %11s" % ("goal token", "n", "med|k|max",
                                          "frac k!=0", "mean k^2"))
    for t in sorted(set(int(x) for x in lat)):
        m = lat == t
        nm = LAT[t] if 0 <= t < len(LAT) else str(t)
        print("  %-14s %5d %11.5f %11.4f %11.6f"
              % (nm, int(m.sum()), float(np.median(absk[m])),
                 float((absk[m] > 1e-9).mean()), float((k[m] ** 2).mean())))


def control(a_name, a, b_name, b):
    """THE WITHIN-ARM KNOWN-VALUE CONTROL."""
    (ca, la, wa, _, _), (cb, lb, wb, _, _) = a, b
    print("=" * 100)
    print("WITHIN-ARM CONTROL: %s vs %s" % (a_name, b_name))
    print("=" * 100)
    if not np.array_equal(wa, wb):
        print("  *** WINDOW GRIDS DIFFER -- VOID ***")
        return
    if not np.array_equal(la, lb):
        n = int((la != lb).sum())
        print("  *** DECODED GOAL DIFFERS ON %d/%d WINDOWS ***" % (n, len(la)))
        print("      The goal head does not depend on the cost, so this must be 0.")
        return
    print("  window grid identical (n=%d); decoded goal identical  [PRECONDITION OK]"
          % len(wa))
    for label, toks in (("LANE_KEEP-goal (weight UNCHANGED -> must be IDENTICAL)", LK_TOK),
                        ("TURN-goal      (weight 15.11245 -> 0 -> must DIFFER)", TURN_TOK),
                        ("other          (LANE_CHANGE/NUDGE/ABORT)", None)):
        if toks is None:
            m = ~np.isin(la, list(LK_TOK | TURN_TOK))
        else:
            m = np.isin(la, list(toks))
        n = int(m.sum())
        if n == 0:
            print("  %-58s n=0  (no windows)" % label)
            continue
        d = np.abs(ca[m] - cb[m]).max()
        nident = int((np.abs(ca[m] - cb[m]).reshape(n, -1).max(-1) == 0).sum())
        print("  %-58s n=%-3d max|dctl|=%.6e  identical %d/%d"
              % (label, n, float(d), nident, n))
    print()


def rec_row(path):
    d = json.load(open(path, encoding="utf-8"))
    cl = (d.get("arms") or {}).get("cl")
    ff = cl["four_families"]
    lat, tac, lon = ff["lateral"], ff["tactical"], ff["longitudinal"]
    pc = (tac.get("lateral_decision") or {}).get("per_class") or {}
    cost = d.get("cost") or {}
    return dict(name=os.path.basename(path)[4:-5],
                wk=(cost.get("weights") or {}).get("W_KAPPA"),
                wkg=cost.get("w_kappa_by_goal"),
                ade=cl["legacy_epmean_row"]["ade_m"],
                curv=lat["curvature_mae_1pm"], head=lat["heading_mae_deg"],
                yaw=lat["yaw_rate_mae_degps"], cross=lat["cross_mae_m"],
                spd=lon["speed_mae_mps"], along=lon["along_mae_m"],
                acc=lon["accel_mae_mps2"],
                lat_acc=(tac.get("lateral_decision") or {}).get("accuracy"),
                lat_kappa=(tac.get("lateral_decision") or {}).get("kappa"),
                lk=(pc.get("lane_keep") or {}).get("recall"),
                tl=(pc.get("turn_left") or {}).get("recall"),
                tr=(pc.get("turn_right") or {}).get("recall"),
                prog=(lon.get("ego_progress") or {}).get("progress_ratio_mean"),
                dk=(lon.get("distance_keeping") or {}).get("status"),
                strat=((ff.get("strategic") or {}).get("status")))


def main(argv):
    out, lon = argv[1], argv[2]
    recs = sorted(glob.glob(out + "/rec_*.json")) + sorted(glob.glob(lon + "/rec_T_wk15.json"))
    rows = [rec_row(p) for p in recs]
    print("=" * 118)
    print("A3 -- GOAL-CONDITIONED LATERAL COST, THOR, T1 'cl', p4 40 windows / 8 episodes")
    print("⛔ FOUR FAMILIES, never ADE alone: this package exists because the best-ADE")
    print("   arm executes ZERO turns and ADE cannot see it.")
    print("=" * 118)
    h = ("%-16s %10s %9s %8s %8s %8s | %8s %8s %8s | %8s %8s"
         % ("arm", "W_KAPPA", "ADE_m", "turnL", "turnR", "laneK",
            "curvMAE", "headMAE", "yawMAE", "spdMAE", "alongMAE"))
    print(h); print("-" * len(h))
    for r in rows:
        def f(x, w=8, p=4):
            return ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) else "%*s" % (w, "-")
        print("%-16s %10s %s %s %s %s | %s %s %s | %s %s"
              % (r["name"], str(r["wk"]), f(r["ade"], 9), f(r["tl"]), f(r["tr"]),
                 f(r["lk"]), f(r["curv"], 8, 5), f(r["head"], 8, 2),
                 f(r["yaw"], 8, 2), f(r["spd"]), f(r["along"])))
    print()
    for r in rows:
        print("  %-16s w_kappa_by_goal=%s  lat_acc=%s kappa=%s ego_progress=%s"
              % (r["name"], r["wkg"], r["lat_acc"], r["lat_kappa"], r["prog"]))
    print("  ⚠️ distance_keeping=%s  strategic=%s  -- UNAVAILABLE with a reason is a"
          % (rows[0]["dk"], rows[0]["strat"]))
    print("     WORK ITEM, not a pass (no lead block / no route label on this panel).")
    print()

    dumps = {}
    for tag, d in (("G_gkappa", out + "/dump_G_gkappa"),
                   ("G_gkappa_inv", out + "/dump_G_gkappa_inv"),
                   ("T_wk15", lon + "/dump_T_wk15")):
        r = load(d)
        if r is not None:
            dumps[tag] = r
            print("=" * 100)
            print("REALISED CURVATURE BY DECODED GOAL TOKEN -- %s" % tag)
            print("=" * 100)
            by_goal(tag, r[0], r[1])
            wke, wkc = r[3], r[4]
            if np.isfinite(wke).any():
                cls = ["lane_keep", "turn", "shift"]
                seen = {}
                for c, w in zip(wkc, wke):
                    seen.setdefault(int(c), set()).add(round(float(w), 6))
                print("  realised per-window weight by class: %s"
                      % {(cls[k] if 0 <= k < 3 else k): sorted(v)
                         for k, v in sorted(seen.items())})
            print()
    if "G_gkappa" in dumps and "T_wk15" in dumps:
        control("G_gkappa", dumps["G_gkappa"], "T_wk15", dumps["T_wk15"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
