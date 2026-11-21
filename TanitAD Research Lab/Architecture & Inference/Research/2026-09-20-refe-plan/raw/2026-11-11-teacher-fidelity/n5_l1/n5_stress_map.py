"""n5_stress_map.py - CONSTRUCTION stress set B2: a genuinely TWO-SIDED corridor made by ERODING the map (the navtrain RJ5 population has none: MEASURED).
Everything it produces is CONSTRUCTION (a synthetic map, real tokens, real teacher targets). It never reads a repair result to select.
Per candidate token (DAC = 1 on its unmodified label; unbounded corridors skipped; one per token, smallest sha1 first):
  1. executed state of the unmodified label -> per-step lateral free interval [lo_k, hi_k] at margin 0 (n4_repair_l1.step_intervals); wmin = min_k (hi_k - lo_k).
  2. slack s drawn from {0.25, 0.40, 0.60} m by sha1; erosion e = (wmin - s) / 2 (> 0): the drivable union is eroded by e on EVERY side (each polygon of the
     scorer's map is intersected with union.buffer(-e), so shared seams stay closed), in BOTH the scorer's map (parts[5]) and cx.drivable.
  3. the base label (unmodified target) is re-scored on the eroded map: it must FAIL DAC (else 'no_failure', excluded).
  4. KNOWN FEASIBLE ANSWER = an independent ORACLE, not L1: the target shifted laterally per pose by the MIDPOINT of the original free interval
     ((lo+hi)/2 at the pose's step, along the executed heading normal, ramped over 1 s), relabelled by the unchanged pipeline and scored on the eroded map:
     kept only if ORACLE DAC = 1 and the full certificate holds against the eroded base label (MEASURED, recorded per token).
  5. on kept tokens run A1 REPAIR-D, L1 and the sign-flipped L1x exactly as n5_run does.
usage: python n5_stress_map.py SCAN.jsonl OUT.jsonl [--n 120] [--shard i/n] [--arms A1,L1,L1x]"""
import argparse
import hashlib
import json
import math
import os

import n5_env  # noqa: F401
import numpy as np

import n4_geom as G
import n4_repair as RP
import n4_repair_l1 as L1
import n5_run as RUN

SLACK = (0.25, 0.40, 0.60)
T20 = 0.2 * np.arange(1, 21)


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def eroded_parts(cx, e):
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    dam = cx.parts[5]
    E = cx.drivable.buffer(-e)
    geoms = [g.intersection(E) for g in dam._geometries]
    dam2 = PDMDrivableMap(dam._tokens, dam._map_types, geoms, dam._node_capacity)
    return tuple(cx.parts[:5]) + (dam2,), E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--arms", default="A1,L1,L1x")
    ap.add_argument("--slack", default="0.25,0.40,0.60", help="slack set (m): free width beyond the footprint at the tightest step; B3 uses 0.12,0.18,0.25")
    ap.add_argument("--oracle-ramp", type=float, default=1.0, help="oracle ramp time (s); B3 uses 0.2")
    ap.add_argument("--wiggle", action="store_true", help="B2w: the target also carries a lateral sinusoid (amplitude {0.2,0.35,0.5} m, period {2,4} s by sha1) "
                    "so the executed path swings between the walls (violations on BOTH sides at different times); the oracle stays the unwiggled midpoint path")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(x) for x in open(a.scan, encoding="utf-8") if x.strip().endswith("}")]
    cand = [r for r in rows if "error" not in r and r["base"]["dac"] == 1.0 and "width_min_m" in r and r["width_min_m"] < 2.9]
    cand.sort(key=lambda r: h(r["token"]))
    cand = cand[:a.n]
    cand = [r for i, r in enumerate(cand) if i % sn == si]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["token"] for x in open(a.out, encoding="utf-8") if x.strip().endswith("}")}
    by = {}
    meta = {}
    for r in cand:
        if r["token"] not in done:
            by.setdefault(r["log"], []).append(r["token"])
            meta[r["token"]] = r
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg, tk, cx in G.contexts(by):
            r = meta[tk]
            rec = {"id": "STRESSMAP:" + tk, "token": tk, "log": lg, "tag": "CONSTRUCTION"}
            try:
                t20 = np.asarray(r["t20"], float)
                lab0 = cx.labels(t20)["R"]
                res0, sim0 = cx.score([lab0])
                corners, heading = L1._corners_heading(cx, sim0[0])
                lo, hi, ne = L1.step_intervals(cx.drivable, corners, heading, 0.0)
                if not ne[1:].all() or res0[0]["dac"] != 1.0:
                    rec["skip"] = "unbounded_or_failing_base"
                else:
                    wmin = float(np.min(hi[1:] - lo[1:]))
                    SL = [float(x) for x in a.slack.split(",")]
                    s = SL[h("s|" + tk) % len(SL)]
                    e = (wmin - s) / 2
                    rec["construct"] = {"wmin_m": round(wmin, 4), "slack_m": s, "erosion_m": round(e, 4)}
                    if e <= 0:
                        rec["skip"] = "no_erosion"
                    else:
                        parts2, E = eroded_parts(cx, e)
                        cx.parts, cx.drivable = parts2, E
                        t_run = t20
                        if a.wiggle:
                            amp, per = (0.2, 0.35, 0.5)[h("a|" + tk) % 3], (2.0, 4.0)[h("p|" + tk) % 2]
                            nt = np.stack([-np.sin(t20[:, 2]), np.cos(t20[:, 2])], -1)
                            t_run = t20.copy()
                            t_run[:, :2] += (amp * np.sin(2 * np.pi * T20 / per))[:, None] * nt
                            rec["construct"].update({"wiggle_amp_m": amp, "wiggle_period_s": per})
                        base = cx.score([cx.labels(t_run)["R"]])[0][0]
                        rec["base"] = {k: float(base[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")}
                        rec["base_dac0"] = bool(base["dac"] == 0.0)
                        # independent oracle: midpoint of the ORIGINAL free interval per pose
                        mid = 0.5 * (lo + hi)
                        hp = heading[L1.POSES]
                        nrm = np.stack([-np.sin(hp), np.cos(hp)], -1)
                        h0 = cx.ego.rear_axle.heading
                        Rm = np.array([[math.cos(-h0), -math.sin(-h0)], [math.sin(-h0), math.cos(-h0)]])
                        d_ego = (np.clip(T20 / a.oracle_ramp, 0, 1)[:, None] * mid[L1.POSES][:, None] * nrm) @ Rm.T
                        to = t20.copy()
                        to[:, :2] += d_ego
                        lo_lab = cx.labels(to)["R"]
                        ro = cx.score([lo_lab])[0][0]
                        rec["oracle"] = {"dac": float(ro["dac"]), "certified": bool(RP.certified(ro, base)), "max_shift_m": float(np.abs(d_ego).max())}
                        rec["scorer_vs_geometry"] = None
                        if rec["base_dac0"] and rec["oracle"]["certified"]:
                            rec["base_dac0"] = True
                            rec["arms"] = {arm: RUN.run_arm(arm, cx, t_run, base, {}) for arm in a.arms.split(",")}
                        else:
                            rec["base_dac0"] = bool(rec["base_dac0"] and rec["oracle"]["certified"])
                            rec["skip"] = "no_failure" if not rec["base_dac0"] and base["dac"] == 1.0 else "oracle_not_certified"
            except Exception as ex:                                  # noqa: BLE001
                rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(rec["id"], rec.get("skip", ""), {k: v.get("status") for k, v in rec.get("arms", {}).items()}, rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
