"""Are the three teacher-derived training signals CONSISTENT with the camera frame they sit on?

PI question, 2026-09-21: *"Is the problem of using the teacher signals consistently to the frames
solved?"* This file is the answer as an instrument, so the answer can be re-checked rather than
re-asserted.

Every REFe training tuple carries THREE teacher-derived signals, and the paper requires all three to
describe the SAME pose as the image:

  (1) the TRAJECTORY TARGET -- "rolling out the frozen teacher at each frame" (p. 7);
  (2) the GOAL / navigation command -- "derived from the goal point GIVEN TO THE TEACHER, so student
      and teacher share one driving intent" (pp. 7-8);
  (3) the six PDM SCORER targets -- scored on candidates built around the teacher trajectory.

⛔ WHAT WAS WRONG, ALL MEASURED 2026-09-21. The historical bank harvested all three out of one
CLOSED-LOOP simulation, whose ego drifted off the log, while the camera frames exist only for the
LOGGED trajectory:
  * pose drift: **83.0 %** of states > 0.5 m from the logged ego, max **24.70 m**;
  * goal: up to **8.39 m** from the goal the teacher actually received at the logged frame
    (mean 4.15 m) -- identical at step 0, diverging after, i.e. the same drift in a second field;
  * scorer: context history, teacher trajectory, lane-graph anchor AND candidate set all centred on
    the drifted ego.

⭐ WHY THIS INSTRUMENT IS BUILT THE WAY IT IS. Tonight produced two checks of mine that could not go
red: a pairing check computing `dist(cam_xy, cam_xy)` (zero by definition), and a buffer-sensitivity
test run at frame 0, where the clamp made the knob immovable. So:
  * every expectation reads an INDEPENDENT source -- the DB's `ego_pose` and `image` tables, and the
    CLOSED-LOOP log as a CONTRAST -- never the scenario object the builder itself used;
  * every arm has a MUTATION in `--self-test` that reintroduces the historical defect and must go RED;
  * arm G2 is DISCRIMINATING by design: the stored goal must MATCH the closed-loop goal at step 0
    (where the two constructions coincide) and must DIFFER from it where the ego had drifted. A goal
    that matched everywhere would mean the fix did nothing; one that differed at step 0 would mean
    it broke the part that was already right.

Usage:
  python diag_signal_consistency.py --bank <perframe dir> --scorer <scorer jsonl> [--self-test]
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

DB = "D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14"
# navtrain logs are NOT under the val14 link dir -- they are ordinary split DBs. Searched in order,
# so the val14 bank keeps reading exactly the file it always read.
DB_ROOTS = [DB,
            "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval",
            "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
            "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/mini"]
DRIFT_RUN = {0: "C:/dzo/m-nr-n", 1: "C:/dzo/m-nr-r1"}


def _db_path(log: str) -> str:
    for root in DB_ROOTS:
        p = os.path.join(root, f"{log}.db")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no nuPlan DB for log {log} under any of {DB_ROOTS}. P1/P2 read the DB INDEPENDENTLY of "
        f"the bank; without it they cannot run, and they must not be reported as passing.")


class LogIndex:
    """Independent reads from the nuPlan DB -- the object the camera frame actually belongs to."""

    def __init__(self):
        self._ego, self._img = {}, {}

    def _load(self, log):
        if log not in self._ego:
            c = sqlite3.connect(_db_path(log))
            self._ego[log] = c.execute(
                "SELECT timestamp, x, y FROM ego_pose ORDER BY timestamp").fetchall()
            self._img[log] = dict(c.execute("SELECT filename_jpg, timestamp FROM image").fetchall())
            c.close()

    def image_ts(self, log, rel):
        self._load(log)
        return self._img[log].get(rel)

    def ego_xy_at(self, log, t):
        self._load(log)
        r = self._ego[log]
        ts = [a[0] for a in r]
        i = min(max(bisect.bisect_left(ts, t), 0), len(ts) - 1)
        if i and abs(ts[i - 1] - t) < abs(ts[i] - t):
            i -= 1
        return r[i][1], r[i][2]


def closed_loop_goals(rank: int) -> dict:
    """The goal each CLOSED-LOOP step used -- the CONTRAST for arm G2, not a reference."""
    from nuplan.planning.simulation.simulation_log import SimulationLog
    out = {}
    for lp in sorted(glob.glob(f"{DRIFT_RUN[rank]}/**/*.msgpack.xz", recursive=True)):
        log = SimulationLog.load_data(file_path=Path(lp))
        for i, s in enumerate(log.simulation_history.data):
            g = getattr(s.trajectory, "goal_points", None)
            if g is not None:
                flat = [float(v) for v in __import__("numpy").asarray(g).reshape(-1)[:2]]
                # ⛔ keyed with the SCENARIO TOKEN -- one log carries up to three scenarios
                out[(log.scenario.log_name, log.scenario.scenario_name, i)] = flat
    return out


def arms(rows, scorer, cl_goals, idx) -> dict:
    """Pure over its inputs, so the self-test can feed it a deliberately broken bank."""
    # P1 image <-> ego: where the ego was when the CAMERA fired, vs the pose implied by the tuple's
    # own recorded dt and speed. The tuple contributes only dt_ms and speed.
    resid = []
    for r in rows:
        t = idx.image_ts(r["log_name"], r["image"][0])
        if t is None:
            continue
        spd = r["ego"][6] if len(r["ego"]) > 6 else 0.0
        resid.append(spd * r.get("dt_ms", 0.0) / 1000.0)

    # P2 the tuple's ORIGIN must be the LOGGED ego -- read from the DB, not from the builder.
    # `origin_world` is written by build_teacher_rollouts.py; a closed-loop bank has none, which is
    # itself a failure (the bank cannot prove where its origin was).
    origin_gap = []
    for r in rows:
        ow = r.get("origin_world")
        t = idx.image_ts(r["log_name"], r["image"][0])
        if ow is None or t is None:
            origin_gap.append(float("inf"))
            continue
        origin_gap.append(math.dist(idx.ego_xy_at(r["log_name"], t), ow[:2]))

    # G1/G2 the goal: MATCH the closed-loop goal at step 0, DIFFER where the ego had drifted
    g0_match = g_late_diff = n0 = nlate = 0
    for r in rows:
        cl = cl_goals.get((r["log_name"], r["token"], int(r["step"])))
        if cl is None or not r.get("goal"):
            continue
        d = math.dist(r["goal"][:2], cl)
        if int(r["step"]) == 0:
            n0 += 1
            g0_match += d < 0.05
        elif int(r["step"]) >= 40:
            nlate += 1
            g_late_diff += d > 0.5

    # S1 the scorer's TEACHER candidate must be the per-frame rollout for the same (log, step)
    s_match = s_n = 0
    for r in rows:
        k = (r["log_name"], r["token"], int(r["step"]))
        st = scorer.get(k)
        if st is None:
            continue
        s_n += 1
        a = [p[:2] for p in r["traj"]]
        b = [p[:2] for p in st]
        s_match += (len(a) == len(b) and max(math.dist(x, y) for x, y in zip(a, b)) < 1e-3)

    fin = [g for g in origin_gap if g != float("inf")]
    return {
        "_resid": resid, "_origin": origin_gap, "_g0": (g0_match, n0), "_glate": (g_late_diff, nlate),
        "_s": (s_match, s_n),
        "P1 image-to-ego residual is sub-metre (the rig's own sync term only)":
            bool(resid) and max(resid) < 1.0,
        # ⚠️ 1.0 m, NOT 0.5. P2 reads the DB ego at the CAMERA's timestamp, so it inherits the
        # rig's camera-to-state sync term (the same one P1 measures, MEASURED max 0.543 m). The
        # first version used 0.5 m and PASSED AT 0.486 m on a 60-tuple pilot -- a pass by
        # 3 %, which the full bank's faster segments would have turned into a false FAIL on
        # correct data. The threshold must sit ABOVE the sync term and FAR BELOW the drift it
        # exists to catch (83 % of states > 0.5 m, max 24.70 m); the self-test's 5 m mutation
        # still goes red at 1.0 m.
        "P2 every tuple's ORIGIN is the LOGGED ego (independent DB read, < 1.0 m)":
            bool(origin_gap) and all(g < 1.0 for g in origin_gap),
        # ⛔⛔ None means NOT APPLICABLE, and it must never read as a pass. G1/G2 compare the bank's
        # goals against a CLOSED-LOOP RUN; navtrain has none, so nothing overlaps and the old
        # `n0 == 0 or ...` / `nlate == 0 or ...` were VACUOUSLY TRUE -- [PASS] printed for two arms
        # that had compared nothing. MEASURED on the first navtrain bank: G1 0/0 [PASS],
        # G2 0/0 [PASS].
        # ⚠️ The FIRST fix here keyed on `not cl_goals` and DID NOT WORK, because `cl_goals` is not
        # empty: it is loaded from `DRIFT_RUN[rank]`, a val14 run that exists on disk, whose keys
        # simply never match a navtrain row. **The condition is OVERLAP, not emptiness** -- n0 and
        # nlate already count only rows that HAVE a closed-loop counterpart, so 0 means "compared
        # nothing" no matter why. Testing the dict instead of the comparison is the same mistake
        # one level up.
        "G1 CONTROL at step 0 the goal MATCHES the closed-loop one (they coincide there)":
            None if n0 == 0 else (g0_match == n0),
        "G2 where the ego had drifted, the goal DIFFERS from the closed-loop one":
            None if nlate == 0 else (g_late_diff > 0.5 * nlate),
        "S1 the scorer's teacher candidate IS the per-frame rollout":
            s_n > 0 and s_match == s_n,
        "C CONTROL rows were actually read": bool(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="per-frame bank dir")
    ap.add_argument("--scorer", required=True, help="scorer jsonl built with --perframe-bank")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(
        os.path.join(a.bank, f"targets_rank{a.rank}.jsonl"), encoding="utf-8") if l.strip()]
    scorer = {}
    for l in open(a.scorer, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            if d.get("candidate") == "teacher":
                scorer[(d["log_name"], d["token"], int(d["step"]))] = d["traj"]
    idx = LogIndex()
    cl = closed_loop_goals(a.rank)
    r = arms(rows, scorer, cl, idx)

    res = r["_resid"]
    fin = [g for g in r["_origin"] if g != float("inf")]
    print(f"  {len(rows)} tuples, {r['_s'][1]} with scorer targets")
    print(f"  P1 image-to-ego residual: mean {sum(res)/max(len(res),1):.3f} m  "
          f"max {max(res) if res else float('nan'):.3f} m")
    print(f"  P2 origin vs LOGGED ego:  max {max(fin) if fin else float('nan'):.4f} m "
          f"over {len(fin)} tuples")
    print(f"  G1 step-0 goals matching the closed-loop goal: {r['_g0'][0]}/{r['_g0'][1]}")
    print(f"  G2 late goals DIFFERING from the closed-loop goal: {r['_glate'][0]}/{r['_glate'][1]}")
    print(f"  S1 scorer teacher == per-frame rollout: {r['_s'][0]}/{r['_s'][1]}")
    print()
    n_na = 0
    for k, v in r.items():
        if not k.startswith("_"):
            if v is None:
                n_na += 1
                print(f"  [N/A ] {k}")
            else:
                print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if n_na:
        # Stated as a DEFICIT, not a footnote: this bank has fewer live arms than the val14 bank,
        # so "all green" here is a weaker statement. Read the skip reason, never just the colour.
        # ASCII ONLY in printed strings -- the console is cp1252 and a decorative character here
        # raised UnicodeEncodeError AFTER every arm had printed, turning a passing gate into a
        # non-zero exit that reads exactly like a failure.
        print(f"\n  !! {n_na} arm(s) N/A -- no closed-loop counterpart exists for any row in this "
              f"bank (navtrain has no closed-loop run), so the goal-vs-closed-loop contrast could "
              f"not run. Verified by P1/P2/S1/C only; G1/G2 are NOT evidence here.")

    if a.self_test:
        print("\n== SELF-TEST: each arm must go RED under its historical defect ==")
        bad = []
        # P2 <- the closed-loop bank's defect: an origin that drifted 5 m off the log
        drift = [dict(x, origin_world=[x["origin_world"][0] + 5.0, x["origin_world"][1],
                                       x["origin_world"][2]]) for x in rows if x.get("origin_world")]
        m = arms(drift, scorer, cl, idx)
        ok = not m["P2 every tuple's ORIGIN is the LOGGED ego (independent DB read, < 1.0 m)"]
        print(f"  [{'RED ' if ok else 'FAIL'}] P2 on an origin drifted 5 m off the log")
        bad += [] if ok else ["P2"]
        # G2 <- the goal read from the CLOSED-LOOP log (what line 330 used to do)
        clg = [dict(x, goal=cl.get((x["log_name"], x["token"], int(x["step"])), x["goal"][:2])
                    + [0.0, 0.0])
               for x in rows]
        m = arms(clg, scorer, cl, idx)
        g2 = m["G2 where the ego had drifted, the goal DIFFERS from the closed-loop one"]
        ok = (g2 is False)   # ⛔ NOT `not g2`: None is N/A, and `not None` would print a fake RED
        n_late = m["_glate"][1]
        if g2 is None:
            print("  [N/A ] G2 self-test -- no closed-loop run for this bank, nothing to mutate "
                  "toward; the arm is inapplicable, not proven")
        elif n_late == 0:
            print("  [SKIP] G2 -- no late-step tuples in this bank to test it on")
        else:
            print(f"  [{'RED ' if ok else 'FAIL'}] G2 on goals read from the closed-loop log")
            bad += [] if ok else ["G2"]
        # S1 <- a scorer whose teacher is shifted (built around a different trajectory)
        shifted = {k: [[p[0] + 1.0, p[1]] + p[2:] for p in v] for k, v in scorer.items()}
        m = arms(rows, shifted, cl, idx)
        ok = not m["S1 the scorer's teacher candidate IS the per-frame rollout"]
        print(f"  [{'RED ' if ok else 'FAIL'}] S1 on a scorer built around a different trajectory")
        bad += [] if ok else ["S1"]
        if bad:
            print(f"\nSELF_TEST_FAILED -- inert: {bad}")
            return 2
        print("\nSELF_TEST_OK")

    # ⛔ The verdict is taken over the APPLICABLE arms only. `None` must neither pass (it compared
    # nothing) nor fail (nothing was wrong) -- and the marker carries the deficit so a green line
    # cannot be quoted without it. `all()` over a None would have read SIGNALS_INCONSISTENT and
    # sent someone hunting a defect that does not exist.
    live = [v for k, v in r.items() if not k.startswith("_") and v is not None]
    good = bool(live) and all(live)
    marker = "SIGNALS_CONSISTENT" if good else "SIGNALS_INCONSISTENT"
    if good and n_na:
        marker += f"_PARTIAL_{n_na}_ARM_NA"
    print("\n" + marker)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
