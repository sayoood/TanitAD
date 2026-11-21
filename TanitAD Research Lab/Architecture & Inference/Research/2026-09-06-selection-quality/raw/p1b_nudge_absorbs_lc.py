"""P1b - IS THE LANE CHANGE ABSORBED BY `NUDGE`?

`ego_manoeuvre.py` calls NUDGE on `abs(lat_peak_m) >= NUDGE_LAT_M` (1.0 m) and
the class doc says, verbatim, that NUDGE has **NO UPPER BOUND** and that
`lat_peak_m` is "the only field that can distinguish a 1 m wobble from an
absorbed lane change".  That is exactly the question here, and the field is
persisted in the label blobs, so it is answerable with zero GPU.

UNITS, asserted from source before any arithmetic (CLAUDE.md units rule):
  ego_manoeuvre.py, `lat_peak_m` docstring -- "Signed peak lateral offset in
  METRES from the key-frame heading (`+` = left)".

A US/EU lane is 3.0-3.7 m wide.  A completed lane change therefore parks its
lateral peak at |lat_peak_m| ~ 3-4 m; a genuine in-lane wobble sits near 1-2 m.
"""
import collections
import gzip
import json
import sys

import numpy as np

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BLOBS = {
    "EVAL ": r"C:\Users\Admin\navcomp\data\s2_labels_v7.2_eval.jsonl.gz",
    "TRAIN": r"C:\Users\Admin\rlgate\s2_labels_v7.2_train.jsonl.gz",
}
LANE_LO, LANE_HI = 2.5, 5.0          # a lane-width band, stated up front


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    out = {}
    for tag, path in BLOBS.items():
        by_lat = collections.defaultdict(list)
        goal_lc_clips = []
        n = 0
        miss = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                at = r.get("a_tac") or {}
                lat = at.get("lat")
                lp = (at.get("lat_args") or {}).get("lat_peak_m")
                if lp is None:
                    miss += 1
                else:
                    by_lat[lat].append(float(lp))
                goals = (r.get("g_tac") or {}).get("goals") or {}
                gset = set(goals if isinstance(goals, dict) else goals)
                if gset & {"LANE_CHANGE_L", "LANE_CHANGE_R", "MERGE",
                           "OVERTAKE_VEHICLE", "TAKE_EXIT_L", "TAKE_EXIT_R"}:
                    goal_lc_clips.append(
                        (r.get("clip_id"), lat, at.get("lon"), lp,
                         sorted(gset & {"LANE_CHANGE_L", "LANE_CHANGE_R",
                                        "MERGE", "OVERTAKE_VEHICLE",
                                        "TAKE_EXIT_L", "TAKE_EXIT_R"})))
        print("=" * 78)
        print(f"{tag}  n_clips={n}  lat_peak_m present on {n - miss}/{n}")
        if miss == n:
            print("  INCONCLUSIVE: the field is absent from this blob "
                  "(not evidence about lane changes)")
            continue
        print(f"\n  |lat_peak_m| DISTRIBUTION BY EMITTED LATERAL ACTION "
              f"(metres; lane band [{LANE_LO}, {LANE_HI}) m)")
        print(f"    {'a_tac.lat':<14}{'n':>6}{'p50':>8}{'p90':>8}{'p99':>8}"
              f"{'max':>9}{'>=2.5m':>9}{'in lane band':>14}")
        for k in sorted(by_lat, key=lambda z: -len(by_lat[z])):
            a = np.abs(np.asarray(by_lat[k], dtype=float))
            band = int(((a >= LANE_LO) & (a < LANE_HI)).sum())
            print(f"    {str(k):<14}{len(a):>6}{q(a,50):>8.2f}{q(a,90):>8.2f}"
                  f"{q(a,99):>8.2f}{a.max():>9.2f}"
                  f"{int((a>=LANE_LO).sum()):>9}"
                  f"{band:>8} / {len(a)}")
        nud = np.abs(np.asarray(by_lat.get("NUDGE_L", []) +
                                by_lat.get("NUDGE_R", []), dtype=float))
        if len(nud):
            print(f"\n  NUDGE_L + NUDGE_R pooled: n={len(nud)}")
            for lo, hi in ((1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0),
                           (3.0, 3.5), (3.5, 4.0), (4.0, 5.0), (5.0, 1e9)):
                c = int(((nud >= lo) & (nud < hi)).sum())
                bar = "#" * min(60, int(60 * c / max(1, len(nud)) * 3))
                print(f"    [{lo:>4.1f}, {hi:>5.1f}) m  {c:>5} / {len(nud)}  "
                      f"{100.0*c/len(nud):5.2f} %  {bar}")
            print(f"    >= {LANE_LO} m (>= a lane width): "
                  f"{int((nud>=LANE_LO).sum())} / {len(nud)}")
        print(f"\n  CLIPS CARRYING A LANE-CHANGE-FAMILY *GOAL*: "
              f"{len(goal_lc_clips)} / {n}")
        cnt = collections.Counter(c[1] for c in goal_lc_clips)
        print(f"    their emitted a_tac.lat: {dict(cnt)}")
        strict = [c for c in goal_lc_clips
                  if set(c[4]) & {"LANE_CHANGE_L", "LANE_CHANGE_R"}]
        print(f"    of which the goal is literally LANE_CHANGE_L/R: "
              f"{len(strict)}")
        cnt2 = collections.Counter(c[1] for c in strict)
        print(f"      their emitted a_tac.lat: {dict(cnt2)}")
        for c in strict[:12]:
            print(f"        {c[0][:8]}  lat={c[1]:<10} lon={str(c[2]):<22} "
                  f"lat_peak_m={c[3]}  goals={c[4]}")
        out[tag] = {"n_clips": n, "n_with_lat_peak": n - miss,
                    "by_lat_n": {k: len(v) for k, v in by_lat.items()},
                    "nudge_ge_lane": int((nud >= LANE_LO).sum())
                    if len(nud) else None,
                    "nudge_n": int(len(nud)),
                    "goal_lc_clips": len(goal_lc_clips),
                    "goal_lc_strict": len(strict),
                    "goal_lc_strict_emitted_lat": dict(cnt2)}
    with open("out_p1b_nudge_absorbs_lc.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote out_p1b_nudge_absorbs_lc.json")


if __name__ == "__main__":
    main()
