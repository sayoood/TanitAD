"""Build the turn-enriched window list -- EXACTLY the rule published in
SPEC_TURN_ASYMMETRY.md section 2, and nothing else.

GT-only: the strata come from ground-truth ego poses through the programme's own
canonical gate. No arm output is read here.
"""
import collections
import json
import sys

LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
QUOTA = {"turn_left": 30, "turn_right": 30, "lane_keep": 15}
RULE = ("SPEC_TURN_ASYMMETRY.md section 2: strata from GT only via "
        "refc_tactical.factor_from_kinematics (v1 branch, |dyaw| > 0.15 rad); "
        "quota 30 turn_left + 30 turn_right + 15 lane_keep; allocated across "
        "the episodes carrying each stratum by round-robin (largest remaining "
        "availability first); within an (episode, stratum) cell taken evenly "
        "spaced in t over that cell's available indices; ordered by "
        "(episode, t). The episode SET is unchanged -- parity is preserved.")


def evenly(xs, k):
    """k items from xs, evenly spaced, endpoints included. Deterministic."""
    n = len(xs)
    if k >= n:
        return list(xs)
    if k == 1:
        return [xs[n // 2]]
    return [xs[round(i * (n - 1) / (k - 1))] for i in range(k)]


def main(census_path, out_path):
    rows = json.load(open(census_path))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    names = {}
    for r in rows:
        names[r["ei"]] = r["name"]
        by[LAT[r["lat"]]][r["ei"]].append(r["t"])
    for s in by:
        for e in by[s]:
            by[s][e].sort()

    picked = []
    report = []
    for stratum, quota in QUOTA.items():
        cells = by[stratum]
        take = {e: 0 for e in cells}
        avail = {e: len(v) for e, v in cells.items()}
        # round-robin, largest remaining availability first
        for _ in range(quota):
            cand = [e for e in cells if take[e] < avail[e]]
            if not cand:
                break
            e = min(cand, key=lambda e: (take[e], -avail[e], e))
            take[e] += 1
        got = 0
        for e in sorted(cells):
            if not take[e]:
                continue
            ts = evenly(cells[e], take[e])
            assert len(set(ts)) == len(ts), (stratum, e)
            for t in ts:
                picked.append((names[e], int(t), stratum, int(e)))
            got += len(ts)
            report.append((stratum, e, names[e], avail[e], len(ts),
                           min(ts), max(ts),
                           min(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else None))
        print("  %-11s quota %2d -> got %2d over %d episodes"
              % (stratum, quota, got, sum(1 for e in take if take[e])))

    picked.sort(key=lambda x: (x[3], x[1]))
    pairs = [[p[0], p[1]] for p in picked]
    assert len(set(map(tuple, pairs))) == len(pairs), "duplicate (episode, t)"

    print("\n  %-11s %-14s %5s %5s %6s %6s %8s"
          % ("stratum", "episode", "avail", "take", "t_min", "t_max", "min_gap"))
    for r in report:
        print("  %-11s %-14s %5d %5d %6d %6d %8s"
              % (r[0], r[2][:13], r[3], r[4], r[5], r[6], r[7]))

    doc = {"rule": RULE, "windows": pairs,
           "n": len(pairs),
           "strata": {s: sum(1 for p in picked if p[2] == s) for s in QUOTA},
           "episodes_per_stratum": {
               s: sorted({p[3] for p in picked if p[2] == s}) for s in QUOTA},
           "source_census": census_path}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    print("\n  TOTAL %d windows over %d episodes -> %s"
          % (len(pairs), len({p[3] for p in picked}), out_path))
    print("  strata: %s" % doc["strata"])
    print("  episodes per stratum: %s" % doc["episodes_per_stratum"])
    print("\n  POWER TARGETS (SPEC section 1):")
    for s, need in (("turn_left", 30), ("turn_right", 30)):
        n = doc["strata"][s]
        eps = len(doc["episodes_per_stratum"][s])
        print("    %-11s n=%2d (target >=27, adopted 30): %s | clusters=%d "
              "(target >=5): %s | recall granularity 1/n = %.4f vs floor 0.0750: %s"
              % (s, n, "MET" if n >= 27 else "**MISSED**", eps,
                 "MET" if eps >= 5 else "**MISSED**", 1.0 / n,
                 "resolvable" if 1.0 / n <= 0.0750 / 2 else "**too coarse**"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
