"""n5_stress.py - build the CONSTRUCTION stress set (synthetic; every artifact derived from it is labelled CONSTRUCTION).
Source: scan.jsonl rows (RJ5 re-roll rows of finished batches, builder-scored). Selection is fixed BEFORE any repair arm runs and reads no repair result:
  candidate = DAC = 1 on the UNSHIFTED label (this is how the feasible answer is known: that label executes inside the drivable area, MEASURED at
              scan) and the nearer boundary within CLEAR_MAX (0.48 m) of the executed path. No two-sidedness filter (none exists in this population).
  one candidate per token (smallest sha1 first), N = up to N_MAX.
  shift: delta from the grid {0.1,0.2,0.3,0.4,0.5} m, drawn by sha1(token) among the grid values > clearance toward the nearer boundary + 0.02 m (so the
         shifted label is expected to fail; the runner records whether it did), direction = toward the NEARER boundary, ramp(t / 1 s) like REPAIR-B,
         applied along the left normal of each target pose heading.
usage: python n5_stress.py SCAN.jsonl OUT.jsonl [--n 80]"""
import argparse
import hashlib
import json

import n5_env  # noqa: F401
import numpy as np

CLEAR_MAX = 0.48     # nearer boundary reachable by the largest shift in the grid; NO two-sidedness filter: the navtrain RJ5 rows have 0 corridors with
                     # both clearances <= 0.9 m (MEASURED scan.jsonl), so the brief's literal construction is one-sided in practice (reported as such)
GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
T20 = 0.2 * np.arange(1, 21)


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.scan, encoding="utf-8") if x.strip().endswith("}")]
    cand = [r for r in rows if "error" not in r and r["base"]["dac"] == 1.0 and "width_min_m" in r
            and min(r["clear_left_m"], r["clear_right_m"]) <= CLEAR_MAX]
    cand.sort(key=lambda r: h(r["token"]))
    out, seen = [], set()
    for r in cand:
        if r["token"] in seen or len(out) >= a.n:
            continue
        left = r["clear_left_m"] <= r["clear_right_m"]
        c = r["clear_left_m"] if left else r["clear_right_m"]
        opts = [g for g in GRID if g > c + 0.02]
        if not opts:
            continue
        d = opts[h("d|" + r["token"]) % len(opts)]
        t20 = np.asarray(r["t20"], float)
        nrm = np.stack([-np.sin(t20[:, 2]), np.cos(t20[:, 2])], -1)
        ramp = np.clip(T20 / 1.0, 0.0, 1.0)
        t2 = t20.copy()
        t2[:, :2] += (1.0 if left else -1.0) * d * ramp[:, None] * nrm
        seen.add(r["token"])
        out.append({"id": "STRESS:" + r["token"], "token": r["token"], "log": r["log"], "tag": "CONSTRUCTION", "t20": t2.round(9).tolist(),
                    "construct": {"side": "left" if left else "right", "delta_m": d, "clear_toward_m": c, "clear_left_m": r["clear_left_m"],
                                  "clear_right_m": r["clear_right_m"], "width_min_m": r["width_min_m"], "unshifted_dac": 1.0}})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("candidates", len(cand), "selected", len(out))


if __name__ == "__main__":
    main()
