"""Score the merged per-frame context-label release against the PI's 60 audited frames (pilot_audit_answers_PI_60.json). Reports on the 40 random frames and on all 60.
These frames also guided the prompt, probe and smoothing choices (PROMPT_DECISION.md, merge_full.py), so every number here is optimistic as an estimate of quality on new frames. Output: SCORE_RELEASE.json."""
import collections
import glob
import json
import math
import os
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot_common as P

FIELDS = ("weather", "surface", "visibility", "illumination", "tod")
FOLD = {"glare": "day"}


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)]


def rows_for(want):
    cols = ["clip_id", "frame_idx", "weather", "surface", "visibility", "illumination", "weather_src", "surface_src", "visibility_src", "illumination_src", "sun_band_est", "period_of_day"]
    got = {}
    for p in sorted(glob.glob("F:/strat_out/release/frames/*.parquet")):
        t = pq.read_table(p, columns=cols)
        t = t.filter(pc.is_in(t.column("clip_id"), value_set=__import__("pyarrow").array(sorted({c for c, _ in want}))))
        for r in t.to_pylist():
            if (r["clip_id"], r["frame_idx"]) in want:
                got[(r["clip_id"], r["frame_idx"])] = r
    return got


def main():
    A = json.load(open(HERE + "/pilot_audit_answers_PI_60.json", encoding="utf-8"))["frames"]
    got = rows_for({(a["clip_id"], a["frame_idx"]) for a in A})
    missing = [(a["clip_id"][:8], a["frame_idx"]) for a in A if (a["clip_id"], a["frame_idx"]) not in got]
    res = {"n_frames_audit": len(A), "missing_in_release": missing}
    for scope, sel in (("random40", lambda a: a["stratum"] == "random"), ("all60", lambda a: True)):
        out = {}
        for f in FIELDS:
            n = k = unk = 0
            pi_cnt, conf, by_src = collections.Counter(), collections.Counter(), collections.Counter()
            for a in A:
                if not sel(a) or a[f] in (None, "cant_tell") or (a["clip_id"], a["frame_idx"]) not in got:
                    continue
                r = got[(a["clip_id"], a["frame_idx"])]
                truth = FOLD.get(a[f], a[f])
                if f == "tod":
                    lab = P.ILLUM_TO_TOD.get(r["illumination"], "unknown")
                    src = r["illumination_src"]
                else:
                    lab = r[f]
                    src = r[f + "_src"]
                n += 1
                hit = lab == truth
                k += hit
                unk += lab == "unknown"
                pi_cnt[truth] += 1
                conf[(truth, lab)] += 1
                by_src[(src, bool(hit))] += 1
            maj = pi_cnt.most_common(1)[0] if pi_cnt else (None, 0)
            out[f] = {"correct": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "wilson95": wilson(k, n), "unknown": unk,
                      "always_majority": {"class": maj[0], "correct": maj[1], "pct": round(100 * maj[1] / n, 1) if n else None},
                      "recall": {c: f"{conf[(c, c)]}/{m}" for c, m in sorted(pi_cnt.items())},
                      "by_source": {f"{s}:{'hit' if h else 'miss'}": v for (s, h), v in sorted(by_src.items())}}
        res[scope] = out
    json.dump(res, open(HERE + "/SCORE_RELEASE.json", "w", encoding="utf-8"), indent=1)
    for scope in ("random40", "all60"):
        print(scope)
        for f, v in res[scope].items():
            print(f"  {f}: {v['correct']}/{v['n']} = {v['pct']} % {v['wilson95']}  always {v['always_majority']['class']} {v['always_majority']['pct']} %  unknown {v['unknown']}  recall {v['recall']}")
    print("missing", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
