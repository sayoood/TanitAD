"""DESCRIPTIVE probe after the T1-HEADS read (post-data, decides nothing): does the T head's tick>=1 F1 drop depend on how far the matched
label row r'_T sits from the row whose LOGGED cameras the model saw (j - age_t)? T1-kin replays the logged cameras at the clock row, so a car
that lags or leads the human sees a view from a different place than its own. Reuses t1hd_read's own matching and cells unchanged.
python -I probe_lag.py <run_real.sh args...>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1hd_read as T  # noqa: E402

a = T.parse(sys.argv[1:])
T.setup_imports(a.tool_root)
ctx = T.Ctx(a)


def macro_f1(true, pred):
    m = true >= 0
    t, p = true[m], pred[m]
    f = []
    for c in np.unique(t):
        tp = int(((p == c) & (t == c)).sum()); fp = int(((p == c) & (t != c)).sum()); fn = int(((p != c) & (t == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return (float(np.mean(f)) if f else float("nan")), int(m.sum())


out = {"note": "DESCRIPTIVE, post-data; plain macro-F1 over the classes present (not the T0 reducer); decides nothing", "runs": {}}
BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 999)]
for tag in ("HACC_s0_m10_causal_heads", "HACC_s1_m10_causal_heads"):
    tt = T.load_ticks(os.path.join(a.run_dir, "ticks", f"ticks_{tag}.npz"))
    R = T.analyze_run(ctx, tt, a, tag, 0, dump_dir=None, want_rows=False)
    seen = tt["j"] - tt["age_t"]                                    # the row whose logged cameras / inputs T was computed from
    lag = R["rT"] - seen                                            # > 0: the car is ahead of the clock row's place ... < 0: behind (crawl)
    res = {"lag_rows_ticks_ge1": {"p10": float(np.percentile(lag[:, 1:], 10)), "median": float(np.median(lag[:, 1:])),
                                  "p90": float(np.percentile(lag[:, 1:], 90))}}
    for hd in ("lat", "lon"):
        tr, pr = R["cells"][hd]["true"], R["cells"][hd]["pred"]
        f0, n0 = macro_f1(tr[:, 0], pr[:, 0])
        rows = {"tick0": {"f1": f0, "n": n0}}
        L = np.abs(lag[:, 1:]); t1, p1 = tr[:, 1:], pr[:, 1:]
        for lo, hi in BINS:
            m = (L >= lo) & (L <= hi)
            f, n = macro_f1(np.where(m, t1, -1), p1)
            rows[f"|lag| {lo}-{hi}"] = {"f1": f, "n": n}
        res[hd] = rows
    out["runs"][tag] = res
    print(tag, json.dumps(res))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROBE_LAG.json"), "w"), indent=1)
