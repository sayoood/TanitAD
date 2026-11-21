"""D9 regression-arm diagnosis (Master Mind 2026-10-16). Replays build_metric_cache.exactness EXACTLY (same tokens, rng seed 0,
4 sets per token, same proposals, same 5 m rear-axle shift) and records per set: whether identity held, whether the regression
was detected, and the fresh scores, so the 22 undetected sets can be classified (saturated / all-zero / genuinely blind).
Read-only: writes one JSON. Run from the P6 tree's refe/ dir with the chain's env, nice 19, OMP 1."""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import build_metric_cache as B  # noqa: E402
import navsim_pdm_targets as PT  # noqa: E402

cache, db_root, map_root, toks, out = sys.argv[1:6]
items = [tuple(t.split(":", 1)) for t in toks.split(",")]
rng = np.random.default_rng(0)
rows = []
for log, tok in items:
    sc = B._scenarios(db_root, map_root, log, {tok}).get(tok)
    if sc is None:
        rows.append({"log": log, "token": tok, "scenario": None})
        continue
    fresh = PT.metric_cache_parts(sc)
    cached = B.OL.load_parts(B.OL.parts_path(cache, log, tok))
    for s in range(4):
        props = np.stack([np.stack([np.linspace(0.5, 30.0, 20) * (0.8 + 0.4 * rng.random()), rng.normal(0, 1.0, 20).cumsum() * 0.1, np.zeros(20)], -1)
                          for _ in range(8)])
        plans8 = PT.executed_plans8(props, repair=True)
        a, b = PT.score_plans(fresh, plans8), PT.score_plans(cached, plans8)
        bad = list(cached)
        ego = copy.deepcopy(bad[1])
        x0 = float(ego.rear_axle.x)
        ego.rear_axle.x += 5.0
        x1 = float(ego.rear_axle.x)
        bad[1] = ego
        try:
            c = PT.score_plans(tuple(bad), plans8)
            err = None
        except Exception as e:  # noqa: BLE001
            c, err = None, repr(e)[:200]
        r = {"log": log, "token": tok, "set": s, "identity_equal": all(np.array_equal(a[k], b[k]) for k in a),
             "regression_equal": (c is not None) and all(np.array_equal(a[k], c[k]) for k in a), "regression_error": err,
             "shift_applied": x1 - x0, "ego_center_x_moved": None,
             "fresh": {k: [round(float(v), 6) for v in a[k]] for k in a}}
        try:
            r["ego_center_x_moved"] = float(ego.center.x) - float(cached[1].center.x)
        except Exception:  # noqa: BLE001
            pass
        if c is not None:
            r["keys_differing"] = [k for k in a if not np.array_equal(a[k], c[k])]
        rows.append(r)
s = [r for r in rows if "set" in r]
und = [r for r in s if r["regression_equal"]]
summary = {"n_sets": len(s), "identity_equal": sum(r["identity_equal"] for r in s), "regression_undetected": len(und),
           "undetected_all_pdms_zero": sum(all(v == 0 for v in r["fresh"]["pdms"]) for r in und),
           "undetected_all_dac_zero": sum(all(v == 0 for v in r["fresh"]["dac"]) for r in und),
           "detected_all_pdms_zero": sum(all(v == 0 for v in r["fresh"]["pdms"]) for r in s if not r["regression_equal"]),
           "undetected_tokens": sorted({r["token"] for r in und}),
           "tokens_with_no_detected_set": sorted({r["token"] for r in s} - {r["token"] for r in s if not r["regression_equal"]})}
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary))
print("ZZD9DIAG_DONE")
