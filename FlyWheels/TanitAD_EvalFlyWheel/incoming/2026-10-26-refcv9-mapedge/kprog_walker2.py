"""kprog_walker2.py -- PREREG_MAPPATH AMENDMENT 2 K-PROG split: walker vs float64 transform output; stored == float32(float64) exactly."""
import json, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l1/code")
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapedge")
import l1_fit_score as F  # noqa: E402
from kprog_walker import walk, arc_nat, fan0, fan1, sel, n, N, S  # noqa: E402  (re-runs walker v1 on import: same seeded sample, harmless)
AL, PH = F.AL, F.PH
rng = np.random.default_rng(0)
lens = np.hypot(*np.diff(np.concatenate([np.zeros((n, N, 1, 2)), fan0], 2), axis=2).transpose(3, 0, 1, 2)).sum(-1)
ok = np.argwhere(lens > 1e-6)
pick = ok[rng.choice(len(ok), 5000, replace=False)]
m64 = m32 = mut = 0.0
eq32 = True
for (w, c) in pick:
    k = int(rng.integers(0, S))
    snat = AL.arc(fan0[w, sel[w]][None])[0]
    o64 = PH.at_progress(fan0[w, c][None], snat[None])[0].astype(np.float64)
    p = walk(fan0[w, c], snat[k])
    m64 = max(m64, abs(p[0] - o64[k, 0]), abs(p[1] - o64[k, 1]))
    eq32 &= bool(np.array_equal(o64[k].astype(np.float32), fan1[w, c, k].astype(np.float32)))
    q = walk(fan0[w, c], snat[k] + 0.05)
    mut = max(mut, abs(q[0] - o64[k, 0]), abs(q[1] - o64[k, 1]))
res = {"i_walker_vs_float64_max_abs_m": m64, "i_pass": bool(m64 <= 1e-6), "ii_stored_equals_float32_cast_exactly": eq32,
       "iii_mutation_max_abs_m": mut, "iii_red": bool(mut > 1e-6), "PASS": bool(m64 <= 1e-6 and eq32 and mut > 1e-6), "n": 5000}
json.dump(res, open("result/MAPPATH_KPROG2.json", "w"), indent=1)
print(json.dumps(res))
