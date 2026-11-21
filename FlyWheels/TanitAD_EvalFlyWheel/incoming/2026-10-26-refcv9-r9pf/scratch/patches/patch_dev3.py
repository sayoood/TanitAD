import json
from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/deviations_r9pf.json")
d = json.loads(p.read_text(encoding="utf-8"))
d = [x for x in d if x["id"] not in ("D20", "D21")]
d += [
 {"id": "D20", "text": "K-OFF: `wp_seq` differs ON vs OFF on 515/515 calls at both seeds because it IS the emitted plan (refc_v3.py:2742 out['wp_seq'] = traj, "
   "set after the R9 re-time at :2723). The compare first read it as a non-R9 output (K-OFF FAIL), then classified it as a plan alias from data -- "
   "digest(wp_seq) == digest(traj) on every call of both runs -- and checks it like traj (ON traj_preretime == OFF wp_seq: 0 / 515 differ). "
   "No other output differs. K-OFF then PASSES at both seeds. Disclosed because the classification was made after seeing the first result."},
 {"id": "D21", "text": "K-EQ FAILS as written at both seeds: decision agreement 100 % (515/515), but max |d| 3.05e-5 / 4.58e-5 m exceeds 2 ulp_float32 of the "
   "window's own max |coordinate| on 28 / 24 of 515 windows (max ratio 13). The stricter reading of the coordinate scope was used (the brief: "
   "conservative on ambiguity). BESIDE, not registered: the residual survives the forward's OWN s_prior (17 / 19 windows over) and a tolerance "
   "that includes the prior's coordinates (26 / 22 over), so it is not the prior. Unresolved: GPU-vs-CPU arithmetic in the head is the "
   "candidate (the in-forward head runs on CUDA, the offline one on CPU); the discriminating test is the same roll on CPU, which was not run."},
]
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); print([x["id"] for x in d][-4:])
