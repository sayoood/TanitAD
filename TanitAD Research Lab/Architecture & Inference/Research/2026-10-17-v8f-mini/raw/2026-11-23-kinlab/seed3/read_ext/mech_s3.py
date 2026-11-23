"""mech_s3.py (SEED3 build, from mech_ki.py; card PREREG_V8F_SEED3 sha256 75f3b191 section 3, beside reads): at T0 ADE6, paired, the SAME instrument and estimator as mech_ki.py / rh_t0.py
(per-window `ade6` / `eid` of out/t0/eval_<ARM>.npz; taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0).  Tier T0, a diagnostic, never driving performance.
  MECH4 at s2 = KI_s2 - RH_s2 (below_zero = hi < 0), printed with the THREE-seed list of MECH4 points (s0, s1 read from the finished KI read's MECH4_KINPUT.json, s2 new); no bar.
  beside: KI_s2 - B0_s2 and RH_s2 - B0_s2 (the G2 pair), each against 0.1174 and 0.0705.
  the s0 / s1 G-T0 points of RH, RL, RA, RK and KI under 0.1174 (and 0.0705), read from their finished GT0 JSON files: PRINTED ONLY (that data has been seen), decides nothing.
Usage: python -I mech_s3.py <read dir> <out json>      (MECH_T0_DIR / MECH_PRIOR_DIR exist only for the planted check; unset, evals are read from <read dir>/out/t0 and the finished reads from <read dir>/out)."""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

RD = sys.argv[1]
OUTF = sys.argv[2]
sys.path.insert(0, RD + "/tree/taniteval")
sys.path.insert(0, RD + "/tree/stack")
from taniteval.ci import paired_episode_cluster_bootstrap as pb  # noqa: E402
import taniteval.ci as ci  # noqa: E402

T0 = os.environ.get("MECH_T0_DIR") or (RD + "/out/t0")
PRIOR = os.environ.get("MECH_PRIOR_DIR") or (RD + "/out")
BAR4, BAR1 = 0.1174, 0.0705
GT0_FILES = {"RH": "read/GT0_ROLLHIST.json", "RL": "read_rl/GT0_ROLLHIST_RL.json", "RA": "read_ra/GT0_ROLLHIST_RA.json", "RK": "read_rk/GT0_ROLLHIST_RK.json", "KI": "read_ki/GT0_ROLLHIST_KI.json"}


def load(arm):
    z = np.load(f"{T0}/eval_{arm}.npz", allow_pickle=True)
    return z["ade6"], z["eid"]


def pair(a, b):
    A, ea = load(a)
    B, eb = load(b)
    assert len(A) == len(B) == 2399 and np.array_equal(ea, eb), (a, b)
    r = pb(A, B, ea, n_boot=2000, seed=0)
    m = {"A": a, "B": b, "point_m": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "mean_A": float(np.mean(A)), "mean_B": float(np.mean(B)),
         "n_windows": int(len(A)), "n_episodes": int(len(np.unique(ea)))}
    m["upper_le_0p1174"] = bool(m["hi"] <= BAR4)
    m["upper_le_0p0705"] = bool(m["hi"] <= BAR1)
    return m


def fmt(m):
    return f"{m['point_m']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]"


out = {"read": "SEED3 beside reads at T0 ADE6 (PREREG_V8F_SEED3 section 3): MECH4 at s2 = KI_s2 - RH_s2 with the three-seed list; KI_s2 - B0_s2 and RH_s2 - B0_s2 against both bars; the s0/s1 G-T0 points printed only",
       "tier": "T0", "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0 ('another draw of EPISODES' only; the seeds answer 'another TRAINING run')",
       "ci_module": os.path.abspath(ci.__file__).replace("\\", "/"), "bars_m": {"F_T0_4seed": BAR4, "F_T0_inherited": BAR1}}
m4 = pair("KI_s2", "RH_s2")
m4["below_zero"] = bool(m4["hi"] < 0)
out["MECH4_s2"] = m4
out["beside"] = {"KI_s2_minus_B0_s2": pair("KI_s2", "B0_s2"), "RH_s2_minus_B0_s2": pair("RH_s2", "B0_s2")}
three = []
pf = f"{PRIOR}/read_ki/MECH4_KINPUT.json"
if os.path.exists(pf):
    pr = json.load(open(pf, encoding="utf-8"))
    for s in ("0", "1"):
        v = pr["per_seed"][s]
        three.append({"seed": int(s), "point_m": v["point_m"], "lo": v["lo"], "hi": v["hi"], "below_zero": bool(v["hi"] < 0), "source": "finished KI read, MECH4_KINPUT.json (read only)"})
else:
    three.append({"seed": "0,1", "status": "NOT FOUND", "path": pf})
three.append({"seed": 2, "point_m": m4["point_m"], "lo": m4["lo"], "hi": m4["hi"], "below_zero": m4["below_zero"], "source": "this read"})
out["MECH4_three_seed_list"] = {"points": three, "n_below_zero": int(sum(1 for t in three if t.get("below_zero") is True)), "bar": "none (printed, decides nothing under the SEED3 card)"}
seen = {}
for arm, rel in GT0_FILES.items():
    f = f"{PRIOR}/{rel}"
    if not os.path.exists(f):
        seen[arm] = {"status": "NOT FOUND", "path": f}
        continue
    g = json.load(open(f, encoding="utf-8"))
    seen[arm] = {"source": rel, "per_seed": {s: {"point_m": v["point_m"], "lo": v["lo"], "hi": v["hi"], "upper_le_0p1174": bool(v["hi"] <= BAR4), "upper_le_0p0705": bool(v["hi"] <= BAR1)}
                                           for s, v in (g.get("per_seed") or {}).items() if isinstance(v, dict) and "hi" in v}}
out["GT0_s0_s1_under_0p1174_PRINTED_ONLY"] = {"note": "that data has already been seen: printed only, decides nothing here", "arms": seen}
json.dump(out, open(OUTF, "w", encoding="utf-8"), indent=1)
print("MECH4 s2 KI-RH", fmt(m4), "below_zero", m4["below_zero"], "| three-seed list", [(t["seed"], round(t["point_m"], 4), t["below_zero"]) for t in three if "point_m" in t])
for k, v in out["beside"].items():
    print(k, fmt(v), "upper<=0.1174", v["upper_le_0p1174"], "upper<=0.0705", v["upper_le_0p0705"])
for arm, v in seen.items():
    print("G-T0 s0/s1", arm, {s: (round(x["point_m"], 4), round(x["hi"], 4), x["upper_le_0p1174"]) for s, x in (v.get("per_seed") or {}).items()} if "per_seed" in v else v)
