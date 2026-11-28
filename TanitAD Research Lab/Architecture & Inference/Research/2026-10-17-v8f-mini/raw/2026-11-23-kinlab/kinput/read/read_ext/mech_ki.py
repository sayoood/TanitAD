"""mech_ki.py (KINPUT build, from mech_rk.py): the KINPUT card's MECH4 read (PREREG_V8F_KINPUT sha256 05e786a8, section 3): KI - RH at T0 ADE6, paired, per seed.
MECH4-YES when the paired interval sits below 0 at both seeds (the logged-base input against RH's own-base input on the SAME bank, dose, label and init: a one-lever contrast).
Same instrument and estimator as rh_t0.py's G-T0 and mech_ra.py / mech_rl.py (per-window `ade6` and `eid` of out/t0/eval_<ARM>.npz; taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0).
Tier T0 (a WM / tactical diagnostic, never driving performance).  Beside: KI - B0 (the G-T0 pair, re-read), KI - RK, RH - B0.
Usage: python -I mech_ki.py <read dir> <out json>
Control (planted check, see FACTS_READ_KI.md): MECH_T0_DIR points at synthetic eval_<ARM>_s<k>.npz files.
MECH_T0_DIR exists only for that control; unset, the evals are read from <read dir>/out/t0."""
import json
import os
import sys

sys.dont_write_bytecode = True   # the taniteval import below must not write __pycache__ into the read tree (dry_check blobcheck counts extra files)

import numpy as np

RD = sys.argv[1]
OUTF = sys.argv[2]
sys.path.insert(0, RD + "/tree/taniteval")
sys.path.insert(0, RD + "/tree/stack")
from taniteval.ci import paired_episode_cluster_bootstrap as pb  # noqa: E402
import taniteval.ci as ci  # noqa: E402

T0 = os.environ.get("MECH_T0_DIR") or (RD + "/out/t0")
TREATED, CONTRAST = "KI", "RH"          # the lever arm and the arm it differs from by exactly one lever (the base input: logged against own)


def load(arm):
    z = np.load(f"{T0}/eval_{arm}.npz", allow_pickle=True)
    return z["ade6"], z["eid"]


def pair(a, b):
    A, ea = load(a)
    B, eb = load(b)
    assert len(A) == len(B) == 2399 and np.array_equal(ea, eb), (a, b)
    r = pb(A, B, ea, n_boot=2000, seed=0)
    return {"A": a, "B": b, "point_m": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
            "mean_A": float(np.mean(A)), "mean_B": float(np.mean(B)), "n_windows": int(len(A)), "n_episodes": int(len(np.unique(ea)))}


out = {"read": f"MECH4 (PREREG_V8F_KINPUT section 3): {TREATED} - {CONTRAST} at T0 ADE6, paired, per seed; MECH4-YES when the interval sits below 0 at both seeds",
       "tier": "T0", "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0 (answers 'another draw of EPISODES'; the two seeds answer 'another TRAINING run')",
       "ci_module": os.path.abspath(ci.__file__).replace("\\", "/"), "per_seed": {}, "beside": {}}
below = []
for s in (0, 1):
    m = pair(f"{TREATED}_s{s}", f"{CONTRAST}_s{s}")
    m["below_zero"] = bool(m["hi"] < 0)
    below.append(m["below_zero"])
    out["per_seed"][str(s)] = m
    out["beside"][f"{TREATED}_minus_B0_s{s}"] = pair(f"{TREATED}_s{s}", f"B0_s{s}")
    out["beside"][f"{TREATED}_minus_RK_s{s}"] = pair(f"{TREATED}_s{s}", f"RK_s{s}")
    out["beside"][f"RH_minus_B0_s{s}"] = pair(f"RH_s{s}", f"B0_s{s}")
out["MECH4"] = "MECH4-YES" if all(below) else "MECH4-NO"
json.dump(out, open(OUTF, "w", encoding="utf-8"), indent=1)
print(out["MECH4"])


def fmt(m):
    return f"{m['point_m']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]"


for s in ("0", "1"):
    b = out["beside"]
    print(f"s{s} {TREATED}-{CONTRAST} {fmt(out['per_seed'][s])}  {TREATED}-B0 {fmt(b[f'{TREATED}_minus_B0_s{s}'])}  {TREATED}-RK {fmt(b[f'{TREATED}_minus_RK_s{s}'])}  RH-B0 {fmt(b[f'RH_minus_B0_s{s}'])}")
