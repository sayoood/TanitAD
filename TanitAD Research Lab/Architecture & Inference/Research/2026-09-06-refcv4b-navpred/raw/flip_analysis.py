"""os_navflip: does the E13 nav path carry CONTENT, or only PRESENCE?

Usage: python flip_analysis.py <flip_dump_dir> <out.json>

Whole-set margins dilute a token intervention over the windows it never touched.
Every margin here is therefore reported TWICE: over all 4,823 windows, and over
the TREATED SUBSET -- the windows whose fed token actually changed. Restricting
to the treated subset is selection on the INTERVENTION, never on the outcome,
which is legitimate; it is stated so a reader can check.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TANITEVAL_ROOT", "."))
from taniteval import ci as _ci                                   # noqa: E402

DUMP, OUT = sys.argv[1], sys.argv[2]
N_BOOT, SEED = 2000, 0
ARMS = ["os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "os_navflip",
        "os_navpred"]

A, G, E = {}, [], []
for i, f in enumerate(sorted(glob.glob(os.path.join(DUMP, "ep*.npz")))):
    z = np.load(f)
    G.append(z["g"]); E += [i] * len(z["g"])
    for a in ARMS:
        if a in z.files:
            A.setdefault(a, []).append(z[a])
A = {k: np.concatenate(v) for k, v in A.items()}
G = np.concatenate(G); E = np.array(E)

DEC = {}
for f in sorted(glob.glob(os.path.join(DUMP, "decisions", "*.npz"))):
    z = np.load(f)
    for k in z.files:
        if z[k].ndim >= 1 and len(z[k]) == len(z["ws"]):
            DEC.setdefault(k, []).append(z[k])
DEC = {k: np.concatenate(v) for k, v in DEC.items()}

ADE = {a: np.linalg.norm(A[a] - G, axis=-1).mean(axis=-1) for a in A}
N = len(G)
res = {"_is": "os_navflip / os_navpred: PRESENCE vs CONTENT of the E13 nav token",
       "evidence_class": "MEASURED (ours)", "tier": "T1",
       "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                    "n_boot=%d, seed=%d" % (N_BOOT, SEED),
       "n_windows": int(N), "n_episodes": int(E.max() + 1), "dump": DUMP,
       "ade": {}, "paired_full": {}, "paired_treated_subset": {}, "subsets": {}}

for a in ARMS:
    if a in ADE:
        b = _ci.episode_cluster_bootstrap(ADE[a], list(E), n_boot=N_BOOT, seed=SEED)
        res["ade"][a] = {"mean": round(float(ADE[a].mean()), 4),
                         "ci95": [round(float(b["lo"]), 4), round(float(b["hi"]), 4)]}

nc = DEC["nav_cmd"].astype(int)
masks = {}
if "nav_cmd_flip" in DEC:
    masks["os_navflip"] = DEC["nav_cmd_flip"].astype(int) != nc
if "nav_cmd_pred" in DEC:
    masks["os_navpred"] = DEC["nav_cmd_pred"].astype(int) != nc
if "nav_cmd_shuf" in DEC:
    masks["os_navshuf"] = DEC["nav_cmd_shuf"].astype(int) != nc
masks["os_navzero"] = np.ones(N, dtype=bool)      # the null touches every window


def pair(store, name, a, b, m=None):
    if a not in ADE or b not in ADE:
        return
    m = np.ones(N, dtype=bool) if m is None else m
    if not m.any():
        return
    e = [x for x, k in zip(E, m) if k]
    r = _ci.paired_episode_cluster_bootstrap(ADE[a][m], ADE[b][m], e,
                                             n_boot=N_BOOT, seed=SEED)
    lo, hi = float(r["lo"]), float(r["hi"])
    store[name] = {"delta_m": round(float(r["delta"]), 4),
                   "ci95": [round(lo, 4), round(hi, 4)],
                   "separated": bool(lo > 0 or hi < 0),
                   "n_windows": int(m.sum()), "n_episodes": int(r["n_episodes"])}


for arm in ("os_navflip", "os_navpred", "os_navshuf", "os_navzero"):
    pair(res["paired_full"], "%s_minus_os" % arm, arm, "os")
    pair(res["paired_full"], "%s_minus_ha0_ext" % arm, arm, "ha0_ext")
    if arm in masks:
        pair(res["paired_treated_subset"], "%s_minus_os_TREATED" % arm, arm, "os",
             masks[arm])
        res["subsets"]["%s_n_treated" % arm] = int(masks[arm].sum())
# ⭐ THE CLEAN PRESENCE-vs-CONTENT READ: every nav intervention scored on the
# SAME 1,743 windows the FLIP touches (the commanded ones). Comparing `flip`
# (a token that is WRONG on all of them) against `zero` (no token at all) on
# one identical subset is what separates the two hypotheses; whole-set margins
# dilute both over the 3,080 `follow` windows where a flip is a no-op.
if "os_navflip" in masks:
    fm = masks["os_navflip"]
    res["subsets"]["flip_treated_n"] = int(fm.sum())
    for arm in ("os_navflip", "os_navpred", "os_navshuf", "os_navzero"):
        pair(res.setdefault("paired_on_FLIP_TREATED_windows", {}),
             "%s_minus_os" % arm, arm, "os", fm)
    pair(res["paired_on_FLIP_TREATED_windows"], "os_navflip_minus_os_navzero",
         "os_navflip", "os_navzero", fm)

pair(res["paired_full"], "os_navflip_minus_os_navshuf", "os_navflip", "os_navshuf")
pair(res["paired_full"], "os_navflip_minus_os_navpred", "os_navflip", "os_navpred")
pair(res["paired_full"], "os_navflip_minus_os_navzero", "os_navflip", "os_navzero")
pair(res["paired_full"], "os_minus_ha0_ext", "os", "ha0_ext")
pair(res["paired_full"], "os_minus_ha", "os", "ha")

# the presence/content split, on the arms that are on ONE surface
if "os_navzero" in ADE and "os_navflip" in ADE and "os" in ADE:
    z, fl, o = (float(ADE["os_navzero"].mean()), float(ADE["os_navflip"].mean()),
                float(ADE["os"].mean()))
    res["PRESENCE_VS_CONTENT"] = {
        "no_token_minus_true_token_m": round(z - o, 4),
        "WRONG_token_minus_true_token_m": round(fl - o, 4),
        "content_share_of_the_nav_margin":
            round((fl - o) / (z - o), 4) if abs(z - o) > 1e-9 else None,
        "presence_share": round(1 - (fl - o) / (z - o), 4) if abs(z - o) > 1e-9 else None,
        "_is": ("CONTENT is what a DELIBERATELY WRONG token costs (os_navflip - os); "
                "PRESENCE is the remainder of what having no token at all costs "
                "(os_navzero - os). The flip is the sharpest intervention in the "
                "harness: it disagrees with the truth on EVERY commanded window."),
    }

# a control that must read a known value: model-free arms unchanged
res["CONTROL_model_free"] = {
    "ha": round(float(ADE["ha"].mean()), 4), "ha_known": 0.2996,
    "ha0": round(float(ADE["ha0"].mean()), 4), "ha0_known": 0.6723,
    "verdict": "PASS" if (abs(float(ADE["ha"].mean()) - 0.2996) < 5e-4 and
                          abs(float(ADE["ha0"].mean()) - 0.6723) < 5e-4) else "FAIL"}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps({k: res[k] for k in ("ade", "paired_full",
                                      "paired_on_FLIP_TREATED_windows",
                                      "paired_treated_subset", "subsets",
                                      "PRESENCE_VS_CONTENT", "CONTROL_model_free")
                  if k in res}, indent=2))
print("[out]", OUT)
