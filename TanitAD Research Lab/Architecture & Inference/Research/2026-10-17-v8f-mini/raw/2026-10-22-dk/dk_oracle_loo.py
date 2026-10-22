"""Which group of T's code carries the early-closing fix? Leave-one-out on the banked GT-code oracles, 2 s distance keeping.

  python dk_oracle_loo.py <v8f-mini package dir> <out.json>        (PYTHONPATH must reach stack and taniteval)

EXPLORATORY, post hoc; follows dk_oracle_read.py, whose committed rule read UPPER for M2b (the GT code removes M2b's 2 s excess of
< 5 s TTC windows over P10) and refuted stoplead as the carrier. It guides the next lever and decides nothing.
The reading rule below was written BEFORE this script was run, and the output carries it verbatim.

ARMS per checkpoint c in {M2b_s0, M2b_s1, M1_s0, M1_s1}: c_orc (the whole GT code, raw/2026-10-20-tac/eval/eval_<c>-tacorc.npz) and, for each
group g in goal latdec londec latcons lonspeed stoplead, c_all-g (GT everywhere EXCEPT g, T's own code there;
raw/2026-10-21-tacgrp2/eval/eval_<c>-tacorc-all-<g>.npz). All T0 (they read GT). Leave-one-out is used because O was trained on whole
codes only, so the mix closest to a whole GT code is the one O can read (v8f_tac_groups2_read.py docstring).

READOUT: S5 = share of LEAD windows with min-TTC < 5 s at 2 s (the banked K=10 block). loss_g = S5(c_all-g) - S5(c_orc), paired
episode-cluster bootstrap, B 2,000 (positive = putting T's own g back brings the tail back).

READING RULE (M2b decides; M1 is reported beside it):
  g CARRIES the fix iff loss_g > 0, separated, at both M2b seeds.
  NONE if no group carries it (the fix is spread over groups, or needs the whole code together).
"""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

PKG, OUT = sys.argv[1], sys.argv[2]
RULE = __doc__[__doc__.index("READING RULE"):].strip()
spec = importlib.util.spec_from_file_location("v8f_dk_read", os.path.join(PKG, "code", "v8f_dk_read.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from taniteval import ci  # noqa: E402

GROUPS = ("goal", "latdec", "londec", "latcons", "lonspeed", "stoplead")
CK = ("M2b_s0", "M2b_s1", "M1_s0", "M1_s1")
RAW = os.path.join(PKG, "raw")
a = SimpleNamespace(gt_npz=None, eval_grid=os.path.join(RAW, "eval_grid_windows.json"), eval_kin=os.path.join(RAW, "eval141_kin.npz"),
                    eval_join=os.path.join(RAW, "join_eval141.npz"))
G = R.ground_truth(a)
preds, saved, prov = {}, {}, {}


def add(name, p):
    z = np.load(p)
    assert np.array_equal(z["eid"], G["eid"]), f"{p} is not on the eval grid"
    preds[name] = z["pred"].astype(np.float64)
    saved[name] = np.asarray(z["ade6"], np.float64)
    prov[name] = {"path": os.path.relpath(p, PKG).replace("\\", "/"), "sha256": R.sha256(p)}


for c in CK:
    add(c + "_orc", os.path.join(RAW, "2026-10-20-tac", "eval", f"eval_{c}-tacorc.npz"))
    for g in GROUPS:
        add(f"{c}_all-{g}", os.path.join(RAW, "2026-10-21-tacgrp2", "eval", f"eval_{c}-tacorc-all-{g}.npz"))
blk = R.load_block(str(R.LEAD_BLOCK_DEFAULT))
res = R.read_dk(preds=preds, gt=G["gt"], eid=G["eid"], clip=G["clip"], frame=G["frame"], v0=G["v0"], block=blk, k=blk["_k"], dt=G["dt"],
                n_boot=2000, seed=0, saved_ade6=saved, pairs=[])
pw = res["perwindow"]
eid, lead = pw["eid"], pw["is_lead"]
out = {"schema": "v8f-dk-oracle-loo-v1", "exploratory": True, "reading_rule_written_before_the_run": RULE, "tier": "T0 (every arm reads GT code)",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over episodes; blind to training-seed variance (H-ESTIM-SEED-1)",
       "horizon_s": 2.0, "lead_block_sha256": R.sha256(str(R.LEAD_BLOCK_DEFAULT)), "contract": res["read"]["human_reference"]["kinematic_contract"]["status"],
       "inputs": prov, "loss": {}}
for c in CK:
    for g in GROUPS:
        ta, tb = pw[f"{c}_all-{g}__min_ttc_s"], pw[f"{c}_orc__min_ttc_s"]
        m = lead & np.isfinite(ta) & np.isfinite(tb)
        q = ci.paired_episode_cluster_bootstrap((ta[m] < 5).astype(float), (tb[m] < 5).astype(float), eid[m], n_boot=2000, seed=0)
        qt = ci.paired_episode_cluster_bootstrap(ta[m], tb[m], eid[m], n_boot=2000, seed=0)
        out["loss"][f"{c}:{g}"] = {"n_used": int(m.sum()), "S5_loss": {"delta": q["delta"], "lo": q["lo"], "hi": q["hi"], "sep": bool(q["lo"] > 0 or q["hi"] < 0)},
                                   "ttc_mean_delta_s": {"delta": qt["delta"], "lo": qt["lo"], "hi": qt["hi"], "sep": bool(qt["lo"] > 0 or qt["hi"] < 0)}}


def carriers(arm):
    return [g for g in GROUPS if all(out["loss"][f"{arm}_s{s}:{g}"]["S5_loss"]["sep"] and out["loss"][f"{arm}_s{s}:{g}"]["S5_loss"]["delta"] > 0 for s in (0, 1))]


out["verdict"] = {"M2b (decides)": carriers("M2b") or "NONE", "M1 (reported)": carriers("M1") or "NONE"}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(R._clean(out), f, indent=1)
for k, v in out["loss"].items():
    s, t = v["S5_loss"], v["ttc_mean_delta_s"]
    print(f"{k:20s} n {v['n_used']}  S5 loss {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] {'sep' if s['sep'] else 'ns'}"
          f"   TTC {t['delta']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}] {'sep' if t['sep'] else 'ns'}")
print("VERDICT", out["verdict"])
