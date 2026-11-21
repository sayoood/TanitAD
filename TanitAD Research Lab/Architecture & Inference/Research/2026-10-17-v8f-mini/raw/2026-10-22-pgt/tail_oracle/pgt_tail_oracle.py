"""After PGT removed the latcons leak, does T's code still move the low-TTC tail?

  python pgt_tail_oracle.py <v8f-mini package dir at the PGT tip> <box2_ship logs dir> <out.json>   (PYTHONPATH must reach stack and taniteval)

EXPLORATORY, post hoc, ONE training seed (PGT stopped at seed 0); it points at the next lever and decides nothing.
The reading rule below was written BEFORE this script was run, and the output carries it verbatim.

WHY (PREREG_V8F_PGT seed 0, pgt_read_s0.json): the latcons leak LAMBDA fell 0.268 -> 0.046 m (separated), but the S5 tail did not move
(+0.73 pp, ns), and the GT-code oracle ADE6 ROSE 0.971 -> 1.293 m (separated): O leans less on the code. RESULT_V8F_DK.md 9.4-9.5 found
that for M2b the GT code closes the tail and that latcons carries that fix. If the code no longer moves M2b-pgt's tail, the tail is O's own.

ARMS (the same oracle sources pgt_read_s0.json used; all on the eval-141 grid):
  M2b_s0          v8f_mini/eval/eval_M2b_s0.npz                         own code (T1-class open loop)
  M2b_s0_orc      v8f_tacgrp2/eval/eval_M2b_s0-tacorc-all-goal.npz       whole GT code (T0). M2b's -tacorc predates tac_code (WC Amendment 1)
  M2b_s0_alll     v8f_tacgrp2/eval/eval_M2b_s0-tacorc-all-latcons.npz    GT everywhere EXCEPT latcons (T0)
  M2b-pgt_s0      v8f_pgt/eval/eval_M2b-pgt_s0.npz                       own code (T1-class open loop)
  M2b-pgt_s0_orc  v8f_pgt/eval/eval_M2b-pgt_s0-tacorc.npz                whole GT code (T0)
  M2b-pgt_s0_alll v8f_pgt/eval/eval_M2b-pgt_s0-tacorc-all-latcons.npz    GT everywhere EXCEPT latcons (T0)
  beside: M2b_s0_orcold = v8f_cz/eval/eval_M2b_s0-tacorc.npz, the oracle RESULT_V8F_DK 9.4 used (reproduction check only).

READOUT: S5 = share of LEAD windows with min-TTC < 5 s at the 2 s horizon (K=10 lead block), paired episode-cluster bootstrap, B 2,000.

READING RULE (M2b-pgt_s0 decides; M2b_s0 is the reference):
  CODE-STILL-MOVES  d(M2b-pgt_s0_orc - M2b-pgt_s0) S5 < 0 separated
                    => T's code still reaches the tail after PGT; a more accurate T (GRID) has a route to it.
  O-OWN             d(M2b-pgt_s0_orc - M2b-pgt_s0) S5 not separated  AND  d(M2b-pgt_s0_orc - P10) S5 > 0 separated
                    => with the GT code O still closes early: the tail is O's own (its longitudinal head or input), not T's code.
  otherwise MIXED.
  LATCONS (read beside, not a verdict) d(*_alll - *_orc) S5: the tail that comes back when T's own latcons is put back.
"""
import hashlib
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

PKG, LG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RULE = __doc__[__doc__.index("READING RULE"):].strip()
spec = importlib.util.spec_from_file_location("v8f_dk_read", os.path.join(PKG, "code", "v8f_dk_read.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from taniteval import ci  # noqa: E402

RAW = os.path.join(PKG, "raw")
a = SimpleNamespace(gt_npz=None, eval_grid=os.path.join(RAW, "eval_grid_windows.json"), eval_kin=os.path.join(RAW, "eval141_kin.npz"),
                    eval_join=os.path.join(RAW, "join_eval141.npz"))
G = R.ground_truth(a)
SRC = {"M2b_s0": "v8f_mini/eval/eval_M2b_s0.npz",
       "M2b_s0_orc": "v8f_tacgrp2/eval/eval_M2b_s0-tacorc-all-goal.npz",
       "M2b_s0_alll": "v8f_tacgrp2/eval/eval_M2b_s0-tacorc-all-latcons.npz",
       "M2b_s0_orcold": "v8f_cz/eval/eval_M2b_s0-tacorc.npz",
       "M2b-pgt_s0": "v8f_pgt/eval/eval_M2b-pgt_s0.npz",
       "M2b-pgt_s0_orc": "v8f_pgt/eval/eval_M2b-pgt_s0-tacorc.npz",
       "M2b-pgt_s0_alll": "v8f_pgt/eval/eval_M2b-pgt_s0-tacorc-all-latcons.npz"}
preds, saved, prov = {}, {}, {}
for nm, rel in SRC.items():
    p = os.path.join(LG, rel)
    z = np.load(p)
    assert np.array_equal(z["eid"], G["eid"]), f"{p} is not on the eval grid"
    preds[nm] = z["pred"].astype(np.float64)
    saved[nm] = np.asarray(z["ade6"], np.float64)
    prov[nm] = {"path": "box2_ship/logs/" + rel, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
p10 = str(R.P10_DEFAULT)
preds["P10"] = np.load(p10).astype(np.float64)
prov["P10"] = {"path": "P10_DEFAULT", "sha256": R.sha256(p10)}
floor = np.load(os.path.join(LG, "v8f_mini", "eval", "floors_perwindow.npz"))["P10_refit__ade6"]

blk = R.load_block(str(R.LEAD_BLOCK_DEFAULT))
res = R.read_dk(preds=preds, gt=G["gt"], eid=G["eid"], clip=G["clip"], frame=G["frame"], v0=G["v0"], block=blk, k=blk["_k"], dt=G["dt"],
                n_boot=2000, seed=0, saved_ade6=saved, p10_floor_ade6=floor, pairs=[])
pw, eid, lead = res["perwindow"], res["perwindow"]["eid"], res["perwindow"]["is_lead"]
out = {"schema": "v8f-pgt-tail-oracle-v1", "exploratory": True, "one_training_seed": True, "reading_rule_written_before_the_run": RULE,
       "tier": {"own": "T1-class open loop", "_orc / _alll / _orcold": "T0 (O reads GT code)"},
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over episodes; blind to training-seed variance (H-ESTIM-SEED-1)",
       "inputs": prov, "lead_block_sha256": R.sha256(str(R.LEAD_BLOCK_DEFAULT)), "window_states": res["read"]["join"]["window_states"],
       "arms": {}, "pairs": {}}
for nm in preds:
    t = pw[f"{nm}__min_ttc_s"]
    m = lead & np.isfinite(t)
    out["arms"][nm] = {"n": int(m.sum()), "S5": float((t[m] < 5).mean()), "ttc_mean_s": float(t[m].mean())}
for c in ("M2b_s0", "M2b-pgt_s0"):
    for x, y in ((c + "_orc", c), (c + "_orc", "P10"), (c, "P10"), (c + "_alll", c + "_orc")):
        ta, tb = pw[f"{x}__min_ttc_s"], pw[f"{y}__min_ttc_s"]
        m = lead & np.isfinite(ta) & np.isfinite(tb)
        q = ci.paired_episode_cluster_bootstrap((ta[m] < 5).astype(float), (tb[m] < 5).astype(float), eid[m], n_boot=2000, seed=0)
        qt = ci.paired_episode_cluster_bootstrap(ta[m], tb[m], eid[m], n_boot=2000, seed=0)
        out["pairs"][f"{x}__minus__{y}"] = {"n_used": int(m.sum()),
                                            "S5": {"delta": q["delta"], "lo": q["lo"], "hi": q["hi"], "sep": bool(q["lo"] > 0 or q["hi"] < 0)},
                                            "ttc_mean_s": {"delta": qt["delta"], "lo": qt["lo"], "hi": qt["hi"], "sep": bool(qt["lo"] > 0 or qt["hi"] < 0)}}
ta, tb = pw["M2b_s0_orcold__min_ttc_s"], pw["M2b_s0__min_ttc_s"]
m = lead & np.isfinite(ta) & np.isfinite(tb)
q = ci.paired_episode_cluster_bootstrap((ta[m] < 5).astype(float), (tb[m] < 5).astype(float), eid[m], n_boot=2000, seed=0)
out["reproduction_dk_9_4"] = {"M2b_s0_orcold__minus__M2b_s0_S5": {"delta": q["delta"], "lo": q["lo"], "hi": q["hi"]}, "banked": "-2.6 pp [-5.1, -0.4]"}

d = out["pairs"]["M2b-pgt_s0_orc__minus__M2b-pgt_s0"]["S5"]
dp = out["pairs"]["M2b-pgt_s0_orc__minus__P10"]["S5"]
if d["sep"] and d["delta"] < 0:
    v = "CODE-STILL-MOVES"
elif (not d["sep"]) and dp["sep"] and dp["delta"] > 0:
    v = "O-OWN"
else:
    v = "MIXED"
out["verdict"] = v
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(R._clean(out), f, indent=1)
print("states", out["window_states"])
for nm, x in out["arms"].items():
    print(f"  {nm:16s} n {x['n']:4d}  S5 {x['S5']:.3f}  TTC {x['ttc_mean_s']:.2f}")
for nm, x in out["pairs"].items():
    s, t = x["S5"], x["ttc_mean_s"]
    print(f"  {nm:34s} S5 {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] {'sep' if s['sep'] else 'ns'}"
          f"   TTC {t['delta']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}] {'sep' if t['sep'] else 'ns'}")
print("reproduction (DK 9.4)", out["reproduction_dk_9_4"])
print("VERDICT", v)
