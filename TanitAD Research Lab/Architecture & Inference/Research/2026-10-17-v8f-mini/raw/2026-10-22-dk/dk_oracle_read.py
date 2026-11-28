"""Does v8F-mini's early closing on the lead come from T's code or from O? Distance keeping on the banked GT-code oracle evals.

  python dk_oracle_read.py <v8f-mini package dir> <out.json>        (PYTHONPATH must reach stack and taniteval)

EXPLORATORY, post hoc (the DK read of 2026-10-22 found the effect first); it guides the next lever and decides nothing. The reading rule
below was written BEFORE this script was run, and the output carries it verbatim.

THE EFFECT BEING EXPLAINED (RESULT_V8F_DK.md 9): at 2 s the share of LEAD windows whose min-TTC is < 5 s is higher for the learned arms than
for P10 (M2b - P10 +4.4 / +2.5 pp, separated at both seeds), and M2b - M1 is flat.

ARMS, per checkpoint c in {M1_s0, M1_s1, M2b_s0, M2b_s1}, all on the eval-141 grid, all banked:
  c              eval_<c>.npz                       T's own code (T1-class open loop)                 raw/full/eval/
  c_orc          eval_<c>-tacorc.npz                O driven by the WHOLE GT code (T0: reads GT)      raw/2026-10-20-tac/eval/
  c_alls         eval_<c>-tacorc-all-stoplead.npz   GT everywhere EXCEPT stoplead, T's own there (T0) raw/2026-10-21-tacgrp2/eval/
The single-group oracle (-tacorc-stoplead) is NOT used: O was trained on whole codes only, and a single GT group beside T's code is a mix it
never saw (v8f_tac_groups2_read.py docstring, Amendment 1's finding that single groups HURT alone).

READOUT: S5 = share of LEAD windows with min-TTC < 5 s at the 2 s horizon (the banked K=10 block). Paired episode-cluster bootstrap, B 2,000.
Also reported, not read: S10 (< 10 s), mean min-TTC, mean headway, and the same at 6 s.

READING RULE (M2b decides; M1 is reported beside it; "both seeds" = s0 and s1 each):
  UPPER   d(c_orc - c) < 0 separated at both seeds  AND  d(c_orc - P10) not separated-positive at either seed
          => with the GT code O no longer closes early: the early closing comes from T's code.
  LOWER   d(c_orc - P10) > 0 separated at both seeds
          => O closes early even with the GT code: the lever is O (or O's own input), not T's accuracy.
  STOPLEAD (read only under UPPER) d(c_alls - c_orc) > 0 separated at both seeds
          => T's stoplead group (stop_x/y, lead_gap_m, lead_tg_s, lead_gap_min_m, lead_tg_min_s; v8f_mini.py:63) is needed for it.
  otherwise MIXED.
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

RAW = os.path.join(PKG, "raw")
a = SimpleNamespace(gt_npz=None, eval_grid=os.path.join(RAW, "eval_grid_windows.json"), eval_kin=os.path.join(RAW, "eval141_kin.npz"),
                    eval_join=os.path.join(RAW, "join_eval141.npz"))
G = R.ground_truth(a)
SRC = {"": os.path.join(RAW, "full", "eval", "eval_{c}.npz"),
       "_orc": os.path.join(RAW, "2026-10-20-tac", "eval", "eval_{c}-tacorc.npz"),
       "_alls": os.path.join(RAW, "2026-10-21-tacgrp2", "eval", "eval_{c}-tacorc-all-stoplead.npz")}
preds, saved, prov = {}, {}, {}
for c in ("M1_s0", "M1_s1", "M2b_s0", "M2b_s1"):
    for suf, pat in SRC.items():
        p = pat.format(c=c)
        z = np.load(p)
        assert np.array_equal(z["eid"], G["eid"]), f"{p} is not on the eval grid"
        preds[c + suf] = z["pred"].astype(np.float64)
        saved[c + suf] = np.asarray(z["ade6"], np.float64)
        prov[c + suf] = {"path": os.path.relpath(p, PKG).replace("\\", "/"), "sha256": R.sha256(p)}
p10 = str(R.P10_DEFAULT)
preds["P10"] = np.load(p10).astype(np.float64)
prov["P10"] = {"path": "P10_DEFAULT", "sha256": R.sha256(p10)}
floor = np.load(os.path.join(RAW, "full", "eval", "floors_perwindow.npz"))["P10_refit__ade6"]

out = {"schema": "v8f-dk-oracle-read-v1", "exploratory": True, "reading_rule_written_before_the_run": RULE,
       "tier": {"c": "T1-class open loop", "c_orc": "T0 (O reads the GT code)", "c_alls": "T0 (GT code except stoplead)"},
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over episodes; blind to training-seed variance (H-ESTIM-SEED-1)",
       "inputs": prov, "reads": {}}
blocks = {"k10": str(R.LEAD_BLOCK_DEFAULT), "k30": str(R.LEAD_BLOCK_K30)}
for key, bp in blocks.items():
    blk = R.load_block(bp)
    res = R.read_dk(preds=preds, gt=G["gt"], eid=G["eid"], clip=G["clip"], frame=G["frame"], v0=G["v0"], block=blk, k=blk["_k"], dt=G["dt"],
                    n_boot=2000, seed=0, saved_ade6=saved, p10_floor_ade6=floor, pairs=[])
    pw, eid, lead = res["perwindow"], res["perwindow"]["eid"], res["perwindow"]["is_lead"]
    r = {"lead_block_sha256": R.sha256(bp), "window_states": res["read"]["join"]["window_states"],
         "contract": res["read"]["human_reference"]["kinematic_contract"]["status"], "arms": {}, "pairs": {}}
    for nm in preds:
        t, h = pw[f"{nm}__min_ttc_s"], pw[f"{nm}__headway_min_m"]
        m = lead & np.isfinite(t)
        r["arms"][nm] = {"n": int(m.sum()), "S5": float((t[m] < 5).mean()), "S10": float((t[m] < 10).mean()), "ttc_mean_s": float(t[m].mean()),
                         "headway_mean_m": float(np.nanmean(h[lead & np.isfinite(h)]))}
    th = pw["human__min_ttc_s"]
    mh = lead & np.isfinite(th)
    r["arms"]["human"] = {"n": int(mh.sum()), "S5": float((th[mh] < 5).mean()), "S10": float((th[mh] < 10).mean()), "ttc_mean_s": float(th[mh].mean())}
    for c in ("M1_s0", "M1_s1", "M2b_s0", "M2b_s1"):
        for x, y in ((c + "_orc", c), (c + "_orc", "P10"), (c, "P10"), (c + "_alls", c + "_orc"), (c + "_alls", c)):
            ta, tb = pw[f"{x}__min_ttc_s"], pw[f"{y}__min_ttc_s"]
            m = lead & np.isfinite(ta) & np.isfinite(tb)
            row = {"n_used": int(m.sum())}
            for thr in (5.0, 10.0):
                q = ci.paired_episode_cluster_bootstrap((ta[m] < thr).astype(float), (tb[m] < thr).astype(float), eid[m], n_boot=2000, seed=0)
                row[f"S{thr:g}"] = {"delta": q["delta"], "lo": q["lo"], "hi": q["hi"], "sep": bool(q["lo"] > 0 or q["hi"] < 0)}
            q = ci.paired_episode_cluster_bootstrap(ta[m], tb[m], eid[m], n_boot=2000, seed=0)
            row["ttc_mean_s"] = {"delta": q["delta"], "lo": q["lo"], "hi": q["hi"], "sep": bool(q["lo"] > 0 or q["hi"] < 0)}
            r["pairs"][f"{x}__minus__{y}"] = row
    out["reads"][key] = r


def s5(key, x, y):
    return out["reads"][key]["pairs"][f"{x}__minus__{y}"]["S5"]


def verdict(arm):
    cs = [f"{arm}_s0", f"{arm}_s1"]
    neg = all(s5("k10", c + "_orc", c)["sep"] and s5("k10", c + "_orc", c)["delta"] < 0 for c in cs)
    pos_any = any(s5("k10", c + "_orc", "P10")["sep"] and s5("k10", c + "_orc", "P10")["delta"] > 0 for c in cs)
    pos_all = all(s5("k10", c + "_orc", "P10")["sep"] and s5("k10", c + "_orc", "P10")["delta"] > 0 for c in cs)
    if neg and not pos_any:
        lab = "UPPER"
        if all(s5("k10", c + "_alls", c + "_orc")["sep"] and s5("k10", c + "_alls", c + "_orc")["delta"] > 0 for c in cs):
            lab = "UPPER+STOPLEAD"
        return lab
    return "LOWER" if pos_all else "MIXED"


out["verdict"] = {"M2b (decides)": verdict("M2b"), "M1 (reported)": verdict("M1")}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(R._clean(out), f, indent=1)
for key, r in out["reads"].items():
    print(f"[{key}] states {r['window_states']} contract {r['contract']}")
    for nm, v in r["arms"].items():
        print(f"  {nm:12s} n {v['n']:4d}  S5 {v['S5']:.3f}  S10 {v['S10']:.3f}  TTC {v['ttc_mean_s']:.2f}")
    for nm, v in r["pairs"].items():
        print(f"  {nm:32s} S5 {v['S5']['delta']:+.4f} [{v['S5']['lo']:+.4f}, {v['S5']['hi']:+.4f}] {'sep' if v['S5']['sep'] else 'ns'}"
              f"   TTC {v['ttc_mean_s']['delta']:+.3f} [{v['ttc_mean_s']['lo']:+.3f}, {v['ttc_mean_s']['hi']:+.3f}] {'sep' if v['ttc_mean_s']['sep'] else 'ns'}")
print("VERDICT", out["verdict"])
