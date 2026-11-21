"""LEAD-VIOL controls on the REAL 6 s lead block (no model): hold-v0 constant-velocity and STOP plans
vs the GT human path, eval139 clips, lead windows = state LEAD.  MEASURED (ours); NOT the refcv7 baseline."""
import os
import re
import sys
import json
import numpy as np

X7 = r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/x7tree"
sys.path[:0] = [X7 + "/stack", X7 + "/taniteval"]
from taniteval import lead_metrics as lm  # noqa: E402

d = "D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl"
ids = sorted({re.match(r"([0-9a-f-]{36})", f).group(1) for f in os.listdir(d)
              if re.match(r"[0-9a-f-]{36}", f)})
out = {}
for name, f in (("k12_dt0p5", "b1_eval_lead_block_k12_dt0p5.npz"),
                ("k30_dt0p2", "b1_eval_lead_block_k30_dt0p2.npz"),
                ("k10_dt0p2_banked_2s", None)):
    p = (r"C:/Users/Admin/qland/hold_x7/artifacts/" + f) if f else \
        r"C:/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"
    z = np.load(p)
    m = np.isin(z["clip_id"], ids)
    state = z["state"][m]
    lead = state == "LEAD"
    leads = z["leads"][m]
    lens = z["lead_lens"][m]
    v0 = z["speeds"][m]
    dt = float(z["dt_s"][0])
    K = leads.shape[1]
    ts = z["ts_rel_s"]
    eid = z["eid"][m]
    human = {"headway_min_m": z["gt_headway_min_m"][m], "time_gap_min_s": z["gt_time_gap_min_s"][m],
             "min_ttc_s": z["gt_min_ttc_s"][m]}
    cv = np.zeros((m.sum(), K, 2))
    cv[:, :, 0] = v0[:, None] * ts[None, :]
    stop = np.zeros((m.sum(), K, 2))
    res = {"n_rows_eval139": int(m.sum()), "n_lead_rows": int(lead.sum()),
           "horizon_s": float(ts[-1]), "dt_s": dt}
    for nm, path in (("hold_v0_CV", cv), ("STOP", stop), ("human_as_plan", None)):
        if path is None:
            r = lm.lead_viol(human, human, eid, lead_valid=lead, n_boot=300)
        else:
            r = lm.lead_viol_vs_human(path, None, leads, lens, v0, dt, eid=eid, lead_valid=lead,
                                      n_boot=300) if False else None
            pdk = lm.distance_keeping(path, leads, lens, v0, dt)
            r = lm.lead_viol(pdk, human, eid, lead_valid=lead, n_boot=300)
        res[nm] = {k: r[k] for k in ("n_lead_windows", "n_viol", "share", "n_at_risk",
                                     "n_human_tg_below_thr", "n_human_tg_undefined",
                                     "n_plan_lost_lead", "share_incl_lost_lead",
                                     "n_plan_ttc_finite", "n_plan_ttc_censored")}
        res[nm]["ci_lo_hi"] = [r["ci"]["lo"], r["ci"]["hi"]]
        res[nm]["n_episodes"] = r["ci"]["n_episodes"]
    out[name] = res
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:/Users/Admin/qland/hold_x7/artifacts/real_block_controls.json", "w"), indent=1)
