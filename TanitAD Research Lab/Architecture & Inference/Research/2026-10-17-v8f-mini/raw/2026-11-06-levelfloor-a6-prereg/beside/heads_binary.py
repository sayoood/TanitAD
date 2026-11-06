#!/usr/bin/env python3
"""heads_binary.py -- descriptive, after the A5 read (never deciding): the LLR level head as a binary go/no-go classifier on the A5 RILF100 ticks.
Rows: logged route speed class (lf_read.logged_speed); columns: v_hat >= 5.2 m/s. Also the planner's own planned speed over its first second (raw plan, before F) against 5.5 m/s.
usage: python -I heads_binary.py <out_json>"""
import json, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/a6_diag")
import f_harm_split as H
res = {}
for s in (0, 1):
    tag = f"RILF100_s{s}"
    t = np.load(f"{H.OUT}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=False)
    vlog, ok = H.LFR.logged_speed(t["gt"], t["gt_ok"])
    vh = t["lf_vhat"].astype(np.float64)
    pr = t["plan_raw"]
    vplan = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(pr), 1, 2)), pr[:, :5]], 1), axis=1), axis=-1).sum(1) / 1.0
    m1 = H.LFR.m1_ticks(t["v0"], vlog, ok)
    out = {"n_ok": int(ok.sum())}
    for name, m in (("vlog_lt3", ok & (vlog < 3)), ("vlog_3to5.2", ok & (vlog >= 3) & (vlog < 5.2)), ("vlog_ge5.2", ok & (vlog >= 5.2)), ("m1_ticks", m1)):
        out[name] = {"n": int(m.sum()), "share_vhat_ge5.2": float((vh[m] >= 5.2).mean()), "share_vhat_ge3": float((vh[m] >= 3.0).mean()),
                     "median_vhat": float(np.median(vh[m])), "median_vlog": float(np.median(vlog[m])), "share_planraw_lt5.5": float((vplan[m] < 5.5).mean())}
    res[tag] = out
    print(tag, json.dumps(out))
json.dump(res, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
