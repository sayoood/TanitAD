#!/usr/bin/env python3
"""Census every v7-tiny run dir on Thor: what exists, what finished, key args."""
import json, os, sys, gzip

ROOT = "/home/nvidia/v7tiny"
KEYS = ["steps", "seed", "batch", "lr", "window", "stage", "o5_k", "o5_target",
        "w_o5", "w_o6", "w_o11_cf", "o11_k", "o11_negs", "w_o1", "w_o9_ema",
        "w_o10_psg", "freeze_encoder", "init_from", "clip", "grad_clip",
        "ema_decay", "ema_decay_ramp", "tau", "run", "out", "max_horizon",
        "w_o14", "o14_mode", "nav_labels", "w_plan", "w_tactical", "w_traj"]

out = {}
for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        continue
    rec = {"files": {}}
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            rec["files"][f] = os.path.getsize(p)
        except Exception as e:
            rec["files"][f] = "ERR:%s" % e
    # config
    cp = os.path.join(d, "config.json")
    if os.path.exists(cp):
        try:
            c = json.load(open(cp))
            a = c.get("args", c)
            rec["config_nkeys"] = len(a) if isinstance(a, dict) else None
            rec["args"] = {k: a[k] for k in KEYS if isinstance(a, dict) and k in a}
            if isinstance(a, dict):
                # anything nonzero-weight we did not list
                rec["nonzero_w"] = {k: v for k, v in a.items()
                                    if k.startswith("w_") and isinstance(v, (int, float)) and v != 0}
            if "argv" in c:
                rec["argv"] = " ".join(c["argv"]) if isinstance(c["argv"], list) else str(c["argv"])
        except Exception as e:
            rec["config_err"] = str(e)
    # metrics / summary
    for fn in ("metrics.json", "summary.json", "stage_gate.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            try:
                j = json.load(open(p))
                if fn == "metrics.json":
                    rec["metrics_step"] = j.get("step")
                    rec["metrics_keys"] = sorted(j.keys())[:40]
                    rec["metrics"] = {k: v for k, v in j.items()
                                      if isinstance(v, (int, float, str, bool))}
                elif fn == "summary.json":
                    rec["summary"] = j if len(str(j)) < 3000 else {k: j[k] for k in list(j)[:25]}
                else:
                    rec["stage_gate"] = j if len(str(j)) < 3000 else str(j)[:3000]
            except Exception as e:
                rec[fn + "_err"] = str(e)
    # train_log.jsonl -> first/last row
    for lg in ("train_log.jsonl", "log.jsonl", "trainlog.jsonl"):
        p = os.path.join(d, lg)
        if os.path.exists(p):
            try:
                lines = open(p, "r", errors="replace").read().strip().split("\n")
                rec["log_rows"] = len(lines)
                rec["log_first"] = json.loads(lines[0]) if lines else None
                rec["log_last"] = json.loads(lines[-1]) if lines else None
            except Exception as e:
                rec["log_err"] = str(e)
    out[name] = rec

print(json.dumps(out, indent=None, default=str))
