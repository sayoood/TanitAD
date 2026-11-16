"""resume_floor.py -- the noise floor the G-CKPT resume smoke never measured (pod, GPU, about 2 min).

resume.json reads `resumed_equals_straight_bitwise: false` (241 tensors differ) with no straight-vs-straight control, so it cannot say
whether the resume drops state or the trainer is simply not bitwise reproducible on this GPU.  This runs the uninterrupted smoke a
SECOND time with the identical argv (resume_smoke.py's own argv construction) and compares, tensor by tensor:
  F  = straight vs straight2   (the run-to-run floor, nothing resumed)
  RS = resumed vs straight     (what resume.json counted)
Writes /root/dscale10k/gate/resume_floor.json; makes all four runs itself and writes only resume_floor/.
"""
import json
import os
import shutil
import subprocess
import sys

import torch

D = "/root/dscale10k"
TREE = D + "/tree"
PY = "/root/venv/bin/python"
O = D + "/gate/resume_smoke"
F = D + "/gate/resume_floor"
ENV = dict(os.environ, PYTHONPATH=":".join([TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"]), HF_HUB_DISABLE_XET="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
ENV.pop("CUDA_VISIBLE_DEVICES", None)


def setf(t, f, vals):
    t = list(t)
    if f in t:
        i = t.index(f)
        j = i + 1
        while j < len(t) and not t[j].startswith("--"):
            j += 1
        return t[:i + 1] + list(vals) + t[j:]
    return t + [f] + list(vals)


def delf(t, f):
    t = list(t)
    if f not in t:
        return t
    i = t.index(f)
    j = i + 1
    while j < len(t) and not t[j].startswith("--"):
        j += 1
    return t[:i] + t[j:]


def state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("step"), ck["model"]


def diff(a, b):
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b or a[k].shape != b[k].shape:
            rows.append((k, float("inf"), float("inf")))
            continue
        x, y = a[k].double(), b[k].double()
        if torch.equal(a[k], b[k]):
            continue
        m = float((x - y).abs().max())
        rows.append((k, m, m / max(float(y.abs().max()), 1e-12)))
    return rows


def summ(rows, n_total):
    if not rows:
        return {"n_differ": 0, "n_total": n_total}
    ms = sorted(r[1] for r in rows)
    rel = sorted(r[2] for r in rows)
    return {"n_differ": len(rows), "n_total": n_total, "max_abs_max": ms[-1], "max_abs_median": ms[len(ms) // 2], "rel_max": rel[-1], "rel_median": rel[len(rel) // 2],
            "top5": [[r[0], r[1]] for r in sorted(rows, key=lambda r: -r[1])[:5]]}


if os.path.isdir(F):
    shutil.rmtree(F)                                                     # own scratch dir of an earlier run of this very script
os.makedirs(F)
base = json.load(open(D + "/argv/U_s0.A.json"))["argv"]
t = delf(delf(base, "--fit-join"), "--steps-mult")
for f, v in (("--steps", ["40"]), ("--ckpt-every", ["20"]), ("--log-every", ["10"]), ("--max-clips", ["200"])):
    t = setf(t, f, v)
def run(out_dir, log, extra=None):
    e = dict(ENV)
    e.update(extra or {})
    with open(F + "/" + log, "w") as lf:
        return subprocess.run([PY, TREE + "/stack/scripts/v8f_mini_train.py"] + setf(t, "--out", [F + "/" + out_dir]), stdout=lf, stderr=subprocess.STDOUT, env=e, cwd=TREE, timeout=3000, stdin=subprocess.DEVNULL).returncode


# the smoke's own checkpoints were removed by the gate's cleanup, so all four runs are made here (same argv, same order as resume_smoke.py)
rcs = {"crash": run("crash", "crash.log", {"V8F_TRAIN_CRASH_AT": "30"})}
rcs["resume"] = run("crash", "resume.log")
rcs["straight"] = run("straight", "straight.log")
rcs["straight2"] = run("straight2", "straight2.log")
rc = rcs["straight2"]
s1, a = state(F + "/straight/ckpt_last.pt")
s2, b = state(F + "/straight2/ckpt_last.pt") if rc == 0 else (None, {})
sr, r = state(F + "/crash/ckpt_last.pt")
out = {"rcs": rcs, "straight2_rc": rc, "steps": {"straight": s1, "straight2": s2, "resumed": sr}, "n_tensors": len(a),
       "F_straight_vs_straight2": summ(diff(b, a), len(a)) if rc == 0 else None,
       "RS_resumed_vs_straight": summ(diff(r, a), len(a)),
       "self_control_straight_vs_itself": summ(diff(a, a), len(a))}
json.dump(out, open(D + "/gate/resume_floor.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "top5"}) for k, v in out.items()}))
print("ZZ RESUME FLOOR END ZZ")
