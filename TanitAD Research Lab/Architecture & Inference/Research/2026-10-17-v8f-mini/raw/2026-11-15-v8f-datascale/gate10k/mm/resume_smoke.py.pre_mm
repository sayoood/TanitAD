"""G-CKPT resume half (pod, GPU, about 6 min): a U-set smoke (--max-clips 200, 40 steps, ckpt every 20), the trainer's V8F_TRAIN_CRASH_AT hook (v8f_mini_train.py:3614) at step 30, then the SAME argv
again (the trainer resumes from ckpt_last.pt: --resume is the default), against an uninterrupted run of the same argv. Writes /root/dscale10k/gate/resume.json.
Bars: the crashed run dies at the hook (rc != 0, the hook's message in its log) and leaves a checkpoint at step 20; the second run exits 0 and its final checkpoint is at step 40; its log shows
the resume; the resumed run's final state is compared with the uninterrupted run's tensor by tensor (REPORTED: the card asks for a resume test, and the trainer states "a resumed run is the same run").
Deliberate regressions (each must read RED): a run whose final step is 20 (no resume), a run with no checkpoint at the crash, a log without the hook message, and a one-element change in the compared state."""
import json
import os
import shutil
import subprocess
import sys
import time

D = "/root/dscale10k"
TREE = D + "/tree"
PY = "/root/venv/bin/python"
O = D + "/gate/resume_smoke"
ENV = dict(os.environ, PYTHONPATH=":".join([TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"]), HF_HUB_DISABLE_XET="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
ENV.pop("CUDA_VISIBLE_DEVICES", None)
sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"]


def setf(t, f, vals):
    t = list(t)
    if f not in t:
        return t + [f] + list(vals)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


def delf(t, f):
    t = list(t)
    if f not in t:
        return t
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i] + t[j:]


def resume_problems(rec):
    bad = []
    if rec.get("crash_rc") in (0, None) or not rec.get("crash_hook_in_log"):
        bad.append("the crash run did not die at the V8F_TRAIN_CRASH_AT hook")
    if rec.get("ckpt_step_at_crash") != 20:
        bad.append(f"the checkpoint left at the crash is at step {rec.get('ckpt_step_at_crash')}, expected 20")
    if rec.get("resume_rc") != 0:
        bad.append(f"the resumed run exited {rec.get('resume_rc')}")
    if rec.get("resume_final_step") != 40:
        bad.append(f"the resumed run's final checkpoint is at step {rec.get('resume_final_step')}, expected 40")
    if not rec.get("resume_logged"):
        bad.append("the resumed run's log does not show a resume")
    return bad


def compare(a, b):
    n = 0
    for k in a:
        if k not in b or a[k].shape != b[k].shape or not bool((a[k] == b[k]).all()):
            n += 1
    return n + len(set(b) - set(a))


def run(argv, log, env_extra=None, timeout=3000):
    e = dict(ENV)
    e.update(env_extra or {})
    with open(log, "w") as f:
        r = subprocess.run([PY, TREE + "/stack/scripts/v8f_mini_train.py"] + argv, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=TREE, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode


def ck_step(path):
    import torch
    if not os.path.isfile(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("step"), ck["model"]


if os.path.isdir(O):
    shutil.rmtree(O)                                                     # own scratch dir of an earlier run of this very script
os.makedirs(O)
base = json.load(open(D + "/argv/U_s0.A.json"))["argv"]
t = delf(delf(base, "--fit-join"), "--steps-mult")
for f, v in (("--steps", ["40"]), ("--ckpt-every", ["20"]), ("--log-every", ["10"]), ("--max-clips", ["200"])):
    t = setf(t, f, v)
rec = {"smoke": "U_s0.A argv, --max-clips 200 --steps 40 --ckpt-every 20, crash at step 30"}
t0 = time.time()
rec["crash_rc"] = run(setf(t, "--out", [O + "/crash"]), O + "/crash.log", {"V8F_TRAIN_CRASH_AT": "30"})
rec["crash_hook_in_log"] = "V8F_TRAIN_CRASH_AT" in open(O + "/crash.log", errors="replace").read()
s = ck_step(O + "/crash/ckpt_last.pt")
rec["ckpt_step_at_crash"] = s[0] if s else None
rec["resume_rc"] = run(setf(t, "--out", [O + "/crash"]), O + "/resume.log")
s2 = ck_step(O + "/crash/ckpt_last.pt")
rec["resume_final_step"] = s2[0] if s2 else None
txt = open(O + "/resume.log", errors="replace").read().lower()
rec["resume_logged"] = ("[resume] step" in txt)
rec["straight_rc"] = run(setf(t, "--out", [O + "/straight"]), O + "/straight.log")
s3 = ck_step(O + "/straight/ckpt_last.pt")
rec["straight_final_step"] = s3[0] if s3 else None
if s2 and s3:
    rec["n_tensors_differ_resumed_vs_straight"] = compare(s2[1], s3[1])
    rec["resumed_equals_straight_bitwise"] = rec["n_tensors_differ_resumed_vs_straight"] == 0
    sd = {k: v.clone() for k, v in s2[1].items()}
    k0 = next(k for k, v in sd.items() if v.dtype.is_floating_point and v.numel() > 1)
    sd[k0].view(-1)[0] += 1e-6
    rec["regression_compare_one_element_changed_n_differ"] = compare(sd, s3[1])
rec["wall_s"] = round(time.time() - t0, 1)
probs = resume_problems(rec)
regs = []
for nm, m in (("no_resume_final_step_20", {"resume_final_step": 20}), ("no_checkpoint_at_crash", {"ckpt_step_at_crash": None}), ("no_hook_in_log", {"crash_hook_in_log": False})):
    r2 = dict(rec)
    r2.update(m)
    p2 = resume_problems(r2)
    regs.append({"arm": nm, "red": bool(p2), "problems": p2[:2]})
regs.append({"arm": "one_element_changed_in_compared_state", "red": rec.get("regression_compare_one_element_changed_n_differ", 0) > rec.get("n_tensors_differ_resumed_vs_straight", 0), "problems": []})
out = {"item": "resume", "pass": not probs and all(r["red"] for r in regs), "problems": probs, "record": rec, "regression": regs, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(out, open(D + "/gate/resume.json", "w"), indent=1, default=str)
print("resume", "PASS" if out["pass"] else "FAIL", probs, [r["red"] for r in regs], {k: rec.get(k) for k in ("resumed_equals_straight_bitwise", "n_tensors_differ_resumed_vs_straight", "wall_s")})
