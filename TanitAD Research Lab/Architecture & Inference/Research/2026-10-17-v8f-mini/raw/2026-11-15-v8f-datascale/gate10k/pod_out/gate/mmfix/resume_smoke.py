"""G-CKPT resume half (pod, GPU, about 6 min): a U-set smoke (--max-clips 200, 40 steps, ckpt every 20), the trainer's V8F_TRAIN_CRASH_AT hook (v8f_mini_train.py:3614) at step 30, then the SAME argv
again (the trainer resumes from ckpt_last.pt: --resume is the default), against an uninterrupted run of the same argv. Writes /root/dscale10k/gate/resume.json.
Bars: the crashed run dies at the hook (rc != 0, the hook's message in its log) and leaves a checkpoint at step 20; the second run exits 0 and its final checkpoint is at step 40; its log shows
the resume; the resumed run's final state is compared with the uninterrupted run's tensor by tensor (REPORTED: the card asks for a resume test, and the trainer states "a resumed run is the same run").
Deliberate regressions (each must read RED): a run whose final step is 20 (no resume), a run with no checkpoint at the crash, a log without the hook message, and a one-element change in the compared state.
MM 2026-11-17 (the bitwise comparison could not be a bar): MEASURED on this smoke (resume_floor.json), two UNINTERRUPTED runs of the same argv differ in
238 of 276 tensors (max abs 1.95e-5): the trainer is not bitwise reproducible on this GPU, and the old fourth arm could never read RED once every
compared tensor already differed.  Now: a second uninterrupted run gives the run-to-run floor F (max abs over tensors); the resume PASSES iff
resumed-vs-straight max abs <= TOL_FLOOR x F (TOL_FLOOR 2, set after the floor measurement read resumed/floor = 0.048: this is an instrument repair,
and its teeth are the defect arm below).  Regression arms added: a resume from the step-20 checkpoint with the Adam moments zeroed (a resume that
drops the optimizer state) must exceed the bound; the compare function must count 0 on a state against itself and 1 on a one-element change."""
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


TOL_FLOOR = 2.0


def max_abs(a, b):
    """max over tensors of max |a - b| (inf on a missing key or a shape change)"""
    if set(a) != set(b):
        return float("inf")
    m = 0.0
    for k in a:
        if a[k].shape != b[k].shape:
            return float("inf")
        if a[k].numel() and not bool((a[k] == b[k]).all()):
            m = max(m, float((a[k].double() - b[k].double()).abs().max()))
    return m


def resume_problems(rec):
    bad = []
    F, RS = rec.get("floor_max_abs"), rec.get("resumed_vs_straight_max_abs")
    if F is None or RS is None:
        bad.append("the run-to-run floor or the resumed-vs-straight difference was not measured")
    elif RS > TOL_FLOOR * F:
        bad.append(f"resumed vs straight max abs {RS:.3g} exceeds {TOL_FLOOR} x the straight-vs-straight floor {F:.3g}")
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
if os.path.isfile(O + "/crash/ckpt_last.pt"):
    shutil.copy2(O + "/crash/ckpt_last.pt", O + "/ckpt20.pt")      # kept for the optimizer-state defect arm
rec["resume_rc"] = run(setf(t, "--out", [O + "/crash"]), O + "/resume.log")
s2 = ck_step(O + "/crash/ckpt_last.pt")
rec["resume_final_step"] = s2[0] if s2 else None
txt = open(O + "/resume.log", errors="replace").read().lower()
rec["resume_logged"] = ("[resume] step" in txt)
rec["straight_rc"] = run(setf(t, "--out", [O + "/straight"]), O + "/straight.log")
s3 = ck_step(O + "/straight/ckpt_last.pt")
rec["straight_final_step"] = s3[0] if s3 else None
rec["straight2_rc"] = run(setf(t, "--out", [O + "/straight2"]), O + "/straight2.log")
s4 = ck_step(O + "/straight2/ckpt_last.pt")
bad_rs = None
if s2 and s3 and s4:
    rec["n_tensors_differ_resumed_vs_straight"] = compare(s2[1], s3[1])
    rec["n_tensors_differ_straight_vs_straight2"] = compare(s4[1], s3[1])
    rec["resumed_equals_straight_bitwise"] = rec["n_tensors_differ_resumed_vs_straight"] == 0
    rec["floor_max_abs"] = max_abs(s4[1], s3[1])
    rec["resumed_vs_straight_max_abs"] = max_abs(s2[1], s3[1])
    sd = {k: v.clone() for k, v in s3[1].items()}
    k0 = next(k for k, v in sd.items() if v.dtype.is_floating_point and v.numel() > 1)
    sd[k0].view(-1)[0] += 1e-6
    rec["compare_self_n_differ"] = compare(s3[1], s3[1])
    rec["compare_one_element_changed_n_differ"] = compare(sd, s3[1])
    if os.path.isfile(O + "/ckpt20.pt"):                                 # the defect: resume from step 20 with the Adam moments zeroed
        import torch
        ck = torch.load(O + "/ckpt20.pt", map_location="cpu", weights_only=False)
        for st in ck["opt"]["state"].values():
            st["exp_avg"].zero_()
            st["exp_avg_sq"].zero_()
        torch.save(ck, O + "/crash/ckpt_last.pt")
        rec["bad_resume_rc"] = run(setf(t, "--out", [O + "/crash"]), O + "/resume_bad.log")
        sb = ck_step(O + "/crash/ckpt_last.pt")
        rec["bad_resume_final_step"] = sb[0] if sb else None
        bad_rs = max_abs(sb[1], s3[1]) if (sb and rec["bad_resume_rc"] == 0 and sb[0] == 40) else float("inf")
        rec["bad_resume_vs_straight_max_abs"] = bad_rs
rec["wall_s"] = round(time.time() - t0, 1)
probs = resume_problems(rec)
regs = []
for nm, m in (("no_resume_final_step_20", {"resume_final_step": 20}), ("no_checkpoint_at_crash", {"ckpt_step_at_crash": None}), ("no_hook_in_log", {"crash_hook_in_log": False}),
              ("resume_drops_optimizer_state", {"resumed_vs_straight_max_abs": bad_rs})):
    r2 = dict(rec)
    r2.update(m)
    p2 = resume_problems(r2)
    regs.append({"arm": nm, "red": bool(p2) and not (nm == "resume_drops_optimizer_state" and bad_rs is None), "problems": p2[:2]})      # the defect arm counts only when it ran
regs.append({"arm": "compare_counts_a_one_element_change", "red": rec.get("compare_self_n_differ") == 0 and rec.get("compare_one_element_changed_n_differ") == 1, "problems": []})
out = {"item": "resume", "pass": not probs and all(r["red"] for r in regs), "problems": probs, "record": rec, "regression": regs, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(out, open(D + "/gate/resume.json", "w"), indent=1, default=str)
print("resume", "PASS" if out["pass"] else "FAIL", probs, [r["red"] for r in regs], {k: rec.get(k) for k in ("floor_max_abs", "resumed_vs_straight_max_abs", "bad_resume_vs_straight_max_abs", "wall_s")})
