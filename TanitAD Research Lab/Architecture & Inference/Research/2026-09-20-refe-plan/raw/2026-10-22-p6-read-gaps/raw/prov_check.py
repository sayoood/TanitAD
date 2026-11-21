"""Provenance of the banked harness copies: is each scratchpad file == the repo original (mirror tip 6723462) after the documented
`D:/ -> E:/` sed, once CRLF is normalised?  Prints one JSON line per file; also reports repo-original candidates that match WITHOUT the sed."""
import hashlib, json, re, subprocess, sys
R = "6723462"
REPO = "C:/Users/Admin/tanitad-push"
SPBANK = "C:/Users/Admin/par18/p6gaps/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/harness"
def git_show(path):
    r = subprocess.run(["git", "-C", REPO, "show", f"{R}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def lf(b): return b.replace(b"\r\n", b"\n")
def sha(b): return hashlib.sha256(b).hexdigest()
W3 = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/"
cands = {
 "w3e/code": W3,
 "ev6e": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/",
}
import os
out = []
for sub, repo_dir in cands.items():
    d = f"{SPBANK}/{sub}"
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"): continue
        b = open(f"{d}/{f}", "rb").read()
        orig = git_show(repo_dir + f)
        rec = {"file": f"{sub}/{f}", "sp_lf_sha": sha(lf(b))[:16]}
        if orig is None:
            rec["repo_original"] = None
        else:
            o = lf(orig)
            rec["repo_original"] = repo_dir + f
            rec["equal_without_sed"] = (lf(b) == o)
            seds = [("D:/", "E:/"), ("D:\\", "E:\\")]
            t = o
            for a, c in seds: t = t.replace(a.encode(), c.encode())
            rec["equal_after_D_to_E_sed"] = (lf(b) == t)
            if not rec["equal_after_D_to_E_sed"] and not rec["equal_without_sed"]:
                # first differing line
                A, T = lf(b).split(b"\n"), t.split(b"\n")
                diffs = [(i + 1, A[i][:140].decode("utf-8", "replace"), T[i][:140].decode("utf-8", "replace")) for i in range(min(len(A), len(T))) if A[i] != T[i]][:3]
                rec["n_lines"] = [len(A), len(T)]; rec["first_diffs"] = diffs
        out.append(rec)
for r in out: print(json.dumps(r))
