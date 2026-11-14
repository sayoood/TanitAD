"""lead_diag_check.py -- control of card PREREG_V8F_PROOF_ADD18 section 5: `--wm-l1-diag on` against off gives the same weights after the same steps.
Runs the build's worker (checks/m_worker.py, synthetic trainer, CPU) twice in tree_patch for each arm shape of the card, 40 steps, and compares the
final checkpoints tensor by tensor (max abs difference must be exactly 0.0).  Control that the comparison can see a difference: the PR shape against
the P shape (rollout term on against off) must differ.  Writes lead_diag_check.json next to this file.
  python lead_diag_check.py"""
import glob
import json
import os
import subprocess
import sys

S = "C:/Users/Admin/v8proof_c/motion"
PY = sys.executable
STEPS = 40
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
SHAPES = {"PR": G + ["--wm-l1-pred", "cell", "--wm-l1-roll-k", "10", "--w-wm-roll", "1.0"],
          "RR": G + ["--wm-l1-residual", "on", "--wm-l1-roll-k", "10", "--w-wm-roll", "1.0"],
          "RES": G + ["--wm-l1-residual", "on"],
          "ROLL": G + ["--wm-l1-roll-k", "10", "--w-wm-roll", "1.0"],
          "P_control": G + ["--wm-l1-pred", "cell"]}


def w(*p):
    return os.sep.join(p).replace("/", os.sep)


def run(name, extra):
    root = S + "/tree_patch"
    out = S + "/tmp/lead_diag/" + name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1", TMP=w(S, "tmp"), TEMP=w(S, "tmp"), MW_NORESUME="1",
               PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
    r = subprocess.run([PY, "F:/TanitAD-scratch/v8proof/spatial/motion/checks/m_worker.py", root, out, str(STEPS), json.dumps(extra)], env=env,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-3000:]
    cks = sorted(glob.glob(out + "/run/**/ckpt_last.pt", recursive=True))
    assert len(cks) == 1, cks
    return cks[0], out


def final(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"], int(ck.get("step", -1))


def maxdiff(a, b):
    assert list(a.keys()) == list(b.keys()), "state-dict keys differ"
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].numel())


def log_keys(out):
    p = glob.glob(out + "/run/**/train_log.jsonl", recursive=True)[0]
    last = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()][-1]
    return sorted(k for k in last if k.startswith("wm"))


res = {"steps": STEPS, "shapes": {}}
finals = {}
for name, extra in SHAPES.items():
    if name == "P_control":
        ck, out = run(name, extra)
        finals[name] = final(ck)[0]
        continue
    ck_off, out_off = run(name + "_off", extra)
    ck_on, out_on = run(name + "_on", extra + ["--wm-l1-diag", "on"])
    a, sa = final(ck_off)
    b, sb = final(ck_on)
    d = maxdiff(a, b)
    finals[name] = a
    res["shapes"][name] = {"argv_extra": extra, "final_step": [sa, sb], "max_abs_diff_weights_diag_on_vs_off": d, "wm_log_keys_off": log_keys(out_off),
                           "wm_log_keys_on": log_keys(out_on), "pass": bool(d == 0.0 and sa == sb == STEPS)}
ctl = maxdiff(finals["PR"], finals["P_control"])
res["control_rollout_term_on_vs_off_differs"] = {"max_abs_diff_PR_vs_P": ctl, "pass": bool(ctl > 0.0)}
res["PASS"] = bool(all(v["pass"] for v in res["shapes"].values()) and ctl > 0.0)
json.dump(res, open(S + "/checks_lead/lead_diag_check.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["PASS"] else 2)
