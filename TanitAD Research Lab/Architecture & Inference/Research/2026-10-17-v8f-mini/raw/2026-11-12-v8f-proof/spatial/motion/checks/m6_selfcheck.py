"""m6: the trainer selfcheck (--selfcheck-only, synthetic world, --exclude 1:3 as the previous lever did) of the tree_patch for
  (A on)            grid state, flat predictor, --wm-l1-residual on
  (B on)            grid state, flat predictor, --wm-l1-roll-k 6 --w-wm-roll 0.3
  (A and B, cell)   grid state, cell predictor, both levers
  (default)         grid state, flat predictor, no lever: the built levers read off / 0 / 0.0
It reads the BUILT levers out of selfcheck.json, runs the gate_extra expectations (right ones PASS, a deliberately wrong one per lever is REFUSED, rc 1), and
re-reads that the trainer's own declared_vs_built list is empty. Output: checks/m6_selfcheck.json and checks/selfchecks/*.json|log. Usage: m6_selfcheck.py"""
import json
import os
import subprocess

S = "C:/Users/Admin/v8proof_c/motion"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
GRIDARM = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
ARMS = {
    "default": ([], {"wm_l1_residual": "off", "wm_l1_roll_k": "0", "w_wm_roll": "0"}),
    "A": (["--wm-l1-residual", "on"], {"wm_l1_residual": "on", "wm_l1_roll_k": "0", "w_wm_roll": "0"}),
    "B": (["--wm-l1-roll-k", "6", "--w-wm-roll", "0.3"], {"wm_l1_residual": "off", "wm_l1_roll_k": "6", "w_wm_roll": "0.3"}),
    "AB_cell": (["--wm-l1-pred", "cell", "--wm-l1-residual", "on", "--wm-l1-roll-k", "6", "--w-wm-roll", "0.3", "--wm-l1-diag", "on"],
                {"wm_l1_residual": "on", "wm_l1_roll_k": "6", "w_wm_roll": "0.3", "wm_l1_diag": "on", "wm_l1_pred": "cell"}),
}
WRONG = {"wm_l1_residual": {"off": "on", "on": "off"}, "wm_l1_roll_k": {"0": "6", "6": "0"}, "w_wm_roll": {"0": "0.3", "0.3": "0"}}
res = {}
os.makedirs(f"{S}/checks/selfchecks", exist_ok=True)
for tag, (extra, want) in ARMS.items():
    out = f"{S}/tmp/sc_{tag}"
    r = subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", "--synthetic", *GRIDARM, *extra, "--exclude", "1:3", "--threads", "2", "--selfcheck-only", "--out", out],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{S}/checks/selfchecks/selfcheck_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-4000:] + "\nSTDERR\n" + (r.stderr or "")[-2000:] + f"\nRC {r.returncode}\n")
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{S}/checks/selfchecks/selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv = sc["levers"]
    keys = ("wm_l1_residual", "wm_l1_roll_k", "w_wm_roll", "wm_l1_diag", "wm_l1_pred", "wm_l1_pred_width", "wm_l1_pred_params", "wm_l1_pred_cross_cell", "wm_state")
    rec = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"), "built": {k: lv.get(k) for k in keys}}
    exp = ["levers.wm_state=grid", "info.n_clips=4"] + [f"levers.{k}={v}" for k, v in want.items()]
    g = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "MOTION_SELFTEST", "--out", f"{out}/extra_token.json", "--expect", *exp],
                       capture_output=True, text=True, encoding="utf-8")
    rec["gate_extra_right_expectations"] = {"rc": g.returncode, "out": g.stdout.strip()[:240]}
    rec["gate_extra_wrong_expectation_refused"] = {}
    for k, flip in WRONG.items():
        bad = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "x", "--out", f"{out}/never.json", "--expect", f"levers.{k}={flip[want[k]]}"],
                             capture_output=True, text=True, encoding="utf-8")
        rec["gate_extra_wrong_expectation_refused"][k] = {"expected": flip[want[k]], "rc": bad.returncode, "out": bad.stdout.strip()[:200]}
    rec["PASS"] = (r.returncode == 0 and rec["declared_vs_built"] in ([], None, "PASS") and g.returncode == 0
                   and all(v["rc"] != 0 for v in rec["gate_extra_wrong_expectation_refused"].values())
                   and all(str(lv.get(k)) == v or (v == "0" and float(lv.get(k)) == 0.0) or (k == "w_wm_roll" and float(lv.get(k)) == float(v)) for k, v in want.items()))
    res[tag] = rec
    print(tag, json.dumps(rec))
res["PASS"] = all(v["PASS"] for v in res.values())
json.dump(res, open(f"{S}/checks/m6_selfcheck.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("M6", "PASS" if res["PASS"] else "FAIL")
