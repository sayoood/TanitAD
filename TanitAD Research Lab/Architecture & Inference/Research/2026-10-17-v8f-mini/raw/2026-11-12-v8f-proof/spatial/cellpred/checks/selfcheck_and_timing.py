"""(1) selfcheck of the new arm shape and of the flat grid arm (synthetic, --exclude 1:3 as stage 1 did), gate_extra parse of the card's expectations
(with n_clips 4 for the 4-clip synthetic world) and the deliberate regression (declared cell, expected flat -> REFUSED);
(2) 30 synthetic training steps of the cell arm and of the flat grid arm on the CPU, interleaved twice, seconds per step. Not a pod number.
Usage: selfcheck_and_timing.py"""
import json
import os
import subprocess
import sys

S = "F:/TanitAD-scratch/v8proof/spatial/cellpred"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GATE = "F:/TanitAD-scratch/v8proof/pod/chain2/gate_extra.py"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
           PYTHONDONTWRITEBYTECODE="1", TMP=w(S, "tmp"), TEMP=w(S, "tmp"),
           PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
GRIDARM = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
res = {}
os.makedirs(f"{S}/checks/selfchecks", exist_ok=True)
for tag, extra in (("G", []), ("P", ["--wm-l1-pred", "cell"])):
    out = f"{S}/tmp/sc_{tag}"
    cmd = [PY, f"{root}/stack/scripts/v8f_mini_train.py", "--synthetic", *GRIDARM, *extra, "--exclude", "1:3", "--threads", "2", "--selfcheck-only", "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{S}/checks/selfchecks/selfcheck_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-6000:] + "\nSTDERR\n" + (r.stderr or "")[-3000:] + f"\nRC {r.returncode}\n")
    sc = json.load(open(f"{out}/selfcheck.json", encoding="utf-8"))
    with open(f"{S}/checks/selfchecks/selfcheck_{tag}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sc, fh, indent=1)
    lv = sc["levers"]
    res[tag] = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": sc.get("live", {}).get("pass"),
                "levers": {k: lv.get(k) for k in ("wm_state", "wm_z1_width", "wm_l1_dim", "wm_l1_pred_width", "wm_cell_dim", "wm_l1_pred", "wm_l1_pred_params",
                                                  "wm_l1_pred_params_flat_ref", "wm_l1_pred_params_le_flat", "wm_l1_pred_cross_cell")},
                "n_clips": sc.get("info", {}).get("n_clips")}
    exp = ["levers.wm_state=grid", "levers.wm_z1_width=640", "levers.wm_cell_dim=16", "info.n_clips=4"]
    exp += (["levers.wm_l1_pred=cell", "levers.wm_l1_pred_width=128"] if tag == "P" else ["levers.wm_l1_pred=flat", "levers.wm_l1_pred_width=256"])
    g = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "PREREG_V8F_PROOF_ADD16", "--out", f"{out}/extra_token.json", "--expect", *exp],
                       capture_output=True, text=True, encoding="utf-8")
    res[tag]["gate_extra"] = {"rc": g.returncode, "out": g.stdout.strip()[:300]}
    bad = subprocess.run([PY, GATE, "check", "--selfcheck", f"{out}/selfcheck.json", "--card", "x", "--out", f"{out}/never.json", "--expect",
                          "levers.wm_l1_pred=" + ("flat" if tag == "P" else "cell")], capture_output=True, text=True, encoding="utf-8")
    res[tag]["gate_extra_deliberate_regression"] = {"rc": bad.returncode, "out": bad.stdout.strip()[:300]}
    print(tag, json.dumps(res[tag]))

# ---- timing
def run_worker(name, extra, steps=30):
    out = f"{S}/tmp/timing/{name}"
    r = subprocess.run([PY, f"{S}/checks/c1_worker.py", root, out, str(steps), json.dumps(extra)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(l) for l in open(f"{out}/run/train_log.jsonl", encoding="utf-8")]
    secs = [x["sec"] for x in rows]
    return {"n": len(rows), "total_first_last": [rows[0]["total"], rows[-1]["total"]], "all_finite": all(x["total"] == x["total"] and abs(x["total"]) < 1e30 for x in rows),
            "sec_logged": secs[:3] + ["..."] + secs[-2:], "mean_step_s_excl_first5": None}, rows


def per_step(rows):
    # train_log 'sec' = elapsed seconds since the start of the loop at each logged step: per-step = successive differences
    s = [r["sec"] for r in rows]
    d = [b - a for a, b in zip(s, s[1:])]
    d = d[4:]
    return sum(d) / len(d)


timing = {"flat": [], "cell": []}
logs = {}
for rep in range(2):
    for name, extra in (("flat", GRIDARM), ("cell", GRIDARM + ["--wm-l1-pred", "cell"])):
        info, rows = run_worker(f"{name}_{rep}", extra)
        timing[name].append(per_step(rows))
        logs[f"{name}_{rep}"] = {"total_first_last": info["total_first_last"], "all_finite": info["all_finite"], "n": info["n"],
                                 "total_by_step": [round(r["total"], 5) for r in rows]}
        print(name, rep, timing[name][-1], info["total_first_last"], info["all_finite"])
res["timing_cpu_s_per_step"] = {k: v for k, v in timing.items()}
res["timing_cpu_best_of_2_s_per_step"] = {k: min(v) for k, v in timing.items()}
res["relative_cost_cell_over_flat_best_of_2"] = min(timing["cell"]) / min(timing["flat"])
res["relative_cost_cell_over_flat_mean"] = (sum(timing["cell"]) / 2) / (sum(timing["flat"]) / 2)
res["loss_logs"] = logs
json.dump(res, open(f"{S}/checks/selfcheck_and_timing.json", "w", encoding="utf-8"), indent=1)
print("relative cost", res["relative_cost_cell_over_flat_best_of_2"], res["relative_cost_cell_over_flat_mean"])
