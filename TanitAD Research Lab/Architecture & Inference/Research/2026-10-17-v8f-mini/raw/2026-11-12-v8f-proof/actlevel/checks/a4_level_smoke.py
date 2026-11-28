"""Tiny end-to-end fine-tune smoke of ACTLEVEL on the CPU (tiny synthetic world, tiny LS-shape base checkpoint of the leadcond checks, read only). Runs the trainer of tree_level as a subprocess:
 (1) --w-act-cons 0.001 --w-act-level 0.01, 20 steps from the base checkpoint: rc 0, every logged value finite, act_level and act_level_n in every logged row, selfcheck levers.w_act_level == 0.01,
     the probe passes with its level keys, summary keys present, checkpoint cfg EQUAL to the base cfg and to the V = 0 run, state-dict keys EQUAL to the V = 0 run;
 (2) the same argv without the level flag: no act_level key in log / selfcheck / summary / probe; the step-1 total of (1) minus the step-1 total of (2) equals 0.01 x act_level of step 1;
 (3) MUTATION: a single-file copy of the trainer with the level add deleted: the step-1 delta reads 0.0 (RED), act_level still logged.
Usage: a4_level_smoke.py [tree_root]   (default: the tree_level next to this script)"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AL = os.path.abspath(os.path.join(HERE, ".."))
root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(AL, "work", "tree_level")
PY = sys.executable
BASE = "F:/TanitAD-scratch/v8proof/leadcond/work/tmp/base_ls/ckpt_last.pt"                    # the tiny LS-shape checkpoint written by the leadcond checks; read only
O = os.path.join(AL, "work", "tmp", "a4l")
os.makedirs(O, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=os.path.join(AL, "work", "tmp"), TEMP=os.path.join(AL, "work", "tmp"), PYTHONPATH=os.pathsep.join([os.path.join(root, "stack"), os.path.join(root, "taniteval"), os.path.join(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
LSF = ["--lead-sup", "on", "--w-lead-sup", "1.0"]
CONS = ["--w-act-cons", "0.001"]


def train(tag, extra, steps, script=None, log_every=5):
    out = os.path.join(O, tag)
    cmd = [PY, script or os.path.join(root, "stack", "scripts", "v8f_mini_train.py"), *TINY, *G, *LSF, "--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "100", "--seed", "0", "--steps", str(steps),
           "--log-every", str(log_every), "--ckpt-every", "100", *CONS, *extra, "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    with open(os.path.join(HERE, f"a4_level_{tag}.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-1500:] + "\nSTDERR\n" + (r.stderr or "")[-2000:] + f"\nRC {r.returncode}\n")
    rows = [json.loads(line) for line in open(os.path.join(out, "train_log.jsonl"), encoding="utf-8")] if r.returncode == 0 else []
    return r.returncode, rows, out


res = {"tree": root}
rcL, rowsL, outL = train("ft_V", ["--w-act-level", "0.01"], 20)
rc0, rows0, out0 = train("ft_V0", [], 20)
bck = torch.load(BASE, map_location="cpu", weights_only=False)
ckL = torch.load(os.path.join(outL, "ckpt_last.pt"), map_location="cpu", weights_only=False)
ck0 = torch.load(os.path.join(out0, "ckpt_last.pt"), map_location="cpu", weights_only=False)
ld = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
scL, smL = ld(os.path.join(outL, "selfcheck.json")), ld(os.path.join(outL, "summary.json"))
sc0, sm0 = ld(os.path.join(out0, "selfcheck.json")), ld(os.path.join(out0, "summary.json"))
fin = all(bool(np.isfinite(x[k])) for x in rowsL for k in x if isinstance(x[k], (int, float)))
n_pos = [x["act_level_n"] > 0 for x in rowsL]
probe = scL.get("act_cons_probe") or {}
res["V_finetune"] = {"rc": rcL, "n_log": len(rowsL), "all_logged_values_finite": fin, "act_level_in_every_row": all(("act_level" in x and "act_level_n" in x) for x in rowsL),
                     "act_level_n_positive_share": float(np.mean(n_pos)) if n_pos else None, "act_level_min": min(x["act_level"] for x in rowsL), "act_level_max": max(x["act_level"] for x in rowsL),
                     "selfcheck_levers": {k: v for k, v in scL["levers"].items() if "act_" in k}, "probe_pass": probe.get("pass"), "probe_level_keys": {k: v for k, v in probe.items() if "level" in k},
                     "summary": {k: v for k, v in smL.items() if "act_" in k}, "ckpt_cfg_equal_to_base_cfg": ckL["cfg"] == bck["cfg"], "ckpt_cfg_equal_to_V0_cfg": ckL["cfg"] == ck0["cfg"],
                     "state_keys_equal_to_V0": list(ckL["model"]) == list(ck0["model"]), "state_keys_equal_to_base": list(ckL["model"]) == list(bck["model"]),
                     "weights_differ_V_vs_V0_max_abs": max(float((ckL["model"][k].float() - ck0["model"][k].float()).abs().max()) for k in ck0["model"])}
res["V0_control"] = {"rc": rc0, "no_act_level_keys_in_log": not any("act_level" in k for x in rows0 for k in x), "no_act_level_in_selfcheck": not [k for k in sc0["levers"] if "level" in k],
                     "no_level_key_in_probe": not [k for k in (sc0.get("act_cons_probe") or {}) if "level" in k], "no_act_level_in_summary": not [k for k in sm0 if "level" in k]}
_, a1, _ = train("s1_V", ["--w-act-level", "0.01"], 1, log_every=1)
_, a0, _ = train("s1_V0", [], 1, log_every=1)
dl = a1[0]["total"] - a0[0]["total"]
res["step1_total_moves_by_V_times_level"] = {"total_V": a1[0]["total"], "total_V0": a0[0]["total"], "delta": dl, "V_x_act_level": 0.01 * a1[0]["act_level"], "abs_diff": abs(dl - 0.01 * a1[0]["act_level"]),
                                             "ok": bool(abs(dl - 0.01 * a1[0]["act_level"]) < 1e-3 and a1[0]["act_level"] > 0)}
src = open(os.path.join(root, "stack", "scripts", "v8f_mini_train.py"), encoding="utf-8", newline="").read()
old = 'T["total"] = T["total"] + a.w_act_level * ac_r["level"]'
assert src.count(old) == 1
os.makedirs(os.path.join(AL, "work", "mut_smoke", "stack", "scripts"), exist_ok=True)
mp = os.path.join(AL, "work", "mut_smoke", "stack", "scripts", "v8f_mini_train.py")
open(mp, "w", encoding="utf-8", newline="").write(src.replace(old, 'T["total"] = T["total"]'))
_, m1, _ = train("s1_mutV", ["--w-act-level", "0.01"], 1, script=mp, log_every=1)
dm = m1[0]["total"] - a0[0]["total"]
res["mutation_level_add_deleted"] = {"delta_vs_V0": dm, "act_level_still_logged": m1[0]["act_level"], "red": bool(abs(dm - 0.01 * m1[0]["act_level"]) > 1e-3)}
v, c, st, mu = res["V_finetune"], res["V0_control"], res["step1_total_moves_by_V_times_level"], res["mutation_level_add_deleted"]
res["PASS"] = bool(v["rc"] == 0 and v["all_logged_values_finite"] and v["act_level_in_every_row"] and v["act_level_n_positive_share"] >= 0.99 and v["selfcheck_levers"].get("w_act_level") == 0.01 and v["probe_pass"]
                   and v["ckpt_cfg_equal_to_base_cfg"] and v["ckpt_cfg_equal_to_V0_cfg"] and v["state_keys_equal_to_V0"] and v["state_keys_equal_to_base"] and v["weights_differ_V_vs_V0_max_abs"] > 0.0
                   and c["rc"] == 0 and c["no_act_level_keys_in_log"] and c["no_act_level_in_selfcheck"] and c["no_level_key_in_probe"] and c["no_act_level_in_summary"] and st["ok"] and mu["red"])
with open(os.path.join(HERE, "a4_level_smoke.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, default=str)[:3000])
print("A4L", "PASS" if res["PASS"] else "FAIL")
