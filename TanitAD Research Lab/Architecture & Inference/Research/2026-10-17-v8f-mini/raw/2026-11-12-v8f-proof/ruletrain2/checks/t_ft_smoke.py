"""Fine-tune smoke of the three new argv shapes (GOW, GOW4, GRW6), CPU, tiny synthetic model, from the tiny base checkpoint (stand-in of G_s0 / G_s1).
For each shape: (1) --selfcheck-only with the card argv as the arms file will name it (the card steps, lr 1e-4, warmup 100, the flags): rc 0, declared_vs_built PASS, init_ckpt read back with 0 mismatching tensors, the lever strings;
(2) a real 20-step fine-tune of the same argv with --steps 20: rc 0, every logged loss finite, the checkpoint cfg equals the base's, the lever keys in the log.
Usage: t_ft_smoke.py"""
import json
import os
import subprocess

import torch

S = "F:/TanitAD-scratch/v8proof/ruletrain2/work"
CK = "F:/TanitAD-scratch/v8proof/ruletrain2/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
TR = f"{root}/stack/scripts/v8f_mini_train.py"
BASE = f"{S}/tmp/ft/base/ckpt_last.pt"
O = f"{S}/tmp/r3ft_smoke"
SHAPES = {"GOW_s0": (2000, ["--w-rule-own", "1.0"], "0"), "GOW_s1": (2000, ["--w-rule-own", "1.0"], "1"), "GOW4_s0": (2000, ["--w-rule-own", "4.0"], "0"),
          "GRW6_s0": (6000, ["--rule-relabel", "on", "--rule-relabel-weight", "4"], "0"), "GRW6_s1": (6000, ["--rule-relabel", "on", "--rule-relabel-weight", "4"], "1")}


def run(tag, args, out):
    r = subprocess.run([PY, TR, *TINY, *G, *args, "--out", out], env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/ft_smoke_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-2500:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r


bck = torch.load(BASE, map_location="cpu", weights_only=False)
res = {}
for tag, (steps, flags, seed) in SHAPES.items():
    common = ["--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "100", "--seed", seed, *flags]
    r = run(f"sc_{tag}", [*common, "--steps", str(steps), "--selfcheck-only"], f"{O}/sc_{tag}")
    sc = json.load(open(f"{O}/sc_{tag}/selfcheck.json", encoding="utf-8")) if r.returncode == 0 else {}
    lv = sc.get("levers") or {}
    ic = lv.get("init_ckpt") or {}
    r2 = run(f"run_{tag}", [*common, "--steps", "20", "--log-every", "5", "--ckpt-every", "100"], f"{O}/run_{tag}")
    rows = [json.loads(line) for line in open(f"{O}/run_{tag}/train_log.jsonl", encoding="utf-8")]
    ck = torch.load(f"{O}/run_{tag}/ckpt_last.pt", map_location="cpu", weights_only=False)
    rec = {"selfcheck_rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "init_ckpt_n_mismatch_after_load": ic.get("n_mismatch_after_load"), "steps_total": sc.get("steps_total"),
           "levers_w_rule_own": lv.get("w_rule_own"), "levers_rule_relabel_weight": lv.get("rule_relabel_weight"), "levers_rule_relabel": lv.get("rule_relabel"),
           "info_rule_relabel_n": (sc.get("info") or {}).get("rule_relabel_n"), "probe_n_lead": (sc.get("rule_own_probe") or {}).get("n_lead"), "probe_n_active": (sc.get("rule_own_probe") or {}).get("n_active"),
           "run_rc": r2.returncode, "n_log": len(rows), "loss_finite": all(bool(torch.isfinite(torch.tensor(x[k]))) for x in rows for k in ("ade", "total")), "total": [x["total"] for x in rows],
           "log_keys": sorted({k for x in rows for k in x if "rule_own" in k or "weighted" in k or "rule_relabel" in k}), "ckpt_cfg_equals_base": ck["cfg"] == bck["cfg"], "init_ckpt_n_mismatch": (ck.get("init_ckpt") or {}).get("n_mismatch_after_load")}
    rec["PASS"] = bool(rec["selfcheck_rc"] == 0 and rec["declared_vs_built"] == "PASS" and rec["init_ckpt_n_mismatch_after_load"] == 0 and rec["steps_total"] == steps and rec["run_rc"] == 0 and rec["loss_finite"]
                       and rec["n_log"] == 4 and rec["ckpt_cfg_equals_base"] and rec["init_ckpt_n_mismatch"] == 0)
    res[tag] = rec
    print(tag, json.dumps(rec))
res["PASS"] = all(v["PASS"] for v in res.values())
json.dump(res, open(f"{CK}/t_ft_smoke.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T_FT_SMOKE", "PASS" if res["PASS"] else "FAIL")
