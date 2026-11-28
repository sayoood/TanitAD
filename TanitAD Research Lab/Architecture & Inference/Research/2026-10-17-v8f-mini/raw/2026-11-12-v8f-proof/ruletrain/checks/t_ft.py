"""t_ft: the fine-tune path of card ADD32 section 3, on the CPU with the tiny synthetic model.
1. base: arm-G argv, seed 0, 6 steps -> ckpt (the stand-in of G_s0's ckpt_last.pt).
2. selfcheck-only with the card's flags `--init-ckpt <base> --steps 2000 --lr 1e-4 --warmup 100`, for seed 0 and seed 100, with --rule-relabel off and on: the gate accepts (rc 0, declared_vs_built PASS, init_ckpt read back with 0 mismatching tensors,
   lr_schedule built warmup 100 / peak 1e-4 / total 2000).
3. real (tiny) fine-tunes of 6 steps: seed 0 off, seed 100 off, seed 0 on: the starting weights equal the base (max abs diff 0.0 after load), the seed-100 data order and loss trace differ from seed 0, the config check of --init-ckpt passes for both seeds.
Usage: t_ft.py"""
import json
import os
import subprocess
import sys

import torch

S = "C:/Users/Admin/v8proof_c/ruletrain"
CK = "F:/TanitAD-scratch/v8proof/ruletrain/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{S}/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
TR = f"{root}/stack/scripts/v8f_mini_train.py"
O = f"{S}/tmp/ft_v3"


def run(tag, args, out):
    r = subprocess.run([PY, TR, *TINY, *G, *args, "--out", out], env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/ft_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-2500:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    return r


res = {}
r = run("base", ["--seed", "0", "--steps", "6", "--log-every", "1", "--ckpt-every", "100"], f"{O}/base")
res["base_rc"] = r.returncode
base = f"{O}/base/ckpt_last.pt"
bck = torch.load(base, map_location="cpu", weights_only=False)
res["selfchecks"] = {}
for seed in (0, 100):
    for rr in ("off", "on"):
        tag = f"sc_s{seed}_{rr}"
        ex = ["--init-ckpt", base, "--steps", "2000", "--lr", "1e-4", "--warmup", "100", "--seed", str(seed), "--selfcheck-only", "--rule-relabel", rr]
        r = run(tag, ex, f"{O}/{tag}")
        sc = json.load(open(f"{O}/{tag}/selfcheck.json", encoding="utf-8")) if r.returncode == 0 else {}
        ic = (sc.get("levers") or {}).get("init_ckpt") or {}
        res["selfchecks"][tag] = {"rc": r.returncode, "declared_vs_built": sc.get("declared_vs_built"), "live_pass": (sc.get("live") or {}).get("pass"), "init_ckpt_n_mismatch_after_load": ic.get("n_mismatch_after_load"),
                                  "init_ckpt_sha256": ic.get("sha256"), "lr_schedule": (sc.get("levers") or {}).get("lr_schedule"), "steps_total": sc.get("steps_total"), "rule_relabel": (sc.get("levers") or {}).get("rule_relabel"),
                                  "rule_relabel_n": (sc.get("info") or {}).get("rule_relabel_n")}
        with open(f"{CK}/ft_{tag}.selfcheck.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(sc, fh, indent=1, default=str)
        print(tag, json.dumps(res["selfchecks"][tag]))
res["fine_tunes"] = {}
for tag, seed, rr in (("s0_off", 0, "off"), ("s100_off", 100, "off"), ("s0_on", 0, "on")):
    ex = ["--init-ckpt", base, "--steps", "6", "--lr", "1e-4", "--warmup", "2", "--seed", str(seed), "--rule-relabel", rr, "--log-every", "1", "--ckpt-every", "100"]
    r = run("run_" + tag, ex, f"{O}/run_{tag}")
    rows = [json.loads(l) for l in open(f"{O}/run_{tag}/train_log.jsonl", encoding="utf-8")]
    ck = torch.load(f"{O}/run_{tag}/ckpt_last.pt", map_location="cpu", weights_only=False)
    res["fine_tunes"][tag] = {"rc": r.returncode, "n_log": len(rows), "ade": [x["ade"] for x in rows], "total": [x["total"] for x in rows],
                              "rule_relabel_n": [x.get("rule_relabel_n") for x in rows], "rule_relabel_changed": [x.get("rule_relabel_changed") for x in rows],
                              "cfg_keys_equal_base": sorted(ck["cfg"]) == sorted(bck["cfg"]), "ckpt_seed": ck["seed"],
                              "init_ckpt_n_mismatch": (ck.get("init_ckpt") or {}).get("n_mismatch_after_load"),
                              "max_abs_weight_move_from_base": max(float((ck["model"][k].float() - bck["model"][k].float()).abs().max()) for k in ck["model"])}
    print(tag, json.dumps({k: v for k, v in res["fine_tunes"][tag].items() if k in ("rc", "ade", "rule_relabel_n", "init_ckpt_n_mismatch")}))
f = res["fine_tunes"]
res["seed_changes_order"] = f["s0_off"]["ade"] != f["s100_off"]["ade"]
res["relabel_changes_loss"] = f["s0_off"]["ade"] != f["s0_on"]["ade"]
res["PASS"] = bool(res["base_rc"] == 0 and all(v["rc"] == 0 and v["declared_vs_built"] == "PASS" and v["init_ckpt_n_mismatch_after_load"] == 0 and v["steps_total"] == 2000
                                               and v["lr_schedule"] is not None for v in res["selfchecks"].values())
                   and all(v["rc"] == 0 and v["init_ckpt_n_mismatch"] == 0 and v["cfg_keys_equal_base"] for v in f.values())
                   and res["seed_changes_order"] and res["relabel_changes_loss"]
                   and all((x or 0) > 0 for x in f["s0_on"]["rule_relabel_n"]) and all(x is None for x in f["s0_off"]["rule_relabel_n"]))
json.dump(res, open(f"{CK}/t_ft.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T_FT", "PASS" if res["PASS"] else "FAIL")
