"""c7 fine-tune + eval smoke (LEADCOND), tiny synthetic world, CPU. Base = the tiny LS-shape checkpoint (30 steps of tree_base). Fine-tune 20 steps (--init-ckpt BASE --lr 1e-4 --warmup 100, the arm flags) with
(a) --lead-cond on, (b) --lead-cond on --w-rule-own 4.0 --rule-own-form l1: rc 0, every logged loss term finite, ckpt cfg = the base's plus lead_cond on, state keys = the base's plus the two lead_cond keys,
the two keys moved off zero. v8f_t1_eval.py of tree_patch loads each fine-tuned checkpoint (strict) and runs (finite dumps). With lead_cond weights ZERO (the base checkpoint + two zero keys + cfg lead_cond on) the eval
dumps equal the base checkpoint's, array by array (exact); control: the fine-tuned checkpoint (weights moved) gives dumps that differ from the base's. Usage: c7_finetune_eval.py"""
import json
import os
import subprocess

import numpy as np
import torch

import lc_common as C

LC = C.LC
CK = f"{LC}/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
root = f"{LC}/work/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(LC, "work", "tmp"), TEMP=w(LC, "work", "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
LSF = ["--lead-sup", "on", "--w-lead-sup", "1.0"]
BASE = C.BASE_CKPT
O = f"{LC}/work/tmp/c7"
os.makedirs(O, exist_ok=True)
W = f"{LC}/work/tmp/world_c3"                      # the synthetic world written by c3 (eval split, tracks, grid, join)
res = {}
bck = torch.load(BASE, map_location="cpu", weights_only=False)
SHAPES = {"LC": ["--lead-cond", "on"], "LCO": ["--lead-cond", "on", "--w-rule-own", "4.0", "--rule-own-form", "l1"]}
for tag, extra in SHAPES.items():
    out = f"{O}/ft_{tag}"
    r = subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", *TINY, *G, *LSF, "--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "100", "--seed", "0", "--steps", "20", "--log-every", "5", "--ckpt-every", "100", *extra, "--out", out],
                       env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/c7_ft_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-2000:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    rows = [json.loads(line) for line in open(f"{out}/train_log.jsonl", encoding="utf-8")]
    ck = torch.load(f"{out}/ckpt_last.pt", map_location="cpu", weights_only=False)
    fin = all(bool(torch.isfinite(torch.tensor(x[k]))) for x in rows for k in x if isinstance(x[k], (int, float)) and k not in ("sec", "sec_per_step", "elapsed"))
    cfg_diff = {k: (bck["cfg"].get(k), ck["cfg"].get(k)) for k in set(bck["cfg"]) | set(ck["cfg"]) if bck["cfg"].get(k) != ck["cfg"].get(k)}
    ev = f"{O}/ev_{tag}"
    os.makedirs(ev, exist_ok=True)
    e = subprocess.run([PY, f"{root}/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", f"{out}/ckpt_last.pt", "--tag", tag, "--tracks", f"{W}/native_tracks.pt", "--eval-cache", W, "--eval-grid", f"{W}/eval_grid_windows.json",
                        "--eval-join", f"{W}/join_eval.npz", "--out-dir", ev, "--max-windows", "4", "--batch", "4", "--n-boot", "20", "--device", "cpu", "--no-analyze"], env=env, capture_output=True, text=True, encoding="utf-8")
    dumps = {f: np.load(f"{ev}/dump_{tag}/{f}", allow_pickle=True) for f in sorted(os.listdir(f"{ev}/dump_{tag}"))} if e.returncode == 0 else {}
    ev_fin = bool(dumps) and all(np.isfinite(np.asarray(v[k], dtype=np.float64)).all() for v in dumps.values() for k in v.files if np.asarray(v[k]).dtype.kind == "f")
    res[tag] = {"train_rc": r.returncode, "n_log": len(rows), "loss_finite": fin, "ckpt_cfg_diff_vs_base": cfg_diff, "state_keys_extra": sorted(set(ck["model"]) - set(bck["model"])), "state_keys_missing": sorted(set(bck["model"]) - set(ck["model"])),
                "lead_cond_weight_abs_sum_after": float(ck["model"]["lead_cond.weight"].abs().sum()), "lead_cond_bias_abs_sum_after": float(ck["model"]["lead_cond.bias"].abs().sum()), "init_ckpt_n_mismatch": (ck.get("init_ckpt") or {}).get("n_mismatch_after_load"),
                "init_ckpt_fresh_keys": (ck.get("init_ckpt") or {}).get("fresh_keys"), "eval_rc": e.returncode, "eval_dump_files": len(dumps), "eval_finite": ev_fin, "eval_tail": (e.stdout or e.stderr or "")[-200:].replace("\n", " ")}
    print(tag, json.dumps(res[tag])[:1200])
# base eval and the zero-weight eval
def evaluate(ckpt, tag):
    ev = f"{O}/ev_{tag}"
    os.makedirs(ev, exist_ok=True)
    e = subprocess.run([PY, f"{root}/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ckpt, "--tag", tag, "--tracks", f"{W}/native_tracks.pt", "--eval-cache", W, "--eval-grid", f"{W}/eval_grid_windows.json",
                        "--eval-join", f"{W}/join_eval.npz", "--out-dir", ev, "--max-windows", "4", "--batch", "4", "--n-boot", "20", "--device", "cpu", "--no-analyze"], env=env, capture_output=True, text=True, encoding="utf-8")
    assert e.returncode == 0, (e.stdout or "")[-500:] + (e.stderr or "")[-1500:]
    return {f: np.load(f"{ev}/dump_{tag}/{f}", allow_pickle=True) for f in sorted(os.listdir(f"{ev}/dump_{tag}"))}


zk = dict(bck)
zk["model"] = dict(bck["model"])
zk["model"]["lead_cond.weight"] = torch.zeros(32, 2)
zk["model"]["lead_cond.bias"] = torch.zeros(32)
zk["cfg"] = {**bck["cfg"], "lead_cond": "on"}
torch.save(zk, f"{O}/zero_lead_cond.pt")
D0, DZ = evaluate(BASE, "base"), evaluate(f"{O}/zero_lead_cond.pt", "zero")
DM = evaluate(f"{O}/ft_LC/ckpt_last.pt", "moved")


def maxdiff(a, b):
    m = 0.0
    n = 0
    for f in a:
        for k in a[f].files:
            x, y = np.asarray(a[f][k]), np.asarray(b[f][k])
            if x.dtype.kind == "f":
                n += x.size
                m = max(m, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())) if x.size else m
            else:
                assert (x == y).all() if x.dtype.kind in "iub" else True
    return m, n


res["zero_weights_eval"] = {"files": len(D0), "max_abs_diff_vs_base_ckpt": maxdiff(D0, DZ)[0], "n_float_values": maxdiff(D0, DZ)[1], "control_fine_tuned_ckpt_max_abs_diff_vs_base": maxdiff(D0, DM)[0]}
ok = all(v["train_rc"] == 0 and v["loss_finite"] and v["eval_rc"] == 0 and v["eval_finite"] and v["state_keys_extra"] == ["lead_cond.bias", "lead_cond.weight"] and not v["state_keys_missing"]
         and v["ckpt_cfg_diff_vs_base"] == {"lead_cond": (None, "on")} and v["lead_cond_weight_abs_sum_after"] > 0.0 and v["init_ckpt_n_mismatch"] == 0 for k, v in res.items() if k in SHAPES)
ok = ok and res["zero_weights_eval"]["max_abs_diff_vs_base_ckpt"] == 0.0 and res["zero_weights_eval"]["control_fine_tuned_ckpt_max_abs_diff_vs_base"] > 0.0
res["PASS"] = bool(ok)
C.dump(f"{CK}/c7_finetune_eval.json", res)
print(json.dumps(res["zero_weights_eval"]), "C7", res["PASS"])
