#!/usr/bin/env python3
"""Prerequisite 2: is the read-out's fit pair-compatible with the DAG-OACC / DAG-ORC bases? CPU only; reads checkpoints (torch.load, our own files), writes JSON.

Reference = the fit's X_s (M2b-mixo s0 / s1: the checkpoints SPEEDREAD/SPEEDSUB pinned). Candidates = DAG-OACC s0/s1 (= DAG-ORC's CTRL), and for completeness DAG-ORC A / REP s0/s1.
Checks: ckpt sha256; cfg.kin_mode / hist / n_tok / d_feat / grid; built_cfg diff (SPEEDAVG pair_violations' own function); state-dict keys+shapes; cons_mu / cons_sd bitwise;
the five speed-cell code indices; the argv levers that decide the ego block (kin-mode, grid, exclude, speed-train sidecar)."""
import hashlib
import importlib.util
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TREE = "F:/TanitAD-scratch/speedsub_tmp/tree"
sys.path.insert(0, TREE + "/stack")
sys.path.insert(0, TREE + "/stack/scripts")
sys.path.insert(0, TREE + "/taniteval")
import torch  # noqa: E402

torch.set_num_threads(2)
OUT = "F:/TanitAD-scratch/speedsub_t1_prep"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


REF = {0: "C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt", 1: "C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt"}
REF_SHA = {0: "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", 1: "68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915"}
A = "E:/Projects/TanitAD-artifacts"
CAND = {
    "DAGOACC_s0(CTRL)": (0, f"{A}/v8f-dagoacc/M2b_s0_dagoacc"), "DAGOACC_s1(CTRL)": (1, f"{A}/v8f-dagoacc/M2b_s1_dagoacc"),
    "DAGORC_A_s0": (0, f"{A}/v8f-dag-orc-oacc/M2b_s0_dagoaccorc"), "DAGORC_A_s1": (1, f"{A}/v8f-dag-orc-oacc/M2b_s1_dagoaccorc"),
    "DAGORC_REP_s0": (0, f"{A}/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep"), "DAGORC_REP_s1": (1, f"{A}/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep"),
}

import v8f_mini_eval_speedavg as SA  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

res = {"schema": "speedsub-t1-prep-compat-v1", "tree": TREE, "model_py_sha256": sha(TREE + "/stack/tanitad/models/v8f_mini.py"), "speed_idx_tip_module": SA.speed_idx(M),
       "n_code": int(M.N_GOAL + M.N_LAT + M.N_LON), "ref": {}, "cand": {}}
ref_ck = {}
for s in (0, 1):
    ref_ck[s] = load(REF[s])
    res["ref"][s] = {"path": REF[s], "sha256": sha(REF[s]), "sha_ok": sha(REF[s]) == REF_SHA[s]}
    r = ref_ck[s]
    c = r["cfg"]
    res["ref"][s].update({"kin_mode": c.get("kin_mode", "steer(default)"), "hist": c.get("hist"), "grid": r.get("grid"), "cfg_keys": sorted(c), "ckpt_keys": sorted(r.keys()),
                          "cons_mu": [float(v) for v in r["model"]["cons_mu"].double()[:3]], "n_cons": int(r["model"]["cons_mu"].numel())})
res["ref_pair_violations"] = SA.pair_violations(ref_ck[0], ref_ck[1])
ci = list(M.CI[n] for n in SA.SPEED_CELLS)
res["speed_cell_cons_index"] = dict(zip(SA.SPEED_CELLS, ci))
for name, (s, d) in CAND.items():
    p = f"{d}/ckpt_last.pt"
    ck = load(p)
    r = ref_ck[s]
    ms = ck["model"]
    out = {"path": p, "sha256": sha(p), "sums_file_sha256": [l.split()[0] for l in open(f"{d}/SHA256SUMS.txt") if l.strip().endswith("ckpt_last.pt")][0]}
    out["sha_matches_SHA256SUMS"] = out["sha256"] == out["sums_file_sha256"]
    c = ck["cfg"]
    out.update({"kin_mode": c.get("kin_mode", "steer(default)"), "hist": c.get("hist"), "grid": ck.get("grid"), "ckpt_keys": sorted(ck.keys())})
    out["pair_violations_vs_ref_same_seed"] = SA.pair_violations(r, ck)
    b0, b1 = SA.built_cfg(r["cfg"]), SA.built_cfg(c)
    out["built_cfg_diff_vs_ref"] = {k: [b0.get(k), b1.get(k)] for k in sorted(set(b0) | set(b1)) if b0.get(k) != b1.get(k)}
    out["state_keys_equal_ref"] = set(ms) == set(r["model"])
    out["state_shapes_equal_ref"] = all(tuple(ms[k].shape) == tuple(r["model"][k].shape) for k in ms if k in r["model"])
    for k in ("cons_mu", "cons_sd"):
        out[f"{k}_bitwise_equal_ref_s0"] = bool(torch.equal(ms[k], ref_ck[0]["model"][k]))
        out[f"{k}_bitwise_equal_ref_s1"] = bool(torch.equal(ms[k], ref_ck[1]["model"][k]))
        out[f"{k}_max_abs_dev_vs_ref_same_seed"] = float((ms[k].double() - r["model"][k].double()).abs().max())
    out["speed_cells_mu_sd"] = {"mu": [float(ms["cons_mu"][i]) for i in ci], "sd": [float(ms["cons_sd"][i]) for i in ci]}
    # run record: argv levers that decide the ego block
    av = json.load(open(f"{d}/argv.json"))["argv"]
    kv = {a: (av[i + 1] if i + 1 < len(av) else None) for i, a in enumerate(av) if a.startswith("--")}
    out["train_argv"] = {k: kv.get(k) for k in ("--kin-mode", "--grid", "--exclude", "--seed", "--steps", "--init-ckpt", "--ego-perturb-p", "--dagger-p", "--p-gt-start", "--p-gt-end", "--w-acc", "--ego-perturb-with-dagger")}
    sm = json.load(open(f"{d}/summary.json"))
    out["summary"] = {k: sm.get(k) for k in ("n_clips", "n_windows", "excluded_sha256", "train_sha256", "train_window_list_sha256", "exclude")}
    out["summary_levers_kin_mode"] = sm["levers"].get("kin_mode")
    ini = sm["levers"].get("init_ckpt")
    out["init_ckpt"] = {k: ini.get(k) for k in ("path", "sha256", "applied", "source_arm", "source_seed", "source_step")} if ini else None
    res["cand"][name] = out
    print(name, out["sha256"][:12], out["kin_mode"], out["grid"], "viol:", out["pair_violations_vs_ref_same_seed"], flush=True)
    del ck
# parity of the reference run (what the fit extracted)
for s, d in ((0, "C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo"), (1, "C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo")):
    try:
        sm = json.load(open(f"{d}/summary.json"))
        res["ref"][s]["summary"] = {k: sm.get(k) for k in ("n_clips", "n_windows", "excluded_sha256", "train_sha256", "train_window_list_sha256", "exclude")}
        res["ref"][s]["summary_levers_kin_mode"] = sm["levers"].get("kin_mode")
        av = json.load(open(f"{d}/argv.json"))["argv"]
        kv = {a: (av[i + 1] if i + 1 < len(av) else None) for i, a in enumerate(av) if a.startswith("--")}
        res["ref"][s]["train_argv"] = {k: kv.get(k) for k in ("--kin-mode", "--grid", "--exclude", "--seed", "--steps", "--init-ckpt", "--code-mix", "--p-gt-start", "--p-gt-end")}
    except Exception as e:  # a second path is probed by the caller; record the failure instead of hiding it
        res["ref"][s]["summary_error"] = repr(e)
json.dump(res, open(f"{OUT}/compat.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
print("[done]", f"{OUT}/compat.json")
