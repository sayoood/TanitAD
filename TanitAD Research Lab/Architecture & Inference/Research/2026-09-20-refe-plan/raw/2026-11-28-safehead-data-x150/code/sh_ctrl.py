#!/usr/bin/env python3
"""Controls for the SAFE-HEAD cache (each can FAIL; results -> W/ctrl/ctrl_results.json). No PDMS aggregate is computed or printed.

    sh_ctrl.py prep  [--logs 6 --per-log 10]   pick navtest tokens whose pixels are local; write ctrl/job_ctrl.json, ctrl/commands_ctrl.json, ctrl/bank_ctrl.npz
    sh_ctrl.py pick                            CTRL-PICK + proposals/logits agreement vs the banked C1 navtest seam dump (stage2_p5c_c1.proposals.npz)
    sh_ctrl.py label                           CTRL-LABEL: tracked labels of the PICKED proposal vs the harness csv (refe_stage2_p5c_c1.csv), max abs diff per component
    sh_ctrl.py logit <out.npz>...              CTRL-LOGIT on final cache files: score_head(s) from the checkpoint's head weights vs cached logits
    sh_ctrl.py det <a.npz> <b.npz>             CTRL-DET: byte identity of two dumps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_lib as L  # noqa: E402

CT = L.wp("ctrl")
BANK_NPZ = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz"
BANK_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p5c_c1/refe_stage2_p5c_c1.csv"
EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_FRAMES = "E:/Projects/TanitAD/data/refe_navtest/frames"
TEST_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"


def record(name, res):
    p = os.path.join(CT, "ctrl_results.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = res
    json.dump(d, open(p, "w"), indent=1, default=str)
    print(f"ZZCTRL {name} {'PASS' if res.get('pass') else 'FAIL'} " + json.dumps({k: v for k, v in res.items() if k != 'pass'}, default=str)[:600], flush=True)


def prep(a):
    os.makedirs(CT, exist_ok=True)
    exp = L.load_json_gz(EXPORT)["tokens"]
    bank = np.load(BANK_NPZ)
    btok = [str(t) for t in bank["token"]]
    bidx = {t: i for i, t in enumerate(btok)}
    by_log = {}
    for t in btok:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = []
    for lg in sorted(by_log):
        if len(logs) >= a.logs:
            break
        if not all(os.path.exists(f"{TEST_FRAMES}/{lg}_{c}.zip") for c in L.CAMS):
            continue
        db = f"{TEST_DB}/{lg}.db"
        if not os.path.exists(db):
            continue
        ok = L.guard_tokens(db, sorted(by_log[lg]))
        if len(ok) >= a.per_log:
            step = max(1, len(ok) // a.per_log)
            logs.append({"log": lg, "db": db, "split": "navtest_ctrl", "tokens": ok[::step][:a.per_log]})
    cmds = {}
    for e in logs:
        for t in e["tokens"]:
            cmds[f"{e['log']}/{t}"] = exp[t]["ego_statuses"][-1]["driving_command"]
    json.dump({"frames": TEST_FRAMES, "yaw": L.YAW_TEST, "commands": os.path.join(CT, "commands_ctrl.json"), "logs": logs}, open(os.path.join(CT, "job_ctrl.json"), "w"))
    json.dump(cmds, open(os.path.join(CT, "commands_ctrl.json"), "w"))
    toks = [t for e in logs for t in e["tokens"]]
    ix = [bidx[t] for t in toks]
    np.savez(os.path.join(CT, "bank_ctrl.npz"), token=np.array(toks), pick=bank["pick"][ix], proposals=bank["proposals"][ix], logits=bank["logits"][ix])
    print(f"prep: {len(toks)} navtest tokens over {len(logs)} logs")


def _load_dumps(pattern_dir):
    out = {}
    for f in sorted(os.listdir(pattern_dir)):
        if f.endswith(".dump.npz"):
            d = np.load(os.path.join(pattern_dir, f))
            for i, t in enumerate(d["token"]):
                out[str(t)] = {k: d[k][i] for k in ("pick", "props8", "logits", "s")}
    return out


def pick(a):
    bank = np.load(os.path.join(CT, "bank_ctrl.npz"))
    mine = _load_dumps(a.dumps)
    toks = [str(t) for t in bank["token"]]
    have = [i for i, t in enumerate(toks) if t in mine]
    eq = sum(int(mine[toks[i]]["pick"]) == int(bank["pick"][i]) for i in have)
    dp = max(float(np.abs(mine[toks[i]]["props8"] - bank["proposals"][i]).max()) for i in have)
    dl = max(float(np.abs(mine[toks[i]]["logits"] - bank["logits"][i]).max()) for i in have)
    record("CTRL-PICK", {"pass": len(have) >= 50 and eq == len(have), "n": len(have), "pick_equal": eq, "props8_max_abs_diff": dp, "logits_max_abs_diff": dl,
                         "bank": BANK_NPZ, "bar": ">= 50 tokens, 100 % pick equality"})


def label(a):
    import csv
    mine = {}
    for f in sorted(os.listdir(a.lab)):
        if f.endswith(".lab.npz"):
            d = np.load(os.path.join(a.lab, f))
            for i, t in enumerate(d["token"]):
                mine[str(t)] = {k: d[k][i] for k in L.LAB_KEYS}
    dumps = _load_dumps(a.dumps)
    rows = {r["token"]: r for r in csv.DictReader(open(BANK_CSV, encoding="utf-8"))}
    cmap = {"tr_nc": "no_at_fault_collisions", "tr_dac": "drivable_area_compliance", "tr_ep": "ego_progress", "tr_ttc": "time_to_collision_within_bound",
            "tr_c": "comfort", "tr_ddc": "driving_direction_compliance", "tr_pdms": "score"}
    diffs = {k: 0.0 for k in cmap}
    sums = {k: 0.0 for k in cmap}
    nbad = {k: 0 for k in cmap}
    n = 0
    worst = {}
    for t, m in mine.items():
        r = rows.get(t)
        if r is None or t not in dumps:
            continue
        k = int(dumps[t]["pick"])
        n += 1
        for key, col in cmap.items():
            d = abs(float(m[key][k]) - float(r[col]))
            sums[key] += d
            nbad[key] += int(d > 1e-4)
            if d > diffs[key]:
                diffs[key], worst[key] = d, t
    mx = max(diffs.values()) if diffs else None
    means = {k: (v / n if n else None) for k, v in sums.items()}
    exact = all(diffs[k] <= 1e-4 for k in ("tr_nc", "tr_dac", "tr_ttc", "tr_c", "tr_ddc"))
    # strict bar (1e-4 on every component) is reported; the PASS bar is the one the programme validated its REBUILT-cache labeller against
    # (PREREG_SFT4.md:19-22: DAC/C/DDC 100 %, NC 99.93 %, TTC 99.89 %, PDMS mean |d| 0.0008 -- "a rebuilt cache is not NAVSIM's file"):
    # the five non-progress components agree to 1e-4 on every token and EP / PDMS mean |d| <= 1e-3.
    record("CTRL-LABEL", {"pass": n >= 50 and exact and means["tr_ep"] <= 1e-3 and means["tr_pdms"] <= 1e-3, "strict_1e-4_all_components_pass": n >= 50 and mx <= 1e-4, "n": n,
                          "max_abs_diff_per_component": diffs, "mean_abs_diff_per_component": means, "n_tokens_over_1e-4_per_component": nbad, "worst_token": worst, "csv": BANK_CSV,
                          "bar": "pass: NC/DAC/TTC/C/DDC <= 1e-4 on every picked proposal and EP, PDMS mean |d| <= 1e-3 (PREREG_SFT4.md:19-22 rebuilt-cache validation); strict 1e-4 reported beside it"})


def logit(a):
    import torch
    sd = torch.load(L.C1_CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
    W = sd["score_head.weight"].float()
    B = sd["score_head.bias"].float()
    worst, n = 0.0, 0
    for p in a.files:
        d = np.load(p)
        s = torch.from_numpy(d["s"]).float()
        with torch.no_grad():
            y = (s @ W.T + B).numpy()
        worst = max(worst, float(np.abs(y - d["logits"]).max()))
        n += len(d["token"])
    record("CTRL-LOGIT", {"pass": worst <= 1e-5, "n_tokens": n, "max_abs_diff": worst, "bar": "<= 1e-5 (s stored float32)"})


def det(a):
    ha, hb = (hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (a.files[0], a.files[1]))
    record("CTRL-DET", {"pass": ha == hb, "sha256_a": ha, "sha256_b": hb, "files": a.files})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "pick", "label", "logit", "det"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dumps", default=L.wp("ctrl", "dump"))
    ap.add_argument("--lab", default=L.wp("ctrl", "lab"))
    ap.add_argument("--logs", type=int, default=6)
    ap.add_argument("--per-log", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(CT, exist_ok=True)
    {"prep": prep, "pick": pick, "label": label, "logit": logit, "det": det}[a.cmd](a)
