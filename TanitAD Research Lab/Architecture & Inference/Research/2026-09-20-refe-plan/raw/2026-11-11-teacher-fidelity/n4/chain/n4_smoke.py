#!/usr/bin/env python3
"""TF-N4 navtrain smoke of the section 7 mutation arms (NEVER navtest). Real arm + each mutation on the 5 RJ5 DAC failures + 20 unchanged tokens of the
same two logs (navtrain held-out, the DEV hold's data; the banked rows tar of TF-N2 reroll is the source). The N2-R plan stand-in = the dev R label;
the reference "harness CSV" stand-in = the banked W-TRANSP tracked rows of those labels (a navtrain identity check, NOT a harness check).

  python n4_smoke.py <workdir>      (env: N4_ROWS = dir with rolls/ and scores/ of RJ5, N4_W, N4_P5R, PYTHONPATH devkit + refe; see CHAIN_N4.md)
Writes <workdir>/smoke_results.json and prints one GREEN / RED line per arm."""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("N4_PY", sys.executable)
N_UNCH = 20


def run(args, log):
    r = subprocess.run([PY, os.path.join(HERE, "n4_nt_build.py")] + args, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, cwd=HERE)
    open(log, "w", encoding="utf-8").write(r.stdout + "\n--STDERR--\n" + r.stderr)
    return r.returncode, r.stdout + r.stderr


def setup(w):
    sys.path.insert(0, HERE)
    import numpy as np
    import n4_common as N
    import n4_geom as G
    S, Rl = N.load_scores(runs=("RJ5",))["RJ5"], N.load_rolls(runs=("RJ5",))["RJ5"]
    fail = sorted(t for t in S if S[t]["arms"]["R"]["tracked"]["dac"] == 0.0)
    logs = sorted({S[t]["log"] for t in fail})
    ok = sorted(t for t in S if S[t]["log"] in logs and S[t]["arms"]["R"]["tracked"]["dac"] == 1.0)[:N_UNCH]
    toks = fail + ok
    by: dict = {}
    for t in toks:
        by.setdefault(S[t]["log"], []).append(t)
    plans, tk, lg = {}, [], []
    for _lg, tok, cx in G.contexts(by):
        plans[tok] = cx.labels(np.asarray(Rl[tok]["traj20"], float))["R"]
    for t in toks:
        tk.append(t)
        lg.append(S[t]["log"])
    P = np.stack([plans[t] for t in tk])
    np.savez(f"{w}/n2r_plans.npz", tokens=np.array(tk), logs=np.array(lg), plans20=P, arm=np.array("N2R"))
    p8 = np.stack([G.ND.to_navsim(p).astype(np.float32) for p in P])
    for nm in ("t_seam", "n2r_seam"):
        np.savez(f"{w}/{nm}.npz", token=np.array(tk), fingerprint=np.array(["fp" + t for t in tk]), poses=p8, sampling=np.array([8, 0.5]), arm=np.array("smoke"))
    json.dump({"tokens": tk, "n_fail": len(fail), "n_unchanged": len(ok)}, open(f"{w}/tokens.json", "w", encoding="utf-8"))
    with open(f"{w}/n2r_ref.csv", "w", encoding="utf-8", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
                     "driving_direction_compliance", "score"])
        for i, t in enumerate(tk):
            m = S[t]["arms"]["R"]["tracked"]
            wr.writerow([i, t, True, m["nc"], m["dac"], m["ep"], m["ttc"], m["c"], m["ddc"], m["pdms"]])
    with open(f"{w}/rolls_RJ5.jsonl", "w", encoding="utf-8") as fo:
        for t in tk:
            fo.write(json.dumps({"token": t, "log": S[t]["log"], "traj20": Rl[t]["traj20"]}) + "\n")
    return fail, ok


def build(w, tag, mutate="none", allow=False):
    out = f"{w}/{tag}"
    os.makedirs(out, exist_ok=True)
    common = ["--route", "navtrain", "--n2r-plans", f"{w}/n2r_plans.npz", "--rolls", f"{w}/rolls_RJ5.jsonl"]
    d = ["derive"] + common + ["--out", f"{out}/derive_s0of1.jsonl", "--shard", "0/1", "--mutate", mutate] + (["--smoke-allow-mutation"] if allow else [])
    rc_d, txt_d = run(d, f"{out}/derive.log")
    if rc_d:
        return {"derive_rc": rc_d, "refused": "REFUSING" in txt_d, "tail": txt_d.strip().splitlines()[-1][:300]}
    m = ["merge"] + common + ["--rows", f"{out}/derive_s*of*.jsonl", "--seam", f"{w}/t_seam.npz", "--n2r-seam", f"{w}/n2r_seam.npz",
                              "--harness-csv-n2r", f"{w}/n2r_ref.csv", "--out-dir", f"{out}/out"]
    rc_m, txt_m = run(m, f"{out}/merge.log")
    rep = json.load(open(f"{out}/out/n4_build_report.json", encoding="utf-8"))
    return {"derive_rc": 0, "merge_rc": rc_m, "verdict": rep["verdict"], "controls": {k: v["pass"] for k, v in rep["controls"].items()},
            "repaired": {k: len(v) for k, v in rep["repaired"].items()}, "dac0": rep["dac0_tokens_builder"], "rate": rep["diagnostics"]["builder_repair_rate"],
            "unch": rep["controls"]["C-UNCH"].get("spurious_applied_on_dac1_tokens")}


def main() -> int:
    w = sys.argv[1]
    os.makedirs(w, exist_ok=True)
    fail, ok = setup(w)
    res = {"tokens": {"failing": len(fail), "unchanged": len(ok)}}
    res["real"] = build(w, "real")
    res["cert_off_guard"] = build(w, "cert_off_guard", "cert_off", False)
    res["cert_off"] = build(w, "cert_off", "cert_off", True)
    res["repair_all"] = build(w, "repair_all", "repair_all", True)
    res["margin_neg_guard"] = build(w, "margin_neg_guard", "margin_neg", False)
    res["margin_neg"] = build(w, "margin_neg", "margin_neg", True)
    # 2e-6 perturbation of ONE unchanged plan in the written files, then the on-disk C-UNCH
    import numpy as np
    pd = f"{w}/perturb"
    shutil.rmtree(pd, ignore_errors=True)
    shutil.copytree(f"{w}/real/out", pd)
    z = dict(np.load(f"{pd}/n4_plans_N4D.npz", allow_pickle=False))
    rep = json.load(open(f"{pd}/n4_build_report.json", encoding="utf-8"))
    un = [i for i, t in enumerate(z["tokens"]) if str(t) not in set(rep["repaired"]["N4D"])][0]
    z["plans20"] = z["plans20"].copy()
    z["plans20"][un, 3, 1] += 2e-6
    np.savez(f"{pd}/n4_plans_N4D.npz", **z)
    uargs = ["unch", "--out-dir", pd, "--n2r-plans", f"{w}/n2r_plans.npz", "--n2r-seam", f"{w}/n2r_seam.npz"]
    res["perturb_2e-6_unch_on_disk"] = {"rc": run(uargs, f"{pd}/unch.log")[0]}
    res["unperturbed_unch_on_disk"] = {"rc": run(["unch", "--out-dir", f"{w}/real/out", "--n2r-plans", f"{w}/n2r_plans.npz", "--n2r-seam", f"{w}/n2r_seam.npz"], f"{w}/real/unch.log")[0]}
    json.dump(res, open(f"{w}/smoke_results.json", "w", encoding="utf-8"), indent=1)
    r = res["real"]
    print("REAL   ", "GREEN" if r["verdict"] == "PASS" and r["repaired"]["N4D"] == len(fail) else "RED", json.dumps(r))
    for k in ("cert_off_guard", "margin_neg_guard"):
        print(k, "RED(refused)" if res[k].get("refused") else "NOT-RED", res[k].get("tail"))
    for k in ("cert_off", "repair_all", "margin_neg"):
        print(k, "RED" if res[k].get("verdict") == "FAIL" or (k == "margin_neg" and res[k]["rate"]["N4D"] < 0.75) else "NOT-RED", json.dumps(res[k]))
    print("perturb", "RED" if res["perturb_2e-6_unch_on_disk"]["rc"] else "NOT-RED", "| unperturbed", "GREEN" if res["unperturbed_unch_on_disk"]["rc"] == 0 else "RED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
