"""G4 end-to-end on CPU (no GPU, no navtest): the G4 srttc_s0_dump.py on P6's epoch-6 checkpoint, S0 plan tokens (NAVTRAIN), three runs.
  A  no --commands            -> must be REFUSED (rc 2): the kept refusal
  B  --commands (P6 table)    -> must finish, report.commands.set == tokens, and every token's command must reach the model
  C  --commands (LEFT<->RIGHT swapped table) -> LEFT/RIGHT tokens' outputs must change, STRAIGHT tokens' outputs must be IDENTICAL (wiring control)
"""
import json, os, subprocess, sys, time
import numpy as np
PK = "C:/Users/Admin/par18/p6gaps/sandbox/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DUMP = f"{PK}/raw/2026-10-18-sel-rank/code/srttc_s0_dump.py"
CK = "D:/Projects/TanitAD-artifacts/p6dry/hf/run/ckpt_last.pt"
CMDS = "D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"
S0 = "C:/Users/Admin/srttc_s0"
OUT = "C:/Users/Admin/par18/p6gaps/raw/g4_e2e"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{PK}/eval")
import eval_checkpoint as EC
cmds = json.load(open(CMDS, encoding="utf-8"))
unit = "L1b95fda46f"
bt = json.load(open(f"{S0}/run/batches/{unit}/batch_tokens.json", encoding="utf-8"))
names = {(1, 0, 0, 0): "LEFT", (0, 1, 0, 0): "STRAIGHT", (0, 0, 1, 0): "RIGHT"}
pick, seen = [], set()
for t in bt["tokens"]:
    lg = bt["token_log"][t]
    c = names.get(tuple(cmds.get(f"{lg}/{t}", [])))
    if c and c not in seen:
        seen.add(c)
        pick.append((t, lg, c))
    if len(seen) == 3:
        break
print("picked", pick, flush=True)
toks = [t for t, _l, _c in pick]
plan = {"tokens": toks, "token_log": {t: l for t, l, _c in pick}, "token_root": {t: bt["token_root"][t] for t in toks}, "db": {l: bt["db"][l] for _t, l, _c in pick}}
json.dump(plan, open(f"{OUT}/plan.json", "w"), indent=1)
swap = {k: ([v[2], v[1], v[0], v[3]]) for k, v in cmds.items()}
json.dump(swap, open(f"{OUT}/commands_swapped.json", "w"))
env = EC.env_driverl()
env.update({"CUDA_VISIBLE_DEVICES": "-1", "P2HO_TREE": PK})
PYD = EC.DRIVERL_PY
res = {"tokens": [{"token": t, "log": l, "command": c} for t, l, c in pick]}


def run(tag, extra):
    t0 = time.time()
    cmd = [PYD, "-I", DUMP, "--ckpt", CK, "--plan", f"{OUT}/plan.json", "--out", f"{OUT}/{tag}.npz", "--device", "cpu", "--lean", "--arm", f"g4e2e_{tag}", *extra]
    with open(f"{OUT}/{tag}.log", "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=OUT)
    txt = open(f"{OUT}/{tag}.log", encoding="utf-8", errors="replace").read()
    print(tag, "rc", rc, f"{time.time() - t0:.0f}s", flush=True)
    return rc, txt


rcA, txtA = run("A_nocmd", [])
res["A_no_commands"] = {"rc": rcA, "refused": "REFUSED" in txtA}
rcB, txtB = run("B_cmds", ["--commands", CMDS])
repB = json.load(open(f"{OUT}/B_cmds.report.json")) if os.path.exists(f"{OUT}/B_cmds.report.json") else None
res["B_commands"] = {"rc": rcB, "ok_marker": "ZZS0DUMP_OK" in txtB, "report_commands": (repB or {}).get("commands"), "nav_mode": (repB or {}).get("nav_mode"), "misses": (repB or {}).get("misses")}
rcC, txtC = run("C_swapped", ["--commands", f"{OUT}/commands_swapped.json"])
if os.path.exists(f"{OUT}/B_cmds.npz") and os.path.exists(f"{OUT}/C_swapped.npz"):
    with np.load(f"{OUT}/B_cmds.npz") as b, np.load(f"{OUT}/C_swapped.npz") as c:
        tb = [str(x) for x in b["token"]]
        rows = []
        for i, t in enumerate(tb):
            j = [str(x) for x in c["token"]].index(t)
            d = float(np.abs(b["logits"][i].astype(np.float64) - c["logits"][j].astype(np.float64)).max())
            dp = float(np.abs(b["proposals"][i].astype(np.float64) - c["proposals"][j].astype(np.float64)).max())
            rows.append({"token": t, "command": dict((x[0], x[2]) for x in pick)[t], "max_abs_logit_diff": d, "max_abs_proposal_diff_m": dp})
        res["C_swap_control"] = rows
        res["C_swap_ok"] = all((r["max_abs_logit_diff"] == 0.0 and r["max_abs_proposal_diff_m"] == 0.0) if r["command"] == "STRAIGHT" else (r["max_abs_proposal_diff_m"] > 0.0) for r in rows)
res["C_rc"] = rcC
json.dump(res, open(f"{OUT}/g4_e2e_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
