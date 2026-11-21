"""G4 e2e analysis step (g4_e2e.py crashed in its own analysis block: the --lean dump stores no `proposals` rows, only logits + pick; the three model runs had finished and are on disk).
Reads the logs/npz/report of runs A, B, C and writes g4_e2e_result.json."""
import json, os
import numpy as np
OUT = "C:/Users/Admin/par18/p6gaps/raw/g4_e2e"
plan = json.load(open(f"{OUT}/plan.json", encoding="utf-8"))
cmds = json.load(open("D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json", encoding="utf-8"))
names = {(1, 0, 0, 0): "LEFT", (0, 1, 0, 0): "STRAIGHT", (0, 0, 1, 0): "RIGHT"}
cmd_of = {t: names[tuple(cmds[f"{plan['token_log'][t]}/{t}"])] for t in plan["tokens"]}
res = {"tokens": [{"token": t, "log": plan["token_log"][t], "command": cmd_of[t]} for t in plan["tokens"]]}
txtA = open(f"{OUT}/A_nocmd.log", encoding="utf-8", errors="replace").read()
res["A_no_commands"] = {"refused_marker": "REFUSED" in txtA, "ok_marker": "ZZS0DUMP_OK" in txtA, "rc": 2, "rc_source": "g4_e2e.out line 'A_nocmd rc 2 62s'"}
rep = {k: json.load(open(f"{OUT}/{k}.report.json")) for k in ("B_cmds", "C_swapped")}
for k, tag in (("B_cmds", "B_commands"), ("C_swapped", "C_swapped_commands")):
    txt = open(f"{OUT}/{k}.log", encoding="utf-8", errors="replace").read()
    r = rep[k]
    res[tag] = {"ok_marker": "ZZS0DUMP_OK 3 0" in txt, "nav_mode": r.get("nav_mode"), "commands_set": (r.get("commands") or {}).get("set"), "table_rows": (r.get("commands") or {}).get("rows"),
                "table_sha256": (r.get("commands") or {}).get("sha256"), "misses": r.get("misses"), "seconds": r.get("seconds")}
with np.load(f"{OUT}/B_cmds.npz") as b, np.load(f"{OUT}/C_swapped.npz") as c:
    tb, tc = [str(x) for x in b["token"]], [str(x) for x in c["token"]]
    rows = []
    for i, t in enumerate(tb):
        j = tc.index(t)
        d = float(np.abs(b["logits"][i].astype(np.float64) - c["logits"][j].astype(np.float64)).max())
        rows.append({"token": t, "command": cmd_of[t], "max_abs_logit_diff_B_vs_C": d, "pick_B": int(b["pick"][i]), "pick_C": int(c["pick"][j]), "logits_bit_identical": bool(np.array_equal(b["logits"][i], c["logits"][j]))})
res["swap_control"] = rows
res["swap_control_ok"] = all(r["logits_bit_identical"] if r["command"] == "STRAIGHT" else (not r["logits_bit_identical"]) for r in rows)
res["verdict_ok"] = bool(res["A_no_commands"]["refused_marker"] and not res["A_no_commands"]["ok_marker"] and res["B_commands"]["ok_marker"] and res["B_commands"]["commands_set"] == 3
                         and res["B_commands"]["nav_mode"] == "command" and res["C_swapped_commands"]["ok_marker"] and res["swap_control_ok"])
json.dump(res, open(f"{OUT}/g4_e2e_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
