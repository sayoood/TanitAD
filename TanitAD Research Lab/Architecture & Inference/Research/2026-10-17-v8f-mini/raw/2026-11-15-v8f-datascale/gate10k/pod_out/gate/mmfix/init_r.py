"""init_r.py -- MM 2026-11-17: the G-HYG / C-INIT record of R_s0 (pod, CPU), which the gate's init item never built (it built B_s0, U_s0 and B_rep).

Same function as the gate (gate_items.build_init: the trainer's own order, seed, build_data, model, CPU), compared with the B_s0 record of
gate/init.json by the checker's own hyg_problems.  init.json is NOT rewritten: the B_s0 / U_s0 / B_rep tokens' evidence refers to it.
Writes gate/init_r.json (record: B_s0 copied from init.json, R_s0 built here, c_init_report, the sha256 of init.json).
Regression arm: an R_s0 record with one parameter more must read RED (as the gate's own init item).
The checker is imported from study/ (gate/ holds a stale copy; it is appended to sys.path AFTER the PYTHONPATH entries)."""
import copy
import hashlib
import json
import os
import sys

D = "/root/dscale10k"
G = D + "/gate"
sys.path.append(G)
import dscale10k_check as C  # noqa: E402
import gate_items as GI  # noqa: E402

if os.path.realpath(C.__file__) != os.path.realpath(D + "/study/dscale10k_check.py") or os.path.realpath(GI.C.__file__) != os.path.realpath(C.__file__):
    sys.exit(f"REFUSED: checker imported from {C.__file__} / {GI.C.__file__}, not {D}/study/dscale10k_check.py")
init_p = G + "/init.json"
ini = json.load(open(init_p))
if not ini.get("pass"):
    sys.exit("REFUSED: gate/init.json does not pass")
base = ini["record"]["B_s0"]
r = GI.build_init(f"{D}/argv/R_s0.A.json", "R_s0")
print("R_s0", r["n_trainable"], r["params_only_sha256"][:12], flush=True)
probs = list(C.hyg_problems(r, base))
rep = dict(ini["record"]["c_init_report"])
rep["B_s0_vs_R_s0_params_equal"] = r["params_only_sha256"] == base["params_only_sha256"]
rep["B_s0_vs_R_s0_state_equal"] = r["state_sha256"] == base["state_sha256"]
rep["B_s0_vs_R_s0_buffers_equal"] = r["buffers_sha256"] == base["buffers_sha256"]
bad = copy.deepcopy(r)
bad["n_trainable"] += 1
regs = [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))]
GI.save("init_r", {"B_s0": base, "R_s0": r, "c_init_report": rep, "init_json_sha256": hashlib.sha256(open(init_p, "rb").read()).hexdigest()}, probs, regs)
