"""init_s1.py -- DSCALE10K block 2 (pod, CPU): the G-HYG / C-INIT record of B_s1 and U_s1 (same function as the gate's init item and mmfix/init_r.py: gate_items.build_init = the trainer's own order,
seed, build_data, model, on CPU), compared with the B_s0 record of gate/init.json by the checker's own hyg_problems. Also refuses when the pod tree is no longer the tree the block-1 G-SUITE item
ran on (its record carries tree_manifest_sha256): the seed-1 tokens reuse the block-1 G-SUITE / G-EVAL / G-CKPT items and may do so only on that tree.
Writes gate/init_s1.json (the record: B_s0 copied from init.json, B_s1, U_s1, c_init_report). init.json is not rewritten. Regression arm: a record with one parameter more must read RED.
The checker is imported from study/ (gate/ holds a stale copy); PYTHONSAFEPATH=1 is set by the caller."""
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
ini = json.load(open(G + "/init.json"))
if not ini.get("pass"):
    sys.exit("REFUSED: gate/init.json does not pass")
suite = json.load(open(G + "/suite.json"))
tree_now = C.tree_manifest_sha(D + "/tree")
probs = []
if suite["record"]["tree_manifest_sha256"] != tree_now:
    probs.append(f"the pod tree manifest {tree_now[:12]} is not the one G-SUITE ran on ({suite['record']['tree_manifest_sha256'][:12]}): re-run the block-1 suite item before minting")
base = ini["record"]["B_s0"]
out = {"B_s0": base}
for arm in ("B_s1", "U_s1"):
    out[arm] = GI.build_init(f"{D}/argv/{arm}.A.json", arm)
    print(arm, out[arm]["n_trainable"], out[arm]["params_only_sha256"][:12], flush=True)
    probs += C.hyg_problems(out[arm], base)
out["c_init_report"] = {"B_s1_vs_B_s0_params_equal": out["B_s1"]["params_only_sha256"] == base["params_only_sha256"],
                        "B_s1_vs_U_s1_params_equal": out["B_s1"]["params_only_sha256"] == out["U_s1"]["params_only_sha256"],
                        "B_s1_vs_U_s1_state_equal": out["B_s1"]["state_sha256"] == out["U_s1"]["state_sha256"],
                        "B_s1_vs_U_s1_buffers_equal": out["B_s1"]["buffers_sha256"] == out["U_s1"]["buffers_sha256"],
                        "tree_manifest_sha256": tree_now, "init_json_sha256": hashlib.sha256(open(G + "/init.json", "rb").read()).hexdigest(),
                        "reading": "params_equal true = the same step-0 weights (the lever is data alone); false = Delta is read as data plus init (card 2.3). Seed 1 differs from B_s0's seed 0 by design."}
bad = copy.deepcopy(out["U_s1"])
bad["n_trainable"] += 1
GI.save("init_s1", out, probs, [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))])
