"""init_b3.py -- DSCALE10K block 3 (pod, CPU): the G-HYG / C-INIT record of UE_s0 and BL_s0 (seed 0, as U_s0 / B_s0) (same function as the gate's init item and mmfix/init_r.py: gate_items.build_init = the trainer's own order,
seed, build_data, model, on CPU), compared with the B_s0 record of gate/init.json by the checker's own hyg_problems. Also refuses when the pod tree is no longer the tree the block-1 G-SUITE item
ran on (its record carries tree_manifest_sha256): the block-3 tokens reuse the block-1 G-SUITE / G-EVAL / G-CKPT items and may do so only on that tree.
Writes gate/init_b3.json (the record: B_s0 copied from init.json, B_s1, U_s1, c_init_report). init.json is not rewritten. Regression arm: a record with one parameter more must read RED.
The checker is imported from block3/study/ (the MM-ruled copy; gate/ holds a stale copy); PYTHONSAFEPATH=1 is set by the caller."""
import copy
import hashlib
import json
import os
import sys

D = "/root/dscale10k"
G = D + "/gate"
B3 = D + "/block3"
sys.path.insert(0, B3 + "/study")
sys.path.append(G)
import dscale10k_check as C  # noqa: E402
import gate_items as GI  # noqa: E402

if os.path.realpath(C.__file__) != os.path.realpath(B3 + "/study/dscale10k_check.py") or os.path.realpath(GI.C.__file__) != os.path.realpath(C.__file__):
    sys.exit(f"REFUSED: checker imported from {C.__file__} / {GI.C.__file__}, not {B3}/study/dscale10k_check.py")
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
for arm in ("UE_s0", "BL_s0"):
    out[arm] = GI.build_init(f"{D}/argv/{arm}.A.json", arm)
    print(arm, out[arm]["n_trainable"], out[arm]["params_only_sha256"][:12], flush=True)
    probs += C.hyg_problems(out[arm], base)
out["c_init_report"] = {"BL_s0_vs_B_s0_params_equal": out["BL_s0"]["params_only_sha256"] == base["params_only_sha256"],
                        "BL_s0_vs_UE_s0_params_equal": out["BL_s0"]["params_only_sha256"] == out["UE_s0"]["params_only_sha256"],
                        "BL_s0_vs_UE_s0_state_equal": out["BL_s0"]["state_sha256"] == out["UE_s0"]["state_sha256"],
                        "BL_s0_vs_UE_s0_buffers_equal": out["BL_s0"]["buffers_sha256"] == out["UE_s0"]["buffers_sha256"],
                        "tree_manifest_sha256": tree_now, "init_json_sha256": hashlib.sha256(open(G + "/init.json", "rb").read()).hexdigest(),
                        "reading": "params_equal true = the same step-0 weights (the lever is data alone); false = Delta is read as data plus init (card 2.3). Seed 0 = B_s0's seed: the step-0 weights should equal B_s0's and UE_s0's."}
bad = copy.deepcopy(out["UE_s0"])
bad["n_trainable"] += 1
GI.save("init_b3", out, probs, [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))])
