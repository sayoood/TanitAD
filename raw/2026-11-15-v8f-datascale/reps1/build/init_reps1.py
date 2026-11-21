"""init_reps1.py -- DSCALE10K REPS1 (pod, CPU): the G-HYG / C-INIT record of the seed-1 pair. usage: init_reps1.py ue | um   (cd $D first; PYTHONSAFEPATH=1; CUDA_VISIBLE_DEVICES= )
ue -> gate/init_ue_s1.json  UE_s1 on /root/dscale10k/tree (cdbff971, UE_s0's tree): gate_items.build_init (the trainer's own order, seed 1, CPU) against B_s0's record of gate/init.json by hyg_problems;
                            C-INIT report: UE_s1's step-0 state / params / buffers sha256 against U_s1's record of gate/init_s1.json (the seed-1 union arm). Also refuses when the tree is not the tree the
                            block-1 G-SUITE ran on (as init_b3.py). Regression arm: one parameter more reads RED. (fork of block3/init_b3.py)
um -> gate/init_um_s1.json  UM_s1 on /root/dscale10k/tree_um (648d4ca3, UM_s0's tree): gate_items_um.build_init, against B_s0 by hyg_problems; C-INIT: step-0 state equals UE_s1's and U_s1's
                            (the stage draws no random number). Regression arm: one parameter more reads RED. (fork of blockum/init_um.py init())
The checker is imported from blockreps1/study (path order forced, realpath asserted)."""
import copy
import hashlib
import json
import os
import sys

D = "/root/dscale10k"
G = D + "/gate"
RP = D + "/blockreps1"
which = sys.argv[1]
TREE = D + ("/tree" if which == "ue" else "/tree_um")
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study", G, RP, RP + "/study"):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if which == "ue":
    sys.path.remove(G)
    sys.path.append(G)                      # gate/gate_items.py (TREE = tree); the study copy of the checker must win over gate/dscale10k_check.py
    import gate_items as GI  # noqa: E402
else:
    import gate_items_um as GI  # noqa: E402
import dscale10k_check as C  # noqa: E402

if os.path.realpath(C.__file__) != os.path.realpath(RP + "/study/dscale10k_check.py") or os.path.realpath(GI.C.__file__) != os.path.realpath(C.__file__):
    sys.exit(f"REFUSED: checker imported from {C.__file__} / {GI.C.__file__}, not {RP}/study/dscale10k_check.py")
if GI.TREE != TREE:
    sys.exit(f"REFUSED: gate items TREE is {GI.TREE}, expected {TREE}")


def L(p):
    return json.load(open(p))


ini, is1 = L(G + "/init.json"), L(G + "/init_s1.json")
if not (ini.get("pass") and is1.get("pass")):
    sys.exit("REFUSED: gate/init.json or gate/init_s1.json does not pass")
base, u1 = ini["record"]["B_s0"], is1["record"]["U_s1"]
probs = []
if which == "ue":
    suite = L(G + "/suite.json")
    now = C.tree_manifest_sha(TREE)
    if suite["record"]["tree_manifest_sha256"] != now:
        probs.append(f"the pod tree manifest {now[:12]} is not the one G-SUITE ran on ({suite['record']['tree_manifest_sha256'][:12]})")
    ue = GI.build_init(f"{D}/argv/UE_s1.A.json", "UE_s1")
    print("UE_s1", ue["n_trainable"], ue["params_only_sha256"][:12], ue["state_sha256"][:12], flush=True)
    probs += C.hyg_problems(ue, base)
    rep = {"UE_s1_params_equal_U_s1": ue["params_only_sha256"] == u1["params_only_sha256"], "UE_s1_state_equal_U_s1": ue["state_sha256"] == u1["state_sha256"],
           "UE_s1_buffers_equal_U_s1": ue["buffers_sha256"] == u1["buffers_sha256"], "UE_s1_cfg_equal_U_s1": ue["cfg"] == u1["cfg"], "tree_manifest_sha256": now,
           "init_json_sha256": hashlib.sha256(open(G + "/init.json", "rb").read()).hexdigest(), "init_s1_json_sha256": hashlib.sha256(open(G + "/init_s1.json", "rb").read()).hexdigest(),
           "reading": "seed 1 (differs from UE_s0's seed 0 by design); state equal to U_s1's = the same step-0 weights as the seed-1 union arm"}
    bad = copy.deepcopy(ue)
    bad["n_trainable"] += 1
    GI.save("init_ue_s1", {"UE_s1": ue, "B_s0": base, "U_s1": u1, "c_init_report": rep}, probs, [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))])
else:
    iue = L(G + "/init_ue_s1.json")
    if not iue.get("pass"):
        sys.exit("REFUSED: gate/init_ue_s1.json does not pass (run `init_reps1.py ue` first)")
    ue = iue["record"]["UE_s1"]
    um = GI.build_init(f"{D}/argv/UM_s1.A.json", "UM_s1")
    print("UM_s1", um["n_trainable"], um["params_only_sha256"][:12], um["state_sha256"][:12], flush=True)
    probs += C.hyg_problems(um, base)
    c = {"UM_s1_state_equals_UE_s1": um["state_sha256"] == ue["state_sha256"], "UM_s1_state_equals_U_s1": um["state_sha256"] == u1["state_sha256"],
         "UM_s1_params_equal_UE_s1": um["params_only_sha256"] == ue["params_only_sha256"], "UM_s1_buffers_equal_UE_s1": um["buffers_sha256"] == ue["buffers_sha256"],
         "UM_s1_cfg_equal_UE_s1": um["cfg"] == ue["cfg"], "state_sha256": um["state_sha256"], "ue_state_sha256": ue["state_sha256"], "u_state_sha256": u1["state_sha256"],
         "reading": "C-INIT: UM_s1's step-0 state-dict sha256 equals UE_s1's and U_s1's (the new stage draws no random number); G-HYG: cfg equal to UE_s1's, 7,214,495 trainable"}
    for k, v in c.items():
        if k.startswith("UM_s1_") and v is not True:
            probs.append(f"C-INIT: {k} is {v}")
    bad = copy.deepcopy(um)
    bad["n_trainable"] += 1
    GI.save("init_um_s1", {"UM_s1": um, "B_s0": base, "UE_s1": ue, "c_init_report": c, "tree_manifest_sha256": C.tree_manifest_sha(TREE)}, probs, [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))])
