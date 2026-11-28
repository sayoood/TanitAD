"""init_reps2.py -- DSCALE10K REPS2 (pod, CPU): the G-HYG / C-INIT record of UE_s2 / UE_s3. usage: init_reps2.py ue_s2 | ue_s3   (cd $D first; PYTHONSAFEPATH=1; CUDA_VISIBLE_DEVICES= )
Writes gate/init_ue_s2.json (resp. _s3) on /root/dscale10k/tree (cdbff971, UE_s0's and UE_s1's tree): gate_items.build_init (the trainer's own order, the arm's seed, CPU) against B_s0's record of gate/init.json
by hyg_problems (cfg + 7,214,495 trainable parameters), and the C-INIT report against UE_s0's record (gate/init_b3.json) and UE_s1's (gate/init_ue_s1.json): the CONFIG must equal theirs; the step-0 state must DIFFER
from both (the seed took effect; there is no union seed-2/3 arm to equal, unlike UE_s1 vs U_s1) and, when UE_s2 and UE_s3 are both known, from each other. Refuses when the tree is not the tree the block-1 G-SUITE ran on.
Regression arm: one parameter more reads RED. The checker is imported from blockreps2/study (path order forced, realpath asserted).
``compare()`` is a pure function (tests/test_reps2_checker.py)."""
import copy
import hashlib
import json
import os
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")


def compare(arm, new, ue0, ue1, other_new, hyg):
    """Problems of the C-INIT report. ``hyg`` = the checker's hyg_problems(new, base) list. Every record carries cfg, params_only_sha256, state_sha256, buffers_sha256, n_trainable."""
    probs = list(hyg)
    if new["cfg"] != ue0["cfg"]:
        probs.append(f"C-INIT: {arm} cfg differs from UE_s0's")
    if new["cfg"] != ue1["cfg"]:
        probs.append(f"C-INIT: {arm} cfg differs from UE_s1's")
    if new["buffers_sha256"] != ue0["buffers_sha256"]:
        probs.append(f"C-INIT: {arm} buffers differ from UE_s0's (buffers are seed-free here)")
    for nm, r in (("UE_s0", ue0), ("UE_s1", ue1), ("the other new arm", other_new)):
        if r is not None and new["state_sha256"] == r["state_sha256"]:
            probs.append(f"C-INIT: {arm}'s step-0 state equals {nm}'s: the seed did not change the initialisation")
    return probs


def main():
    which = sys.argv[1]
    assert which in ("ue_s2", "ue_s3"), "init_reps2.py serves ue_s2 | ue_s3"
    arm = which.upper().replace("UE_S", "UE_s")
    G, RP, TREE = D + "/gate", D + "/blockreps2", D + "/tree"
    for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study", G, RP, RP + "/study"):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    sys.path.remove(G)
    sys.path.append(G)                      # gate/gate_items.py (TREE = tree); the blockreps2 copy of the checker must win over gate/dscale10k_check.py
    import gate_items as GI  # noqa: E402
    import dscale10k_check as C  # noqa: E402
    if os.path.realpath(C.__file__) != os.path.realpath(RP + "/study/dscale10k_check.py") or os.path.realpath(GI.C.__file__) != os.path.realpath(C.__file__):
        sys.exit(f"REFUSED: checker imported from {C.__file__} / {GI.C.__file__}, not {RP}/study/dscale10k_check.py")
    if GI.TREE != TREE:
        sys.exit(f"REFUSED: gate items TREE is {GI.TREE}, expected {TREE}")
    L = lambda p: json.load(open(p))
    ini, ib3, iu1 = L(G + "/init.json"), L(G + "/init_b3.json"), L(G + "/init_ue_s1.json")
    if not (ini.get("pass") and ib3.get("pass") and iu1.get("pass")):
        sys.exit("REFUSED: gate/init.json, init_b3.json or init_ue_s1.json does not pass")
    base, ue0, ue1 = ini["record"]["B_s0"], ib3["record"]["UE_s0"], iu1["record"]["UE_s1"]
    suite = L(G + "/suite.json")
    now = C.tree_manifest_sha(TREE)
    pre = []
    if suite["record"]["tree_manifest_sha256"] != now:
        pre.append(f"the pod tree manifest {now[:12]} is not the one G-SUITE ran on ({suite['record']['tree_manifest_sha256'][:12]})")
    new = GI.build_init(f"{D}/argv/{arm}.A.json", arm)
    print(arm, new["n_trainable"], new["params_only_sha256"][:12], new["state_sha256"][:12], flush=True)
    other = None
    op = G + "/init_ue_s3.json" if arm == "UE_s2" else G + "/init_ue_s2.json"
    if os.path.isfile(op):
        o = L(op)
        other = o["record"].get(("UE_s3" if arm == "UE_s2" else "UE_s2"))
    probs = pre + compare(arm, new, ue0, ue1, other, C.hyg_problems(new, base))
    rep = {f"{arm}_cfg_equal_UE_s0": new["cfg"] == ue0["cfg"], f"{arm}_cfg_equal_UE_s1": new["cfg"] == ue1["cfg"], f"{arm}_buffers_equal_UE_s0": new["buffers_sha256"] == ue0["buffers_sha256"],
           f"{arm}_state_equal_UE_s0": new["state_sha256"] == ue0["state_sha256"], f"{arm}_state_equal_UE_s1": new["state_sha256"] == ue1["state_sha256"],
           f"{arm}_params_equal_UE_s0": new["params_only_sha256"] == ue0["params_only_sha256"], "tree_manifest_sha256": now,
           "init_json_sha256": hashlib.sha256(open(G + "/init.json", "rb").read()).hexdigest(), "init_b3_json_sha256": hashlib.sha256(open(G + "/init_b3.json", "rb").read()).hexdigest(),
           "init_ue_s1_json_sha256": hashlib.sha256(open(G + "/init_ue_s1.json", "rb").read()).hexdigest(),
           "reading": f"{arm}: seed {new['seed']}; cfg and buffers equal UE_s0's, step-0 state DIFFERENT from seed 0 and seed 1 by design (no union arm of this seed to equal)"}
    bad = copy.deepcopy(new)
    bad["n_trainable"] += 1
    GI.save("init_" + arm.lower(), {arm: new, "B_s0": base, "UE_s0": ue0, "UE_s1": ue1, "c_init_report": rep}, probs, [GI.reg("hyg_param_count_differs", C.hyg_problems(bad, base))])


if __name__ == "__main__":
    main()
