"""mkforks_reps1.py (pod): derives the REPS1 forks from the seed-0 scripts by literal substitution, each asserted by occurrence count. Writes ONLY under /root/dscale10k/blockreps1/. Re-runnable.
  um_checks.py               -> um_checks_reps1.py            UM_s0 -> UM_s1, UE_s0 -> UE_s1, '/blockum' -> '/blockreps1'  (the K-TACSRC bar `tac_src_empty_batches > 5` is NOT touched)
  reach_probe_um.py          -> reach_probe_um_s1.py          UM_s0 -> UM_s1, UE_s0 -> UE_s1 (tenk_ref = probe2/UE_s1.A.json), checker path blockum/study -> blockreps1/study
  assemble_evidence_um.py    -> assemble_evidence_reps1_um.py UM_s0 -> UM_s1, UE_s0 -> UE_s1, item init_um -> init_um_s1 (our init record), checker unchanged
  assemble_evidence_b3.py    -> assemble_evidence_reps1_ue.py UE_s0 -> UE_s1, item init_b3 -> init_ue_s1
  gate_items_um.py           -> gate_items_um.py              verbatim copy (so blockreps1 imports its own, with the tree_um TREE)
  fake_leaf3.sh (block3)     -> see fake_leaf_reps1.sh (hand-written)
"""
import os
import shutil

D = "/root/dscale10k"
B = D + "/blockreps1"


def fork(src, dst, subs):
    t = open(src, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    for old, new, n in subs:
        assert t.count(old) == n, (src, old, t.count(old), n)
        t = t.replace(old, new)
    open(dst, "w", encoding="utf-8", newline="\n").write(t)
    print("wrote", dst)


def count(src, old):
    return open(src, encoding="utf-8", newline="").read().replace("\r\n", "\n").count(old)


UC = D + "/blockum/um_checks.py"
fork(UC, B + "/um_checks_reps1.py", [("UM_s0", "UM_s1", count(UC, "UM_s0")), ("UE_s0", "UE_s1", count(UC, "UE_s0")), ('BU = D + "/blockum"', 'BU = D + "/blockreps1"', 1)])
RP = D + "/blockum/reach_probe_um.py"
fork(RP, B + "/reach_probe_um_s1.py", [("UM_s0", "UM_s1", count(RP, "UM_s0")), ("UE_s0", "UE_s1", count(RP, "UE_s0")), ("/root/dscale10k/blockum/study", "/root/dscale10k/blockreps1/study", count(RP, "/root/dscale10k/blockum/study"))])
AU = D + "/blockum/assemble_evidence_um.py"
fork(AU, B + "/assemble_evidence_reps1_um.py", [("UM_s0", "UM_s1", count(AU, "UM_s0")), ("UE_s0", "UE_s1", count(AU, "UE_s0")), ('item("init_um")', 'item("init_um_s1")', 1)])
AB = D + "/block3/assemble_evidence_b3.py"
fork(AB, B + "/assemble_evidence_reps1_ue.py", [("UE_s0", "UE_s1", count(AB, "UE_s0")), ('item("init_b3")', 'item("init_ue_s1")', 1)])
shutil.copyfile(D + "/blockum/gate_items_um.py", B + "/gate_items_um.py")
print("copied gate_items_um.py")
