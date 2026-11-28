"""Generate the RBDPOD pod-side scripts (LF only) from the pod's own navrbd_chain.sh, the b4 gsuite/dump scripts and render_argv.py. Each replacement is asserted to hit the expected number of times."""
import os
import re

H = "F:/TanitAD-scratch/strat1/navrbd"
O = H + "/rbdpod/stage"
os.makedirs(O, exist_ok=True)
C0 = "0c3b374fb1036b0ca4ca7620f98137dd6c59cca4"
C1 = "5d474c9135298a40b3d50b56d611477d9fe97ca3"


def rd(p):
    return open(p, "rb").read().decode("utf-8").replace("\r\n", "\n")


def wr(name, s):
    assert "\r" not in s
    open(O + "/" + name, "wb").write(s.encode("utf-8"))


def sub(s, old, new, n):
    assert s.count(old) == n, (old, s.count(old), n)
    return s.replace(old, new)


# ---- chain
s = rd(H + "/rbdpod/pod_chain.sh")
s = sub(s, "TREE=$N/tree;", "TREE=$N/tree_fix;", 1)
s = sub(s, "COMMIT=" + C0, "COMMIT=" + C1, 1)
s = sub(s, "SUITE=$R/gsuite_result_navrbd.json", "SUITE=$R/gsuite_result_navrbd_fix.json", 1)
s = sub(s, "logs/navrbd_gsuite.log", "logs/navrbd_gsuite_fix.log", 2)
s = sub(s, "navrbd_tree.sha256", "navrbd_tree_fix.sha256", 2)
s = sub(s, "navrbd_tree_verify.log", "navrbd_tree_verify_fix.log", 2)
s = sub(s, "0c3b374f blobs", "5d474c91 blobs", 1)
s = sub(s, 'ARMS="N1D:s0 N1D:s1 N1Dx:s0"', 'ARMS="N1Dx:s0"', 1)
s = sub(s, "$N/render_argv.py", "$N/render_argv_fix.py", 1)
s = sub(s, "# STRAT1 NAVRBD chain", "# RBDPOD copy (tree_fix, commit 5d474c91): STRAT1 NAVRBD chain", 1)
wr("navrbd_chain_fix.sh", s)

# ---- render_argv
s = rd(H + "/b4/render_argv.py")
s = sub(s, 'TREE = "/root/strat1/navrbd/tree"', 'TREE = "/root/strat1/navrbd/tree_fix"', 1)
s = sub(s, "The tree is /root/strat1/navrbd/tree;", "The tree is /root/strat1/navrbd/tree_fix;", 1)
wr("render_argv_fix.py", s)

# ---- dump
s = rd(H + "/b4/dump_rbd.sh")
s = sub(s, "T=/root/strat1/navrbd/tree\n", "T=/root/strat1/navrbd/tree_fix\n", 1)
assert s.count("navrbd/tree_fix") == 1
wr("dump_rbd_fix.sh", s)

# ---- gsuite
s = rd(H + "/b4/navrbd_gsuite.sh")
s = sub(s, "TREE=$N/tree\nBT=$R/navrb2/tree_fix\n", 'SFX=${DRY:+_dry}\nTREE=$N/tree_fix$SFX\nBT=$N/tree\n', 1)
s = sub(s, "LOG=$R/logs/navrbd_gsuite.log", "LOG=$R/logs/navrbd_gsuite_fix$SFX.log", 1)
s = sub(s, "JB=$R/junit/navrb2_fix2\nJN=$R/junit/navrbd\nT=$R/tmp/navrbdsuite\n", "JB=$R/junit/navrbd\nJN=$R/junit/navrbd_fix$SFX\nT=$R/tmp/navrbdfixsuite$SFX\n", 1)
s = sub(s, "COMMIT=" + C0 + "\nBASE=e71782a3b48cc975ab4c2db9f559c4a3c99624b3\nNFILES=16", "COMMIT=" + C1 + "\nBASE=" + C0 + "\nNFILES=5", 1)
s = sub(s, "navrbd_tree.sha256", "navrbd_tree_fix.sha256", 1)
s = sub(s, "are not the 0c3b374f blobs", "are not the 5d474c91 blobs", 1)
s = sub(s, "the tree differs from tree_fix (e71782a3) in", "tree_fix differs from tree (0c3b374f) in", 1)
s = sub(s, "tree == e71782a3 + $NFILES NAVRBD files", "tree_fix == tree (0c3b374f) + $NFILES rbdfix files", 1)
s = sub(s, "(junit/navrb2_fix2 composition)", "(junit/navrbd composition)", 1)
s = sub(s, "--out $R/gsuite_result_navrbd.json > $R/logs/suite_compare_navrbd.log", "--out $R/gsuite_result_navrbd_fix$SFX.json > $R/logs/suite_compare_navrbd_fix$SFX.log", 1)
s = sub(s, "(see $R/gsuite_result_navrbd.json)", "(see $R/gsuite_result_navrbd_fix$SFX.json)", 2)
s = sub(s, "json.load(open('$R/gsuite_result_navrbd.json'))", "json.load(open('$R/gsuite_result_navrbd_fix$SFX.json'))", 1)
# the dry run stops after the import check (no pytest)
s = sub(s, 'for wave in "0 1 2 3" "4 5 6 7"; do', '[ -n "$DRY" ] && { echo "ZZ DRY OK ZZ (shards composed, import resolves in the tree, no pytest run)" >> "$LOG"; exit 0; }\nfor wave in "0 1 2 3" "4 5 6 7"; do', 1)
NL = chr(10)
assert not [l for l in s.split(NL) if "navrb2" in l and not l.startswith("#")]
s = "#!/bin/bash" + NL + "# RBDPOD copy of navrbd_gsuite.sh for tree_fix (5d474c91): base = junit/navrbd (the 0c3b374f run, today), 5 files differ; DRY=1 stops before pytest." + NL + s.split(NL, 1)[1]
wr("navrbd_gsuite_fix.sh", s)
print("generated", os.listdir(O))
