#!/usr/bin/env python
"""gen_argv_ktree.py (pod side): K-TREE argv of AMENDMENT UM-A3.
A = UE_s1's argv (argv/UE_s1.A.json) with --seed 1 (already), --steps 20, --log-every 1, --ckpt-every 20, --out ktree/A_tree
B = the same tokens, --out ktree/B_treeum (run on tree_um); asserts no --tac-src-* flag
C = A again, --out ktree/C_tree_rep
Writes argv/{A,B,C}.json and ARGV_DIFF.txt (full command lines, unified diff). Nothing here touches a trainer or a token."""
import difflib, json, os, sys

D = "/root/dscale10k"
K = D + "/ktree"
TREE = {"A": D + "/tree", "B": D + "/tree_um", "C": D + "/tree"}
OUT = {"A": K + "/A_tree", "B": K + "/B_treeum", "C": K + "/C_tree_rep"}
TRAINER = "stack/scripts/v8f_mini_train.py"

base = json.load(open(D + "/argv/UE_s1.A.json"))["argv"]
assert "--seed" in base and base[base.index("--seed") + 1] == "1", "UE_s1 argv is not seed 1"
bad = [t for t in base if t.startswith("--tac-src")]
assert not bad, "UE_s1 argv carries tac-src flags: %s" % bad


def setf(t, f, v):
    t = list(t)
    if f in t:
        t[t.index(f) + 1] = v
    else:
        t += [f, v]
    return t


t = list(base)
t = setf(t, "--steps", "20")
t = setf(t, "--log-every", "1")
t = setf(t, "--ckpt-every", "20")
os.makedirs(K + "/argv", exist_ok=True)
lines = {}
for k in "ABC":
    tk = setf(t, "--out", OUT[k])
    assert not [x for x in tk if x.startswith("--tac-src")], k
    json.dump({"argv": tk}, open("%s/argv/%s.json" % (K, k), "w"), indent=1)
    lines[k] = ["python " + TREE[k] + "/" + TRAINER] + [x for x in tk]
# one token per line, so the diff names the token
def render(k):
    return ["%s %s" % ("trainer", lines[k][0].split(" ", 1)[1])] + lines[k][1:]
out = []
for a, b in (("A", "B"), ("A", "C")):
    out += list(difflib.unified_diff(render(a), render(b), "%s (%s)" % (a, TREE[a]), "%s (%s)" % (b, TREE[b]), lineterm="", n=0))
    out.append("")
changed = sorted({l[1:] for l in out if l[:1] in "+-" and not l.startswith(("+++", "---"))})
out.append("tokens differing between A and B/C (besides the lines above): none other than the trainer path and --out value")
out.append("no --tac-src-* token in A, B or C: verified (assert)")
out.append("A tokens: %d, B tokens: %d, C tokens: %d" % tuple(len(lines[k]) - 1 for k in "ABC"))
open(K + "/ARGV_DIFF.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
