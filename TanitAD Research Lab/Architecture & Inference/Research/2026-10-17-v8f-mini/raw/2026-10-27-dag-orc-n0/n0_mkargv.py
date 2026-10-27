"""N0 argv builder: DAG's banked deadreckon argv (from dagt1/logs/chain.out 'start DAG_s{s}_m10_causal_heads' line) with ONLY --tag / --out-dir / --dump-ticks / --nav changed.
Writes argv/N0_s{s}.argv (one token per line) and argv/DAGBANKED_s{s}.argv, and the diff."""
import re, shlex, sys, difflib, json, os
OUT = "E:/Projects/TanitAD-artifacts/dag-orc-n0"
LOG = "E:/Projects/TanitAD-artifacts/dagt1/logs/chain.out"
lines = open(LOG, encoding="utf-8").read().splitlines()
res = {}
for s in (0, 1):
    pref = f"start DAG_s{s}_m10_causal_heads"
    ln = [l for l in lines if pref in l]
    assert len(ln) == 1, (s, len(ln))
    cmd = ln[0].split("): ", 1)[1]
    banked = shlex.split(cmd, posix=True)
    new = list(banked)
    def setv(flag, val):
        i = new.index(flag); old = new[i + 1]; new[i + 1] = val; return old
    assert new[new.index("--nav") + 1] == "deadreckon"
    setv("--nav", "logged")
    setv("--tag", f"N0DAG_s{s}_m10_causal_heads_navlogged")
    setv("--out-dir", OUT)
    setv("--dump-ticks", f"{OUT}/ticks")
    open(f"{OUT}/argv/DAGBANKED_s{s}.argv", "w", newline="\n").write("\n".join(banked) + "\n")
    open(f"{OUT}/argv/N0_s{s}.argv", "w", newline="\n").write("\n".join(new) + "\n")
    # diff
    d = [(i, a, b) for i, (a, b) in enumerate(zip(banked, new)) if a != b]
    assert len(banked) == len(new)
    res[s] = {"n_tokens": len(new), "differences": [{"token_index": i, "banked": a, "n0": b, "flag": banked[i - 1]} for i, a, b in d]}
    print(f"s{s}: {len(new)} tokens, {len(d)} differing tokens: " + "; ".join(f"{banked[i-1]}: {a} -> {b}" for i, a, b in d))
json.dump(res, open(f"{OUT}/argv/ARGV_DIFF.json", "w"), indent=1)
