import sys
sys.argv = ["x", "F:/TanitAD-scratch/compose_fix/mut", "F:/TanitAD-scratch/compose_fix/tmp/dry_out.txt"]
src = open("F:/TanitAD-scratch/compose_fix/tmp/run_mutations.py", encoding="utf-8").read()
assert src.rstrip().endswith("main()")
src = src.rstrip()[: -len("main()")]
g = {}
exec(compile(src, "run_mutations.py", "exec"), g)
TREE = g["TREE"]
bad = 0
for name, rule, edits, k in g["M"]:
    texts = {}
    for f, old, new in edits:
        texts.setdefault(f, open(f"{TREE}/{f}", encoding="utf-8", newline="").read())
    for f, old, new in edits:
        c = texts[f].count(old)
        if c != 1:
            bad += 1
            print("ANCHOR PROBLEM", name, f, c, repr(old[:70]))
print("mutants:", len(g["M"]), "anchor problems:", bad)
print("distinct -k selections (controls):", len({m[3] for m in g["M"]}))
