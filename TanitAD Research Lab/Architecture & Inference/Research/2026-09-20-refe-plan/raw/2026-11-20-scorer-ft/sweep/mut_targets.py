"""MM 2026-11-20: static check of source-mutation tables. For every tuple literal (file, old, new, ...) in a test whose first element names a
refe source file, count `old` in that file in the pristine (p) and patched (q) trees. A target found exactly once in p but not exactly once in
q is a mutation the patch broke (its full mode would raise 'mutation target not found exactly once').  usage: mut_targets.py <p refe> <q refe>"""
import ast, pathlib, sys
P, Q = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SRC = {"launch_gate.py", "gate_checks.py", "train.py", "rank_loss.py", "weight_rule_rank.py", "online_labeller.py", "model.py", "p5c_postgate.py"}
bad, n = [], 0
for t in sorted(Q.glob("test_*.py")):
    try:
        tree = ast.parse(t.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print("PARSE_FAIL", t.name, e); continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) >= 3 and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts[:3]):
            f, old = node.elts[0].value, node.elts[1].value
            if f not in SRC or not old:
                continue
            n += 1
            cp = (P / f).read_text(encoding="utf-8").count(old) if (P / f).is_file() else -1
            cq = (Q / f).read_text(encoding="utf-8").count(old) if (Q / f).is_file() else -1
            if cq != 1 and cp == 1:
                bad.append((t.name, f, node.lineno, cp, cq, old[:90]))
            elif cq != 1:
                print("ALREADY_NOT_ONCE", t.name, f, node.lineno, cp, cq, repr(old[:70]))
print("targets checked", n)
for b in bad:
    print("BROKEN", b)
print("ZZMUT_BROKEN", len(bad))
