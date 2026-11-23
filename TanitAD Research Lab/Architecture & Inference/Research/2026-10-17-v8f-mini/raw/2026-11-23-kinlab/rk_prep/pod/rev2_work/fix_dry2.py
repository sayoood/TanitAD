p = "test_chain_dry_rk.py"
s = open(p, encoding="utf-8").read()
old = "        vals[n.targets[0].id] = ast.literal_eval(n.value)"
new = "        vals[n.targets[0].id] = eval(compile(ast.Expression(n.value), 'x', 'eval'), {'BSHA': vals.get('BSHA')})"
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
