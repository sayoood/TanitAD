p = "tree/stack/tests/test_strat1_n1dh_check.py"
t = open(p, encoding="utf-8", newline="").read()
old = r'''"gate_drop_identity_clause": ('    for key in ("identity_terms_differing", "identity_outputs_differing"):'''
new = r'''"gate_drop_identity_clause": ('    for key in ("identity_outputs_differing", "identity_terms_differing"):'''
assert t.count(old) == 1
open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))
print("ok")
