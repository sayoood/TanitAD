p = "test_launch_gate.py"
s = open(p, encoding="utf-8").read()
old = '''    ev = cons(r0, add={"G14_something_new": {"ok": True}})                                                   # (G13 is a KNOWN gate since SAFEOBJ: the unknown-gate fixture moves to G14)'''
assert s.count(old) == 1
s = s.replace(old, '''    ev = cons(r0, add={"G16_something_new": {"ok": True}})                                                   # (G13 is a KNOWN gate since SAFEOBJ, G14 / G15 since P6-HN-S: the unknown-gate fixture moves to G16)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
