p = "test_launch_gate.py"
s = open(p, encoding="utf-8").read()
old = '''        rows = rls1_pf_rows(argv, add={k14: (pf_kw or {}).get("g14", hn_g14()), k15: (pf_kw or {}).get("g15", hn_g15(man_sha))}, data_mut=data_ev(), **{k: v for k, v in (pf_kw or {}).items() if k not in ("g14", "g15")})'''
new = '''        dropped = (pf_kw or {}).get("drop", ())
        adds = {k: v for g_, k, v in (("G14", k14, (pf_kw or {}).get("g14", hn_g14())), ("G15", k15, (pf_kw or {}).get("g15", hn_g15(man_sha)))) if g_ not in dropped}
        rows = rls1_pf_rows(argv, add=adds, data_mut=data_ev(), **{k: v for k, v in (pf_kw or {}).items() if k not in ("g14", "g15")})'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
