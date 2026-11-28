def sub(p,old,new,cnt=1):
    t=open(p,encoding="utf-8",newline="").read()
    crlf="\r\n" in t
    if crlf: old=old.replace("\n","\r\n"); new=new.replace("\n","\r\n")
    assert t.count(old)==cnt,(p,t.count(old),old[:70]); open(p,"w",encoding="utf-8",newline="").write(t.replace(old,new))
for p in ("patch_scripts/p3_strat1_check.py","tree/stack/scripts/strat1_check.py"):
    sub(p,'''    for key in ("identity_terms_differing", "identity_outputs_differing"):
        if pr.get(key) != []:
            bad.append(f"aux_route_probe''','''    for key in ("identity_outputs_differing", "identity_terms_differing"):
        if pr.get(key) != []:
            bad.append(f"aux_route_probe''')
sub("tree/stack/tests/test_strat1_n1dh_check.py",'''('    for key in ("identity_terms_differing", "identity_outputs_differing"):\n        if pr.get(key) != []:\n            bad.append(f"aux_route_probe', '    for key in ():''','''('    for key in ("identity_outputs_differing", "identity_terms_differing"):\n        if pr.get(key) != []:\n            bad.append(f"aux_route_probe', '    for key in ():''')
sub("tree/stack/tests/test_strat1_n1dw_check.py","REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS and \"--rb-window-w\"","REAL.D_KINDS + REAL.W_KINDS + REAL.C_KINDS + REAL.H_KINDS and \"--rb-window-w\"")
