def edit(P, pairs):
    s=open(P,encoding="utf-8",newline="").read()
    for a,b in pairs:
        assert s.count(a)==1,(P,a[:60]); s=s.replace(a,b)
    open(P,"w",encoding="utf-8",newline="").write(s)
R="C:/Users/Admin/v8proof_c/motion/"
edit(R+"tree_patch/stack/tanitad/models/v8f_mini.py",[('            if cfg.wm_l1_diag == "on":                                      # MOTION diagnostics: logged only, detached, not in the total\n',
 '            if cfg.wm_l1_roll_k > 0:\n                T["wm_roll"] = wm["roll/L1"].detach()                       # MOTION lever B: the rollout term, logged (and required by G-LIVE) whenever it is declared; it is inside T["wm"] already\n'
 '            if cfg.wm_l1_diag == "on":                                      # MOTION diagnostics: logged only, detached, not in the total\n')])
edit(R+"tree_patch/stack/scripts/v8f_mini_train.py",[('    if cfg.wm != "off":\n        expect.add("wm")\n','    if cfg.wm != "off":\n        expect.add("wm")\n    if cfg.wm_l1_roll_k > 0:\n        expect.add("wm_roll")                                            # MOTION lever B: a declared rollout term that never appears is a launch blocker\n')])
edit(R+"checks/m6_selfcheck.py",[('rec["declared_vs_built"] in ([], None)','rec["declared_vs_built"] in ([], None, "PASS")')])
