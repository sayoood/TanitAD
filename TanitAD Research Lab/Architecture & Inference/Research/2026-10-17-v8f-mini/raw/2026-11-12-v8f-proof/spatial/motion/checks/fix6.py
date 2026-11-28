P="C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tests/test_v8f_motion_levers.py"
s=open(P,encoding="utf-8",newline="").read()
a='    assert not [k for k in off["terms"] if k.startswith(("wm_pred", "wm_copy", "wm_roll"))]\n'
b='    assert not [k for k in off["terms"] if k.startswith(("wm_pred", "wm_copy", "wm_roll_"))]      # with K > 0 the plain rollout term wm_roll is logged (G-LIVE requires it); the diagnostics are not\n    assert "wm_roll" in off["terms"] and abs(float(off["terms"]["wm_roll"]) - float(on["terms"]["wm_roll_l1"])) < 1e-6\n'
assert s.count(a)==1; s=s.replace(a,b)
a=s[s.index('    a, cfg, model, opt, info = _built(world, GRID)\n    res = TR.live_check(model, dict(world["batch"]), cfg)\n    assert "wm_roll" not in res["terms"]'):]
b='    a, cfg, model, opt, info = _built(world, GRID)\n    res = TR.live_check(model, dict(world["batch"]), cfg)\n    assert "wm_roll" not in res["terms"] and res["missing_terms"] == []         # no lever, no key\n'
s=s.replace(a,b)
open(P,"w",encoding="utf-8",newline="").write(s)
