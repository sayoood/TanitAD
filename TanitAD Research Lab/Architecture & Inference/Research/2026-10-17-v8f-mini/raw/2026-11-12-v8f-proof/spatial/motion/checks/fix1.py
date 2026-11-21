P="C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tests/test_v8f_motion_levers.py"
s=open(P,encoding="utf-8",newline="").read()
a="""    _, cfg2, m2, _, _ = _built(world, GRID + B_ON)
    m2.wm.stack.cfg[0].roll_weight = 0.0                                                            # the loss's own weight changed under an argv that declares 0.3
    assert any("w_wm_roll" in m for m in TR.declared_vs_built(_args(GRID + B_ON), cfg2, m2, opt, info))"""
b="""    a2, cfg2, m2, o2, i2 = _built(world, GRID + B_ON)
    m2.wm.stack.cfg[0].roll_weight = 0.0                                                            # the loss's own weight changed under an argv that declares 0.3
    assert any("w_wm_roll" in m for m in TR.declared_vs_built(a2, cfg2, m2, o2, i2))"""
assert s.count(a)==1; s=s.replace(a,b)
a="""    _, c_ab, m_ab, _, _ = _built(world, AB_CELL, **ov)
    with pytest.raises(ValueError, match="wm_l1_residual"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m_ab, c_ab)                  # an older checkpoint (key absent = off) is refused by an 'on' argv
"""
b="""    _, c_a, m_a, _, _ = _built(world, GRID + A_ON, **ov)
    with pytest.raises(ValueError, match=r"wm_l1_residual: 'off' != 'on'"):
        TR.load_init_ckpt(str(tmp_path / "grid" / "ckpt_last.pt"), m_a, c_a)                    # an older checkpoint (key absent = off) is refused by an 'on' argv
    _, c_ab, m_ab, _, _ = _built(world, AB_CELL, **ov)
"""
assert s.count(a)==1; s=s.replace(a,b)
open(P,"w",encoding="utf-8",newline="").write(s)
