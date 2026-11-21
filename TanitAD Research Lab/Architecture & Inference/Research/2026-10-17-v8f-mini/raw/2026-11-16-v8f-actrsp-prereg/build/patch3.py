import sys
p=sys.argv[1]
s=open(p,encoding='utf-8',newline='').read()
def rep(old,new):
    global s
    assert s.count(old)==1,(old[:70],s.count(old)); s=s.replace(old,new)
rep('''    terms_first = None                                                # AMEND_ACTSENS_1: the terms of step 0, taken BEFORE any opt.step() (terms_seen is the LAST step's)
''','''    terms_first = None                                                # AMEND_ACTSENS_1: the terms of step 0, taken BEFORE any opt.step() (terms_seen is the LAST step's)
    c3 = None                                                         # PREREG_V8F_ACTRSP control C3: step 0 against the SAME init and batch with the response term switched off (lever-ON launches only)
    if cfg.wm_rsp_w > 0.0:
        with torch.no_grad():
            m_ref = copy.deepcopy(m)
            m_ref.wm.stack.cfg[0].rsp_w = 0.0                         # the reference is the flag-off build of this very copy: same weights, same batch, same seed below
            torch.manual_seed(0)
            ref_terms = {k: float(v.detach()) for k, v in m_ref.wm_only_loss(batch)["terms"].items()}
        del m_ref
''')
rep('''        if s == 0:
            terms_first = dict(terms_seen)                                # AMEND_ACTSENS_1: step 0 (mutation MA1 records the last step instead)
''','''        if s == 0:
            terms_first = dict(terms_seen)                                # AMEND_ACTSENS_1: step 0 (mutation MA1 records the last step instead)
            if cfg.wm_rsp_w > 0.0:                                        # PREREG_V8F_ACTRSP C3: every unweighted term equal, level/L1 differs by exactly rsp_w x rsp/L1 (1e-5 relative)
                unw = sorted(k for k in terms_first if k.startswith("wm/") and not k.startswith("wm/level/") and "rsp" not in k)
                diff = terms_first["wm/level/L1"] - ref_terms["wm/level/L1"]
                want = cfg.wm_rsp_w * terms_first["wm/rsp/L1"]
                c3 = {"n_unweighted": len(unw), "unweighted_equal": all(terms_first[k] == ref_terms.get(k) for k in unw), "level_L1_diff": diff, "rsp_w_x_rsp": want,
                      "levels_23_equal": all(terms_first[f"wm/level/L{l}"] == ref_terms[f"wm/level/L{l}"] for l in (2, 3)), "ref_has_no_rsp_keys": not any("rsp" in k for k in ref_terms),
                      "rel_err": abs(diff - want) / max(abs(want), 1e-30)}
                c3["pass"] = bool(c3["unweighted_equal"] and c3["levels_23_equal"] and c3["ref_has_no_rsp_keys"] and c3["n_unweighted"] > 0 and c3["rel_err"] <= 1e-5)
''')
rep('''    ok = (not missing and not dead_groups and not unexpected_dead and res["control_nonwm_grad_sum"] == 0.0 and res["n_wm_params"] > 0 and res["n_nonwm_params"] > 0)
''','''    if c3 is not None:                                                # PREREG_V8F_ACTRSP: lever-ON launches only (a flag-off live record keeps its key set)
        res["c3"] = c3
    ok = (not missing and not dead_groups and not unexpected_dead and res["control_nonwm_grad_sum"] == 0.0 and res["n_wm_params"] > 0 and res["n_nonwm_params"] > 0 and (c3 is None or c3["pass"]))
''')
open(p,'w',encoding='utf-8',newline='').write(s)
