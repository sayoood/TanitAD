import sys
p=sys.argv[1]; s=open(p,encoding='utf-8',newline='').read()
old="""            slope = ((ir_c * ar_c).mean(dim=(0, 1)) / (ar_c.pow(2).mean(dim=(0, 1))).clamp_min(1e-12)).clamp_min(RSP_S_MIN)   # (2,)
"""
new=old+"""        slope = slope.to(base.dtype)                                                 # a no-op in float32; in float64 (the analytic-identity test) it keeps 2 * delta * slope out of float32 promotion rules
"""
assert s.count(old)==1; s=s.replace(old,new); open(p,'w',encoding='utf-8',newline='').write(s)
p=sys.argv[2]; s=open(p,encoding='utf-8',newline='').read()
old="""                      "rel_err": abs(diff - want) / max(abs(want), 1e-30)}
                c3["pass"] = bool(c3["unweighted_equal"] and c3["levels_23_equal"] and c3["ref_has_no_rsp_keys"] and c3["n_unweighted"] > 0 and c3["rel_err"] <= 1e-5)
"""
new="""                      "rel_err": abs(diff - want) / max(abs(want), 1e-30)}
                c3["tol"] = max(1e-5 * abs(want), 8.0 * 1.1920929e-07 * abs(terms_first["wm/level/L1"]))   # 1e-5 relative to the term, or 8 float32 ulps of level/L1 (level/L1 ~ 100 at step 0 and is a float32 sum: its own rounding is ~1e-5 of a term of size 0.5)
                c3["pass"] = bool(c3["unweighted_equal"] and c3["levels_23_equal"] and c3["ref_has_no_rsp_keys"] and c3["n_unweighted"] > 0 and abs(diff - want) <= c3["tol"] and want > 0.0)
"""
assert s.count(old)==1; s=s.replace(old,new); open(p,'w',encoding='utf-8',newline='').write(s)
