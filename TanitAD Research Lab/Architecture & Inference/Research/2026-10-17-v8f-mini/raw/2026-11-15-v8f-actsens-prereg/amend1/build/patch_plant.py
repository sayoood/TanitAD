import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''"--wm-idm-gamma" "--wm-sampler"''', '''"--wm-idm-gamma" "live_first" "--wm-sampler"''')
rep('''    terms = dict(r35_terms(init))
    if idm:                                                                       # the IDM arm: unweighted terms equal R35's (same init, same first batch), level/L1 = R35's + (G1 - 100) x idm/L1
        terms["wm/level/L1"] = terms["wm/level/L1"] + (idm[0] - 100.0) * terms["wm/idm/L1"]
''', '''    terms = dict(r35_terms(init))
    first = None
    if idm:                                                                       # AMEND_ACTSENS_1: a lever-NAMED launch records live_first = G-LIVE step 0 (taken before any update): R35's terms x 0.9 (a stand-in batch), the dose only on level/L1
        first = {k: v * 0.9 for k, v in terms.items()}
        first["wm/level/L1"] = first["wm/level/L1"] + (idm[0] - 100.0) * first["wm/idm/L1"]
        if idm[0] != 100.0:                                                       # live.terms is the LAST of three AdamW steps: the weights have moved under the dose, so it does NOT equal R35's (what stopped the chain at 02:38)
            terms = {k: v * 1.003 for k, v in terms.items()}
            terms["wm/level/L1"] = terms["wm/level/L1"] + (idm[0] - 100.0) * terms["wm/idm/L1"]
''')
rep('''    if idm:
        lv["wm_idm_gamma_readback"] = {"declared": idm, "built": idm, "wm_aux_gamma": 100.0}
''', '''    if idm:
        lv["wm_idm_gamma_readback"] = {"declared": idm, "built": idm, "wm_aux_gamma": 100.0}
        sc["live_first"] = first
''')
rep('''    elif mode == "bad_steps":''', '''    elif mode == "drifted_first" and idm:                                         # live_first taken from the LAST step (the mutation of the trainer test): equals live.terms
        sc["live_first"] = dict(terms)
    elif mode == "other_batch_first" and idm:                                     # live_first from another batch: one unweighted term differs
        sc["live_first"]["wm/pred/L2"] = sc["live_first"]["wm/pred/L2"] * 1.0000001
    elif mode == "ref_term_changed" and idm:                                      # a REF whose live.terms has ONE term changed (a flipped last bit is enough)
        sc["live"]["terms"]["wm/sigreg/L3"] = sc["live"]["terms"]["wm/sigreg/L3"] * (1 + 1e-12)
    elif mode == "no_first" and idm:
        del sc["live_first"]
    elif mode == "bad_steps":''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
