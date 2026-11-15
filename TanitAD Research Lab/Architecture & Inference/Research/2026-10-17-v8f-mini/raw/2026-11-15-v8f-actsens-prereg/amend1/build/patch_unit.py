import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''def put(p, data=b""):''', '''def GP(t, toks, sc, ref=None):
    """gate_problems with the reference gate's selfcheck supplied the way do_gate does it: an IDM tag is read against REF_s{k}'s (the clean one unless `ref` is given); REF / DIV / NULL need none"""
    if CK.TABLE[CK.base_tag(t)]["lever"] == "idm" and ref is None:
        ref = sc_of(CK.ref_tag(CK.TABLE[t]["init"]))[1]
    return CK.gate_problems(t, toks, sc, ref)


def only_idm(s):
    d = s["levers"].get("wm_idm_gamma_readback")
    if not d or d.get("declared", [0])[0] != 1000.0:
        raise KeyError("not an IDM arm")


def only_ref(s):
    d = s["levers"].get("wm_idm_gamma_readback")
    if not d or d.get("declared") != [100.0, 100.0, 100.0]:
        raise KeyError("not a REF arm")


def put(p, data=b""):''')
rep('''    runnable = CK.RUN_ORDER + [CK.PILOT_TAG]''', '''    runnable = CK.RUN_ORDER + [CK.PILOT_TAG, "REF_s0", "REF_s1"]''')
rep('''    check("no blocked tag", all(not v["blocked"] for v in CK.TABLE.values()))''', '''    check("no blocked tag", all(not v["blocked"] for v in CK.TABLE.values()))
    check("AMEND_ACTSENS_1: REF_s0 / REF_s1 are gate-only tags: lever ref, order 0, not in RUN_ORDER", [t for t, v in CK.TABLE.items() if v["lever"] == "ref"] == ["REF_s0", "REF_s1"] and all(CK.TABLE[t]["order"] == 0 and t not in CK.RUN_ORDER for t in ("REF_s0", "REF_s1")))
    check("AMEND_ACTSENS_1: the pilot job gates REF_s0, IDM_s0, DIV_s0 in that order", CK.GATE_AT_PILOT == ["REF_s0", "IDM_s0", "DIV_s0"])
    check("AMEND_ACTSENS_1: the trainer needles carry live_first", '"live_first"' in CK.NEEDLES[CK.TRAINER_REL])
    for k in (0, 1):
        rt, it, nt = CK.tokens(f"REF_s{k}"), CK.tokens(f"NULL_s{k + 2}"), CK.tokens(f"IDM_s{k}")
        check(f"AMEND_ACTSENS_1: REF_s{k} = R35_s{k}'s argv + --wm-idm-gamma 100 100 100, --out = the REF gate out", rt[-4:] == ["--wm-idm-gamma", "100", "100", "100"] and rt[rt.index("--out") + 1] == CK.gate_out_decl(f"REF_s{k}")
              and [x for x in rt[:-4] if x != rt[rt.index("--out") + 1]] == [x for x in nt[:-4] if x != nt[nt.index("--out") + 1]] and rt[rt.index("--seed") + 1] == str(k))
''')
rep('''        lev = CK.TABLE[t]["lever"]
        check(f"{t}: lever''', '''        lev = CK.TABLE[t]["lever"]
        if lev == "ref":
            continue
        check(f"{t}: lever''')
rep('''    orig = os.path.join(PKG, "..", "argvs_actsens.txt")
''', '''    orig = os.path.join(PKG, "..", "argvs_actsens_amend1.txt")
''')
rep('''        check(f"clean selfcheck passes {t}", CK.gate_problems(t, toks, sc) == [])''', '''        check(f"clean selfcheck passes {t}", GP(t, toks, sc) == [])
        check(f"{t}: live_first is present exactly when --wm-idm-gamma is named (IDM and REF)", ("live_first" in sc) == (t.startswith(("IDM", "REF"))))''')
rep('''        "idm readback built at the old weights": lambda s: s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 100.0),''',
    '''        "idm readback built at the old weights": lambda s: (only_idm(s), s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 100.0)),
        "ref readback built at the dose": lambda s: (only_ref(s), s["levers"]["wm_idm_gamma_readback"]["built"].__setitem__(0, 1000.0)),
        "ref readback declared at the dose": lambda s: (only_ref(s), s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [1000.0, 100.0, 100.0])),''')
rep('''        "idm readback declared": lambda s: s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [100.0, 100.0, 100.0]),''',
    '''        "idm readback declared": lambda s: (only_idm(s), s["levers"]["wm_idm_gamma_readback"].__setitem__("declared", [100.0, 100.0, 100.0])),''')
i, j = s.index('        "C3 an unweighted term moved"'), s.index('        "ladder readback absent"')
s = s[:i] + '''        "C3 live_first: an unweighted term moved": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/pred/L2", s["live_first"]["wm/pred/L2"] * 1.0000001)),
        "C3 live_first: level/L2 moved": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/level/L2", s["live_first"]["wm/level/L2"] * 1.0000001)),
        "C3 live_first: level/L1 not REF + 900 x idm": lambda s: (only_idm(s), s["live_first"].__setitem__("wm/level/L1", s["live_first"]["wm/level/L1"] + 1.0)),
        "C3 live_first: a term missing": lambda s: (only_idm(s), s["live_first"].pop("wm/sigreg/L3")),
        "C3 live_first absent on an IDM arm": lambda s: (only_idm(s), s.pop("live_first")),
        "C3 live_first equal to live.terms (the last step)": lambda s: (only_idm(s), s.__setitem__("live_first", dict(s["live"]["terms"]))),
        "REF live.terms: one term changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/sigreg/L3", s["live"]["terms"]["wm/sigreg/L3"] * (1 + 1e-12))),
        "REF live.terms: level/L1 changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/level/L1", s["live"]["terms"]["wm/level/L1"] + 1e-9)),
        "REF live.terms: total changed": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("total", s["live"]["terms"].get("total", 1.0) + 1e-9)) if "total" in s["live"]["terms"] else (_ for _ in ()).throw(KeyError()),
        "REF live.terms: an extra key": lambda s: (only_ref(s), s["live"]["terms"].__setitem__("wm/extra", 1.0)),
        "REF live.terms: a key missing": lambda s: (only_ref(s), s["live"]["terms"].pop("wm/rollout_masked/L3")),
        "REF live_first absent": lambda s: (only_ref(s), s.pop("live_first")),
        "REF live_first holds a non-finite term": lambda s: (only_ref(s), s["live_first"].__setitem__("wm/pred/L1", float("nan"))),
        "live_first on a non-IDM, non-REF arm": lambda s: s.__setitem__("live_first", {"wm/pred/L1": 0.1}) if "live_first" not in s else (_ for _ in ()).throw(KeyError()),
''' + s[j:]
rep('''            check(f"{t}: planted '{name}' is refused", len(CK.gate_problems(t, toks, s2)) >= 1)''', '''            check(f"{t}: planted '{name}' is refused", len(GP(t, toks, s2)) >= 1)''')
# the deliberate regressions of AMEND_ACTSENS_1 A2 (whole-selfcheck stand-in modes) + the reference-gate plumbing
rep('''    check("the planted mutations ran on every run tag (>= 40 each)", n_ref >= 40 * len(runnable))''', '''    check("the planted mutations ran on every run tag (>= 40 each)", n_ref >= 40 * len(runnable))
    # ---- AMEND_ACTSENS_1 A2: the three deliberate regressions, each must make the gate refuse (stand-in modes) and the clean pair must pass
    for k in (0, 1):
        it, rt = f"IDM_s{k}", f"REF_s{k}"
        itoks = CK.tokens(it)
        gi, gr = CK.gate_argv(it, itoks), CK.gate_argv(rt, CK.tokens(rt))
        clean_i, clean_r = PT.good_selfcheck(gi), PT.good_selfcheck(gr)
        check(f"{it}: the clean IDM gate passes against the clean {rt}", CK.gate_problems(it, itoks, clean_i, clean_r) == [])
        r35 = CK.r35_terms(k)
        check(f"{rt}: the stand-in REF writes live.terms equal to R35_s{k}'s on every key, bit for bit", clean_r["live"]["terms"] == r35)
        check(f"{it}: its live.terms are NOT R35's (the last of three AdamW steps has moved under the dose: the 02:38 case), so a C3 that read live.terms would refuse it",
              clean_i["live"]["terms"] != r35 and any(clean_i["live"]["terms"][u] != r35[u] for u in CK.UNWEIGHTED))
        check(f"{it}: regression 1, C3 reading live.terms instead of live_first (live_first := the last step): refused", any("C3" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "drifted_first"), clean_r)))
        check(f"{it}: regression 3, live_first from another batch: refused and the unweighted term is named", any("C3" in x and "unweighted" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "other_batch_first"), clean_r)))
        check(f"{it}: live_first absent: refused", any("live_first" in x for x in CK.gate_problems(it, itoks, PT.good_selfcheck(gi, "no_first"), clean_r)))
        check(f"{it}: the REF's live_first is from another batch than the IDM's: refused", any("C3" in x for x in CK.gate_problems(it, itoks, clean_i, PT.good_selfcheck(gr, "other_batch_first")))
        r_bad = PT.good_selfcheck(gr, "ref_term_changed")
        check(f"{rt}: regression 2, a REF record with one term changed: refused, the term named", any("REF" in x and "differ" in x for x in CK.gate_problems(rt, CK.tokens(rt), r_bad)))
        check(f"{it}: no REF gate output at all: the IDM arm is refused", any("no " + rt in x for x in CK.gate_problems(it, itoks, clean_i, None)))
        check(f"{it}: a REF whose live_first is absent refuses the IDM arm", any("live_first" in x for x in CK.gate_problems(it, itoks, clean_i, PT.good_selfcheck(gr, "no_first"))))
        check(f"C3 identity at the numbers: level/L1 difference = 900 x idm/L1 on the stand-in pair", abs((clean_i["live_first"]["wm/level/L1"] - clean_r["live_first"]["wm/level/L1"]) - 900.0 * clean_r["live_first"]["wm/idm/L1"]) <= 1e-5 * 900.0 * clean_r["live_first"]["wm/idm/L1"])''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
