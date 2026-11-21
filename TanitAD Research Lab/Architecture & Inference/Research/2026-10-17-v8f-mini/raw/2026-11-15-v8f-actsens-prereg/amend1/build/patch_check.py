"""Apply AMEND_ACTSENS_1 to box2_build/actsens_check.py. Every replacement must match exactly once."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("""'"--wm-idm-gamma"', '"--wm-sampler"', "def bad_wm_idm_gamma",""", """'"--wm-idm-gamma"', '"live_first"', '"--wm-sampler"', "def bad_wm_idm_gamma",""")
rep("""    "NULL_s3": {"lever": "null", "seed": 3, "init": 1, "order": 6, "blocked": None},
}""", """    "NULL_s3": {"lever": "null", "seed": 3, "init": 1, "order": 6, "blocked": None},
    # AMEND_ACTSENS_1 (A2): gate-only reference gates. R35_s{k}'s argv + `--wm-idm-gamma 100 100 100` (the default dose, NAMED so that the trainer records live_first). They never train and get no run token; order 0 keeps them out of RUN_ORDER.
    "REF_s0":  {"lever": "ref",  "seed": 0, "init": 0, "order": 0, "blocked": None},
    "REF_s1":  {"lever": "ref",  "seed": 1, "init": 1, "order": 0, "blocked": None},
}""")
rep("""GATE_AT_PILOT = ["IDM_s0", "DIV_s0"]          # the pilot job also gates these full-length argvs, so run 1 starts under a token, steps_total is read before the 72 h decision and both speed modes are read back""",
    """GATE_AT_PILOT = ["REF_s0", "IDM_s0", "DIV_s0"]   # the pilot job also gates these argvs IN THIS ORDER (REF_s0 first: IDM_s0's C3 reads its live_first), so run 1 starts under a token, steps_total is read before the 72 h decision and both speed modes are read back""")
# ---- C3
i, j = s.index("def c3_problems(init, terms):"), s.index("def base_tag(tag):")
s = s[:i] + '''def ref_tag(init):
    return f"REF_s{init}"


def _first_problems(who, lf):
    """live_first (G-LIVE step 0, taken before any optimiser step) must carry every unweighted term the card compares, finite, and wm/level/L1 and wm/idm/L1."""
    if not isinstance(lf, dict) or not lf:
        return [f"{who}: live_first is missing (the trainer records it when --wm-idm-gamma is named)"]
    miss = [k for k in UNWEIGHTED + ("wm/level/L1", "wm/idm/L1") if not finite(lf.get(k))]
    return [f"{who}: live_first lacks finite {miss[:3]!r} ({len(miss)} terms)"] if miss else []


def c3_problems(init, live_first, ref_first):
    """card C3 as amended (AMEND_ACTSENS_1 A1): IDM_s{k}'s live_first (G-LIVE step 0: same init, same first batch) against REF_s{k}'s live_first (R35's argv + the default dose, same tree, read from REF's gate output on box 2):
    the unweighted terms equal bit for bit, level/L2 and level/L3 equal, and level/L1 difference = 900 x idm/L1 (1e-5 relative). The record is live_first, NEVER live.terms (the last of three AdamW steps)."""
    p = _first_problems(f"C3 IDM_s{init}", live_first) + _first_problems(f"C3 REF_s{init}", ref_first)
    if p:
        return p
    bad = [k for k in UNWEIGHTED if live_first[k] != ref_first[k]]
    if bad:
        p.append(f"C3: {len(bad)} unweighted live_first terms differ from REF_s{init}'s (first: {bad[0]} {live_first[bad[0]]!r} vs {ref_first[bad[0]]!r})")
    d, want = live_first["wm/level/L1"] - ref_first["wm/level/L1"], 900.0 * ref_first["wm/idm/L1"]
    if abs(d - want) > 1e-5 * abs(want):
        p.append(f"C3: level/L1 difference {d!r} != 900 x idm/L1 {want!r} (1e-5 relative)")
    return p


def ref_problems(init, terms, live_first):
    """AMEND_ACTSENS_1 A2, the REF gate: live.terms equal R35_s{k}'s banked terms on EVERY key, bit for bit (C1 on real data; also shows the NAMED default dose equals the absent flag); live_first present."""
    r, p = r35_terms(init), []
    if set(terms) != set(r):
        p.append(f"REF: live.terms keys differ from R35_s{init}'s: only here {sorted(set(terms) - set(r))[:3]}, only there {sorted(set(r) - set(terms))[:3]}")
    bad = [k for k in r if k in terms and terms[k] != r[k]]
    if bad:
        p.append(f"REF: {len(bad)} live.terms differ from R35_s{init}'s banked terms (first: {bad[0]} {terms[bad[0]]!r} vs {r[bad[0]]!r})")
    return p + _first_problems(f"REF_s{init}", live_first)


''' + s[j:]
# ---- gate_problems
rep("def gate_problems(tag, toks, sc):", "def gate_problems(tag, toks, sc, ref_sc=None):")
rep("""    if lever == "idm":
        if not isinstance(ir, dict):
            p.append("levers.wm_idm_gamma_readback is missing: the per-level IDM weight was not read back from the built stack")
        else:
            eq("wm_idm_gamma_readback.declared", ir.get("declared"), [1000.0, 100.0, 100.0])
            eq("wm_idm_gamma_readback.built (the weights the loss multiplies by, levels 1 / 2 / 3)", ir.get("built"), [1000.0, 100.0, 100.0])""",
    """    if lever in ("idm", "ref"):
        want_g = [1000.0, 100.0, 100.0] if lever == "idm" else [100.0, 100.0, 100.0]
        if not isinstance(ir, dict):
            p.append("levers.wm_idm_gamma_readback is missing: the per-level IDM weight was not read back from the built stack")
        else:
            eq("wm_idm_gamma_readback.declared", ir.get("declared"), want_g)
            eq("wm_idm_gamma_readback.built (the weights the loss multiplies by, levels 1 / 2 / 3)", ir.get("built"), want_g)""")
rep("""    if lever == "idm":
        p += c3_problems(init, terms)
""", """    if lever == "idm":                                                     # AMEND_ACTSENS_1 A1: C3 reads live_first (step 0) against REF_s{k}'s live_first, never live.terms (step 3)
        ref_first = (ref_sc or {}).get("live_first") if isinstance((ref_sc or {}).get("live_first"), dict) else None
        if ref_sc is None:
            p.append(f"C3: no {ref_tag(init)} gate output to compare with (the reference gate is missing or refused): the IDM arm is refused")
        else:
            p += c3_problems(init, sc.get("live_first"), ref_first)
    elif lever == "ref":
        p += ref_problems(init, terms, sc.get("live_first"))
    elif "live_first" in sc:
        p.append(f"a {lever} arm carries live_first: the IDM flag must be absent")
""")
# ---- do_gate
rep('''    toks = line.split(" ")
    rd = run_dir(tag)
    gd = os.path.join(rd, "gate")
    if not token_ok(tag, line):
        print(f"gate-cached {tag}")
        return []
''', '''    toks = line.split(" ")
    rd = run_dir(tag)
    gd = os.path.join(rd, "gate")
    ref_sc, ref_sha = None, None
    if TABLE[base_tag(tag)]["lever"] == "idm":                              # AMEND_ACTSENS_1: the REF gate of the same chassis must have passed on THIS tree; its selfcheck is what C3 reads
        rt = ref_tag(TABLE[base_tag(tag)]["init"])
        rp = token_ok(rt, " ".join(tokens(rt)))
        rsp = os.path.join(run_dir(rt), "gate", "selfcheck.json")
        if rp or not os.path.exists(rsp):
            return [f"C3: the reference gate {rt} is missing or refused ({'; '.join(rp)[:300] or 'no selfcheck'}): the IDM arm is refused"]
        ref_sc, ref_sha = json.load(open(rsp, encoding="utf-8")), sha_file(rsp)
    if not token_ok(tag, line):
        print(f"gate-cached {tag}")
        return []
''')
rep("    bad = gate_problems(tag, toks, sc)\n", "    bad = gate_problems(tag, toks, sc, ref_sc)\n")
rep('''"selfcheck_sha256": sha_file(os.path.join(gd, "selfcheck.json")), **b}, indent=1))''',
    '''"selfcheck_sha256": sha_file(os.path.join(gd, "selfcheck.json")), **({"c3_ref_selfcheck_sha256": ref_sha} if ref_sha else {}), **b}, indent=1))''')
# ---- cmd_stage
rep('''    if TABLE.get(tag, {}).get("blocked"):
        return stage_fail(tag, gid, "pin-fail", f"{tag} is blocked: {TABLE[tag]['blocked']}")
    rd = run_dir(tag)''', '''    if TABLE.get(tag, {}).get("blocked"):
        return stage_fail(tag, gid, "pin-fail", f"{tag} is blocked: {TABLE[tag]['blocked']}")
    if TABLE.get(tag, {}).get("lever") == "ref":
        return stage_fail(tag, gid, "pin-fail", f"{tag} is a gate-only reference: it never trains and gets no run token")
    rd = run_dir(tag)''')
rep('''    jobs = [(tag, line)]
    if is_pilot(tag):''', '''    jobs = [(tag, line)]
    if not is_pilot(tag) and TABLE[tag]["lever"] == "idm":     # AMEND_ACTSENS_1 A2: the reference gate of the chassis runs on this tree BEFORE the IDM gate (cached if the pilot job already passed it)
        rt = ref_tag(TABLE[tag]["init"])
        os.makedirs(run_dir(rt), exist_ok=True)
        rfull = " ".join(tokens(rt))
        write_text(os.path.join(run_dir(rt), "argv.txt"), rfull + "\\n")
        jobs = [(rt, rfull)] + jobs
    if is_pilot(tag):''')
# ---- finish_pilot: keep the REF evidence with the pilot
rep('''    shutil.copy2(token_path(tag), os.path.join(pd, "gate_token.json"))
''', '''    shutil.copy2(token_path(tag), os.path.join(pd, "gate_token.json"))
    shutil.copy2(os.path.join(run_dir(ref_tag(0)), "gate", "selfcheck.json"), os.path.join(pd, "ref_selfcheck.json"))     # AMEND_ACTSENS_1: REF_s0's gate output (C1 on real data, and what IDM_s0's C3 was read against)
''')
# ---- cleanup
rep('''    for t in ([tag] + GATE_AT_PILOT) if is_pilot(tag) else [tag]:
        d = rm_own_temp(loc(gate_out_decl(t)), "gate_out")''', '''    for t in ([tag] + GATE_AT_PILOT) if is_pilot(tag) else ([tag, ref_tag(TABLE[tag]["init"])] if TABLE.get(tag, {}).get("lever") == "idm" else [tag]):
        d = rm_own_temp(loc(gate_out_decl(t)), "gate_out")''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
