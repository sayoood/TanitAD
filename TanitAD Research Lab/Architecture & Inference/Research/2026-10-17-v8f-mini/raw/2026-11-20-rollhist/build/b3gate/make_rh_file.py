"""Applies the ROLLHIST additions to dscale10k_check_rh.py (already holds the header and ARMS edits). Idempotent guard: refuses to run twice. Every replacement asserts it matches exactly once."""
import sys

p = "dscale10k_check_rh.py"
s = open(p, encoding="utf-8").read()
if "def rh_argv_problems" in s:
    sys.exit("already patched")
blk = open("_block_rh.py.txt", encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


HEAD = "# ------------------------------------------------------------------------------------------------ the token\n"
rep(HEAD, blk + HEAD)

rep('''REQUIRED_STAGE_B = ("C-BANK", "G-DAGGER")''', '''REQUIRED_RH_RH = ("C-BANK", "G-DAGGER", "C-RH-PAIR", "C-RH-BANKGEN")           # ROLLHIST RH: the stage-B checks + the pair + the bank's generator record
REQUIRED_RH_B0 = ("C-RH-PAIR", "G-DAGGER-B0")                                   # ROLLHIST B0: no bank, no DAgger probe; the pair + the trainer's own selfcheck of the p = 0 argv
REQUIRED_STAGE_B = ("C-BANK", "G-DAGGER")''')

rep('''def required_checks(arm: str, stage: str = "A") -> tuple:
    return''', '''def required_checks(arm: str, stage: str = "A") -> tuple:
    if arm in RH_ARMS:
        return REQUIRED_DATA + REQUIRED_UNION + REQUIRED_TRAINER + (REQUIRED_RH_RH if RH_ARMS[arm][0] == "RH" else REQUIRED_RH_B0)
    return''')

rep('''    ue = ev.get("ue_steps")
    put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue))
''', '''    ue = ev.get("ue_steps")
    kind = RH_ARMS[arm][0] if arm in RH_ARMS else None
    if kind:
        tmpl = json.load(open(a.template, encoding="utf-8")) if getattr(a, "template", "") else None
        tmpl = tmpl["argv"] if isinstance(tmpl, dict) else tmpl
        pair = json.load(open(a.pair_argv, encoding="utf-8")) if getattr(a, "pair_argv", "") else None
        pair = pair["argv"] if isinstance(pair, dict) else pair
        rtok = getattr(a, "readout_token", "") or None
        pre = ([f"{arm} is a stage-B arm: stage {stage} refused"] if stage != "B" else []) + ([] if tmpl else ["no --template (the arm's U_sK.B.tmpl.json)"])
        if kind == "RH" and not rtok:
            pre.append("an RH arm needs --readout-token (causal_g1x | causal_g1): the declaration is a CLI argument of the mint")
        put("C-ARGV", pre + (rh_argv_problems(base, tmpl, toks, arm, rtok) if tmpl else []))
        if pair is None:
            put("C-RH-PAIR", ["no --pair-argv (the other arm's filled argv: RH <-> B0 of the same seed)"])
        else:
            p_tok = rtok if kind == "RH" else ((parse_argv(pair).get("--dagger-bank-readout") or [""])[0])
            rh_t, b0_t = (toks, pair) if kind == "RH" else (pair, toks)
            bsha = (parse_argv(rh_t).get("--dagger-bank-sha256") or [""])[0] or None
            pc = getattr(a, "pair_check", "")
            put("C-RH-PAIR", rh_pair_problems(rh_t, b0_t, p_tok, bsha, pc), readout_token=p_tok, pair_argv_sha256=argv_sha(pair), pair_check_sha256=sha_file(pc) if pc and os.path.isfile(pc) else None)
    else:
        put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue))
''')

rep('''    if stage == "B":
        bank_dir = (parse_argv(toks).get("--dagger-bank") or [""])[0]
        init = (parse_argv(toks).get("--init-ckpt") or [""])[0]
        put("C-BANK", bank_problems(bank_dir, init, dry), n_shards=BANK_SHARDS)
        sc = json.load(open(a.selfcheck, encoding="utf-8")) if getattr(a, "selfcheck", "") and os.path.isfile(a.selfcheck) else None
        put("G-DAGGER", dagger_problems(sc, toks) if sc else ["no --selfcheck: the trainer's --selfcheck-only record of THIS stage-B argv (real bank, real init checkpoint)"],
            selfcheck_sha256=sha_file(a.selfcheck) if sc else None)
''', '''    if stage == "B":
        bank_dir = (parse_argv(toks).get("--dagger-bank") or [""])[0]
        init = (parse_argv(toks).get("--init-ckpt") or [""])[0]
        if kind != "B0":
            put("C-BANK", bank_problems(bank_dir, init, dry), n_shards=BANK_SHARDS)
        sc = json.load(open(a.selfcheck, encoding="utf-8")) if getattr(a, "selfcheck", "") and os.path.isfile(a.selfcheck) else None
        no_sc = ["no --selfcheck: the trainer's --selfcheck-only record of THIS stage-B argv (real bank, real init checkpoint)"]
        if kind == "B0":
            put("G-DAGGER-B0", b0_selfcheck_problems(sc, toks) if sc else no_sc, selfcheck_sha256=sha_file(a.selfcheck) if sc else None)
        else:
            put("G-DAGGER", (dagger_problems(sc, toks) + (rh_levers_problems(sc, "RH", rtok) if kind == "RH" else [])) if sc else no_sc,
                selfcheck_sha256=sha_file(a.selfcheck) if sc else None)
        if kind == "RH":
            put("C-RH-BANKGEN", rh_bank_generator_problems(bank_dir, (parse_argv(toks).get("--dagger-bank-sha256") or [""])[0], rtok or "", RH_KAPPA), readout_token=rtok, kappa=RH_KAPPA)
''')

rep('''"selfcheck": getattr(a, "selfcheck", ""), "tacsrc_list": getattr(a, "tacsrc_list", ""), "ue_argv": getattr(a, "ue_argv", ""), "b_dry": getattr(a, "b_dry", "")}
    b, bad = binding(''', '''"selfcheck": getattr(a, "selfcheck", ""), "tacsrc_list": getattr(a, "tacsrc_list", ""), "ue_argv": getattr(a, "ue_argv", ""), "b_dry": getattr(a, "b_dry", ""),
             "template": getattr(a, "template", ""), "pair_argv": getattr(a, "pair_argv", ""), "pair_check": getattr(a, "pair_check", "")}
    b, bad = binding(''')

rep('''"selfcheck": g("selfcheck"), "tacsrc_list": g("tacsrc_list"), "ue_argv": g("ue_argv"), "b_dry": g("b_dry")}                                               # gate patch 2026-11-17: the same extra files mint bound
''', '''"selfcheck": g("selfcheck"), "tacsrc_list": g("tacsrc_list"), "ue_argv": g("ue_argv"), "b_dry": g("b_dry"),
             "template": g("template"), "pair_argv": g("pair_argv"), "pair_check": g("pair_check")}                                               # gate patch 2026-11-17: the same extra files mint bound
''')

rep('''    print(f"TOKEN-OK {a.arm}")
''', '''    if a.arm in RH_ARMS:                                                  # ROLLHIST: stage B only, and the declared token is the one the token records
        tk = json.load(open(os.path.join(a.gate_dir, "gate_token.json"), encoding="utf-8"))
        rec = ((tk.get("checks") or {}).get("C-RH-PAIR") or {}).get("readout_token")
        if a.stage != "B":
            print(f"REFUSED {a.arm}: a stage-B arm")
            return 2
        if RH_ARMS[a.arm][0] == "RH" and g("readout_token") != rec:
            print(f"REFUSED {a.arm}: --readout-token {g('readout_token')!r} is not the token's {rec!r}")
            return 2
    print(f"TOKEN-OK {a.arm}")
''')

# CLI
rep('''p.add_argument("--tacsrc-list", default=""); p.add_argument("--ue-argv", default=""); p.add_argument("--b-dry", default="")
        if nm == "mint":''', '''p.add_argument("--tacsrc-list", default=""); p.add_argument("--ue-argv", default=""); p.add_argument("--b-dry", default="")
            p.add_argument("--template", default=""); p.add_argument("--pair-argv", default=""); p.add_argument("--pair-check", default=""); p.add_argument("--readout-token", default="")
        if nm == "mint":''')
rep('''p.add_argument("--b-dry", default=""); p.add_argument("--holdout-manifest", default="");''', '''p.add_argument("--b-dry", default=""); p.add_argument("--template", default=""); p.add_argument("--pair-argv", default="")
            p.add_argument("--pair-check", default=""); p.add_argument("--readout-token", default=""); p.add_argument("--holdout-manifest", default="");''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
