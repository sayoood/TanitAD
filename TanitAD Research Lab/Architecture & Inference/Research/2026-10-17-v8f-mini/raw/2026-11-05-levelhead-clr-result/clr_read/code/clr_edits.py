"""clr_edits.py -- the exact edits make_clr_from_cl.py applies to the (role-renamed) CL reader to produce code/clr_read.py.  Entries: ("edit", name, old, new) = exact-once replacement;
("span", name, start, end, new) = replace from the exactly-once ``start`` up to (not including) the exactly-once ``end``."""

DOC_NEW = r'''clr_read.py -- the T1-kin read of arm CLR (PREREG_LEVELHEAD_A2.md: the level head inside RAPR-P01's recovery recipe).  Cloned from cl_read.py (make_clr_from_cl.py lists every difference).

The rules P, G, F, D and LEAK-BORNE are the CL card's section 4 with CL := CLR (A2 card, Reads), implemented exactly as cl_read.py implements them; the A2 card adds the ATTRIBUTION (CLR - RAPRP01 on M1 and all 19 rows) and the outcome mapping.
Reuses the RECOVER-NEXT reader's functions (heavy_init, extract, read_arrays, rows_of, Aof, Bof, m1_cell, guards_rule, floor_rule, read_a_block, GD.classify), loaded BY FILE PATH (pins checked by heavy_init).

Roles per seed s (tag = role + "_s{s}"):
  NEW (this package's rollouts, F:/TanitAD-scratch/motion_c/clr_read/out)  CLR logged | CLRC logged + kbox | FZCLR hold_now | RICLR reindexed | RICLRC reindexed + kbox | RIRAPRP01 = RAPR-P01's checkpoint reindexed
  REUSED (read only, sha256-pinned)  C0, C0C, FZC0, C0REP, RAPRP01 (recover_next/read/raw/arrays; RAPRP01 = the banked LOGGED rollout)   RIC0, RIC0C (motion_c/ri_read/raw/arrays)
Rules (ALL evaluated on `reindexed` unless named):
  P  per seed: M1 of RICLR - RIC0 separated-better AND share = -(RICLR - RIC0)[M1] / (RIC0 - P10)[M1] >= 0.5; BOTH seeds.
  G  guards_rule on RICLR - RIC0: ADE6 point <= +0.0604 m and no non-crawl row separated-worse, both seeds.
  F  |RICLR - RIC0| on M1 (point) > |C0REP - C0| on M1 (LOGGED replicate pair), both seeds.
  D  composite RICLRC - P10: D1, D2, D3 over FOUR families (STRATEGIC included; the three-family form beside) at both seeds, AND P AND G.  Only this counts.
  LEAK-BORNE  (logged share - reindexed share) > half the logged share at either seed (logged share > 0 required).
  ATTRIBUTION  (A2 card) M1 of RICLR - RIRAPRP01 (reindexed, deciding) separated-better at >= 1 seed (the CL card's reading (d)); CLR - RAPRP01 on `logged` beside.
  hold_now (FZCLR vs FZC0): beside, never deciding.
Outcome (A2 card, Outcomes), one branch, with the next arm: D-HOLDS | P-HOLDS-G-OR-D-FAILS | P-FAILS-ATTR-NOT-SEPARATED-BETTER (the level adds nothing even when needed: LLR next) |
  P-FAILS-ATTR-SEPARATED-BETTER (an UNCOMMITTED cell: the card names no branch; reported, the PI / Master Mind decides); LEAK-BORNE overrides the label.
Tier T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2).  Estimator: paired episode-cluster bootstrap B = 2000, seed 0 (gapdecomp.decompose) over the 141 eval episodes /
2,399 windows: answers another draw of EPISODES only; the training-variance floor is rule F, the inference seed does not apply (deterministic planner).

usage: python -I clr_read.py --mode production [--out-root F:/TanitAD-scratch/motion_c/clr_read/out] [--workers 6]     (controls first; a failed control REFUSES the read, exit 3)
       python -I clr_read.py --mode ka-null | ka-ri | ka-cl                                                          (known answers on EXISTING arrays, no rollout; raw/KA_READER_<mode>.json)'''

EDITS = []


def edit(name, old, new):
    EDITS.append(("edit", name, old, new))


def span(name, start, end, new):
    EDITS.append(("span", name, start, end, new))


edit("paths", r'''CARD = f"{DL}/motion_c/cl_prereg/PREREG_LEVELHEAD_CL.md"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/cl_read/out"
WORK_DIR = "F:/TanitAD-scratch/motion_c/cl_read/work"''', r'''CL_CARD = f"{DL}/motion_c/cl_prereg/PREREG_LEVELHEAD_CL.md"           # sections 4-5 hold the rules P, G, F, D, LEAK-BORNE, which the A2 card takes by reference (CL := CLR)
CARD = f"{DL}/motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md"             # the A2 card: reads, attribution, outcomes
A2_FROZEN = f"{DL}/motion_c/a2_prereg/FROZEN_SHA.txt"
CL_RAW = f"{DL}/motion_c/cl_read/raw"
CL_READ_JSON = f"{CL_RAW}/CL_READ.json"
CL_ARR = f"{CL_RAW}/arrays"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/clr_read/out"
WORK_DIR = "F:/TanitAD-scratch/motion_c/clr_read/work"''')
edit("card_rules_path", "def card_rules_sha(path=CARD):", "def card_rules_sha(path=CL_CARD):")
edit("a2_card_sha", 'CARD_RULES_SHA = "ff6d672a5f53dfdab62e18922fe5653ce67a794a75fd902b094c535fb9c25bf8"',
     'A2_CARD_SHA = "9a117b201d74db13aa7ccb43b0407de7634d319141203550047929bda5b66a31"      # the A2 card as frozen 2026-11-05T15:52 (a2_prereg/FROZEN_SHA.txt): any edit is a refused control\n'
     'CARD_RULES_SHA = "ff6d672a5f53dfdab62e18922fe5653ce67a794a75fd902b094c535fb9c25bf8"')
edit("new_roles", '''NEW_ROLES = {"CLR": ("logged", "off"), "CLRC": ("logged", "kbox"), "FZCLR": ("hold_now", "off"), "RICLR": ("reindexed", "off"), "RICLRC": ("reindexed", "kbox")}''',
     '''NEW_ROLES = {"CLR": ("logged", "off"), "CLRC": ("logged", "kbox"), "FZCLR": ("hold_now", "off"), "RICLR": ("reindexed", "off"), "RICLRC": ("reindexed", "kbox"), "RIRAPRP01": ("reindexed", "off")}
RUN_OF = {role: "CLR" for role in NEW_ROLES}
RUN_OF["RIRAPRP01"] = "RAPRP01"          # the run dir M2b_s{s}_recft_<RUN_OF[role]> each role is rolled from (RIRAPRP01 = the attribution base: RAPR-P01's own checkpoint under `reindexed`)''')
edit("reused", 'REUSED = {"C0": REF_ARR, "C0C": REF_ARR, "FZC0": REF_ARR, "C0REP": REF_ARR, "RIC0": RI_ARR, "RIC0C": RI_ARR}',
     'REUSED = {"C0": REF_ARR, "C0C": REF_ARR, "FZC0": REF_ARR, "C0REP": REF_ARR, "RAPRP01": REF_ARR, "RIC0": RI_ARR, "RIC0C": RI_ARR}')
edit("reused_sha", '"RIC0C_s1": "7ef6cc2a30600d11c822a2b609b4aea40eb5f20c1f1b72759e6290dc5973050f"}',
     '''"RIC0C_s1": "7ef6cc2a30600d11c822a2b609b4aea40eb5f20c1f1b72759e6290dc5973050f",
              "RAPRP01_s0": "79c449422a9d59d799d6869b19e42ba9566ee879efb55239f7d9b13936e577ca", "RAPRP01_s1": "057289b16f691fd54c505d3075901c8cff8ad1fe78a80ea9ef7539fc75046598"}
CL_ARR_SHA = {"CL_s0": "cfd58740a967c0ae5d666da921c2834c8eda8bb0e5fee945a627a6b4ab8c1070", "CL_s1": "9a7323a52a6e21929d078b64f772eb04ac9dd3b9036d738cab2fb7b6d5e939c8",       # CL's own arrays: the ka-cl known answer only
              "CLC_s0": "e2c2a3c532bfe7c0587359c2a25fffefbe6d7673e1c47d5f74843e99bb463571", "CLC_s1": "fec315481b9a6b81c0e4ad4a64d72ae15aa268194669e254013f25d0f54ec64f",
              "FZCL_s0": "82675a88cabfd2bf681d738e78d6ba6e6e4ccc5cbb8f69acfba1e084d087cab5", "FZCL_s1": "f2e94871b3c4a24841130baa864bbd31ba594bc56a027dbb52d24fecda2af591",
              "RICL_s0": "9335cc19736711294d7d9da1f5c9e29a8cd0d11fb14fda2596c4223d6e9f4ff0", "RICL_s1": "16f84ec94cfe3530c0b70a94ea11cb1622b9889d4ce5c46a4336552fe5c621cc",
              "RICLC_s0": "7e416cf5c0b16518af2dea09c5c236c1ba634862c73a13b927cc0c1c5f5c5278", "RICLC_s1": "7aca44562915fed8de5fd6b44c4ae944d0c46d310da41cbe7fbcf9d0ea5385f7"}''')

span("next_arm", "NEXT_ARM = {", "rr = SR = GD = None", r'''NEXT_ARM = {
    "D-HOLDS": "counted DOMINANT: signal the PI for scale (data and size), with the video evidence the PI asked for",
    "P-HOLDS-G-OR-D-FAILS": "next arm CLR-W2 = the arm's dose (W / 2) under the same card (A2 card, Outcomes); then the composite's failing row is the target",
    "P-FAILS-ATTR-NOT-SEPARATED-BETTER": "the level adds nothing even when needed (CLR - RAPRP01 on M1 not separated-better): LLR (the lead-aware head) is the next arm, wiring per the O-sensitivity of CLR (readout/ and WIRING.md)",
    "P-FAILS-ATTR-SEPARATED-BETTER": "UNCOMMITTED CELL: the A2 card commits no branch for 'P fails, CLR - RAPRP01 on M1 separated-better' (the level helps relative to RAPR-P01 without clearing P): the Master Mind / the PI names "
                                     "the next arm before any launch; LLR (the committed branch of the not-separated cell) is the nearest candidate; both numbers are reported",
}
''')

span("decide", "def decide(P, G, D, n_m1_separated_better, leak):", "def read_all(arrays, n_boot=2000, log=print):", r'''def rule_attr(cells_ri, cells_lg):
    """ATTRIBUTION (A2 card, Reads and Outcomes): CLR - RAPRP01 on M1.  Deciding scene = reindexed (RICLR - RIRAPRP01); logged (CLR - the banked RAPRP01 arrays) beside.  'Separated-better' at >= 1 seed counts as
    moved (the CL card's reading (d)); the per-seed cells are reported.  ``cells_*`` = {seed key: m1_cell} with the base excess RAPRP01 - P10."""
    def blk(cells):
        per = {k: {"separated_better": bool(c["better"]), "share_of_RAPRP01_excess_closed": c["share_closed"], "xy_delta": c["xy_delta"]} for k, c in cells.items()}
        return per, int(sum(p["separated_better"] for p in per.values()))
    pr, nr = blk(cells_ri)
    pl, nl = blk(cells_lg)
    return {"per_seed_reindexed": pr, "n_seeds_separated_better_reindexed": nr, "per_seed_logged_beside": pl, "n_seeds_separated_better_logged_beside": nl,
            "SEPARATED_BETTER_AT_ANY_SEED": bool(nr >= 1), "deciding_scene": "reindexed"}


def decide(P, G, D, n_attr_separated_better, leak):
    """The A2 card's Outcomes as one function.  Returns {label, branch, underlying, next, counts_as_result, notes}.
    branch: D-HOLDS (D counted: D1-D3 at both seeds with P and G) | P-HOLDS-G-OR-D-FAILS | P-FAILS-ATTR-NOT-SEPARATED-BETTER (CLR - RAPRP01 on M1 separated-better at NO seed) | P-FAILS-ATTR-SEPARATED-BETTER (>= 1 seed; uncommitted).
    LEAK-BORNE overrides the LABEL whatever else the row reads and is reported as the camera-clock effect, never as a result; its next arm is the P-fails branch's."""
    notes = []
    if D:
        branch = "D-HOLDS"
    elif P:
        branch = "P-HOLDS-G-OR-D-FAILS"
        notes.append("G fails" if not G else "G holds, D fails")
    elif n_attr_separated_better >= 1:
        branch = "P-FAILS-ATTR-SEPARATED-BETTER"
    else:
        branch = "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
    if not P and not G:
        notes.append("G fails too (the guards read worse alongside the failed P)")
    if branch == "P-FAILS-ATTR-SEPARATED-BETTER" and n_attr_separated_better == 1:
        notes.append("CLR - RAPRP01 on M1 separated-better at ONE seed only: read as 'separated-better' (the committed not-separated cell is no separation at either seed)")
    if leak:
        pf = "P-FAILS-ATTR-SEPARATED-BETTER" if n_attr_separated_better >= 1 else "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
        return {"label": "LEAK-BORNE", "branch": "LEAK-BORNE", "underlying": branch, "next": NEXT_ARM[pf] + " (LEAK-BORNE: the effect is the camera-clock effect, never a result; next as for 'P fails')",
                "counts_as_result": False, "notes": notes}
    return {"label": branch, "branch": branch, "underlying": branch, "next": NEXT_ARM[branch], "counts_as_result": True, "notes": notes}


# ------------------------------------------------------------------------------------------------ the read (per-window arrays in, one result dict out)
''')

edit("init_attr", "    cells_ri, cells_lg, cells_fz, xy_ri, rows_ri, ra_riclc, eff_ri, eff_lg, rep = {}, {}, {}, {}, {}, {}, {}, {}, {}",
     "    cells_ri, cells_lg, cells_fz, xy_ri, rows_ri, ra_riclc, eff_ri, eff_lg, rep = {}, {}, {}, {}, {}, {}, {}, {}, {}\n    attr_ri, attr_lg = {}, {}")
edit("attr_rows", '''        b["FZCLR_minus_FZC0"], b["FZC0_minus_P10"], b["FZCLR_minus_P10"] = rows("FZCLR", "FZC0")[0], vs_p10("FZC0"), vs_p10("FZCLR")''',
     '''        b["FZCLR_minus_FZC0"], b["FZC0_minus_P10"], b["FZCLR_minus_P10"] = rows("FZCLR", "FZC0")[0], vs_p10("FZC0"), vs_p10("FZCLR")
        # ATTRIBUTION (A2 card): CLR against RAPR-P01's own checkpoint, on the reindexed clock (RIRAPRP01 = new rollout) and on the logged clock (the banked RAPRP01 arrays)
        b["RICLR_minus_RIRAPRP01"], b["M1_per_class_split_RICLR_minus_RIRAPRP01"] = rows("RICLR", "RIRAPRP01", M1)
        b["CLR_minus_RAPRP01"], b["M1_per_class_split_CLR_minus_RAPRP01"] = rows("CLR", "RAPRP01", M1)
        b["RIRAPRP01_minus_P10"], b["RAPRP01_minus_P10"] = vs_p10("RIRAPRP01"), vs_p10("RAPRP01")
        b["RIRAPRP01_minus_RIC0"], b["RAPRP01_minus_C0"] = rows("RIRAPRP01", "RIC0")[0], rows("RAPRP01", "C0")[0]
        attr_ri[k] = rr.m1_cell(b["RICLR_minus_RIRAPRP01"][M1], b["RIRAPRP01_minus_P10"][M1])
        attr_lg[k] = rr.m1_cell(b["CLR_minus_RAPRP01"][M1], b["RAPRP01_minus_P10"][M1])
        b["m1_cell_attribution_reindexed"], b["m1_cell_attribution_logged"] = attr_ri[k], attr_lg[k]''')
edit("family_tables_attr", 'for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0")}',
     'for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01")}')
edit("decide_call", '''    out = decide(P["PASS"], G["PASS"], D["DOMINANT_COUNTED"], P["n_seeds_separated_better"], leak["LEAK_BORNE"])''',
     '''    A = rule_attr(attr_ri, attr_lg)
    out = decide(P["PASS"], G["PASS"], D["DOMINANT_COUNTED"], A["n_seeds_separated_better_reindexed"], leak["LEAK_BORNE"])''')
edit("rules_dict", '''    res["rules"] = {"P": P, "G": G, "F": F, "D": D, "LEAK_BORNE": leak, "deciding_scene": "reindexed",''',
     '''    res["rules"] = {"P": P, "G": G, "F": F, "D": D, "LEAK_BORNE": leak, "ATTRIBUTION": A, "deciding_scene": "reindexed",''')

edit("ctl_expected_run", '''"ckpt_is_the_CLR_run": f"M2b_s{s}_recft_CLR" in d["ckpt"].replace("\\\\", "/"),''',
     '''"ckpt_is_the_expected_run": f"/M2b_s{s}_recft_{RUN_OF[role]}/ckpt_last.pt" in d["ckpt"].replace("\\\\", "/"),''')
edit("ctl_ckpts_key", 'ckpts.setdefault(s, set()).add(d["ckpt_sha256"])', 'ckpts.setdefault((s, RUN_OF[role]), set()).add(d["ckpt_sha256"])')
edit("ctl_one_ckpt", '''ctl["one_checkpoint_per_seed"] = {"PASS": bool(all(len(v) == 1 for v in ckpts.values()) and len(ckpts) == len(SEEDS)), "ckpt_sha256": {f"s{s}": sorted(v) for s, v in ckpts.items()}}''',
     '''ctl["one_checkpoint_per_seed"] = {"PASS": bool(all(len(v) == 1 for v in ckpts.values()) and len(ckpts) == 2 * len(SEEDS)), "ckpt_sha256": {f"s{s}_{r}": sorted(v) for (s, r), v in ckpts.items()}}''')
edit("ctl_ka_new", '''ka = json.load(open(f"{raw}/KA_NEW.json", encoding="utf-8"))''', '''ka = json.load(open(f"{CL_RAW}/KA_NEW.json", encoding="utf-8"))''')
edit("ctl_precheck_same", '''same = all(pc["seeds"][f"s{s}"]["ckpt_sha256"] in ckpts.get(s, set()) for s in SEEDS)''',
     '''same = all(pc["seeds"][f"s{s}"]["ckpt_sha256"] in ckpts.get((s, "CLR"), set()) and pc["seeds"][f"s{s}"].get("raprp01_ckpt_sha256") in ckpts.get((s, "RAPRP01"), set()) for s in SEEDS)''')
edit("ctl_cards", '''"pinned": CARD_RULES_SHA, "card_sha256": sha256_file(CARD)}''',
     '''"pinned": CARD_RULES_SHA, "card_sha256": sha256_file(CL_CARD)}
    a2, fz = sha256_file(CARD), open(A2_FROZEN, encoding="utf-8").readline().split()[0]
    ctl["card_a2"] = {"PASS": bool(a2 == A2_CARD_SHA == fz), "sha256": a2, "pinned": A2_CARD_SHA, "frozen_sha_file": fz}''')

span("ka_arrays", "def ka_arrays(mode, s):", "def ka_check(mode, res):", r'''def ka_arrays(mode, s):
    z, ctl = reused_arrays(s)
    base = {"CLR": "C0", "CLRC": "C0C", "FZCLR": "FZC0", "RICLR": "RIC0", "RICLRC": "RIC0C"}
    if mode == "ka-null":       # the arm IS its base (C0's family) and the attribution base is the null too: RIRAPRP01 := RIC0, RAPRP01 := C0
        z.update({r: z[b] for r, b in base.items()})
    elif mode == "ka-ri":       # the RI rollout fed in the LOGGED slot (CLR := RIC0, CLRC := RIC0C), the reindexed slots null: CLR - C0 must be RI_C0_READ.json's RIC0 - C0, row for row
        z.update({**{r: z[b] for r, b in base.items()}, "CLR": z["RIC0"], "CLRC": z["RIC0C"]})
    else:                       # ka-cl: CL's own five arrays fed as CLR's; every CLR contrast must equal CL_READ.json's CL contrast (checked by ka_cl_checks)
        for r, cl in (("CLR", "CL"), ("CLRC", "CLC"), ("FZCLR", "FZCL"), ("RICLR", "RICL"), ("RICLRC", "RICLC")):
            p = f"{CL_ARR}/{cl}_s{s}.npz"
            got = sha256_file(p)
            ctl[f"{cl}_s{s}_fed_as_{r}"] = {"path": p, "sha256": got, "pinned": CL_ARR_SHA[f"{cl}_s{s}"], "PASS": got == CL_ARR_SHA[f"{cl}_s{s}"]}
            z[r] = rr.read_arrays(p)
    z["RIRAPRP01"], z["RAPRP01"] = z["RIC0"], z["C0"]          # null attribution base in every known-answer mode (the contrasts then equal the contrasts against C0)
    return z, ctl


def _cl_name(c):
    """The CL reader's name of a CLR contrast key (the reverse of the role renaming)."""
    for a, b in (("RICLRC", "RICLC"), ("RICLR", "RICL"), ("FZCLR", "FZCL"), ("CLRC", "CLC"), ("CLR", "CL")):
        c = re.sub(r"(?<![A-Za-z0-9])" + a + r"(?![A-Za-z0-9])", b, c)
    return c


def ka_cl_checks(res):
    """ka-cl: CL's arrays fed as CLR's (null attribution base) must give CL_READ.json back: every 19-row contrast row for row (delta, lo, hi, separated), every rule value, and the label's branch (M1 moved <-> attribution)."""
    cl, chk, n = json.load(open(CL_READ_JSON, encoding="utf-8")), {}, 0
    for s in SEEDS:
        k = f"s{s}"
        for c, v in res["seeds"][k].items():
            if not isinstance(v, dict) or "RAPRP01" in c or c.startswith("family_tables") or c.startswith("m1_cell") or c.startswith("read_") or not all(lab in v for lab in GD.LABELS[:3]):
                continue
            old = cl["seeds"][k].get(_cl_name(c))
            if old is None:
                continue
            ok = all(float(v[lab][q]) == float(old[lab][q]) for lab in old for q in ("delta", "lo", "hi")) and all(bool(v[lab]["separated"]) == bool(old[lab]["separated"]) for lab in old)
            chk[f"{k}_{c}_equals_CL_READ[{_cl_name(c)}]"] = bool(ok)
            n += len(old)
        chk[f"{k}_m1_cells_equal"] = all(res["seeds"][k][a] == cl["seeds"][k][b] for a, b in (("m1_cell_reindexed", "m1_cell_reindexed"), ("m1_cell_logged", "m1_cell_logged"), ("m1_cell_hold_now", "m1_cell_hold_now")))
        chk[f"{k}_attribution_cell_equals_CL_m1_cell"] = bool(res["seeds"][k]["m1_cell_attribution_reindexed"]["better"] == cl["seeds"][k]["m1_cell_reindexed"]["better"])
    chk["rows_compared_at_least_500"] = n >= 500
    R, Rc = res["rules"], cl["rules"]
    chk["rule_P_equals_CL"] = all(R["P"]["per_seed"][k] == Rc["P"]["per_seed"][k] for k in ("s0", "s1")) and R["P"]["PASS"] == Rc["P"]["PASS"]
    chk["rule_G_F_equal_CL"] = R["G"]["PASS"] == Rc["G"]["PASS"] and R["F"]["PASS"] == Rc["F"]["PASS"] and R["F"]["per_seed"] == Rc["F"]["per_seed"]
    chk["rule_D_leak_equal_CL"] = R["D"]["DOMINANT_COUNTED"] == Rc["D"]["DOMINANT_COUNTED"] and R["LEAK_BORNE"]["LEAK_BORNE"] == Rc["LEAK_BORNE"]["LEAK_BORNE"] and R["LEAK_BORNE"]["per_seed"] == Rc["LEAK_BORNE"]["per_seed"]
    mapped = {"P-FAILS-M1-NOT-MOVED": "P-FAILS-ATTR-NOT-SEPARATED-BETTER", "P-FAILS-M1-SEPARATED-BETTER": "P-FAILS-ATTR-SEPARATED-BETTER"}
    chk["label_is_CLs_branch_mapped"] = res["outcome"]["label"] == mapped.get(cl["outcome"]["label"], cl["outcome"]["label"])
    return chk


''')
edit("ka_check_cl", '''    chk = {}
    ri = json.load(open(RI_READ_JSON, encoding="utf-8"))''', '''    chk = {}
    if mode == "ka-cl":
        return ka_cl_checks(res)
    ri = json.load(open(RI_READ_JSON, encoding="utf-8"))''')
edit("ka_ri_branch", '''        else:
            for lab, r in ri["seeds"][k]["RIC0_minus_C0"].items():''', '''        elif mode == "ka-ri":
            for lab, r in ri["seeds"][k]["RIC0_minus_C0"].items():''')
edit("ka_null_label", '''        chk["label_is_M1_not_moved"] = res["outcome"]["label"] == "P-FAILS-M1-NOT-MOVED"''',
     '''        chk["label_is_attribution_not_separated_better"] = res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
        for k in ("s0", "s1"):          # the arm is its base and the attribution base is the null: every attribution row is exactly zero (no separation)
            chk[f"{k}_attribution_rows_exactly_zero"] = all(r["delta"] == 0.0 and not r["separated"] for c in ("RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01") for r in res["seeds"][k][c].values())''')

edit("md_title", '''f"# CLR read: arm CLR (level-from-image head), T1-kin, generated {res.get('clock', '')}"''',
     '''f"# CLR read: arm CLR (level-from-image head inside RAPR-P01's recovery recipe), T1-kin, generated {res.get('clock', '')}"''')
edit("md_next", '''f"Next (card section 5): {o['next']}."''', '''f"Next (A2 card, Outcomes): {o['next']}."''')
edit("md_rules_head", '"## Rules (card section 4; deciding scene = reindexed)"', '"## Rules (CL card section 4 by reference, CL := CLR; attribution and outcomes of the A2 card; deciding scene = reindexed)"')
edit("md_attr_row", '''    L += ["", f"Beside: P mean-over-seeds form''', '''    At = R["ATTRIBUTION"]
    L.append("| ATTRIBUTION: M1 of RICLR - RIRAPRP01 (reindexed) separated-better at >= 1 seed | " + " | ".join(fmt(res["seeds"][k]["RICLR_minus_RIRAPRP01"][M1]) for k in ("s0", "s1")) + f" | **{At['SEPARATED_BETTER_AT_ANY_SEED']}** |")
    L.append("| beside: M1 of CLR - RAPRP01 (logged), RAPRP01 = the banked rollout | " + " | ".join(fmt(res["seeds"][k]["CLR_minus_RAPRP01"][M1]) for k in ("s0", "s1")) + f" | {At['n_seeds_separated_better_logged_beside']} of 2 seeds separated-better |")
    L += ["", f"Beside: P mean-over-seeds form''')
edit("md_19rows", '''("RICLRC_minus_P10", "RICLRC - P10 (reindexed composite, D)")):''',
     '''("RICLRC_minus_P10", "RICLRC - P10 (reindexed composite, D)"),
                     ("RICLR_minus_RIRAPRP01", "RICLR - RIRAPRP01 (reindexed, ATTRIBUTION: CLR against RAPR-P01's own checkpoint)"), ("CLR_minus_RAPRP01", "CLR - RAPRP01 (logged, attribution beside; RAPRP01 = the banked rollout)")):''')
edit("md_family_loop", '''    for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0"):''',
     '''    for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01"):''')

edit("main_header", '''res = {"card": "PREREG_LEVELHEAD_CL.md", "card_sha256": sha256_file(CARD), "card_rules_sha256": card_rules_sha(),''',
     '''res = {"card": "PREREG_LEVELHEAD_A2.md (arm CLR; the rules by reference to PREREG_LEVELHEAD_CL.md sections 4-5)", "card_sha256": sha256_file(CARD), "card_rules_sha256": card_rules_sha(), "cl_card_sha256": sha256_file(CL_CARD),''')
edit("main_modes", '''choices=("production", "ka-null", "ka-ri")''', '''choices=("production", "ka-null", "ka-ri", "ka-cl")''')
