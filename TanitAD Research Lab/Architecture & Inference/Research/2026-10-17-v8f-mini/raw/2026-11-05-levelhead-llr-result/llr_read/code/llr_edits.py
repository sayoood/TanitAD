"""llr_edits.py -- the exact edits make_llr_from_clr.py applies to the role-renamed CLR reader to produce code/llr_read.py.  Entries: ("edit", name, old, new) = exact-once replacement;
("span", name, start, end, new) = replace from the exactly-once ``start`` up to (not including) the exactly-once ``end``."""
import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
_PK = os.path.dirname(_HERE)
_TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


CLR_SHAS = dict(l.split() for l in open(f"{_PK}/raw/CLR_ARRAY_SHAS.txt", encoding="utf-8") if l.strip())
MODEL_SHA, DATA_SHA, TRAINER_SHA = (_sha(f"{_TREE}/{p}") for p in ("stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/scripts/v8f_mini_train.py"))

DOC_NEW = r'''llr_read.py -- the T1-kin read of arm LLR (PREREG_LEVELHEAD_A2.md: the lead-aware vision head + the level head, token wiring, inside RAPR-P01's recovery recipe).  Cloned from clr_read.py (make_llr_from_clr.py lists every difference).

The rules P, G, F, D and LEAK-BORNE are the CL card's section 4 with CL := LLR (A2 card, Reads), implemented exactly as clr_read.py implements them; the A2 card's ATTRIBUTION is LLR - CLR on M1 and all 19 rows (reported; the card's outcome does not branch on it).
Reuses the RECOVER-NEXT reader's functions (heavy_init, extract, read_arrays, rows_of, Aof, Bof, m1_cell, guards_rule, floor_rule, read_a_block, GD.classify), loaded BY FILE PATH (pins checked by heavy_init).

Roles per seed s (tag = role + "_s{s}"):
  NEW (this package's rollouts, F:/TanitAD-scratch/motion_c/llr_read/out, the LLR read tree = the 83c9ee0 lead-head stack)  LLR logged | LLRC logged + kbox | FZLLR hold_now | RILLR reindexed | RILLRC reindexed + kbox
  REUSED (read only, sha256-pinned)  C0, C0C, FZC0, C0REP (recover_next/read/raw/arrays)   RIC0, RIC0C (motion_c/ri_read/raw/arrays)   CLR (logged) and RICLR (reindexed) = the BANKED attribution base (motion_c/clr_read/raw/arrays)
Rules (ALL evaluated on `reindexed` unless named):
  P  per seed: M1 of RILLR - RIC0 separated-better AND share = -(RILLR - RIC0)[M1] / (RIC0 - P10)[M1] >= 0.5; BOTH seeds.
  G  guards_rule on RILLR - RIC0: ADE6 point <= +0.0604 m and no non-crawl row separated-worse, both seeds.
  F  |RILLR - RIC0| on M1 (point) > |C0REP - C0| on M1 (LOGGED replicate pair), both seeds.
  D  composite RILLRC - P10: D1, D2, D3 over FOUR families (STRATEGIC included; the three-family form beside) at both seeds, AND P AND G.  Only this counts.
  LEAK-BORNE  (logged share - reindexed share) > half the logged share at either seed (logged share > 0 required).
  ATTRIBUTION  (A2 card, reported) M1 of RILLR - RICLR (reindexed, deciding) and LLR - CLR (logged, beside), all 19 rows each; the banked CLR arrays are the base (no new base rollout).
  hold_now (FZLLR vs FZC0): beside, never deciding.
Outcome (A2 card, Outcomes): D-HOLDS (counted DOMINANT, signal the PI) | P-HOLDS-G-OR-D-FAILS (the arm's dose) | P-FAILS-FAMILY-EXHAUSTED (LLR P fails: the vision-head family on frozen single-frame features is exhausted at these forms;
  the remaining levers are the named blockers of option (c): T2 re-perception, the A40); LEAK-BORNE overrides the label.
Tier T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2).  Estimator: paired episode-cluster bootstrap B = 2000, seed 0 (gapdecomp.decompose) over the 141 eval episodes /
2,399 windows: answers another draw of EPISODES only; the training-variance floor is rule F, the inference seed does not apply (deterministic planner).

usage: python -I llr_read.py --mode production [--out-root F:/TanitAD-scratch/motion_c/llr_read/out] [--workers 6]     (controls first; a failed control REFUSES the read, exit 3)
       python -I llr_read.py --mode ka-null | ka-ri | ka-clr                                                          (known answers on EXISTING arrays, no rollout; raw/KA_READER_<mode>.json; ka-clr feeds CLR's five arrays as LLR's)'''

EDITS = []


def edit(name, old, new):
    EDITS.append(("edit", name, old, new))


def span(name, start, end, new):
    EDITS.append(("span", name, start, end, new))


edit("paths", '''CL_RAW = f"{DL}/motion_c/cl_read/raw"
CL_READ_JSON = f"{CL_RAW}/CL_READ.json"
CL_ARR = f"{CL_RAW}/arrays"
''', '''CLR_RAW = f"{DL}/motion_c/clr_read/raw"                              # the CLR read: the banked attribution base (CLR, RICLR arrays) and the ka-clr known answer
CLR_READ_JSON = f"{CLR_RAW}/CLR_READ.json"
CLR_ARR = f"{CLR_RAW}/arrays"
CLR_PRECHECK_JSON = f"{CLR_RAW}/CLR_PRECHECK.json"
A2_DIR = f"{DL}/motion_c/a2_prereg"                                  # FROZEN_SHA.txt lists the card, A2_RULING_1.md and FROZEN_LLR.txt (WIRING=token, W_LEAD=3.362)
''')
edit("pins_stack", 'TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"\n',
     f'TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"\nMODEL_SHA, DATA_SHA, TRAINER_SHA = "{MODEL_SHA}", "{DATA_SHA}", "{TRAINER_SHA}"       # the 83c9ee0 lead-head blobs of the LLR read tree (KA-NEW2 records them)\n')
edit("new_roles", ', "RICLR": ("reindexed", "off")}', '}')
span("run_of", 'RUN_OF["RICLR"] = "CLR"', 'REUSED = {', '')
edit("reused_dirs", '"CLR": REF_ARR,', '"CLR": CLR_ARR, "RICLR": CLR_ARR,')
edit("reused_sha", '"CLR_s0": "79c449422a9d59d799d6869b19e42ba9566ee879efb55239f7d9b13936e577ca", "CLR_s1": "057289b16f691fd54c505d3075901c8cff8ad1fe78a80ea9ef7539fc75046598"}',
     ', '.join(f'"{r}_s{s}": "{CLR_SHAS[f"{r}_s{s}"]}"' for r in ("CLR", "RICLR") for s in (0, 1)) + '}')
span("clr_arr_sha", 'CL_ARR_SHA = {', 'FOUR_FAMILIES = (',
     'CLR_ARR_SHA = {' + ', '.join(f'"{k}": "{v}"' for k, v in CLR_SHAS.items()) + '}       # CLR\'s own five arrays x two seeds: the ka-clr known answer (and the base: CLR, RICLR)\n')
span("next_arm", 'NEXT_ARM = {', 'rr = SR = GD = None', '''NEXT_ARM = {
    "D-HOLDS": "counted DOMINANT: signal the PI for scale (data and size), with the video evidence the PI asked for",
    "P-HOLDS-G-OR-D-FAILS": "next arm LLR-W2 = the arm's dose halved under the same card (A2 card, Outcomes: 'the arm's dose (W / 2)'; the card does not say whether W, W_lead or both are halved for LLR: the Master Mind names it); the composite's failing row is the target",
    "P-FAILS-FAMILY-EXHAUSTED": "the vision-head family on frozen single-frame features is exhausted at these forms (A2 card, Outcomes: LLR P fails); the remaining levers are the named blockers of option (c) (T2 re-perception, the A40), reported to the PI with the numbers",
}
''')
span("decide", 'def decide(', 'def read_all(', '''def decide(P, G, D, n_attr_separated_better, leak):
    """The A2 card's Outcomes for arm LLR as one function.  Returns {label, branch, underlying, next, counts_as_result, notes}.
    branch: D-HOLDS (D counted: D1-D3 at both seeds with P and G) | P-HOLDS-G-OR-D-FAILS (the arm's dose under the same card) | P-FAILS-FAMILY-EXHAUSTED (A2 card: 'LLR P fails').  The attribution LLR - CLR on M1 is
    REPORTED in the notes and never branches the outcome (the card branches on P only).  LEAK-BORNE overrides the LABEL whatever else the row reads and is reported as the camera-clock effect, never as a result;
    its next arm is the P-fails branch's."""
    notes = []
    if D:
        branch = "D-HOLDS"
    elif P:
        branch = "P-HOLDS-G-OR-D-FAILS"
        notes.append("G fails" if not G else "G holds, D fails")
    else:
        branch = "P-FAILS-FAMILY-EXHAUSTED"
    if not P and not G:
        notes.append("G fails too (the guards read worse alongside the failed P)")
    notes.append("attribution LLR - CLR on M1 (reindexed) separated-better at %d of 2 seeds (reported; the card's outcome does not branch on it)" % int(n_attr_separated_better))
    if leak:
        return {"label": "LEAK-BORNE", "branch": "LEAK-BORNE", "underlying": branch, "next": NEXT_ARM["P-FAILS-FAMILY-EXHAUSTED"] + " (LEAK-BORNE: the effect is the camera-clock effect, never a result; next as for 'P fails')",
                "counts_as_result": False, "notes": notes}
    return {"label": branch, "branch": branch, "underlying": branch, "next": NEXT_ARM[branch], "counts_as_result": True, "notes": notes}


''')
edit("read_all_doc", 'roles LLR LLRC FZLLR RILLR RILLRC C0 C0C FZC0 C0REP RIC0 RIC0C.', 'roles LLR LLRC FZLLR RILLR RILLRC C0 C0C FZC0 C0REP RIC0 RIC0C and the banked attribution base CLR RICLR.')
edit("attr_comment", "LLR against RAPR-P01's own checkpoint, on the reindexed clock (RICLR = new rollout)", "LLR against CLR's own checkpoint (the BANKED arrays of the CLR read), on the reindexed clock (RICLR)")
edit("helpers", 'def run_controls(out_root, raw):', '''def frozen_sha_of(name):
    """The sha256 FROZEN_SHA.txt lists for ``name`` (lines '<sha256> *<name>'), None if unlisted."""
    for line in open(f"{A2_DIR}/FROZEN_SHA.txt", encoding="utf-8"):
        p = line.split()
        if len(p) == 2 and p[1].lstrip("*") == name:
            return p[0]
    return None


def parse_frozen_llr():
    """FROZEN_LLR.txt's KEY=VALUE lines (comments skipped)."""
    d = {}
    for line in open(f"{A2_DIR}/FROZEN_LLR.txt", encoding="utf-8"):
        t = line.strip()
        if t and not t.startswith("#") and "=" in t:
            k, v = t.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def run_controls(out_root, raw):''')
edit("ckpt_count", 'len(ckpts) == 2 * len(SEEDS)', 'len(ckpts) == len(SEEDS)')
edit("ka_new", '''        ka = json.load(open(f"{CL_RAW}/KA_NEW.json", encoding="utf-8"))
        ctl["KA_NEW"] = {"PASS": bool(ka["PASS"] is True and ka["logged_first64_16t_vs_banked_C0_s0"]["tool_sha256"] == TOOL_SHA), "max_plan_diff_m": ka["max_plan_diff_m"]}''',
     '''        ka = json.load(open(f"{PK}/raw/KA_NEW2.json", encoding="utf-8"))
        ctl["KA_NEW2"] = {"PASS": bool(ka["PASS"] is True and ka["tool_sha256"] == TOOL_SHA and all(ka["cases"][c]["PASS"] for c in "ABC") and ka["model_sha256"] == MODEL_SHA and ka["data_sha256"] == DATA_SHA
                                       and ka["trainer_sha256"] == TRAINER_SHA), "max_plan_diff_m": ka["max_plan_diff_m"]}''')
edit("ka_new_err", 'ctl["KA_NEW"] = {"PASS": False,', 'ctl["KA_NEW2"] = {"PASS": False,')
edit("precheck_same", ''' and pc["seeds"][f"s{s}"].get("raprp01_ckpt_sha256") in ckpts.get((s, "CLR"), set()) for s in SEEDS)''',
     ''' for s in SEEDS)
        cp = json.load(open(CLR_PRECHECK_JSON, encoding="utf-8"))
        base_ok = all(pc["seeds"][f"s{s}"].get("clr_ckpt_sha256") is not None and pc["seeds"][f"s{s}"]["clr_ckpt_sha256"] == cp["seeds"][f"s{s}"]["ckpt_sha256"] for s in SEEDS)      # the base = the CLR checkpoint the banked CLR arrays came from''')
edit("precheck_pass", 'bool(pc["PASS"] is True and same)', 'bool(pc["PASS"] is True and same and base_ok)')
edit("precheck_w", '"w": pc.get("w")}', '"w": pc.get("w"), "w_lead": pc.get("w_lead"), "aux_cond": pc.get("aux_cond"), "base_clr_ckpt_sha256_is_the_banked_arrays_checkpoint": base_ok, "e_aux_abs_sum": {k: v.get("e_aux_abs_sum") for k, v in pc["seeds"].items()}}')
edit("frozen_controls", '    return ctl\n\n\ndef extract_new(', '''    for nm in ("A2_RULING_1.md", "FROZEN_LLR.txt"):
        got_f, want_f = sha256_file(f"{A2_DIR}/{nm}"), frozen_sha_of(nm)
        ctl["frozen_" + nm] = {"PASS": bool(want_f is not None and got_f == want_f), "sha256": got_f, "frozen_sha_file_line": want_f}
    fl = parse_frozen_llr()
    ctl["frozen_llr_values"] = {"PASS": bool(fl.get("WIRING") == "token" and fl.get("W_LEAD") == "3.362"), "values": fl}
    return ctl


def extract_new(''')
span("ka_arrays", 'def ka_arrays(mode, s):', 'def ka_check(mode, res):', '''def ka_arrays(mode, s):
    z, ctl = reused_arrays(s)            # C0, C0C, FZC0, C0REP, RIC0, RIC0C and the banked attribution base CLR, RICLR
    base = {"LLR": "C0", "LLRC": "C0C", "FZLLR": "FZC0", "RILLR": "RIC0", "RILLRC": "RIC0C"}
    if mode == "ka-null":       # the arm IS its base (C0's family) and the attribution base is the null too: RICLR := RIC0, CLR := C0
        z.update({r: z[b] for r, b in base.items()})
        z["RICLR"], z["CLR"] = z["RIC0"], z["C0"]
    elif mode == "ka-ri":       # the RI rollout fed in the LOGGED slot (LLR := RIC0, LLRC := RIC0C), the reindexed slots null: LLR - C0 must be RI_C0_READ.json's RIC0 - C0, row for row
        z.update({**{r: z[b] for r, b in base.items()}, "LLR": z["RIC0"], "LLRC": z["RIC0C"]})
        z["RICLR"], z["CLR"] = z["RIC0"], z["C0"]
    else:                       # ka-clr: CLR's own five arrays fed as LLR's with the REAL attribution base (CLR's arrays): every LLR contrast equals CLR_READ.json's CLR contrast, every attribution row is exactly 0
        for r, c in (("LLR", "CLR"), ("LLRC", "CLRC"), ("FZLLR", "FZCLR"), ("RILLR", "RICLR"), ("RILLRC", "RICLRC")):
            p = f"{CLR_ARR}/{c}_s{s}.npz"
            got = sha256_file(p)
            ctl[f"{c}_s{s}_fed_as_{r}"] = {"path": p, "sha256": got, "pinned": CLR_ARR_SHA[f"{c}_s{s}"], "PASS": got == CLR_ARR_SHA[f"{c}_s{s}"]}
            z[r] = rr.read_arrays(p)
    return z, ctl


def _clr_name(c):
    """The CLR reader's name of an LLR contrast key (the reverse of the role renaming)."""
    for a, b in (("RILLRC", "RICLRC"), ("RILLR", "RICLR"), ("FZLLR", "FZCLR"), ("LLRC", "CLRC"), ("LLR", "CLR")):
        c = re.sub(r"(?<![A-Za-z0-9])" + a + r"(?![A-Za-z0-9])", b, c)
    return c


def ka_clr_checks(res):
    """ka-clr: CLR's arrays fed as LLR's with the real base must give CLR_READ.json back: every arm contrast row for row (delta, lo, hi, separated), every rule value, the label's branch, and EXACT-ZERO attribution rows."""
    cr, chk, n = json.load(open(CLR_READ_JSON, encoding="utf-8")), {}, 0
    for s in SEEDS:
        k = f"s{s}"
        for c, v in res["seeds"][k].items():
            if not isinstance(v, dict) or "LLR" not in c or c.startswith("family_tables") or c.startswith("m1_cell") or c.startswith("read_") or not all(lab in v for lab in GD.LABELS[:3]):
                continue
            old = cr["seeds"][k].get(_clr_name(c))
            if old is None:
                continue
            ok = all(float(v[lab][q]) == float(old[lab][q]) for lab in old for q in ("delta", "lo", "hi")) and all(bool(v[lab]["separated"]) == bool(old[lab]["separated"]) for lab in old)
            chk[f"{k}_{c}_equals_CLR_READ[{_clr_name(c)}]"] = bool(ok)
            n += len(old)
        chk[f"{k}_m1_cells_equal"] = all(res["seeds"][k][a] == cr["seeds"][k][a] for a in ("m1_cell_reindexed", "m1_cell_logged", "m1_cell_hold_now"))
        chk[f"{k}_attribution_rows_exactly_zero"] = all(r["delta"] == 0.0 and not r["separated"] for c in ("RILLR_minus_RICLR", "LLR_minus_CLR") for r in res["seeds"][k][c].values())
    chk["rows_compared_at_least_400"] = n >= 400      # 12 arm contrasts x 19 rows x 2 seeds = 456
    R, Rc = res["rules"], cr["rules"]
    chk["rule_P_equals_CLR"] = all(R["P"]["per_seed"][k] == Rc["P"]["per_seed"][k] for k in ("s0", "s1")) and R["P"]["PASS"] == Rc["P"]["PASS"]
    chk["rule_G_F_equal_CLR"] = R["G"]["PASS"] == Rc["G"]["PASS"] and R["F"]["PASS"] == Rc["F"]["PASS"] and R["F"]["per_seed"] == Rc["F"]["per_seed"]
    chk["rule_D_leak_equal_CLR"] = R["D"]["DOMINANT_COUNTED"] == Rc["D"]["DOMINANT_COUNTED"] and R["LEAK_BORNE"]["LEAK_BORNE"] == Rc["LEAK_BORNE"]["LEAK_BORNE"] and R["LEAK_BORNE"]["per_seed"] == Rc["LEAK_BORNE"]["per_seed"]
    chk["attribution_not_separated_at_any_seed"] = not R["ATTRIBUTION"]["SEPARATED_BETTER_AT_ANY_SEED"]
    mapped = {"P-FAILS-ATTR-NOT-SEPARATED-BETTER": "P-FAILS-FAMILY-EXHAUSTED", "P-FAILS-ATTR-SEPARATED-BETTER": "P-FAILS-FAMILY-EXHAUSTED"}
    chk["label_is_CLRs_branch_mapped"] = res["outcome"]["label"] == mapped.get(cr["outcome"]["label"], cr["outcome"]["label"])
    return chk


''')
edit("ka_check_head", '''    if mode == "ka-cl":
        return ka_cl_checks(res)''', '''    if mode == "ka-clr":
        return ka_clr_checks(res)''')
edit("ka_null_label", 'chk["label_is_attribution_not_separated_better"] = res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"', 'chk["label_is_P_fails_family_exhausted"] = res["outcome"]["label"] == "P-FAILS-FAMILY-EXHAUSTED"')
edit("md_title", "# LLR read: arm LLR (level-from-image head inside RAPR-P01's recovery recipe)", "# LLR read: arm LLR (lead-aware head + level head, token wiring, inside RAPR-P01's recovery recipe)")
edit("md_attr", "| ATTRIBUTION: M1 of RILLR - RICLR (reindexed) separated-better at >= 1 seed |", "| ATTRIBUTION (reported; the outcome does not branch on it): M1 of RILLR - RICLR (reindexed) separated-better at >= 1 seed |")
edit("md_titles", "(reindexed, ATTRIBUTION: LLR against RAPR-P01's own checkpoint)", "(reindexed, ATTRIBUTION: LLR against CLR's own checkpoint, the banked CLR arrays)")
edit("main_choices", 'choices=("production", "ka-null", "ka-ri", "ka-cl")', 'choices=("production", "ka-null", "ka-ri", "ka-clr")')
