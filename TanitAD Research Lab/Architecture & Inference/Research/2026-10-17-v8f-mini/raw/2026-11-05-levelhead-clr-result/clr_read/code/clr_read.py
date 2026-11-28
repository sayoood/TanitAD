#!/usr/bin/env python3
"""clr_read.py -- the T1-kin read of arm CLR (PREREG_LEVELHEAD_A2.md: the level head inside RAPR-P01's recovery recipe).  Cloned from cl_read.py (make_clr_from_cl.py lists every difference).

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
       python -I clr_read.py --mode ka-null | ka-ri | ka-cl                                                          (known answers on EXISTING arrays, no rollout; raw/KA_READER_<mode>.json)
"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
DL = "E:/Projects/TanitAD-artifacts/dagland"
RR = f"{DL}/recover_next/read/code/recover_read.py"
RR_SHA = "de5cfbf572ba2184e5f85a942ac79a6a6fa4df422f83367bbd9fa0e6966e95d1"
REF_ARR = f"{DL}/recover_next/read/raw/arrays"
RI_ARR = f"{DL}/motion_c/ri_read/raw/arrays"
RI_READ_JSON = f"{DL}/motion_c/ri_read/raw/RI_C0_READ.json"
CL_CARD = f"{DL}/motion_c/cl_prereg/PREREG_LEVELHEAD_CL.md"           # sections 4-5 hold the rules P, G, F, D, LEAK-BORNE, which the A2 card takes by reference (CL := CLR)
CARD = f"{DL}/motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md"             # the A2 card: reads, attribution, outcomes
A2_FROZEN = f"{DL}/motion_c/a2_prereg/FROZEN_SHA.txt"
CL_RAW = f"{DL}/motion_c/cl_read/raw"
CL_READ_JSON = f"{CL_RAW}/CL_READ.json"
CL_ARR = f"{CL_RAW}/arrays"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/clr_read/out"
WORK_DIR = "F:/TanitAD-scratch/motion_c/clr_read/work"
TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
SEEDS = (0, 1)
M1 = "M1 crawl share"
SHARE_BAR, LEAK_FRACTION = 0.5, 0.5
NEW_ROLES = {"CLR": ("logged", "off"), "CLRC": ("logged", "kbox"), "FZCLR": ("hold_now", "off"), "RICLR": ("reindexed", "off"), "RICLRC": ("reindexed", "kbox"), "RIRAPRP01": ("reindexed", "off")}
RUN_OF = {role: "CLR" for role in NEW_ROLES}
RUN_OF["RIRAPRP01"] = "RAPRP01"          # the run dir M2b_s{s}_recft_<RUN_OF[role]> each role is rolled from (RIRAPRP01 = the attribution base: RAPR-P01's own checkpoint under `reindexed`)
REUSED = {"C0": REF_ARR, "C0C": REF_ARR, "FZC0": REF_ARR, "C0REP": REF_ARR, "RAPRP01": REF_ARR, "RIC0": RI_ARR, "RIC0C": RI_ARR}
REUSED_SHA = {"C0_s0": "5b04b8f92c710444bb1517d802d0b9a57b8134c41f4304d635417585cbc163ab", "C0_s1": "8280c3f47d90de9f68432bc1a97fe584b87d1e0fee94ac5a00c465dfc71d7ef9",
              "C0C_s0": "265c804427eb66d6b541236697e941c184e487c8a31c03ee894a377d6141c8dc", "C0C_s1": "33e78abcf17519b425f81be7a8091b23595f5fae9d825e32e6977b1ab972b791",
              "FZC0_s0": "cbce1e7f46995f71891d46c0e30027cc9929e164106b261b78131c5d8a9461c4", "FZC0_s1": "61ea757c32b1af7694a5b2bef6613cbb3ece8d2de919c06154fb3a4e50b07e52",
              "C0REP_s0": "f6b6336fd1d6421f910f78e3bde6f9c236a03643c2ac845ef25eda7feb2b4a7b", "C0REP_s1": "7ee047c07ca387a76453ef23f7e53f67376ca9d30fbb98d368bded6663cc7b80",
              "RIC0_s0": "31e2ad8ddb4f6e8e6bfe6ee780f47b7c0adcba1cc1f9047071eb64939042a244", "RIC0_s1": "3b9b814d7b285cb05fcefbdfeb21c71eab01d728383ba1d737da84f1b322601d",
              "RIC0C_s0": "f88c7a80aec9d3711cd616c9b45aa064109368e6a16d806c5e7bb65af275efab", "RIC0C_s1": "7ef6cc2a30600d11c822a2b609b4aea40eb5f20c1f1b72759e6290dc5973050f",
              "RAPRP01_s0": "79c449422a9d59d799d6869b19e42ba9566ee879efb55239f7d9b13936e577ca", "RAPRP01_s1": "057289b16f691fd54c505d3075901c8cff8ad1fe78a80ea9ef7539fc75046598"}
CL_ARR_SHA = {"CL_s0": "cfd58740a967c0ae5d666da921c2834c8eda8bb0e5fee945a627a6b4ab8c1070", "CL_s1": "9a7323a52a6e21929d078b64f772eb04ac9dd3b9036d738cab2fb7b6d5e939c8",       # CL's own arrays: the ka-cl known answer only
              "CLC_s0": "e2c2a3c532bfe7c0587359c2a25fffefbe6d7673e1c47d5f74843e99bb463571", "CLC_s1": "fec315481b9a6b81c0e4ad4a64d72ae15aa268194669e254013f25d0f54ec64f",
              "FZCL_s0": "82675a88cabfd2bf681d738e78d6ba6e6e4ccc5cbb8f69acfba1e084d087cab5", "FZCL_s1": "f2e94871b3c4a24841130baa864bbd31ba594bc56a027dbb52d24fecda2af591",
              "RICL_s0": "9335cc19736711294d7d9da1f5c9e29a8cd0d11fb14fda2596c4223d6e9f4ff0", "RICL_s1": "16f84ec94cfe3530c0b70a94ea11cb1622b9889d4ce5c46a4336552fe5c621cc",
              "RICLC_s0": "7e416cf5c0b16518af2dea09c5c236c1ba634862c73a13b927cc0c1c5f5c5278", "RICLC_s1": "7aca44562915fed8de5fd6b44c4ae944d0c46d310da41cbe7fbcf9d0ea5385f7"}
FOUR_FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NEXT_ARM = {
    "D-HOLDS": "counted DOMINANT: signal the PI for scale (data and size), with the video evidence the PI asked for",
    "P-HOLDS-G-OR-D-FAILS": "next arm CLR-W2 = the arm's dose (W / 2) under the same card (A2 card, Outcomes); then the composite's failing row is the target",
    "P-FAILS-ATTR-NOT-SEPARATED-BETTER": "the level adds nothing even when needed (CLR - RAPRP01 on M1 not separated-better): LLR (the lead-aware head) is the next arm, wiring per the O-sensitivity of CLR (readout/ and WIRING.md)",
    "P-FAILS-ATTR-SEPARATED-BETTER": "UNCOMMITTED CELL: the A2 card commits no branch for 'P fails, CLR - RAPRP01 on M1 separated-better' (the level helps relative to RAPR-P01 without clearing P): the Master Mind / the PI names "
                                     "the next arm before any launch; LLR (the committed branch of the not-separated cell) is the nearest candidate; both numbers are reported",
}
rr = SR = GD = None


# ------------------------------------------------------------------------------------------------ loading
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def init(path=None):
    """Load the RECOVER-NEXT reader BY FILE PATH (its sha256 is checked first) and its pinned instruments (heavy_init)."""
    global rr, SR, GD
    path = path or RR
    got = sha256_file(path)
    if path == RR and got != RR_SHA:
        raise SystemExit(f"REFUSED: {path} is {got}, not the pinned {RR_SHA}")
    spec = importlib.util.spec_from_file_location("recover_read_cl", path)
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    rr.heavy_init()
    SR, GD = rr.SR, rr.GD
    assert rr.M1 == M1 and tuple(rr.SEEDS) == SEEDS
    return rr


def card_rules_sha(path=CL_CARD):
    """sha256 of the card's sections 4 and 5 (LF-normalised): the committed rules, from '## 4.' up to '## 6.'."""
    t = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    i, j = t.index("## 4. Committed rules"), t.index("## 6. Variance named")
    return hashlib.sha256(t[i:j].encode("utf-8")).hexdigest()


A2_CARD_SHA = "9a117b201d74db13aa7ccb43b0407de7634d319141203550047929bda5b66a31"      # the A2 card as frozen 2026-11-05T15:52 (a2_prereg/FROZEN_SHA.txt): any edit is a refused control
CARD_RULES_SHA = "ff6d672a5f53dfdab62e18922fe5653ce67a794a75fd902b094c535fb9c25bf8"      # sections 4 and 5 of the card as committed 2026-11-05 (a later edit of the rules is a refused control, never silent)


# ------------------------------------------------------------------------------------------------ pure layer: the card's rules (no file access)
def family_table(rows):
    """Per family (the four of the card, STRATEGIC included) the rows separated on the better / worse side, from {label: {delta, lo, hi, separated}} (19 rows)."""
    out = {f: {"n_rows": 0, "separated_better": [], "separated_worse": []} for f in FOUR_FAMILIES}
    for lab, r in rows.items():
        f = SR.FAMILY[lab]
        if f not in out:
            continue
        out[f]["n_rows"] += 1
        if SR.better(GD, lab, r):
            out[f]["separated_better"].append(lab)
        if SR.worse(GD, lab, r):
            out[f]["separated_worse"].append(lab)
    return out


def rule_P(cells):
    """P (card section 4): at EACH seed M1 of RICLR - RIC0 separated-better AND share >= 0.5 (an undefined share, no excess to close, fails).  ``cells`` = {seed key: m1_cell}.
    The RECOVER reader's mean-over-seeds form (``primary_rule``) is reported beside and does not decide."""
    per = {}
    for k, c in cells.items():
        sh = c["share_closed"]
        per[k] = {"separated_better": bool(c["better"]), "share_closed": sh, "share_at_least_bar": bool(sh is not None and sh >= SHARE_BAR)}
        per[k]["holds"] = bool(per[k]["separated_better"] and per[k]["share_at_least_bar"])
    return {"per_seed": per, "share_bar": SHARE_BAR, "n_seeds_separated_better": int(sum(p["separated_better"] for p in per.values())),
            "PASS": bool(len(per) == len(SEEDS) and all(p["holds"] for p in per.values())), "mean_form_reported_beside": rr.primary_rule(cells)}


def rule_G(xy_by_seed, rows_by_seed=None):
    """G: the RECOVER reader's ``guards_rule`` (ADE6 point <= +0.0604 m, no separated-worse non-crawl row) on RICLR - RIC0.  Beside: ADE6's CI upper end against the same bound."""
    g = rr.guards_rule(xy_by_seed)
    if rows_by_seed:
        g["ade6_ci_upper_end_within_guard_reported_beside"] = {k: bool(rows_by_seed[k]["ADE6"]["hi"] <= rr.ADE_GUARD_M) for k in rows_by_seed}
    return g


def rule_F(effect_m1_delta, replicate_m1_delta, logged_effect_m1_delta=None):
    """F: ``floor_rule`` with the deciding effect (RICLR - RIC0 on M1) against the replicate pair's (C0REP - C0, LOGGED).  Beside: the logged effect (CLR - C0) against the same floor."""
    f = rr.floor_rule(effect_m1_delta, replicate_m1_delta)
    if logged_effect_m1_delta is not None:
        f["logged_effect_form_reported_beside"] = rr.floor_rule(logged_effect_m1_delta, replicate_m1_delta)
    return f


def rule_leak(cells_logged, cells_reindexed):
    """LEAK-BORNE (card section 4): at either seed (logged share - reindexed share) > LEAK_FRACTION x the logged share.  A logged share that is undefined or not positive cannot be leak-borne (nothing was closed on the
    camera clock); an undefined reindexed share counts 0 (the conservative reading ``share_mean`` uses)."""
    per, hit = {}, False
    for k in cells_logged:
        sl, sr = cells_logged[k]["share_closed"], cells_reindexed[k]["share_closed"]
        srv = 0.0 if sr is None else float(sr)
        h = bool(sl is not None and sl > 0 and (float(sl) - srv) > LEAK_FRACTION * float(sl))
        per[k] = {"logged_share": sl, "reindexed_share": sr, "excess_over_half_logged": None if sl is None else float(sl) - srv - LEAK_FRACTION * float(sl), "leak_borne": h}
        hit = hit or h
    return {"per_seed": per, "fraction": LEAK_FRACTION, "LEAK_BORNE": bool(hit)}


def rule_D(ra_by_seed, P, G):
    """D (counted DOMINANT): Read A (D1, D2, D3 over four families) of RICLRC - P10 at BOTH seeds, with P and G holding.  ``ra_by_seed`` = {seed key: read_a_block}."""
    both = bool(len(ra_by_seed) == len(SEEDS) and all(v["DOMINANT"] for v in ra_by_seed.values()))
    both3 = bool(len(ra_by_seed) == len(SEEDS) and all(v["DOMINANT_three_families_strategic_paused"] for v in ra_by_seed.values()))
    return {"read_a_both_seeds": both, "P_holds": bool(P), "G_holds": bool(G), "DOMINANT_COUNTED": bool(both and P and G), "three_family_form_reported_beside": bool(both3),
            "by_seed": {k: {"DOMINANT": bool(v["DOMINANT"]), "D1": bool(v["D1"]), "D2": bool(v["D2"]), "D3": bool(v["D3"]), "D3_by_family": v["D3_by_family"], "separated_worse_rows": v["separated_worse_rows"],
                            "failing_terms": v["failing_terms"]} for k, v in ra_by_seed.items()}}


def rule_attr(cells_ri, cells_lg):
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
def read_all(arrays, n_boot=2000, log=print):
    """``arrays`` = {seed: {role: array dict}}, roles CLR CLRC FZCLR RICLR RICLRC C0 C0C FZC0 C0REP RIC0 RIC0C.  Every role of a seed must carry the same windows.  Returns the result dict (rules, outcome, 19-row blocks)."""
    res = {"seeds": {}}
    cells_ri, cells_lg, cells_fz, xy_ri, rows_ri, ra_riclc, eff_ri, eff_lg, rep = {}, {}, {}, {}, {}, {}, {}, {}, {}
    attr_ri, attr_lg = {}, {}
    for s in SEEDS:
        k, ar = f"s{s}", arrays[s]
        eid = ar["C0"]["eid"]
        for role, z in ar.items():
            assert np.array_equal(z["eid"], eid), f"s{s} {role}: windows differ from C0's"
        cls, _ = GD.classify(ar["C0"]["v0"], ar["C0"]["g"])

        def rows(x, y, full=None):
            return rr.rows_of(rr.Aof(ar[x]), rr.Aof(ar[y]), eid, cls, n_boot, full)

        def vs_p10(x):
            return rr.rows_of(rr.Aof(ar[x]), rr.Bof(ar[x]), eid, cls, n_boot)[0]
        b = {}
        b["RICLR_minus_RIC0"], b["M1_per_class_split_RICLR_minus_RIC0"] = rows("RICLR", "RIC0", M1)
        b["RIC0_minus_P10"], b["RICLR_minus_P10"] = vs_p10("RIC0"), vs_p10("RICLR")
        b["RICLRC_minus_P10"], b["RIC0C_minus_P10"] = vs_p10("RICLRC"), vs_p10("RIC0C")
        b["RICLRC_minus_RICLR"], b["RICLRC_minus_RIC0C"] = rows("RICLRC", "RICLR")[0], rows("RICLRC", "RIC0C")[0]
        b["CLR_minus_C0"], b["M1_per_class_split_CLR_minus_C0"] = rows("CLR", "C0", M1)
        b["C0_minus_P10"], b["CLR_minus_P10"] = vs_p10("C0"), vs_p10("CLR")
        b["CLRC_minus_P10"], b["C0C_minus_P10"] = vs_p10("CLRC"), vs_p10("C0C")
        b["CLRC_minus_C0C"], b["CLRC_minus_CLR"] = rows("CLRC", "C0C")[0], rows("CLRC", "CLR")[0]
        b["C0REP_minus_C0"] = rows("C0REP", "C0")[0]
        b["FZCLR_minus_FZC0"], b["FZC0_minus_P10"], b["FZCLR_minus_P10"] = rows("FZCLR", "FZC0")[0], vs_p10("FZC0"), vs_p10("FZCLR")
        # ATTRIBUTION (A2 card): CLR against RAPR-P01's own checkpoint, on the reindexed clock (RIRAPRP01 = new rollout) and on the logged clock (the banked RAPRP01 arrays)
        b["RICLR_minus_RIRAPRP01"], b["M1_per_class_split_RICLR_minus_RIRAPRP01"] = rows("RICLR", "RIRAPRP01", M1)
        b["CLR_minus_RAPRP01"], b["M1_per_class_split_CLR_minus_RAPRP01"] = rows("CLR", "RAPRP01", M1)
        b["RIRAPRP01_minus_P10"], b["RAPRP01_minus_P10"] = vs_p10("RIRAPRP01"), vs_p10("RAPRP01")
        b["RIRAPRP01_minus_RIC0"], b["RAPRP01_minus_C0"] = rows("RIRAPRP01", "RIC0")[0], rows("RAPRP01", "C0")[0]
        attr_ri[k] = rr.m1_cell(b["RICLR_minus_RIRAPRP01"][M1], b["RIRAPRP01_minus_P10"][M1])
        attr_lg[k] = rr.m1_cell(b["CLR_minus_RAPRP01"][M1], b["RAPRP01_minus_P10"][M1])
        b["m1_cell_attribution_reindexed"], b["m1_cell_attribution_logged"] = attr_ri[k], attr_lg[k]
        cells_ri[k] = rr.m1_cell(b["RICLR_minus_RIC0"][M1], b["RIC0_minus_P10"][M1])
        cells_lg[k] = rr.m1_cell(b["CLR_minus_C0"][M1], b["C0_minus_P10"][M1])
        cells_fz[k] = rr.m1_cell(b["FZCLR_minus_FZC0"][M1], b["FZC0_minus_P10"][M1])
        xy_ri[k], rows_ri[k] = SR.read_xy(GD, b["RICLR_minus_RIC0"]), b["RICLR_minus_RIC0"]
        ra_riclc[k] = rr.read_a_block(SR, GD, b["RICLRC_minus_P10"])
        eff_ri[k], eff_lg[k], rep[k] = b["RICLR_minus_RIC0"][M1]["delta"], b["CLR_minus_C0"][M1]["delta"], b["C0REP_minus_C0"][M1]["delta"]
        b["read_a_RICLRC_minus_P10"], b["read_a_CLRC_minus_P10_logged_beside"] = ra_riclc[k], rr.read_a_block(SR, GD, b["CLRC_minus_P10"])
        b["read_a_RIC0C_base_beside"], b["read_a_C0C_base_beside"] = rr.read_a_block(SR, GD, b["RIC0C_minus_P10"]), rr.read_a_block(SR, GD, b["C0C_minus_P10"])
        b["read_xy_RICLR_minus_RIC0"], b["read_xy_CLR_minus_C0_logged_beside"] = xy_ri[k], SR.read_xy(GD, b["CLR_minus_C0"])
        b["m1_cell_reindexed"], b["m1_cell_logged"], b["m1_cell_hold_now"] = cells_ri[k], cells_lg[k], cells_fz[k]
        b["family_tables"] = {c: family_table(b[c]) for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01")}
        b["floor_max_abs_dev_vs_C0_m"] = {role: float(max(np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) for u, v in zip(rr.Bof(ar[role]), rr.Bof(ar["C0"])))) for role in ar if role != "C0"}
        res["seeds"][k] = b
        log(k, "M1 RICLR-RIC0 %+.4f [%+.4f, %+.4f] sep %s | RIC0-P10 %+.4f | share %s | logged share %s | ADE6 RICLR-RIC0 %+.4f | RICLRC read A DOMINANT %s" % (
            b["RICLR_minus_RIC0"][M1]["delta"], b["RICLR_minus_RIC0"][M1]["lo"], b["RICLR_minus_RIC0"][M1]["hi"], b["RICLR_minus_RIC0"][M1]["separated"], b["RIC0_minus_P10"][M1]["delta"],
            cells_ri[k]["share_closed"], cells_lg[k]["share_closed"], b["RICLR_minus_RIC0"]["ADE6"]["delta"], ra_riclc[k]["DOMINANT"]))
    P, G = rule_P(cells_ri), rule_G(xy_ri, rows_ri)
    F = rule_F(eff_ri, rep, eff_lg)
    leak = rule_leak(cells_lg, cells_ri)
    D = rule_D(ra_riclc, P["PASS"], G["PASS"])
    A = rule_attr(attr_ri, attr_lg)
    out = decide(P["PASS"], G["PASS"], D["DOMINANT_COUNTED"], A["n_seeds_separated_better_reindexed"], leak["LEAK_BORNE"])
    if out["branch"] in ("D-HOLDS", "P-HOLDS-G-OR-D-FAILS") and not F["PASS"]:
        out["notes"].append("F fails: |RICLR - RIC0| on M1 does not exceed the C0REP replicate floor at every seed; the card's D does not require F, the training-variance reading is the PI's / the Master Mind's")
    freeze_reading = rr.freeze_reading(bool(rule_P(cells_lg)["PASS"]), cells_fz)
    res["rules"] = {"P": P, "G": G, "F": F, "D": D, "LEAK_BORNE": leak, "ATTRIBUTION": A, "deciding_scene": "reindexed",
                    "P_logged_form_beside": rule_P(cells_lg), "hold_now_beside": {"m1_cells": cells_fz, "reading": freeze_reading, "deciding": False,
                                                                                  "note": "FZCLR vs FZC0 (frame freeze): beside, never deciding (card section 4). hold_now repeats the CURRENT frame over the history rows and the level head reads only the current frame"}}
    res["outcome"] = out
    res["strategic_family"] = rr.STRATEGIC_NOTE
    res["variance_questions"] = {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)", "training_run": "answered ONLY by the C0REP floor (rule F); a separated CI is necessary, not sufficient (H-ESTIM-SEED-1)",
                                 "inference": "the planner is deterministic at eval (no sampling): no inference seed applies"}
    return res


# ------------------------------------------------------------------------------------------------ arrays (files) and controls
def reused_arrays(s):
    z, ctl = {}, {}
    for role, d in REUSED.items():
        p = f"{d}/{role}_s{s}.npz"
        got = sha256_file(p)
        ctl[f"{role}_s{s}"] = {"path": p, "sha256": got, "pinned": REUSED_SHA[f"{role}_s{s}"], "PASS": got == REUSED_SHA[f"{role}_s{s}"]}
        z[role] = rr.read_arrays(p)
    return z, ctl


def expected_flags(role):
    scene, feas = NEW_ROLES[role]
    return f"--smooth sg72 --scene {scene}" + (f" --feasible {feas}" if feas != "off" else "")


def run_controls(out_root, raw):
    """Production controls before any row: done markers (flags, tool sha, one checkpoint sha per seed), run headers (scene, feasible, smooth, windows), the KA-NEW file, the pre-check file, the reused arrays' sha256, the card's rules."""
    ctl, ckpts = {}, {}
    for s in SEEDS:
        for role in NEW_ROLES:
            tag = f"{role}_s{s}"
            c = {"PASS": False}
            try:
                d = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
                h = rr.run_header(out_root, tag)
                feas = (h.get("feasible") or {}).get("mode", "off") if isinstance(h.get("feasible"), dict) else ("off" if not h.get("feasible") else str(h.get("feasible")))
                smooth = h["smooth"]["mode"] if isinstance(h.get("smooth"), dict) else h.get("smooth")
                c = {"flags_as_expected": d["flags"].startswith(expected_flags(role)), "tool_sha256_pinned": d["tool_sha256"] == TOOL_SHA, "ckpt_is_the_expected_run": f"/M2b_s{s}_recft_{RUN_OF[role]}/ckpt_last.pt" in d["ckpt"].replace("\\", "/"),
                     "header_scene": h["scene"] == NEW_ROLES[role][0], "header_feasible": feas == NEW_ROLES[role][1], "header_smooth_sg72": smooth == "sg72", "header_n_windows_2399": int(h["n_windows"]) == 2399,
                     "header_kin_yaw": h["kin_mode"] == "yaw", "ckpt_sha256": d["ckpt_sha256"], "threads": d.get("threads")}
                ckpts.setdefault((s, RUN_OF[role]), set()).add(d["ckpt_sha256"])
                c["PASS"] = all(v for k, v in c.items() if isinstance(v, bool))
            except Exception as e:      # a missing marker / header is a failed control, named
                c["error"] = f"{type(e).__name__}: {e}"
            ctl[f"rollout_{tag}"] = c
    ctl["one_checkpoint_per_seed"] = {"PASS": bool(all(len(v) == 1 for v in ckpts.values()) and len(ckpts) == 2 * len(SEEDS)), "ckpt_sha256": {f"s{s}_{r}": sorted(v) for (s, r), v in ckpts.items()}}
    try:
        ka = json.load(open(f"{CL_RAW}/KA_NEW.json", encoding="utf-8"))
        ctl["KA_NEW"] = {"PASS": bool(ka["PASS"] is True and ka["logged_first64_16t_vs_banked_C0_s0"]["tool_sha256"] == TOOL_SHA), "max_plan_diff_m": ka["max_plan_diff_m"]}
    except Exception as e:
        ctl["KA_NEW"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    try:
        pc = json.load(open(f"{raw}/CLR_PRECHECK.json", encoding="utf-8"))
        same = all(pc["seeds"][f"s{s}"]["ckpt_sha256"] in ckpts.get((s, "CLR"), set()) and pc["seeds"][f"s{s}"].get("raprp01_ckpt_sha256") in ckpts.get((s, "RAPRP01"), set()) for s in SEEDS)
        ctl["CLR_PRECHECK"] = {"PASS": bool(pc["PASS"] is True and same), "ckpt_sha256_equals_rollouts": same, "e_level_abs_sum": {k: v.get("e_level_abs_sum") for k, v in pc["seeds"].items()}, "w": pc.get("w")}
    except Exception as e:
        ctl["CLR_PRECHECK"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    got = card_rules_sha()
    ctl["card_rules"] = {"PASS": got == CARD_RULES_SHA, "sha256": got, "pinned": CARD_RULES_SHA, "card_sha256": sha256_file(CL_CARD)}
    a2, fz = sha256_file(CARD), open(A2_FROZEN, encoding="utf-8").readline().split()[0]
    ctl["card_a2"] = {"PASS": bool(a2 == A2_CARD_SHA == fz), "sha256": a2, "pinned": A2_CARD_SHA, "frozen_sha_file": fz}
    return ctl


def extract_new(out_root, arr_dir, workers):
    """The ten CLR rollouts to per-window arrays (the reader's extractor, one thread each, ``workers`` at a time)."""
    jobs = [(f"{role}_s{s}", s) for s in SEEDS for role in NEW_ROLES]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(rr.extract, tag, s, out_root, arr_dir): tag for tag, s in jobs}
        for f in cf.as_completed(futs):
            f.result()
    return {tag: f"{arr_dir}/{tag}.npz" for tag, _ in jobs}


# ------------------------------------------------------------------------------------------------ known answers of the reader (existing arrays, no rollout)
def ka_arrays(mode, s):
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


def ka_check(mode, res):
    """The expected answer of each known-answer mode, from first principles (ka-null) and from the banked RI_C0_READ.json (ka-ri)."""
    chk = {}
    if mode == "ka-cl":
        return ka_cl_checks(res)
    ri = json.load(open(RI_READ_JSON, encoding="utf-8"))
    for s in SEEDS:
        k = f"s{s}"
        b = res["seeds"][k]
        if mode == "ka-null":
            chk[f"{k}_all_RICLR_minus_RIC0_rows_exactly_zero"] = all(r["delta"] == 0.0 and not r["separated"] for r in b["RICLR_minus_RIC0"].values())
            chk[f"{k}_all_CLR_minus_C0_rows_exactly_zero"] = all(r["delta"] == 0.0 and not r["separated"] for r in b["CLR_minus_C0"].values())
            chk[f"{k}_shares_zero_or_undefined"] = all((c["share_closed"] in (0.0, None) or abs(c["share_closed"]) == 0.0) for c in (b["m1_cell_reindexed"], b["m1_cell_logged"]))
            chk[f"{k}_RICLRC_read_a_equals_banked_RIC0C"] = bool(b["read_a_RICLRC_minus_P10"]["DOMINANT"] == ri["seeds"][k]["readA_RIC0C"]["DOMINANT"]
                                                                and b["read_a_RICLRC_minus_P10"]["separated_worse_rows"] == ri["seeds"][k]["readA_RIC0C"]["separated_worse_rows"])
        elif mode == "ka-ri":
            for lab, r in ri["seeds"][k]["RIC0_minus_C0"].items():
                got = b["CLR_minus_C0"][lab]
                chk[f"{k}_CLR_minus_C0[{lab}]_equals_banked_RIC0_minus_C0"] = all(float(got[q]) == float(r[q]) for q in ("delta", "lo", "hi")) and bool(got["separated"]) == bool(r["separated"])
            chk[f"{k}_CLR_minus_P10_equals_banked_RIC0_minus_P10"] = all(float(b["CLR_minus_P10"][lab][q]) == float(r[q]) for lab, r in ri["seeds"][k]["RIC0_minus_P10"].items() for q in ("delta", "lo", "hi"))
            chk[f"{k}_CLRC_minus_P10_equals_banked_RIC0C_minus_P10"] = all(float(b["CLRC_minus_P10"][lab][q]) == float(r[q]) for lab, r in ri["seeds"][k]["RIC0C_minus_P10"].items() for q in ("delta", "lo", "hi"))
            chk[f"{k}_C0_minus_P10_equals_banked"] = all(float(b["C0_minus_P10"][lab][q]) == float(r[q]) for lab, r in ri["seeds"][k]["C0_minus_P10"].items() for q in ("delta", "lo", "hi"))
            ex = ri["seeds"][k]["C0_minus_P10"][M1]["delta"]
            want = (-ri["seeds"][k]["RIC0_minus_C0"][M1]["delta"] / ex) if ex > 0 else None
            got = b["m1_cell_logged"]["share_closed"]
            chk[f"{k}_m1_share_is_minus_delta_over_excess"] = bool((want is None and got is None) or (want is not None and got is not None and abs(got - want) < 1e-12))
    if mode == "ka-null":
        chk["label_is_attribution_not_separated_better"] = res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
        for k in ("s0", "s1"):          # the arm is its base and the attribution base is the null: every attribution row is exactly zero (no separation)
            chk[f"{k}_attribution_rows_exactly_zero"] = all(r["delta"] == 0.0 and not r["separated"] for c in ("RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01") for r in res["seeds"][k][c].values())
        chk["not_leak_borne"] = not res["rules"]["LEAK_BORNE"]["LEAK_BORNE"]
        chk["G_holds"] = bool(res["rules"]["G"]["PASS"])
        chk["P_fails_F_fails"] = (not res["rules"]["P"]["PASS"]) and (not res["rules"]["F"]["PASS"])
    return chk


# ------------------------------------------------------------------------------------------------ RESULTS.md
def fmt(r):
    return "%+.4f [%+.4f, %+.4f]%s" % (r["delta"], r["lo"], r["hi"], " *" if r["separated"] else "")


def render_md(res):
    o, R = res["outcome"], res["rules"]
    banner = [] if res.get("mode", "production") == "production" else [f"**KNOWN-ANSWER RUN `{res['mode']}` on EXISTING arrays (no CLR rollout): a test of this reader, not a read of arm CLR.**", ""]
    L = banner + [f"# CLR read: arm CLR (level-from-image head inside RAPR-P01's recovery recipe), T1-kin, generated {res.get('clock', '')}",
         "", f"**Outcome: {o['label']}**" + (f" (underlying: {o['underlying']})" if o["label"] != o["underlying"] else ""), "", f"Next (A2 card, Outcomes): {o['next']}.",
         "" if not o["notes"] else "Notes: " + "; ".join(o["notes"]) + ".", "",
         f"Tier: {res.get('tier', '')}. Estimator: {res.get('estimator', '')}. `*` = interval excludes 0. STRATEGIC: {res['strategic_family']}", "",
         "## Rules (CL card section 4 by reference, CL := CLR; attribution and outcomes of the A2 card; deciding scene = reindexed)", "", "| rule | s0 | s1 | holds |", "|---|---|---|---|"]
    P, G, F, D, LK = R["P"], R["G"], R["F"], R["D"], R["LEAK_BORNE"]
    sh = lambda c: "None" if c is None else "%.3f" % (float(c) + 0.0)          # + 0.0 turns -0.0 into 0.0
    L.append("| P: M1 separated-better and share >= 0.5 | " + " | ".join(f"sep {P['per_seed'][k]['separated_better']}, share {sh(P['per_seed'][k]['share_closed'])}" for k in ("s0", "s1")) + f" | **{P['PASS']}** |")
    L.append("| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | " + " | ".join(f"ADE6 {G['per_seed'][k]['ade6_delta']:+.4f}, worse: {', '.join(G['per_seed'][k]['non_crawl_separated_worse']) or 'none'}" for k in ("s0", "s1")) + f" | **{G['PASS']}** |")
    L.append("| F: abs(RICLR-RIC0)[M1] > abs(C0REP-C0)[M1] | " + " | ".join(f"{F['per_seed'][k]['abs_R_minus_C0']:.4f} vs {F['per_seed'][k]['abs_C0rep_minus_C0']:.4f}" for k in ("s0", "s1")) + f" | **{F['PASS']}** |")
    L.append("| LEAK-BORNE: logged share - reindexed share > half the logged share | " + " | ".join(f"logged {sh(LK['per_seed'][k]['logged_share'])}, reindexed {sh(LK['per_seed'][k]['reindexed_share'])}" for k in ("s0", "s1")) + f" | **{LK['LEAK_BORNE']}** |")
    L.append("| D: RICLRC - P10 read A (D1, D2, D3 four families) + P + G | " + " | ".join(f"D1 {D['by_seed'][k]['D1']}, D2 {D['by_seed'][k]['D2']}, D3 {D['by_seed'][k]['D3']}" for k in ("s0", "s1")) + f" | **{D['DOMINANT_COUNTED']}** |")
    At = R["ATTRIBUTION"]
    L.append("| ATTRIBUTION: M1 of RICLR - RIRAPRP01 (reindexed) separated-better at >= 1 seed | " + " | ".join(fmt(res["seeds"][k]["RICLR_minus_RIRAPRP01"][M1]) for k in ("s0", "s1")) + f" | **{At['SEPARATED_BETTER_AT_ANY_SEED']}** |")
    L.append("| beside: M1 of CLR - RAPRP01 (logged), RAPRP01 = the banked rollout | " + " | ".join(fmt(res["seeds"][k]["CLR_minus_RAPRP01"][M1]) for k in ("s0", "s1")) + f" | {At['n_seeds_separated_better_logged_beside']} of 2 seeds separated-better |")
    L += ["", f"Beside: P mean-over-seeds form {P['mean_form_reported_beside']['PASS']} (mean share {P['mean_form_reported_beside']['mean_share_closed']:.3f}); P on logged scene {R['P_logged_form_beside']['PASS']}; "
          f"D three-family form {D['three_family_form_reported_beside']}; F logged form {F['logged_effect_form_reported_beside']['PASS']}; hold_now (FZCLR vs FZC0) reading {R['hold_now_beside']['reading']['reading']}, "
          f"shares " + ", ".join(f"{k} {sh(c['share_closed'])}" for k, c in R["hold_now_beside"]["m1_cells"].items()) + " (never deciding).", "", "## D failing terms", ""]
    for k in ("s0", "s1"):
        L.append(f"* {k}: " + ("; ".join(D["by_seed"][k]["failing_terms"]) or "none"))
    for c, title in (("RICLR_minus_RIC0", "RICLR - RIC0 (reindexed, the deciding contrast)"), ("CLR_minus_C0", "CLR - C0 (logged)"), ("RICLRC_minus_P10", "RICLRC - P10 (reindexed composite, D)"),
                     ("RICLR_minus_RIRAPRP01", "RICLR - RIRAPRP01 (reindexed, ATTRIBUTION: CLR against RAPR-P01's own checkpoint)"), ("CLR_minus_RAPRP01", "CLR - RAPRP01 (logged, attribution beside; RAPRP01 = the banked rollout)")):
        L += ["", f"## 19 rows: {title}", "", "| row | family | s0 | s1 |", "|---|---|---|---|"]
        for lab in GD.LABELS:
            L.append(f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(res["seeds"][k][c][lab]) for k in ("s0", "s1")) + " |")
    L += ["", "## Families (separated-better / separated-worse row counts)", "", "| contrast | family | s0 better / worse | s1 better / worse |", "|---|---|---|---|"]
    for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01"):
        for f in FOUR_FAMILIES:
            t = [res["seeds"][k]["family_tables"][c][f] for k in ("s0", "s1")]
            L.append(f"| {c} | {f} | " + " | ".join(f"{len(x['separated_better'])} / {len(x['separated_worse'])} (of {x['n_rows']})" for x in t) + " |")
    L += ["", "## Controls", "", "| control | PASS |", "|---|---|"]
    for k, v in res.get("controls", {}).items():
        L.append(f"| {k} | {v.get('PASS')} |")
    L += ["", "## Variance named", ""] + [f"* {k}: {v}" for k, v in res["variance_questions"].items()]
    return "\n".join(x for x in L if x is not None) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("production", "ka-null", "ka-ri", "ka-cl"), default="production")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--arr-dir", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--results-md", default=f"{PK}/RESULTS.md", help="where the generated RESULTS.md goes (the package's own by default; the rehearsal points it elsewhere)")
    a = ap.parse_args(argv)
    init()
    rr.PK = PK                                           # the extractor logs go to THIS package, not the RECOVER-NEXT one
    rr.WORK = f"{a.work_dir}/rig"
    os.makedirs(a.raw, exist_ok=True)
    arr_dir = a.arr_dir or f"{a.raw}/arrays"
    res = {"card": "PREREG_LEVELHEAD_A2.md (arm CLR; the rules by reference to PREREG_LEVELHEAD_CL.md sections 4-5)", "card_sha256": sha256_file(CARD), "card_rules_sha256": card_rules_sha(), "cl_card_sha256": sha256_file(CL_CARD), "reader": os.path.abspath(__file__).replace("\\", "/"), "reader_sha256": sha256_file(__file__),
           "recover_read_sha256": sha256_file(RR), "mode": a.mode, "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only", "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    arrays, rctl = {}, {}
    if a.mode == "production":
        ctl = run_controls(a.out_root, a.raw)
        for s in SEEDS:
            z, c = reused_arrays(s)
            arrays[s], rctl = z, {**rctl, **c}
        ctl["reused_arrays"] = {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}
        res["controls"] = ctl
        bad = [k for k, v in ctl.items() if not v.get("PASS")]
        if bad:
            res["refused"] = f"controls failed: {bad}"
            json.dump(res, open(f"{a.raw}/CLR_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
            print("REFUSED:", res["refused"])
            return 3
        paths = extract_new(a.out_root, arr_dir, a.workers)
        for s in SEEDS:
            for role in NEW_ROLES:
                arrays[s][role] = rr.read_arrays(paths[f"{role}_s{s}"])
        out_json = f"{a.raw}/CLR_READ.json"
    else:
        ctl = {}
        for s in SEEDS:
            arrays[s], c = ka_arrays(a.mode, s)
            rctl.update(c)
        res["controls"] = {"reused_arrays": {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}}
        if not res["controls"]["reused_arrays"]["PASS"]:
            print("REFUSED: a reused array differs from its pinned sha256")
            return 3
        out_json = f"{a.raw}/KA_READER_{a.mode}.json"
    res.update(read_all(arrays, a.n_boot))
    if a.mode != "production":
        chk = ka_check(a.mode, res)
        res["ka_checks"], res["KA_PASS"] = chk, bool(all(chk.values()))
        print(f"KA {a.mode}: {sum(chk.values())}/{len(chk)} checks hold; label {res['outcome']['label']}")
        for k, v in chk.items():
            if not v:
                print("  RED", k)
    json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    if a.mode == "production":
        open(a.results_md, "w", encoding="utf-8", newline="\n").write(render_md(res))
        print("OUTCOME", res["outcome"]["label"], "|", res["outcome"]["next"])
    else:
        open(f"{a.raw}/KA_READER_{a.mode}_RESULTS.md", "w", encoding="utf-8", newline="\n").write(render_md(res))      # the generated format on a known answer
    return 0 if res.get("KA_PASS", True) else 2


if __name__ == "__main__":
    sys.exit(main())
