#!/usr/bin/env python3
"""dag2_read.py -- the dev-box T1-kin read of PREREG_V8F_DAGGER2 (claim row H-V8F-DAGGER2-1).  BUILT AND SELF-TESTED, NOT RUN ON REAL DAG2 OUTPUTS.

PREREG (binding, nothing here changes it): Project Steering/PREREG_V8F_DAGGER2.md @ 1446333d (as amended by its Amendment 1, A1-2026-10-27), sha256 PREREG2_SHA256 below.  It inherits PREREG_V8F_DAGGER (+ Amendments 1, 2) verbatim,
including the reader's controls and tiers, and adds the crawl set C, R-CRAWL, R-MOVE, the guards, the fixed replicate floor RF and the outcome table.

NOTHING IS RE-DERIVED.  Every number comes from a function of an existing reader, run on these arms:
  layer G  dag_read.read() (the DAGGER reader, pinned sha256): R-TLAT, R-TLON, R-ADE, R-T0, every K-control, the bank controls, COVERAGE, S-POS vs CTRV.  The reader's slots are
           DAG := DAG2 and DAG0 := DAG2-0.  Two single-purpose substitutions, each one a PREREG_V8F_DAGGER2 change and each checked, never skipped:
             g_argv      the init checkpoint is DAG_s{seed} (change 1/2), so `--init-ckpt` must end `M2b_s{seed}_dag/ckpt_last.pt`; the reader's own check then runs with that tail
                         presented as the M2b tail it is written for.  Every other clause of g_argv (steps, lr, warmup, p-gt, ego-perturb-p, dagger-p, the DAG-only flags) is the reader's.
             g_replicate change 3: no replicate of DAG2-0 is trained.  The reader's ORC0 slot is filled with DAG2-0's OWN entry (so its RF row reads exactly 0.0, a control),
                         and RF is the PREREG's constant (RF_FIXED); bar = dag_read.bar_for(hd, RF) = max(2F, 2*RF), inclusive.
  layer A  Read A's driver code (readA_code/: rr_lib.py, readA.py, build_report.py, sha256-pinned copies of dagreread/code/) over the pinned reader sources (dagreread/reader_src/, the files
           committed at bdf8d561): comps / L1 / L2 / M1 rows (readA.rows_block), TAC + DK rows (t1kf_read_a1.prep / rows), the Read A table (build_report.collect / verdict / outcome,
           extracted by AST exactly as rr_lib extracts the readers).  Pairs: DAG2 - P10, DAG2 - DAG2-0, DAG - P10 (re-computed from DAG's banked dumps AND required to equal the banked
           READ_A.json: control RC-DAG), DAG2 - DAG (beside).

The only new logic is the PREREG's own composition and it is the part the self-test mutates (dag2t1_selftest.py): the crawl set C per seed, R-CRAWL, R-MOVE, the guards, the newly-worse
clause with its two exemptions, the fixed-RF bars, and the outcome partition coded cell by cell from the PREREG table.

AMENDMENT 1 (A1-2026-10-27, commit 1446333d) made that table complete: the 16 guarded (R-CRAWL, R-MOVE) x 2-seed cells are partitioned on R-MOVE alone (FIXED 1, PARTIAL 11, NO EFFECT 4, none
unplaced).  PARTIAL is `R-MOVE at both seeds but R-CRAWL fails somewhere; or R-MOVE at one seed only`.  A one-seed R-MOVE is reported as PARTIAL (one seed) and adopts nothing (H-ESTIM-SEED-1):
only FIXED, and PARTIAL with R-MOVE at both seeds, adopt DAG2.  The UNPLACED path (rc 3) is kept as a defensive check; through real cells it is unreachable.

    python dag2_read.py --manifest runs.json --out DAG2_READ.json --tool-root <tree> ... (see build_parser)      rc 0 outcome written | 2 REFUSED (nothing written) | 3 UNPLACED (defensive, unreachable) | 4 BLOCKED (written)
    python dag2_read.py --static-check ...                                                                       pins and static inputs only; touches no run output
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
H2 = HERE.replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_DAGGER2 (committed numbers)
PREREG2_ID = "Project Steering/PREREG_V8F_DAGGER2.md @ 1446333d (Amendment 1, A1-2026-10-27)"
PREREG2_SHA256 = "c4239e4590ee26e30ce6f525987aca82e3fb8ff034b7264321e2d0faa724f1c6"      # sha256 of the 1446333d blob (raw bytes; git cat-file blob)
CLAIM_ROW = "H-V8F-DAGGER2-1"
SEEDS = (0, 1)
HALF_F = 0.0604                                                         # R-T0: point <= +F/2 = 0.0604 m, inclusive
RF_FIXED = {"lat": {0: 0.0067, 1: 0.0043}, "lon": {0: 0.0251, 1: 0.0139}}   # change 3: DAGGER's measured replicate floor, fixed
BARS_PREREG = {"lat": {0: 0.0248, 1: 0.0248}, "lon": {0: 0.0502, 1: 0.0278}}  # change 3: max(2F, 2*RF), inclusive, as printed in the PREREG
CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),                       # change 4: the rows separated-worse for plain DAG - P10 at that seed
             1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE")}       # (RESULT_V8F_DAG_REREAD section 1) except accel / DK-VIOL, which belong to DAG-OACC
EXEMPT_NEWLY = ("accel MAE", "DK-VIOL")                                  # change 5, last guard: exempt from the newly-worse clause, reported
COVERAGE_MIN = 0.5
N_BOOT = 2000

# The PREREG's outcome table AS AMENDED (Amendment 1, section `The outcome table, as read from now on`), cell by cell (markdown bold markers are markup, not text; a control checks every
# row against that section of the PREREG file).  Only PARTIAL differs from the as-landed table: its condition and its committed next.
OUTCOME_TABLE_2 = (
    ("FIXED", "R-CRAWL and R-MOVE at both seeds; guards held",
     "adopt DAG2 as the base. Read A is re-run on DAG2 and on DAG2 with DAG-OACC's lever when it exists; a DOMINANT read there is the dominance claim"),
    ("PARTIAL", "R-MOVE at both seeds but R-CRAWL fails somewhere; or R-MOVE at one seed only; guards held",
     "adopt DAG2 as the base if R-MOVE holds at both seeds, else DAG stays the base; report. The residual crawl is then a launch-intent defect (O's v0 at tick 1, `R76`). "
     "Next: the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on the base's states, under its own PREREG"),
    ("NO EFFECT", "R-MOVE fails at both seeds; guards held",
     "the residual crawl is not an own-state-distribution defect. Next: as PARTIAL, on DAG states"),
    ("HARMS", "a guard fails at ≥ 1 seed", "DAG2 is not adopted; DAG stays the base. Next: as PARTIAL, on DAG states"),
    ("BLOCKED", "COVERAGE < 0.5", "the rejoin-target form (DAGGER §7)"),
    ("REFUSED", "any control fails", "fix and re-run"),
)
OUTCOME_NAMES_2 = tuple(r[0] for r in OUTCOME_TABLE_2)
UNPLACED_TEXT = ("UNPLACED: this reading matches no row of the PREREG's outcome table. Amendment 1 (A1-2026-10-27) made the table complete over the 16 guarded cells, so this path is unreachable "
                 "through real cells: it means a row of the table is missing from this reader, or the PREREG changed: a reader / PREREG gap, not a result. No outcome word is invented; "
                 "the PI / Master Mind decides, WITHOUT looking at which way the numbers fall.")
SPOS_ROWS = ("S-POS@30", "S-POS@50", "S-POS@80")


# ---------------------------------------------------------------------------------------------------------------- pure logic (no numpy, no torch): this is what the self-test mutates
def sep_worse(row, direction="lower"):
    """separated AND the point on the wrong side.  direction 'lower' = an error row (positive delta is worse); 'higher' = an agreement / F1 row (negative delta is worse)."""
    return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))


def sep_better(row, direction="lower"):
    return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))


def rf_bars(bar_for):
    """bar(h, s) = bar_for(h, RF_FIXED[h][s]) = max(2F_h, 2*RF_h,s) (the DAGGER reader's own bar_for), inclusive."""
    return {hd: {s: bar_for(hd, RF_FIXED[hd][s]) for s in SEEDS} for hd in ("lat", "lon")}


def clears_bar(row, bar):
    """The DAGGER definition of 'a head clears': separated-better AND point >= bar, inclusive.  Reported beside; the DAG2 table's guards are 'not separated-worse'."""
    return bool(sep_better(row, "higher") and row["delta"] >= bar)


def crawl_flags(seed, rows_vs_p10, rows_vs_dag20):
    """R-CRAWL: every row of C_s reads NOT separated-worse for DAG2 - P10.  R-MOVE: every row of C_s reads separated-better for DAG2 - DAG2-0.  All rows are error-like (lower is better)."""
    c = CRAWL_SET[seed]
    crawl = {l: not sep_worse(rows_vs_p10[l]) for l in c}
    move = {l: sep_better(rows_vs_dag20[l]) for l in c}
    return {"crawl_set": list(c), "R-CRAWL_rows": crawl, "R-MOVE_rows": move, "R-CRAWL": all(crawl.values()), "R-MOVE": all(move.values())}


def guard_flags(tlat, tlon, ade, t0_point):
    """Each guard HELD?  R-TLAT / R-TLON (F1, higher better) and R-ADE (error) are 'not separated-worse' for DAG2 - DAG2-0; R-T0 is the point <= HALF_F (inclusive)."""
    return {"R-TLAT": not sep_worse(tlat, "higher"), "R-TLON": not sep_worse(tlon, "higher"), "R-ADE": not sep_worse(ade, "lower"), "R-T0": bool(t0_point <= HALF_F)}


def newly_worse(rows_dag, rows_dag2, directions):
    """Rows (Read A labels) that were NOT separated-worse for DAG - P10 and ARE separated-worse for DAG2 - P10.  Accel and DK-VIOL are exempt (and reported).  A row that is NOT COMPUTED on
    either side cannot be judged and is returned under 'not_judged' (it never silently counts as passing)."""
    viol, exempt_hit, not_judged = [], [], []
    for lab, r2 in rows_dag2.items():
        r1 = rows_dag.get(lab)
        if r2.get("status") == "NOT COMPUTED" or r1 is None or r1.get("status") == "NOT COMPUTED":
            not_judged.append(lab)
            continue
        d = directions.get(lab, "lower")
        was_worse = sep_worse(r1, d)
        now_worse = sep_worse(r2, d)
        if now_worse and not was_worse:
            (exempt_hit if lab in EXEMPT_NEWLY else viol).append(lab)
    return {"violations": viol, "exempt_but_newly_worse_reported": exempt_hit, "not_judged": not_judged, "held": not viol and not not_judged}


def table_row_2(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE_2 if nm == name)


def decide2(seed_flags, guards):
    """The PREREG's table as amended (A1-2026-10-27), coded cell by cell.  ``seed_flags[s]`` = {"R-CRAWL": bool, "R-MOVE": bool}; ``guards[s]`` = {guard name: held}.  REFUSED and BLOCKED are decided
    before this runs.  HARMS first (a guard fails at >= 1 seed); with the guards held the 16 (R-CRAWL, R-MOVE) x 2-seed cells are partitioned on R-MOVE: FIXED (R-CRAWL and R-MOVE at both seeds) /
    PARTIAL (clause a: R-MOVE at both seeds but R-CRAWL fails somewhere; clause b: R-MOVE at exactly one seed, whatever R-CRAWL reads) / NO EFFECT (R-MOVE at neither).  Every matching row is returned
    in table order.  None matching = UNPLACED: a defensive path, unreachable through real cells since Amendment 1.
    Adoption (Amendment 1, `What R-MOVE can and cannot claim`): only FIXED, and PARTIAL with R-MOVE at both seeds, adopt DAG2.  A one-seed R-MOVE is reported as PARTIAL (one seed) and adopts nothing:
    ``PARTIAL_one_seed`` / ``outcome_qualifier`` ("one seed") / ``adopts_DAG2``."""
    if tuple(sorted(seed_flags)) != SEEDS or tuple(sorted(guards)) != SEEDS:
        raise ValueError(f"decide2 needs exactly seeds {SEEDS}")
    cr = {s: bool(seed_flags[s]["R-CRAWL"]) for s in SEEDS}
    mv = {s: bool(seed_flags[s]["R-MOVE"]) for s in SEEDS}
    held = all(all(guards[s].values()) for s in SEEDS)
    both = {s: cr[s] and mv[s] for s in SEEDS}
    fixed = held and all(both.values())
    partial_a = held and all(mv.values()) and not all(cr.values())                    # "R-MOVE at both seeds but R-CRAWL fails somewhere"
    partial_b = held and sum(mv.values()) == 1                                        # "R-MOVE at one seed only" (Amendment 1: whatever R-CRAWL reads there)
    no_effect = held and not any(mv.values())                                         # "R-MOVE fails at both seeds"
    match = {"FIXED": fixed, "PARTIAL": partial_a or partial_b, "NO EFFECT": no_effect, "HARMS": not held}
    matching = [n for n in OUTCOME_NAMES_2 if match.get(n)]
    out = matching[0] if matching else "UNPLACED"
    partial_one_seed = out == "PARTIAL" and sum(mv.values()) == 1                     # PARTIAL reached with R-MOVE at exactly one seed
    adopts = out == "FIXED" or (out == "PARTIAL" and all(mv.values()))                # only FIXED, and PARTIAL with R-MOVE at BOTH seeds, adopt DAG2
    qualifier = "one seed" if partial_one_seed else None
    flags = {"R-CRAWL_per_seed": {f"s{s}": cr[s] for s in SEEDS}, "R-MOVE_per_seed": {f"s{s}": mv[s] for s in SEEDS}, "guards_held": held,
             "guards_per_seed": {f"s{s}": dict(guards[s]) for s in SEEDS}, "PARTIAL_clause_a": partial_a, "PARTIAL_clause_b": partial_b}
    return {"OUTCOME": out, "outcome_qualifier": qualifier, "PARTIAL_one_seed": partial_one_seed, "adopts_DAG2": adopts, "matching_rows_in_table_order": matching, "flags": flags}


def outcome_block_2(d):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, nxt = table_row_2(out)
    return {**d, "condition": cond, "committed_next": nxt}


def outcome_rc(dec):
    """The reader's exit code for a composed outcome block: 3 for UNPLACED (defensive path, unreachable under the amended table), 0 for every placed outcome (BLOCKED / REFUSED return before this)."""
    return {"UNPLACED": 3}.get(dec["OUTCOME"], 0)


def blocked_block_2(cov):
    cond, nxt = table_row_2("BLOCKED")
    return {"OUTCOME": "BLOCKED", "matching_rows_in_table_order": ["BLOCKED"], "condition": cond, "committed_next": nxt, "coverage": cov, "coverage_min": COVERAGE_MIN}


def compose(g_rows, lay_rows, directions):
    """The PREREG's composition from the readers' rows.  g_rows[s] = layer G's rows of seed s (R-TLAT, R-TLON, R-ADE, R-T0 with 'point'); lay_rows = {"DAG2_minus_P10": {s: Read A rows},
    "DAG2_minus_DAG20": {s: ..}, "DAG_minus_P10": {s: ..}}.  R-CRAWL reads DAG2 - P10, R-MOVE reads DAG2 - DAG2-0, the newly-worse clause compares DAG2 - P10 with DAG - P10."""
    seed_flags, guards, nw, rows = {}, {}, {}, {}
    for s in SEEDS:
        g = g_rows[s]
        sf = crawl_flags(s, lay_rows["DAG2_minus_P10"][s], lay_rows["DAG2_minus_DAG20"][s])
        seed_flags[s] = sf
        gf = guard_flags(g["R-TLAT"], g["R-TLON"], g["R-ADE"], g["R-T0"]["point"])
        nw[s] = newly_worse(lay_rows["DAG_minus_P10"][s], lay_rows["DAG2_minus_P10"][s], directions)
        gf["NEWLY-WORSE"] = nw[s]["held"]
        guards[s] = gf
        rows[f"s{s}"] = {"R-TLAT": g["R-TLAT"], "R-TLON": g["R-TLON"], "R-ADE": g["R-ADE"], "R-T0": g["R-T0"], "crawl": sf, "newly_worse": nw[s], "guards_held": gf}
    return outcome_block_2(decide2(seed_flags, guards)), rows


AMENDED_TABLE_HEADING = "The outcome table, as read from now on"          # Amendment 1's heading; the file also holds the AS-LANDED table above it, which must NOT be the one coded


def amended_table_rows(text):
    """The rows of the table that follows the Amendment 1 heading (markdown bold markers stripped) as (outcome, condition, committed next) tuples; header and separator rows dropped.  None when the
    heading is absent (an as-landed PREREG)."""
    t = text.replace("**", "")
    if AMENDED_TABLE_HEADING not in t:
        return None
    rows = []
    for ln in t.split(AMENDED_TABLE_HEADING, 1)[1].splitlines():
        ln = ln.strip()
        if ln.startswith("|"):
            rows.append(tuple(c.strip() for c in ln.strip("|").split("|")))
        elif rows:
            break
    return [r for r in rows if r != ("outcome", "condition", "committed next") and not set("".join(r)) <= set("-: ")]


def table_problems(text):
    """-> [] iff the table under the Amendment 1 heading equals OUTCOME_TABLE_2 row by row, cell by cell; otherwise one line per difference."""
    rows = amended_table_rows(text)
    if rows is None:
        return [f"the PREREG has no section {AMENDED_TABLE_HEADING!r}: it is not the amended text (Amendment 1, A1-2026-10-27)"]
    want = [tuple(r) for r in OUTCOME_TABLE_2]
    probs = [f"the amended table has {len(rows)} rows, the reader codes {len(want)}"] if len(rows) != len(want) else []
    for i, w in enumerate(want):
        r = rows[i] if i < len(rows) else None
        if r != w:
            probs.append(f"row {w[0]}: the PREREG reads {r!r}, the reader codes {w!r}")
    return probs


def check_prereg2_text(path):
    """The file is the pinned one (raw-byte sha256 of the 1446333d blob) AND the six rows of the amended outcome table are the ones this reader codes, cell by cell."""
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != PREREG2_SHA256:
        return False, f"{path} is not the pinned PREREG_V8F_DAGGER2.md (sha256 {PREREG2_SHA256[:12]}...)"
    miss = table_problems(raw.decode("utf-8"))
    return not miss, miss or "the six rows of the amended outcome table (Amendment 1) are verbatim in the pinned PREREG_V8F_DAGGER2.md"


# ---------------------------------------------------------------------------------------------------------------- pins
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


PINS = {   # every file this reader imports or compares, sha256 (MEASURED 2026-10-27; sources named)
    "dag_read.py": "1cf56c7c30195ac47718bb07feef11531a6309cb01e0b28a34f926c6a696084c",             # dagreread/raw/INPUT_SHA256.txt
    "reader_src/hmc_controls.py": "0bdba3f65c5df8b0efc66702d61f6d8a793d24c4295bc344ab7e40a4cc0bd6c2",
    "reader_src/hmc_lib.py": "7792d7f001b66c9618b7fd40c933f8532e1a86b3ca2d50c0facc7d61275cc9b6",
    "reader_src/lcdiag_common.py": "798089e8abfeb56dfe9e8f6a755448f416be8dfbec948df08638fa997c038a06",
    "reader_src/t1k_read.py": "5610114d3e09501347ed0ac65e44790eb2954eb5d5b703af089a64e845dbf5e4",
    "reader_src/t1kf_read_a1.py": "d16ddd607caf4e05d0bbb667b830299df1f5c1eb0923d995988667afc940e838",
    "reader_src/t1klc_read.py": "9a5623e5f737677e6fe0cf6df41c3588b11282423e27008830a7914253f7a7ca",
    "reader_src/v8f_flat_read.py": "d369639086b4a0b63331977eaeb3fa300229a18b66fd79cbb29cd897bfedc257",
    "readA_code/rr_lib.py": "c3800878cae80bf3018284528e68ed5f88dcebbab7d1125749942735e6cebc99",   # dagreread/MANIFEST_SHA256.txt (code/)
    "readA_code/readA.py": "722913b785f20e0458daed322ecd486fbc2a33106b44b1294f6b59d08d936fa6",
    "readA_code/build_report.py": "c4e94920487381bfb22d06acdff2d79cbb7d97ba52ebb0707003c3e0968e820a",
    "banked/DAG_READ.json": "3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d",       # v8f-dagger-t1/read/DAG_READ.json (hold sha, mk_hold_dag_reread.py)
    "banked/READ_A.json": "1f2b24edcaea3df50b7e56756eb9d46311664a84843148dffb0518ab3afa428a",           # dagreread/raw/READ_A.json
    "banked/T1KF_READ_A1_DAG.json": "deed4c7229c5689267ffe3c4301da184c0ede8656594bd375462d3ed6c002a7f",
    "lead_block_k30.npz": "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9",         # t1kf_read_a1.LEAD_BLOCK_SHA
}
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}   # v8f-dagger/M2b_s{S}_dag/SHA256SUMS.txt
DEFAULTS = {
    "dag_read": "C:/Users/Admin/dagread/dag_read.py",
    "readers": f"{ART}/dagreread/reader_src",
    "reada_code": f"{H2}/readA_code",
    "lead_block": "E:/Projects/TanitAD-artifacts/dag2t1/inputs/b1_eval_lead_block_k30.npz",     # sha256-verified copy of the repo blob (pin lead_block_k30.npz); staged here because the earlier default lived in a session scratchpad
    "dag_t1": f"{ART}/v8f-dagger-t1",
    "dag_read_json": f"{ART}/v8f-dagger-t1/read/DAG_READ.json",
    "reada_banked": f"{ART}/dagreread/raw/READ_A.json",
    "t1kf_banked": f"{ART}/dagreread/raw/t1kf_DAG/T1KF_READ_A1_DAG.json",
    "prereg2": f"{ART}/dag2t1/inputs/PREREG_V8F_DAGGER2.md",           # the 1446333d blob, sha256-verified copy staged here (dagland/'s copy is the as-landed 1d4f4fbb text and is not touched); absolute, like lead_block
}


def pin_problems(paths):
    """paths: {pin name: file path}.  -> list of problems (a missing file or a different sha256)."""
    bad = []
    for k, p in paths.items():
        if not os.path.isfile(p):
            bad.append(f"{k}: {p} is missing")
        elif sha256_file(p) != PINS[k]:
            bad.append(f"{k}: {p} has sha256 {sha256_file(p)[:16]}..., pinned {PINS[k][:16]}...")
    return bad


def load_dag_read(path):
    spec = importlib.util.spec_from_file_location("dag_read", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dag_read"] = m
    spec.loader.exec_module(m)
    return m


def extract_from(src, funcs=(), assigns=(), ns=None):
    """The AST extraction rr_lib / t1klc_read / hmc_lib use: execute only the named top-level functions / assignments of ``src`` (the rest of a module-level script never runs)."""
    ns = {} if ns is None else ns
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in funcs) or (isinstance(node, ast.Assign) and any(getattr(t, "id", "") in assigns for t in node.targets)):
            exec(compile(ast.Module([node], []), src, "exec"), ns)
    return ns


# ---------------------------------------------------------------------------------------------------------------- layer G: the DAGGER reader on DAG2 / DAG2-0
def tail_norm(p):
    return str(p).replace("\\", "/")


def make_g_argv(DR):
    orig = DR.g_argv

    def g_argv(tag, arm, seed, r):
        fl = dict(r["flags"])
        ok, v = DR.one_value(fl, "--init-ckpt")
        want_tail = f"M2b_s{seed}_dag/ckpt_last.pt"
        if not ok:
            return False, ["--init-ckpt is not named exactly once"]
        if not tail_norm(v).endswith(want_tail):
            return False, [f"--init-ckpt {v} is not .../{want_tail} (PREREG_V8F_DAGGER2 change 1/2: the init is DAG_s{seed})"]
        fl["--init-ckpt"] = [[f"M2b_s{seed}/ckpt_last.pt"]]                     # the reader's own clause, presented the tail it is written for
        r2 = dict(r)
        r2["flags"] = fl
        return orig(tag, arm, seed, r2)
    return g_argv


def make_g_replicate(DR):
    def g_replicate(seed, r0, ro, key):
        if tail_norm(ro["dir"]) != tail_norm(r0["dir"]):
            return False, f"fixed-RF mode: the ORC0 slot must be DAG2-0's own run directory ({r0['dir']}), it is {ro['dir']}"
        return True, "PREREG_V8F_DAGGER2 change 3: no replicate of DAG2-0 is trained; the ORC0 slot is DAG2-0's own entry (its RF row must read exactly 0.0) and RF is RF_FIXED"
    return g_replicate


def layer_g(a, DR, pins_dag_sha):
    T = DR.load_t1hd(a.t1hd_read, a.tool_root)
    K = DR.load_t1k(a.t1k_read, a.tool_root)
    if T.SEED_FLOOR != DR.SEED_FLOOR:
        raise DR.Refusal(f"t1hd_read's seed floor {T.SEED_FLOOR} is not the PREREG's {DR.SEED_FLOOR}")
    manifest = DR.load_manifest(a.manifest)
    if set(manifest["arms"]) - {"DAG", "DAG0"}:
        raise DR.Refusal(f"the DAG2 manifest must hold exactly the arms DAG (:= DAG2) and DAG0 (:= DAG2-0), it holds {sorted(manifest['arms'])}")
    manifest["arms"]["ORC0"] = copy.deepcopy(manifest["arms"]["DAG0"])
    init_sha = {s: v.lower() for s, v in DR.parse_kv(a.init_sha, "--init-sha").items()}
    work = a.work_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "_dag2_work")
    os.makedirs(work, exist_ok=True)
    ns = T.parse(["--mode", "real", "--tool-root", a.tool_root, "--eval-cache", a.eval_cache, "--tracks", a.tracks, "--eval-grid", a.eval_grid, "--eval-join", a.eval_join,
                  "--banked-dir", ".", "--out", os.path.join(work, "t1hd_unused.json")])
    backend = DR.RealBackend(T, ns, N_BOOT, work)
    cfg = {"manifest": manifest, "m2b_sha": init_sha, "banked_sha": dict(pins_dag_sha), "n_boot": N_BOOT, "n_windows": DR.N_WINDOWS, "n_episodes": DR.N_EPISODES,
           "keys": {"n": "dagger_n", "eligible": "dagger_eligible", "built_p": "dagger_p", "bank_sha": "dagger_bank_sha256", "init_sha": "init_ckpt.sha256",
                    "window_sha": "train_window_list_sha256", "train_sha": "train_sha256"}, "prereg": a.prereg}
    saved = (DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate)
    DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate = dict(pins_dag_sha), make_g_argv(DR), make_g_replicate(DR)
    try:
        return DR.read(cfg, backend, T, K), T
    finally:
        DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate = saved


# ---------------------------------------------------------------------------------------------------------------- layer A: Read A's rows on DAG2 / DAG2-0 / DAG
class ReadA:
    def __init__(self, a):
        self.a = a
        os.environ["RR_READERS"] = a.readers
        sys.path.insert(0, a.reada_code)
        import numpy as np
        self.np = np
        import rr_lib as R
        import readA as RA
        self.R, self.RA = R, RA
        if os.path.normpath(R.TREE) != os.path.normpath(a.tool_root):
            raise RuntimeError(f"Read A's instruments are bound to {R.TREE} (rr_lib.TREE), the tool root is {a.tool_root}")
        # build_report's own collect / judge / verdict / outcome (AST-extracted; the module-level script around them never runs)
        self.BR = extract_from(os.path.join(a.reada_code, "build_report.py"), funcs=("judge", "tacdk_mean", "collect", "verdict", "outcome"), assigns=("SPEC", "SPOS_KEY", "TWO_F"))
        self.BR["tacdk_mean"] = lambda arm, s, key, ref="P10": (None, None)      # descriptive arm-mean columns of TAC / DK rows only (tacdk_means.py is a module-level script): not carried
        self.dirs = {}
        import argparse as ap
        argv0 = sys.argv[:]
        sys.argv = ["t1kf", "--tree", a.tool_root, "--t1k-read", os.path.join(a.readers, "t1k_read.py"), "--lead-block", a.lead_block, "--out", "unused.json"]
        try:
            spec = importlib.util.spec_from_file_location("t1kf", os.path.join(a.readers, "t1kf_read_a1.py"))
            self.T = importlib.util.module_from_spec(spec)
            sys.modules["t1kf"] = self.T
            spec.loader.exec_module(self.T)
        finally:
            sys.argv = argv0
        self.DIRECTIONS = {sp[1]: sp[4] for sp in self.BR["SPEC"]}
        self.DIRECTIONS["M1 crawl share"] = "lower"

    def dump(self, d):
        return self.R.load_dir(d)

    def tacdk_context(self, d):
        T, np = self.T, self.np
        join = np.load(os.path.join(self.a.eval_join), allow_pickle=False)
        clip = join["clip"].astype(str)[d["clip_index"]]
        blk = T.DK.load_block(self.a.lead_block)
        Jd = T.DK.join_block(blk, clip, 2 * d["ws"].astype(np.int64), d["v0"], k=T.K, dt=T.DT)
        hdk = T.LM.distance_keeping(d["g"][:, :T.K].astype(np.float64), Jd["leads"], Jd["lead_lens"], Jd["speeds"], T.DT)
        return T.Ctx(d["g"].astype(np.float64), d["eid"], Jd, hdk)

    def pair_rows(self, X, Y, d, PX, PY):
        """-> (comps_json, t1kf_json) in the shapes build_report.collect reads."""
        blk = self.RA.rows_block(X, Y, d["g"], d["eid"])
        tk = self.T.rows(PX, PY, d["eid"])
        return blk, tk

    def table(self, comps, tacdk, dagread, spos_arm):
        """build_report.collect over both seeds: {0: rows, 1: rows} with the Read A labels; ``dagread`` supplies the STRATEGIC rows (S-POS vs CTRV, within run)."""
        cj = {"seeds": {f"s{s}": {"rows_arm_minus_P10": comps[s]} for s in SEEDS}}
        tj = {"configs": {"decisive": {"seeds": {f"s{s}": {"rows_X_minus_P10": tacdk[s]} for s in SEEDS}}}}
        self.BR["dagread"] = dagread
        return self.BR["collect"]("X", cj, tj, "decisive", "X", spos_arm)


def rows_equal(a, b, tol=1e-4):
    """Two Read-A row dicts (label -> row) equal: same labels, same separated flag, delta / lo / hi within tol.  -> list of differences."""
    bad = []
    for lab in sorted(set(a) | set(b)):
        ra, rb = a.get(lab), b.get(lab)
        if ra is None or rb is None:
            bad.append(f"{lab}: present on one side only")
            continue
        if ra.get("status") == "NOT COMPUTED" or rb.get("status") == "NOT COMPUTED":
            if ra.get("status") != rb.get("status"):
                bad.append(f"{lab}: NOT COMPUTED on one side only")
            continue
        for k in ("delta", "lo", "hi"):
            if abs(float(ra[k]) - float(rb[k])) > tol:
                bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        if bool(ra["separated"]) != bool(rb["separated"]):
            bad.append(f"{lab}.separated: {ra['separated']} vs {rb['separated']}")
    return bad


# ---------------------------------------------------------------------------------------------------------------- the read
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--tool-root", default=f"{ART}/dagt1_tree_b43c7a9a")
    ap.add_argument("--t1hd-read")
    ap.add_argument("--t1k-read")
    ap.add_argument("--eval-cache")
    ap.add_argument("--tracks")
    ap.add_argument("--eval-grid")
    ap.add_argument("--eval-join")
    ap.add_argument("--init-sha", action="append", help="SEED=sha256 of the checkpoint DAG2 / DAG2-0 started from (the runs' own records; must equal DAG's banked ckpt)")
    ap.add_argument("--prereg", help="PREREG_V8F_DAGGER.md (the DAGGER reader checks its table cells in it)")
    ap.add_argument("--prereg2", default=DEFAULTS["prereg2"], help="PREREG_V8F_DAGGER2.md @ 1446333d (pinned sha256; the amended DAG2 table rows are checked in it)")
    ap.add_argument("--dag-read", default=DEFAULTS["dag_read"])
    ap.add_argument("--readers", default=DEFAULTS["readers"])
    ap.add_argument("--reada-code", default=DEFAULTS["reada_code"])
    ap.add_argument("--lead-block", default=DEFAULTS["lead_block"])
    ap.add_argument("--dag-t1", default=DEFAULTS["dag_t1"], help="DAG's banked T1-kin outputs (the DAG - P10 baseline of the newly-worse clause)")
    ap.add_argument("--dag-read-json", default=DEFAULTS["dag_read_json"])
    ap.add_argument("--reada-banked", default=DEFAULTS["reada_banked"])
    ap.add_argument("--t1kf-banked", default=DEFAULTS["t1kf_banked"])
    ap.add_argument("--work-dir")
    ap.add_argument("--static-check", action="store_true", help="pins and static inputs only (no run output is read)")
    ap.add_argument("--preflight", action="store_true", help="the bank and run-record controls on whatever DAG2 / DAG2-0 runs are pulled (no rollout output is read)")
    ap.add_argument("--repro-only", action="store_true", help="layer A only: re-compute DAG - P10 from DAG's banked dumps and compare with the banked READ_A.json (control RC-DAG); writes --out as a small report")
    return ap


def static_paths(a):
    return {"dag_read.py": a.dag_read, "reader_src/hmc_controls.py": f"{a.readers}/hmc_controls.py", "reader_src/hmc_lib.py": f"{a.readers}/hmc_lib.py",
            "reader_src/lcdiag_common.py": f"{a.readers}/lcdiag_common.py", "reader_src/t1k_read.py": f"{a.readers}/t1k_read.py", "reader_src/t1kf_read_a1.py": f"{a.readers}/t1kf_read_a1.py",
            "reader_src/t1klc_read.py": f"{a.readers}/t1klc_read.py", "reader_src/v8f_flat_read.py": f"{a.readers}/v8f_flat_read.py",
            "readA_code/rr_lib.py": f"{a.reada_code}/rr_lib.py", "readA_code/readA.py": f"{a.reada_code}/readA.py", "readA_code/build_report.py": f"{a.reada_code}/build_report.py",
            "banked/DAG_READ.json": a.dag_read_json, "banked/READ_A.json": a.reada_banked, "banked/T1KF_READ_A1_DAG.json": a.t1kf_banked, "lead_block_k30.npz": a.lead_block}


def main(argv=None):
    a = build_parser().parse_args(argv)
    t_start = time.time()
    bad = pin_problems(static_paths(a))
    ok2, det2 = check_prereg2_text(a.prereg2) if os.path.isfile(a.prereg2) else (False, f"{a.prereg2} is missing")
    if not ok2:
        bad.append(f"PREREG2: {det2}")
    if a.static_check:
        print(json.dumps({"pins_checked": len(static_paths(a)), "problems": bad, "prereg2": det2 if ok2 else None, "bars_prereg": BARS_PREREG}, indent=1, default=str))
        print("DAG2-READ-STATIC-OK" if not bad else "DAG2-READ-STATIC-REFUSED")
        return 0 if not bad else 2
    if bad:
        print("DAG2 READ REFUSED (nothing written): pinned inputs:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 2
    DR = load_dag_read(a.dag_read)
    if a.preflight:
        ctl, state = preflight(a, DR)
        for k, v in ctl.items.items():
            print(("PASS " if v["PASS"] else "FAIL ") + k + "  " + str(v["detail"])[:170].replace(chr(10), " "))
        print(json.dumps({"state": state, "controls": len(ctl.items), "failed": sorted(ctl.failed())}, indent=1, default=str))
        if a.out:
            write_out(a, {"preflight": True, "state": state, "controls_PASS": ctl.flags(), "controls": ctl.items, "provenance": provenance(a, t_start)}, 0)
        return 0 if not ctl.failed() else 1
    ctl = DR.Ctl()
    ctl.add("PINS", True, f"{len(static_paths(a))} pinned inputs re-hashed")
    ctl.add("PREREG2-TEXT", ok2, det2)
    bars = rf_bars(DR.bar_for)
    ctl.add("RF-BARS", all(abs(bars[hd][s] - BARS_PREREG[hd][s]) < 1e-12 for hd in bars for s in SEEDS), {"computed": bars, "PREREG": BARS_PREREG})
    man_raw = json.load(open(a.manifest, encoding="utf-8")) if (a.manifest and os.path.isfile(a.manifest)) else {"arms": {}}
    arms_raw = man_raw.get("arms", {})
    a.d2_dump = lambda s: arms_raw["DAG"][str(s)]["t1"]["dump_dir"]
    a.d20_dump = lambda s: arms_raw["DAG0"][str(s)]["t1"]["dump_dir"]
    a.dag_dump = lambda s: f"{a.dag_t1}/dump_DAG_s{s}_m10_causal_heads"
    if a.repro_only:      # control RC-DAG in isolation: DAG's banked dumps stand in for DAG2 / DAG2-0 (DAG, DAG-0), so the whole of layer A runs on real banked data and nothing else is read
        a.d2_dump = lambda s: f"{a.dag_t1}/dump_DAG_s{s}_m10_causal_heads"
        a.d20_dump = lambda s: f"{a.dag_t1}/dump_DAG0_s{s}_m10_causal_heads"
    A = ReadA(a)
    np = A.np
    res_g, rows = None, {}
    if not a.repro_only:
        try:
            res_g, T = layer_g(a, DR, DAG_CKPT_SHA)
        except DR.Refusal as e:
            print("DAG2 READ REFUSED (nothing written):", file=sys.stderr)
            for m_ in e.msgs:
                print("  -", m_, file=sys.stderr)
            return 2
        except (OSError, KeyError, ValueError) as e:
            print(f"DAG2 READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        if res_g["OUTCOME"]["OUTCOME"] == "BLOCKED":
            cov = res_g["OUTCOME"].get("coverage")
            return write_out(a, {"OUTCOME": blocked_block_2(cov), "controls_PASS": res_g["controls_PASS"], "provenance": provenance(a, t_start), "note": "no arm was read (COVERAGE below the minimum)"}, 4)
        for s in SEEDS:
            sd = res_g["rows"][f"s{s}"]
            for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
                r = sd[nm]
                ctl.add(f"RF-SLOT-ZERO:{nm}:s{s}", float(r["RF"]) == 0.0, {"RF_from_the_self_replicate_slot": r["RF"], "note": "must be exactly 0.0: the ORC0 slot IS DAG2-0"})
                r["RF_slot_selfreplicate"] = r["RF"]
                r["RF"], r["RF_note"] = RF_FIXED[hd][s], "PREREG_V8F_DAGGER2 change 3: fixed at DAGGER's measured value (an estimate from another init with the same recipe)"
                r["bar"] = DR.bar_for(hd, r["RF"])
                r["bar_binding"] = "2RF" if 2.0 * r["RF"] > DR.TWO_F[hd] else "2F"
                r["clears_dagger_bar_beside"] = clears_bar(r, r["bar"])
    # ---- layer A
    sd_ = {}
    for s in SEEDS:
        d2 = A.dump(a.d2_dump(s))
        d20 = A.dump(a.d20_dump(s))
        dg = A.dump(a.dag_dump(s))
        sd_[s] = (d2, d20, dg)
    lay = layer_a(a, A, DR, ctl, sd_, res_g)
    if a.repro_only:
        return write_out(a, {"RC-DAG": lay["RC-DAG"], "controls_PASS": ctl.flags(), "controls": ctl.items, "provenance": provenance(a, t_start)}, 0 if not ctl.failed() else 2)
    if ctl.failed():
        print("DAG2 READ REFUSED (nothing written):", file=sys.stderr)
        for k, v in ctl.failed().items():
            print("  -", k, v["detail"], file=sys.stderr)
        return 2
    # ---- the PREREG's composition
    dec, rows = compose({s: res_g["rows"][f"s{s}"] for s in SEEDS}, lay["rows"], A.DIRECTIONS)
    out = {"OUTCOME": dec, "controls_PASS": ctl.flags(), "rows": rows, "read_a": lay["read_a_tables"], "beside": lay["beside"],
           "bars": {"RF_FIXED": RF_FIXED, "computed": bars, "PREREG": BARS_PREREG, "note": "the bars have no consumer in the DAG2 table (its guards are 'not separated-worse'); reported beside as clears_dagger_bar_beside"},
           "layer_g_beside": res_g.get("beside"), "bank": res_g.get("bank"), "controls": {**{k: v for k, v in res_g["controls"].items()}, **ctl.items},
           "tier": "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic; plain arms); R-T0 at T0 (box 2's standard eval). Never 'closed loop'.",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes, per seed, paired. It answers 'would another draw of EPISODES say this?'; the training-seed "
                        "question is answered by the two seeds and by RF_FIXED; the planner is deterministic.",
           "provenance": provenance(a, t_start)}
    return write_out(a, out, outcome_rc(dec))


def layer_a(a, A, DR, ctl, dumps, res_g):
    """Read A's rows.  dumps[s] = (DAG2, DAG2-0, DAG) per-window dumps (causal).  Returns {"rows": {...}, "read_a_tables": ..., "beside": ..., "RC-DAG": ...}."""
    np, T, RA = A.np, A.T, A.RA
    banked_reada = json.load(open(a.reada_banked, encoding="utf-8"))["READ_A"]["rows"]
    banked_dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    comps = {k: {} for k in ("D2P", "D2D20", "DGP", "D2DG")}
    tacdk = {k: {} for k in comps}
    ident_ok = {}
    for s in SEEDS:
        d2, d20, dg = dumps[s]
        for nm, d in (("DAG2-0", d20), ("DAG", dg)):
            same = all(np.array_equal(d[k], d2[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
            ctl.add(f"C4-WINDOWS:{nm}-vs-DAG2:s{s}", bool(same and len(d["eid"]) == DR.N_WINDOWS and len(np.unique(d["eid"])) == DR.N_EPISODES),
                    {"n_windows": int(len(d["eid"])), "n_episodes": int(len(np.unique(d["eid"]))), "note": "clip_index, ws, eid, g, v0 bit-equal"})
            ctl.add(f"C3-FLOOR:{nm}-vs-DAG2:s{s}", bool(np.array_equal(d["fl"], d2["fl"])), "P10's tracks read no model: max |delta| must be exactly 0.0")
        cx = A.tacdk_context(d2)
        P = {"DAG2": T.prep("DAG2", d2["cl"], cx), "DAG2-0": T.prep("DAG2-0", d20["cl"], cx), "DAG": T.prep("DAG", dg["cl"], cx), "P10": T.prep("P10", d2["fl"], cx)}
        for k, (X, Y, PX, PY) in {"D2P": (d2["cl"], d2["fl"], P["DAG2"], P["P10"]), "D2D20": (d2["cl"], d20["cl"], P["DAG2"], P["DAG2-0"]),
                                  "DGP": (dg["cl"], dg["fl"], P["DAG"], P["P10"]), "D2DG": (d2["cl"], dg["cl"], P["DAG2"], P["DAG"])}.items():
            comps[k][s], tacdk[k][s] = A.pair_rows(X, Y, d2, PX, PY)
        idc = RA.ident_block(d2["cl"], d2["g"], d2["eid"])
        idt = T.rows(P["DAG2"], P["DAG2"], d2["eid"])
        ident_ok[s] = bool(all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values()))
        ctl.add(f"C2-IDENTITY:DAG2-DAG2:s{s}", ident_ok[s], "(DAG2, DAG2) reads +0 [0, 0] on every comps and TAC / DK row")
        if res_g is not None:                                                         # the two code paths agree on DAG2 - DAG2-0 ADE6 (layer G's R-ADE vs readA.rows_block)
            ra = comps["D2D20"][s]["ADE6"]["delta"]
            rg = res_g["rows"][f"s{s}"]["R-ADE"]["delta"]
            ctl.add(f"ADE-LINK:s{s}", abs(float(ra) - float(rg)) <= 1e-9, {"readA_rows_block": ra, "layer_G_R-ADE": rg})
    spos = {"DAG2": None, "DAG": banked_dagread}
    tabs = {}
    tabs["DAG2_minus_P10"] = A.table(comps["D2P"], tacdk["D2P"], res_g if res_g is not None else banked_dagread, "DAG")
    tabs["DAG2_minus_DAG20"] = A.table(comps["D2D20"], tacdk["D2D20"], banked_dagread, None)
    tabs["DAG_minus_P10"] = A.table(comps["DGP"], tacdk["DGP"], banked_dagread, "DAG")
    tabs["DAG2_minus_DAG"] = A.table(comps["D2DG"], tacdk["D2DG"], banked_dagread, None)
    # RC-DAG: DAG - P10 re-computed here == the banked READ_A.json, row for row
    rc = {}
    for s in SEEDS:
        diffs = rows_equal(tabs["DAG_minus_P10"][s], banked_reada[f"s{s}"])
        rc[f"s{s}"] = {"differences": diffs, "n_rows": len(banked_reada[f"s{s}"])}
        ctl.add(f"RC-DAG:s{s}", not diffs, {"rows_compared": len(banked_reada[f"s{s}"]), "tolerance": 1e-4, "differences": diffs[:6]})
    # C-SET: the PREREG's crawl set is the banked set of separated-worse rows, minus the two longitudinal rows that belong to DAG-OACC
    for s in SEEDS:
        derived = sorted(l for l, r in banked_reada[f"s{s}"].items() if r.get("separated_worse") and l not in EXEMPT_NEWLY)
        ctl.add(f"C-SET:s{s}", derived == sorted(CRAWL_SET[s]), {"derived_from_banked_READ_A": derived, "PREREG_C": sorted(CRAWL_SET[s])})
    rowsd = {"DAG2_minus_P10": tabs["DAG2_minus_P10"], "DAG2_minus_DAG20": tabs["DAG2_minus_DAG20"], "DAG_minus_P10": tabs["DAG_minus_P10"]}
    beside = {"DAG2_minus_DAG": tabs["DAG2_minus_DAG"],
              "read_a_on_DAG2": {"verdict": {f"s{s}": A.BR["verdict"](tabs["DAG2_minus_P10"])[s] for s in SEEDS}, "outcome_if_applied": A.BR["outcome"](A.BR["verdict"](tabs["DAG2_minus_P10"]))},
              "t1_kin_four_families_DAG2_minus_DAG20": "the 'DAG2_minus_DAG20' table above (Read A's rows are the T1-kin four families; STRATEGIC is unpaired and NOT COMPUTED there)",
              "t0_four_families_DAG2_minus_DAG20": "NOT BUILT: no paired T0 four-family instrument exists in dag_read.py or in Read A's readers (dag_read's own beside.not_computed says the same); R-T0 (ADE6) is the only T0 pairing",
              "dag2_with_hacc_refit": hacc_status(),
              "s_pos_against_ctrv": "in layer_g_beside.per_seed.s*.per_run_t1heads.DAG['S-POS vs CTRV (ticks>=1, within this run)'] (DAG := DAG2) and in read_a.DAG2_minus_P10 (STRATEGIC rows)"}
    return {"rows": rowsd, "read_a_tables": {k: {f"s{s}": v[s] for s in SEEDS} for k, v in tabs.items() if k in ("DAG2_minus_P10", "DAG_minus_P10", "DAG2_minus_DAG20")}, "beside": beside, "RC-DAG": rc}


def hacc_status():
    """Beside: DAG2 with HACC's rule re-fitted by PREREG_V8F_DAG_HACC's recipe, 'if the fit inputs exist'.  The recipe fits on a checkpoint's TRAIN-side outputs (daghacc/fit_dag.py); this reader does not run it."""
    want = [f"{ART}/daghacc/fit/fit_DAG2_s{s}.json" for s in SEEDS]
    return {"status": "NOT RUN", "fit_inputs_expected": want, "present": [os.path.isfile(p) for p in want],
            "reason": "the re-fit is a separate recipe (daghacc/fit_dag.py + bake_daghacc.py) on DAG2's checkpoints; this reader reads Read A on plain arms and composes no post-map"}


def preflight(a, DR):
    """Early warning, no rollout output read: the DAGGER reader's bank and run-record controls on the PULLED DAG2 / DAG2-0 runs (whatever has been pulled), patched exactly as the read patches them.
    Mirrors the first half of dag_read.read() control for control; the controls themselves are the reader's functions."""
    ctl = DR.Ctl()
    manifest = DR.load_manifest(a.manifest)
    init = {s: v.lower() for s, v in DR.parse_kv(a.init_sha, "--init-sha").items()}
    ki, kw, kt = "init_ckpt.sha256", "train_window_list_sha256", "train_sha256"
    kn, ke, kp, kb = "dagger_n", "dagger_eligible", "dagger_p", "dagger_bank_sha256"
    saved = (DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate)
    DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate = dict(DAG_CKPT_SHA), make_g_argv(DR), make_g_replicate(DR)
    state = {}
    try:
        eval_ids, train_ids = DR.jload(manifest["eval141_clip_ids"]), DR.jload(manifest["train_clip_ids"])
        banks = {}
        for s in SEEDS:
            ctl.add(f"M2B-CKPT:s{s}", init.get(s) == DAG_CKPT_SHA[s], {"passed": str(init.get(s))[:16], "DAG_s%d" % s: DAG_CKPT_SHA[s][:16]})
            be = (manifest.get("bank") or {}).get(str(s))
            if not be or not os.path.isfile(be["report"]):
                state[f"bank_s{s}"] = "not pulled"
                continue
            B = DR.load_bank(be["report"], be.get("bank_file"), what=f"bank seed {s} report")
            DR.bank_stage(ctl, s, B, init.get(s, ""), eval_ids, train_ids)
            banks[s] = B
            cov = DR.coverage_of(B)
            state[f"coverage_s{s}"] = cov
            ctl.add(f"COVERAGE:s{s}", cov["eligible_fraction"] >= COVERAGE_MIN, cov)
        recs = {}
        for s in SEEDS:
            for arm in ("DAG", "DAG0"):
                e = DR.entry(manifest, arm, s)
                tag = f"{arm}_s{s}"
                if not os.path.isfile(os.path.join(e["train_dir"], "summary.json")):
                    state[tag] = "not pulled"
                    continue
                tr = DR.train_records(e, ki)
                recs[(arm, s)] = tr
                ctl.run(f"SUMMARY:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_summary(tag, s, tr))
                ctl.run(f"INIT-CKPT:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_init(tag, s, tr, init.get(s, ""), ki))
                ctl.run(f"ARGV:{tag}", lambda tr=tr, arm=arm, s=s, tag=tag: DR.g_argv(tag, arm, s, tr))
                ctl.run(f"DAGGER-REACHED:{tag}", lambda tr=tr, arm=arm, tag=tag: DR.g_reached(tag, arm, tr, kn, ke, kp))
                ctl.run(f"GATE-TOKEN:{tag}", lambda tr=tr, e=e, tag=tag: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
                if arm == "DAG" and s in banks:
                    ctl.run(f"BANK-DECLARED:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_bank_declared(tag, s, banks[s], tr, init.get(s, ""), ki, kb))
            if ("DAG", s) in recs and ("DAG0", s) in recs and s in banks:
                rd, r0 = recs[("DAG", s)], recs[("DAG0", s)]
                ctl.run(f"PAIR:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_pair(s, rd, r0, ki))
                ctl.run(f"ORC0-SLOT-SELF:s{s}", lambda r0=r0, s=s: DR.g_replicate(s, r0, r0, ki))
                ctl.run(f"K-B3.2:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_kb32(banks[s], rd, r0, kw))
                ctl.run(f"TRAIN-LIST:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_train_list(banks[s], train_ids, rd, r0, kt))
    finally:
        DR.BANKED_M2B_SHA, DR.g_argv, DR.g_replicate = saved
    return ctl, state


def provenance(a, t_start):
    return {"prereg": PREREG2_ID, "prereg_sha256": PREREG2_SHA256, "claim_row": CLAIM_ROW, "reader_sha256": sha256_file(os.path.abspath(__file__)),
            "pins": {k: PINS[k] for k in PINS}, "manifest_sha256": sha256_file(a.manifest) if a.manifest and os.path.isfile(a.manifest) else None, "wall_s": time.time() - t_start,
            "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}


def write_out(a, out, rc):
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    oc = out.get("OUTCOME")
    if isinstance(oc, dict):
        print(json.dumps({"OUTCOME": oc["OUTCOME"], "outcome_qualifier": oc.get("outcome_qualifier"), "adopts_DAG2": oc.get("adopts_DAG2"), "matching_rows": oc["matching_rows_in_table_order"],
                         "committed_next": oc.get("committed_next")}, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}  sha256 {sha256_file(a.out)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
