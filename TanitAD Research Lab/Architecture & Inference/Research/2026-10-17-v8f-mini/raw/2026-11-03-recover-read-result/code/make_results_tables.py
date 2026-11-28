#!/usr/bin/env python3
"""make_results_tables.py -- raw/RESULTS_TABLES.md of the RECOVER-READ package, generated from raw/RECOVER_READ.json ONLY.

Run (from anywhere):  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I code/make_results_tables.py [path/to/RECOVER_READ.json]

Every number in the output is formatted from the JSON; no number is typed in this file. Nothing is recomputed: deltas, intervals, shares, flags and
lists are the reader's own values (code/recover_read.py, sha256 in PINS.txt). The only logic here is layout, plus asserts that fail loudly if the
JSON has a different row set, family set or a marker list that disagrees with a cell.

Row -> family: the JSON does NOT assign rows to families (it carries the four family NAMES as the keys of D3_by_family and nothing per row), so the
grouping below is the brief's. It equals the FAMILY map of the pinned dagland/smooth_t1/code/smooth_t1_read.py except that M1 crawl share is shown
under LONGITUDINAL (the pinned map tags it 'crawl', so the reader's own D3 does not count M1 toward LONGITUDINAL).

Marker per cell (the JSON carries no per-row direction table; its own better / worse lists are used where it has them):
  ns     the JSON cell has separated = false
  sep-B  separated, and the row is in the JSON's separated-better list for that block
  sep-W  separated, and the row is in the JSON's separated-worse list for that block
  sep    separated, direction not listed by the JSON for that block
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
PK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else f"{PK}/raw/RECOVER_READ.json"
OUT_PATH = f"{PK}/raw/RESULTS_TABLES.md"

FAMILY_ROWS = [
    ("PRIMARY (ADE6 on top)", ["ADE6"]),
    ("LONGITUDINAL", ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "M1 crawl share"]),
    ("LATERAL", ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"]),
    ("TACTICAL", ["TAC-LAT", "TAC-LON"]),
    ("STRATEGIC", ["S-POS@30", "S-POS@50", "S-POS@80"]),
]
ROWS = [r for _, rs in FAMILY_ROWS for r in rs]
M1 = "M1 crawl share"
SEEDS = ("s0", "s1")
# display name of each seeds.<seed>.<key> block
BLOCK_NAME = {"R_minus_C0": "R - C0", "C0REP_minus_C0": "C0rep - C0", "R_minus_P10": "R - P10", "RC_minus_P10": "RC - P10", "C0C_minus_P10": "C0C - P10",
              "RC_minus_R": "RC - R", "C0_minus_P10": "C0 - P10", "C0C_minus_C0": "C0C - C0", "FZC0_minus_P10": "FZ-C0 - P10", "FZR_minus_FZC0": "FZ-R - FZ-C0"}
MAIN_BLOCKS = ["R_minus_C0", "C0REP_minus_C0", "R_minus_P10", "RC_minus_P10", "C0C_minus_P10", "RC_minus_R"]
REF_BLOCKS = ["C0_minus_P10", "C0C_minus_C0", "FZC0_minus_P10", "FZR_minus_FZC0"]

j = json.load(open(JSON_PATH, encoding="utf-8"))
J_SHA = hashlib.sha256(open(JSON_PATH, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ formatting
def f4(x):
    return f"{x:+.4f}"


def u4(x):
    return f"{x:.4f}"


def tf(b):
    return "true" if b else "false"


def cell_txt(c, mk):
    return f"{f4(c['delta'])} [{f4(c['lo'])}, {f4(c['hi'])}] {mk}"


def point_txt(c, mk):
    return f"{f4(c['delta'])} {mk}"


def lst(xs):
    return ", ".join(xs) if xs else "(none)"


# ------------------------------------------------------------------------------------------------ structure asserts
for s in SEEDS:
    for b in MAIN_BLOCKS + REF_BLOCKS:
        assert set(j["seeds"][s][b].keys()) == set(ROWS), (s, b, sorted(set(j["seeds"][s][b].keys()) ^ set(ROWS)))
        for r, c in j["seeds"][s][b].items():
            assert set(c.keys()) == {"delta", "lo", "hi", "separated"}, (s, b, r)
FAM_NAMES = [n for n, _ in FAMILY_ROWS[1:]]
for s in SEEDS:
    assert list(j["read_a_R_vs_P10"]["by_seed"][s]["D3_by_family"].keys()) == FAM_NAMES, "family names differ from the JSON's D3_by_family"
assert j["mode"] == "production" and j["tier"] == "T1-kin" and j["refused"] is None and j["controls_failed"] == []


# ------------------------------------------------------------------------------------------------ the JSON's own direction lists
def dir_lists(block, s):
    """(better_rows or None, worse_rows or None, complete) from the JSON's own lists; None = the JSON does not list it for this block."""
    if block == "R_minus_P10":
        ra = j["read_a_R_vs_P10"]["by_seed"][s]
        return set(ra["separated_better_rows"]), set(ra["separated_worse_rows"]), True
    if block == "C0C_minus_P10":
        ra = j["composite"]["read_a_C0C_vs_P10"][s]
        return set(ra["separated_better_rows"]), set(ra["separated_worse_rows"]), True
    if block == "RC_minus_P10":
        return None, set(j["composite"]["by_seed"][s]["separated_worse_rows"]), False
    if block == "R_minus_C0":
        xy = j["seeds"][s]["read_xy_R_minus_C0"]
        return {r for r, v in xy["crawl_separated_better"].items() if v}, set(xy["non_crawl_separated_worse"]), False
    return None, None, False


def marker(block, s, row):
    c = j["seeds"][s][block][row]
    if not c["separated"]:
        return "ns"
    better, worse, _ = dir_lists(block, s)
    if better is not None and row in better:
        return "sep-B"
    if worse is not None and row in worse:
        return "sep-W"
    return "sep"


# the JSON's lists must agree with its cells (a marker is only as good as that)
for s in SEEDS:
    for b in ("R_minus_P10", "C0C_minus_P10", "RC_minus_P10", "R_minus_C0"):
        better, worse, complete = dir_lists(b, s)
        for r in (better or set()) | (worse or set()):
            assert j["seeds"][s][b][r]["separated"], (s, b, r, "listed by the JSON but its cell is not separated")
        if better is not None and worse is not None:
            assert not (better & worse), (s, b)
        if complete:
            for r in ROWS:
                if j["seeds"][s][b][r]["separated"]:
                    assert r in better or r in worse, (s, b, r, "separated but in neither JSON list")


# ------------------------------------------------------------------------------------------------ table builders
def family_table(s, blocks, with_ci=True):
    """one markdown table per family; columns = blocks."""
    out = []
    for fam, rows in FAMILY_ROWS:
        out.append(f"**{fam}**")
        out.append("")
        out.append("| row | " + " | ".join(BLOCK_NAME[b] for b in blocks) + " |")
        out.append("|---|" + "---|" * len(blocks))
        for r in rows:
            cells = []
            for b in blocks:
                c = j["seeds"][s][b][r]
                cells.append(cell_txt(c, marker(b, s, r)) if with_ci else point_txt(c, marker(b, s, r)))
            out.append(f"| {r} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def compact_table(blocks):
    """19 rows; columns = (block, seed) point + marker."""
    heads = [f"{BLOCK_NAME[b]} {s}" for b in blocks for s in SEEDS]
    out = ["| row | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
    for fam, rows in FAMILY_ROWS:
        out.append(f"| **{fam}** |" + " |" * len(heads))
        for r in rows:
            out.append(f"| {r} | " + " | ".join(point_txt(j["seeds"][s][b][r], marker(b, s, r)) for b in blocks for s in SEEDS) + " |")
    out.append("")
    return out


def kv_table(title_cols, rows):
    out = ["| item | " + " | ".join(title_cols) + " |", "|---|" + "---|" * len(title_cols)]
    for name, vals in rows:
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    out.append("")
    return out


# ------------------------------------------------------------------------------------------------ the document
L = []
A = L.append
A("# RECOVER-READ results tables (GENERATED by code/make_results_tables.py from raw/RECOVER_READ.json; do not hand-edit)")
A("")
A(f"* source: `raw/RECOVER_READ.json` sha256 `{J_SHA}`; mode `{j['mode']}`; tier `{j['tier']}`; refused `{j['refused']}`; controls_failed `{j['controls_failed']}`; run_clock `{j['run_clock']}`")
A(f"* evidence class (the JSON's): {j['evidence_class']}")
A(f"* estimator (the JSON's): {j['estimator']}")
A(f"* cards: composite `{j['cards']['composite']}`; recover_ft `{j['cards']['recover_ft']}`; recover_ft_amendment_1 `{j['cards']['recover_ft_amendment_1']}`; reader sha256 `{j['reader_sha256']}`")
A("* cell format: `delta [lo, hi] marker`. delta = point of the paired episode-cluster bootstrap on the row; [lo, hi] = its interval; the unit is the row's own unit (the JSON carries no unit).")
A("* markers: `ns` = the JSON cell has separated = false; `sep-B` / `sep-W` = separated, and the row is in the JSON's separated-better / separated-worse list for that block; `sep` = separated, direction not listed by the JSON for that block.")
A("  The JSON lists directions for R - P10 and C0C - P10 (both lists), RC - P10 (worse list only), R - C0 (crawl rows better, non-crawl rows worse). Every other block shows `sep` / `ns` only.")
A("* bounds are printed at 4 decimals; separation is the JSON's own `separated` flag and is never re-derived from the printed bounds (a bound that prints as +0.0000 can sit on either side of zero).")
A("* families: the JSON does not assign rows to families (it carries only the four family names as `D3_by_family` keys). The grouping is the brief's; it equals the FAMILY map of the pinned `smooth_t1_read.py`")
A("  except that `M1 crawl share` is shown under LONGITUDINAL (the pinned map tags it `crawl`, so the reader's own D3 does not count M1 toward LONGITUDINAL). ADE6 is on top.")
A(f"* STRATEGIC (the JSON's `strategic_family` note, verbatim): {j['strategic_family']}")
A("")

# ---- Part A: verdict facts
A("## A. Verdict facts (every value is a JSON field)")
A("")
A(f"**label (H-RECOVER-FT-1)** = `{j['label']}`")
A("")
pr, gd, fl, ff = j["primary"], j["guards"], j["floor"], j["frame_freeze"]
A("### A1. PRIMARY / GUARDS / FLOOR / frame-freeze")
A("")
rows = []
for name, fn in [
    ("M1 R - C0, logged frames", lambda s: cell_txt(j["seeds"][s]["R_minus_C0"][M1], marker("R_minus_C0", s, M1))),
    ("M1 separated-better (m1_cell_logged.better)", lambda s: tf(j["seeds"][s]["m1_cell_logged"]["better"])),
    ("M1 C0 - P10, the excess (m1_cell_logged.yp_delta)", lambda s: f4(j["seeds"][s]["m1_cell_logged"]["yp_delta"])),
    ("M1 C0 - P10 separated-worse (m1_cell_logged.yp_separated_worse)", lambda s: tf(j["seeds"][s]["m1_cell_logged"]["yp_separated_worse"])),
    ("share closed (primary.share_closed)", lambda s: u4(pr["share_closed"][s])),
    ("ADE6 R - C0", lambda s: cell_txt(j["seeds"][s]["R_minus_C0"]["ADE6"], marker("R_minus_C0", s, "ADE6"))),
    ("ADE6 within the guard (guards.per_seed.ade6_within_guard)", lambda s: tf(gd["per_seed"][s]["ade6_within_guard"])),
    ("non-crawl separated-worse rows in R - C0", lambda s: lst(gd["per_seed"][s]["non_crawl_separated_worse"])),
    ("FLOOR abs(R - C0) on M1 (point)", lambda s: u4(fl["per_seed"][s]["abs_R_minus_C0"])),
    ("FLOOR abs(C0rep - C0) on M1 (point)", lambda s: u4(fl["per_seed"][s]["abs_C0rep_minus_C0"])),
    ("FLOOR exceeds_floor", lambda s: tf(fl["per_seed"][s]["exceeds_floor"])),
    ("FREEZE M1 FZ-R - FZ-C0 (xy_delta)", lambda s: f4(ff["m1_cells_freeze"][s]["xy_delta"])),
    ("FREEZE M1 FZ-R - FZ-C0, cell with interval", lambda s: cell_txt(j["seeds"][s]["FZR_minus_FZC0"][M1], marker("FZR_minus_FZC0", s, M1))),
    ("FREEZE separated-better (m1_cells_freeze.better)", lambda s: tf(ff["m1_cells_freeze"][s]["better"])),
    ("FREEZE M1 FZ-C0 - P10, the excess (yp_delta)", lambda s: f4(ff["m1_cells_freeze"][s]["yp_delta"])),
    ("FREEZE share closed", lambda s: u4(ff["share_closed"][s])),
]:
    rows.append((name, [fn(s) for s in SEEDS]))
L += kv_table(list(SEEDS), rows)
A(f"* PRIMARY: n_seeds_separated_better = {pr['n_seeds_separated_better']}; mean_share_closed = {u4(pr['mean_share_closed'])}; share_bar = {pr['share_bar']}; PASS = {tf(pr['PASS'])}")
A(f"* GUARDS: ade_guard_m = {gd['ade_guard_m']}; PASS = {tf(gd['PASS'])}")
A(f"* FLOOR: PASS = {tf(fl['PASS'])}")
A(f"* FRAME-FREEZE: reading = `{ff['reading']}`; n_seeds_separated_better = {ff['n_seeds_separated_better']}; mean_share_closed = {u4(ff['mean_share_closed'])}; robust_bar = {ff['robust_bar']}; artefact_bar = {ff['artefact_bar']}; P_logged_holds = {tf(ff['P_logged_holds'])}; applies_to_label = {tf(ff['applies_to_label'])}")
A(f"* freeze_cells_note (JSON): {ff['freeze_cells_note']}")
A("")

A("### A2. Read A of R vs P10")
A("")
ra = j["read_a_R_vs_P10"]
rows = []
for name, fn in [
    ("ADE6 R - P10", lambda s: cell_txt(j["seeds"][s]["R_minus_P10"]["ADE6"], marker("R_minus_P10", s, "ADE6"))),
    ("D1", lambda s: tf(ra["by_seed"][s]["D1"])),
    ("D2", lambda s: tf(ra["by_seed"][s]["D2"])),
    ("D3 (four families)", lambda s: tf(ra["by_seed"][s]["D3"])),
    ("D3 by family", lambda s: ", ".join(f"{k} {tf(v)}" for k, v in ra["by_seed"][s]["D3_by_family"].items())),
    ("D3 (three families, STRATEGIC set aside)", lambda s: tf(ra["by_seed"][s]["D3_three_families_strategic_paused"])),
    ("DOMINANT (four-family D3)", lambda s: tf(ra["by_seed"][s]["DOMINANT"])),
    ("DOMINANT (three-family D3)", lambda s: tf(ra["by_seed"][s]["DOMINANT_three_families_strategic_paused"])),
    ("separated-worse rows", lambda s: lst(ra["by_seed"][s]["separated_worse_rows"])),
    ("separated-better rows", lambda s: lst(ra["by_seed"][s]["separated_better_rows"])),
    ("STRATEGIC rows separated-better", lambda s: lst(ra["by_seed"][s]["strategic_rows_separated_better"])),
    ("failing_terms", lambda s: lst(ra["by_seed"][s]["failing_terms"])),
]:
    rows.append((name, [fn(s) for s in SEEDS]))
L += kv_table(list(SEEDS), rows)
A(f"* DOMINANT_both_seeds = {tf(ra['DOMINANT_both_seeds'])}; DOMINANT_three_families_strategic_paused_both_seeds = {tf(ra['DOMINANT_three_families_strategic_paused_both_seeds'])}; reported_beside_the_label_not_part_of_it = {tf(ra['reported_beside_the_label_not_part_of_it'])}")
A("")

A("### A3. H-RECOVER-COMPOSITE-1")
A("")
co = j["composite"]
A(f"* status = `{co['status']}`; counts = {tf(co['counts'])}; DOMINANT = {tf(co['DOMINANT'])}; DOMINANT (three-family) = {tf(co['DOMINANT_three_families_strategic_paused'])}; R_freeze_reading = `{co['R_freeze_reading']}`; attribution = `{co['attribution']}`; C0C_dominant_seeds = {co['C0C_dominant_seeds']}")
A(f"* freeze_reading_note (JSON): {co['freeze_reading_note']}")
A("")
rows = []
for arm, getter, pblock in [("RC vs P10", lambda s: co["by_seed"][s], "RC_minus_P10"), ("C0C vs P10", lambda s: co["read_a_C0C_vs_P10"][s], "C0C_minus_P10")]:
    for name, fn in [
        (f"{arm}: ADE6", lambda s, pblock=pblock: cell_txt(j["seeds"][s][pblock]["ADE6"], marker(pblock, s, "ADE6"))),
        (f"{arm}: M1 crawl share", lambda s, pblock=pblock: cell_txt(j["seeds"][s][pblock][M1], marker(pblock, s, M1))),
        (f"{arm}: D1", lambda s, getter=getter: tf(getter(s)["D1"])),
        (f"{arm}: D2", lambda s, getter=getter: tf(getter(s)["D2"])),
        (f"{arm}: D3 (four families)", lambda s, getter=getter: tf(getter(s)["D3"])),
        (f"{arm}: D3 by family", lambda s, getter=getter: ", ".join(f"{k} {tf(v)}" for k, v in getter(s)["D3_by_family"].items())),
        (f"{arm}: D3 (three families)", lambda s, getter=getter: tf(getter(s)["D3_three_families_strategic_paused"]) if "D3_three_families_strategic_paused" in getter(s) else "not in the read JSON"),
        (f"{arm}: DOMINANT (four-family D3)", lambda s, getter=getter: tf(getter(s)["DOMINANT"])),
        (f"{arm}: DOMINANT (three-family D3)", lambda s, getter=getter: tf(getter(s)["DOMINANT_three_families_strategic_paused"])),
        (f"{arm}: separated-worse rows", lambda s, getter=getter: lst(getter(s)["separated_worse_rows"])),
        (f"{arm}: failing_terms", lambda s, getter=getter: lst(getter(s)["failing_terms"])),
    ]:
        rows.append((name, [fn(s) for s in SEEDS]))
L += kv_table(list(SEEDS), rows)
A("RC - R is kbox's own effect on the recovery base (the JSON's `RC_minus_R_is_kboxs_own_effect`, six rows; the full 19-row block is in part C):")
A("")
kb = co["RC_minus_R_is_kboxs_own_effect"]
A("| row | " + " | ".join(SEEDS) + " |")
A("|---|" + "---|" * len(SEEDS))
for r in kb["s0"].keys():
    A(f"| {r} | " + " | ".join(cell_txt(kb[s][r], "sep" if kb[s][r]["separated"] else "ns") for s in SEEDS) + " |")
A("")

A("### A4. Replicate floor (C0rep vs C0, same recipe, training seed 10+s; the training-variance check)")
A("")
rows = []
for r in ("ADE6", M1):
    rows.append((f"{r}: C0rep - C0", [cell_txt(j["seeds"][s]["C0REP_minus_C0"][r], marker("C0REP_minus_C0", s, r)) for s in SEEDS]))
for r in ("ADE6", M1):
    rows.append((f"{r}: R - C0 (same contrast axis, for scale)", [cell_txt(j["seeds"][s]["R_minus_C0"][r], marker("R_minus_C0", s, r)) for s in SEEDS]))
L += kv_table(list(SEEDS), rows)
A(f"* variance questions (the JSON's `variance_questions`): episodes: {j['variance_questions']['episodes']}; training_run: {j['variance_questions']['training_run']}; inference: {j['variance_questions']['inference']}")
A("")

A("### A5. Controls")
A("")
A(f"* controls_failed = `{j['controls_failed']}`; refused = `{j['refused']}`; mode = `{j['mode']}`")
A("* " + "; ".join(f"{('_'.join(k.split('_')[:2]) if k.startswith('KCARD') else k.split('_')[0])} {'PASS' if v['PASS'] else 'FAIL'}" for k, v in j["controls"].items()))
ka1 = j["controls"]["KA1_bit_identity_rerun_of_R_s0"]
A(f"* KA1 (bit-identity rerun of R_s0): dump_bit_identical = {tf(ka1['dump_bit_identical'])}, dump_max_abs_dev = {ka1['dump_max_abs_dev']}, ticks_bit_identical = {tf(ka1['ticks_bit_identical'])}, ticks_max_abs_dev = {ka1['ticks_max_abs_dev']}")
k4 = j["controls"]["KFZ4_FZ_HLVL_keeps_its_closure_read_from_FZ_CONTROLS"]["kfz4"]
A(f"* KFZ4 (FZ-HLVL keeps its closure, read from FZ_CONTROLS.json): share_closed s0 = {u4(k4['share_closed']['s0'])}, s1 = {u4(k4['share_closed']['s1'])}; mean = {u4(k4['mean_share_closed'])}; share_bar = {k4['share_bar']}; PASS = {tf(k4['PASS'])}")
A("")

# ---- Part B: compact key tables
A("## B. Key blocks, point estimates and markers, both seeds (all 19 rows)")
A("")
A("### B1. R-centred: R - C0, C0rep - C0, R - P10")
A("")
L += compact_table(["R_minus_C0", "C0REP_minus_C0", "R_minus_P10"])
A("### B2. Composite: RC - P10, C0C - P10, RC - R")
A("")
L += compact_table(["RC_minus_P10", "C0C_minus_P10", "RC_minus_R"])

# ---- Part C: full tables
A("## C. The six requested blocks with intervals, per seed and family")
A("")
for s in SEEDS:
    A(f"### C-{s}")
    A("")
    L += family_table(s, MAIN_BLOCKS)

# ---- Part D: reference blocks
A("## D. Reference blocks (the excess against P10, the control composite, the frame-freeze contrast)")
A("")
for s in SEEDS:
    A(f"### D-{s}")
    A("")
    L += family_table(s, REF_BLOCKS)

text = "\n".join(L).rstrip("\n") + "\n"
open(OUT_PATH, "w", encoding="utf-8", newline="\n").write(text)
print("RESULTS_TABLES.md", len(L), "lines; json sha256", J_SHA[:16])
